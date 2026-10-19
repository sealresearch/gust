/* Work sharing for the float32 operators (GUST_THREADS; see README.md "Threads"). Platforms: WebAssembly shared
   memory + atomics (workers are host Web Workers), or GUST_THREADS_POSIX (pthreads, started by gust_threads_start;
   Linux, Raspberry Pi, Android, macOS, iOS). The executor runs on one thread; an operator splits its work into independent tasks (output
   column slabs, row chunks, heads) and calls gt_parallel, which the calling thread and the parked workers drain.
   Every output value is computed by exactly one task with the same code, so results do not depend on the thread
   count. Without GUST_THREADS the tasks run in order on the calling thread.

   Protocol: `next` holds a 48-bit generation and a 16-bit task index. A worker claims a task with a compare-and-swap
   for the generation it woke up for; the wide tag also rejects workers descheduled across 65536 rounds. The
   caller returns once `done` reaches the task count, i.e. after every claimed task has finished. */

#define GT_MAX 16u      /* threads including the caller */

typedef struct {
    /* Reuse unused pointer slots for the wider pool counters and immutable kind; pool bytes do not grow. */
    const void *p[8];
    uint32_t u[10], kind;
} GtArgs;

static void gt_task(uint32_t kind, const GtArgs *a, uint32_t i, uint32_t tid); /* ops.c */

#ifdef GUST_THREADS
static struct {
    uint64_t gen, next;
    uint32_t n, done, threads;
    GtArgs args;
} gt_pool = {0, 0, 0, 0, 1, {{0}, {0}, 0}};

/* ---- platform layer: park a worker until the generation changes, wake every parked worker ---- */
#if defined(__wasm__)
#define GT_EXPORT(name) __attribute__((export_name(name)))
/* WASM is little-endian: park on the low word, then recheck the full generation in gust_worker. Operators of one
   request follow each other within microseconds, so spin a little first (as the native pools do): a wake from
   memory.atomic.wait costs far more than an operator on small cores. */
static inline void gt_park(uint64_t g) {
    for (uint32_t i = 0; i < 20000u; i++)
        if (__atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) != g) return;
    __builtin_wasm_memory_atomic_wait32((int32_t *)&gt_pool.gen, (int32_t)g, -1);
}
static inline void gt_wake_all(void) { __builtin_wasm_memory_atomic_notify((int32_t *)&gt_pool.gen, 0xffffffffu); }
#elif defined(GUST_THREADS_POSIX)
#include <pthread.h>
#define GT_EXPORT(name)
static pthread_mutex_t gt_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gt_cv = PTHREAD_COND_INITIALIZER;
static inline void gt_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}
/* Operators of one request follow each other within microseconds: spin a little, then sleep on the condition. */
static void gt_park(uint64_t g) {
    for (uint32_t i = 0; i < 20000u; i++) {
        if (__atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) != g) return;
        gt_relax();
    }
    pthread_mutex_lock(&gt_mu);
    while (__atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) == g) pthread_cond_wait(&gt_cv, &gt_mu);
    pthread_mutex_unlock(&gt_mu);
}
static void gt_wake_all(void) { /* gen is already stored; taking the lock orders it before any waiter's check */
    pthread_mutex_lock(&gt_mu);
    pthread_cond_broadcast(&gt_cv);
    pthread_mutex_unlock(&gt_mu);
}
#elif defined(GUST_THREADS_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define GT_EXPORT(name)
static SRWLOCK gt_mu = SRWLOCK_INIT;
static CONDITION_VARIABLE gt_cv = CONDITION_VARIABLE_INIT;
static inline void gt_relax(void) { YieldProcessor(); }
static void gt_park(uint64_t g) {
    for (uint32_t i = 0; i < 20000u; i++) {
        if (__atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) != g) return;
        gt_relax();
    }
    AcquireSRWLockExclusive(&gt_mu);
    while (__atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) == g) SleepConditionVariableSRW(&gt_cv, &gt_mu, INFINITE, 0);
    ReleaseSRWLockExclusive(&gt_mu);
}
static void gt_wake_all(void) {
    AcquireSRWLockExclusive(&gt_mu);
    WakeAllConditionVariable(&gt_cv);
    ReleaseSRWLockExclusive(&gt_mu);
}
#else
#error "GUST_THREADS needs WebAssembly, GUST_THREADS_POSIX or GUST_THREADS_WIN32"
#endif

static void gt_drain(uint64_t gen, uint32_t tid) {
    for (;;) {
        uint64_t v = __atomic_load_n(&gt_pool.next, __ATOMIC_SEQ_CST);
        if ((v >> 16) != (gen & 0xffffffffffffull)) return;
        const uint32_t i = v & 0xffffu;
        if (i >= __atomic_load_n(&gt_pool.n, __ATOMIC_SEQ_CST)) return;
        if (!__atomic_compare_exchange_n(&gt_pool.next, &v, v + 1u, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) continue;
        gt_task(gt_pool.args.kind, &gt_pool.args, i, tid);
        __atomic_fetch_add(&gt_pool.done, 1u, __ATOMIC_SEQ_CST);
    }
}

/* Worker entry (one per worker thread, never returns): waits for a new generation, then drains its tasks. */
GT_EXPORT("gust_worker") void gust_worker(uint32_t tid) {
    uint64_t seen = __atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST);
    for (;;) {
        const uint64_t g = __atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST);
        if (g == seen) {
            gt_park(g);
            continue;
        }
        seen = g;
        gt_drain(g, tid);
    }
}

/* Number of threads (caller included) that may run tasks; set by the host once its workers are running. */
GT_EXPORT("gust_set_threads") uint32_t gust_set_threads(uint32_t n) {
    gt_pool.threads = n < 1u ? 1u : n > GT_MAX ? GT_MAX : n;
    return gt_pool.threads;
}

#if defined(GUST_THREADS_POSIX)
static void *gt_worker_main(void *arg) {
    gust_worker((uint32_t)(uintptr_t)arg);
    return NULL;
}
/* Starts n - 1 worker threads (once per process) and lets operators use n threads; returns the thread count. */
uint32_t gust_threads_start(uint32_t n) {
    static uint32_t started = 1u;
    if (n > GT_MAX) n = GT_MAX;
    for (; started < n; started++) {
        pthread_t t;
        if (pthread_create(&t, NULL, gt_worker_main, (void *)(uintptr_t)started)) break;
        pthread_detach(t);
    }
    return gust_set_threads(started < n ? started : n);
}
#elif defined(GUST_THREADS_WIN32)
static DWORD WINAPI gt_worker_main(LPVOID arg) {
    gust_worker((uint32_t)(uintptr_t)arg);
    return 0;
}
uint32_t gust_threads_start(uint32_t n) {
    static uint32_t started = 1u;
    if (n > GT_MAX) n = GT_MAX;
    for (; started < n; started++) {
        HANDLE t = CreateThread(NULL, 0, gt_worker_main, (LPVOID)(uintptr_t)started, 0, NULL);
        if (!t) break;
        CloseHandle(t);
    }
    return gust_set_threads(started < n ? started : n);
}
#endif

static inline uint32_t gt_threads(void) { return gt_pool.threads; }

static void gt_parallel(uint32_t kind, const GtArgs *a, uint32_t n) {
    if (gt_pool.threads <= 1u || n <= 1u || n > 0xffffu) {
        for (uint32_t i = 0; i < n; i++) gt_task(kind, a, i, 0);
        return;
    }
    const uint64_t g = __atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST) + 1u;
    /* Invalidate the old drain BEFORE changing its bounds or arguments. A worker can still be checking next
       after completing its last task; publishing a larger n first lets it claim an out-of-range old task with
       the new arguments. New-generation workers cannot drain until gen is published below. */
    __atomic_store_n(&gt_pool.next, (g & 0xffffffffffffull) << 16, __ATOMIC_SEQ_CST);
    gt_pool.args = *a;
    gt_pool.args.kind = kind;
    __atomic_store_n(&gt_pool.n, n, __ATOMIC_SEQ_CST);
    __atomic_store_n(&gt_pool.done, 0u, __ATOMIC_SEQ_CST);
    __atomic_store_n(&gt_pool.gen, g, __ATOMIC_SEQ_CST);
    gt_wake_all();
    gt_drain(g, 0);
    while (__atomic_load_n(&gt_pool.done, __ATOMIC_SEQ_CST) < n) {
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
        gt_relax();
#endif
    }
}
#else
static inline uint32_t gt_threads(void) { return 1u; }
static void gt_parallel(uint32_t kind, const GtArgs *a, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) gt_task(kind, a, i, 0);
}
#endif

