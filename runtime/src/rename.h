/* Per-variant public names for the x86-64 multi-variant build (dispatch_x86.c): with GUST_VARIANT=v the runtime's
   entry points become gust_<name>_v. Included first by gust.c. */
#ifdef GUST_VARIANT
#define GUST_CAT_(a, b) a##_##b
#define GUST_CAT(a, b) GUST_CAT_(a, b)
#define gust_context_bytes GUST_CAT(gust_context_bytes, GUST_VARIANT)
#define gust_capabilities GUST_CAT(gust_capabilities, GUST_VARIANT)
#define gust_init GUST_CAT(gust_init, GUST_VARIANT)
#define gust_run GUST_CAT(gust_run, GUST_VARIANT)
#define gust_run_valid GUST_CAT(gust_run_valid, GUST_VARIANT)
#define gust_run_valid_outputs GUST_CAT(gust_run_valid_outputs, GUST_VARIANT)
#define gust_run_valid_inputs GUST_CAT(gust_run_valid_inputs, GUST_VARIANT)
#define gust_run_traced GUST_CAT(gust_run_traced, GUST_VARIANT)
#define gust_run_valid_inputs_traced GUST_CAT(gust_run_valid_inputs_traced, GUST_VARIANT)
#define gust_run_checked GUST_CAT(gust_run_checked, GUST_VARIANT)
#define gust_tensor GUST_CAT(gust_tensor, GUST_VARIANT)
#define gust_tensor_bytes GUST_CAT(gust_tensor_bytes, GUST_VARIANT)
#define gust_reset GUST_CAT(gust_reset, GUST_VARIANT)
#define gust_threads_start GUST_CAT(gust_threads_start, GUST_VARIANT)
#define gust_set_threads GUST_CAT(gust_set_threads, GUST_VARIANT)
#define gust_worker GUST_CAT(gust_worker, GUST_VARIANT)
#define gust_parallel GUST_CAT(gust_parallel, GUST_VARIANT)
#endif
