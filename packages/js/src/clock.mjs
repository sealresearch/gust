// The WASI clock the model modules import, shared by the main thread and the operator workers.
/** WASI clock_time_get: nanoseconds, from the wall clock for id 0 and a monotonic clock for the others. */
export function clockTimeGet(memory, id, out) {
  const ns = id === 0 ? BigInt(Date.now()) * 1000000n : BigInt(Math.round(performance.now() * 1e6));
  new DataView(memory.buffer).setBigUint64(out, ns, true);
  return 0;
}

