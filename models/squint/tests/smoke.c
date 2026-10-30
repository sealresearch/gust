/* Opens the reader and reads one blank frame. */
#include "squint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    char err[200];
    if (sq_open(err, sizeof err)) { puts(err); return 1; }
    static uint8_t gray[1280 * 720];
    memset(gray, 255, sizeof gray);
    sq_read_options o;
    sq_default_options(&o);
    o.budget_ms = o.acquire_budget_ms = o.tracked_acquire_budget_ms = 1e300 * 10; /* infinity */
    static sq_read_result r;
    int rc = sq_read(gray, 1280, 720, &o, &r, err, sizeof err);
    printf("rc=%d accepted=%d reason=%s memory=%zu MB\n", rc, r.accepted, r.reason ? r.reason : "-", sq_memory_bytes() >> 20);
    return 0;
}
