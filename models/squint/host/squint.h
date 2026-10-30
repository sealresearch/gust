/* Squint: a QR code reader for shaky, blurred and badly focused camera frames, with its own decoder and a tracker that
   only returns a payload the current frame supports. This is the host: the whole pipeline in C on top of the Squint
   hosts (geometry, decoder, white patches, optical flow, the verifier with its evidence certificate and the acquisition).

   It is a port of the JavaScript release (Squint.read) and returns the same results for the same frames. The hosts keep
   their state in static memory, so there is one reader per process: sq_open fails while another is open. */
#ifndef SQUINT_H
#define SQUINT_H
#include <stddef.h>
#include <stdint.h>

#define SQ_MAX_WIDTH 1280
#define SQ_MAX_HEIGHT 720
#define SQ_MAX_PAYLOAD 2048
#define SQ_MAX_GRID 3249

typedef struct {
    int accepted;
    const char *reason; /* why not, when not accepted (static string) */
    const char *path;   /* which stage read it (static string) */
    int version;        /* QR version 1 to 10; 0 when unknown */
    int has_grid;       /* grid holds version x version modules (n x n) */
    int payload_len;
    int grid_len;
    double corners[8];  /* clockwise from the top left: x0, y0, x1, y1, ... */
    uint8_t payload[SQ_MAX_PAYLOAD];
    uint8_t grid[SQ_MAX_GRID];
} sq_read_result;

typedef struct {
    double time_ms;              /* frame time; must not go backwards within a stream */
    double budget_ms;            /* tracker time budget; INFINITY for none (results then depend on the frame only) */
    double acquire_budget_ms;    /* full-frame reader budget; INFINITY for none */
    double tracked_acquire_budget_ms;
    int tracked_acquire_every;   /* while a code is tracked, acquire on every n-th frame only */
    int enhance;                 /* alternate 1280 x 720 frames use a lighting-normalized view */
} sq_read_options;

void sq_default_options(sq_read_options *o);
/* Sets the hosts up. Returns 0, or nonzero with a message when a reader is already open. */
int sq_open(char *err, size_t errlen);
void sq_close(void);
/* Forgets tracked codes (a new camera stream or scene). */
void sq_reset(void);
/* Reads one gray frame (width * height bytes). Returns 0, or nonzero with a message when the frame is unusable. */
int sq_read(const uint8_t *gray, int width, int height, const sq_read_options *o, sq_read_result *out, char *err,
            size_t errlen);
/* Bytes the hosts hold (static memory of every instance). */
size_t sq_memory_bytes(void);
#endif
