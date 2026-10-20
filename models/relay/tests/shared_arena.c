/* Host sharing admission: a future state tensor or counted-cache op must retain a private arena. */
#undef NDEBUG
#include "relay_host.h"
#include "gust.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rd(const unsigned char *b) {
    return (unsigned)b[0] | (unsigned)b[1] << 8 | (unsigned)b[2] << 16 | (unsigned)b[3] << 24;
}
int main(int argc, char **argv) {
    assert(argc == 2);
    FILE *f = fopen(argv[1], "rb");
    assert(f);
    fseek(f, 0, SEEK_END);
    const long length = ftell(f);
    rewind(f);
    assert(length > 68);
    unsigned char *blob = malloc((size_t)length);
    assert(blob && fread(blob, 1, (size_t)length, f) == (size_t)length);
    fclose(f);
    const size_t bytes = rh_gmodel_share_bytes(blob, (size_t)length);
    assert(bytes && bytes == ((rd(blob + 40) + 63u) & ~(size_t)63u));
    assert(!rh_gmodel_share_bytes(blob, 63));
    const unsigned flags = blob[64 + 24];
    blob[64 + 24] |= 8;
    assert(!rh_gmodel_share_bytes(blob, (size_t)length));
    blob[64 + 24] = (unsigned char)flags;
    const unsigned offset = rd(blob + 32), opcode = blob[offset];
    blob[offset] = 46;
    assert(!rh_gmodel_share_bytes(blob, (size_t)length));
    blob[offset] = (unsigned char)opcode;
    void *allocation = malloc(bytes + 64);
    void *arena = (void *)(((uintptr_t)allocation + 63u) & ~(uintptr_t)63u);
    assert(allocation);
    int error;
    assert(!rh_gmodel_load_shared(blob, (size_t)length, &error, arena, bytes - 64));
    rh_gmodel *shared = rh_gmodel_load_shared(blob, (size_t)length, &error, arena, bytes);
    assert(shared && !error);
    rh_gmodel *private = rh_gmodel_load(blob, (size_t)length, &error);
    assert(private && !error);
    assert(rh_gmodel_bytes(private) - rh_gmodel_bytes(shared) == bytes);
    rh_gmodel_free(shared);
    /* Destroying the shared graph must not own/free the external allocation. */
    memset(arena, 0x55, bytes);
    rh_gmodel_free(private);
    free(allocation);
    free(blob);
    puts("Stateless sharing accepted; state/cache/truncated graphs and short arenas refused; private loading preserved.");
    return 0;
}
