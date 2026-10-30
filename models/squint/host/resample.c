/* Fixed-arena demo-only classical ROI resampling. No model/engine operator. */
#include <stdint.h>
#define PIXELS (1280 * 720)
#define SIDE   576
static uint8_t source[PIXELS * 4], gray[384 * 384], output[SIDE * SIDE * 4], sharp[SIDE * SIDE * 4];
static uint8_t luminance[PIXELS];
static float horizontal[SIDE * 384];
static int shape[2];
static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : x > hi ? hi : x;
}
static uint8_t byte(float x) {
    return x <= 0 ? 0 : x >= 255 ? 255 : (uint8_t)(x + .5f);
}
static void cubic(float x, float *w) {
    float a = -.75f, z = x + 1, t = 1 - x;
    w[0] = a * z * z * z - 5 * a * z * z + 8 * a * z - 4 * a;
    w[1] = (a + 2) * x * x * x - (a + 3) * x * x + 1;
    w[2] = (a + 2) * t * t * t - (a + 3) * t * t + 1;
    w[3] = 1 - w[0] - w[1] - w[2];
}
uintptr_t sr_input(void) {
    return (uintptr_t)source;
}
uintptr_t sr_output(void) {
    return (uintptr_t)output;
}
uintptr_t sr_sharp(void) {
    return (uintptr_t)sharp;
}
uintptr_t sr_shape(void) {
    return (uintptr_t)shape;
}
int sr_resize(int width, int height, int x, int y, int right, int bottom) {
    if (width < 1 || height < 1 || width > 1280 || height > 1280 || width * height > PIXELS || x < 0 || y < 0 ||
        right > width || bottom > height || right <= x || bottom <= y)
        return -1;
    int iw = right - x, ih = bottom - y, maximum = iw > ih ? iw : ih;
    float down = maximum > 384 ? 384.f / maximum : 1;
    int cw = (int)(iw * down + .5f), ch = (int)(ih * down + .5f);
    if (cw < 1) cw = 1;
    if (ch < 1) ch = 1;
    int ow = (int)(cw * 1.5f + .5f), oh = (int)(ch * 1.5f + .5f);
    shape[0] = ow;
    shape[1] = oh;
    for (int row = 0; row < ch; row++)
        for (int col = 0; col < cw; col++) {
            int sx = clampi(x + (int)((col + .5f) / down), x, right - 1),
                sy = clampi(y + (int)((row + .5f) / down), y, bottom - 1), i = (sy * width + sx) * 4;
            gray[row * cw + col] = byte(source[i] * .299f + source[i + 1] * .587f + source[i + 2] * .114f);
        }
    for (int col = 0; col < ow; col++) {
        float position = (col + .5f) * cw / ow - .5f;
        int base = (int)position;
        if (position < base) base--;
        float w[4];
        cubic(position - base, w);
        for (int row = 0; row < ch; row++) {
            float v = 0;
            for (int k = 0; k < 4; k++) v += w[k] * gray[row * cw + clampi(base + k - 1, 0, cw - 1)];
            horizontal[row * ow + col] = v;
        }
    }
    for (int row = 0; row < oh; row++) {
        float position = (row + .5f) * ch / oh - .5f;
        int base = (int)position;
        if (position < base) base--;
        float w[4];
        cubic(position - base, w);
        for (int col = 0; col < ow; col++) {
            float v = 0;
            for (int k = 0; k < 4; k++) v += w[k] * horizontal[clampi(base + k - 1, 0, ch - 1) * ow + col];
            int i = (row * ow + col) * 4;
            output[i] = output[i + 1] = output[i + 2] = byte(v);
            output[i + 3] = 255;
        }
    }
    return 0;
}
int sr_sharpen(float amount) {
    if (!(amount >= 0 && amount <= 4)) return -1;
    int width = shape[0], height = shape[1];
    if (width < 1 || height < 1) return -1;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * 4, sum = output[(y * width + clampi(x - 1, 0, width - 1)) * 4] +
                                               output[(y * width + clampi(x + 1, 0, width - 1)) * 4] +
                                               output[(clampi(y - 1, 0, height - 1) * width + x) * 4] +
                                               output[(clampi(y + 1, 0, height - 1) * width + x) * 4];
            uint8_t v = byte(output[i] + amount * (output[i] - sum / 4.f));
            sharp[i] = sharp[i + 1] = sharp[i + 2] = v;
            sharp[i + 3] = 255;
        }
    return 0;
}

uintptr_t sr_gray(void) {
    return (uintptr_t)luminance;
}
int sr_luminance(int width, int height) {
    if (width < 1 || height < 1 || width > 1280 || height > 1280 || width * height > PIXELS) return -1;
    for (int i = 0; i < width * height; i++)
        luminance[i] =
            (uint8_t)((306u * source[i * 4] + 601u * source[i * 4 + 1] + 117u * source[i * 4 + 2] + 512u) >> 10);
    return 0;
}
