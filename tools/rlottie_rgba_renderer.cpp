/*
 * MojiLex lossless rlottie adapter.
 *
 * Build this file against the pinned Samsung/rlottie revision documented in
 * docs/media-prerequisites.md. The output format is deliberately tiny and
 * private: a fixed little-endian header followed by tightly packed straight
 * RGBA8 frames. No image encoder is involved.
 */

#ifdef _WIN32
// The pinned upstream header has no separate static-consumer switch. Avoid
// dllimport indirection when linking rlottie.lib into this standalone helper.
#define RLOTTIE_BUILD
#pragma warning(push)
#pragma warning(disable : 4251)  // upstream export annotation exposes a private STL member
#endif
#include <rlottie.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr char kMagic[8] = {'M', 'L', 'X', 'R', 'G', 'B', 'A', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kMaxPixels = 16000000ULL;
constexpr std::uint64_t kMaxFrames = 600ULL;
constexpr std::uint64_t kMaxPayloadBytes = 1500000000ULL;

bool parse_positive(const char *text, std::uint64_t maximum, std::uint64_t &result) {
    if (text == nullptr || *text == '\0') return false;
    std::uint64_t value = 0;
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return false;
        const std::uint64_t digit = static_cast<std::uint64_t>(*cursor - '0');
        if (value > (maximum - digit) / 10ULL) return false;
        value = value * 10ULL + digit;
    }
    if (value == 0 || value > maximum) return false;
    result = value;
    return true;
}

void write_u32_le(std::ostream &stream, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        stream.put(static_cast<char>((value >> shift) & 0xffU));
    }
}

void write_u64_le(std::ostream &stream, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        stream.put(static_cast<char>((value >> shift) & 0xffULL));
    }
}

std::uint8_t unpremultiply(std::uint8_t channel, std::uint8_t alpha) {
    if (alpha == 0) return 0;
    if (alpha == 255) return channel;
    const std::uint32_t straight =
        (static_cast<std::uint32_t>(channel) * 255U + alpha / 2U) / alpha;
    return static_cast<std::uint8_t>(std::min<std::uint32_t>(255U, straight));
}

int fail(const char *message) {
    std::cerr << message << '\n';
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        return fail("usage: mojilex-rlottie-rgba INPUT.json WIDTH HEIGHT FRAMES");
    }

    std::uint64_t width64 = 0;
    std::uint64_t height64 = 0;
    std::uint64_t frames64 = 0;
    if (!parse_positive(argv[2], kMaxPixels, width64) ||
        !parse_positive(argv[3], kMaxPixels, height64) ||
        !parse_positive(argv[4], kMaxFrames, frames64)) {
        return fail("invalid expected dimensions or frame count");
    }
    if (width64 > kMaxPixels / height64) return fail("expected canvas exceeds pixel limit");
    const std::uint64_t pixels = width64 * height64;
    if (pixels > kMaxPixels || pixels > kMaxPayloadBytes / 4ULL / frames64) {
        return fail("expected RGBA stream exceeds payload limit");
    }
    const std::uint64_t frame_bytes = pixels * 4ULL;
    const std::uint64_t payload_bytes = frame_bytes * frames64;

    rlottie::configureModelCacheSize(0);
    auto animation = rlottie::Animation::loadFromFile(argv[1], false);
    if (!animation) return fail("rlottie could not load validated JSON");

    std::size_t native_width = 0;
    std::size_t native_height = 0;
    animation->size(native_width, native_height);
    if (native_width != width64 || native_height != height64) {
        return fail("rlottie canvas does not match validated dimensions");
    }
    // This pinned revision reports end-start+1, while Telegram's playable
    // interval is op-ip. Accept the extra terminal count, but never render it.
    if (animation->totalFrame() < frames64) {
        return fail("rlottie timeline is shorter than the validated interval");
    }

#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        return fail("cannot configure binary RGBA output stream");
    }
#endif
    std::ostream &output = std::cout;
    output.write(kMagic, static_cast<std::streamsize>(sizeof(kMagic)));
    write_u32_le(output, kVersion);
    write_u32_le(output, static_cast<std::uint32_t>(width64));
    write_u32_le(output, static_cast<std::uint32_t>(height64));
    write_u32_le(output, static_cast<std::uint32_t>(frames64));
    write_u64_le(output, payload_bytes);
    if (!output) return fail("cannot write RGBA stream header");

    std::vector<std::uint32_t> argb(static_cast<std::size_t>(pixels));
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(frame_bytes));
    rlottie::Surface surface(
        argb.data(), static_cast<std::size_t>(width64), static_cast<std::size_t>(height64),
        static_cast<std::size_t>(width64 * 4ULL));
    for (std::uint64_t frame = 0; frame < frames64; ++frame) {
        std::fill(argb.begin(), argb.end(), 0U);
        animation->renderSync(static_cast<std::size_t>(frame), surface, false);
        for (std::size_t pixel_index = 0; pixel_index < argb.size(); ++pixel_index) {
            const std::uint32_t pixel = argb[pixel_index];
            const auto alpha = static_cast<std::uint8_t>((pixel >> 24U) & 0xffU);
            const auto red = static_cast<std::uint8_t>((pixel >> 16U) & 0xffU);
            const auto green = static_cast<std::uint8_t>((pixel >> 8U) & 0xffU);
            const auto blue = static_cast<std::uint8_t>(pixel & 0xffU);
            const std::size_t offset = pixel_index * 4U;
            rgba[offset] = unpremultiply(red, alpha);
            rgba[offset + 1U] = unpremultiply(green, alpha);
            rgba[offset + 2U] = unpremultiply(blue, alpha);
            rgba[offset + 3U] = alpha;
        }
        output.write(reinterpret_cast<const char *>(rgba.data()),
                     static_cast<std::streamsize>(rgba.size()));
        if (!output) return fail("cannot write complete RGBA frame");
    }
    output.flush();
    if (!output) return fail("cannot finalize RGBA output stream");
    return 0;
}
