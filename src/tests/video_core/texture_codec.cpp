// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// The NEON tile codec against the scalar reference, for every (direction, format, converted)
// entry of the Morton tables, on random data. On hosts without NEON both instantiations are
// the scalar code and the test is a tautology.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/rasterizer_cache/texture_codec.h"

namespace {

using VideoCore::PixelFormat;

constexpr u32 kWidth = 16;
constexpr u32 kHeight = 24;

std::vector<u8> RandomBytes(std::size_t size, u32 seed) {
    std::mt19937 rng{seed};
    std::vector<u8> v(size);
    for (auto& b : v) {
        b = static_cast<u8>(rng());
    }
    return v;
}

template <bool morton_to_linear, PixelFormat format, bool converted>
void Check(u32 start_offset, u32 end_offset) {
    constexpr u32 bpp = VideoCore::GetFormatBpp(format);
    constexpr u32 linear_bpp = converted ? 4 : VideoCore::GetFormatBytesPerPixel(format);
    const u32 tiled_size = kWidth * kHeight * bpp / 8;
    const u32 linear_size = kWidth * kHeight * linear_bpp;
    if (end_offset == 0) {
        end_offset = tiled_size;
    }

    const auto tiled_in = RandomBytes(tiled_size, 1);
    const auto linear_in = RandomBytes(linear_size, 2);

    std::vector<u8> ref_tiled = tiled_in, simd_tiled = tiled_in;
    std::vector<u8> ref_linear = linear_in, simd_linear = linear_in;
    if constexpr (morton_to_linear) {
        std::fill(ref_linear.begin(), ref_linear.end(), 0xCD);
        std::fill(simd_linear.begin(), simd_linear.end(), 0xCD);
    } else {
        std::fill(ref_tiled.begin(), ref_tiled.end(), 0xCD);
        std::fill(simd_tiled.begin(), simd_tiled.end(), 0xCD);
    }

    // Both directions receive a tiled span whose first byte is the one at start_offset.
    const u32 tiled_span_begin = start_offset;
    const auto span_of = [&](std::vector<u8>& v) {
        return std::span<u8>(v.data() + tiled_span_begin, v.size() - tiled_span_begin);
    };

    VideoCore::MortonCopy<morton_to_linear, format, converted, false>(
        kWidth, kHeight, start_offset, end_offset, ref_linear, span_of(ref_tiled));
    VideoCore::MortonCopy<morton_to_linear, format, converted, true>(
        kWidth, kHeight, start_offset, end_offset, simd_linear, span_of(simd_tiled));

    INFO("format " << VideoCore::PixelFormatAsString(format) << " to_linear " << morton_to_linear
                   << " converted " << converted << " range " << start_offset << ".."
                   << end_offset);
    if constexpr (morton_to_linear) {
        REQUIRE(ref_linear == simd_linear);
    } else {
        REQUIRE(ref_tiled == simd_tiled);
    }
}

template <PixelFormat format, bool converted>
void CheckBoth() {
    constexpr u32 tile_bytes = VideoCore::GetFormatBpp(format) * 64 / 8;
    constexpr bool has_encoder = format != PixelFormat::ETC1 && format != PixelFormat::ETC1A4;
    Check<true, format, converted>(0, 0);
    Check<true, format, converted>(tile_bytes, 3 * tile_bytes);
    if constexpr (has_encoder) {
        Check<false, format, converted>(0, 0);
        // Unaligned download ranges take the temp-tile path on both sides.
        Check<false, format, converted>(tile_bytes / 2 + 1, 5 * tile_bytes - tile_bytes / 4);
    }
}

} // namespace

TEST_CASE("NEON tile codec matches the scalar reference", "[video_core]") {
    CheckBoth<PixelFormat::RGBA8, false>();
    CheckBoth<PixelFormat::RGBA8, true>();
    CheckBoth<PixelFormat::RGB8, false>();
    CheckBoth<PixelFormat::RGB8, true>();
    CheckBoth<PixelFormat::RGB5A1, false>();
    CheckBoth<PixelFormat::RGB5A1, true>();
    CheckBoth<PixelFormat::RGB565, false>();
    CheckBoth<PixelFormat::RGB565, true>();
    CheckBoth<PixelFormat::RGBA4, false>();
    CheckBoth<PixelFormat::RGBA4, true>();
    CheckBoth<PixelFormat::IA8, false>();
    CheckBoth<PixelFormat::RG8, false>();
    CheckBoth<PixelFormat::I8, false>();
    CheckBoth<PixelFormat::A8, false>();
    CheckBoth<PixelFormat::IA4, false>();
    CheckBoth<PixelFormat::I4, false>();
    CheckBoth<PixelFormat::A4, false>();
    CheckBoth<PixelFormat::ETC1, false>();
    CheckBoth<PixelFormat::ETC1A4, false>();
    CheckBoth<PixelFormat::D16, false>();
    CheckBoth<PixelFormat::D16, true>();
    CheckBoth<PixelFormat::D24, false>();
    CheckBoth<PixelFormat::D24, true>();
    CheckBoth<PixelFormat::D24S8, false>();
}

namespace {

template <bool morton_to_linear, PixelFormat format, bool converted, bool simd>
double NsPerTile() {
    constexpr u32 bpp = VideoCore::GetFormatBpp(format);
    constexpr u32 linear_bpp = converted ? 4 : VideoCore::GetFormatBytesPerPixel(format);
    constexpr u32 w = 256, h = 256;
    auto tiled = RandomBytes(w * h * bpp / 8, 3);
    auto linear = RandomBytes(w * h * linear_bpp, 4);
    const int iterations = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        VideoCore::MortonCopy<morton_to_linear, format, converted, simd>(
            w, h, 0, static_cast<u32>(tiled.size()), linear, tiled);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / (iterations * (w / 8) * (h / 8));
}

template <bool morton_to_linear, PixelFormat format, bool converted>
void Bench(const char* name) {
    const double scalar = NsPerTile<morton_to_linear, format, converted, false>();
    const double simd = NsPerTile<morton_to_linear, format, converted, true>();
    std::printf("%-22s scalar %7.0f ns/tile  neon %7.0f ns/tile  x%.1f\n", name, scalar, simd,
                scalar / simd);
}

} // namespace

TEST_CASE("NEON tile codec throughput", "[video_core][!benchmark]") {
    Bench<true, PixelFormat::RGBA8, false>("RGBA8 decode");
    Bench<true, PixelFormat::RGBA8, true>("RGBA8 decode conv");
    Bench<true, PixelFormat::RGB8, true>("RGB8 decode conv");
    Bench<true, PixelFormat::RGB565, false>("RGB565 decode");
    Bench<true, PixelFormat::RGB565, true>("RGB565 decode conv");
    Bench<true, PixelFormat::RGBA4, true>("RGBA4 decode conv");
    Bench<true, PixelFormat::IA8, false>("IA8 decode");
    Bench<true, PixelFormat::I8, false>("I8 decode");
    Bench<true, PixelFormat::IA4, false>("IA4 decode");
    Bench<true, PixelFormat::I4, false>("I4 decode");
    Bench<true, PixelFormat::ETC1, false>("ETC1 decode");
    Bench<true, PixelFormat::ETC1A4, false>("ETC1A4 decode");
    Bench<true, PixelFormat::D24S8, false>("D24S8 decode");
    Bench<false, PixelFormat::RGBA8, false>("RGBA8 encode");
    Bench<false, PixelFormat::RGB565, true>("RGB565 encode conv");
    Bench<false, PixelFormat::D24S8, false>("D24S8 encode");
}
