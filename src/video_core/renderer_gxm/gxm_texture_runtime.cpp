// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include "common/alignment.h"
#include "common/color.h"
#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "common/settings.h"
#include "video_core/custom_textures/material.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_shader_util.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/gxm_texture_runtime.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace GxmRenderer {

using VideoCore::PixelFormat;
using VideoCore::SurfaceFlagBits;
using VideoCore::SurfaceType;
using VideoCore::TextureType;

namespace {

constexpr u32 MappedChunk = 8 * 1024 * 1024;
constexpr u32 MappedBudget = 48 * 1024 * 1024;
constexpr u32 CdramChunk = 8 * 1024 * 1024;
constexpr u32 CdramBudget = 96 * 1024 * 1024;
/// Surfaces at least this big go to CDRAM (256 KiB granularity makes it wasteful below).

constexpr SceGxmTextureFormat TexAbgr8 = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR;
constexpr SceGxmColorFormat ColAbgr8 = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR;
constexpr SceGxmTransferFormat TrAbgr8 = SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR;
constexpr SceGxmTransferFormat TrRaw16 = SCE_GXM_TRANSFER_FORMAT_RAW16;
constexpr SceGxmTransferFormat TrRaw32 = SCE_GXM_TRANSFER_FORMAT_RAW32;

// The cache's formats in GXM terms. The converted ones arrive from the codec as bytes
// R,G,B,A (GXM's "ABGR" word); the 16-bit colour formats keep the PICA bit layout, for which
// GXM has matching texture and colour formats but no matching PTLA format (its 16-bit
// formats are the BGR-ordered ones), so their transfers are RAW.
constexpr std::array<GxmFormat, 18> FormatTable = {{
    // RGBA8: guest bytes A,B,G,R; converted to R,G,B,A.
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, true, true},
    // RGB8: converted to RGBA8.
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, true, true},
    // RGB5A1: r<<11 | g<<6 | b<<1 | a
    {SCE_GXM_TEXTURE_FORMAT_U5U5U5U1_RGBA,
     SCE_GXM_COLOR_FORMAT_U5U5U5U1_RGBA,
     {},
     TrRaw16,
     2,
     false,
     false},
    // RGB565: r<<11 | g<<5 | b
    {SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB,
     SCE_GXM_COLOR_FORMAT_U5U6U5_RGB,
     {},
     TrRaw16,
     2,
     false,
     false},
    // RGBA4: r<<12 | g<<8 | b<<4 | a
    {SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_RGBA,
     SCE_GXM_COLOR_FORMAT_U4U4U4U4_RGBA,
     {},
     TrRaw16,
     2,
     false,
     false},
    // Texture-only formats decode to RGBA8 (bytes R,G,B,A).
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // IA8
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // RG8
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // I8
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // A8
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // IA4
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // I4
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // A4
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // ETC1
    {TexAbgr8, ColAbgr8, {}, TrAbgr8, 4, false, true}, // ETC1A4
    // D16: kept as GXM's S8D24 (stencil unused), the guest's 16-bit unorm widened to 24 bits
    // on upload and narrowed on download here (the codec leaves D16 alone). Console,
    // 2026-09-04: a real linear D16 surface with force load/store never passed a depth test
    // (citro2d's targets: gpusprites and 2d_shapes drew nothing), while linear S8D24 handled
    // the same way is fine. The one D16 user found in the wild (OpenLara) uses the tiled
    // layout; no SDK sample or vitaGL touches D16 at all. S8D24 is the only linear depth
    // layout the console has proven, so it is the backing.
    {SCE_GXM_TEXTURE_FORMAT_X8U24_SD, {}, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24, TrRaw32, 4, false, false},
    {}, // 15: invalid
    // D24: converted to float by the codec, GXM's DF32.
    {SCE_GXM_TEXTURE_FORMAT_F32_R, {}, SCE_GXM_DEPTH_STENCIL_FORMAT_DF32, TrRaw32, 4, true, false},
    // D24S8: the guest word (stencil in the top byte) is GXM's S8D24; the codec hands it
    // over rotated into GL order and it is rotated back on upload.
    {SCE_GXM_TEXTURE_FORMAT_X8U24_SD,
     {},
     SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
     TrRaw32,
     4,
     false,
     false},
}};

/**
 * Row converters between a surface's own storage and a common intermediate, for the CPU blit.
 *
 * The PTLA cannot do these: its only 16-bit formats are the BGR-ordered ones, and the guest's
 * are RGB-ordered, so a colour transfer between two different pixel formats has no hardware
 * path at all. The cache asks for them anyway - CheckFormatsBlittable allows any colour or
 * texture format to reach any other, and a title that renders to RGBA8 and display-transfers
 * to an RGB565 framebuffer does exactly that every frame. Refusing left the destination stale,
 * which is a black screen.
 *
 * The intermediate is RGBA8 in the same byte order the surfaces use (R, G, B, A ascending),
 * so the 4-byte formats pass straight through. Depth converts through a float.
 */
using RowUnpack = void (*)(const u8* src, u8* rgba, u32 count);
using RowPack = void (*)(const u8* rgba, u8* dst, u32 count);

void UnpackRgba8(const u8* src, u8* rgba, u32 count) {
    std::memcpy(rgba, src, count * 4);
}
void PackRgba8(const u8* rgba, u8* dst, u32 count) {
    std::memcpy(dst, rgba, count * 4);
}

void UnpackRgb565(const u8* src, u8* rgba, u32 count) {
    u32 i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // Eight texels a pass. Widening 5 and 6 bits to 8 is the usual replicate-the-high-bits
    // trick ((v << 3) | (v >> 2) and (v << 2) | (v >> 4)), which is what Convert5To8 and
    // Convert6To8 do scalar.
    for (; i + 8 <= count; i += 8) {
        const uint16x8_t p = vld1q_u16(reinterpret_cast<const u16*>(src + i * 2));
        const uint16x8_t r5 = vshrq_n_u16(p, 11);
        const uint16x8_t g6 = vandq_u16(vshrq_n_u16(p, 5), vdupq_n_u16(63));
        const uint16x8_t b5 = vandq_u16(p, vdupq_n_u16(31));
        uint8x8x4_t out;
        out.val[0] = vmovn_u16(vorrq_u16(vshlq_n_u16(r5, 3), vshrq_n_u16(r5, 2)));
        out.val[1] = vmovn_u16(vorrq_u16(vshlq_n_u16(g6, 2), vshrq_n_u16(g6, 4)));
        out.val[2] = vmovn_u16(vorrq_u16(vshlq_n_u16(b5, 3), vshrq_n_u16(b5, 2)));
        out.val[3] = vdup_n_u8(255);
        vst4_u8(rgba + i * 4, out);
    }
#endif
    for (; i < count; i++) {
        u16 pixel;
        std::memcpy(&pixel, src + i * 2, 2);
        rgba[i * 4 + 0] = Common::Color::Convert5To8((pixel >> 11) & 0x1F);
        rgba[i * 4 + 1] = Common::Color::Convert6To8((pixel >> 5) & 0x3F);
        rgba[i * 4 + 2] = Common::Color::Convert5To8(pixel & 0x1F);
        rgba[i * 4 + 3] = 255;
    }
}
void PackRgb565(const u8* rgba, u8* dst, u32 count) {
    u32 i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // Narrowing is a plain right shift, which is what Convert8To5 and Convert8To6 do.
    for (; i + 8 <= count; i += 8) {
        const uint8x8x4_t in = vld4_u8(rgba + i * 4);
        const uint16x8_t r = vshll_n_u8(in.val[0], 8);
        const uint16x8_t g = vshll_n_u8(in.val[1], 8);
        const uint16x8_t b = vshll_n_u8(in.val[2], 8);
        const uint16x8_t packed = vorrq_u16(vandq_u16(r, vdupq_n_u16(0xF800)),
                                            vorrq_u16(vandq_u16(vshrq_n_u16(g, 5),
                                                                vdupq_n_u16(0x07E0)),
                                                      vshrq_n_u16(b, 11)));
        vst1q_u16(reinterpret_cast<u16*>(dst + i * 2), packed);
    }
#endif
    for (; i < count; i++) {
        const u16 pixel = static_cast<u16>((Common::Color::Convert8To5(rgba[i * 4 + 0]) << 11) |
                                           (Common::Color::Convert8To6(rgba[i * 4 + 1]) << 5) |
                                           Common::Color::Convert8To5(rgba[i * 4 + 2]));
        std::memcpy(dst + i * 2, &pixel, 2);
    }
}

void UnpackRgb5A1(const u8* src, u8* rgba, u32 count) {
    for (u32 i = 0; i < count; i++) {
        u16 pixel;
        std::memcpy(&pixel, src + i * 2, 2);
        rgba[i * 4 + 0] = Common::Color::Convert5To8((pixel >> 11) & 0x1F);
        rgba[i * 4 + 1] = Common::Color::Convert5To8((pixel >> 6) & 0x1F);
        rgba[i * 4 + 2] = Common::Color::Convert5To8((pixel >> 1) & 0x1F);
        rgba[i * 4 + 3] = Common::Color::Convert1To8(pixel & 0x1);
    }
}
void PackRgb5A1(const u8* rgba, u8* dst, u32 count) {
    for (u32 i = 0; i < count; i++) {
        const u16 pixel = static_cast<u16>((Common::Color::Convert8To5(rgba[i * 4 + 0]) << 11) |
                                           (Common::Color::Convert8To5(rgba[i * 4 + 1]) << 6) |
                                           (Common::Color::Convert8To5(rgba[i * 4 + 2]) << 1) |
                                           Common::Color::Convert8To1(rgba[i * 4 + 3]));
        std::memcpy(dst + i * 2, &pixel, 2);
    }
}

void UnpackRgba4(const u8* src, u8* rgba, u32 count) {
    for (u32 i = 0; i < count; i++) {
        u16 pixel;
        std::memcpy(&pixel, src + i * 2, 2);
        rgba[i * 4 + 0] = Common::Color::Convert4To8((pixel >> 12) & 0xF);
        rgba[i * 4 + 1] = Common::Color::Convert4To8((pixel >> 8) & 0xF);
        rgba[i * 4 + 2] = Common::Color::Convert4To8((pixel >> 4) & 0xF);
        rgba[i * 4 + 3] = Common::Color::Convert4To8(pixel & 0xF);
    }
}
void PackRgba4(const u8* rgba, u8* dst, u32 count) {
    for (u32 i = 0; i < count; i++) {
        const u16 pixel = static_cast<u16>((Common::Color::Convert8To4(rgba[i * 4 + 0]) << 12) |
                                           (Common::Color::Convert8To4(rgba[i * 4 + 1]) << 8) |
                                           (Common::Color::Convert8To4(rgba[i * 4 + 2]) << 4) |
                                           Common::Color::Convert8To4(rgba[i * 4 + 3]));
        std::memcpy(dst + i * 2, &pixel, 2);
    }
}

/// D16's surface is S8D24: the 16-bit unorm widened to 24 bits (bit-replicated, so 0xFFFF
/// is 0xFFFFFF) with a zero stencil byte, met only at upload and download. A converting
/// blit never sees D16.
void D16ToS8D24(const u8* src, u8* out, u32 count) {
    for (u32 i = 0; i < count; i++) {
        u16 pixel;
        std::memcpy(&pixel, src + i * 2, 2);
        const u32 word = (static_cast<u32>(pixel) << 8) | (pixel >> 8);
        std::memcpy(out + i * 4, &word, 4);
    }
}
void S8D24ToD16(const u8* in, u8* dst, u32 count) {
    for (u32 i = 0; i < count; i++) {
        u32 word;
        std::memcpy(&word, in + i * 4, 4);
        const u16 pixel = static_cast<u16>((word & 0xFFFFFF) >> 8);
        std::memcpy(dst + i * 2, &pixel, 2);
    }
}

/// Null when the format needs no conversion to reach the intermediate (the 4-byte colour
/// storage and D24's float), and null for the formats a converting blit never sees.
RowUnpack UnpackFor(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGB565:
        return UnpackRgb565;
    case PixelFormat::RGB5A1:
        return UnpackRgb5A1;
    case PixelFormat::RGBA4:
        return UnpackRgba4;
    default:
        return UnpackRgba8;
    }
}
RowPack PackFor(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGB565:
        return PackRgb565;
    case PixelFormat::RGB5A1:
        return PackRgb5A1;
    case PixelFormat::RGBA4:
        return PackRgba4;
    default:
        return PackRgba8;
    }
}

constexpr bool IsDepthType(SurfaceType type) {
    return type == SurfaceType::Depth || type == SurfaceType::DepthStencil;
}

/// Row pitch in texels GXM implies for a linear texture of `width`, and the wider pitch a
/// depth-stencil surface needs (a multiple of 32 samples).
constexpr u32 PitchFor(u32 width, bool depth) {
    return Common::AlignUp(width, depth ? 32u : 8u);
}

void RotateWordsRight8(u32* dst, const u32* src, u32 count) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    u32 i = 0;
    for (; i + 4 <= count; i += 4) {
        const uint32x4_t v = vld1q_u32(src + i);
        vst1q_u32(dst + i, vorrq_u32(vshrq_n_u32(v, 8), vshlq_n_u32(v, 24)));
    }
    for (; i < count; i++) {
        dst[i] = (src[i] >> 8) | (src[i] << 24);
    }
#else
    for (u32 i = 0; i < count; i++) {
        dst[i] = (src[i] >> 8) | (src[i] << 24);
    }
#endif
}

void RotateWordsLeft8(u32* dst, const u32* src, u32 count) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    u32 i = 0;
    for (; i + 4 <= count; i += 4) {
        const uint32x4_t v = vld1q_u32(src + i);
        vst1q_u32(dst + i, vorrq_u32(vshlq_n_u32(v, 8), vshrq_n_u32(v, 24)));
    }
    for (; i < count; i++) {
        dst[i] = (src[i] << 8) | (src[i] >> 24);
    }
#else
    for (u32 i = 0; i < count; i++) {
        dst[i] = (src[i] << 8) | (src[i] >> 24);
    }
#endif
}

u8 ToU8(float v) {
    return static_cast<u8>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
}

/// The clear value packed the way the surface stores a texel.
u32 PackClear(const GxmFormat& format, PixelFormat pixel_format,
              const VideoCore::ClearValue& value) {
    switch (pixel_format) {
    case PixelFormat::RGB5A1: {
        const u32 r = ToU8(value.color.r()) >> 3, g = ToU8(value.color.g()) >> 3;
        const u32 b = ToU8(value.color.b()) >> 3, a = ToU8(value.color.a()) >> 7;
        return (r << 11) | (g << 6) | (b << 1) | a;
    }
    case PixelFormat::RGB565: {
        const u32 r = ToU8(value.color.r()) >> 3, g = ToU8(value.color.g()) >> 2;
        const u32 b = ToU8(value.color.b()) >> 3;
        return (r << 11) | (g << 5) | b;
    }
    case PixelFormat::RGBA4: {
        const u32 r = ToU8(value.color.r()) >> 4, g = ToU8(value.color.g()) >> 4;
        const u32 b = ToU8(value.color.b()) >> 4, a = ToU8(value.color.a()) >> 4;
        return (r << 12) | (g << 8) | (b << 4) | a;
    }
    case PixelFormat::D16: {
        const u32 d = static_cast<u32>(
            std::clamp(static_cast<double>(value.depth), 0.0, 1.0) * 65535.0 + 0.5);
        return (d << 8) | (d >> 8);
    }
    case PixelFormat::D24: {
        u32 bits;
        std::memcpy(&bits, &value.depth, sizeof(bits));
        return bits;
    }
    case PixelFormat::D24S8: {
        // In double: 16777215.5f is not a float, it rounds up to 2^24, and masked to 24
        // bits a clear to far (1.0) became a clear to near. SM3DL's world failed LESS
        // against it every frame (console and Vita3K alike, 2026-09-04).
        const u32 d = static_cast<u32>(
            std::clamp(static_cast<double>(value.depth), 0.0, 1.0) * 16777215.0 + 0.5);
        return (static_cast<u32>(value.stencil) << 24) | (d & 0xFFFFFF);
    }
    default: // RGBA8 storage: bytes R,G,B,A
        return static_cast<u32>(ToU8(value.color.r())) |
               (static_cast<u32>(ToU8(value.color.g())) << 8) |
               (static_cast<u32>(ToU8(value.color.b())) << 16) |
               (static_cast<u32>(ToU8(value.color.a())) << 24);
    }
}

} // Anonymous namespace

const GxmFormat& FormatOf(PixelFormat format) {
    const auto index = static_cast<std::size_t>(format);
    return FormatTable[index < FormatTable.size() ? index : 15];
}

// ---- GpuPool --------------------------------------------------------------------------------

GpuPool::GpuPool(Kind kind_, u32 chunk_size_, u32 budget_)
    : kind{kind_}, chunk_size{chunk_size_}, budget{budget_} {}

GpuPool::~GpuPool() {
    if (!HasDevice()) {
        return;
    }
    for (auto& chunk : chunks) {
        Device().free_block(chunk.base);
    }
}

bool GpuPool::AddChunk(u32 min_size) {
    if (reserved + min_size > budget) {
        return false;
    }
    const auto& device = Device();
    const bool cdram = kind == Kind::Cdram;
    // A slab of chunk_size, or the largest the pool below can still hand out when that is
    // less: its last megabytes are not a whole slab, and asking for one anyway leaves them
    // unused for the rest of the run.
    u32 size = std::min(std::max(chunk_size, min_size), budget - reserved);
    if (device.largest_free != nullptr) {
        const u32 available = device.largest_free(cdram);
        if (available >= min_size) {
            size = std::min(size, available);
        }
    }
    void* base = cdram ? device.alloc_cdram(size) : device.alloc_mapped(size);
    if (base == nullptr) {
        return false;
    }
    Chunk chunk;
    chunk.base = static_cast<u8*>(base);
    chunk.size = size;
    chunk.free.emplace(0u, size);
    chunks.push_back(std::move(chunk));
    reserved += size;
    return true;
}

void* GpuPool::Alloc(u32 size, u32 align) {
    if (size == 0 || !HasDevice()) {
        return nullptr;
    }
    size = Common::AlignUp(size, align);
    for (int attempt = 0; attempt < 2; attempt++) {
        for (std::size_t ci = 0; ci < chunks.size(); ci++) {
            Chunk& chunk = chunks[ci];
            for (auto it = chunk.free.begin(); it != chunk.free.end(); ++it) {
                const u32 start = Common::AlignUp(it->first, align);
                const u32 end = it->first + it->second;
                if (start + size > end) {
                    continue;
                }
                const u32 block_start = it->first;
                const u32 block_size = it->second;
                chunk.free.erase(it);
                if (start > block_start) {
                    chunk.free.emplace(block_start, start - block_start);
                }
                if (start + size < end) {
                    chunk.free.emplace(start + size, end - (start + size));
                }
                u8* ptr = chunk.base + start;
                live.emplace(ptr, std::make_pair(ci, size));
                used += size;
                return ptr;
            }
        }
        if (!AddChunk(size)) {
            break;
        }
    }
    // Once a pool is full it is full for every allocation of the frame, and the caller has
    // somewhere else to go (CDRAM falls back to LPDDR); saying so on each one cost more than
    // the failure.
    if (++exhausted_reports <= 4 || exhausted_reports % 1024 == 0) {
        LOG_ERROR(Render,
                  "GXM {} pool exhausted: {} KiB asked, {} KiB used of {} KiB budget, {} "
                  "refusals so far",
                  kind == Kind::Cdram ? "CDRAM" : "LPDDR", size / 1024, used / 1024,
                  budget / 1024, exhausted_reports);
    }
    return nullptr;
}

void GpuPool::Free(void* ptr) {
    const auto it = live.find(static_cast<u8*>(ptr));
    if (it == live.end()) {
        return;
    }
    const auto [ci, size] = it->second;
    live.erase(it);
    used -= size;
    Chunk& chunk = chunks[ci];
    u32 start = static_cast<u32>(static_cast<u8*>(ptr) - chunk.base);
    u32 end = start + size;
    // Coalesce with the neighbours.
    auto next = chunk.free.lower_bound(start);
    if (next != chunk.free.end() && next->first == end) {
        end += next->second;
        next = chunk.free.erase(next);
    }
    if (next != chunk.free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == start) {
            start = prev->first;
            chunk.free.erase(prev);
        }
    }
    chunk.free.emplace(start, end - start);
}

// ---- TextureRuntime -------------------------------------------------------------------------

TextureRuntime::TextureRuntime(VideoCore::RendererBase&)
    : mapped_pool{GpuPool::Kind::Mapped, MappedChunk, MappedBudget},
      cdram_pool{GpuPool::Kind::Cdram, CdramChunk, CdramBudget} {}

TextureRuntime::~TextureRuntime() = default;

u64 TextureRuntime::BeginSceneEpoch(SceGxmNotification* out) {
    if (!epoch_words_ready && HasDevice() && Device().alloc_notification != nullptr) {
        bool all = true;
        for (auto& word : epoch_words) {
            word = Device().alloc_notification();
            if (word == nullptr) {
                all = false;
                break;
            }
            *word = 0;
        }
        epoch_words_ready = all;
    }
    scene_serial++;
    if (epoch_words_ready) {
        out->address = const_cast<u32*>(EpochWord(scene_serial));
        // The notification is 32 bits and the serial is 64. completed_floor is raised by every
        // full wait, and a wait happens at least once a frame, so the live range of serials
        // never comes near a wrap - the low word identifies the scene unambiguously.
        out->value = static_cast<u32>(scene_serial);
    } else {
        // No notification slots: nothing can be proved finished, so every check waits.
        out->address = nullptr;
        out->value = 0;
    }
    return scene_serial;
}

void TextureRuntime::WaitForEpoch(u64 serial) {
    // Only a scene that has been handed to sceGxmEndScene can ever be waited for. A claimed
    // but unsubmitted serial has no notification coming: nothing writes it, and the thread
    // that would submit it is this one. Surface::Allocate reaches here through ReapFreed
    // without ending the open scene, and the renamed block it is waiting on carries exactly
    // that scene's serial - so clamping to the newest claimed scene, as this used to, parked
    // the render thread forever the first time both pools refused an allocation.
    serial = std::min(serial, submitted_serial);
    if (!epoch_words_ready || serial == 0 || CompletedEpoch() >= serial) {
        return;
    }
    SceGxmNotification wait{};
    wait.address = const_cast<u32*>(EpochWord(serial));
    wait.value = static_cast<u32>(serial);
    // sceGxmNotificationWait returns when the word EQUALS the value, so the three numbers
    // here are the whole question if it never returns: the serial waited for, the newest
    // submitted, and what the GPU had finished on the way in. A completed already past the
    // waited-for serial means the value can never come back and the thread is parked for good.
    GXM_PHASE("gpu:wait-epoch", serial, scene_serial, CompletedEpoch());
    sceGxmNotificationWait(&wait);
    GXM_PHASE("gpu:wait-epoch-done", serial, scene_serial, CompletedEpoch());
    completed_floor = std::max(completed_floor, serial);
}

void TextureRuntime::DeferFree(void* data, bool cdram, u64 serial) {
    pending_frees.push_back({data, cdram, serial});
}

void TextureRuntime::ReapFreed(bool wait) {
    if (pending_frees.empty()) {
        return;
    }
    if (wait) {
        u64 newest = 0;
        for (const auto& p : pending_frees) {
            newest = std::max(newest, p.serial);
        }
        WaitForEpoch(newest);
    }
    const u64 completed = CompletedEpoch();
    std::erase_if(pending_frees, [&](const PendingFree& p) {
        if (p.serial > completed) {
            return false;
        }
        (p.cdram ? cdram_pool : mapped_pool).Free(p.data);
        return true;
    });
}

u32 TextureRuntime::CdramHeadroom() const {
    const u32 spare = cdram_pool.Reserved() - std::min(cdram_pool.Used(), cdram_pool.Reserved());
    u32 device_free = 0;
    if (HasDevice() && Device().largest_free != nullptr &&
        cdram_pool.Reserved() < cdram_pool.Budget()) {
        device_free = Device().largest_free(true);
    }
    return spare + device_free;
}

u64 TextureRuntime::CompletedEpoch() const {
    if (!epoch_words_ready) {
        return completed_floor;
    }
    // Scenes on one context finish in submission order, so the newest serial held by any
    // slot is the newest finished scene; the other slots hold older ones. Each is rebuilt
    // to the full serial around the floor a wait already established.
    const u64 base = completed_floor & ~static_cast<u64>(0xFFFFFFFF);
    u64 completed = completed_floor;
    for (const volatile u32* word : epoch_words) {
        u64 value = base | *word;
        if (value < completed_floor) {
            value += 0x100000000ull;
        }
        // A slot never written since the floor moved past a wrap would rebuild far ahead:
        // only values the thread has actually claimed count.
        if (value <= scene_serial) {
            completed = std::max(completed, value);
        }
    }
    return completed;
}

bool TextureRuntime::NeedsWait(const Surface* surface) const {
    if (surface == nullptr) {
        // Not a surface this operation touches - the second end of a one-ended operation.
        return false;
    }
    // A surface the presentation layer is sampling is referenced by a scene this renderer
    // never recorded and cannot have given a serial, so it is always waited for. That is at
    // most the two display framebuffers, once each per frame.
    return surface->presented || surface->gpu_serial > CompletedEpoch();
}

void TextureRuntime::Sync(Surface* a, Surface* b, const char* why) {
    // A deferred depth clear must be in memory before anything touches the memory.
    for (Surface* s : {a, b}) {
        if (s != nullptr && s->HasPendingClear()) {
            FlushPendingClear(*s);
        }
    }
    // Naming no surface at all means "wait for everything"; naming one and leaving the other
    // null is the ordinary one-ended case, and must not be read as the first.
    // Bring-up switch: wait for the GPU on every operation, whatever the serials say. A
    // picture that comes right this way was being read while it was still being drawn.
    static const bool always = GxmFlag("alwayssync");
    const bool unconditional = always || (a == nullptr && b == nullptr);
    if (!unconditional && !NeedsWait(a) && !NeedsWait(b)) {
        stat_syncs_avoided++;
        return;
    }
    if (sync_hook) {
        stat_syncs++;

        // What the wait is actually for: the newest scene that touched either surface, or
        // everything when a surface the presentation layer sampled is involved (that scene
        // is not one of ours and has no serial).
        u64 serial = 0;
        if (!unconditional && (a == nullptr || !a->presented) && (b == nullptr || !b->presented)) {
            serial = std::max(a != nullptr ? a->gpu_serial : 0, b != nullptr ? b->gpu_serial : 0);
        }
        static const bool log_syncs = GxmFlag("logdraw");
        if (log_syncs) {
            static u32 shown = 0;
            if (shown++ < 48) {
                const auto describe = [&](const Surface* s) {
                    return s == nullptr
                               ? std::string{"-"}
                               : fmt::format("{:08X} {}x{} {}{} serial {} (done {})", s->addr,
                                             s->width, s->height,
                                             VideoCore::PixelFormatAsString(s->pixel_format),
                                             s->presented ? " presented" : "", s->gpu_serial,
                                             CompletedEpoch());
                };
                LOG_INFO(Render, "sync wait: {} | {}{}", describe(a), describe(b),
                         unconditional ? " (unconditional)" : "");
            }
        }
        sync_hook(serial);
    }
}

u64 TextureRuntime::GetResourceTick() {
    return current_resource_tick;
}

void TextureRuntime::Finish() {
    current_resource_tick++;
    ReapFreed(false);
}

bool TextureRuntime::NeedsConversion(const Surface& surface) const {
    return FormatOf(surface.pixel_format).converted;
}

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool) {
    if (size > staging_buffer.size()) {
        staging_buffer.resize(size);
    }
    return VideoCore::StagingData{
        .size = size,
        .offset = 0,
        .mapped = std::span{staging_buffer.data(), size},
    };
}

void TextureRuntime::TransferWait() {
    sceGxmTransferFinish();
}

bool TextureRuntime::Reinterpret(Surface& source, Surface& dest, const VideoCore::TextureCopy&) {
    static bool warned = false;
    if (!warned) {
        LOG_WARNING(Render,
                    "GXM: reinterpretation {} -> {} not implemented; validating from "
                    "guest memory instead",
                    VideoCore::PixelFormatAsString(source.pixel_format),
                    VideoCore::PixelFormatAsString(dest.pixel_format));
        warned = true;
    }
    return false;
}

void TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear) {
    if (!surface.HasMemory() || clear.texture_layer != 0) {
        return;
    }
    // A clear of the whole depth surface is what the tiler does for free: the next scene
    // on it starts every tile from the background value instead of loading memory (libgxm
    // Overview, "Depth/Stencil Bandwidth Is Optional", "No Hardware Clear"). Done here
    // instead, it cost a scene break, a full GPU wait, a transfer fill and a transfer wait,
    // two or three times a frame on SM3DL.
    const auto& rect = clear.texture_rect;
    if (surface.IsDepthTarget() && clear.texture_level == 0 && rect.left == 0 &&
        rect.bottom == 0 && rect.GetWidth() == surface.width &&
        rect.GetHeight() == surface.height) {
        surface.pending_clear = true;
        surface.pending_depth = clear.value.depth;
        surface.pending_stencil = clear.value.stencil;
        stat_deferred_clears++;
        return;
    }
    // A colour clear is a quad: drawn into the scene open on the surface, or into the
    // next one to open on it (the rasterizer's hook and BeginScene), which is the tiler's
    // "everything is geometry" model. Tiny surfaces (the cache's null surface) stay on the
    // fill path, as no scene is ever going to open on them.
    if (surface.IsColorTarget() && clear.texture_level == 0 && surface.width >= 64 &&
        surface.height >= 64) {
        if (rect.left == 0 && rect.bottom == 0 && rect.GetWidth() == surface.width &&
            rect.GetHeight() == surface.height) {
            surface.pending_colour.clear(); // covered
        }
        surface.pending_colour.push_back({rect, clear.value.color});
        stat_deferred_clears++;
        if (colour_clear_hook) {
            colour_clear_hook(surface);
        }
        return;
    }
    FlushPendingClear(surface); // an earlier whole clear lands before a partial one
    ClearNow(surface, clear);
}

void TextureRuntime::FlushPendingClear(Surface& surface) {
    if (surface.pending_clear) {
        surface.pending_clear = false;
        VideoCore::TextureClear clear{};
        clear.texture_rect = Common::Rectangle<u32>{0, surface.height, surface.width, 0};
        clear.value.depth = surface.pending_depth;
        clear.value.stencil = surface.pending_stencil;
        ClearNow(surface, clear);
    }
    if (!surface.pending_colour.empty()) {
        std::vector<Surface::PendingColourClear> pending;
        pending.swap(surface.pending_colour);
        for (const auto& p : pending) {
            VideoCore::TextureClear clear{};
            clear.texture_rect = p.rect;
            clear.value.color = p.color;
            ClearNow(surface, clear);
        }
    }
}

void TextureRuntime::ClearNow(Surface& surface, const VideoCore::TextureClear& clear) {
    Sync(&surface, nullptr, "clear");
    const GxmFormat& format = surface.Format();
    const u32 value = PackClear(format, surface.pixel_format, clear.value);
    const auto& rect = clear.texture_rect;
    const u32 width = rect.GetWidth();
    const u32 height = rect.GetHeight();
    if (width == 0 || height == 0) {
        return;
    }
    const u32 level = clear.texture_level;
    u8* base = surface.RowPointer(level, rect.bottom, rect.left);
    const s32 stride = static_cast<s32>(surface.LevelStrideTexels(level) * format.bytes_per_pixel);
    stat_fills++;
    static const bool log_clears = GxmFlag("logdraw");
    if (log_clears) {
        static u32 shown = 0;
        if (shown++ < 64) {
            LOG_INFO(Render, "{} clear: surface {:08X} {}x{} {} rect {}x{} at {},{} value {:08X} (depth {})",
                     IsDepthType(surface.type) ? "depth" : "colour", surface.addr, surface.width, surface.height,
                     VideoCore::PixelFormatAsString(surface.pixel_format), width, height,
                     rect.left, rect.bottom, value, clear.value.depth);
        }
    }
    // PackClear already produced the word in the surface's own bit layout, so the fill is a
    // bit pattern and wants a RAW format: naming a real one would invite the engine to
    // convert a value that is already converted.
    //
    // There is no three-byte raw format, and a 24-bit surface asked to fill as RAW32 writes
    // four bytes per pixel: a third again of every row, at a pixel granularity the memory
    // does not have. The rows then walk forward by a third of their length each time, which
    // is what a cleared background looked like - skewed, and every draw blended onto it.
    // Those surfaces are cleared on the CPU instead, three bytes at a time.
    const bool has_raw_format = format.bytes_per_pixel == 2 || format.bytes_per_pixel == 4;
    const SceGxmTransferFormat fill_format = format.bytes_per_pixel == 2 ? TrRaw16 : TrRaw32;
    const int err = has_raw_format ? sceGxmTransferFill(value, fill_format, base, 0, 0, width,
                                                        height, stride, nullptr, 0, nullptr)
                                   : -1;
    if (err < 0) {
        // The engine refused it, or the format has no raw form: fill on the CPU.
        for (u32 y = 0; y < height; y++) {
            u8* row = base + static_cast<std::ptrdiff_t>(y) * stride;
            switch (format.bytes_per_pixel) {
            case 2:
                std::fill_n(reinterpret_cast<u16*>(row), width, static_cast<u16>(value));
                break;
            case 3:
                for (u32 x = 0; x < width; x++) {
                    row[x * 3 + 0] = static_cast<u8>(value);
                    row[x * 3 + 1] = static_cast<u8>(value >> 8);
                    row[x * 3 + 2] = static_cast<u8>(value >> 16);
                }
                break;
            default:
                std::fill_n(reinterpret_cast<u32*>(row), width, value);
                break;
            }
        }
        return;
    }
    TransferWait();
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  std::span<const VideoCore::TextureCopy> copies) {
    if (!source.HasMemory() || !dest.HasMemory()) {
        return false;
    }
    if (source.texture_type == TextureType::CubeMap || dest.texture_type == TextureType::CubeMap) {
        static bool warned = false;
        if (!warned) {
            LOG_WARNING(Render, "GXM: cube map copies not implemented (P5)");
            warned = true;
        }
        return false;
    }
    const GxmFormat& sf = source.Format();
    const GxmFormat& df = dest.Format();
    if (sf.bytes_per_pixel != df.bytes_per_pixel) {
        return false;
    }
    static const bool log_copies = GxmFlag("logdraw");
    if (log_copies) {
        static u32 shown = 0;
        if (shown++ < 200) {
            for (const auto& copy : copies) {
                LOG_INFO(Render, "copy {:08X} {}x{}{} -> {:08X} {}x{}{}: {}x{} from {},{} to {},{}",
                         source.addr, source.width, source.height,
                         source.IsColorTarget() ? " target" : "", dest.addr, dest.width,
                         dest.height, dest.IsColorTarget() ? " target" : "", copy.extent.width,
                         copy.extent.height, copy.src_offset.x, copy.src_offset.y,
                         copy.dst_offset.x, copy.dst_offset.y);
            }
        }
    }
    // A copy between surfaces the GPU is still working on: the transfer engine below needs
    // the CPU to drain the fragment pipeline first (Sync) and then wait for the transfer.
    // Smash's results screen asked for five of those a frame (four slivers of a blur
    // buffer's tail, one whole 512x400 frame into its 512x512 texture view) and ran at 2 fps
    // on the console. The fragment-pipeline blit is a scene of its own, ordered behind the
    // scenes that wrote the source, and the two surfaces are separate allocations here
    // (aliasing guest addresses, not host memory), so it needs no wait. Same pixel format
    // only: a blit converts pixels, the transfer copies bytes.
    if (blit_hook && source.pixel_format == dest.pixel_format &&
        (NeedsWait(&source) || NeedsWait(&dest))) {
        bool all = true;
        for (const auto& copy : copies) {
            if (copy.extent.width == 0 || copy.extent.height == 0) {
                continue;
            }
            const VideoCore::TextureBlit blit{
                .src_level = copy.src_level,
                .dst_level = copy.dst_level,
                .src_layer = copy.src_layer,
                .dst_layer = copy.dst_layer,
                .src_rect = Common::Rectangle<u32>{copy.src_offset.x,
                                                   copy.src_offset.y + copy.extent.height,
                                                   copy.src_offset.x + copy.extent.width,
                                                   copy.src_offset.y},
                .dst_rect = Common::Rectangle<u32>{copy.dst_offset.x,
                                                   copy.dst_offset.y + copy.extent.height,
                                                   copy.dst_offset.x + copy.extent.width,
                                                   copy.dst_offset.y},
            };
            if (!blit_hook(source, dest, blit)) {
                all = false;
                break;
            }
            stat_gpu_blits++;
        }
        if (all) {
            return true;
        }
    }
    Sync(&source, &dest, "CopyTextures");
    const SceGxmTransferFormat raw = sf.bytes_per_pixel == 2 ? TrRaw16 : TrRaw32;
    for (const auto& copy : copies) {
        if (copy.extent.width == 0 || copy.extent.height == 0) {
            continue;
        }
        const u8* src = source.RowPointer(copy.src_level, copy.src_offset.y, copy.src_offset.x);
        u8* dst = dest.RowPointer(copy.dst_level, copy.dst_offset.y, copy.dst_offset.x);
        const s32 src_stride =
            static_cast<s32>(source.LevelStrideTexels(copy.src_level) * sf.bytes_per_pixel);
        const s32 dst_stride =
            static_cast<s32>(dest.LevelStrideTexels(copy.dst_level) * df.bytes_per_pixel);
        stat_transfers++;
        const int err = sceGxmTransferCopy(
            copy.extent.width, copy.extent.height, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE, raw,
            SCE_GXM_TRANSFER_LINEAR, src, 0, 0, src_stride, raw, SCE_GXM_TRANSFER_LINEAR, dst, 0, 0,
            dst_stride, nullptr, 0, nullptr);
        if (err < 0) {
            const u32 bytes = copy.extent.width * sf.bytes_per_pixel;
            for (u32 y = 0; y < copy.extent.height; y++) {
                std::memcpy(dst + static_cast<std::ptrdiff_t>(y) * dst_stride,
                            src + static_cast<std::ptrdiff_t>(y) * src_stride, bytes);
            }
            continue;
        }
        TransferWait();
    }
    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    if (!source.HasMemory() || !dest.HasMemory()) {
        return false;
    }
    const GxmFormat& sf = source.Format();
    const GxmFormat& df = dest.Format();
    // The cache flags a vertical flip by handing over an upside-down source rectangle.
    const bool flip = blit.src_rect.bottom > blit.src_rect.top;
    const u32 src_w = blit.src_rect.GetWidth();
    const u32 src_h = flip ? blit.src_rect.bottom - blit.src_rect.top : blit.src_rect.GetHeight();
    const u32 dst_w = blit.dst_rect.GetWidth();
    const u32 dst_h = blit.dst_rect.GetHeight();
    if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
        return true;
    }
    // The CPU waits for the GPU only on the paths that touch the memory from outside a
    // scene (the transfer engine, the CPU). The fragment-pipeline blit is a scene of its
    // own, and the GPU runs scenes in order (libgxm Overview, Scene Dependencies), so what
    // an earlier scene drew is there by the time this one samples it. That wait was one
    // full GPU drain per display transfer, two a frame.
    const u32 src_first_row = flip ? blit.src_rect.bottom - 1 : blit.src_rect.bottom;
    const u8* src = source.RowPointer(blit.src_level, src_first_row, blit.src_rect.left);
    u8* dst = dest.RowPointer(blit.dst_level, blit.dst_rect.bottom, blit.dst_rect.left);
    const s32 src_pitch =
        static_cast<s32>(source.LevelStrideTexels(blit.src_level) * sf.bytes_per_pixel);
    const s32 dst_pitch =
        static_cast<s32>(dest.LevelStrideTexels(blit.dst_level) * df.bytes_per_pixel);
    const s32 src_stride = flip ? -src_pitch : src_pitch;

    const bool same_size = src_w == dst_w && src_h == dst_h;
    const bool half_size = src_w == 2 * dst_w && src_h == 2 * dst_h;
    // A PTLA format for each end: real formats convert between each other, RAW copies bits.
    const bool same_bits =
        sf.bytes_per_pixel == df.bytes_per_pixel &&
        (source.pixel_format == dest.pixel_format || (sf.native_transfer && df.native_transfer));
    if (same_size && same_bits) {
        Sync(&source, &dest, "BlitTextures");
        const SceGxmTransferFormat fmt =
            sf.native_transfer ? sf.transfer : (sf.bytes_per_pixel == 2 ? TrRaw16 : TrRaw32);
        stat_transfers++;
        const int err =
            sceGxmTransferCopy(src_w, src_h, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE, fmt,
                               SCE_GXM_TRANSFER_LINEAR, src, 0, 0, src_stride, fmt,
                               SCE_GXM_TRANSFER_LINEAR, dst, 0, 0, dst_pitch, nullptr, 0, nullptr);
        if (err >= 0) {
            TransferWait();
            return true;
        }
    } else if (!same_bits && blit_hook && blit_hook(source, dest, blit)) {
        // A format change: no PTLA path exists, so the fragment pipeline converts it.
        stat_gpu_blits++;
        return true;
    } else if (half_size && sf.native_transfer && df.native_transfer && !flip) {
        Sync(&source, &dest, "BlitTextures");
        stat_transfers++;
        const int err =
            sceGxmTransferDownscale(sf.transfer, src, 0, 0, src_w, src_h, src_pitch, df.transfer,
                                    dst, 0, 0, dst_pitch, nullptr, 0, nullptr);
        if (err >= 0) {
            TransferWait();
            return true;
        }
    }
    Sync(&source, &dest, "BlitTextures");
    return CpuBlit(source, dest, blit);
}

bool TextureRuntime::CpuBlit(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
    const GxmFormat& sf = source.Format();
    const GxmFormat& df = dest.Format();
    const bool src_depth = IsDepthType(source.type);
    if (src_depth != IsDepthType(dest.type)) {
        static bool warned = false;
        if (!warned) {
            LOG_WARNING(Render, "GXM: blit {} -> {} crosses colour and depth; refused",
                        VideoCore::PixelFormatAsString(source.pixel_format),
                        VideoCore::PixelFormatAsString(dest.pixel_format));
            warned = true;
        }
        return false;
    }
    stat_cpu_blits++;
    const bool flip = blit.src_rect.bottom > blit.src_rect.top;
    const u32 src_w = blit.src_rect.GetWidth();
    const u32 src_h = flip ? blit.src_rect.bottom - blit.src_rect.top : blit.src_rect.GetHeight();
    const u32 dst_w = blit.dst_rect.GetWidth();
    const u32 dst_h = blit.dst_rect.GetHeight();
    const u32 src_row0 = flip ? blit.src_rect.top : blit.src_rect.bottom;

    // Identical storage: the rows are bytes, and a same-width blit is a memcpy per row.
    const bool same_storage = source.pixel_format == dest.pixel_format ||
                              (sf.bytes_per_pixel == df.bytes_per_pixel &&
                               sf.native_transfer && df.native_transfer);
    // Otherwise convert through the intermediate: RGBA8 in surface byte order for colour,
    // a float for depth. Both are four bytes a texel.
    const RowUnpack unpack = same_storage ? nullptr : UnpackFor(source.pixel_format);
    const RowPack pack = same_storage ? nullptr : PackFor(dest.pixel_format);
    const u32 src_bpp = sf.bytes_per_pixel;
    const u32 dst_bpp = df.bytes_per_pixel;

    // One source row is read once into a CPU buffer so CDRAM is never read texel by texel,
    // and the horizontal resample happens in the intermediate.
    std::vector<u8> row(static_cast<std::size_t>(src_w) * std::max(src_bpp, 4u));
    std::vector<u8> mid(same_storage ? 0 : static_cast<std::size_t>(dst_w) * 4);
    for (u32 y = 0; y < dst_h; y++) {
        u32 sy = dst_h == src_h ? y : y * src_h / dst_h;
        if (flip) {
            sy = src_h - 1 - sy;
        }
        const u8* src = source.RowPointer(blit.src_level, src_row0 + sy, blit.src_rect.left);
        u8* dst = dest.RowPointer(blit.dst_level, blit.dst_rect.bottom + y, blit.dst_rect.left);
        if (same_storage) {
            std::memcpy(row.data(), src, static_cast<std::size_t>(src_w) * src_bpp);
            if (src_w == dst_w) {
                std::memcpy(dst, row.data(), static_cast<std::size_t>(dst_w) * dst_bpp);
                continue;
            }
            for (u32 x = 0; x < dst_w; x++) {
                std::memcpy(dst + x * dst_bpp, row.data() + (x * src_w / dst_w) * src_bpp,
                            dst_bpp);
            }
            continue;
        }
        unpack(src, row.data(), src_w);
        if (src_w == dst_w) {
            pack(row.data(), dst, dst_w);
            continue;
        }
        for (u32 x = 0; x < dst_w; x++) {
            std::memcpy(mid.data() + x * 4, row.data() + (x * src_w / dst_w) * 4, 4);
        }
        pack(mid.data(), dst, dst_w);
    }
    return true;
}

void TextureRuntime::GenerateMipmaps(Surface&) {}

// ---- Surface --------------------------------------------------------------------------------

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params,
                 const VideoCore::SurfaceFlagBits& initial_flag_bits)
    : SurfaceBase{params, initial_flag_bits}, runtime{&runtime_},
      format{FormatOf(params.pixel_format)} {
    if (pixel_format == PixelFormat::Invalid || type == SurfaceType::Fill ||
        type == SurfaceType::Invalid || width == 0 || height == 0) {
        return;
    }
    if (texture_type == TextureType::CubeMap) {
        // P5: cube maps need a swizzled six-face allocation and PTLA face copies.
        return;
    }
    Allocate();
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceBase& surface,
                 const VideoCore::Material*)
    : SurfaceBase{surface, {}}, runtime{&runtime_}, format{FormatOf(surface.pixel_format)} {
    // Custom textures are not supported on this backend; the surface stays empty and the
    // cache's upload of the original pixels never reaches a custom surface.
}

Surface::~Surface() {
    Release();
}

Surface::Surface(Surface&& o) noexcept
    : SurfaceBase{std::move(o)}, runtime{o.runtime}, format{o.format}, data{o.data},
      in_cdram{o.in_cdram}, stride_texels{o.stride_texels}, alloc_size{o.alloc_size},
      level_offsets{o.level_offsets}, texture{o.texture}, color_surface{o.color_surface},
      ds_surface{o.ds_surface}, is_color_target{o.is_color_target},
      is_depth_target{o.is_depth_target} {
    o.data = nullptr;
}

Surface& Surface::operator=(Surface&& o) noexcept {
    if (this != &o) {
        Release();
        SurfaceBase::operator=(std::move(o));
        runtime = o.runtime;
        format = o.format;
        data = o.data;
        in_cdram = o.in_cdram;
        stride_texels = o.stride_texels;
        alloc_size = o.alloc_size;
        level_offsets = o.level_offsets;
        texture = o.texture;
        color_surface = o.color_surface;
        ds_surface = o.ds_surface;
        is_color_target = o.is_color_target;
        is_depth_target = o.is_depth_target;
        o.data = nullptr;
    }
    return *this;
}

/**
 * The filters a texture will accept. GXM refuses sceGxmTextureSetMinFilter and
 * sceGxmTextureSetMipFilter on a LINEAR_STRIDED texture (SCE_GXM_ERROR_UNSUPPORTED): a strided
 * texture has no mipmaps, so it has no minification filter either, and the call leaves the
 * texture at the point filter it was initialised with. Calling anyway is not harmful, only
 * misleading - it looks like linear minification was asked for and got.
 */
void SetFilters(SceGxmTexture& texture, SceGxmTextureFilter min_filter,
                SceGxmTextureFilter mag_filter) {
    sceGxmTextureSetMagFilter(&texture, mag_filter);
    if (sceGxmTextureGetType(&texture) == SCE_GXM_TEXTURE_LINEAR_STRIDED) {
        return;
    }
    sceGxmTextureSetMinFilter(&texture, min_filter);
    sceGxmTextureSetMipFilter(&texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
}

void Surface::Allocate() {
    const bool depth = IsDepthType(type);
    stride_texels = PitchFor(width, depth);
    // Levels are stored one after the other, each with its own implied pitch; only level 0
    // is exposed to the sampler until GXM's linear mip layout is verified on console.
    u32 offset = 0;
    for (u32 level = 0; level < VideoCore::MAX_PICA_LEVELS; level++) {
        level_offsets[level] = offset;
        if (level < levels) {
            const u32 w = std::max(width >> level, 1u);
            u32 h = std::max(height >> level, 1u);
            if (depth) {
                // "The memory footprint must be a whole number of tiles" (GPU User's Guide
                // 11, linear depth/stencil surfaces): the tiler loads and stores the last
                // tile row in full, so the rows past the framebuffer must exist.
                h = Common::AlignUp(h, 32u);
            }
            offset += PitchFor(w, depth) * h * format.bytes_per_pixel;
        }
    }
    alloc_size = Common::AlignUp(offset, 64u);
    // CDRAM first for every surface, texture or target: 128 MB the game partition does not
    // count, and where the GPU reads fastest; the CPU's uploads into it are rare. LPDDR when
    // the CDRAM pool is out, or everywhere under `texlpddr`.
    in_cdram = true;
    if (in_cdram) {
        data = static_cast<u8*>(runtime->cdram_pool.Alloc(alloc_size, 256));
        if (data == nullptr) {
            in_cdram = false;
        }
    }
    if (data == nullptr) {
        data = static_cast<u8*>(runtime->mapped_pool.Alloc(alloc_size, 256));
    }
    if (data == nullptr) {
        // What was freed this frame is still owed to the GPU: wait for it and try again.
        runtime->ReapFreed(true);
        in_cdram = true;
        if (in_cdram) {
            data = static_cast<u8*>(runtime->cdram_pool.Alloc(alloc_size, 256));
            in_cdram = data != nullptr;
        }
        if (data == nullptr) {
            data = static_cast<u8*>(runtime->mapped_pool.Alloc(alloc_size, 256));
        }
    }
    if (data == nullptr) {
        LOG_ERROR(Render, "GXM: no memory for a {}x{} {} surface", width, height,
                  VideoCore::PixelFormatAsString(pixel_format));
        return;
    }
    // sceGxmTextureInitLinear derives its own pitch, AlignUp(width, 8) texels
    // (SCE_GXM_TEXTURE_IMPLICIT_STRIDE_ALIGNMENT). Depth surfaces are stored at
    // AlignUp(width, 32) instead, because a depth-stencil surface's stride must be a multiple
    // of SCE_GXM_TILE_SIZEX, so for a depth surface the implied pitch is the wrong one -
    // 400 wide is 400 texels to the sampler and 416 to everything else, and the picture skews
    // by a row every row. Those give the sampler the pitch the memory actually has.
    const u32 stride_bytes = stride_texels * format.bytes_per_pixel;
    const int texture_err =
        stride_texels == PitchFor(width, false)
            ? sceGxmTextureInitLinear(&texture, data, format.texture, width, height, 1)
            : sceGxmTextureInitLinearStrided(&texture, data, format.texture, width, height,
                                             stride_bytes);
    if (texture_err < 0) {
        LOG_ERROR(Render, "GXM: texture init failed for {}x{} {} (pitch {} texels): {} ({:#x})",
                  width, height, VideoCore::PixelFormatAsString(pixel_format), stride_texels,
                  GxmErrorName(texture_err), static_cast<u32>(texture_err));
    }
    SetFilters(texture, SCE_GXM_TEXTURE_FILTER_LINEAR, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetUAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    // Every colour and depth format in the table has a GXM surface form, so the only reason
    // either init can fail is a bad argument - which is worth a line, not a silent surface
    // that draws nowhere. A zero enum is not a "no such format" marker here:
    // SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR is itself zero (base format 0 | swizzle ABGR 0).
    if (type == SurfaceType::Color) {
        is_color_target = sceGxmColorSurfaceInit(
                              &color_surface, format.color, SCE_GXM_COLOR_SURFACE_LINEAR,
                              SCE_GXM_COLOR_SURFACE_SCALE_NONE, SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
                              width, height, stride_texels, data) >= 0;
        if (!is_color_target) {
            LOG_ERROR(Render, "GXM: no colour surface for {}x{} {} (stride {})", width, height,
                      VideoCore::PixelFormatAsString(pixel_format), stride_texels);
        }
    } else if (depth) {
        is_depth_target = sceGxmDepthStencilSurfaceInit(&ds_surface, format.depth,
                                                        SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR,
                                                        stride_texels, data, nullptr) >= 0;
        if (!is_depth_target) {
            LOG_ERROR(Render, "GXM: no depth-stencil surface for {}x{} {} (stride {})", width,
                      height, VideoCore::PixelFormatAsString(pixel_format), stride_texels);
        }
        // Memory is the truth for depth too (the guest's clear is a fill into it, and its
        // test direction depends on the value): load at scene start, store at scene end.
        // P6 decides this per scene.
        sceGxmDepthStencilSurfaceSetForceLoadMode(&ds_surface,
                                                  SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
        sceGxmDepthStencilSurfaceSetForceStoreMode(&ds_surface,
                                                   SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
    }
}

bool Surface::Rename() {
    GpuPool& pool = in_cdram ? runtime->cdram_pool : runtime->mapped_pool;
    void* fresh = pool.Alloc(alloc_size, 256);
    if (fresh == nullptr) {
        return false;
    }
    runtime->DeferFree(data, in_cdram, gpu_serial);
    data = static_cast<u8*>(fresh);
    sceGxmTextureSetData(&texture, data);
    gpu_serial = 0;
    return true;
}

void Surface::Release() {
    if (data == nullptr || runtime == nullptr) {
        return;
    }
    runtime->DeferFree(data, in_cdram, gpu_serial);
    data = nullptr;
}

u32 Surface::LevelStrideTexels(u32 level) const noexcept {
    return PitchFor(std::max(width >> level, 1u), IsDepthType(type));
}

u8* Surface::RowPointer(u32 level, u32 y, u32 x) const noexcept {
    return data + LevelOffset(level) +
           (static_cast<std::size_t>(y) * LevelStrideTexels(level) + x) * format.bytes_per_pixel;
}

void Surface::Upload(const VideoCore::BufferTextureCopy& upload,
                     const VideoCore::StagingData& staging) {
    if (data == nullptr) {
        return;
    }
    const auto& rect = upload.texture_rect;
    const u32 w = rect.GetWidth();
    const u32 h = rect.GetHeight();
    // The GPU may still be sampling this surface. A whole-surface upload of a texture
    // (a glyph atlas the guest rewrites every frame, on SM3DL) takes a fresh block instead
    // of waiting for those scenes, which was half the render thread's time at times.
    const bool whole = upload.texture_level == 0 && levels == 1 && rect.left == 0 &&
                       rect.bottom == 0 && w == width && h == height;
    if (whole && !is_color_target && !is_depth_target && runtime->InFlight(*this) && Rename()) {
        runtime->stat_renames++;
    } else {
        runtime->Sync(this, nullptr, "upload");
    }
    const u32 bpp = format.bytes_per_pixel;
    const u32 row_bytes = w * bpp;
    // The staging rows are the codec's: two bytes a pixel for D16, whose surface holds floats.
    const u32 staging_row = pixel_format == PixelFormat::D16 ? w * 2 : row_bytes;
    runtime->stat_uploads++;
    // Which surfaces keep coming back from guest memory: on SM3DL's 3D scenes the console
    // showed 17 uploads and 400 KiB a frame, which is a framebuffer's worth every frame.
    static const bool log_uploads = GxmFlag("logdraw");
    if (log_uploads && w * h * bpp >= 64 * 1024) {
        static u32 shown = 0;
        if (shown++ < 64) {
            LOG_INFO(Render, "upload: surface {:08X} {}x{} {} rect {}x{} at {},{} ({} KiB){}",
                     addr, width, height, VideoCore::PixelFormatAsString(pixel_format), w, h,
                     rect.left, rect.bottom, (w * h * bpp) >> 10,
                     IsColorTarget() ? " colour target" : "");
        }
    }
    runtime->stat_upload_bytes += row_bytes * h;
    const u8* src = staging.mapped.data();
    for (u32 y = 0; y < h; y++) {
        u8* dst = RowPointer(upload.texture_level, rect.bottom + y, rect.left);
        if (pixel_format == PixelFormat::D24S8) {
            RotateWordsRight8(reinterpret_cast<u32*>(dst), reinterpret_cast<const u32*>(src), w);
        } else if (pixel_format == PixelFormat::D16) {
            D16ToS8D24(src, dst, w);
        } else {
            std::memcpy(dst, src, row_bytes);
        }
        src += staging_row;
    }
}

std::span<u8> Surface::DirectUploadSpan(const VideoCore::BufferTextureCopy& upload, u32 bytes) {
    if (data == nullptr || pixel_format == PixelFormat::D24S8 || pixel_format == PixelFormat::D16) {
        return {};
    }
    const auto& rect = upload.texture_rect;
    const u32 w = rect.GetWidth();
    const u32 h = rect.GetHeight();
    if (rect.left != 0 || w != LevelStrideTexels(upload.texture_level) ||
        w * h * format.bytes_per_pixel != bytes) {
        return {};
    }
    const bool whole = upload.texture_level == 0 && levels == 1 && rect.bottom == 0 &&
                       w == width && h == height;
    if (whole && !is_color_target && !is_depth_target && runtime->InFlight(*this) && Rename()) {
        runtime->stat_renames++;
    } else {
        runtime->Sync(this);
    }
    runtime->stat_uploads++;
    runtime->stat_upload_bytes += bytes;
    static const bool log_uploads = GxmFlag("logdraw");
    if (log_uploads && bytes >= 64 * 1024) {
        static u32 shown = 0;
        if (shown++ < 64) {
            LOG_INFO(Render, "upload (direct): surface {:08X} {}x{} {} rect {}x{} at {},{} ({} KiB){}",
                     addr, width, height, VideoCore::PixelFormatAsString(pixel_format), w, h,
                     rect.left, rect.bottom, bytes >> 10, IsColorTarget() ? " colour target" : "");
        }
    }
    return {RowPointer(upload.texture_level, rect.bottom, 0), bytes};
}

void Surface::UploadCustom(const VideoCore::Material*, u32) {}

void Surface::Download(const VideoCore::BufferTextureCopy& download,
                       const VideoCore::StagingData& staging) {
    if (data == nullptr) {
        return;
    }
    runtime->Sync(this, nullptr, "download");
    const auto& rect = download.texture_rect;
    const u32 w = rect.GetWidth();
    const u32 h = rect.GetHeight();
    const u32 bpp = format.bytes_per_pixel;
    const u32 row_bytes = w * bpp;
    runtime->stat_downloads++;
    runtime->stat_download_bytes += row_bytes * h;
    // Through the transfer unit when the format has a raw transfer size and the level fits
    // the download buffer (see its declaration); the CPU reads the copy, not the surface.
    const u8* first_row = RowPointer(download.texture_level, rect.bottom, rect.left);
    const u8* read_from = nullptr;
    u32 read_stride = LevelStrideTexels(download.texture_level) * bpp;
    if ((bpp == 2 || bpp == 4) && row_bytes * h <= TextureRuntime::DownloadMemSize) {
        if (runtime->download_mem == nullptr && HasDevice()) {
            runtime->download_mem =
                static_cast<u8*>(Device().alloc_mapped(TextureRuntime::DownloadMemSize));
        }
    }
    if (read_from == nullptr && runtime->download_mem != nullptr && (bpp == 2 || bpp == 4) &&
        row_bytes * h <= TextureRuntime::DownloadMemSize) {
        const SceGxmTransferFormat raw = bpp == 2 ? TrRaw16 : TrRaw32;
        const int err = sceGxmTransferCopy(
            w, h, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE, raw, SCE_GXM_TRANSFER_LINEAR, first_row,
            0, 0, static_cast<s32>(read_stride), raw, SCE_GXM_TRANSFER_LINEAR,
            runtime->download_mem, 0, 0, static_cast<s32>(row_bytes), nullptr, 0, nullptr);
        if (err >= 0) {
            runtime->TransferWait();
            read_from = runtime->download_mem;
            read_stride = row_bytes;
        }
    }
    u8* dst = staging.mapped.data();
    for (u32 y = 0; y < h; y++) {
        const u8* src = read_from != nullptr
                            ? read_from + static_cast<std::size_t>(y) * read_stride
                            : RowPointer(download.texture_level, rect.bottom + y, rect.left);
        if (pixel_format == PixelFormat::D24S8) {
            RotateWordsLeft8(reinterpret_cast<u32*>(dst), reinterpret_cast<const u32*>(src), w);
        } else if (pixel_format == PixelFormat::D16) {
            S8D24ToD16(src, dst, w);
            dst += w * 2;
            continue;
        } else {
            std::memcpy(dst, src, row_bytes);
        }
        dst += row_bytes;
    }
}

void Surface::ScaleUp(u32) {}

u32 Surface::GetInternalBytesPerPixel() const {
    return format.bytes_per_pixel;
}

// ---- Framebuffer ----------------------------------------------------------------------------

Framebuffer::Framebuffer(TextureRuntime&, const VideoCore::FramebufferParams& params,
                         Surface* color, Surface* depth)
    : VideoCore::FramebufferParams{params} {
    if (shadow_rendering) {
        return; // no image load/store on this backend; the draw is dropped
    }
    Surface* any = color != nullptr && color->HasMemory()
                       ? color
                       : (depth != nullptr && depth->HasMemory() ? depth : nullptr);
    if (any == nullptr) {
        return;
    }
    width = std::max(any->width >> (color != nullptr ? color_level : depth_level), 1u);
    height = std::max(any->height >> (color != nullptr ? color_level : depth_level), 1u);
    if (color != nullptr && color->IsColorTarget() && color_level == 0) {
        color_surface = *color->ColorSurface();
        color_data = color->Data();
        has_color = true;
    } else {
        sceGxmColorSurfaceInitDisabled(&color_surface);
    }
    if (depth != nullptr && depth->IsDepthTarget() && depth_level == 0) {
        ds_surface = *depth->DepthStencilSurface();
        has_depth = true;
    }
    if (!has_color && !has_depth) {
        return;
    }
    SceGxmRenderTargetParams rt{};
    rt.width = static_cast<u16>(width);
    rt.height = static_cast<u16>(height);
    // Titles switch targets mid-frame; the SDK bounds queued scenes per target per frame.
    rt.scenesPerFrame = 8;
    rt.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    rt.driverMemBlock = static_cast<SceUID>(-1);
    if (sceGxmCreateRenderTarget(&rt, &render_target) < 0) {
        LOG_ERROR(Render, "GXM: render target creation failed ({}x{})", width, height);
        render_target = nullptr;
    }
}

Framebuffer::~Framebuffer() {
    if (render_target != nullptr) {
        // The cache removes framebuffers only after a frame's wait, so the GPU is done here.
        sceGxmDestroyRenderTarget(render_target);
    }
}

Framebuffer::Framebuffer(Framebuffer&& o) noexcept
    : VideoCore::FramebufferParams{o}, render_target{o.render_target},
      color_surface{o.color_surface}, ds_surface{o.ds_surface}, color_data{o.color_data},
      width{o.width}, height{o.height}, has_color{o.has_color}, has_depth{o.has_depth} {
    o.render_target = nullptr;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& o) noexcept {
    if (this != &o) {
        if (render_target != nullptr) {
            sceGxmDestroyRenderTarget(render_target);
        }
        VideoCore::FramebufferParams::operator=(o);
        render_target = o.render_target;
        color_surface = o.color_surface;
        ds_surface = o.ds_surface;
        color_data = o.color_data;
        width = o.width;
        height = o.height;
        has_color = o.has_color;
        has_depth = o.has_depth;
        o.render_target = nullptr;
    }
    return *this;
}

// ---- Sampler --------------------------------------------------------------------------------

Sampler::Sampler(TextureRuntime&, VideoCore::SamplerParams params) {
    using TextureFilter = Pica::TexturingRegs::TextureConfig::TextureFilter;
    using WrapMode = Pica::TexturingRegs::TextureConfig::WrapMode;
    const auto filter = [](TextureFilter f) {
        return f == TextureFilter::Linear ? SCE_GXM_TEXTURE_FILTER_LINEAR
                                          : SCE_GXM_TEXTURE_FILTER_POINT;
    };
    // The SDK header says MIRROR is for swizzled textures only, and sceGxmTextureSetUAddrMode
    // refuses it on a linear one (SCE_GXM_ERROR_UNSUPPORTED; PKAS's fight button and SM3DL's
    // title button, each half a texture mirrored, came out stretched). That check is in the
    // user-side library, the addressing unit reads the mode from the control word, and
    // DSVita mirrors linear textures on the console; Apply writes the bits itself when the
    // call refuses. Border modes are still clamp.
    const auto wrap = [](WrapMode m) {
        switch (m) {
        case WrapMode::Repeat:
        case WrapMode::Repeat2:
        case WrapMode::Repeat3:
            return SCE_GXM_TEXTURE_ADDR_REPEAT;
        case WrapMode::MirroredRepeat:
            return SCE_GXM_TEXTURE_ADDR_MIRROR;
        default:
            return SCE_GXM_TEXTURE_ADDR_CLAMP;
        }
    };
    mag_filter = filter(params.mag_filter);
    min_filter = filter(params.min_filter);
    wrap_s = wrap(params.wrap_s);
    wrap_t = wrap(params.wrap_t);
}

void Sampler::Apply(SceGxmTexture& texture) const {
    SetFilters(texture, min_filter, mag_filter);
    const int err_u = sceGxmTextureSetUAddrMode(&texture, wrap_s);
    const int err_v = sceGxmTextureSetVAddrMode(&texture, wrap_t);
    if (err_u < 0 || err_v < 0) {
        // Control word 0 of the texture: the V mode in bits 3 to 5, the U mode in bits 6 to 8
        // (the Vita3K SceGxmTexture layout, which its own setters write into). Flag
        // `nomirrorbits` keeps to what the library accepts, to tell the two apart on the
        // console.
        static const bool no_bits = GxmFlag("nomirrorbits");
        auto* words = reinterpret_cast<u32*>(&texture);
        if (!no_bits) {
            u32 cw0 = words[0];
            if (err_u < 0) {
                cw0 = (cw0 & ~(7u << 6)) | (static_cast<u32>(wrap_s) << 6);
            }
            if (err_v < 0) {
                cw0 = (cw0 & ~(7u << 3)) | (static_cast<u32>(wrap_t) << 3);
            }
            words[0] = cw0;
        }
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_WARNING(Render,
                        "GXM: texture address mode refused: u {} -> {:#x}, v {} -> {:#x}; {}",
                        static_cast<int>(wrap_s), static_cast<u32>(err_u),
                        static_cast<int>(wrap_t), static_cast<u32>(err_v),
                        no_bits ? "left as the library set it"
                                : "written into the control word directly");
        }
    }
}

} // namespace GxmRenderer
