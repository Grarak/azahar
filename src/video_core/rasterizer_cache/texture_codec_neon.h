// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// NEON tile codec: the 8x8 Morton tile as a fixed 64-texel permutation done with vector
// loads and stores, the format widening/packing on whole rows, and an ETC1 subtile decoder.
// Included by texture_codec.h on NEON targets only; every function returns false when it
// has no specialisation for the (direction, format, converted) triple, and the scalar
// per-texel code in texture_codec.h stays the reference. tests/video_core/texture_codec.cpp
// diffs the two on random tiles for every table entry.
//
// Tile layout, from MortonInterleave(x, y) = xlut[x] + ylut[y] with x bits at 0,2,4 and y bits
// at 1,3,5: the tile is four 4x4 quadrants in order (x<4,y<4) (x>=4,y<4) (x<4,y>=4) (x>=4,y>=4);
// a quadrant is four 2x2 blocks in the same order; a block is (x,y) (x+1,y) (x,y+1) (x+1,y+1).
// So the two texels of one row inside a block are adjacent, and one quadrant holds rows y..y+3
// of one 4-wide half of the tile. Row r of the tile is assembled from the two quadrants of its
// half (r/4), block row (r/2)%2, and the low (r even) or high (r odd) half of each block.
//
// The linear side is written bottom-up like the scalar code: tile row y lands in linear row
// 7 - y (the GL origin convention the whole cache uses).

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <cstring>
#include <span>
#include <arm_neon.h>
#include "common/common_types.h"
#include "video_core/rasterizer_cache/pixel_format.h"

#define AZAHAR_TEXTURE_CODEC_NEON 1

namespace VideoCore::NeonCodec {

// One tile row of 8 texels, at each texel width the cache deals in.
struct Row32 {
    uint32x4_t lo, hi; // texels 0-3, 4-7
};
using Row16 = uint16x8_t;
using Row8 = uint8x8_t;
using Row4 = uint8x8_t; // lanes 0-3 hold the 4 packed nibble pairs; lanes 4-7 ignored

inline u8* LinearRow(u8* linear, u32 stride_bytes, u32 y) {
    return linear + (7 - y) * stride_bytes;
}
inline const u8* LinearRow(const u8* linear, u32 stride_bytes, u32 y) {
    return linear + (7 - y) * stride_bytes;
}

// ---- Morton -> rows -------------------------------------------------------------------------

// 32-bit texels: a quadrant is 64 bytes, four blocks of one q register each.
inline void LoadRows32(const u8* tile, Row32 out[8]) {
    for (u32 half = 0; half < 2; half++) {
        const u8* ql = tile + half * 128;
        const u8* qr = ql + 64;
        for (u32 br = 0; br < 2; br++) {
            const uint32x4_t l0 = vld1q_u32(reinterpret_cast<const u32*>(ql + br * 32));
            const uint32x4_t l1 = vld1q_u32(reinterpret_cast<const u32*>(ql + br * 32 + 16));
            const uint32x4_t r0 = vld1q_u32(reinterpret_cast<const u32*>(qr + br * 32));
            const uint32x4_t r1 = vld1q_u32(reinterpret_cast<const u32*>(qr + br * 32 + 16));
            const u32 y = half * 4 + br * 2;
            out[y].lo = vcombine_u32(vget_low_u32(l0), vget_low_u32(l1));
            out[y].hi = vcombine_u32(vget_low_u32(r0), vget_low_u32(r1));
            out[y + 1].lo = vcombine_u32(vget_high_u32(l0), vget_high_u32(l1));
            out[y + 1].hi = vcombine_u32(vget_high_u32(r0), vget_high_u32(r1));
        }
    }
}

// 16-bit texels: a block is 8 bytes (two u32 pairs: row y, row y+1); vld2_u32 over two
// blocks splits the pairs by row.
inline void LoadRows16(const u8* tile, Row16 out[8]) {
    for (u32 half = 0; half < 2; half++) {
        const u8* ql = tile + half * 64;
        const u8* qr = ql + 32;
        for (u32 br = 0; br < 2; br++) {
            const uint32x2x2_t l = vld2_u32(reinterpret_cast<const u32*>(ql + br * 16));
            const uint32x2x2_t r = vld2_u32(reinterpret_cast<const u32*>(qr + br * 16));
            const u32 y = half * 4 + br * 2;
            out[y] = vreinterpretq_u16_u32(vcombine_u32(l.val[0], r.val[0]));
            out[y + 1] = vreinterpretq_u16_u32(vcombine_u32(l.val[1], r.val[1]));
        }
    }
}

// 8-bit texels: a quadrant is 16 bytes; vld2_u16 over it yields (row0 row2 | row1 row3)
// half-rows of the quadrant, zipped with the other quadrant's by u32 lane.
inline void LoadRows8(const u8* tile, Row8 out[8]) {
    for (u32 half = 0; half < 2; half++) {
        const uint16x4x2_t l = vld2_u16(reinterpret_cast<const u16*>(tile + half * 32));
        const uint16x4x2_t r = vld2_u16(reinterpret_cast<const u16*>(tile + half * 32 + 16));
        const uint32x2x2_t even = vzip_u32(vreinterpret_u32_u16(l.val[0]),
                                           vreinterpret_u32_u16(r.val[0])); // rows 0, 2
        const uint32x2x2_t odd = vzip_u32(vreinterpret_u32_u16(l.val[1]),
                                          vreinterpret_u32_u16(r.val[1])); // rows 1, 3
        out[half * 4 + 0] = vreinterpret_u8_u32(even.val[0]);
        out[half * 4 + 1] = vreinterpret_u8_u32(odd.val[0]);
        out[half * 4 + 2] = vreinterpret_u8_u32(even.val[1]);
        out[half * 4 + 3] = vreinterpret_u8_u32(odd.val[1]);
    }
}

// 4-bit texels: a block is 2 bytes (row y, row y+1); vld2_u8 over both quadrants of a half
// gives per row-parity [r0L r2L r0R r2R] as u16 lanes, unzipped into rows.
inline void LoadRows4(const u8* tile, Row4 out[8]) {
    for (u32 half = 0; half < 2; half++) {
        const uint8x8x2_t v = vld2_u8(tile + half * 16);
        const uint16x4x2_t rows =
            vuzp_u16(vreinterpret_u16_u8(v.val[0]), vreinterpret_u16_u8(v.val[1]));
        // rows.val[0] = (r0L r0R r1L r1R), rows.val[1] = (r2L r2R r3L r3R)
        const uint32x2_t r01 = vreinterpret_u32_u16(rows.val[0]);
        const uint32x2_t r23 = vreinterpret_u32_u16(rows.val[1]);
        out[half * 4 + 0] = vreinterpret_u8_u32(vdup_lane_u32(r01, 0));
        out[half * 4 + 1] = vreinterpret_u8_u32(vdup_lane_u32(r01, 1));
        out[half * 4 + 2] = vreinterpret_u8_u32(vdup_lane_u32(r23, 0));
        out[half * 4 + 3] = vreinterpret_u8_u32(vdup_lane_u32(r23, 1));
    }
}

// 24-bit texels: vld3_u8 over the two blocks of one quadrant block-row gives per channel
// [r0x0 r0x1 r1x0 r1x1 r0x2 r0x3 r1x2 r1x3]; as u16 lanes (r0 r1 r0 r1), unzip by row.
struct Row24 {
    uint8x8_t c[3]; // memory byte order: c[0] = byte 0 of every texel, etc.
};
inline void LoadRows24(const u8* tile, Row24 out[8]) {
    for (u32 half = 0; half < 2; half++) {
        const u8* ql = tile + half * 96;
        const u8* qr = ql + 48;
        for (u32 br = 0; br < 2; br++) {
            const uint8x8x3_t l = vld3_u8(ql + br * 24);
            const uint8x8x3_t r = vld3_u8(qr + br * 24);
            const u32 y = half * 4 + br * 2;
            for (u32 ch = 0; ch < 3; ch++) {
                const uint16x4x2_t lu =
                    vuzp_u16(vreinterpret_u16_u8(l.val[ch]), vreinterpret_u16_u8(l.val[ch]));
                const uint16x4x2_t ru =
                    vuzp_u16(vreinterpret_u16_u8(r.val[ch]), vreinterpret_u16_u8(r.val[ch]));
                // lu.val[0] lanes 0,1 = row y left half; lu.val[1] = row y+1
                const uint32x2x2_t r0 =
                    vzip_u32(vreinterpret_u32_u16(lu.val[0]), vreinterpret_u32_u16(ru.val[0]));
                const uint32x2x2_t r1 =
                    vzip_u32(vreinterpret_u32_u16(lu.val[1]), vreinterpret_u32_u16(ru.val[1]));
                out[y].c[ch] = vreinterpret_u8_u32(r0.val[0]);
                out[y + 1].c[ch] = vreinterpret_u8_u32(r1.val[0]);
            }
        }
    }
}

// ---- rows -> Morton -------------------------------------------------------------------------

inline void StoreRows32(u8* tile, const Row32 in[8]) {
    for (u32 half = 0; half < 2; half++) {
        u8* ql = tile + half * 128;
        u8* qr = ql + 64;
        for (u32 br = 0; br < 2; br++) {
            const Row32& a = in[half * 4 + br * 2];
            const Row32& b = in[half * 4 + br * 2 + 1];
            vst1q_u32(reinterpret_cast<u32*>(ql + br * 32),
                      vcombine_u32(vget_low_u32(a.lo), vget_low_u32(b.lo)));
            vst1q_u32(reinterpret_cast<u32*>(ql + br * 32 + 16),
                      vcombine_u32(vget_high_u32(a.lo), vget_high_u32(b.lo)));
            vst1q_u32(reinterpret_cast<u32*>(qr + br * 32),
                      vcombine_u32(vget_low_u32(a.hi), vget_low_u32(b.hi)));
            vst1q_u32(reinterpret_cast<u32*>(qr + br * 32 + 16),
                      vcombine_u32(vget_high_u32(a.hi), vget_high_u32(b.hi)));
        }
    }
}

inline void StoreRows16(u8* tile, const Row16 in[8]) {
    for (u32 half = 0; half < 2; half++) {
        u8* ql = tile + half * 64;
        u8* qr = ql + 32;
        for (u32 br = 0; br < 2; br++) {
            const uint32x4x2_t z = vzipq_u32(vreinterpretq_u32_u16(in[half * 4 + br * 2]),
                                             vreinterpretq_u32_u16(in[half * 4 + br * 2 + 1]));
            vst1q_u32(reinterpret_cast<u32*>(ql + br * 16), z.val[0]);
            vst1q_u32(reinterpret_cast<u32*>(qr + br * 16), z.val[1]);
        }
    }
}

inline void StoreRows24(u8* tile, const Row24 in[8]) {
    for (u32 half = 0; half < 2; half++) {
        u8* ql = tile + half * 96;
        u8* qr = ql + 48;
        for (u32 br = 0; br < 2; br++) {
            const Row24& a = in[half * 4 + br * 2];
            const Row24& b = in[half * 4 + br * 2 + 1];
            uint8x8x3_t l, r;
            for (u32 ch = 0; ch < 3; ch++) {
                const uint16x4x2_t z =
                    vzip_u16(vreinterpret_u16_u8(a.c[ch]), vreinterpret_u16_u8(b.c[ch]));
                l.val[ch] = vreinterpret_u8_u16(z.val[0]);
                r.val[ch] = vreinterpret_u8_u16(z.val[1]);
            }
            vst3_u8(ql + br * 24, l);
            vst3_u8(qr + br * 24, r);
        }
    }
}

// ---- widening -------------------------------------------------------------------------------

inline uint8x8_t Widen5(uint16x8_t v) { // (v << 3) | (v >> 2), v in [0, 31]
    return vmovn_u16(vorrq_u16(vshlq_n_u16(v, 3), vshrq_n_u16(v, 2)));
}
inline uint8x8_t Widen6(uint16x8_t v) {
    return vmovn_u16(vorrq_u16(vshlq_n_u16(v, 2), vshrq_n_u16(v, 4)));
}
inline uint8x8_t Widen4(uint16x8_t v) {
    return vmovn_u16(vorrq_u16(vshlq_n_u16(v, 4), v));
}
inline uint8x8_t Widen4(uint8x8_t v) {
    return vorr_u8(vshl_n_u8(v, 4), v);
}

inline void StoreRGBA(u8* dst, uint8x8_t r, uint8x8_t g, uint8x8_t b, uint8x8_t a) {
    uint8x8x4_t v;
    v.val[0] = r;
    v.val[1] = g;
    v.val[2] = b;
    v.val[3] = a;
    vst4_u8(dst, v);
}

// Decodes one tile of `format` into linear RGBA8 or raw rows. Returns false when the
// format has no NEON path.
template <PixelFormat format, bool converted>
inline bool DecodeTile(u32 stride, const u8* tile, u8* linear) {
    constexpr u32 linear_bpp = converted ? 4 : GetFormatBytesPerPixel(format);
    const u32 stride_bytes = stride * linear_bpp;
    const uint8x8_t ff = vdup_n_u8(255);
    const uint8x8_t zero = vdup_n_u8(0);

    if constexpr (format == PixelFormat::RGBA8 || format == PixelFormat::D24S8) {
        Row32 rows[8];
        LoadRows32(tile, rows);
        for (u32 y = 0; y < 8; y++) {
            uint32x4_t lo = rows[y].lo, hi = rows[y].hi;
            if constexpr (format == PixelFormat::RGBA8 && converted) {
                lo = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(lo)));
                hi = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(hi)));
            } else if constexpr (format == PixelFormat::D24S8) {
                lo = vorrq_u32(vshlq_n_u32(lo, 8), vshrq_n_u32(lo, 24));
                hi = vorrq_u32(vshlq_n_u32(hi, 8), vshrq_n_u32(hi, 24));
            }
            u8* dst = LinearRow(linear, stride_bytes, y);
            vst1q_u32(reinterpret_cast<u32*>(dst), lo);
            vst1q_u32(reinterpret_cast<u32*>(dst + 16), hi);
        }
        return true;
    } else if constexpr (format == PixelFormat::RGB8) {
        Row24 rows[8];
        LoadRows24(tile, rows);
        for (u32 y = 0; y < 8; y++) {
            u8* dst = LinearRow(linear, stride_bytes, y);
            if constexpr (converted) {
                StoreRGBA(dst, rows[y].c[2], rows[y].c[1], rows[y].c[0], ff);
            } else {
                uint8x8x3_t v;
                v.val[0] = rows[y].c[0];
                v.val[1] = rows[y].c[1];
                v.val[2] = rows[y].c[2];
                vst3_u8(dst, v);
            }
        }
        return true;
    } else if constexpr (format == PixelFormat::RGB565 || format == PixelFormat::RGB5A1 ||
                         format == PixelFormat::RGBA4 || format == PixelFormat::D16 ||
                         format == PixelFormat::IA8 || format == PixelFormat::RG8) {
        Row16 rows[8];
        LoadRows16(tile, rows);
        for (u32 y = 0; y < 8; y++) {
            const uint16x8_t p = rows[y];
            u8* dst = LinearRow(linear, stride_bytes, y);
            if constexpr (format == PixelFormat::IA8) {
                const uint8x8_t i = vshrn_n_u16(p, 8);
                StoreRGBA(dst, i, i, i, vmovn_u16(p));
            } else if constexpr (format == PixelFormat::RG8) {
                StoreRGBA(dst, vshrn_n_u16(p, 8), vmovn_u16(p), zero, ff);
            } else if constexpr (!converted) {
                vst1q_u16(reinterpret_cast<u16*>(dst), p);
            } else if constexpr (format == PixelFormat::RGB565) {
                StoreRGBA(dst, Widen5(vshrq_n_u16(p, 11)),
                          Widen6(vandq_u16(vshrq_n_u16(p, 5), vdupq_n_u16(63))),
                          Widen5(vandq_u16(p, vdupq_n_u16(31))), ff);
            } else if constexpr (format == PixelFormat::RGB5A1) {
                const uint16x8_t one = vdupq_n_u16(1);
                const uint8x8_t a = vmovn_u16(vmulq_n_u16(vandq_u16(p, one), 255));
                StoreRGBA(dst, Widen5(vshrq_n_u16(p, 11)),
                          Widen5(vandq_u16(vshrq_n_u16(p, 6), vdupq_n_u16(31))),
                          Widen5(vandq_u16(vshrq_n_u16(p, 1), vdupq_n_u16(31))), a);
            } else if constexpr (format == PixelFormat::RGBA4) {
                const uint16x8_t m = vdupq_n_u16(15);
                StoreRGBA(dst, Widen4(vshrq_n_u16(p, 12)), Widen4(vandq_u16(vshrq_n_u16(p, 8), m)),
                          Widen4(vandq_u16(vshrq_n_u16(p, 4), m)), Widen4(vandq_u16(p, m)));
            } else {
                return false;
            }
        }
        return true;
    } else if constexpr (format == PixelFormat::I8 || format == PixelFormat::A8 ||
                         format == PixelFormat::IA4) {
        Row8 rows[8];
        LoadRows8(tile, rows);
        for (u32 y = 0; y < 8; y++) {
            const uint8x8_t v = rows[y];
            u8* dst = LinearRow(linear, stride_bytes, y);
            if constexpr (format == PixelFormat::I8) {
                StoreRGBA(dst, v, v, v, ff);
            } else if constexpr (format == PixelFormat::A8) {
                StoreRGBA(dst, zero, zero, zero, v);
            } else {
                const uint8x8_t i = Widen4(vshr_n_u8(v, 4));
                StoreRGBA(dst, i, i, i, Widen4(vand_u8(v, vdup_n_u8(15))));
            }
        }
        return true;
    } else if constexpr (format == PixelFormat::I4 || format == PixelFormat::A4) {
        Row4 rows[8];
        LoadRows4(tile, rows);
        const uint8x8_t m = vdup_n_u8(15);
        for (u32 y = 0; y < 8; y++) {
            // Low nibble is the even texel, high nibble the odd one (nibble index = Morton
            // offset, and the offset's low bit is x's low bit).
            const uint8x8x2_t z = vzip_u8(vand_u8(rows[y], m), vshr_n_u8(rows[y], 4));
            const uint8x8_t v = Widen4(z.val[0]);
            u8* dst = LinearRow(linear, stride_bytes, y);
            if constexpr (format == PixelFormat::I4) {
                StoreRGBA(dst, v, v, v, ff);
            } else {
                StoreRGBA(dst, zero, zero, zero, v);
            }
        }
        return true;
    } else {
        return false;
    }
}

// Encodes linear rows back into one Morton tile. Returns false when the format has no
// NEON path (the intensity formats need channel averaging; nothing hot flushes them).
template <PixelFormat format, bool converted>
inline bool EncodeTile(u32 stride, u8* tile, const u8* linear) {
    constexpr u32 linear_bpp = converted ? 4 : GetFormatBytesPerPixel(format);
    const u32 stride_bytes = stride * linear_bpp;

    if constexpr (format == PixelFormat::RGBA8 || format == PixelFormat::D24S8) {
        Row32 rows[8];
        for (u32 y = 0; y < 8; y++) {
            const u8* src = LinearRow(linear, stride_bytes, y);
            uint32x4_t lo = vld1q_u32(reinterpret_cast<const u32*>(src));
            uint32x4_t hi = vld1q_u32(reinterpret_cast<const u32*>(src + 16));
            if constexpr (format == PixelFormat::RGBA8 && converted) {
                lo = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(lo)));
                hi = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(hi)));
            } else if constexpr (format == PixelFormat::D24S8) {
                lo = vorrq_u32(vshrq_n_u32(lo, 8), vshlq_n_u32(lo, 24));
                hi = vorrq_u32(vshrq_n_u32(hi, 8), vshlq_n_u32(hi, 24));
            }
            rows[y].lo = lo;
            rows[y].hi = hi;
        }
        StoreRows32(tile, rows);
        return true;
    } else if constexpr (format == PixelFormat::RGB8) {
        Row24 rows[8];
        for (u32 y = 0; y < 8; y++) {
            const u8* src = LinearRow(linear, stride_bytes, y);
            if constexpr (converted) {
                const uint8x8x4_t v = vld4_u8(src);
                rows[y].c[0] = v.val[2];
                rows[y].c[1] = v.val[1];
                rows[y].c[2] = v.val[0];
            } else {
                const uint8x8x3_t v = vld3_u8(src);
                rows[y].c[0] = v.val[0];
                rows[y].c[1] = v.val[1];
                rows[y].c[2] = v.val[2];
            }
        }
        StoreRows24(tile, rows);
        return true;
    } else if constexpr (format == PixelFormat::RGB565 || format == PixelFormat::RGB5A1 ||
                         format == PixelFormat::RGBA4 || format == PixelFormat::D16) {
        Row16 rows[8];
        for (u32 y = 0; y < 8; y++) {
            const u8* src = LinearRow(linear, stride_bytes, y);
            if constexpr (!converted) {
                rows[y] = vld1q_u16(reinterpret_cast<const u16*>(src));
            } else {
                const uint8x8x4_t v = vld4_u8(src);
                const uint16x8_t r = vmovl_u8(v.val[0]);
                const uint16x8_t g = vmovl_u8(v.val[1]);
                const uint16x8_t b = vmovl_u8(v.val[2]);
                const uint16x8_t a = vmovl_u8(v.val[3]);
                if constexpr (format == PixelFormat::RGB565) {
                    rows[y] = vorrq_u16(vorrq_u16(vshlq_n_u16(vshrq_n_u16(r, 3), 11),
                                                  vshlq_n_u16(vshrq_n_u16(g, 2), 5)),
                                        vshrq_n_u16(b, 3));
                } else if constexpr (format == PixelFormat::RGB5A1) {
                    rows[y] =
                        vorrq_u16(vorrq_u16(vshlq_n_u16(vshrq_n_u16(r, 3), 11),
                                            vshlq_n_u16(vshrq_n_u16(g, 3), 6)),
                                  vorrq_u16(vshlq_n_u16(vshrq_n_u16(b, 3), 1), vshrq_n_u16(a, 7)));
                } else if constexpr (format == PixelFormat::RGBA4) {
                    rows[y] =
                        vorrq_u16(vorrq_u16(vshlq_n_u16(vshrq_n_u16(r, 4), 12),
                                            vshlq_n_u16(vshrq_n_u16(g, 4), 8)),
                                  vorrq_u16(vshlq_n_u16(vshrq_n_u16(b, 4), 4), vshrq_n_u16(a, 4)));
                } else {
                    return false;
                }
            }
        }
        StoreRows16(tile, rows);
        return true;
    } else {
        return false;
    }
}

// ---- ETC1 -----------------------------------------------------------------------------------

// Expands the 16 low bits of `bits` to 16 byte lanes of 0x00/0xFF, lane i = bit i.
inline uint8x16_t SpreadBits16(u32 bits) {
    const uint8x8_t mask = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t lo = vtst_u8(vdup_n_u8(static_cast<u8>(bits)), mask);
    const uint8x8_t hi = vtst_u8(vdup_n_u8(static_cast<u8>(bits >> 8)), mask);
    return vcombine_u8(lo, hi);
}

// Decodes one 4x4 ETC1 subtile (8 bytes of colour, optionally preceded by 8 bytes of
// packed 4-bit alpha) into 4 linear rows of RGBA8 at `dst` (row r at dst + r * stride_bytes,
// rows counted upward: the caller passes the row for subtile row 0 and a negative step for
// the bottom-up layout). Bit-exact with Pica::Texture::DecodeETC1TileRGBA8.
inline void DecodeETC1Subtile(const u8* colour, const u8* alpha, u8* dst, s32 row_step) {
    u64 word;
    std::memcpy(&word, colour, sizeof(word));
    const u32 sub_bits = static_cast<u32>(word & 0xFFFF);
    const u32 neg_bits = static_cast<u32>((word >> 16) & 0xFFFF);
    const bool flip = (word >> 32) & 1;
    const bool diff = (word >> 33) & 1;
    const u32 table2 = (word >> 34) & 7;
    const u32 table1 = (word >> 37) & 7;

    static constexpr u8 modifier_table[8][2] = {{2, 8},   {5, 17},  {9, 29},   {13, 42},
                                                {18, 60}, {24, 80}, {33, 106}, {47, 183}};

    // Base colours per block, exactly as the scalar code: 5-bit + signed 3-bit delta, or
    // two 4-bit colours, then widened.
    int base[2][3];
    if (diff) {
        const int r = (word >> 59) & 31, g = (word >> 51) & 31, b = (word >> 43) & 31;
        const auto sext3 = [](u64 v) { return static_cast<int>((v & 7) ^ 4) - 4; };
        const int dr = sext3(word >> 56), dg = sext3(word >> 48), db = sext3(word >> 40);
        const int c0[3] = {r, g, b};
        const int c1[3] = {r + dr, g + dg, b + db};
        for (int ch = 0; ch < 3; ch++) {
            // Convert5To8 on the (possibly out-of-range) int truncated to u8, as the scalar
            // path does through its u8 parameter.
            const u8 v0 = static_cast<u8>(c0[ch]);
            const u8 v1 = static_cast<u8>(c1[ch]);
            base[0][ch] = static_cast<u8>((v0 << 3) | (v0 >> 2));
            base[1][ch] = static_cast<u8>((v1 << 3) | (v1 >> 2));
        }
    } else {
        const u8 c[2][3] = {
            {static_cast<u8>((word >> 60) & 15), static_cast<u8>((word >> 52) & 15),
             static_cast<u8>((word >> 44) & 15)},
            {static_cast<u8>((word >> 56) & 15), static_cast<u8>((word >> 48) & 15),
             static_cast<u8>((word >> 40) & 15)},
        };
        for (int blk = 0; blk < 2; blk++) {
            for (int ch = 0; ch < 3; ch++) {
                base[blk][ch] = (c[blk][ch] << 4) | c[blk][ch];
            }
        }
    }

    // Lane t = texel index 4*x + y (column-major). Block 1 is x >= 2 unflipped, y >= 2
    // flipped; the table follows the block.
    const uint8x16_t blk1 =
        flip ? uint8x16_t{0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255}
             : uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255};
    const uint8x16_t sub = SpreadBits16(sub_bits);
    const uint8x16_t neg = SpreadBits16(neg_bits);

    const uint8x16_t mod1 =
        vbslq_u8(sub, vdupq_n_u8(modifier_table[table1][1]), vdupq_n_u8(modifier_table[table1][0]));
    const uint8x16_t mod2 =
        vbslq_u8(sub, vdupq_n_u8(modifier_table[table2][1]), vdupq_n_u8(modifier_table[table2][0]));
    const uint8x16_t mod = vbslq_u8(blk1, mod2, mod1);
    // Signed modifier in 16-bit lanes: (m ^ neg) - neg.
    const int16x8_t neg_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(neg)));
    const int16x8_t neg_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(neg)));
    const int16x8_t mod_lo =
        vsubq_s16(veorq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mod))), neg_lo), neg_lo);
    const int16x8_t mod_hi =
        vsubq_s16(veorq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mod))), neg_hi), neg_hi);

    uint8x16_t chan[4];
    for (int ch = 0; ch < 3; ch++) {
        const uint8x16_t b = vbslq_u8(blk1, vdupq_n_u8(static_cast<u8>(base[1][ch])),
                                      vdupq_n_u8(static_cast<u8>(base[0][ch])));
        const int16x8_t lo = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))), mod_lo);
        const int16x8_t hi = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b))), mod_hi);
        chan[ch] = vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
    }
    if (alpha) {
        // Nibble t of the 64-bit word is texel t: low nibble first.
        const uint8x8_t a = vld1_u8(alpha);
        const uint8x8x2_t z = vzip_u8(vand_u8(a, vdup_n_u8(15)), vshr_n_u8(a, 4));
        chan[3] = vcombine_u8(Widen4(z.val[0]), Widen4(z.val[1]));
    } else {
        chan[3] = vdupq_n_u8(255);
    }

    // Column-major lanes -> row-major: row y, column x is lane 4*x + y. Two rows per 8-lane
    // table lookup, then the channels are zipped into 16-byte rows.
    const uint8x8_t idx_lo = {0, 4, 8, 12, 1, 5, 9, 13};
    const uint8x8_t idx_hi = {2, 6, 10, 14, 3, 7, 11, 15};
    uint8x8x2_t tbl[4];
    for (int ch = 0; ch < 4; ch++) {
        tbl[ch].val[0] = vget_low_u8(chan[ch]);
        tbl[ch].val[1] = vget_high_u8(chan[ch]);
    }
    for (int pair = 0; pair < 2; pair++) {
        const uint8x8_t idx = pair == 0 ? idx_lo : idx_hi;
        const uint8x8x2_t rg = vzip_u8(vtbl2_u8(tbl[0], idx), vtbl2_u8(tbl[1], idx));
        const uint8x8x2_t ba = vzip_u8(vtbl2_u8(tbl[2], idx), vtbl2_u8(tbl[3], idx));
        const uint16x4x2_t row0 =
            vzip_u16(vreinterpret_u16_u8(rg.val[0]), vreinterpret_u16_u8(ba.val[0]));
        const uint16x4x2_t row1 =
            vzip_u16(vreinterpret_u16_u8(rg.val[1]), vreinterpret_u16_u8(ba.val[1]));
        u8* d0 = dst + (2 * pair) * row_step;
        u8* d1 = dst + (2 * pair + 1) * row_step;
        vst1_u16(reinterpret_cast<u16*>(d0), row0.val[0]);
        vst1_u16(reinterpret_cast<u16*>(d0 + 8), row0.val[1]);
        vst1_u16(reinterpret_cast<u16*>(d1), row1.val[0]);
        vst1_u16(reinterpret_cast<u16*>(d1 + 8), row1.val[1]);
    }
}

// One ETC1/ETC1A4 tile (four subtiles in Morton order) into linear RGBA8, rows bottom-up.
inline void DecodeETC1Tile(u32 stride, const u8* tile, u8* linear, bool has_alpha) {
    const u32 stride_bytes = stride * 4;
    const std::size_t subtile_size = has_alpha ? 16 : 8;
    for (u32 sy = 0; sy < 2; sy++) {
        for (u32 sx = 0; sx < 2; sx++) {
            const u8* sub = tile + (sx + 2 * sy) * subtile_size;
            const u8* alpha = has_alpha ? sub : nullptr;
            const u8* colour = has_alpha ? sub + 8 : sub;
            // Subtile row r is tile row sy*4 + r, i.e. linear row 7 - (sy*4 + r).
            u8* dst = LinearRow(linear, stride_bytes, sy * 4) + sx * 16;
            DecodeETC1Subtile(colour, alpha, dst, -static_cast<s32>(stride_bytes));
        }
    }
}

} // namespace VideoCore::NeonCodec

#endif // NEON
