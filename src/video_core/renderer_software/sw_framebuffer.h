// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstring>
#include <map>
#include <vector>
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/utils.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct FramebufferRegs;
}

namespace SwRenderer {

/**
 * Host-side storage for the colour and depth surfaces the software renderer draws into.
 *
 * The software renderer is the PICA: colour and depth are its output and the guest expects them in
 * its own memory. Producing them there as they are drawn puts the render thread and its stripe
 * workers in guest memory at the same time as the natively executing guest, which wedges it. This
 * is the same answer the GL rasterizer cache gives: surfaces live in host memory and reach guest
 * memory only at the flush points the emulation thread waits on.
 *
 * Every live surface is held, not one: a title renders to several and swaps between them within a
 * frame, so a single-entry cache would have to write back on eviction - mid-frame, on the render
 * thread, which is exactly where a write cannot be made to wait for anything.
 */
class SurfaceCache {
public:
    explicit SurfaceCache(Memory::MemorySystem& memory) : memory{memory} {}

    /// Host storage for [addr, addr+size), loaded from guest memory if this is the first sight of
    /// it or if something has rewritten it since. Marks it as holding content guest memory does
    /// not, so a later flush hands it back.
    u8* GetForWrite(PAddr addr, u32 size);

    /// Writes back everything held for [addr, addr+size). Only ever called from a context the
    /// emulation thread is waiting on.
    void Flush(PAddr addr, u32 size);

    /// Something else is about to rewrite [addr, addr+size). Hands back what is held first - the
    /// rewrite is queued behind this and wins - then marks it for reload.
    void Invalidate(PAddr addr, u32 size);

    /// Every byte of [addr, addr+size) in guest memory has just been rewritten (a fill or
    /// transfer, ordered after everything drawn). Surfaces absorb the bytes: data takes them as
    /// current content and shadow takes them as baseline, so nothing is dropped and nothing has
    /// to reload. Bytes past a surface's loaded span go into data only - the shadow keeps its
    /// length, and the write-back tail still treats everything past it as content to hand back.
    void UpdateFromGuest(PAddr addr, u32 size);

    /// Number of surfaces held, for the eviction warning.
    [[nodiscard]] std::size_t Size() const {
        return surfaces.size();
    }

private:
    struct Surface {
        u32 size{};
        std::vector<u8> data;
        /// What guest memory held when this surface was loaded. Write-back sends only the bytes
        /// that differ from it, so guest writes the cache could not observe are not overwritten.
        std::vector<u8> shadow;
        /// Host storage holds content guest memory does not.
        bool dirty{};
        /// Guest memory holds content host storage does not.
        bool reload{true};
    };

    /// Hands back and drops every entry overlapping [addr, addr+size) other than one starting
    /// exactly there, so no byte is ever held by two surfaces at once.
    void EvictOverlapping(PAddr addr, u32 size);

    void WriteBack(PAddr addr, Surface& surface);
    void Reload(PAddr addr, Surface& surface);

    Memory::MemorySystem& memory;
    std::map<PAddr, Surface> surfaces;
};

class Framebuffer {
public:
    explicit Framebuffer(Memory::MemorySystem& memory, const Pica::FramebufferRegs& framebuffer);
    ~Framebuffer();

    /// Points the colour and depth pointers at their host-side surfaces, creating or reloading
    /// them as needed.
    void Bind();

    /// Hands [addr, addr+size) back to guest memory.
    void FlushRange(PAddr addr, u32 size) {
        cache.Flush(addr, size);
    }

    /// Marks [addr, addr+size) as about to be rewritten by something else.
    void InvalidateRange(PAddr addr, u32 size) {
        cache.Invalidate(addr, size);
    }

    /// Absorbs a finished guest-memory rewrite of [addr, addr+size) into live surfaces.
    void UpdateFromGuest(PAddr addr, u32 size) {
        cache.UpdateFromGuest(addr, size);
    }

    /// Draws a pixel at the specified coordinates.
    void DrawPixel(u32 x, u32 y, const Common::Vec4<u8>& color) const;

    /// Returns the current color at the specified coordinates.
    [[nodiscard]] const Common::Vec4<u8> GetPixel(u32 x, u32 y) const;

    /// Returns the depth value at the specified coordinates.
    [[nodiscard]] u32 GetDepth(u32 x, u32 y) const;

    /// Returns the stencil value at the specified coordinates.
    [[nodiscard]] u8 GetStencil(u32 x, u32 y) const;

    /// Stores the provided depth value at the specified coordinates.
    void SetDepth(u32 x, u32 y, u32 value) const;

    /// Stores the provided stencil value at the specified coordinates.
    void SetStencil(u32 x, u32 y, u8 value) const;

    /**
     * Byte offsets, into the colour and depth buffers, of `count` horizontally consecutive
     * pixels starting at (x, y). Either output may be null. The per-row work - the vertical
     * flip, the coarse-row stride, the format's pixel size and the y half of the Morton
     * coordinate - is done once for the whole block; the accessors above repeat all of it on
     * every call, and a span pixel goes through up to six of them. Defined here because the
     * span loops call it once per eight pixels and the call was measurable.
     */
    void BlockOffsets(u32 x, u32 y, u32 count, u32* color_offsets, u32* depth_offsets) const {
        const auto& framebuffer = regs.framebuffer;
        // The render framebuffer is laid out from bottom to top.
        // NOTE: The framebuffer height register contains the actual FB height minus one.
        const u32 fy = framebuffer.height - y;
        const u32 coarse_y = fy & ~7;
        // The Morton coordinate is separable and the block shares a row, so its y half and
        // the coarse-row byte offset are evaluated once instead of once per pixel.
        const u32 ymorton = VideoCore::MortonInterleaveY(fy);
        const u32 color_bpp =
            Pica::BytesPerPixel(Pica::PixelFormat(framebuffer.color_format.Value()));
        const u32 depth_bpp =
            Pica::FramebufferRegs::BytesPerDepthPixel(framebuffer.depth_format);
        const u32 color_base = coarse_y * framebuffer.width * color_bpp;
        const u32 depth_base = coarse_y * framebuffer.width * depth_bpp;
        for (u32 j = 0; j < count; ++j) {
            const u32 px = x + j;
            const u32 i = VideoCore::MortonInterleaveX(px) + ymorton + (px & ~7) * 8;
            if (color_offsets != nullptr) {
                color_offsets[j] = i * color_bpp + color_base;
            }
            if (depth_offsets != nullptr) {
                depth_offsets[j] = i * depth_bpp + depth_base;
            }
        }
    }

    /**
     * Reads, and writes back, the colours of a block of pixels named by BlockOffsets, for the
     * lanes set in `mask`. Colours are four bytes in r, g, b, a order, the layout the span
     * loops already keep them in. One format decision covers the block instead of one per
     * pixel, and the common RGBA8 case is then a byte-reversed word each.
     */
    void GetPixels(const u32* offsets, u32 mask, u8 (*out)[4]) const;
    void DrawPixels(const u32* offsets, u32 mask, const u8 (*colors)[4]) const;

    /// The same accessors, addressed by a byte offset from BlockOffsets.
    [[nodiscard]] Common::Vec4<u8> GetPixelAt(u32 offset) const;
    void DrawPixelAt(u32 offset, const Common::Vec4<u8>& color) const;
    [[nodiscard]] u32 GetDepthAt(u32 offset) const;
    void SetDepthAt(u32 offset, u32 value) const;
    [[nodiscard]] u8 GetStencilAt(u32 offset) const;
    void SetStencilAt(u32 offset, u8 value) const;

    /// True when the depth buffer is D24S8, whose pixel is one 32-bit word: depth in the low
    /// three bytes, stencil in the top one. The block depth/stencil test works on those words.
    [[nodiscard]] bool DepthIsD24S8() const {
        return regs.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8;
    }

    /// Loads and stores whole D24S8 words for the lanes set in `mask`. The test reads depth
    /// and stencil and writes both back, so one word each way replaces up to four accesses.
    void LoadDepthWords(const u32* offsets, u32 mask, u32* out) const {
        for (u32 lane = 0; lane < 8; ++lane) {
            if ((mask >> lane) & 1) {
                std::memcpy(&out[lane], depth_buffer + offsets[lane], sizeof(u32));
            }
        }
    }

    void StoreDepthWords(const u32* offsets, u32 mask, const u32* in) const {
        for (u32 lane = 0; lane < 8; ++lane) {
            if ((mask >> lane) & 1) {
                std::memcpy(depth_buffer + offsets[lane], &in[lane], sizeof(u32));
            }
        }
    }

    /// Draws a pixel to the shadow buffer.
    void DrawShadowMapPixel(u32 x, u32 y, u32 depth, u8 stencil) const;

private:
    Memory::MemorySystem& memory;
    const Pica::FramebufferRegs& regs;
    // Impossible physical addresses, so the first Bind() always resolves the pointers. Left
    // indeterminate, a reallocated Framebuffer can inherit its predecessor's cached addresses from
    // the reused heap block and then never refresh the null buffer pointers beside them.
    PAddr color_addr = 0xFFFFFFFF;
    u8* color_buffer{};
    PAddr depth_addr = 0xFFFFFFFF;
    u8* depth_buffer{};

    /// Byte extent of each surface, from the registers BlockOffsets addresses through.
    [[nodiscard]] u32 ColorBufferSize() const;
    [[nodiscard]] u32 DepthBufferSize() const;

    SurfaceCache cache;
};

u8 PerformStencilAction(Pica::FramebufferRegs::StencilAction action, u8 old_stencil, u8 ref);

Common::Vec4<u8> EvaluateBlendEquation(const Common::Vec4<u8>& src,
                                       const Common::Vec4<u8>& srcfactor,
                                       const Common::Vec4<u8>& dest,
                                       const Common::Vec4<u8>& destfactor,
                                       Pica::FramebufferRegs::BlendEquation equation);

u8 LogicOp(u8 src, u8 dest, Pica::FramebufferRegs::LogicOp op);

} // namespace SwRenderer
