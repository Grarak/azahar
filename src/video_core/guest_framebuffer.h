// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"
#include "video_core/pica/regs_external.h"

namespace Memory {
class MemorySystem;
}
namespace Pica {
class PicaCore;
}

namespace VideoCore {

class GPU;

/// What a frontend that samples guest memory itself needs to know about one screen's
/// framebuffer this frame: where it is, how it is laid out, and a pointer into the emulator's
/// view of it. `screen` is 0 for the top screen, 1 for the bottom.
struct GuestFramebuffer {
    PAddr addr;       ///< guest physical address of the framebuffer
    const u8* pixels; ///< host pointer to the guest framebuffer (null when unmapped)
    /// `pixels` is the GPU's VBlank snapshot of a CPU-written framebuffer rather than guest
    /// memory: DisplaySnapshotBytes long, page-aligned, and to be sampled as it is, never
    /// through the surface cache.
    bool snapshot;
    /// A renderer that keeps the framebuffer on the GPU may hand the frontend a texture of
    /// it instead (a SceGxmTexture on the Vita), laid out like the guest buffer: 240-texel
    /// rows, one per scanline column. Null when guest memory is the source.
    const void* texture;
    /// The framebuffer's place inside `texture`, in texels (a title may render a larger
    /// buffer and present part of it); all zero when the texture is the framebuffer.
    u32 tex_x, tex_y, tex_width, tex_height;
    u32 stride;       ///< bytes per 240-pixel scanline column
    u32 height;       ///< 400 (top) or 320 (bottom)
    Pica::PixelFormat format;
    /// The LCD color fill overrides the framebuffer entirely when enabled; sampling guest
    /// memory would miss it, so the frontend draws a solid quad instead.
    bool fill_enabled;
    u8 fill_r, fill_g, fill_b;
};

/// Which of a screen's candidate framebuffer addresses to present, for renderers whose output
/// lands in guest memory. Prefers the buffer the render thread most recently finished writing
/// (monotonic content); falls back to the register selection with the fill-hold for titles no
/// blit ever targets. Shared by the software and GXM renderers.
class GuestFramebufferPicker {
public:
    GuestFramebufferPicker(Pica::PicaCore& pica, Memory::MemorySystem& memory);

    [[nodiscard]] GuestFramebuffer Get(GPU* owner_gpu, int screen);
    [[nodiscard]] PAddr PickFramebufferAddr(GPU* owner_gpu, u32 fb_id);

private:
    Pica::PicaCore& pica;
    Memory::MemorySystem& memory;
    /// Last framebuffer address handed out per screen, for the fill-hold: while a fill to the
    /// new address is still queued, the old one keeps being presented.
    std::array<PAddr, 2> held_addr{};
};

} // namespace VideoCore
