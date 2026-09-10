// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/gxm.h>
#include "common/common_types.h"

namespace GxmRenderer {

/**
 * What the renderer needs from the frontend's GXM layer (citra_vita/gxm_present),
 * which owns the library, the immediate context and the shader patcher. The frontend
 * fills this in after its own init and clears it before shutdown; the renderer reads it on
 * the render thread, which is the one thread that may use the context.
 */
struct GxmDevice {
    SceGxmContext* context{};
    SceGxmShaderPatcher* patcher{};
    /// Long-lived GPU-visible memory, uncached LPDDR: rings, index buffers, small tables.
    void* (*alloc_mapped)(u32 size){};
    /// The same, but ordinary cached memory. Cached blocks are GPU-snoop-coherent here, so
    /// the only question is which side pays: the CPU for uncached stores, or the GPU for
    /// snooped reads. Uncached is right for anything the GPU reads repeatedly; a buffer the
    /// CPU fills once and the vertex fetch reads once may not be that.
    void* (*alloc_mapped_cached)(u32 size){};
    /// Long-lived CDRAM: render targets and textures the GPU alone touches.
    void* (*alloc_cdram)(u32 size){};
    /// The largest block alloc_cdram (or alloc_mapped) could still return, so a caller that
    /// takes memory in slabs can ask for one that fits instead of guessing. Zero when there
    /// is no pool to ask, in which case an allocation is the only way to find out.
    u32 (*largest_free)(bool cdram){};
    /// Transient GPU-visible memory valid for the frame being built (vertices, uniforms).
    void* (*frame_alloc)(u32 size, u32 align){};
    /// Releases a block from alloc_mapped or alloc_cdram. The caller has made sure the GPU
    /// is done with it (sceGxmFinish or a notification).
    void (*free_block)(void* block){};
    /**
     * Claims one 32-bit slot in libgxm's notification region, so the renderer and the
     * presentation layer cannot pick the same one. Null when none is left.
     *
     * A scene that ends with a fragment notification naming this slot writes its value there
     * once the GPU has finished the scene's fragment processing - which makes "has the GPU
     * finished with this memory yet" a plain load rather than a wait.
     */
    volatile u32* (*alloc_notification)(){};
    /// The Razor GPU capture module is loaded, so user markers reach a capture and are worth
    /// the string that names them. False on a normal run, where they would be pure cost.
    bool razor{};
};

void SetDevice(const GxmDevice& device);
[[nodiscard]] const GxmDevice& Device();
[[nodiscard]] bool HasDevice();

} // namespace GxmRenderer
