// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"
#include "video_core/pica/output_vertex.h"

namespace Pica {

/// Byte layout of a draw's snapshot arena: per-loader offsets plus the index slice at the end.
/// Computed identically on both threads from the same (shipped) registers.
struct DrawArenaLayout {
    std::array<u32, 12> loader_offset{};
    u32 index_offset = 0;
    u32 total = 0;
    /// Ring layout: every loader block 16-byte aligned, the indices 16-bit and rebased to
    /// vertex_min (so the block is what the GXM vertex streams read as it is), the total a
    /// multiple of the software vertex stride. Off: the plain arena layout the GL backend and
    /// the software path read.
    bool ring = false;
    /// Bit per loader that owns arena bytes.
    u16 used_mask = 0;
    /// A used loader with byte_count 0 exists; the GL vertex-array path cannot express it.
    bool zero_stride = false;
};

/**
 * One draw call, frozen for execution on the render-thread mirror: the attribute and index bytes
 * it references live in a snapshot arena, so the guest may rewrite its buffers the moment the
 * emulation thread moves on. Immediate-mode draws carry the assembled input vertex instead of an
 * arena.
 */
struct DrawPayload {
    const u8* arena = nullptr;
    DrawArenaLayout layout{};
    u32 vertex_min = 0;
    u32 vertex_max = 0;
    u32 num_vertices = 0;
    u32 vertex_offset = 0;
    bool is_indexed = false;
    bool index_u16 = false;
    /// Execute through the GLSL hardware-shader path (fed from the arena); on failure the
    /// software mirror path runs instead.
    bool hw = false;
    /// The arena is inside the renderer's GPU-visible vertex ring, written there by the
    /// emulation thread; `arena` points into it, the draw reads it in place, and `ring_end` is
    /// the ring position the render thread retires once the GPU is done with the draw.
    bool in_ring = false;
    u32 ring_offset = 0;
    u64 ring_end = 0;
    bool immediate = false;
    bool immediate_reset_geometry = false;
    AttributeBuffer immediate_input{};
};

} // namespace Pica
