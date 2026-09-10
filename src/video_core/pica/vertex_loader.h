// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "core/memory.h"
#include "video_core/pica/draw_payload.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/regs_pipeline.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {

class VertexLoader {
public:
    explicit VertexLoader(Memory::MemorySystem& memory_, const PipelineRegs& regs);
    ~VertexLoader();

    /**
     * Copies every attribute byte the draw can reference — each loader array's
     * [min_vertex, max_vertex] range — into the arena, and repoints the per-attribute base
     * pointers at it. After this, LoadVertex touches no guest memory at all: fetches become one
     * pointer add each, where they used to be a GetPhysicalPointer region lookup per attribute
     * per vertex. The snapshot is also what unties the fetch from the emulation timeline — the
     * guest may rewrite its buffers the moment P3D fires without affecting anything read from
     * here.
     */
    DrawArenaLayout ComputeArenaLayout(u32 min_vertex, u32 max_vertex, u32 index_bytes,
                                       bool ring = false) const;

    /// Writes a draw's indices into a ring-layout arena: 16-bit, rebased to min_vertex.
    static void WriteRingIndices(u8* dst, const u8* src, bool index_u16, u32 count,
                                 u32 min_vertex);

    /// Emulation thread: copies each loader array's [min, max] range (clamped to its physical
    /// region) into dst following the layout. The index slice is copied by the caller.
    void CopyIntoArena(PAddr base_address, u32 min_vertex, u32 max_vertex,
                       const DrawArenaLayout& layout, u8* dst) const;

    /// Render thread: points the per-attribute bases into a shipped arena, pre-biased by
    /// -min_vertex*stride so LoadVertex indexes with absolute vertex ids.
    void AttachArena(const u8* arena, const DrawArenaLayout& layout, u32 min_vertex);

    /**
     * The no-copy variant: resolves each loader array's base to a host pointer once per draw and
     * repoints the per-attribute bases at guest memory directly. Same fetch fast path as
     * Snapshot — one pointer add per attribute, no region lookup per access — without the copy;
     * the fetch stays tied to the emulation timeline, which is correct while shading runs on it.
     */
    void Rebase(PAddr base_address);

    void LoadVertex(u32 vertex, AttributeBuffer& input,
                    AttributeBuffer& input_default_attributes) const;

    template <typename T>
    void LoadAttribute(const u8* source, u32 attrib, AttributeBuffer& out) const {
        const T* data = reinterpret_cast<const T*>(source);
        for (u32 comp = 0; comp < vertex_attribute_elements[attrib]; ++comp) {
            out[attrib][comp] = f24::FromFloat32(data[comp]);
        }
    }

    int GetNumTotalAttributes() const {
        return num_total_attributes;
    }

private:
    Memory::MemorySystem& memory;
    std::array<u32, 16> vertex_attribute_sources;
    std::array<u32, 16> vertex_attribute_strides{};
    std::array<PipelineRegs::VertexAttributeFormat, 16> vertex_attribute_formats;
    std::array<u32, 16> vertex_attribute_elements{};
    std::array<bool, 16> vertex_attribute_is_default;
    std::array<u8, 16> vertex_attribute_loader{};
    int num_total_attributes = 0;

    /// Loader-array geometry recorded at construction, consumed by Snapshot.
    std::array<u32, 12> loader_data_offset{};
    std::array<u32, 12> loader_byte_count{};
    std::array<bool, 12> loader_used{};

    /// Per-attribute base pointers into the snapshot arena, pre-biased by -min_vertex*stride so
    /// LoadVertex indexes with the absolute vertex id.
    std::array<const u8*, 16> vertex_attribute_data{};
};

} // namespace Pica
