// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <algorithm>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "video_core/pica/vertex_loader.h"

namespace Pica {

VertexLoader::VertexLoader(Memory::MemorySystem& memory_, const PipelineRegs& regs)
    : memory{memory_} {
    const auto& attribute_config = regs.vertex_attributes;
    num_total_attributes = attribute_config.GetNumTotalAttributes();

    vertex_attribute_sources.fill(0xdeadbeef);

    for (u32 i = 0; i < 16; i++) {
        vertex_attribute_is_default[i] = attribute_config.IsDefaultAttribute(i);
    }

    // Setup attribute data from loaders
    for (u32 loader = 0; loader < 12; ++loader) {
        const auto& loader_config = attribute_config.attribute_loaders[loader];

        u32 offset = 0;

        // TODO: What happens if a loader overwrites a previous one's data?
        for (u32 component = 0; component < loader_config.component_count; ++component) {
            if (component >= 12) {
                LOG_ERROR(HW_GPU,
                          "Overflow in the vertex attribute loader {} trying to load component {}",
                          loader, component);
                continue;
            }

            u32 attribute_index = loader_config.GetComponent(component);
            if (attribute_index < 12) {
                offset = Common::AlignUp(offset,
                                         attribute_config.GetElementSizeInBytes(attribute_index));
                vertex_attribute_sources[attribute_index] = loader_config.data_offset + offset;
                vertex_attribute_strides[attribute_index] =
                    static_cast<u32>(loader_config.byte_count);
                vertex_attribute_loader[attribute_index] = static_cast<u8>(loader);
                loader_data_offset[loader] = loader_config.data_offset;
                loader_byte_count[loader] = static_cast<u32>(loader_config.byte_count);
                loader_used[loader] = true;
                vertex_attribute_formats[attribute_index] =
                    attribute_config.GetFormat(attribute_index);
                vertex_attribute_elements[attribute_index] =
                    attribute_config.GetNumElements(attribute_index);
                offset += attribute_config.GetStride(attribute_index);
            } else if (attribute_index < 16) {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings,
                // respectively
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            } else {
                UNREACHABLE(); // This is truly unreachable due to the number of bits for each
                               // component
            }
        }
    }
}

VertexLoader::~VertexLoader() = default;

DrawArenaLayout VertexLoader::ComputeArenaLayout(u32 min_vertex, u32 max_vertex,
                                                 u32 index_bytes, bool ring) const {
    // A loader with byte_count 0 is a constant stream: every vertex reads the same bytes, so it
    // occupies one fixed-size cell. Must match CopyIntoArena's layout exactly.
    constexpr u32 ZERO_STRIDE_BYTES = 64;
    DrawArenaLayout layout;
    layout.ring = ring;
    for (u32 loader = 0; loader < 12; ++loader) {
        if (!loader_used[loader]) {
            continue;
        }
        layout.used_mask |= static_cast<u16>(1u << loader);
        if (loader_byte_count[loader] == 0) {
            layout.zero_stride = true;
        }
        if (ring) {
            layout.total = Common::AlignUp(layout.total, 16u);
        }
        layout.loader_offset[loader] = layout.total;
        layout.total += loader_byte_count[loader] == 0
                            ? ZERO_STRIDE_BYTES
                            : (max_vertex - min_vertex + 1) * loader_byte_count[loader];
    }
    if (ring) {
        layout.total = Common::AlignUp(layout.total, 16u);
    }
    layout.index_offset = layout.total;
    layout.total += index_bytes;
    if (ring) {
        layout.total = Common::AlignUp(std::max(layout.total, 1u),
                                       static_cast<u32>(sizeof(OutputVertex)));
    }
    return layout;
}

void VertexLoader::WriteRingIndices(u8* dst, const u8* src, bool index_u16, u32 count,
                                    u32 min_vertex) {
    u16* const out = reinterpret_cast<u16*>(dst);
    if (index_u16) {
        const u16* const in = reinterpret_cast<const u16*>(src);
        for (u32 n = 0; n < count; n++) {
            out[n] = static_cast<u16>(in[n] - min_vertex);
        }
    } else {
        for (u32 n = 0; n < count; n++) {
            out[n] = static_cast<u16>(src[n] - min_vertex);
        }
    }
}

void VertexLoader::CopyIntoArena(PAddr base_address, u32 min_vertex, u32 max_vertex,
                                 const DrawArenaLayout& layout, u8* dst) const {
    constexpr u32 ZERO_STRIDE_BYTES = 64;
    for (u32 loader = 0; loader < 12; ++loader) {
        if (!loader_used[loader]) {
            continue;
        }
        const PAddr start =
            base_address + loader_data_offset[loader] + min_vertex * loader_byte_count[loader];
        u32 bytes = loader_byte_count[loader] == 0
                        ? ZERO_STRIDE_BYTES
                        : (max_vertex - min_vertex + 1) * loader_byte_count[loader];
        const u8* const src = memory.GetPhysicalPointer(start);
        if (src == nullptr) {
            std::memset(dst + layout.loader_offset[loader], 0, bytes);
            continue;
        }
        // Clamp to the end of the physical region so a buffer straddling the region tail cannot
        // read past the mapping; the hardware would have wrapped or read garbage there anyway.
        const auto region = memory.GetPhysMemRegionInfo(start);
        if (region.valid()) {
            bytes = std::min<u64>(bytes, region.region_end - start);
        }
        std::memcpy(dst + layout.loader_offset[loader], src, bytes);
    }
}

void VertexLoader::AttachArena(const u8* arena, const DrawArenaLayout& layout, u32 min_vertex) {
    for (s32 i = 0; i < num_total_attributes; ++i) {
        if (vertex_attribute_is_default[i]) {
            continue;
        }
        const u32 loader = vertex_attribute_loader[i];
        const u32 within_loader = vertex_attribute_sources[i] - loader_data_offset[loader];
        // Biased by -min_vertex*stride so lookups use the absolute vertex id.
        vertex_attribute_data[i] = arena + layout.loader_offset[loader] + within_loader -
                                   min_vertex * vertex_attribute_strides[i];
    }
}

void VertexLoader::Rebase(PAddr base_address) {
    std::array<const u8*, 12> loader_host{};
    for (u32 loader = 0; loader < 12; ++loader) {
        if (loader_used[loader] && loader_byte_count[loader] != 0) {
            loader_host[loader] =
                memory.GetPhysicalPointer(base_address + loader_data_offset[loader]);
        }
    }
    static constexpr std::array<u8, 64> zeroes{};
    for (s32 i = 0; i < num_total_attributes; ++i) {
        if (vertex_attribute_is_default[i]) {
            continue;
        }
        const u32 loader = vertex_attribute_loader[i];
        const u8* const host = loader_host[loader];
        if (host == nullptr) {
            // Invalid base: read zeroes, with a stride of zero folded in by pointing every
            // vertex at the same block.
            vertex_attribute_data[i] = zeroes.data();
            continue;
        }
        vertex_attribute_data[i] = host + (vertex_attribute_sources[i] - loader_data_offset[loader]);
    }
}

void VertexLoader::LoadVertex(u32 vertex, AttributeBuffer& input,
                              AttributeBuffer& input_default_attributes) const {
    for (s32 i = 0; i < num_total_attributes; ++i) {
        // Load the default attribute if we're configured to do so
        if (vertex_attribute_is_default[i]) {
            input[i] = input_default_attributes[i];
            continue;
        }

        // TODO(yuriks): In this case, no data gets loaded and the vertex
        // remains with the last value it had. This isn't currently maintained
        // as global state, however, and so won't work in Citra yet.
        if (vertex_attribute_elements[i] == 0) {
            LOG_ERROR(HW_GPU, "Vertex retension unimplemented");
            continue;
        }

        // Per-vertex data comes from the snapshot arena, never from guest memory.
        const u8* const source = vertex_attribute_data[i] + vertex_attribute_strides[i] * vertex;

        switch (vertex_attribute_formats[i]) {
        case PipelineRegs::VertexAttributeFormat::BYTE:
            LoadAttribute<s8>(source, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::UBYTE:
            LoadAttribute<u8>(source, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::SHORT:
            LoadAttribute<s16>(source, i, input);
            break;
        case PipelineRegs::VertexAttributeFormat::FLOAT:
            LoadAttribute<f32>(source, i, input);
            break;
        }

        // Default attribute values set if array elements have < 4 components. This
        // is *not* carried over from the default attribute settings even if they're
        // enabled for this attribute.
        for (u32 comp = vertex_attribute_elements[i]; comp < 4; comp++) {
            input[i][comp] = comp == 3 ? f24::One() : f24::Zero();
        }
    }
}

} // namespace Pica
