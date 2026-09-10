// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace GxmRenderer::Usse {

/**
 * The PICA vertex emitter: a GXP vertex program for a PICA vertex program, written directly
 * as USSE instructions. The counterpart of cg_vs_shader_gen.cpp with the same interface:
 * the same attribute names (vs_in_attrN), the same uniforms (uniforms_f, uniforms_i,
 * flip_viewport, depth_scale, depth_offset, vs_in_regN), and the same fragment interface
 * by semantic (POSITION, COLOR0, TEXCOORD0-6), so the rasterizer's binding and uniform
 * upload are unchanged and every generated fragment program links against it. Where
 * SceShaccCg took seconds to tens of seconds per program on the console, this takes
 * microseconds.
 *
 * Returns the container, or empty (with `refusal` set) for a program it cannot express:
 * an instruction outside the set below, a bool uniform not baked in, a jump out of its
 * subroutine, deeper loop nesting than two, more live registers than the unified store
 * holds. The caller then falls back to the Cg path.
 */

/// Where a fragment-interface semantic comes from: an output attribute's component, or the
/// constant 1.0 when `attribute` is past the output count.
struct VsSemantic {
    uint8_t attribute;
    uint8_t component;
};

struct VsRequest {
    std::span<const uint32_t> code;    ///< the PICA program code
    std::span<const uint32_t> swizzle; ///< the operand descriptors
    uint32_t code_size;                ///< the written extent of `code`
    uint32_t main_offset;
    uint32_t num_outputs;              ///< output attributes (gs_output_attributes_count)
    std::array<uint32_t, 16> output_map; ///< PICA output register -> output attribute
    std::array<VsSemantic, 24> semantics; ///< by VSOutputAttributes::Semantic
    /// The draw's shape, as cg_vs_shader_gen's VSExtra: input registers no loader feeds
    /// (uniforms vs_in_regN), the bool uniforms baked in, components each loader supplies.
    uint16_t default_regs;
    uint16_t bools;
    uint16_t bool_mask;
    std::array<uint8_t, 16> reg_components;
};

/// `level` 2 runs the IR's optimising passes (the second tier).
[[nodiscard]] std::vector<uint8_t> EmitVertexProgram(const VsRequest& request, uint32_t& input_regs,
                                                     std::string* refusal = nullptr, int level = 1);

} // namespace GxmRenderer::Usse
