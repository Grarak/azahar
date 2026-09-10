// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GxmRenderer::Usse {

/**
 * Writes a GXP container (the file format psp2cgc emits and libgxm loads) around a
 * fragment program's instruction words, so a program made by the emitter is
 * indistinguishable to sceGxmProgramCheck, the shader patcher and
 * sceGxmProgramFindParameterByName from a compiled one.
 *
 * The layout follows psp2cgc's output field for field, with the field meanings taken
 * from Vita3K's gxm/types.h and gxp_parser.cpp and the constant bits from reference
 * programs: header, the varyings block with
 * one descriptor per interpolated input, the phase table, the secondary and primary
 * instruction words, the dependent sampler table, the parameter containers, the parameter
 * table, the names.
 *
 * Fragment programs: every uniform is a float (F32) in the default uniform buffer
 * (container 14), register-resident; every sampler is a dependent read (SMP from a
 * coordinate the program computed).
 *
 * Vertex programs (`vertex`): the attributes are F32x4 parameters at resource index 4n
 * (four primary attribute registers each, as the compiler lays them out whatever the
 * declared width), the register-resident uniforms are the first `uniform_floats` floats of
 * the default uniform buffer, and a uniform past them is read from memory through the
 * buffer pointer libgxm plants in the data container's first secondary attribute
 * (`buffer_pointer`, value = buffer address + `ldst_base_value`; the compiler's convention
 * is the byte offset of the memory-resident part minus four, see usse_encoder Lda32).
 */

/// Where a sampler's control words sit in the secondary attributes: after the uniforms
/// and the literals, four registers per sampler in declaration order.
inline uint32_t SamplerSaOffset(uint32_t uniform_floats, uint32_t literal_count, uint32_t index) {
    return uniform_floats + literal_count + index * 4;
}
/// Where literal `index` sits.
inline uint32_t LiteralSaOffset(uint32_t uniform_floats, uint32_t index) {
    return uniform_floats + index;
}

/// An interpolated input as the hardware names it: the id in the descriptor's bits 12-15.
enum class FragmentInput : uint32_t {
    TexCoord0 = 0x0,
    TexCoord1 = 0x1,
    TexCoord2 = 0x2,
    TexCoord3 = 0x3,
    TexCoord4 = 0x4,
    TexCoord5 = 0x5,
    TexCoord6 = 0x6,
    TexCoord7 = 0x7,
    TexCoord8 = 0x8,
    TexCoord9 = 0x9,
    Color0 = 0xA,
    Color1 = 0xB,
    Fog = 0xC,
    Position = 0xD, ///< WPOS
};

struct GxpInput {
    FragmentInput input;
    uint8_t components; ///< 1-4, each one PA register (f32)
};

struct GxpUniform {
    std::string name;
    uint32_t sa_offset;  ///< first SA register, in the default uniform buffer
    uint8_t components;  ///< 1-4
    uint32_t array_size; ///< 1 for a scalar or vector
};

struct GxpSampler {
    std::string name;
    uint32_t unit; ///< TEXUNITn, 0-15
};

struct GxpAttribute {
    std::string name;
    uint32_t resource; ///< the first of its four primary attribute registers
};

struct GxpProgram {
    /// In PA order: the first input starts at pa0, each takes `components` registers.
    std::vector<GxpInput> inputs;
    std::vector<GxpUniform> uniforms;
    std::vector<GxpSampler> samplers;
    uint32_t pa_count = 0;   ///< primary attribute registers the program touches
    uint32_t temp_count = 0; ///< temporaries (r0..) the program touches
    /// The default uniform buffer's size in floats: the highest SA a uniform occupies + 1.
    uint32_t uniform_floats = 0;
    /// Constants libgxm writes into the data container's first SA registers at load, one
    /// float each, in this order; raw bits, so any 32-bit pattern can be planted.
    std::vector<uint32_t> literals;
    bool uses_discard = false;
    bool writes_depth = false;
    /// Instruction index at which each phase begins (the first is 0); one entry per PHAS.
    std::vector<uint32_t> phase_starts{0};
    std::vector<uint64_t> primary;
    std::vector<uint64_t> secondary;
    /// Identity fields, informational: psp2cgc writes hashes here, libgxm does not read them.
    uint32_t binary_guid = 0;
    uint32_t source_guid = 0;

    // --- vertex programs ---
    bool vertex = false;
    /// Program flag 0x2: one vertex per USSE instance. The compiler sets it on every
    /// program that branches on per-vertex data (probes p_e, p_f: "per-instance mode" in
    /// psp2shaderperf's report); the others run several vertices per instance.
    bool per_instance = false;
    std::vector<GxpAttribute> attributes;
    /// Bit n: primary attribute register n is written by the vertex fetch.
    uint64_t attrib_pa_regs = 0;
    /// The varyings block's output description: (output register count << 24) | 0x1000 |
    /// (0x800 with COLOR0), and three bits per TEXCOORDn (1: one or two floats, 3: three,
    /// 7: four), from the compiled programs with the same interface.
    uint32_t vertex_outputs1 = 0;
    uint32_t vertex_outputs2 = 0;
    bool buffer_pointer = false;
    int32_t ldst_base_value = 0;
    /// The default uniform buffer's size in floats when it extends past the resident part.
    uint32_t buffer_floats = 0;
    /// Secondary attributes the secondary program writes, past the containers.
    uint32_t secondary_reg_count = 0;
};

/// The data container's first SA holds the buffer pointer in a vertex program that has one;
/// the literals follow.
inline uint32_t DataHead(const GxpProgram& p) {
    return p.buffer_pointer ? 1u : 0u;
}

[[nodiscard]] std::vector<uint8_t> WriteGxp(const GxpProgram& program);

} // namespace GxmRenderer::Usse
