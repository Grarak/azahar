// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include "video_core/renderer_gxm/usse/gxp_writer.h"

namespace GxmRenderer::Usse {

namespace {

// Header layout (SceGxmProgram). Offsets are of the field; offset-valued fields are
// relative to their own position.
constexpr uint32_t HeaderSize = 0x9c;
constexpr uint32_t FieldSize = 0x08;
constexpr uint32_t FieldBinaryGuid = 0x0c;
constexpr uint32_t FieldSourceGuid = 0x10;
constexpr uint32_t FieldProgramFlags = 0x14;
constexpr uint32_t FieldBufferFlags = 0x18;
constexpr uint32_t FieldTexunitFlags0 = 0x1c;
constexpr uint32_t FieldParameterCount = 0x24;
constexpr uint32_t FieldParametersOffset = 0x28;
constexpr uint32_t FieldVaryingsOffset = 0x2c;
constexpr uint32_t FieldPrimaryRegCount = 0x30;
constexpr uint32_t FieldSecondaryRegCount = 0x32;
constexpr uint32_t FieldTempRegCount1 = 0x34;
constexpr uint32_t FieldPhaseCount = 0x3a;
constexpr uint32_t FieldPrimaryInstrCount = 0x3c;
constexpr uint32_t FieldPrimaryOffset = 0x40;
constexpr uint32_t FieldUniformBufferCount = 0x78;
constexpr uint32_t FieldSecondaryInstrCount = 0x44;
constexpr uint32_t FieldSecondaryOffset = 0x48;
constexpr uint32_t FieldSecondaryOffsetEnd = 0x4c;
constexpr uint32_t FieldDataBufferCount = 0x5c;
constexpr uint32_t FieldDefaultUniformBufferCount = 0x64;
constexpr uint32_t FieldLiteralBufferDataOffset = 0x68;
constexpr uint32_t FieldCompilerVersion = 0x6c;
constexpr uint32_t FieldLiteralsCount = 0x70;
constexpr uint32_t FieldLiteralsOffset = 0x74;
constexpr uint32_t FieldUniformBufferOffset = 0x7c;
constexpr uint32_t FieldDependentSamplerCount = 0x80;
constexpr uint32_t FieldDependentSamplerOffset = 0x84;
constexpr uint32_t FieldTextureBufferDependentSamplerOffset = 0x8c;
constexpr uint32_t FieldContainerCount = 0x90;
constexpr uint32_t FieldContainerOffset = 0x94;
constexpr uint32_t FieldSamplerQueryInfoOffset = 0x98;

// What psp2cgc 3.50 writes: version 1.5, SDK 0x350, its own version word.
constexpr uint8_t MajorVersion = 1;
constexpr uint8_t MinorVersion = 5;
constexpr uint16_t SdkVersion = 0x350;
constexpr uint32_t CompilerVersion = 0x33dc0;

// program_flags: bit 0 fragment; 0x1000 with interpolated inputs, and 0x4 with them too,
// discard or not (psp2cgc sets both on all 148 of SM3DL's programs, alpha test and all, and
// on the ubershader; the emitter's earlier reading that discard drops 0x4 came from a
// reference without inputs); 0x180000 always set by this compiler version; 0x8 discard
// used; 0x10 depth written. 0x2 the compiler sets on some lit programs and on the
// ubershader, meaning unknown, not set here.
constexpr uint32_t FlagFragment = 1u << 0;
constexpr uint32_t FlagIterators = 0x1000;
constexpr uint32_t FlagIteratorsPresent = 0x4;
constexpr uint32_t FlagAlways = 0x180000;
constexpr uint32_t FlagDiscardUsed = 1u << 3;
constexpr uint32_t FlagDepthUsed = 1u << 4;
// Vertex programs: 0x190000 always (the compiler's, all 44 SM3DL programs and the probes),
// 0x2 per-instance mode.
constexpr uint32_t FlagVertexAlways = 0x190000;
constexpr uint32_t FlagPerInstance = 1u << 1;
// buffer_flags: two bits per buffer, 1 = loaded into registers; buffer 14 is the default
// uniform buffer.
constexpr uint32_t BufferFlagsDefaultInRegisters = 1u << 28;
// Bit 29: the default uniform buffer is also read from memory (through the pointer).
constexpr uint32_t BufferFlagsDefaultInMemory = 1u << 29;
// texunit_flags: four bits per unit, 2 = dependent read.
constexpr uint32_t TexunitDependent = 2;

constexpr uint16_t ContainerDefaultUniforms = 14;
constexpr uint16_t ContainerData = 19; ///< literals, then four SAs of control words per sampler
constexpr uint32_t SamplerWords = 4;

// Parameter categories and types (SceGxmParameterCategory / SceGxmParameterType).
constexpr uint8_t CategoryAttribute = 0;
constexpr uint8_t CategoryUniform = 1;
constexpr uint8_t CategorySampler = 2;
constexpr uint8_t TypeF32 = 0;
constexpr uint8_t TypeF16 = 1;
// Sampler parameters carry semantic 1 in the reference programs (2 when the read is
// non-dependent); uniforms carry 0.
constexpr uint8_t SemanticSampler = 1;

class Writer {
public:
    std::vector<uint8_t> out;

    uint32_t Size() const {
        return static_cast<uint32_t>(out.size());
    }
    void Pad(uint32_t n) {
        out.insert(out.end(), n, 0);
    }
    void AlignTo(uint32_t a) {
        while (out.size() % a != 0) {
            out.push_back(0);
        }
    }
    void U8(uint8_t v) {
        out.push_back(v);
    }
    void U16(uint16_t v) {
        out.push_back(static_cast<uint8_t>(v));
        out.push_back(static_cast<uint8_t>(v >> 8));
    }
    void U32(uint32_t v) {
        for (int i = 0; i < 4; i++) {
            out.push_back(static_cast<uint8_t>(v >> (i * 8)));
        }
    }
    void U64(uint64_t v) {
        for (int i = 0; i < 8; i++) {
            out.push_back(static_cast<uint8_t>(v >> (i * 8)));
        }
    }
    void PatchU16(uint32_t at, uint16_t v) {
        out[at] = static_cast<uint8_t>(v);
        out[at + 1] = static_cast<uint8_t>(v >> 8);
    }
    void PatchU32(uint32_t at, uint32_t v) {
        for (int i = 0; i < 4; i++) {
            out[at + i] = static_cast<uint8_t>(v >> (i * 8));
        }
    }
    /// An offset field: the distance from the field to `target`.
    void PatchOffset(uint32_t field, uint32_t target) {
        PatchU32(field, target - field);
    }
};

} // Anonymous namespace

std::vector<uint8_t> WriteGxp(const GxpProgram& program) {
    Writer w;
    w.out.reserve(1024);

    // --- header, offsets patched as the sections land ---
    w.out.insert(w.out.end(), {'G', 'X', 'P', '\0'});
    w.U8(MajorVersion);
    w.U8(MinorVersion);
    w.U16(SdkVersion);
    w.Pad(HeaderSize - w.Size());

    uint32_t flags = FlagFragment | FlagAlways;
    if (program.vertex) {
        flags = FlagVertexAlways | (program.per_instance ? FlagPerInstance : 0);
    } else {
        if (!program.inputs.empty()) {
            flags |= FlagIterators | FlagIteratorsPresent;
        }
        if (program.uses_discard) {
            flags |= FlagDiscardUsed;
        }
        if (program.writes_depth) {
            flags |= FlagDepthUsed;
        }
    }
    w.PatchU32(FieldBinaryGuid, program.binary_guid);
    w.PatchU32(FieldSourceGuid, program.source_guid);
    w.PatchU32(FieldProgramFlags, flags);
    const bool has_uniforms = program.uniform_floats != 0;
    w.PatchU32(FieldBufferFlags, (has_uniforms ? BufferFlagsDefaultInRegisters : 0) |
                                     (program.buffer_pointer ? BufferFlagsDefaultInMemory : 0));
    uint32_t texunit_flags = 0;
    for (const auto& sampler : program.samplers) {
        texunit_flags |= TexunitDependent << (sampler.unit * 4);
    }
    w.PatchU32(FieldTexunitFlags0, texunit_flags);
    w.PatchU16(FieldPrimaryRegCount, static_cast<uint16_t>(program.pa_count));
    // Secondary attributes: the uniforms, then the data container behind them.
    const uint32_t data_base = program.uniform_floats;
    const uint32_t data_head = DataHead(program);
    const uint32_t literal_count = static_cast<uint32_t>(program.literals.size());
    const uint32_t sampler_count = static_cast<uint32_t>(program.samplers.size());
    const uint32_t data_size = data_head + literal_count + sampler_count * SamplerWords;
    w.PatchU16(FieldSecondaryRegCount, static_cast<uint16_t>(std::max(
                                           data_base + data_size, program.secondary_reg_count)));
    w.PatchU32(FieldDataBufferCount, data_size);
    w.PatchU32(FieldLiteralsCount, literal_count);
    w.PatchU32(FieldTempRegCount1, program.temp_count);
    w.PatchU16(FieldPhaseCount, static_cast<uint16_t>(program.phase_starts.size()));
    w.PatchU32(FieldPrimaryInstrCount, static_cast<uint32_t>(program.primary.size()));
    w.PatchU32(FieldSecondaryInstrCount, static_cast<uint32_t>(program.secondary.size()));
    w.PatchU32(FieldDefaultUniformBufferCount,
               std::max(program.uniform_floats, program.buffer_floats));
    w.PatchU32(FieldCompilerVersion, CompilerVersion);

    // --- varyings: 8 unknown zero bytes, output description, then the descriptors ---
    w.PatchOffset(FieldVaryingsOffset, w.Size());
    if (program.vertex) {
        w.U64(program.attrib_pa_regs);
        w.U64(0); // untyped_pa_regs
        w.U32(program.vertex_outputs1);
        w.U32(program.vertex_outputs2);
        w.U32(0); // texcoord_pack_format: all float
        w.U16(0); // semantic_index_offset
        w.U16(0); // semantic_instance_offset
    } else {
        w.Pad(8);
        w.U8(0);       // fragment_output_start
        w.U8(0);       // unk1
        w.U8(TypeF16); // the colour leaves the program as half4 in pa0
        w.U8(4);
        w.U16(static_cast<uint16_t>(program.inputs.size()));
        w.U16(0); // non-dependent sampler count
        w.U32(4); // the descriptors follow this field directly
        for (size_t i = 0; i < program.inputs.size(); i++) {
            const auto& input = program.inputs[i];
            // 0x0c00000f: constant bits and "no texture query"; the input id in bits 12-15; the
            // component count minus one in bits 22-23; COLOR0 marked float (the default type
            // of a colour iterator is otherwise fixed-point); bit 25 on the last descriptor.
            uint32_t info = 0x0c00000f | (static_cast<uint32_t>(input.input) << 12) |
                            (static_cast<uint32_t>(input.components - 1) << 22);
            if (input.input == FragmentInput::Color0 || input.input == FragmentInput::Color1) {
                info |= 0x00100000;
            }
            if (i + 1 == program.inputs.size()) {
                info |= 0x02000000;
            }
            w.U32(info);
            w.U32(0);                                              // resource_index
            w.U32(static_cast<uint32_t>(input.components - 1) << 4); // registers minus one
            w.U32(0);                                              // component_info
        }
    }

    // --- secondary program, phase table, primary program ---
    const uint32_t secondary_at = w.Size();
    w.PatchOffset(FieldSecondaryOffset, secondary_at);
    for (const uint64_t word : program.secondary) {
        w.U64(word);
    }
    w.PatchOffset(FieldSecondaryOffsetEnd, w.Size());
    for (const uint32_t start : program.phase_starts) {
        w.U32(start);
    }
    w.PatchOffset(FieldPrimaryOffset, w.Size());
    for (const uint64_t word : program.primary) {
        w.U64(word);
    }

    // --- literals: (SA offset within the data container, raw value) ---
    const uint32_t after_code = w.Size();
    w.PatchOffset(FieldLiteralsOffset, after_code);
    for (uint32_t i = 0; i < literal_count; i++) {
        w.U32(LiteralSaOffset(data_head, i));
        w.U32(program.literals[i]);
    }

    // --- dependent samplers: four control words per sampler, each entry naming the unit's
    // word (unit * 4 + k) and the word's SA within the data container ---
    const uint32_t samplers_at = w.Size();
    w.PatchOffset(FieldDependentSamplerOffset, samplers_at);
    w.PatchOffset(FieldLiteralBufferDataOffset, samplers_at);
    w.PatchU32(FieldDependentSamplerCount, sampler_count * SamplerWords);
    for (uint32_t j = 0; j < sampler_count; j++) {
        for (uint32_t k = 0; k < SamplerWords; k++) {
            w.U16(static_cast<uint16_t>(program.samplers[j].unit * SamplerWords + k));
            w.U16(static_cast<uint16_t>(SamplerSaOffset(data_head, literal_count, j) + k));
        }
    }

    // --- containers ---
    const uint32_t containers_at = w.Size();
    uint32_t container_count = 0;
    if (has_uniforms) {
        w.U16(ContainerDefaultUniforms);
        w.U16(0);
        w.U16(0);
        w.U16(static_cast<uint16_t>(program.uniform_floats));
        container_count++;
    }
    if (data_size != 0) {
        w.U16(ContainerData);
        w.U16(0);
        w.U16(static_cast<uint16_t>(data_base));
        w.U16(static_cast<uint16_t>(data_size));
        container_count++;
    }
    w.PatchU32(FieldContainerCount, container_count);
    w.PatchOffset(FieldContainerOffset, containers_at);
    w.PatchOffset(FieldTextureBufferDependentSamplerOffset, containers_at);

    // --- uniform buffer info: where libgxm plants the buffer pointer (SceGxmUniformBufferInfo:
    // reside_buffer, ldst_base_offset, ldst_base_value), right after the containers ---
    uint32_t buffer_info_at = 0;
    if (program.buffer_pointer) {
        buffer_info_at = w.Size();
        w.U16(ContainerDefaultUniforms);
        w.U16(0);
        w.U32(static_cast<uint32_t>(program.ldst_base_value));
        w.PatchU32(FieldUniformBufferCount, 1);
    }

    // --- parameters, then their names ---
    const uint32_t params_at = w.Size();
    const uint32_t param_count = static_cast<uint32_t>(
        program.attributes.size() + program.uniforms.size() + program.samplers.size());
    w.PatchU32(FieldParameterCount, param_count);
    w.PatchOffset(FieldParametersOffset, params_at);
    std::vector<uint32_t> name_fields;
    for (const auto& a : program.attributes) {
        name_fields.push_back(w.Size());
        w.U32(0);
        w.U16(static_cast<uint16_t>(CategoryAttribute | (TypeF32 << 4) | (4 << 8)));
        w.U8(0);
        w.U8(0);
        w.U32(1);
        w.U32(a.resource);
    }
    for (const auto& u : program.uniforms) {
        name_fields.push_back(w.Size());
        w.U32(0); // name offset, patched below
        w.U16(static_cast<uint16_t>(CategoryUniform | (TypeF32 << 4) | (u.components << 8) |
                                    (ContainerDefaultUniforms << 12)));
        w.U8(0); // semantic
        w.U8(0); // semantic index
        w.U32(u.array_size);
        w.U32(u.sa_offset);
    }
    for (const auto& s : program.samplers) {
        name_fields.push_back(w.Size());
        w.U32(0);
        w.U16(static_cast<uint16_t>(CategorySampler | (TypeF32 << 4) | (4 << 8)));
        w.U8(SemanticSampler);
        w.U8(0);
        w.U32(1);
        w.U32(s.unit);
    }
    // --- sampler query info: one u16 per texture unit, 0x0301 for a unit the program
    // reads with a dependent 2D read; 32 bytes, present only when there are samplers ---
    if (sampler_count != 0) {
        w.PatchOffset(FieldSamplerQueryInfoOffset, w.Size());
        uint16_t units[16]{};
        for (const auto& s : program.samplers) {
            units[s.unit & 15] = 0x0301;
        }
        for (const uint16_t u : units) {
            w.U16(u);
        }
    }

    size_t name_index = 0;
    const auto put_name = [&](const std::string& name) {
        const uint32_t field = name_fields[name_index++];
        w.PatchU32(field, w.Size() - field);
        w.out.insert(w.out.end(), name.begin(), name.end());
        w.U8(0);
    };
    for (const auto& a : program.attributes) {
        put_name(a.name);
    }
    for (const auto& u : program.uniforms) {
        put_name(u.name);
    }
    for (const auto& s : program.samplers) {
        put_name(s.name);
    }

    // Without a buffer info table the offset points where the compiler points it.
    w.PatchOffset(FieldUniformBufferOffset, program.buffer_pointer ? buffer_info_at : params_at);

    w.PatchU32(FieldSize, w.Size());
    w.AlignTo(4);
    return w.out;
}

} // namespace GxmRenderer::Usse
