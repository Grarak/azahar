// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <string>
#include "common/common_types.h"
#include "video_core/shader/generator/shader_gen.h"

namespace Pica {
struct ShaderSetup;
}

namespace Pica::Shader::Generator::Cg {

/**
 * What a generated vertex program expects from the draw that binds it, beyond the PICA
 * configuration in PicaVSConfig.
 *
 * `default_regs` is the set of input registers that no vertex loader feeds: the PICA
 * pipeline fills those from the fixed default attribute values, and GXM has no constant
 * vertex attribute, so the program declares them as uniforms instead. That makes them part
 * of the program's identity: the same PICA shader with a different loader layout is a
 * different GXP.
 */
/// Bisect switches for the translation: programs using a construct are refused (empty
/// source), so the software shader takes their draws. Bits for SetVsRefuse.
constexpr u32 VsRefuseMova = 1;
constexpr u32 VsRefuseLoop = 2;
constexpr u32 VsRefuseBranch = 4;
void SetVsRefuse(u32 mask);

struct VSExtra {
    u16 default_regs = 0;
    /// The bool uniforms the program branches on, baked in: a branch on one is resolved at
    /// generation and the untaken side is never emitted. PICA programs use them as feature
    /// toggles (skinning, lighting paths), so the untaken side is often most of the
    /// program, and what the compiler never sees it never spends twenty seconds on. Only
    /// the bools the program reads take part (UsedBoolUniforms), so the variants a title
    /// reaches are the combinations it actually uses.
    u16 bools = 0;
    u16 bool_mask = 0; ///< which bits of `bools` are baked; zero leaves them uniforms
    /// Components the loader supplies per input register (1-4; 0 or 4 means all four).
    /// The PICA fills what the loader does not supply with (0, 0, 0, 1), and libgxm does
    /// not say what a typed attribute's missing components become, so the program takes
    /// exactly the supplied ones and pads them itself.
    std::array<u8, 16> reg_components{};
};

/// The bool uniforms the program's flow control reads, as a bit per uniform, scanning the
/// program's written extent. Over-approximate for code the program never reaches, which
/// only costs key bits.
[[nodiscard]] u16 UsedBoolUniforms(const ShaderSetup& setup);

/**
 * The GXM Cg counterpart of GLSL::GenerateVertexShader: sce_vp_psp2 source for a PICA
 * vertex program, decompiled from its bytecode, with the same output interface (by
 * semantic) as the fixed pass-through program the software-shaded path uses, so every
 * generated fragment program links against either.
 *
 * `input_regs` receives the input registers the program reads: the draw binds a vertex
 * attribute for each of those that a loader feeds, and only those, because the patcher
 * refuses an attribute the program does not name. Empty on a program the decompiler
 * cannot express; the draw then shades in software.
 */
std::string GenerateVertexShader(const ShaderSetup& setup, const PicaVSConfig& config,
                                 const VSExtra& extra, u32& input_regs);

/// The identity of a generated program: the PICA configuration and the extra, hashed.
[[nodiscard]] u64 VertexShaderKey(const PicaVSConfig& config, const VSExtra& extra);

/// Emitter-validation harness, like the fragment MaybeDump: writes vs_<hash>.cg into the
/// dump directory set through SetDumpDir, so the psp2cgc corpus check covers vertex programs.
void MaybeDumpVertex(const ShaderSetup& setup, const PicaVSConfig& config, u64 hash);

} // namespace Pica::Shader::Generator::Cg
