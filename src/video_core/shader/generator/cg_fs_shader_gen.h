// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/shader/generator/pica_fs_config.h"

namespace Pica::Shader::Generator::Cg {

/**
 * The GXM Cg counterpart of GLSL::GenerateFragmentShader: emits sce_fp_psp2 Cg source for a
 * PICA fragment configuration. Structure follows the GLSL generator statement for statement
 * so the two stay comparable; the differences are the language surface (float4/lerp/frac/
 * tex2D, semantics instead of varyings, flat uniforms instead of a uniform block) and the
 * GXM profile's hard limits (no storage images, so shadow rendering and the shadow texture
 * types emit marked stubs).
 *
 * The emitted source targets the runtime compiler (SceShaccCg) on console, and psp2cgc on
 * the pi5 harness for validation - every emitted file must compile there.
 */
std::string GenerateFragmentShader(const FSConfig& config, const UserConfig& user,
                                   const Profile& profile);

/// Bring-up: replace every fragment program's final colour with one intermediate, so a
/// frame dump per mode bisects the fragment pipeline on hardware. 0 is off.
enum class DebugOutput : u32 { None, VertexColor, Texture0, Lighting, Normal, View, Texcoord0, Texcoord1, Texcoord2, Alpha, TevStage0 };
void SetDebugOutput(DebugOutput mode);
/// Leave the fog stage out of every program, to tell a wrong fog from a wrong colour.
void SetFogDisabled(bool disabled);
[[nodiscard]] bool FogDisabled();
/// TevStage0 plus n stops after TEV stage n, so fs_tev0..fs_tev5 walk the combiner.
[[nodiscard]] DebugOutput GetDebugOutput();

/// The Profile a GXM target implies: no texture buffers, no texelFetch, no extensions,
/// 0-to-1 depth range, no logic ops.
Profile MakeGxmProfile();

/// Emitter-validation harness: when a dump directory is set, MaybeDump writes the Cg source
/// for a fragment config to <dir>/fs_<hash>.cg. The GL frontend calls it for every new
/// config it compiles, so a play session produces the corpus psp2cgc then has to accept.
void SetDumpDir(std::string dir);
/// The directory SetDumpDir set, empty when dumping is off. Shared with the vertex emitter.
const std::string& DumpDir();
void MaybeDump(const FSConfig& config, const UserConfig& user, u64 hash);

} // namespace Pica::Shader::Generator::Cg
