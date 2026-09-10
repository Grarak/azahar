// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "video_core/shader/generator/pica_fs_config.h"

namespace GxmRenderer::Usse {

/**
 * The PICA fragment emitter: a GXP program for a fragment configuration, written directly
 * as USSE instructions. Mirrors cg_fs_shader_gen.cpp
 * pass for pass and names its uniforms the same, so the rasterizer's uniform upload and
 * texture binding are unchanged; what the compiler took seconds for takes microseconds.
 *
 * Returns the container, or empty when the configuration is outside what is implemented
 * so far (the caller compiles the Cg instead): lighting, fog, procedural textures,
 * scissor, projection/cube/shadow textures, border colours, shadow rendering, W-buffer
 * depth, min/max blend emulation, bitwise logic ops.
 */
/// `refusal`, when given, receives why an empty result came back. `level` 2 runs the IR's
/// optimising passes (the second tier).
[[nodiscard]] std::vector<uint8_t> EmitFragmentProgram(const Pica::Shader::FSConfig& config,
                                                       std::string* refusal = nullptr,
                                                       int level = 1);

} // namespace GxmRenderer::Usse
