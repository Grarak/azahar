// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdint>
#include <vector>

namespace GxmRenderer::Usse {

/**
 * The rasterizer's fixed programs, written as USSE directly (they used to be Cg compiled
 * by SceShaccCg on the console at first boot; nothing compiles on the console any more):
 *
 *  - the pass-through vertex program for software-shaded draws: the PICA output vertex's
 *    attributes to the fragment interface, with the depth mapping on clip z
 *    (vert_position, vert_color, vert_texcoord0/1/2, vert_texcoord0_w, vert_normquat,
 *    vert_view; uniforms flip_viewport, depth_scale, depth_offset);
 *  - the blit quad's vertex program (vert_position float2, vert_texcoord float2 to
 *    POSITION and TEXCOORD0);
 *  - the blit's fragment program (one fetch of blit_source at TEXUNIT0);
 *  - the clear quad's fragment program (uniform float4 fill_colour).
 *
 * Each is the compiler's own program for the same Cg, instruction for instruction where
 * a compiled twin existed (the pass-through and blit vertex programs), so the words are
 * known to run.
 */
[[nodiscard]] std::vector<uint8_t> EmitPassThroughVertexProgram();
[[nodiscard]] std::vector<uint8_t> EmitBlitVertexProgram();
[[nodiscard]] std::vector<uint8_t> EmitBlitFragmentProgram();
[[nodiscard]] std::vector<uint8_t> EmitFillFragmentProgram();

} // namespace GxmRenderer::Usse
