// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_gxm/usse/fixed_usse_gen.h"
#include "video_core/renderer_gxm/usse/gxp_writer.h"
#include "video_core/renderer_gxm/usse/usse_encoder.h"

namespace GxmRenderer::Usse {

namespace {

const Src ConstOne{Reg::C(2), XXXX};
const Src ConstZeroOne{Reg::C(1), XYZW};
const Src ConstHalf{Reg::C(12), XXXX};
const Reg I0 = Reg::I(0);
const Reg I1 = Reg::I(1);

// The fragment interface's output registers, as vs_usse_gen lays them out.
constexpr uint8_t OutPosition = 0;
constexpr uint8_t OutColor = 4;
constexpr uint8_t OutTexcoord0 = 8;
constexpr uint8_t OutTexcoord1 = 10;
constexpr uint8_t OutTexcoord2 = 12;
constexpr uint8_t OutTexcoord0W = 14;
constexpr uint8_t OutNormquat = 16;
constexpr uint8_t OutNormquatFlat = 20;
constexpr uint8_t OutView = 24;

} // Anonymous namespace

std::vector<uint8_t> EmitPassThroughVertexProgram() {
    // Attributes at four primary attributes each: position pa0, colour pa4, texcoord0 pa8,
    // texcoord1 pa12, texcoord2 pa16, texcoord0_w pa20, normquat pa24, view pa28.
    GxpProgram gxp;
    gxp.vertex = true;
    const char* names[] = {"vert_position",   "vert_color",    "vert_texcoord0", "vert_texcoord1",
                           "vert_texcoord2",  "vert_texcoord0_w", "vert_normquat", "vert_view"};
    const uint8_t widths[] = {4, 4, 2, 2, 2, 1, 4, 3};
    for (uint32_t i = 0; i < 8; i++) {
        gxp.attributes.push_back({names[i], i * 4});
        gxp.attrib_pa_regs |= ((1ull << widths[i]) - 1) << (i * 4);
    }
    gxp.pa_count = 32;
    gxp.uniforms.push_back({"flip_viewport", 0, 1, 1});
    gxp.uniforms.push_back({"depth_scale", 1, 1, 1});
    gxp.uniforms.push_back({"depth_offset", 2, 1, 1});
    gxp.uniform_floats = 4;
    gxp.vertex_outputs1 = 0x1b001800;
    gxp.vertex_outputs2 = 0xff249;

    Encoder e;
    e.skip_invalid = true;
    e.Phas(true);
    e.Nop();
    e.MoveInternal({Reg::O(OutColor), 3}, {Reg::Pa(4), XYZW});
    e.MoveInternal({Reg::O(OutColor + 2), 3}, {Reg::Pa(6), XYZW});
    e.MoveInternal({Reg::O(OutTexcoord0), 3}, {Reg::Pa(8), XYZW});
    e.MoveInternal({Reg::O(OutTexcoord1), 3}, {Reg::Pa(12), XYZW});
    e.MoveInternal({Reg::O(OutTexcoord2), 3}, {Reg::Pa(16), XYZW});
    e.MoveInternal({Reg::O(OutTexcoord0W), 1}, {Reg::Pa(20), XYZW});
    e.MoveInternal({Reg::O(OutNormquat), 3}, {Reg::Pa(24), XYZW});
    e.MoveInternal({Reg::O(OutNormquat + 2), 3}, {Reg::Pa(26), XYZW});
    e.MoveInternal({Reg::O(OutNormquatFlat), 3}, {Reg::Pa(24), XYZW});
    e.MoveInternal({Reg::O(OutNormquatFlat + 2), 3}, {Reg::Pa(26), XYZW});
    e.MoveInternal({Reg::O(OutView), 3}, {Reg::Pa(28), XYZW});
    e.MoveInternal({Reg::O(OutView + 2), 1}, {Reg::Pa(30), XYZW});
    // Position: y flipped when flip_viewport > 0.5, z = (z / w * depth_scale +
    // depth_offset) * w.
    e.Vec(VecOp::Mul, {I0, 0xf}, {Reg::Pa(0), XYZW}, ConstOne);
    e.Test(Cond::Gt, 2, {Reg::Sa(0)}, 0, ConstHalf, false);
    e.Vec(VecOp::Mul, {I0, 0x2}, {I0, YYYY, Mod::Neg}, ConstOne, Pred::P2);
    e.Comp(CompOp::Rcp, {I1, 0x1}, {I0}, 3);
    e.Vec(VecOp::Mul, {I1, 0x2}, {I0, ZZZZ}, {I1, XXXX});
    e.Vec(VecOp::Mul, {I1, 0x2}, {I1, YYYY}, {Reg::Sa(0), YYYY});
    e.Vec(VecOp::Add, {I1, 0x2}, {I1, YYYY}, {Reg::Sa(2), XXXX});
    e.Vec(VecOp::Mul, {I0, 0x4}, {I1, YYYY}, {I0, WWWW});
    e.MoveInternal({Reg::O(OutPosition), 3}, {I0, XYZW});
    e.MoveInternal({Reg::O(OutPosition + 2), 3}, {I0, ZWZW});
    e.EndVertex();
    e.FinishNosched();
    gxp.primary = e.Words();
    return WriteGxp(gxp);
}

std::vector<uint8_t> EmitBlitVertexProgram() {
    GxpProgram gxp;
    gxp.vertex = true;
    gxp.attributes.push_back({"vert_position", 0});
    gxp.attributes.push_back({"vert_texcoord", 4});
    gxp.attrib_pa_regs = 0x33;
    gxp.pa_count = 8;
    gxp.vertex_outputs1 = 0x6001000; // six output registers: position, texcoord0
    gxp.vertex_outputs2 = 0x1;

    Encoder e;
    e.skip_invalid = true;
    e.Phas(true);
    e.MoveInternal({Reg::O(OutPosition), 3}, {Reg::Pa(0), XYZW});
    e.MoveInternal({Reg::O(OutPosition + 2), 3}, ConstZeroOne);
    e.MoveInternal({Reg::O(OutColor), 3}, {Reg::Pa(4), XYZW}); // o4: TEXCOORD0 here
    e.EndVertex();
    e.FinishNosched();
    gxp.primary = e.Words();
    return WriteGxp(gxp);
}

std::vector<uint8_t> EmitBlitFragmentProgram() {
    GxpProgram gxp;
    gxp.inputs.push_back({FragmentInput::TexCoord0, 2});
    gxp.samplers.push_back({"blit_source", 0});
    gxp.pa_count = 4;
    gxp.temp_count = 4;

    Encoder e;
    e.skip_invalid = false;
    e.Phas(true);
    e.Nop();
    e.Sample2D(Reg::R(0), Reg::Pa(0), static_cast<uint8_t>(SamplerSaOffset(0, 0, 0)), nullptr, 0);
    e.Wdf(0);
    e.PackF16({Reg::Pa(0), 0xf}, Reg::R(0), Reg::R(2));
    e.FinishNosched();
    gxp.primary = e.Words();
    return WriteGxp(gxp);
}

std::vector<uint8_t> EmitFillFragmentProgram() {
    GxpProgram gxp;
    gxp.uniforms.push_back({"fill_colour", 0, 4, 1});
    gxp.uniform_floats = 4;
    gxp.pa_count = 4;

    Encoder e;
    e.skip_invalid = false;
    e.Phas(true);
    e.Nop();
    e.PackF16({Reg::Pa(0), 0xf}, Reg::Sa(0), Reg::Sa(2));
    e.FinishNosched();
    gxp.primary = e.Words();
    return WriteGxp(gxp);
}

} // namespace GxmRenderer::Usse
