// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/pica/output_vertex.h"
#include "video_core/pica/regs_rasterizer.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace Pica {

void OutputVertexMap::Build(const RasterizerRegs& regs) {
    constexpr u32 num_slots = sizeof(OutputVertex) / sizeof(f24);
    num_attributes = regs.vs_output_total & 7;
    for (u32 attrib = 0; attrib < num_attributes; ++attrib) {
        const auto reg_map = regs.vs_output_attributes[attrib];
        auto& attr = attributes[attrib];
        const std::array<u32, 4> src{reg_map.map_x, reg_map.map_y, reg_map.map_z, reg_map.map_w};
        for (u32 comp = 0; comp < 4; ++comp) {
            attr.dst[comp] = src[comp] < num_slots ? static_cast<u8>(src[comp]) : 0xFF;
        }
        attr.contiguous4 = attr.dst[0] != 0xFF && attr.dst[1] == attr.dst[0] + 1 &&
                           attr.dst[2] == attr.dst[0] + 2 && attr.dst[3] == attr.dst[0] + 3;
    }
}

OutputVertex::OutputVertex(const OutputVertexMap& map, const AttributeBuffer& output) {
    // Attributes can be used without being set in GPUREG_SH_OUTMAP_Oi
    // Hardware tests have shown that they are initialized to 1 in this case.
    static constexpr std::array<f24, sizeof(OutputVertex) / sizeof(f24)> ones = [] {
        std::array<f24, sizeof(OutputVertex) / sizeof(f24)> init{};
        init.fill(f24::One());
        return init;
    }();
    std::memcpy(this, ones.data(), sizeof(OutputVertex));

    f24* const slots = reinterpret_cast<f24*>(this);
    for (u32 attrib = 0; attrib < map.num_attributes; ++attrib) {
        const auto& attr = map.attributes[attrib];
        if (attr.contiguous4) {
            std::memcpy(slots + attr.dst[0], &output[attrib], 4 * sizeof(f24));
        } else {
            for (u32 comp = 0; comp < 4; ++comp) {
                if (attr.dst[comp] != 0xFF) {
                    slots[attr.dst[comp]] = output[attrib][comp];
                }
            }
        }
    }

    // The hardware takes the absolute and saturates vertex colors, *before* doing interpolation
#ifdef __ARM_NEON
    // abs, then pick |c| only where |c| < 1 — the select keeps the scalar path's NaN result (1.0)
    // where vminq would propagate the NaN.
    float* const color_f32 = reinterpret_cast<float*>(color.AsArray());
    const float32x4_t abs_color = vabsq_f32(vld1q_f32(color_f32));
    const float32x4_t one = vdupq_n_f32(1.0f);
    vst1q_f32(color_f32, vbslq_f32(vcltq_f32(abs_color, one), abs_color, one));
#else
    for (u32 i = 0; i < 4; ++i) {
        const f32 c = std::fabs(color[i].ToFloat32());
        color[i] = f24::FromFloat32(c < 1.0f ? c : 1.0f);
    }
#endif
}

#define ASSERT_POS(var, pos)                                                                       \
    static_assert(offsetof(OutputVertex, var) == pos * sizeof(f24), "Semantic at wrong "           \
                                                                    "offset.")

ASSERT_POS(pos, RasterizerRegs::VSOutputAttributes::POSITION_X);
ASSERT_POS(quat, RasterizerRegs::VSOutputAttributes::QUATERNION_X);
ASSERT_POS(color, RasterizerRegs::VSOutputAttributes::COLOR_R);
ASSERT_POS(tc0, RasterizerRegs::VSOutputAttributes::TEXCOORD0_U);
ASSERT_POS(tc1, RasterizerRegs::VSOutputAttributes::TEXCOORD1_U);
ASSERT_POS(tc0_w, RasterizerRegs::VSOutputAttributes::TEXCOORD0_W);
ASSERT_POS(view, RasterizerRegs::VSOutputAttributes::VIEW_X);
ASSERT_POS(tc2, RasterizerRegs::VSOutputAttributes::TEXCOORD2_U);

#undef ASSERT_POS

} // namespace Pica
