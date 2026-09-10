// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <span>
#include <utility>

#include "common/quaternion.h"
#include "common/vector_math.h"
#include "video_core/pica/pica_core.h"

namespace SwRenderer {

/**
 * The lighting registers decoded once per draw.
 *
 * Every field here is a bitfield extract, a fixed-point conversion or a sampler-support test
 * that the per-pixel model used to redo on every call - the light index indirection, three f16
 * position components and up to seven LUT selectors per light, each with its own scale lookup
 * and support test. None of it depends on the pixel.
 */
struct SwLightingState {
    struct Lut {
        bool enabled = false;
        Pica::LightingRegs::LightingLutInput input{};
        bool abs = false;
        f32 scale = 1.0f;
        std::size_t sampler = 0;
    };

    struct Light {
        Common::Vec3f position;
        Common::Vec3f spot_dir; ///< already divided by 2047
        Common::Vec3f specular_0, specular_1, diffuse, ambient;
        f32 dist_atten_scale = 0.0f, dist_atten_bias = 0.0f;
        std::size_t dist_atten_lut = 0;
        std::size_t spot_lut = 0;
        bool directional = false;
        bool two_sided_diffuse = false;
        bool geometric_factor_0 = false, geometric_factor_1 = false;
        bool dist_atten_enabled = false;
        bool spot_atten_enabled = false;
        bool shadow_enabled = false;
    };

    bool enable_shadow = false, shadow_invert = false, shadow_alpha = false;
    bool shadow_primary = false, shadow_secondary = false;
    u32 shadow_selector = 0, bump_selector = 0;
    Pica::LightingRegs::LightingBumpMode bump_mode{};
    bool disable_bump_renorm = false;
    bool clamp_highlights = false;
    bool enable_primary_alpha = false, enable_secondary_alpha = false;
    bool config7 = false; ///< the CP LUT input is only defined for Config7
    Common::Vec3f global_ambient;
    u32 max_light_index = 0;
    Lut d0, d1, fr, rr, rg, rb, sp;
    std::array<Light, 8> lights{};
};

/// Decodes the lighting registers. Call once per draw, not per pixel.
SwLightingState DecodeLighting(const Pica::LightingRegs& lighting);

std::pair<Common::Vec4<u8>, Common::Vec4<u8>> ComputeFragmentsColors(
    const SwLightingState& state, const Pica::PicaCore::Lighting& lighting_state,
    const Common::Quaternion<f32>& normquat, const Common::Vec3f& view,
    std::span<const Common::Vec4<u8>, 4> texture_color);

} // namespace SwRenderer
