// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/math_util.h"
#include <algorithm>
#include "video_core/renderer_software/sw_lighting.h"

namespace SwRenderer {

using Pica::f16;
using Pica::LightingRegs;

static float LookupLightingLut(const Pica::PicaCore::Lighting& lighting, std::size_t lut_index,
                               u8 index, float delta) {
    ASSERT_MSG(lut_index < lighting.luts.size(), "Out of range lut");
    ASSERT_MSG(index < lighting.luts[lut_index].size(), "Out of range index");

    const auto& lut = lighting.luts[lut_index][index];

    const float lut_value = lut.ToFloat();
    const float lut_diff = lut.DiffToFloat();

    return lut_value + lut_diff * delta;
}

SwLightingState DecodeLighting(const Pica::LightingRegs& lighting) {
    SwLightingState st;
    st.enable_shadow = lighting.config0.enable_shadow != 0;
    st.shadow_invert = lighting.config0.shadow_invert != 0;
    st.shadow_alpha = lighting.config0.shadow_alpha != 0;
    st.shadow_primary = lighting.config0.shadow_primary != 0;
    st.shadow_secondary = lighting.config0.shadow_secondary != 0;
    st.shadow_selector = lighting.config0.shadow_selector;
    st.bump_selector = lighting.config0.bump_selector;
    st.bump_mode = lighting.config0.bump_mode;
    st.disable_bump_renorm = lighting.config0.disable_bump_renorm != 0;
    st.clamp_highlights = lighting.config0.clamp_highlights != 0;
    st.enable_primary_alpha = lighting.config0.enable_primary_alpha != 0;
    st.enable_secondary_alpha = lighting.config0.enable_secondary_alpha != 0;
    st.config7 = lighting.config0.config == LightingRegs::LightingConfig::Config7;
    st.global_ambient = lighting.global_ambient.ToVec3f();
    st.max_light_index = lighting.max_light_index;

    // A LUT is live when its disable bit is clear and the lighting config supports its sampler;
    // both are draw state, and the scale and input selector with them.
    const auto lut = [&](bool disabled, LightingRegs::LightingSampler sampler,
                         LightingRegs::LightingLutInput input, bool abs,
                         LightingRegs::LightingScale scale) {
        SwLightingState::Lut l;
        l.enabled = !disabled &&
                    LightingRegs::IsLightingSamplerSupported(lighting.config0.config, sampler);
        l.input = input;
        l.abs = abs;
        l.scale = std::remove_cvref_t<decltype(lighting.lut_scale)>::GetScale(scale);
        l.sampler = static_cast<std::size_t>(sampler);
        return l;
    };
    st.d0 = lut(lighting.config1.disable_lut_d0 != 0, LightingRegs::LightingSampler::Distribution0,
                lighting.lut_input.d0, lighting.abs_lut_input.disable_d0 == 0,
                lighting.lut_scale.d0);
    st.d1 = lut(lighting.config1.disable_lut_d1 != 0, LightingRegs::LightingSampler::Distribution1,
                lighting.lut_input.d1, lighting.abs_lut_input.disable_d1 == 0,
                lighting.lut_scale.d1);
    st.fr = lut(lighting.config1.disable_lut_fr != 0, LightingRegs::LightingSampler::Fresnel,
                lighting.lut_input.fr, lighting.abs_lut_input.disable_fr == 0,
                lighting.lut_scale.fr);
    st.rr = lut(lighting.config1.disable_lut_rr != 0, LightingRegs::LightingSampler::ReflectRed,
                lighting.lut_input.rr, lighting.abs_lut_input.disable_rr == 0,
                lighting.lut_scale.rr);
    st.rg = lut(lighting.config1.disable_lut_rg != 0, LightingRegs::LightingSampler::ReflectGreen,
                lighting.lut_input.rg, lighting.abs_lut_input.disable_rg == 0,
                lighting.lut_scale.rg);
    st.rb = lut(lighting.config1.disable_lut_rb != 0, LightingRegs::LightingSampler::ReflectBlue,
                lighting.lut_input.rb, lighting.abs_lut_input.disable_rb == 0,
                lighting.lut_scale.rb);
    // The spotlight sampler is per light; the rest of the selector is not.
    st.sp = lut(false, LightingRegs::LightingSampler::SpotlightAttenuation, lighting.lut_input.sp,
                lighting.abs_lut_input.disable_sp == 0, lighting.lut_scale.sp);
    st.sp.enabled = LightingRegs::IsLightingSamplerSupported(
        lighting.config0.config, LightingRegs::LightingSampler::SpotlightAttenuation);

    for (u32 light_index = 0; light_index <= lighting.max_light_index; ++light_index) {
        const u32 num = lighting.light_enable.GetNum(light_index);
        const auto& cfg = lighting.light[num];
        auto& l = st.lights[light_index];
        l.position = {f16::FromRaw(cfg.x).ToFloat32(), f16::FromRaw(cfg.y).ToFloat32(),
                      f16::FromRaw(cfg.z).ToFloat32()};
        l.spot_dir = Common::Vec3<s32>{cfg.spot_x.Value(), cfg.spot_y.Value(), cfg.spot_z.Value()}
                         .Cast<float>() /
                     2047.0f;
        l.specular_0 = cfg.specular_0.ToVec3f();
        l.specular_1 = cfg.specular_1.ToVec3f();
        l.diffuse = cfg.diffuse.ToVec3f();
        l.ambient = cfg.ambient.ToVec3f();
        l.directional = cfg.config.directional != 0;
        l.two_sided_diffuse = cfg.config.two_sided_diffuse != 0;
        l.geometric_factor_0 = cfg.config.geometric_factor_0 != 0;
        l.geometric_factor_1 = cfg.config.geometric_factor_1 != 0;
        l.dist_atten_enabled = !lighting.IsDistAttenDisabled(num);
        l.dist_atten_scale = Pica::f20::FromRaw(cfg.dist_atten_scale).ToFloat32();
        l.dist_atten_bias = Pica::f20::FromRaw(cfg.dist_atten_bias).ToFloat32();
        l.dist_atten_lut =
            static_cast<std::size_t>(LightingRegs::LightingSampler::DistanceAttenuation) + num;
        l.spot_atten_enabled = !lighting.IsSpotAttenDisabled(num);
        l.spot_lut =
            static_cast<std::size_t>(LightingRegs::SpotlightAttenuationSampler(num));
        l.shadow_enabled = !lighting.IsShadowDisabled(num);
    }
    return st;
}

std::pair<Common::Vec4<u8>, Common::Vec4<u8>> ComputeFragmentsColors(
    const SwLightingState& st, const Pica::PicaCore::Lighting& lighting_state,
    const Common::Quaternion<f32>& normquat, const Common::Vec3f& view,
    std::span<const Common::Vec4<u8>, 4> texture_color) {

    Common::Vec4f shadow;
    if (st.enable_shadow) {
        shadow = texture_color[st.shadow_selector].Cast<float>() / 255.0f;
        if (st.shadow_invert) {
            shadow = Common::MakeVec(1.0f, 1.0f, 1.0f, 1.0f) - shadow;
        }
    } else {
        shadow = Common::MakeVec(1.0f, 1.0f, 1.0f, 1.0f);
    }

    Common::Vec3f surface_normal{};
    Common::Vec3f surface_tangent{};

    if (st.bump_mode != LightingRegs::LightingBumpMode::None) {
        Common::Vec3f perturbation =
            texture_color[st.bump_selector].xyz().Cast<float>() / 127.5f -
            Common::MakeVec(1.0f, 1.0f, 1.0f);
        if (st.bump_mode == LightingRegs::LightingBumpMode::NormalMap) {
            if (!st.disable_bump_renorm) {
                const f32 z_square = 1 - perturbation.xy().Length2();
                perturbation.z = std::sqrt(std::max(z_square, 0.0f));
            }
            surface_normal = perturbation;
            surface_tangent = Common::MakeVec(1.0f, 0.0f, 0.0f);
        } else if (st.bump_mode == LightingRegs::LightingBumpMode::TangentMap) {
            surface_normal = Common::MakeVec(0.0f, 0.0f, 1.0f);
            surface_tangent = perturbation;
        } else {
            LOG_ERROR(HW_GPU, "Unknown bump mode {}",
                      static_cast<u32>(st.bump_mode));
        }
    } else {
        surface_normal = Common::MakeVec(0.0f, 0.0f, 1.0f);
        surface_tangent = Common::MakeVec(1.0f, 0.0f, 0.0f);
    }

    // Use the normalized the quaternion when performing the rotation
    auto normal = Common::QuaternionRotate(normquat, surface_normal);
    auto tangent = Common::QuaternionRotate(normquat, surface_tangent);

    Common::Vec4f diffuse_sum = {0.0f, 0.0f, 0.0f, 1.0f};
    Common::Vec4f specular_sum = {0.0f, 0.0f, 0.0f, 1.0f};

    for (u32 light_index = 0; light_index <= st.max_light_index; ++light_index) {
        const auto& light = st.lights[light_index];
        Common::Vec3f refl_value{};
        Common::Vec3f light_vector{};

        if (light.directional) {
            light_vector = light.position;
        } else {
            light_vector = light.position + view;
        }

        [[maybe_unused]] const f32 length = light_vector.Normalize();

        Common::Vec3f norm_view = view.Normalized();
        Common::Vec3f half_vector = norm_view + light_vector;

        f32 dist_atten = 1.0f;
        if (light.dist_atten_enabled) {
            const std::size_t lut = light.dist_atten_lut;
            const f32 sample_loc =
                std::clamp(light.dist_atten_scale * length + light.dist_atten_bias, 0.0f, 1.0f);

            const u8 lutindex =
                static_cast<u8>(std::clamp(Common::FastFloor(sample_loc * 256.0f), 0.0f, 255.0f));
            const f32 delta = sample_loc * 256 - lutindex;

            dist_atten = LookupLightingLut(lighting_state, lut, lutindex, delta);
        }

        auto get_lut_value = [&](const SwLightingState::Lut& sel, std::size_t sampler) {
            f32 result = 0.0f;

            switch (sel.input) {
            case LightingRegs::LightingLutInput::NH:
                result = Common::Dot(normal, half_vector.Normalized());
                break;
            case LightingRegs::LightingLutInput::VH:
                result = Common::Dot(norm_view, half_vector.Normalized());
                break;
            case LightingRegs::LightingLutInput::NV:
                result = Common::Dot(normal, norm_view);
                break;
            case LightingRegs::LightingLutInput::LN:
                result = Common::Dot(light_vector, normal);
                break;
            case LightingRegs::LightingLutInput::SP: {
                result = Common::Dot(light_vector, light.spot_dir);
                break;
            }
            case LightingRegs::LightingLutInput::CP:
                if (st.config7) {
                    const Common::Vec3f norm_half_vector = half_vector.Normalized();
                    const Common::Vec3f half_vector_proj =
                        norm_half_vector - normal * Common::Dot(normal, norm_half_vector);
                    result = Common::Dot(half_vector_proj, tangent);
                } else {
                    result = 0.0f;
                }
                break;
            default:
                LOG_CRITICAL(HW_GPU, "Unknown lighting LUT input {}", static_cast<u32>(sel.input));
                UNIMPLEMENTED();
                result = 0.0f;
            }

            u8 index;
            f32 delta;

            if (sel.abs) {
                if (light.two_sided_diffuse) {
                    result = std::abs(result);
                } else {
                    result = std::max(result, 0.0f);
                }

                const f32 flr = Common::FastFloor(result * 256.0f);
                index = static_cast<u8>(std::clamp(flr, 0.0f, 255.0f));
                delta = result * 256 - index;
            } else {
                const f32 flr = Common::FastFloor(result * 128.0f);
                const s8 signed_index = static_cast<s8>(std::clamp(flr, -128.0f, 127.0f));
                delta = result * 128.0f - signed_index;
                index = static_cast<u8>(signed_index);
            }

            return sel.scale * LookupLightingLut(lighting_state, sampler, index, delta);
        };

        // If enabled, compute spot light attenuation value
        f32 spot_atten = 1.0f;
        if (light.spot_atten_enabled && st.sp.enabled) {
            spot_atten = get_lut_value(st.sp, light.spot_lut);
        }

        // Specular 0 component
        f32 d0_lut_value = 1.0f;
        if (st.d0.enabled) {
            d0_lut_value = get_lut_value(st.d0, st.d0.sampler);
        }

        Common::Vec3f specular_0 = d0_lut_value * light.specular_0;

        // If enabled, lookup ReflectRed value, otherwise, 1.0 is used
        if (st.rr.enabled) {
            refl_value.x = get_lut_value(st.rr, st.rr.sampler);
        } else {
            refl_value.x = 1.0f;
        }

        // If enabled, lookup ReflectGreen value, otherwise, ReflectRed value is used
        if (st.rg.enabled) {
            refl_value.y = get_lut_value(st.rg, st.rg.sampler);
        } else {
            refl_value.y = refl_value.x;
        }

        // If enabled, lookup ReflectBlue value, otherwise, ReflectRed value is used
        if (st.rb.enabled) {
            refl_value.z = get_lut_value(st.rb, st.rb.sampler);
        } else {
            refl_value.z = refl_value.x;
        }

        // Specular 1 component
        f32 d1_lut_value = 1.0f;
        if (st.d1.enabled) {
            d1_lut_value = get_lut_value(st.d1, st.d1.sampler);
        }

        Common::Vec3f specular_1 = d1_lut_value * refl_value * light.specular_1;

        // Fresnel
        // Note: only the last entry in the light slots applies the Fresnel factor
        if (light_index == st.max_light_index && st.fr.enabled) {
            const f32 lut_value = get_lut_value(st.fr, st.fr.sampler);

            // Enabled for diffuse lighting alpha component
            if (st.enable_primary_alpha) {
                diffuse_sum.a() = lut_value;
            }

            // Enabled for the specular lighting alpha component
            if (st.enable_secondary_alpha) {
                specular_sum.a() = lut_value;
            }
        }

        auto dot_product = Common::Dot(light_vector, normal);
        if (light.two_sided_diffuse) {
            dot_product = std::abs(dot_product);
        } else {
            dot_product = std::max(dot_product, 0.0f);
        }

        f32 clamp_highlights = 1.0f;
        if (st.clamp_highlights) {
            clamp_highlights = dot_product == 0.0f ? 0.0f : 1.0f;
        }

        if (light.geometric_factor_0 || light.geometric_factor_1) {
            f32 geo_factor = half_vector.Length2();
            geo_factor = geo_factor == 0.0f ? 0.0f : std::min(dot_product / geo_factor, 1.0f);
            if (light.geometric_factor_0) {
                specular_0 *= geo_factor;
            }
            if (light.geometric_factor_1) {
                specular_1 *= geo_factor;
            }
        }

        const bool shadow_primary_enable =
            st.shadow_primary && light.shadow_enabled;
        const bool shadow_secondary_enable =
            st.shadow_secondary && light.shadow_enabled;
        const auto shadow_primary =
            shadow_primary_enable ? shadow.xyz() : Common::MakeVec(1.f, 1.f, 1.f);
        const auto shadow_secondary =
            shadow_secondary_enable ? shadow.xyz() : Common::MakeVec(1.f, 1.f, 1.f);

        const auto diffuse = (light.diffuse * dot_product * shadow_primary +
                              light.ambient) *
                             dist_atten * spot_atten;
        const auto specular = (specular_0 + specular_1) * clamp_highlights * dist_atten *
                              spot_atten * shadow_secondary;

        diffuse_sum += Common::MakeVec(diffuse, 0.0f);
        specular_sum += Common::MakeVec(specular, 0.0f);
    }

    if (st.shadow_alpha) {
        // Alpha shadow also uses the Fresnel selecotr to determine which alpha to apply
        // Enabled for diffuse lighting alpha component
        if (st.enable_primary_alpha) {
            diffuse_sum.a() *= shadow.w;
        }

        // Enabled for the specular lighting alpha component
        if (st.enable_secondary_alpha) {
            specular_sum.a() *= shadow.w;
        }
    }

    diffuse_sum += Common::MakeVec(st.global_ambient, 0.0f);

    const auto diffuse = Common::MakeVec(std::clamp(diffuse_sum.x, 0.0f, 1.0f) * 255,
                                         std::clamp(diffuse_sum.y, 0.0f, 1.0f) * 255,
                                         std::clamp(diffuse_sum.z, 0.0f, 1.0f) * 255,
                                         std::clamp(diffuse_sum.w, 0.0f, 1.0f) * 255)
                             .Cast<u8>();
    const auto specular = Common::MakeVec(std::clamp(specular_sum.x, 0.0f, 1.0f) * 255,
                                          std::clamp(specular_sum.y, 0.0f, 1.0f) * 255,
                                          std::clamp(specular_sum.z, 0.0f, 1.0f) * 255,
                                          std::clamp(specular_sum.w, 0.0f, 1.0f) * 255)
                              .Cast<u8>();
    return std::make_pair(diffuse, specular);
}

} // namespace SwRenderer
