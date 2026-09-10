// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"

namespace Pica::Shader {

struct VKFormatTraits {
    u8 transfer_support{};
    u8 blit_support{};
    u8 attachment_support{};
    u8 storage_support{};
    u8 needs_conversion{};
    u8 needs_emulation{};
    u32 usage_flags{};
    u32 aspect_flags{};
    u32 native_format{};

    auto operator<=>(const VKFormatTraits&) const = default;
};

/**
 * Geometry of the 2D LUT textures used when texture buffers are unavailable. The fragment
 * shader bakes the shape into its index arithmetic (i -> i & 255, i >> 8), so the rasterizer
 * has to allocate exactly this, and the two must agree - hence one definition.
 */
constexpr u32 LUT_TEX_WIDTH = 256;
constexpr u32 LUT_LF_ROWS = 512;   ///< lighting refreshes up to 25 rows (24 LUTs plus fog); the
                                   ///< GXM backend recycles the ring in halves, each of which
                                   ///< must outlive the scenes that sampled it (1 MiB at 8 B)
constexpr u32 LUT_RG_ROWS = 32;
constexpr u32 LUT_RGBA_ROWS = 32;  ///< proctex refreshes up to 5

struct Profile {
    u8 enable_accurate_mul{};
    u8 has_separable_shaders{};
    u8 has_clip_planes{};
    u8 has_geometry_shader{};
    u8 has_texture_buffer{};
    /// Whether the shading language has texelFetch. Without it the LUT reads fall back to
    /// normalised sampling, which is exact because the LUT textures are nearest-filtered.
    /// GXM-backed GLES drivers (the Vita) are the case this exists for.
    u8 has_texel_fetch{};
    u8 has_custom_border_color{};
    u8 has_fragment_shader_interlock{};
    u8 has_fragment_shader_barycentric{};
    u8 has_blend_minmax_factor{};
    u8 has_minus_one_to_one_range{};
    u8 has_logic_op{};

    u8 has_gl_ext_framebuffer_fetch{};
    u8 has_gl_arm_framebuffer_fetch{};
    u8 has_gl_nv_fragment_shader_interlock{};
    u8 has_gl_intel_fragment_shader_ordering{};
    u8 has_gl_nv_fragment_shader_barycentric{};

    u8 vk_disable_spirv_optimizer{};
    u8 vk_use_spirv_generator{};
    std::array<VKFormatTraits, 16> vk_format_traits{};

    u8 is_vulkan{};

    auto operator<=>(const Profile&) const = default;
};

} // namespace Pica::Shader
