// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/renderer_software/sw_texturing.h"

namespace SwRenderer {

using TevStageConfig = Pica::TexturingRegs::TevStageConfig;

int GetWrappedTexCoord(Pica::TexturingRegs::TextureConfig::WrapMode mode, s32 val, u32 size) {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;

    // PICA texture dimensions are powers of two, and the ARMv7-A baseline this builds for
    // (Cortex-A9) has no divide instruction: the % below is a libgcc call per texel coordinate.
    // The mask is exact for power-of-two sizes; the division stays as the fallback for any
    // degenerate register value. Size zero must keep the old behaviour - AEABI's division by
    // zero returns zero - because the mask form would pass the coordinate through unwrapped.
    const auto wrap = [size](u32 v) {
        if (size == 0) {
            return 0u;
        }
        return (size & (size - 1)) == 0 ? v & (size - 1) : v % size;
    };

    switch (mode) {
    case TextureConfig::ClampToEdge2:
        // For negative coordinate, ClampToEdge2 behaves the same as Repeat
        if (val < 0) {
            return static_cast<s32>(wrap(static_cast<u32>(val)));
        }
        [[fallthrough]];
    case TextureConfig::ClampToEdge:
        val = std::max(val, 0);
        val = std::min(val, static_cast<s32>(size) - 1);
        return val;
    case TextureConfig::ClampToBorder:
        return val;
    case TextureConfig::ClampToBorder2:
    // For ClampToBorder2, the case of positive coordinate beyond the texture size is already
    // handled outside. Here we only handle the negative coordinate in the same way as Repeat.
    case TextureConfig::Repeat2:
    case TextureConfig::Repeat3:
    case TextureConfig::Repeat:
        return static_cast<s32>(wrap(static_cast<u32>(val)));
    case TextureConfig::MirroredRepeat: {
        const u32 period = 2 * size;
        u32 coord = period == 0                    ? 0
                    : (period & (period - 1)) == 0 ? static_cast<u32>(val) & (period - 1)
                                                   : static_cast<u32>(val) % period;
        if (coord >= size) {
            coord = 2 * size - 1 - coord;
        }
        return static_cast<s32>(coord);
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown texture coordinate wrapping mode {:x}", (int)mode);
        UNIMPLEMENTED();
        return 0;
    }
};

} // namespace SwRenderer
