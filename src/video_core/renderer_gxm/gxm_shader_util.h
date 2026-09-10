// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string_view>
#include "common/common_types.h"

namespace GxmRenderer {

enum class ShaderStage { Vertex, Fragment };

/**
 * A libgxm error code as its SDK name, or "unknown" when it is not one of them.
 *
 * Worth having because a bare "it failed" costs a console round trip to turn into a cause,
 * and because vitasdk's `SceGxmErrorCode` enum stops at 0x805B0027 and misses the whole
 * out-of-memory range the shader patcher reports (OUT_OF_BUFFER_MEMORY 0x805B0022,
 * OUT_OF_VERTEX_USSE_MEMORY 0x805B0023, OUT_OF_FRAGMENT_USSE_MEMORY 0x805B0024) - and gets
 * SCE_GXM_ERROR_INVALID_TEXTURE wrong, naming 0x805B0018 (which the SDK header calls
 * INVALID_THREAD) instead of 0x805B0020. The names here follow the SDK's own error.h.
 */
[[nodiscard]] std::string_view GxmErrorName(int error);

} // namespace GxmRenderer
