// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdio>
#include <iterator>
#include <cstring>
#include <psp2/gxm.h>
#include "common/logging/log.h"
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_shader_util.h"

namespace GxmRenderer {

std::string_view GxmErrorName(int error) {
    switch (static_cast<u32>(error)) {
    case 0x805B0000: return "UNINITIALIZED";
    case 0x805B0001: return "ALREADY_INITIALIZED";
    case 0x805B0002: return "OUT_OF_MEMORY";
    case 0x805B0003: return "INVALID_VALUE";
    case 0x805B0004: return "INVALID_POINTER";
    case 0x805B0005: return "INVALID_ALIGNMENT";
    case 0x805B0006: return "NOT_WITHIN_SCENE";
    case 0x805B0007: return "WITHIN_SCENE";
    case 0x805B0008: return "NULL_PROGRAM";
    case 0x805B0009: return "UNSUPPORTED";
    case 0x805B000A: return "PATCHER_INTERNAL";
    case 0x805B000B: return "RESERVE_FAILED";
    case 0x805B000C: return "PROGRAM_IN_USE";
    case 0x805B000D: return "INVALID_INDEX_COUNT";
    case 0x805B000E: return "INVALID_POLYGON_MODE";
    case 0x805B000F: return "INVALID_SAMPLER_RESULT_TYPE_PRECISION";
    case 0x805B0010: return "INVALID_SAMPLER_RESULT_TYPE_COMPONENT_COUNT";
    case 0x805B0011: return "UNIFORM_BUFFER_NOT_RESERVED";
    case 0x805B0014: return "INVALID_PRECOMPUTED_DRAW";
    case 0x805B0015: return "INVALID_PRECOMPUTED_VERTEX_STATE";
    case 0x805B0016: return "INVALID_PRECOMPUTED_FRAGMENT_STATE";
    case 0x805B0017: return "DRIVER";
    case 0x805B0018: return "INVALID_THREAD";
    case 0x805B0019: return "INVALID_TEXTURE_DATA_POINTER";
    case 0x805B001A: return "INVALID_TEXTURE_PALETTE_POINTER";
    case 0x805B001B: return "INVALID_OUTPUT_REGISTER_SIZE";
    case 0x805B001C: return "INVALID_FRAGMENT_MSAA_MODE";
    case 0x805B001D: return "INVALID_VISIBILITY_BUFFER_POINTER";
    case 0x805B001E: return "INVALID_VISIBILITY_INDEX";
    case 0x805B001F: return "INVALID_DEPTH_STENCIL_CONFIGURATION";
    case 0x805B0020: return "INVALID_TEXTURE";
    case 0x805B0021: return "OUT_OF_HOST_MEMORY";
    case 0x805B0022: return "OUT_OF_BUFFER_MEMORY";
    case 0x805B0023: return "OUT_OF_VERTEX_USSE_MEMORY";
    case 0x805B0024: return "OUT_OF_FRAGMENT_USSE_MEMORY";
    case 0x805B0025: return "INVALID_PRIMITIVE_TYPE";
    case 0x805B0026: return "INVALID_MAPPING";
    case 0x805B0027: return "OUT_OF_RENDER_TARGETS";
    case 0x805B0028: return "INVALID_VISIBILITY_OP";
    case 0x805B0029: return "RAZOR";
    case 0x805B002A: return "INVALID_SAMPLER_FILTER_MODE";
    case 0x805B002B: return "INVALID_REGION_CLIP_IN_COMMAND_LIST";
    case 0x805B002C: return "WITHIN_COMMAND_LIST";
    case 0x805B002D: return "NOT_WITHIN_COMMAND_LIST";
    case 0x805B002E: return "BUFFER_OVERRUN";
    default: return "unknown";
    }
}

namespace {

const GxmDevice* device_instance = nullptr;
GxmDevice device_storage{};

} // Anonymous namespace

void SetDevice(const GxmDevice& device) {
    device_storage = device;
    device_instance = device.context != nullptr ? &device_storage : nullptr;
}

const GxmDevice& Device() {
    return device_storage;
}

bool HasDevice() {
    return device_instance != nullptr;
}

} // namespace GxmRenderer
