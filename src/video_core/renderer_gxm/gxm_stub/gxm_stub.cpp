// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <psp2/gxm.h>
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_stub/gxm_stub.h"

namespace {

// The GXP header as the SDK lays it out (Vita3K's gxm/types.h names the fields; our own
// gxp_writer emits the same words, since the console accepts them). Only what the parameter
// queries need.
struct GxpHeader {
    uint32_t magic;
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t sdk_version;
    uint32_t size;
    uint32_t binary_guid;
    uint32_t source_guid;
    uint32_t program_flags;
    uint32_t buffer_flags;
    uint32_t texunit_flags[2];
    uint32_t parameter_count;
    uint32_t parameters_offset; ///< from the start of this field
    uint32_t varyings_offset;
    uint16_t primary_reg_count;
    uint16_t secondary_reg_count;
    uint32_t temp_reg_count1;
    uint16_t temp_reg_count2;
    uint16_t primary_program_phase_count;
    uint32_t primary_program_instr_count;
    uint32_t primary_program_offset;
    uint32_t secondary_program_instr_count;
    uint32_t secondary_program_offset;
    uint32_t secondary_program_offset_end;
    uint32_t scratch_buffer_count;
    uint32_t thread_buffer_count;
    uint32_t literal_buffer_count;
    uint32_t data_buffer_count;
    uint32_t texture_buffer_count;
    uint32_t default_uniform_buffer_count;
};

struct GxpParameter {
    int32_t name_offset; ///< from the start of this structure
    uint8_t category_type;          ///< category in the low nibble, type in the high
    uint8_t components_container;   ///< component count low, container index high
    uint8_t semantic;
    uint8_t semantic_index;
    uint32_t array_size;
    uint32_t resource_index;
};
static_assert(sizeof(GxpParameter) == 16);

const GxpParameter* Parameters(const SceGxmProgram* program) {
    const auto* header = reinterpret_cast<const GxpHeader*>(program);
    const auto* base = reinterpret_cast<const uint8_t*>(&header->parameters_offset);
    return reinterpret_cast<const GxpParameter*>(base + header->parameters_offset);
}

const GxpParameter* Param(const SceGxmProgramParameter* parameter) {
    return reinterpret_cast<const GxpParameter*>(parameter);
}

// One uniform buffer per reserve: the renderer fills it before the draw and never reads it
// back, so a small ring of scratch buffers is all a GPU-less context needs.
constexpr uint32_t UniformRingSlots = 64;
constexpr uint32_t UniformSlotBytes = 16 * 1024;
alignas(16) uint8_t uniform_ring[UniformRingSlots * UniformSlotBytes];
uint32_t uniform_ring_next = 0;

void* ReserveUniforms() {
    void* slot = uniform_ring + (uniform_ring_next % UniformRingSlots) * UniformSlotBytes;
    uniform_ring_next++;
    return slot;
}

// Device memory: malloc, 4 KiB aligned like the console's memblocks. A generous frame arena
// wraps; with no GPU reading behind the CPU, wrapping over an older frame is harmless.
constexpr uint32_t FrameArenaBytes = 64u << 20;
uint8_t* frame_arena = nullptr;
uint32_t frame_arena_used = 0;
std::mutex device_mutex;

void* AllocBlock(uint32_t size) {
    void* block = nullptr;
    if (posix_memalign(&block, 4096, size == 0 ? 4096 : size) != 0) {
        return nullptr;
    }
    return block;
}

void* FrameAlloc(uint32_t size, uint32_t align) {
    std::scoped_lock lock{device_mutex};
    if (frame_arena == nullptr) {
        frame_arena = static_cast<uint8_t*>(AllocBlock(FrameArenaBytes));
    }
    const uint32_t a = align == 0 ? 16 : align;
    uint32_t at = (frame_arena_used + a - 1) & ~(a - 1);
    if (at + size > FrameArenaBytes) {
        at = 0;
    }
    frame_arena_used = at + size;
    return frame_arena + at;
}

uint32_t LargestFree(bool) {
    // No pool to ask, and malloc rarely refuses. Report the console's CDRAM pool so the
    // default surface trim (headroom under 12 MB) does not run every frame here: with 0
    // it trimmed the Smash title's textures each frame and the stub reloaded 352 KiB a
    // frame that the console never would (2026-09-07). `smallcache` still forces it.
    return 64u * 1024u * 1024u;
}

void FreeBlock(void* block) {
    std::free(block);
}

volatile uint32_t notifications[512];
std::atomic<uint32_t> notifications_used{0};

volatile uint32_t* AllocNotification() {
    const uint32_t i = notifications_used.fetch_add(1);
    return i < 512 ? &notifications[i] : nullptr;
}

// The opaque objects. libgxm's are private; ours only need to be distinct non-null pointers.
struct StubContext {
    int unused;
};
struct StubPatcher {
    int unused;
};
StubContext stub_context;
StubPatcher stub_patcher;

struct StubVertexProgram {
    const SceGxmProgram* program;
};
struct StubFragmentProgram {
    const SceGxmProgram* program;
};
struct StubRenderTarget {
    SceGxmRenderTargetParams params;
};

} // namespace

namespace GxmStub {

void InstallDevice() {
    GxmRenderer::SetDevice(GxmRenderer::GxmDevice{
        .context = reinterpret_cast<SceGxmContext*>(&stub_context),
        .patcher = reinterpret_cast<SceGxmShaderPatcher*>(&stub_patcher),
        .alloc_mapped = AllocBlock,
        .alloc_mapped_cached = AllocBlock,
        .alloc_cdram = AllocBlock,
        .largest_free = LargestFree,
        .frame_alloc = FrameAlloc,
        .free_block = FreeBlock,
        .alloc_notification = AllocNotification,
        .razor = false,
    });
}

} // namespace GxmStub

extern "C" {

// Scenes: complete as soon as they end.
int sceGxmBeginScene(SceGxmContext*, unsigned int, const SceGxmRenderTarget*,
                     const SceGxmValidRegion*, SceGxmSyncObject*, SceGxmSyncObject*,
                     const SceGxmColorSurface*, const SceGxmDepthStencilSurface*) {
    return 0;
}

int sceGxmEndScene(SceGxmContext*, const SceGxmNotification* vertex,
                   const SceGxmNotification* fragment) {
    if (vertex != nullptr && vertex->address != nullptr) {
        *vertex->address = vertex->value;
    }
    if (fragment != nullptr && fragment->address != nullptr) {
        *fragment->address = fragment->value;
    }
    return 0;
}

void sceGxmFinish(SceGxmContext*) {}

int sceGxmNotificationWait(const SceGxmNotification*) {
    return 0;
}

int sceGxmDraw(SceGxmContext*, SceGxmPrimitiveType, SceGxmIndexFormat, const void*,
               unsigned int) {
    return 0;
}

// Surfaces and render targets.
int sceGxmColorSurfaceInit(SceGxmColorSurface* surface, SceGxmColorFormat,
                           SceGxmColorSurfaceType, SceGxmColorSurfaceScaleMode,
                           SceGxmOutputRegisterSize, unsigned int, unsigned int, unsigned int,
                           void* data) {
    std::memset(surface, 0, sizeof(*surface));
    surface->pbeEmitWords[0] = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(data));
    return 0;
}

int sceGxmColorSurfaceInitDisabled(SceGxmColorSurface* surface) {
    std::memset(surface, 0, sizeof(*surface));
    return 0;
}

int sceGxmDepthStencilSurfaceInit(SceGxmDepthStencilSurface* surface, SceGxmDepthStencilFormat,
                                  SceGxmDepthStencilSurfaceType, unsigned int, void* depth,
                                  void* stencil) {
    std::memset(surface, 0, sizeof(*surface));
    surface->depthData = depth;
    surface->stencilData = stencil;
    return 0;
}

void sceGxmDepthStencilSurfaceSetBackgroundDepth(SceGxmDepthStencilSurface* surface,
                                                 float depth) {
    surface->backgroundDepth = depth;
}
void sceGxmDepthStencilSurfaceSetBackgroundStencil(SceGxmDepthStencilSurface*, unsigned char) {}
void sceGxmDepthStencilSurfaceSetForceLoadMode(SceGxmDepthStencilSurface*,
                                               SceGxmDepthStencilForceLoadMode) {}
void sceGxmDepthStencilSurfaceSetForceStoreMode(SceGxmDepthStencilSurface*,
                                                SceGxmDepthStencilForceStoreMode) {}

int sceGxmCreateRenderTarget(const SceGxmRenderTargetParams* params,
                             SceGxmRenderTarget** target) {
    auto* rt = new StubRenderTarget{*params};
    *target = reinterpret_cast<SceGxmRenderTarget*>(rt);
    return 0;
}

int sceGxmDestroyRenderTarget(SceGxmRenderTarget* target) {
    delete reinterpret_cast<StubRenderTarget*>(target);
    return 0;
}

// Programs: real GXP blobs, parsed for their parameter tables.
int sceGxmProgramCheck(const SceGxmProgram* program) {
    const auto* header = reinterpret_cast<const GxpHeader*>(program);
    return header != nullptr && header->magic == 0x00505847u ? 0 : SCE_GXM_ERROR_INVALID_VALUE;
}

const SceGxmProgramParameter* sceGxmProgramFindParameterByName(const SceGxmProgram* program,
                                                                const char* name) {
    if (program == nullptr || name == nullptr) {
        return nullptr;
    }
    const auto* header = reinterpret_cast<const GxpHeader*>(program);
    const GxpParameter* params = Parameters(program);
    for (uint32_t i = 0; i < header->parameter_count; i++) {
        const auto* p = &params[i];
        const char* pname = reinterpret_cast<const char*>(p) + p->name_offset;
        if (std::strcmp(pname, name) == 0) {
            return reinterpret_cast<const SceGxmProgramParameter*>(p);
        }
    }
    return nullptr;
}

unsigned int sceGxmProgramGetDefaultUniformBufferSize(const SceGxmProgram* program) {
    return reinterpret_cast<const GxpHeader*>(program)->default_uniform_buffer_count * 4;
}

unsigned int sceGxmProgramParameterGetArraySize(const SceGxmProgramParameter* parameter) {
    return Param(parameter)->array_size;
}
SceGxmParameterCategory sceGxmProgramParameterGetCategory(
    const SceGxmProgramParameter* parameter) {
    return static_cast<SceGxmParameterCategory>(Param(parameter)->category_type & 0xF);
}
unsigned int sceGxmProgramParameterGetComponentCount(const SceGxmProgramParameter* parameter) {
    return Param(parameter)->components_container & 0xF;
}
unsigned int sceGxmProgramParameterGetContainerIndex(const SceGxmProgramParameter* parameter) {
    return Param(parameter)->components_container >> 4;
}
unsigned int sceGxmProgramParameterGetResourceIndex(const SceGxmProgramParameter* parameter) {
    return Param(parameter)->resource_index;
}
SceGxmParameterType sceGxmProgramParameterGetType(const SceGxmProgramParameter* parameter) {
    return static_cast<SceGxmParameterType>(Param(parameter)->category_type >> 4);
}

// Uniforms: the same float placement the renderer's direct path uses (resource index in
// floats, then the component offset).
int sceGxmReserveFragmentDefaultUniformBuffer(SceGxmContext*, void** buffer) {
    *buffer = ReserveUniforms();
    return 0;
}
int sceGxmReserveVertexDefaultUniformBuffer(SceGxmContext*, void** buffer) {
    *buffer = ReserveUniforms();
    return 0;
}
int sceGxmSetUniformDataF(void* buffer, const SceGxmProgramParameter* parameter,
                          unsigned int component_offset, unsigned int component_count,
                          const float* data) {
    if (buffer == nullptr || parameter == nullptr || data == nullptr) {
        return SCE_GXM_ERROR_INVALID_POINTER;
    }
    const uint32_t at = Param(parameter)->resource_index + component_offset;
    if ((at + component_count) * sizeof(float) > UniformSlotBytes) {
        return SCE_GXM_ERROR_INVALID_VALUE;
    }
    std::memcpy(static_cast<float*>(buffer) + at, data, component_count * sizeof(float));
    return 0;
}

// State: nothing to keep.
void sceGxmSetBackDepthFunc(SceGxmContext*, SceGxmDepthFunc) {}
void sceGxmSetBackDepthWriteEnable(SceGxmContext*, SceGxmDepthWriteMode) {}
void sceGxmSetBackStencilFunc(SceGxmContext*, SceGxmStencilFunc, SceGxmStencilOp, SceGxmStencilOp,
                              SceGxmStencilOp, unsigned char, unsigned char) {}
void sceGxmSetBackStencilRef(SceGxmContext*, unsigned int) {}
void sceGxmSetCullMode(SceGxmContext*, SceGxmCullMode) {}
void sceGxmSetFragmentProgram(SceGxmContext*, const SceGxmFragmentProgram*) {}
int sceGxmSetFragmentTexture(SceGxmContext*, unsigned int, const SceGxmTexture*) {
    return 0;
}
void sceGxmSetFrontDepthFunc(SceGxmContext*, SceGxmDepthFunc) {}
void sceGxmSetFrontDepthWriteEnable(SceGxmContext*, SceGxmDepthWriteMode) {}
void sceGxmSetFrontStencilFunc(SceGxmContext*, SceGxmStencilFunc, SceGxmStencilOp,
                               SceGxmStencilOp, SceGxmStencilOp, unsigned char, unsigned char) {}
void sceGxmSetFrontStencilRef(SceGxmContext*, unsigned int) {}
void sceGxmSetRegionClip(SceGxmContext*, SceGxmRegionClipMode, unsigned int, unsigned int,
                         unsigned int, unsigned int) {}
int sceGxmSetUserMarker(SceGxmContext*, const char*) {
    return 0;
}
void sceGxmSetVertexProgram(SceGxmContext*, const SceGxmVertexProgram*) {}
int sceGxmSetVertexStream(SceGxmContext*, unsigned int, const void*) {
    return 0;
}
void sceGxmSetViewport(SceGxmContext*, float, float, float, float, float, float) {}
void sceGxmSetViewportEnable(SceGxmContext*, SceGxmViewportMode) {}

// Shader patcher: a registered program is its header, a patched program remembers it.
int sceGxmShaderPatcherRegisterProgram(SceGxmShaderPatcher*, const SceGxmProgram* program,
                                       SceGxmShaderPatcherId* id) {
    *id = reinterpret_cast<SceGxmShaderPatcherId>(const_cast<SceGxmProgram*>(program));
    return 0;
}
int sceGxmShaderPatcherUnregisterProgram(SceGxmShaderPatcher*, SceGxmShaderPatcherId) {
    return 0;
}
int sceGxmShaderPatcherCreateVertexProgram(SceGxmShaderPatcher*, SceGxmShaderPatcherId id,
                                           const SceGxmVertexAttribute*, unsigned int,
                                           const SceGxmVertexStream*, unsigned int,
                                           SceGxmVertexProgram** out) {
    *out = reinterpret_cast<SceGxmVertexProgram*>(
        new StubVertexProgram{reinterpret_cast<const SceGxmProgram*>(id)});
    return 0;
}
int sceGxmShaderPatcherCreateFragmentProgram(SceGxmShaderPatcher*, SceGxmShaderPatcherId id,
                                             SceGxmOutputRegisterFormat, SceGxmMultisampleMode,
                                             const SceGxmBlendInfo*, const SceGxmProgram*,
                                             SceGxmFragmentProgram** out) {
    *out = reinterpret_cast<SceGxmFragmentProgram*>(
        new StubFragmentProgram{reinterpret_cast<const SceGxmProgram*>(id)});
    return 0;
}
int sceGxmShaderPatcherReleaseVertexProgram(SceGxmShaderPatcher*, SceGxmVertexProgram* p) {
    delete reinterpret_cast<StubVertexProgram*>(p);
    return 0;
}
int sceGxmShaderPatcherReleaseFragmentProgram(SceGxmShaderPatcher*, SceGxmFragmentProgram* p) {
    delete reinterpret_cast<StubFragmentProgram*>(p);
    return 0;
}

// Textures: the control words hold what the renderer reads back, the type and nothing else.
int sceGxmTextureInitLinear(SceGxmTexture* texture, const void* data, SceGxmTextureFormat,
                            unsigned int, unsigned int, unsigned int) {
    std::memset(texture, 0, sizeof(*texture));
    texture->generic2.type = SCE_GXM_TEXTURE_LINEAR >> 29;
    texture->data_addr = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(data)) >> 2;
    return 0;
}
int sceGxmTextureInitLinearStrided(SceGxmTexture* texture, const void* data,
                                   SceGxmTextureFormat, unsigned int, unsigned int,
                                   unsigned int) {
    std::memset(texture, 0, sizeof(*texture));
    texture->generic2.type = SCE_GXM_TEXTURE_LINEAR_STRIDED >> 29;
    texture->data_addr = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(data)) >> 2;
    return 0;
}
SceGxmTextureType sceGxmTextureGetType(const SceGxmTexture* texture) {
    return static_cast<SceGxmTextureType>(texture->generic2.type << 29);
}
int sceGxmTextureSetData(SceGxmTexture* texture, const void* data) {
    texture->data_addr = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(data)) >> 2;
    return 0;
}
int sceGxmTextureSetMagFilter(SceGxmTexture*, SceGxmTextureFilter) {
    return 0;
}
int sceGxmTextureSetMinFilter(SceGxmTexture*, SceGxmTextureFilter) {
    return 0;
}
int sceGxmTextureSetMipFilter(SceGxmTexture*, SceGxmTextureMipFilter) {
    return 0;
}
int sceGxmTextureSetUAddrMode(SceGxmTexture*, SceGxmTextureAddrMode) {
    return 0;
}
int sceGxmTextureSetVAddrMode(SceGxmTexture*, SceGxmTextureAddrMode) {
    return 0;
}

// Transfers: the readback path copies through the transfer unit, and a copy that copies
// keeps the render thread's later decisions the same as on the console.
static uint32_t TransferBytesPerPixel(SceGxmTransferFormat format) {
    switch (format) {
    case SCE_GXM_TRANSFER_FORMAT_U8_R:
        return 1;
    case SCE_GXM_TRANSFER_FORMAT_RAW16:
    case SCE_GXM_TRANSFER_FORMAT_U4U4U4U4_ABGR:
    case SCE_GXM_TRANSFER_FORMAT_U1U5U5U5_ABGR:
    case SCE_GXM_TRANSFER_FORMAT_U5U6U5_BGR:
    case SCE_GXM_TRANSFER_FORMAT_U8U8_GR:
        return 2;
    case SCE_GXM_TRANSFER_FORMAT_U8U8U8_BGR:
        return 3;
    default:
        return 4;
    }
}

int sceGxmTransferCopy(uint32_t width, uint32_t height, uint32_t, uint32_t,
                       SceGxmTransferColorKeyMode, SceGxmTransferFormat src_format,
                       SceGxmTransferType src_type, const void* src, uint32_t src_x,
                       uint32_t src_y, int32_t src_stride, SceGxmTransferFormat dst_format,
                       SceGxmTransferType dst_type, void* dst, uint32_t dst_x, uint32_t dst_y,
                       int32_t dst_stride, SceGxmSyncObject*, uint32_t,
                       const SceGxmNotification* notification) {
    const uint32_t bpp = TransferBytesPerPixel(src_format);
    if (src != nullptr && dst != nullptr && src_type == SCE_GXM_TRANSFER_LINEAR &&
        dst_type == SCE_GXM_TRANSFER_LINEAR && bpp == TransferBytesPerPixel(dst_format)) {
        for (uint32_t y = 0; y < height; y++) {
            const auto* s = static_cast<const uint8_t*>(src) + (src_y + y) * src_stride +
                            src_x * bpp;
            auto* d = static_cast<uint8_t*>(dst) + (dst_y + y) * dst_stride + dst_x * bpp;
            std::memcpy(d, s, width * bpp);
        }
    }
    if (notification != nullptr && notification->address != nullptr) {
        *notification->address = notification->value;
    }
    return 0;
}

int sceGxmTransferDownscale(SceGxmTransferFormat, const void*, unsigned int, unsigned int,
                            unsigned int, unsigned int, int, SceGxmTransferFormat, void*,
                            unsigned int, unsigned int, int, SceGxmSyncObject*, unsigned int,
                            const SceGxmNotification* notification) {
    if (notification != nullptr && notification->address != nullptr) {
        *notification->address = notification->value;
    }
    return 0;
}

int sceGxmTransferFill(uint32_t color, SceGxmTransferFormat format, void* dst, uint32_t dst_x,
                       uint32_t dst_y, uint32_t width, uint32_t height, int32_t stride,
                       SceGxmSyncObject*, uint32_t, const SceGxmNotification* notification) {
    const uint32_t bpp = TransferBytesPerPixel(format);
    if (dst != nullptr && bpp == 4) {
        for (uint32_t y = 0; y < height; y++) {
            auto* row = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(dst) +
                                                    (dst_y + y) * stride) +
                        dst_x;
            for (uint32_t x = 0; x < width; x++) {
                row[x] = color;
            }
        }
    }
    if (notification != nullptr && notification->address != nullptr) {
        *notification->address = notification->value;
    }
    return 0;
}

int sceGxmTransferFinish(void) {
    return 0;
}

} // extern "C"
