// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <map>
#include <bit>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <fmt/format.h>
#ifdef __vita__
#include <malloc.h>
#include <psp2/io/stat.h>
#else
// The stub libgxm build: the dump directories are the console's, and a dump off the console
// just fails to open, which is fine.
static inline int sceIoMkdir(const char*, int) {
    return 0;
}
#endif
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "common/timer.h"
#include "core/memory.h"
#include "video_core/pica/draw_payload.h"
#include "video_core/pica/pica_core.h"
#include "video_core/rasterizer_cache/rasterizer_cache.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/gxm_rasterizer.h"
#include "video_core/renderer_gxm/pica_to_gxm.h"
#include "video_core/renderer_gxm/usse/fixed_usse_gen.h"
#include "video_core/renderer_gxm/usse/fs_usse_gen.h"
#include "video_core/renderer_gxm/usse/vs_usse_gen.h"
#include "video_core/shader/generator/cg_fs_shader_gen.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"

namespace GxmRenderer {

namespace {

// Sized from what a title actually produces: Super Mario 3D Land reaches ~50k vertices a
// frame at 96 bytes each, so 4 MiB was under one frame's worth and the producer stalled on it
// every frame - and a stall here is a full sceGxmFinish, the thing everything else in this
// backend now works to avoid.
constexpr u32 RingSize = 16 * 1024 * 1024;
constexpr u32 MaxVerticesPerDraw = 65535 - 65535 % 3;

// The fixed programs (the pass-through vertex program for software-shaded draws, the blit
// quad's pair, the clear quad's fragment program) come from usse/fixed_usse_gen: nothing is
// compiled on the console.

constexpr u32 BlitRingSize = 64 * 1024;

struct BlitVertex {
    float x, y, u, v;
};

/// Adds the time between construction and destruction to a counter, when timing is on.
class PhaseTimer {
public:
    PhaseTimer(bool enabled, u64& sink)
        : sink{enabled ? &sink : nullptr}, start{enabled ? Common::PipelineStats::NowUs() : 0} {}
    ~PhaseTimer() {
        if (sink != nullptr) {
            *sink += Common::PipelineStats::NowUs() - start;
        }
    }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;

private:
    u64* sink;
    u64 start;
};

using VideoCore::RasterizerAccelerated;
using VideoCore::SurfaceType;
static_assert(sizeof(RasterizerAccelerated::HardwareVertex) == 88);

/**
 * sceGxmSetUniformDataF with the two things a silent call hides: the parameter's real
 * capacity, and failure.
 *
 * The compiler is free to shrink a uniform array to what the program reaches - the 128-entry
 * bank a vertex program reads only statically can come out as however many entries it
 * names - and a write past the parameter's range is refused whole, with no data landing
 * and no sign of it but a return code nobody read. So the write is clamped to the
 * parameter's array size times its component count, and a refusal is logged once per
 * parameter.
 */
// logdraw: a breadcrumb per draw step for the first draws, to place a crash.
void DrawTrace(const char* step) {
    static const bool on = GxmFlag("logdraw");
    static u32 shown = 0;
    if (on && shown++ < 600) {
        LOG_INFO(Render, "trace: {}", step);
    }
}

/// A uniform's offset and size, asked of the program once.
RasterizerGxm::UniformSlot MakeSlot(const SceGxmProgramParameter* param) {
    if (param == nullptr) {
        return {};
    }
    const u32 array_size = sceGxmProgramParameterGetArraySize(param);
    const u32 components = sceGxmProgramParameterGetComponentCount(param);
    // One vector, or a scalar array, or an array whose elements are a multiple of eight
    // bytes: the floats are contiguous from the resource index. An array of float3 is not -
    // each element is padded to eight bytes - and neither is anything but f32.
    const bool direct = sceGxmProgramParameterGetType(param) == SCE_GXM_PARAMETER_TYPE_F32 &&
                        (array_size == 1 || components == 1 || components % 2 == 0);
    return {param, static_cast<u16>(sceGxmProgramParameterGetResourceIndex(param)),
            static_cast<u16>(array_size * components), direct};
}

void SetUniformF(void* buffer, const RasterizerGxm::UniformSlot& slot, const float* data,
                 u32 count, const char* what) {
    if (!slot) {
        return;
    }
    const u32 n = std::min(count, static_cast<u32>(slot.capacity));
    if (slot.direct && n <= 4) {
        // A scalar or one vector: a store each, not a libc call (a draw sets a hundred of
        // these; the calls were 11.7M eighteen-byte memcpys in an SM3DL level).
        float* dst = static_cast<float*>(buffer) + slot.offset;
        for (u32 i = 0; i < n; i++) {
            dst[i] = data[i];
        }
        return;
    }
    // The default uniform buffer is a flat float array and a parameter's resource index is
    // its offset in it (what vitaGL writes through). sceGxmSetUniformDataF does the same
    // after two more calls to ask the program what this parameter is, and a draw sets about
    // a hundred uniforms; `slowuniforms` puts the library calls back for comparison.
    if (!slot.direct) {
        const int err = sceGxmSetUniformDataF(buffer, slot.param, 0, n, data);
        if (err < 0) {
            static std::unordered_map<const SceGxmProgramParameter*, bool> reported;
            if (!reported[slot.param]) {
                reported[slot.param] = true;
                LOG_ERROR(Render, "GXM: uniform '{}' refused {} of {} floats: {} ({:#x})", what,
                          n, count, GxmErrorName(err), static_cast<u32>(err));
            }
        }
        return;
    }
    std::memcpy(static_cast<float*>(buffer) + slot.offset, data, n * sizeof(float));
}

} // Anonymous namespace

RasterizerGxm::RasterizerGxm(Memory::MemorySystem& memory_, Pica::PicaCore& pica_,
                             VideoCore::CustomTexManager& custom_tex_manager,
                             VideoCore::RendererBase& renderer_)
    : VideoCore::RasterizerAccelerated{memory_, pica_}, renderer{renderer_}, runtime{renderer_},
      res_cache{memory_, custom_tex_manager, runtime, regs, renderer_},
      profile{Pica::Shader::Generator::Cg::MakeGxmProfile()} {
    // The stuck-queue log's scene autopsy: one rasterizer per GXM device, so a static owner.
    static RasterizerGxm* trace_owner = nullptr;
    trace_owner = this;
    Common::PipelineStats::scene_report.store([] { trace_owner->ReportSceneTrace(); },
                                              std::memory_order_relaxed);
    // trimlog: every surface the trim gives up and every reload of one, in trace-probe builds.
    res_cache.SetProbeTrimLog(GxmFlag("trimlog"));
    // On by default: nine clock reads a draw, and the render thread's own account of its
    // frame is what tells a GPU-bound frame from a submission-bound one (2026-09-05).
    // Off unless asked (`timedraw`): the phase timers read the clock nine times a draw and
    // convert nanoseconds to microseconds on the host, 15-19% of the render thread on the
    // pi5 in a Smash fight and an SM3DL level, and on the
    // console each read is the sceKernelGetProcessTimeWide syscall, 150k a second in SM3DL.
    time_draws = GxmFlag("timedraw");
    runtime.SetSyncHook([this](u64 serial) { SyncToGpu(serial); });
    runtime.SetColourClearHook([this](Surface& surface) {
        if (!in_scene || scene_fb == nullptr || !scene_fb->color_id ||
            &res_cache.GetSurface(scene_fb->color_id) != &surface) {
            return false;
        }
        DrawPendingColourClears();
        return true;
    });
    SetSurfaceDumpHook([](void* self) { static_cast<RasterizerGxm*>(self)->DumpSurfaces(); },
                       this);
    runtime.SetBlitHook([this](Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
        return GpuBlit(source, dest, blit);
    });
    // The ring is claimed here rather than in EnsureResources, because the facade asks for it
    // (SetRealRasterizer) before any draw reaches this rasterizer. Only the memory is needed,
    // which the presentation layer has had mapped since long before a title loads; everything
    // that wants the GXM context still waits for the first draw.
    EnsureRing();
}

void RasterizerGxm::EnsureRing() {
    if (ring != nullptr || !HasDevice()) {
        return;
    }
    // Every vertex the guest draws is stored here once and read once. Uncached stores on this
    // CPU are slow enough that at the ~76k vertices a frame a 3D title reaches - 7 MB - the
    // choice of memory is worth measuring rather than assuming; `cachedring` is that
    // measurement.
    ring = static_cast<u8*>(Device().alloc_mapped(RingSize));
    if (ring == nullptr) {
        LOG_CRITICAL(Render, "GXM rasterizer: no memory for the {} KiB vertex ring",
                     RingSize / 1024);
        return;
    }
    ring_size = RingSize;
    vertex_ring.base = ring;
    vertex_ring.size = RingSize;
}

RasterizerGxm::~RasterizerGxm() {
    SetSurfaceDumpHook(nullptr, nullptr);
    if (in_scene) {
        EndScene();
    }
    if (HasDevice() && Device().context != nullptr) {
        GXM_PHASE("gpu:finish", 0, runtime.CurrentEpoch(), runtime.CompletedEpoch());
    sceGxmFinish(Device().context);
    }
    // Surfaces release their pool memory as the cache dies; the pools free their chunks
    // after that, in the runtime's destructor (member order: runtime before res_cache).
    pipeline_cache.reset();
    if (HasDevice()) {
        if (ring != nullptr) {
            // The facade holds a pointer to this ring; it drops it when the rasterizer that
            // replaces this one publishes its own, and both the GPU and the producer are
            // quiescent by the time a rasterizer is destroyed.
            vertex_ring.base = nullptr;
            Device().free_block(ring);
        }
        if (identity_indices != nullptr) {
            Device().free_block(identity_indices);
        }
        if (white_mem != nullptr) {
            Device().free_block(white_mem);
        }
        if (blit_ring != nullptr) {
            Device().free_block(blit_ring);
        }
        for (LutTexture* lut : {&lut_lf, &lut_rg, &lut_rgba}) {
            if (lut->mem != nullptr) {
                Device().free_block(lut->mem);
            }
        }
    }
    for (auto& [key, target] : blit_targets) {
        sceGxmDestroyRenderTarget(target);
    }
}

bool RasterizerGxm::EnsureResources() {
    if (resources_ready) {
        return true;
    }
    if (resources_failed || !HasDevice()) {
        return false;
    }
    const auto& device = Device();
    pipeline_cache = std::make_unique<PipelineCache>();
    if (!GxmFlag("notier2")) {
        tier2 = std::make_unique<ShaderTier2>();
        if (GxmFlag("tier2now")) {
            tier2_threshold = 1;
        }
    }
    EnsureRing();
    identity_indices = static_cast<u16*>(device.alloc_mapped(65536 * sizeof(u16)));
    white_mem = device.alloc_mapped(64);
    if (ring == nullptr || identity_indices == nullptr || white_mem == nullptr) {
        LOG_CRITICAL(Render, "GXM rasterizer: out of mapped memory at startup");
        resources_failed = true;
        return false;
    }
    for (u32 i = 0; i < 65536; i++) {
        identity_indices[i] = static_cast<u16>(i);
    }
    std::memset(white_mem, 0xFF, 64);
    sceGxmTextureInitLinear(&white_texture, white_mem, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, 1, 1,
                            1);
    vertex_shader = pipeline_cache->UseBuiltinShader(0x7061737376657274ull, "pica_vs_trivial",
                                                     Usse::EmitPassThroughVertexProgram());
    blit_vertex_shader = pipeline_cache->UseBuiltinShader(0x626c69745f767300ull, "blit_vs",
                                                          Usse::EmitBlitVertexProgram());
    blit_fragment_shader = pipeline_cache->UseBuiltinShader(0x626c69745f667300ull, "blit_fs",
                                                            Usse::EmitBlitFragmentProgram());
    fill_fragment_shader = pipeline_cache->UseBuiltinShader(0x66696c6c5f667300ull, "fill_fs",
                                                            Usse::EmitFillFragmentProgram());
    blit_ring = static_cast<u8*>(device.alloc_mapped(BlitRingSize));
    blit_ring_limit = BlitRingSize / 2;
    // The LUT textures, shaped by the generator's constants: the fragment programs bake the
    // row arithmetic in, so the allocation must match them.
    {
        using namespace Pica::Shader;
        const auto make_lut = [&](LutTexture& lut, u32 rows, u32 texel_bytes,
                                  SceGxmTextureFormat format) {
            lut.rows = rows;
            lut.texel_bytes = texel_bytes;
            lut.row_cursor = 0;
            lut.mem = device.alloc_mapped(LUT_TEX_WIDTH * rows * texel_bytes);
            if (lut.mem == nullptr) {
                return false;
            }
            std::memset(lut.mem, 0, LUT_TEX_WIDTH * rows * texel_bytes);
            if (sceGxmTextureInitLinear(&lut.texture, lut.mem, format, LUT_TEX_WIDTH, rows, 1) <
                0) {
                return false;
            }
            sceGxmTextureSetMinFilter(&lut.texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMagFilter(&lut.texture, SCE_GXM_TEXTURE_FILTER_POINT);
            sceGxmTextureSetMipFilter(&lut.texture, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
            sceGxmTextureSetUAddrMode(&lut.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
            sceGxmTextureSetVAddrMode(&lut.texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
            return true;
        };
        if (!make_lut(lut_lf, LUT_LF_ROWS, 8, SCE_GXM_TEXTURE_FORMAT_F32F32_00GR) ||
            !make_lut(lut_rg, LUT_RG_ROWS, 8, SCE_GXM_TEXTURE_FORMAT_F32F32_00GR) ||
            !make_lut(lut_rgba, LUT_RGBA_ROWS, 4, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR)) {
            LOG_CRITICAL(Render, "GXM rasterizer: no memory for the LUT textures");
            resources_failed = true;
            return false;
        }
    }
    // The stream the software-shaded path draws from is Pica::OutputVertex as the vertex
    // shader left it: every component is already a host float, so the attributes are offsets
    // into that struct and no vertex is ever repacked. The GL backend's ring path names the
    // same offsets.
    {
        using Pica::OutputVertex;
        fixed_layout = {};
        fixed_layout.shader = vertex_shader;
        fixed_layout.stream_count = 1;
        fixed_layout.strides[0] = sizeof(OutputVertex);
        fixed_layout.attribute_count = 8;
        const auto at = [](const char* name, std::size_t offset, u8 components) {
            return VertexAttribute{name, static_cast<u16>(offset), SCE_GXM_ATTRIBUTE_FORMAT_F32,
                                   components, 0};
        };
        fixed_layout.attributes[0] = at("vert_position", offsetof(OutputVertex, pos), 4);
        fixed_layout.attributes[1] = at("vert_color", offsetof(OutputVertex, color), 4);
        fixed_layout.attributes[2] = at("vert_texcoord0", offsetof(OutputVertex, tc0), 2);
        fixed_layout.attributes[3] = at("vert_texcoord1", offsetof(OutputVertex, tc1), 2);
        fixed_layout.attributes[4] = at("vert_texcoord2", offsetof(OutputVertex, tc2), 2);
        fixed_layout.attributes[5] = at("vert_texcoord0_w", offsetof(OutputVertex, tc0_w), 1);
        fixed_layout.attributes[6] = at("vert_normquat", offsetof(OutputVertex, quat), 4);
        fixed_layout.attributes[7] = at("vert_view", offsetof(OutputVertex, view), 3);
    }
    resources_ready = true;
    // The vertex program's three uniforms, once. Its compile may still be running, so this
    // is only a first attempt; UploadUniforms retries while any of them is unresolved.
    ResolveVertexUniforms();
    LOG_INFO(Render, "GXM rasterizer up: {} KiB vertex ring, surface cache on", RingSize / 1024);
    return true;
}

SceGxmRenderTarget* RasterizerGxm::BlitTargetFor(u32 width, u32 height) {
    const u32 key = (width << 16) | height;
    const auto it = blit_targets.find(key);
    if (it != blit_targets.end()) {
        return it->second;
    }
    SceGxmRenderTargetParams rt{};
    rt.width = static_cast<u16>(width);
    rt.height = static_cast<u16>(height);
    rt.scenesPerFrame = 4;
    rt.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    rt.driverMemBlock = static_cast<SceUID>(-1);
    SceGxmRenderTarget* target = nullptr;
    const int err = sceGxmCreateRenderTarget(&rt, &target);
    if (err < 0) {
        LOG_ERROR(Render, "GXM: no blit render target for {}x{}: {} ({:#x})", width, height,
                  GxmErrorName(err), static_cast<u32>(err));
        target = nullptr;
    }
    blit_targets.emplace(key, target);
    return target;
}

bool RasterizerGxm::GpuBlit(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
    static const bool no_gpu_blit = GxmFlag("nogpublit");
    if (no_gpu_blit || !resources_ready || !HasDevice() || blit_ring == nullptr) {
        return false;
    }
    // Only a colour destination has a surface the pixel back end can convert into, and only
    // level 0 of it: the render target addresses the surface's own memory and stride.
    if (!dest.IsColorTarget() || blit.dst_level != 0 || !source.HasMemory()) {
        return false;
    }
    // Sampling memory the same scene renders into is undefined, and overlap is enough for
    // that - the two need not be the same surface.
    const u8* src_base = source.Data();
    const u8* dst_base = dest.Data();
    if (src_base < dst_base + dest.AllocSize() && dst_base < src_base + source.AllocSize()) {
        return false;
    }
    // The programs are compiled on the worker like any other; until they are, the CPU path
    // does the work rather than the draw being dropped.
    for (Shader* shader : {blit_vertex_shader, blit_fragment_shader, fill_fragment_shader}) {
        if (shader == nullptr || !shader->IsDone() || shader->failed) {
            return false;
        }
    }

    const bool flip = blit.src_rect.bottom > blit.src_rect.top;
    const u32 src_top = flip ? blit.src_rect.bottom : blit.src_rect.top;
    const u32 src_bottom = flip ? blit.src_rect.top : blit.src_rect.bottom;
    const u32 dst_w = blit.dst_rect.GetWidth();
    const u32 dst_h = blit.dst_rect.GetHeight();
    if (dst_w == 0 || dst_h == 0) {
        return true;
    }
    // A scene costs a scene: the open one has to be submitted for this to have a target of
    // its own, and below a few thousand texels the CPU pass is cheaper than that boundary -
    // unless either surface is still in flight, when the CPU pass first drains the GPU
    // (Smash's results screen: four 8x8 and 24x8 slivers a frame, each behind a full drain).
    constexpr u32 MinGpuBlitTexels = 64 * 64;
    if (dst_w * dst_h < MinGpuBlitTexels && !runtime.InFlight(source) &&
        !runtime.InFlight(dest)) {
        return false;
    }
    // The open scene may already be rendering into this very destination, in which case the
    // quad goes into it and the pair costs one scene instead of two. A scene is the expensive
    // unit on this GPU - Smash's victory screen spends 188 of them on a frame, a third on
    // conversion blits that follow a draw to the same target (2026-09-09) - and the merge is
    // free: GXM context state already persists across scene boundaries, the source cannot be
    // what this scene renders into (overlap was refused above), and the destination is held
    // by the open scene's serial rather than one of its own.
    const bool merged = in_scene && scene_fb != nullptr && scene_fb->color_id &&
                        &res_cache.GetSurface(scene_fb->color_id) == &dest;
    SceGxmRenderTarget* target = nullptr;
    if (!merged) {
        target = BlitTargetFor(dest.width, dest.height);
        if (target == nullptr) {
            return false;
        }
    }

    // Quad memory: bump-allocated and reset with the frame, after the wait that proves the GPU
    // has finished reading it. Two triangles rather than a strip, because that is what every
    // other draw here uses and the index buffer is already the identity.
    constexpr u32 BlitVertices = 6;
    const u32 offset = (blit_ring_used + 15u) & ~15u;
    if (offset + BlitVertices * sizeof(BlitVertex) > blit_ring_limit) {
        return false;
    }
    blit_ring_used = offset + BlitVertices * sizeof(BlitVertex);
    auto* quad = reinterpret_cast<BlitVertex*>(blit_ring + offset);

    // Clip space, with the viewport set to the destination rectangle below. An earlier version
    // put pixel coordinates through a disabled viewport instead; nothing in the SDK samples or
    // in vitaGL does that, the exact convention it expects is unverified, and it produced a
    // black screen. This is the transform the PICA draws and the presentation layer's own
    // quads both already use: a positive y scale, so NDC +1 lands on the highest row index,
    // which is the top of the picture in the bottom-up order these surfaces are stored in.
    // The flip the cache asks for by handing over an upside-down source rectangle is in v.
    const float su = 1.0f / static_cast<float>(std::max(source.width, 1u));
    const float sv = 1.0f / static_cast<float>(std::max(source.height, 1u));
    const float u0 = blit.src_rect.left * su, u1 = blit.src_rect.right * su;
    const float v0 = (flip ? src_top : src_bottom) * sv;
    const float v1 = (flip ? src_bottom : src_top) * sv;
    const BlitVertex bl{-1.0f, -1.0f, u0, v0};
    const BlitVertex br{1.0f, -1.0f, u1, v0};
    const BlitVertex tl{-1.0f, 1.0f, u0, v1};
    const BlitVertex tr{1.0f, 1.0f, u1, v1};
    quad[0] = bl;
    quad[1] = br;
    quad[2] = tl;
    quad[3] = br;
    quad[4] = tr;
    quad[5] = tl;

    // A scene of its own on the destination, unless the open one is already on it.
    SceGxmContext* context = Device().context;
    SceGxmNotification notification{};
    u64 serial;
    if (merged) {
        serial = runtime.CurrentEpoch();
        runtime.stat_gpu_blit_merges++;
        if (scenes_this_frame > 0 && scenes_this_frame <= MaxSceneRecords) {
            scene_records[scenes_this_frame - 1].draws++;
        }
    } else {
    EndScene();
    // This scene samples what an earlier one may still be rendering. Its fragment stage
    // does the sampling, and fragment processing runs scene by scene in order, so no
    // vertex-stage wait is needed: SCE_GXM_SCENE_VERTEX_WAIT_FOR_DEPENDENCY is for a vertex
    // program reading the earlier scene's output, and here it only stalled the tiler.
    const u32 flags = SCE_GXM_SCENE_FRAGMENT_SET_DEPENDENCY;
    serial = runtime.BeginSceneEpoch(&notification);
    GXM_PHASE("gpu:begin-scene-blit", serial);
    if (sceGxmBeginScene(context, flags, target, nullptr, nullptr, nullptr, dest.ColorSurface(),
                         nullptr) < 0) {
        return false;
    }
    stat_scenes++;
    if (scenes_this_frame < MaxSceneRecords) {
        scene_records[scenes_this_frame] = {dest.addr, static_cast<u16>(dest.width),
                                            static_cast<u16>(dest.height), 1, true};
    }
    OpenSceneTrace(dest.addr, static_cast<u16>(dest.width), static_cast<u16>(dest.height), true);
    scenes_this_frame++;
    }
    pipeline_cache->InvalidateBinding();
    state_cache = CachedState{};
    uniform_pipeline = nullptr;

    PipelineInfo info{};
    info.vertex = blit_vertex_shader;
    info.fragment = blit_fragment_shader;
    info.stream_count = 1;
    info.strides[0] = sizeof(BlitVertex);
    info.attribute_count = 2;
    info.attributes[0] = {"vert_position", 0, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2, 0};
    info.attributes[1] = {"vert_texcoord", 8, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2, 0};
    info.blend_enabled = true;
    info.blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
    info.blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
    info.blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
    info.blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
    info.blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
    info.blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
    info.blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
    if (!pipeline_cache->BindPipeline(info)) {
        if (!merged) {
            GXM_PHASE("gpu:end-scene");
            sceGxmEndScene(context, nullptr, &notification);
            runtime.SceneSubmitted();
            CloseSceneTrace();
        }
        return false;
    }
    NoteScenePipeline(info.vertex != nullptr ? info.vertex->name.c_str() : "?",
                      info.fragment != nullptr ? info.fragment->name.c_str() : "?");

    sceGxmSetViewportEnable(context, SCE_GXM_VIEWPORT_ENABLED);
    sceGxmSetViewport(context, blit.dst_rect.left + dst_w * 0.5f, dst_w * 0.5f,
                      blit.dst_rect.bottom + dst_h * 0.5f, dst_h * 0.5f, 0.0f, 1.0f);
    sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE, blit.dst_rect.left,
                        blit.dst_rect.bottom, blit.dst_rect.right - 1, blit.dst_rect.top - 1);
    sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
    sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    sceGxmSetBackDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
    // This scene has no depth-stencil surface, and the SDK refuses any draw whose stencil
    // test is not NEVER or ALWAYS in that case with
    // SCE_GXM_ERROR_INVALID_DEPTH_STENCIL_CONFIGURATION. Stencil is context state, so a PICA
    // draw that left a real test behind would silently take this draw with it.
    for (int face = 0; face < 2; face++) {
        const auto set = face == 0 ? sceGxmSetFrontStencilFunc : sceGxmSetBackStencilFunc;
        set(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
            SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
    }

    SceGxmTexture bound = *source.Texture();
    // Point sampling, so a same-size conversion is exact; the CPU path this replaces was
    // nearest too, and a filtered result would not match what the guest expects of a
    // display transfer.
    sceGxmTextureSetMinFilter(&bound, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMagFilter(&bound, SCE_GXM_TEXTURE_FILTER_POINT);
    sceGxmTextureSetMipFilter(&bound, SCE_GXM_TEXTURE_MIP_FILTER_DISABLED);
    sceGxmTextureSetUAddrMode(&bound, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&bound, SCE_GXM_TEXTURE_ADDR_CLAMP);
    BindFragmentTexture(0, bound);

    sceGxmSetVertexStream(context, 0, quad);
    const int draw_err = sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
                                    SCE_GXM_INDEX_FORMAT_U16, identity_indices, BlitVertices);
    if (!merged) {
        GXM_PHASE("gpu:end-scene");
        sceGxmEndScene(context, nullptr, &notification);
        runtime.SceneSubmitted();
        CloseSceneTrace();
    }
    if (draw_err < 0) {
        // The scene still has to be ended, but the destination was not written: say so, and
        // let the caller fall back rather than leaving stale pixels behind.
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_ERROR(Render, "GXM: the conversion blit was refused: {} ({:#x})",
                      GxmErrorName(draw_err), static_cast<u32>(draw_err));
        }
        return false;
    }

    // Both ends are held until the GPU passes this scene: the destination is written by it and
    // the source is read by it.
    source.gpu_serial = serial;
    dest.gpu_serial = serial;
    static const bool log_draw = GxmFlag("logdraw");
    if (log_draw && stat_gpu_blit_logs < 8) {
        stat_gpu_blit_logs++;
        LOG_INFO(Render,
                 "gpu blit: {} {}x{} [{},{}-{},{}] -> {} {}x{} [{},{}-{},{}] flip {} target "
                 "{}x{} scene {}",
                 VideoCore::PixelFormatAsString(source.pixel_format), source.width,
                 source.height, blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right,
                 blit.src_rect.top, VideoCore::PixelFormatAsString(dest.pixel_format),
                 dest.width, dest.height, blit.dst_rect.left, blit.dst_rect.bottom,
                 blit.dst_rect.right, blit.dst_rect.top, flip, dest.width, dest.height, serial);
    }
    return true;
}

void RasterizerGxm::DropDraw() {
    // A draw this backend cannot make still consumed ring space, and the producer waits on
    // that space coming back. Nothing has been submitted that reads it, so it is free as soon
    // as everything before it is: fold it into the open scene if there is one, queue it behind
    // the last scene still in flight otherwise, and retire it outright when nothing is.
    u64 end_pos = hw_ring_end;
    hw_ring_end = 0;
    for (const auto& range : pending_ring_ranges) {
        end_pos = std::max(end_pos, range.end_pos);
    }
    if (end_pos != 0) {
        if (in_scene) {
            ring_scene_pos = std::max(ring_scene_pos, end_pos);
        } else if (!ring_retire.empty()) {
            ring_retire.back().pos = std::max(ring_retire.back().pos, end_pos);
        } else {
            vertex_ring.Retire(end_pos);
        }
    }
    stat_dropped_draws++;
    vertex_batch.clear();
    pending_ring_ranges.clear();
}

void RasterizerGxm::DrawRingRange(u32 first, u32 count, u64 ring_end_pos) {
    // The shipping facade only takes its zero-copy path - the emulation thread writing
    // triangles straight into this ring - when the real rasterizer implements this. Without
    // it, the facade shipped a std::vector of OutputVertex per batch and the render thread
    // replayed AddTriangle over it, copying every vertex into the ring a second time: at the
    // ~76k vertices a frame Super Mario 3D Land reaches, 7 MB a frame of stores into uncached
    // memory, on the thread that is behind.
    pending_ring_ranges.push_back({first, count, ring_end_pos});
    DrawTriangles();
}

VideoCore::VertexRing* RasterizerGxm::GetVertexRing() {
    EnsureRing();
    return vertex_ring.Active() ? &vertex_ring : nullptr;
}

void RasterizerGxm::RingRetireBlocking() {
    const PhaseTimer timer{time_draws, times.retire};
    // Called when the producer has run out of ring. Giving space back is a load and a
    // comparison, so try that first.
    if (RetireRing()) {
        return;
    }
    // Nothing was retirable, so nothing submitted is holding the space either: submit the
    // open scene so there is something to wait for. This is the one place the ring stalls.
    if (in_scene) {
        EndScene();
    }
    if (ring_retire.empty() || !HasDevice()) {
        // Nothing is in flight at all. The space the producer wants is held by draws this
        // rasterizer has not seen yet, and waiting here would never end.
        return;
    }
    GXM_PHASE("gpu:finish", 0, runtime.CurrentEpoch(), runtime.CompletedEpoch());
    sceGxmFinish(Device().context);
    stat_finishes++;
    runtime.EpochDrained();
    RetireRing();
}

bool RasterizerGxm::RetireRing() {
    const u64 completed = runtime.CompletedEpoch();
    u64 pos = 0;
    while (!ring_retire.empty() && ring_retire.front().serial <= completed) {
        pos = ring_retire.front().pos;
        ring_retire.pop_front();
    }
    if (pos == 0) {
        return false;
    }
    vertex_ring.Retire(pos);
    return true;
}

void RasterizerGxm::SyncToGpu(u64 serial) {
    DrawTrace("sync to gpu");
    // Anything that reads or writes a surface outside a scene must not race the GPU: end
    // the scene and wait. Cheap when nothing is queued (the common case between frames),
    // ruinous when it lands between two draws - the scene is torn down, the tiler runs over
    // the whole target and the CPU waits for it. stat_syncs counts how often the cache asked
    // for one and stat_scene_breaks how many of those had a scene to tear down, which is the
    // number that matters: it is scenes per frame, against a render target sized for 8.
    if (!resources_ready || !HasDevice()) {
        return;
    }
    const PhaseTimer timer{time_draws, times.sync};
    stat_syncs++;
    if (in_scene) {
        stat_scene_breaks++;
    }
    EndScene();
    if (serial != 0) {
        // One scene's notification, not the whole queue: the frames already submitted stay
        // submitted and the GPU keeps working through them. A full drain here was 22% of
        // the render thread on the console (2026-09-06), 60 ms a time, because by then two
        // frames of scenes were in flight behind the one being waited for.
        runtime.WaitForEpoch(serial);
        return;
    }
    GXM_PHASE("gpu:finish", 0, runtime.CurrentEpoch(), runtime.CompletedEpoch());
    sceGxmFinish(Device().context);
    stat_finishes++;
    runtime.EpochDrained();
}

u32 RasterizerGxm::SceneFlagsFor(std::initializer_list<VideoCore::SurfaceId> touched) {
    // Every scene sets the dependency, which only records that it can be waited for.
    // Nothing waits: the surfaces a scene samples are read by its fragment stage, and
    // fragment processing runs scene by scene in order (libgxm Overview, Scene
    // Dependencies). SCE_GXM_SCENE_VERTEX_WAIT_FOR_DEPENDENCY is for a vertex program that
    // reads an earlier scene's output, which no program here does; set on every scene that
    // touched a surface still in flight, it only kept the tiler from overlapping the
    // previous scene's fragment work, once a frame on SM3DL.
    (void)touched;
    const u32 flags = SCE_GXM_SCENE_FRAGMENT_SET_DEPENDENCY;
    return flags;
}

void RasterizerGxm::BeginScene(Framebuffer* fb) {
    if (in_scene && scene_fb == fb) {
        return;
    }
    if (in_scene) {
        EndScene();
    }
    if (fb->RenderTarget() == nullptr) {
        return;
    }
    SceGxmContext* context = Device().context;
    // A deferred depth clear lands here: this scene starts from the background value and
    // loads nothing; the force-store at its end writes the result back, and the scenes
    // after it load again.
    if (SceGxmDepthStencilSurface* ds = fb->MutableDepthStencilSurface(); ds && fb->depth_id) {
        Surface& depth = res_cache.GetSurface(fb->depth_id);
        if (depth.pending_clear) {
            depth.pending_clear = false;
            sceGxmDepthStencilSurfaceSetBackgroundDepth(ds, depth.pending_depth);
            sceGxmDepthStencilSurfaceSetBackgroundStencil(ds, depth.pending_stencil);
            sceGxmDepthStencilSurfaceSetForceLoadMode(ds,
                                                      SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
        } else {
            sceGxmDepthStencilSurfaceSetForceLoadMode(ds,
                                                      SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
        }
    }
    const u32 flags = SceneFlagsFor({fb->color_id, fb->depth_id, bound_surfaces[0],
                                     bound_surfaces[1], bound_surfaces[2]});
    // The serial is claimed before the scene opens so the surfaces it uses can be marked with
    // it as they are bound; the notification goes to sceGxmEndScene.
    const u64 serial = runtime.BeginSceneEpoch(&scene_notification);
    GXM_PHASE("gpu:begin-scene");
    if (sceGxmBeginScene(context, flags, fb->RenderTarget(), nullptr, nullptr, nullptr,
                         fb->ColorSurface(), fb->DepthStencilSurface()) < 0) {
        LOG_ERROR(Render, "sceGxmBeginScene failed ({}x{} colour {} depth {})", fb->Width(),
                  fb->Height(), fb->HasColor(), fb->HasDepth());
        return;
    }
    in_scene = true;
    scene_fb = fb;
    if (scenes_this_frame < MaxSceneRecords) {
        scene_records[scenes_this_frame] = {
            fb->color_id ? res_cache.GetSurface(fb->color_id).addr : 0,
            static_cast<u16>(fb->Width()), static_cast<u16>(fb->Height()), 0, false};
    }
    OpenSceneTrace(fb->color_id ? res_cache.GetSurface(fb->color_id).addr : 0,
                   static_cast<u16>(fb->Width()), static_cast<u16>(fb->Height()), false);
    scenes_this_frame++;
    // Bring-up switch: which surface each scene draws into. Two screens that end up in one
    // surface put both pictures on top of each other, which reads as corruption rather than
    // as the aliasing it is.
    static const bool log_scenes = GxmFlag("logdraw");
    if (log_scenes) {
        static u32 scenes = 0;
        static const u8* last_target = nullptr;
        scenes++;
        if (fb->ColorData() != last_target) {
            last_target = fb->ColorData();
            LOG_INFO(Render, "scene {}: colour {} {}x{}, depth {}", scenes,
                     static_cast<const void*>(fb->ColorData()), fb->Width(), fb->Height(),
                     fb->HasDepth());
        }
    }
    // The attachments are written by this scene for as long as its fragment processing runs.
    for (const VideoCore::SurfaceId id : {fb->color_id, fb->depth_id}) {
        if (id) {
            res_cache.GetSurface(id).gpu_serial = serial;
        }
    }
    stat_scenes++;
    pipeline_cache->InvalidateBinding();
    // BeginScene resets the viewport and the region clip to the render target, so nothing
    // this rasterizer set before it still holds.
    state_cache = CachedState{};
    // The LUT units, and the framebuffer-fetch unit (white until P5c); a texture binding
    // persists indefinitely, so this belongs once a scene rather than once a draw.
    BindFragmentTexture(3, lut_lf.texture);
    BindFragmentTexture(4, lut_rg.texture);
    BindFragmentTexture(5, lut_rgba.texture);
    for (u32 unit = 6; unit < 8; unit++) {
        BindFragmentTexture(unit, white_texture);
    }
    if (fb->color_id && !res_cache.GetSurface(fb->color_id).pending_colour.empty()) {
        DrawPendingColourClears();
    }
}

void RasterizerGxm::DrawPendingColourClears() {
    Surface& surface = res_cache.GetSurface(scene_fb->color_id);
    std::vector<Surface::PendingColourClear> pending;
    pending.swap(surface.pending_colour);
    SceGxmContext* context = Device().context;
    const SceGxmProgramParameter* colour_param = fill_fragment_shader->Param("fill_colour");
    constexpr u32 QuadVertices = 6;
    for (const auto& clear : pending) {
        const u32 w = clear.rect.GetWidth();
        const u32 h = clear.rect.GetHeight();
        const u32 offset = (blit_ring_used + 15u) & ~15u;
        if (w == 0 || h == 0 || offset + QuadVertices * sizeof(BlitVertex) > blit_ring_limit) {
            continue;
        }
        blit_ring_used = offset + QuadVertices * sizeof(BlitVertex);
        auto* quad = reinterpret_cast<BlitVertex*>(blit_ring + offset);
        // Clip space over a viewport set to the rectangle, as the conversion blit does: the
        // quad's edges are the rectangle's, whatever tile the region clip rounds to.
        const BlitVertex bl{-1.0f, -1.0f, 0.0f, 0.0f};
        const BlitVertex br{1.0f, -1.0f, 1.0f, 0.0f};
        const BlitVertex tl{-1.0f, 1.0f, 0.0f, 1.0f};
        const BlitVertex tr{1.0f, 1.0f, 1.0f, 1.0f};
        quad[0] = bl;
        quad[1] = br;
        quad[2] = tl;
        quad[3] = br;
        quad[4] = tr;
        quad[5] = tl;

        PipelineInfo info{};
        info.vertex = blit_vertex_shader;
        info.fragment = fill_fragment_shader;
        info.stream_count = 1;
        info.strides[0] = sizeof(BlitVertex);
        info.attribute_count = 2;
        info.attributes[0] = {"vert_position", 0, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2, 0};
        info.attributes[1] = {"vert_texcoord", 8, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2, 0};
        info.blend_enabled = true;
        info.blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
        info.blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
        info.blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
        info.blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
        info.blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
        info.blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
        info.blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
        if (!pipeline_cache->BindPipeline(info)) {
            continue;
        }
        sceGxmSetViewportEnable(context, SCE_GXM_VIEWPORT_ENABLED);
        sceGxmSetViewport(context, clear.rect.left + w * 0.5f, w * 0.5f,
                          clear.rect.bottom + h * 0.5f, h * 0.5f, 0.0f, 1.0f);
        sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE, clear.rect.left,
                            clear.rect.bottom, clear.rect.right - 1, clear.rect.top - 1);
        sceGxmSetCullMode(context, SCE_GXM_CULL_NONE);
        sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
        sceGxmSetBackDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
        sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
        sceGxmSetBackDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
        for (int face = 0; face < 2; face++) {
            const auto set = face == 0 ? sceGxmSetFrontStencilFunc : sceGxmSetBackStencilFunc;
            set(context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP,
                SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
        }
        void* fbuf = nullptr;
        if (sceGxmReserveFragmentDefaultUniformBuffer(context, &fbuf) >= 0 && fbuf != nullptr) {
            const float colour[4] = {clear.color.x, clear.color.y, clear.color.z, clear.color.w};
            SetUniformF(fbuf, MakeSlot(colour_param), colour, 4, "fill");
        }
        sceGxmSetVertexStream(context, 0, quad);
        const int err = sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                                   identity_indices, QuadVertices);
        if (err < 0) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_ERROR(Render, "GXM: the clear quad was refused: {} ({:#x})",
                          GxmErrorName(err), static_cast<u32>(err));
            }
        }
    }
    // Everything the quads set is PICA state the next draw must set again.
    pipeline_cache->InvalidateBinding();
    state_cache = CachedState{};
    uniform_pipeline = nullptr;
}

bool RasterizerGxm::ReserveLutRows(LutTexture& lut, u32 needed) {
    if (needed == 0) {
        return false;
    }
    bool wrapped = false;
    if (lut.row_cursor + needed > lut.rows) {
        lut.row_cursor = 0;
        wrapped = true;
    }
    const u32 half_rows = lut.rows / 2;
    const u32 last_half = std::min((lut.row_cursor + needed - 1) / half_rows, 1u);
    if (last_half != lut.half) {
        // The rows of the half being left were handed to programs of the scene that is open
        // (or about to open, when the upload comes before its BeginScene): nothing later
        // samples them. The half being entered was left that way a ring-half ago.
        lut.left_serial[lut.half] = runtime.CurrentEpoch() + 1;
        lut.half = last_half;
        const u64 wanted = lut.left_serial[last_half];
        if (wanted != 0 && runtime.CompletedEpoch() < wanted) {
            if (in_scene) {
                EndScene();
            }
            runtime.WaitForEpoch(wanted);
            stat_lut_waits++;
        }
    }
    return wrapped;
}

void RasterizerGxm::SyncAndUploadLUTs() {
    if (!pica.lighting.lut_dirty && !pica.fog.lut_dirty && !pica.proctex.table_dirty) {
        return;
    }
    using namespace Pica::Shader;
    // Row allocation is round-robin, as in the GL backend's 2D path: a wrap hands out rows
    // whose offsets other, non-dirty LUTs still point at, so a wrap restarts the ring and
    // re-uploads every LUT of that texture. The ring is recycled in halves (ReserveLutRows),
    // so no row is rewritten while a scene may still sample it and, with 512 rows against
    // the two dozen a lighting change dirties, the wait for the half's last scene is over
    // before it starts. The full GPU drain a wrap used to do came once every two frames on
    // SM3DL's title scene (128 rows), and broke the scene it landed in.
    const u32 lf_needed = static_cast<u32>(std::popcount(pica.lighting.lut_dirty)) +
                          (pica.fog.lut_dirty ? 1u : 0u);
    const u32 rg_needed = (pica.proctex.noise_lut_dirty ? 1u : 0u) +
                          (pica.proctex.color_map_dirty ? 1u : 0u) +
                          (pica.proctex.alpha_map_dirty ? 1u : 0u);
    const u32 rgba_needed =
        (pica.proctex.lut_dirty ? 1u : 0u) + (pica.proctex.diff_lut_dirty ? 1u : 0u);
    if (ReserveLutRows(lut_lf, lf_needed)) {
        pica.lighting.lut_dirty = pica.lighting.LutAllDirty;
        pica.fog.lut_dirty = true;
        // The wrap needs every row again.
        ReserveLutRows(lut_lf, static_cast<u32>(std::popcount(pica.lighting.lut_dirty)) + 1);
    }
    const bool rg_wrapped = ReserveLutRows(lut_rg, rg_needed);
    const bool rgba_wrapped = ReserveLutRows(lut_rgba, rgba_needed);
    if (rg_wrapped || rgba_wrapped) {
        lut_rg.row_cursor = 0;
        lut_rgba.row_cursor = 0;
        pica.proctex.table_dirty = pica.proctex.TableAllDirty;
        ReserveLutRows(lut_rg, 3);
        ReserveLutRows(lut_rgba, 2);
    }
    // A row is written and its index handed back as the LUT's offset in texels, which the
    // fragment program turns back into a row.
    const auto upload_row = [](LutTexture& lut, const void* data, u32 texel_count) -> int {
        const u32 row = lut.row_cursor++;
        u8* dst = static_cast<u8*>(lut.mem) + static_cast<std::size_t>(row) * LUT_TEX_WIDTH *
                                                  lut.texel_bytes;
        std::memcpy(dst, data, static_cast<std::size_t>(texel_count) * lut.texel_bytes);
        return static_cast<int>(row * LUT_TEX_WIDTH);
    };
    std::array<Common::Vec2f, 256> scratch2;
    while (pica.lighting.lut_dirty) {
        const u32 index = std::countr_zero(pica.lighting.lut_dirty);
        pica.lighting.lut_dirty &= ~(1u << index);
        const auto& source = pica.lighting.luts[index];
        for (u32 i = 0; i < source.size(); i++) {
            scratch2[i] = {source[i].ToFloat(), source[i].DiffToFloat()};
        }
        fs_data.lighting_lut_offset[index / 4][index % 4] =
            upload_row(lut_lf, scratch2.data(), static_cast<u32>(source.size()));
        fs_data_dirty = true;
    }
    if (pica.fog.lut_dirty) {
        for (u32 i = 0; i < pica.fog.lut.size(); i++) {
            scratch2[i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
        }
        fs_data.fog_lut_offset =
            upload_row(lut_lf, scratch2.data(), static_cast<u32>(pica.fog.lut.size()));
        fs_data_dirty = true;
        pica.fog.lut_dirty = false;
    }
    if (pica.proctex.table_dirty) {
        const auto value_lut = [&](const auto& lut, int& lut_offset) {
            for (u32 i = 0; i < lut.size(); i++) {
                scratch2[i] = {lut[i].ToFloat(), lut[i].DiffToFloat()};
            }
            lut_offset = upload_row(lut_rg, scratch2.data(), static_cast<u32>(lut.size()));
            fs_data_dirty = true;
        };
        if (pica.proctex.noise_lut_dirty) {
            value_lut(pica.proctex.noise_table, fs_data.proctex_noise_lut_offset);
        }
        if (pica.proctex.color_map_dirty) {
            value_lut(pica.proctex.color_map_table, fs_data.proctex_color_map_offset);
        }
        if (pica.proctex.alpha_map_dirty) {
            value_lut(pica.proctex.alpha_map_table, fs_data.proctex_alpha_map_offset);
        }
        // The colour tables are bytes already, in the texture's own order.
        static_assert(sizeof(pica.proctex.color_table[0]) == 4);
        if (pica.proctex.lut_dirty) {
            fs_data.proctex_lut_offset =
                upload_row(lut_rgba, pica.proctex.color_table.data(),
                           static_cast<u32>(pica.proctex.color_table.size()));
            fs_data_dirty = true;
        }
        if (pica.proctex.diff_lut_dirty) {
            fs_data.proctex_diff_lut_offset =
                upload_row(lut_rgba, pica.proctex.color_diff_table.data(),
                           static_cast<u32>(pica.proctex.color_diff_table.size()));
            fs_data_dirty = true;
        }
        pica.proctex.table_dirty = 0;
    }
}

/**
 * Gives up the surfaces the cache has not used for longest, when the video memory pool is
 * running out.
 *
 * Nothing else evicts. Upstream a surface leaves the cache when the guest writes over its
 * memory, but this port never sees the guest's writes - the render thread must not flip page
 * attributes under the emulation thread, and the native CPU backend traps no stores - so the
 * cache keeps every surface a title ever makes. A desktop driver pages that out to system
 * memory; the console has a fixed pool and simply runs out. Super Smash Bros reached 419
 * surfaces holding 51.5 MB of 56 MB after two minutes of a fight, and the allocation that
 * could not be satisfied then parked the render thread.
 *
 * A surface's gpu_serial is the scene that last used it, so it is already the least-recently-
 * used key. Dirty-region owners and surfaces used this tick are protected by the shared
 * cache. Anything the GPU may still be holding, or that
 * the presentation layer is sampling, is skipped. This runs at the end of a frame, with no
 * scene open and nothing bound.
 */
void RasterizerGxm::TrimSurfaceCache() {
    // ON BY DEFAULT since 2026-09-07 (`notrim` turns it off): a Smash fight on the console
    // froze with the surface cache at 60954 of 65280 KiB and still growing (443 surfaces,
    // no trim), the render thread stopped inside GXM while the guest ran on at 100%. The
    // reload after a trim is proven under GL on the pi5 (trimmed to 8 MB every frame, every
    // texture reloads, the picture is identical); Vita3K's blank stage after a trim is its
    // readback limit, not the trim. The trim lives in the shared cache
    // (RasterizerCache::TrimSurfaces, least recently used first, never a dirty-region owner),
    // where AZAHAR_SURFACE_BUDGET_MB and AZAHAR_TRIM_LOG (CITRA_TRACE_PROBES builds) trace it.
    static const bool trim = !GxmFlag("notrim");
    if (!trim || !resources_ready) {
        return;
    }
    // Start well before the memory is gone: a surface goes back to the pool only once the
    // collector erases it, a few ticks after it is sentenced, and the pool needs a contiguous
    // run inside one of its chunks, so the last megabytes are the least usable ones.
    constexpr u32 TrimBelow = 12 * 1024 * 1024;
    constexpr u32 TrimUntil = 24 * 1024 * 1024;
    // smallcache: pretend the pool ends at 40 MB, so this runs under Vita3K, where the device
    // has memory to spare and the console's ceiling is never reached.
    static const bool small_cache = GxmFlag("smallcache");
    // 16 MB: below what a fight holds once stale textures are dropped, so it trims every frame.
    constexpr u32 SmallCeiling = 16 * 1024 * 1024;
    const u32 used = runtime.CdramPool().Used();
    const u32 headroom =
        small_cache ? (used < SmallCeiling ? SmallCeiling - used : 0) : runtime.CdramHeadroom();
    if (headroom >= TrimBelow) {
        return;
    }
    const u64 completed = runtime.CompletedEpoch();
    stat_trimmed += res_cache.TrimSurfaces(
        TrimUntil - headroom, [](const Surface& surface) { return surface.AllocSize(); },
        [&](const Surface& surface) {
            // IsColorTarget describes format capability, not usage: ordinary RGB/RGBA
            // textures have color-surface descriptors too. Exempting them makes them
            // immortal. TrimSurfaces already protects dirty owners and current-tick uses;
            // clean old targets can be recreated from guest memory like other textures.
            return surface.presented || !surface.HasMemory() || surface.gpu_serial > completed;
        });
}

void RasterizerGxm::EndScene() {
    DrawTrace("end scene");
    if (!in_scene) {
        return;
    }
    // The fragment notification is what turns "is the GPU done with this surface" into a
    // load: it lands in the notification region once this scene's fragment processing ends.
    {
        const PhaseTimer timer{time_draws, times.scene_end};
        GXM_PHASE("gpu:end-scene");
        sceGxmEndScene(Device().context, nullptr,
                       scene_notification.address != nullptr ? &scene_notification : nullptr);
    }
    // The scene is now the GPU's, so its notification will arrive and a wait for it can end.
    runtime.SceneSubmitted();
    CloseSceneTrace();
    if (ring_scene_pos != 0) {
        ring_retire.push_back({runtime.CurrentEpoch(), ring_scene_pos});
        ring_scene_pos = 0;
    }
    in_scene = false;
    scene_fb = nullptr;
}

void RasterizerGxm::SyncState(const VideoCore::FramebufferHelper<Traits>& helper) {
    SceGxmContext* context = Device().context;
    const Framebuffer* fb = helper.Framebuffer();
    const auto& fb_regs = regs.framebuffer;
    const auto& om = fb_regs.output_merger;
    const bool is_flipped = fb_regs.framebuffer.IsFlipped();

    // Viewport and draw rectangle from the helper, in the surface's own space (a framebuffer
    // may be a sub-rectangle of a bigger cached surface). Rows are stored bottom-up, so the
    // y scale is positive: NDC +1 lands on the highest row index, the top of the picture.
    const auto vp = helper.Viewport();
    const float viewport[6] = {static_cast<float>(vp.x) + vp.width * 0.5f, vp.width * 0.5f,
                               static_cast<float>(vp.y) + vp.height * 0.5f, vp.height * 0.5f,
                               0.0f, 1.0f};
    if (!state_cache.valid || std::memcmp(state_cache.vp, viewport, sizeof(viewport)) != 0) {
        std::memcpy(state_cache.vp, viewport, sizeof(viewport));
        sceGxmSetViewportEnable(context, SCE_GXM_VIEWPORT_ENABLED);
        sceGxmSetViewport(context, viewport[0], viewport[1], viewport[2], viewport[3],
                          viewport[4], viewport[5]);
    }
    const auto draw = helper.DrawRect();
    const u32 fbw = fb->Width();
    const u32 fbh = fb->Height();
    const u32 cx0 = std::min(draw.left, fbw), cx1 = std::min(draw.right, fbw);
    const u32 cy0 = std::min(draw.bottom, fbh), cy1 = std::min(draw.top, fbh);
    const bool clip_to_draw = cx1 > cx0 && cy1 > cy0;
    const u32 clip[4] = {clip_to_draw ? cx0 : 0u, clip_to_draw ? cy0 : 0u,
                         clip_to_draw ? cx1 - 1 : fbw - 1, clip_to_draw ? cy1 - 1 : fbh - 1};
    if (!state_cache.valid || std::memcmp(state_cache.clip, clip, sizeof(clip)) != 0) {
        std::memcpy(state_cache.clip, clip, sizeof(clip));
        sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE, clip[0], clip[1], clip[2],
                            clip[3]);
    }
    // Scissor uniforms (the fragment program's discard test), window-relative like GL's.
    const auto [scissor_x1, scissor_y2, scissor_x2, scissor_y1] = helper.Scissor();
    // Most titles leave the scissor test on with the whole screen as its rectangle. The
    // region clip above already confines the draw to that, and a kill in the program
    // costs the tiler its hidden-surface removal (the phase before the kill runs for every
    // fragment, visible or not), so such a scissor test is left out of the program.
    scissor_covers_draw =
        regs.rasterizer.scissor_test.mode == Pica::RasterizerRegs::ScissorMode::Include &&
        scissor_x1 <= static_cast<s32>(clip[0]) && scissor_y1 <= static_cast<s32>(clip[1]) &&
        scissor_x2 - 1 >= static_cast<s32>(clip[2]) && scissor_y2 - 1 >= static_cast<s32>(clip[3]);
    if (fs_data.scissor_x1 != scissor_x1 || fs_data.scissor_x2 != scissor_x2 ||
        fs_data.scissor_y1 != scissor_y1 || fs_data.scissor_y2 != scissor_y2) {
        fs_data.scissor_x1 = scissor_x1;
        fs_data.scissor_x2 = scissor_x2;
        fs_data.scissor_y1 = scissor_y1;
        fs_data.scissor_y2 = scissor_y2;
        fs_data_dirty = true;
    }

    // One draw's state per 120-frame report, when the switch is on: the first draw of a
    // window, which is the frame's first and the one whose target every later draw follows.
    static const bool log_draw = GxmFlag("logdraw");
    static u32 draw_logs = 0;
    if (log_draw && (stat_draws == 0 || draw_logs++ < 200)) {
        LOG_INFO(Render,
                 "draw: target {}x{} colour {} depth {} flipped {} viewport {},{} {}x{} clip "
                 "{},{}-{},{} cull {} depth test {} func {} write {} blend {} alpha test {} "
                 "func {} colour mask {}{}{}{} scissor mode {} depth range scale {} offset {} "
                 "depth addr {:08X} allow ds write {} colour addr {:08X}",
                 fbw, fbh, fb->HasColor(), fb->HasDepth(), is_flipped, vp.x, vp.y, vp.width,
                 vp.height, cx0, cy0, cx1, cy1,
                 static_cast<int>(regs.rasterizer.cull_mode.Value()),
                 om.depth_test_enable.Value(), static_cast<int>(om.depth_test_func.Value()),
                 om.depth_write_enable.Value(), om.alphablend_enable.Value(),
                 om.alpha_test.enable.Value(), static_cast<int>(om.alpha_test.func.Value()),
                 om.red_enable.Value(), om.green_enable.Value(), om.blue_enable.Value(),
                 om.alpha_enable.Value(),
                 static_cast<int>(regs.rasterizer.scissor_test.mode.Value()),
                 Pica::f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32(),
                 Pica::f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32(),
                 fb_regs.framebuffer.GetDepthBufferPhysicalAddress(),
                 fb_regs.framebuffer.allow_depth_stencil_write.Value(),
                 fb_regs.framebuffer.GetColorBufferPhysicalAddress());
    }

    // Culling: PICA names the winding it keeps; GXM names the winding it drops, judged in its
    // own y-down screen space. Rows are stored bottom-up here (the GL order), so a winding
    // that is clockwise in GL's y-up window is counter-clockwise to GXM: keeping PICA's
    // clockwise faces means dropping what GXM calls clockwise. The framebuffer flip mirrors
    // the raster once more and swaps it back.
    using CullMode = Pica::RasterizerRegs::CullMode;
    const auto cull = regs.rasterizer.cull_mode;
    SceGxmCullMode mode = SCE_GXM_CULL_NONE;
    if (cull == CullMode::KeepClockWise) {
        mode = is_flipped ? SCE_GXM_CULL_CCW : SCE_GXM_CULL_CW;
    } else if (cull == CullMode::KeepCounterClockWise) {
        mode = is_flipped ? SCE_GXM_CULL_CW : SCE_GXM_CULL_CCW;
    }
    if (!state_cache.valid || state_cache.cull != mode) {
        state_cache.cull = mode;
        sceGxmSetCullMode(context, mode);
    }

    // Depth: both faces the same, since culling may be off.
    const bool has_ds = fb->HasDepth();
    const bool depth_write = fb_regs.framebuffer.allow_depth_stencil_write != 0 &&
                             om.depth_write_enable == 1 && has_ds;
    // A write without a test is a test that always passes (see BeginDraw).
    const bool depth_test = om.depth_test_enable == 1 && has_ds;
    const SceGxmDepthFunc depth_func =
        depth_test ? PicaToGxm::CompareFunc(om.depth_test_func) : SCE_GXM_DEPTH_FUNC_ALWAYS;
    const auto write_mode =
        depth_write ? SCE_GXM_DEPTH_WRITE_ENABLED : SCE_GXM_DEPTH_WRITE_DISABLED;
    if (!state_cache.valid || state_cache.depth_func != depth_func) {
        state_cache.depth_func = depth_func;
        sceGxmSetFrontDepthFunc(context, depth_func);
        sceGxmSetBackDepthFunc(context, depth_func);
    }
    if (!state_cache.valid || state_cache.depth_write != write_mode) {
        state_cache.depth_write = write_mode;
        sceGxmSetFrontDepthWriteEnable(context, write_mode);
        sceGxmSetBackDepthWriteEnable(context, write_mode);
    }

    // Stencil, when the framebuffer has one.
    const bool has_stencil = fb_regs.HasStencil() && has_ds;
    const auto& st = om.stencil_test;
    if (has_stencil && st.enable) {
        const u8 write_mask = fb_regs.framebuffer.allow_depth_stencil_write != 0
                                  ? static_cast<u8>(st.write_mask)
                                  : 0u;
        const auto func = PicaToGxm::StencilFunc(st.func);
        const auto fail = PicaToGxm::StencilOp(st.action_stencil_fail);
        const auto zfail = PicaToGxm::StencilOp(st.action_depth_fail);
        const auto zpass = PicaToGxm::StencilOp(st.action_depth_pass);
        const u8 cmp_mask = static_cast<u8>(st.input_mask);
        SetStencil(func, fail, zfail, zpass, cmp_mask, write_mask,
                   static_cast<u8>(st.reference_value));
    } else {
        SetStencil(SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
                   SCE_GXM_STENCIL_OP_KEEP, 0, 0, 0);
    }
    state_cache.valid = true;
}

void RasterizerGxm::SetStencil(SceGxmStencilFunc func, SceGxmStencilOp fail, SceGxmStencilOp zfail,
                               SceGxmStencilOp zpass, u8 compare_mask, u8 write_mask, u8 ref) {
    SceGxmContext* context = Device().context;
    auto& c = state_cache;
    if (!c.valid || !c.stencil_set || c.stencil_func != func || c.stencil_fail != fail ||
        c.stencil_zfail != zfail || c.stencil_zpass != zpass || c.stencil_compare != compare_mask ||
        c.stencil_write != write_mask) {
        c.stencil_set = true;
        c.stencil_func = func;
        c.stencil_fail = fail;
        c.stencil_zfail = zfail;
        c.stencil_zpass = zpass;
        c.stencil_compare = compare_mask;
        c.stencil_write = write_mask;
        sceGxmSetFrontStencilFunc(context, func, fail, zfail, zpass, compare_mask, write_mask);
        sceGxmSetBackStencilFunc(context, func, fail, zfail, zpass, compare_mask, write_mask);
    }
    if (!c.valid || c.stencil_ref != ref) {
        c.stencil_ref = ref;
        sceGxmSetFrontStencilRef(context, ref);
        sceGxmSetBackStencilRef(context, ref);
    }
}

bool RasterizerGxm::BindPrograms(const VertexLayout& layout, u32 vertex_count) {
    const auto& om = regs.framebuffer.output_merger;
    const auto& fb_regs = regs.framebuffer.framebuffer;

    // The fragment program for this PICA configuration, compiled on the worker if new.
    Pica::Shader::FSConfig fs_config{regs};
    // These comparisons cannot reject an 8-bit alpha. Reuse the Always shader before
    // hashing so the USSE emitter omits the alpha kill and its GXP discard flag (unless
    // scissoring still needs it), without putting the reference value in the shader key.
    using CompareFunc = Pica::FramebufferRegs::CompareFunc;
    if ((fs_config.framebuffer.alpha_test_func == CompareFunc::GreaterThanOrEqual &&
         om.alpha_test.ref == 0) ||
        (fs_config.framebuffer.alpha_test_func == CompareFunc::LessThanOrEqual &&
         om.alpha_test.ref == 255)) {
        fs_config.framebuffer.alpha_test_func.Assign(CompareFunc::Always);
    }
    if (scissor_covers_draw && fs_config.framebuffer.scissor_test_mode ==
                                   Pica::RasterizerRegs::ScissorMode::Include) {
        fs_config.framebuffer.scissor_test_mode.Assign(
            Pica::RasterizerRegs::ScissorMode::Disabled);
        stat_scissor_free++;
    }
    // A debug output is a different program, and must not be what the disk cache hands
    // back for the real one.
    const u64 fs_hash = Common::HashCombine(
        static_cast<u64>(fs_config.Hash()),
        static_cast<u64>(Pica::Shader::Generator::Cg::GetDebugOutput()) |
            (Pica::Shader::Generator::Cg::FogDisabled() ? 0x100ull : 0ull));
    // Only generate the Cg when the configuration is new. Both arguments below cost real time
    // - the source is hundreds of string appends and the name is an allocation - and they
    // were being paid on every draw for a shader that had existed since the first one.
    // One program per configuration: the emitter's, ready the moment it is made. What it
    // refuses is skipped. SceShaccCg used to compile the refused ones, and for a while a
    // second, better tier of every configuration, but at 3-5 s a program on a core shared
    // with libgxm's threads it stalled the GPU for as long as it ran (2026-09-05), and
    // that was not worth a third fewer cycles; a better program has to come from a better
    // emitter. The generated Cg still exists for psp2cgc off the console (usse_dump).
    Shader* fs = pipeline_cache->FindShader(fs_hash);
    if (fs == nullptr) {
        std::string refusal;
        std::vector<u8> program = Usse::EmitFragmentProgram(fs_config, &refusal);
        if (program.empty()) {
            if (usse_refused.insert(fs_hash).second) {
                LOG_WARNING(Render, "no fragment program for fs_{:016x}: {}", fs_hash, refusal);
                stat_usse_refused++;
            }
            return false;
        }
        fs = pipeline_cache->UseBuiltinShader(fs_hash, fmt::format("us_{:016x}", fs_hash),
                                              program);
        stat_usse_emitted++;
        static const bool dump = GxmFlag("usse_dump");
        if (dump) {
            sceIoMkdir("ux0:data/azahar/usse", 0777);
            const std::string stem = fmt::format("ux0:data/azahar/usse/us_{:016x}", fs_hash);
            if (FILE* f = std::fopen((stem + ".gxp").c_str(), "wb")) {
                std::fwrite(program.data(), 1, program.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((stem + ".cg").c_str(), "wb")) {
                // The same configuration as Cg, so psp2cgc can compile the twin.
                const std::string cg = Pica::Shader::Generator::Cg::GenerateFragmentShader(
                    fs_config, user_config, profile);
                std::fwrite(cg.data(), 1, cg.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((stem + ".fq").c_str(), "wb")) {
                // The configuration itself, for the host emitter (usse_tests/fq_emit).
                std::fwrite(&fs_config, 1, sizeof(fs_config), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen((stem + ".txt").c_str(), "wb")) {
                const std::string info = fmt::format(
                    "scissor {} alpha_test {} fog {} lighting {} lights {} config {} "
                    "tex0_type {} logic {}\n",
                    static_cast<u32>(fs_config.framebuffer.scissor_test_mode.Value()),
                    static_cast<u32>(fs_config.framebuffer.alpha_test_func.Value()),
                    static_cast<u32>(fs_config.texture.fog_mode.Value()),
                    static_cast<u32>(fs_config.lighting.enable.Value()),
                    static_cast<u32>(fs_config.lighting.src_num.Value()),
                    static_cast<u32>(fs_config.lighting.config.Value()),
                    static_cast<u32>(fs_config.texture.texture0_type.Value()),
                    static_cast<u32>(fs_config.framebuffer.logic_op.Value()));
                std::fwrite(info.data(), 1, info.size(), f);
                std::fclose(f);
            }
        }
    }

    PipelineInfo info{};
    ConsiderTier2(fs, fs_hash, &fs_config, nullptr);
    info.vertex = layout.shader;
    info.fragment = fs;
    info.stream_count = layout.stream_count;
    info.strides = layout.strides;
    info.attribute_count = layout.attribute_count;
    info.attributes = layout.attributes;

    // Blend and colour mask are baked into the fragment program (GXM has no dynamic blend
    // state), so they are part of the pipeline key.
    info.blend_enabled = true;
    SceGxmBlendInfo& blend = info.blend;
    const auto mask = [&](u32 enable, u8 bit) -> u8 {
        return fb_regs.allow_color_write != 0 && enable != 0 ? bit : 0u;
    };
    blend.colorMask =
        mask(om.red_enable, SCE_GXM_COLOR_MASK_R) | mask(om.green_enable, SCE_GXM_COLOR_MASK_G) |
        mask(om.blue_enable, SCE_GXM_COLOR_MASK_B) | mask(om.alpha_enable, SCE_GXM_COLOR_MASK_A);
    // Bring-up switch: every draw written straight to the target. A picture that comes right
    // without blending puts the fault in the blend or in what it reads back, not in what the
    // shader computed.
    if (om.alphablend_enable == 1) {
        blend.colorFunc = PicaToGxm::BlendEquation(om.alpha_blending.blend_equation_rgb);
        blend.alphaFunc = PicaToGxm::BlendEquation(om.alpha_blending.blend_equation_a);
        blend.colorSrc = PicaToGxm::BlendFactor(om.alpha_blending.factor_source_rgb);
        blend.colorDst = PicaToGxm::BlendFactor(om.alpha_blending.factor_dest_rgb);
        blend.alphaSrc = PicaToGxm::BlendFactor(om.alpha_blending.factor_source_a);
        blend.alphaDst = PicaToGxm::BlendFactor(om.alpha_blending.factor_dest_a);
    } else {
        blend.colorFunc = SCE_GXM_BLEND_FUNC_NONE;
        blend.alphaFunc = SCE_GXM_BLEND_FUNC_NONE;
        blend.colorSrc = SCE_GXM_BLEND_FACTOR_ONE;
        blend.colorDst = SCE_GXM_BLEND_FACTOR_ZERO;
        blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
        blend.alphaDst = SCE_GXM_BLEND_FACTOR_ZERO;
    }

    // Waiting here is a whole SceShaccCg compile on the render thread - tens of milliseconds
    // for a fragment program, seconds for a generated vertex program, once per PICA
    // configuration a title reaches. Measured at 48% of the wall while Super Mario 3D Land
    // was still finding its fragment programs, and with vertex programs in the mix the
    // exception the Vulkan backend makes for small draws (menu and HUD quads, exactly six
    // vertices) blocked the render thread for nine seconds on a title screen made of quads,
    // during which the queue grew past three thousand ops and the heap ran out. So nothing
    // waits: a draw whose programs are not ready is skipped, whatever its size, and this
    // backend does not consult Settings::async_shader_compilation either - there is no
    // configuration of this hardware in which blocking on a compile is the better choice.
    // The disk cache makes it a first-run cost.
    (void)vertex_count;
    DrawTrace("bind pipeline");
    if (!pipeline_cache->BindPipeline(info)) {
        return false;
    }
    NoteScenePipeline(info.vertex != nullptr ? info.vertex->name.c_str() : "?",
                      info.fragment != nullptr ? info.fragment->name.c_str() : "?");
    DrawTrace("pipeline bound");
    // Name the draw in a Razor capture: without this every draw in the host tool is an
    // anonymous index, and the whole reason to capture is to find out which PICA
    // configuration produced the pixels being looked at. Inert unless a capture is being
    // written, but the marker only reaches one at all inside a scene - which this is - and
    // the name is only worth building when the capture module is there to read it.
    if (Device().razor) {
        sceGxmSetUserMarker(Device().context, fs->name.c_str());
    }
    {
        const PhaseTimer timer{time_draws, times.uniforms};
        DrawTrace("uniforms");
        UploadUniforms(fs, layout.shader, fs_hash);
        DrawTrace("uniforms done");
    }
    return true;
}

void RasterizerGxm::ResolveVertexUniforms() {
    if (vertex_shader == nullptr || !vertex_shader->IsDone() || vertex_shader->failed) {
        return;
    }
    vs_flip_viewport = MakeSlot(vertex_shader->Param("flip_viewport"));
    vs_depth_scale = MakeSlot(vertex_shader->Param("depth_scale"));
    vs_depth_offset = MakeSlot(vertex_shader->Param("depth_offset"));
    vs_uniforms_resolved = true;
}

const RasterizerGxm::FsUniforms& RasterizerGxm::UniformsOf(Shader* fs) {
    if (fs == cached_fs) {
        return *cached_locs;
    }
    auto [it, inserted] = fs_uniforms.try_emplace(fs);
    FsUniforms& u = it->second;
    if (inserted) {
        const auto p = [fs](const char* name) { return MakeSlot(fs->Param(name)); };
        u.alphatest_ref = p("alphatest_ref");
        u.depth_scale = p("depth_scale");
        u.depth_offset = p("depth_offset");
        u.scissor_x1 = p("scissor_x1");
        u.scissor_y1 = p("scissor_y1");
        u.scissor_x2 = p("scissor_x2");
        u.scissor_y2 = p("scissor_y2");
        u.fog_lut_offset = p("fog_lut_offset");
        u.proctex_noise_lut_offset = p("proctex_noise_lut_offset");
        u.proctex_color_map_offset = p("proctex_color_map_offset");
        u.proctex_alpha_map_offset = p("proctex_alpha_map_offset");
        u.proctex_lut_offset = p("proctex_lut_offset");
        u.proctex_diff_lut_offset = p("proctex_diff_lut_offset");
        u.proctex_bias = p("proctex_bias");
        u.lighting_lut_offset = p("lighting_lut_offset");
        u.fog_color = p("fog_color");
        u.proctex_noise_f = p("proctex_noise_f");
        u.proctex_noise_a = p("proctex_noise_a");
        u.proctex_noise_p = p("proctex_noise_p");
        u.lighting_global_ambient = p("lighting_global_ambient");
        u.const_color = p("const_color");
        u.tev_combiner_buffer_color = p("tev_combiner_buffer_color");
        u.tex_lod_bias = p("tex_lod_bias");
        u.tex_border_color = p("tex_border_color");
        u.blend_color = p("blend_color");
        u.tex_dims = p("tex_dims");
        u.fb_dims = p("fb_dims");
        for (u32 i = 0; i < FsUniforms::Lights; i++) {
            auto& light = u.light_src[i];
            const auto lp = [&](const char* member) {
                return MakeSlot(fs->Param(fmt::format("light_src[{}].{}", i, member).c_str()));
            };
            light.specular_0 = lp("specular_0");
            light.specular_1 = lp("specular_1");
            light.diffuse = lp("diffuse");
            light.ambient = lp("ambient");
            light.position = lp("position");
            light.spot_direction = lp("spot_direction");
            light.dist_atten_bias = lp("dist_atten_bias");
            light.dist_atten_scale = lp("dist_atten_scale");
        }
    }
    cached_fs = fs;
    cached_locs = &u;
    return u;
}

const RasterizerGxm::VsUniforms& RasterizerGxm::VsUniformsOf(Shader* vs) {
    auto [it, inserted] = vs_uniforms.try_emplace(vs);
    VsUniforms& u = it->second;
    if (inserted) {
        u.f = MakeSlot(vs->Param("uniforms_f"));
        if (GxmFlag("logdraw")) {
            const auto describe = [](const char* name, const SceGxmProgramParameter* p) {
                if (p == nullptr) {
                    LOG_INFO(Render, "vertex uniform '{}': absent", name);
                    return;
                }
                LOG_INFO(Render,
                         "vertex uniform '{}': array {}, components {}, container {}, resource "
                         "index {}, category {}",
                         name, sceGxmProgramParameterGetArraySize(p),
                         sceGxmProgramParameterGetComponentCount(p),
                         sceGxmProgramParameterGetContainerIndex(p),
                         sceGxmProgramParameterGetResourceIndex(p),
                         static_cast<u32>(sceGxmProgramParameterGetCategory(p)));
            };
            describe("uniforms_f", u.f.param);
            describe("uniforms_i", vs->Param("uniforms_i"));
            describe("flip_viewport", vs->Param("flip_viewport"));
        }
        u.i = MakeSlot(vs->Param("uniforms_i"));
        u.b = MakeSlot(vs->Param("uniforms_b"));
        u.flip_viewport = MakeSlot(vs->Param("flip_viewport"));
        u.depth_scale = MakeSlot(vs->Param("depth_scale"));
        u.depth_offset = MakeSlot(vs->Param("depth_offset"));
        for (u32 reg = 0; reg < 16; reg++) {
            u.default_regs[reg] = MakeSlot(vs->Param(fmt::format("vs_in_reg{}", reg).c_str()));
            u.any_default |= static_cast<bool>(u.default_regs[reg]);
        }
    }
    return u;
}

void RasterizerGxm::UploadUniforms(Shader* fs, Shader* vs,
                                   u64 fs_hash) {
    SceGxmContext* context = Device().context;
    if (!vs_uniforms_resolved) {
        ResolveVertexUniforms();
    }
    const bool generated = vs != vertex_shader;
    const VsUniforms* vsu = generated ? &VsUniformsOf(vs) : nullptr;

    // What the buffers already on the context were written for. A reservation survives until
    // the next one, so an unchanged draw reserves and writes nothing.
    const float fb_dims[2] = {static_cast<float>(scene_fb ? scene_fb->Width() : 1),
                              static_cast<float>(scene_fb ? scene_fb->Height() : 1)};
    const bool fb_dims_changed =
        fb_dims[0] != uniform_fb_dims[0] || fb_dims[1] != uniform_fb_dims[1];
    const bool rebind = pipeline_cache->CurrentPipeline() != uniform_pipeline ||
                        uniform_scene != runtime.CurrentEpoch();
    // A generated program also carries the PICA uniforms, which the guest rewrites between
    // most draws (its matrices live there), and the default attribute values.
    bool defaults_changed = false;
    if (vsu != nullptr && vsu->any_default &&
        std::memcmp(&pica.input_default_attributes, &uploaded_default_attrs,
                    sizeof(uploaded_default_attrs)) != 0) {
        defaults_changed = true;
    }
    const bool vertex_dirty = rebind || vs_data_dirty ||
                              (generated && (pica.vs_setup.uniforms_dirty || defaults_changed));
    (void)fs_hash;
    const bool fragment_dirty = rebind || fs_data_dirty || tex_dims_dirty || fb_dims_changed;
    if (!vertex_dirty && !fragment_dirty) {
        stat_uniform_skips++;
        return;
    }
    uniform_pipeline = pipeline_cache->CurrentPipeline();
    uniform_scene = runtime.CurrentEpoch();
    stat_uniform_uploads++;

    // Vertex: the buffer belongs to the program until the next reservation and must be
    // complete before the draw that uses it.
    if (vertex_dirty) {
        void* vbuf = nullptr;
        if (sceGxmReserveVertexDefaultUniformBuffer(context, &vbuf) >= 0 && vbuf != nullptr) {
            last_vertex_uniforms = static_cast<const u8*>(vbuf);
            const float flip = vs_data.flip_viewport ? 1.0f : 0.0f;
            if (vsu == nullptr) {
                SetUniformF(vbuf, vs_flip_viewport, &flip, 1, "vs");
                SetUniformF(vbuf, vs_depth_scale, &fs_data.depth_scale, 1, "vs");
                SetUniformF(vbuf, vs_depth_offset, &fs_data.depth_offset, 1, "vs");
            } else {
                const auto set = [&](const UniformSlot& slot, const float* data, u32 count) {
                    SetUniformF(vbuf, slot, data, count, "vs");
                };
                set(vsu->flip_viewport, &flip, 1);
                set(vsu->depth_scale, &fs_data.depth_scale, 1);
                set(vsu->depth_offset, &fs_data.depth_offset, 1);
                const auto& uniforms = pica.vs_setup.uniforms;
                if (vsu->f) {
                    // f24 to float, the whole bank: a program that indexes it through an
                    // address register can reach any of it. The bank is 128 entries with
                    // ones past the 96 real uniforms, which is what the PICA reads there.
                    // Rows 0-3 again past 127: the emitter reads a neighbouring row at an
                    // immediate offset, and the PICA index wraps there. f24 keeps its value
                    // as a float, so the loop below is a copy and clang makes it a NEON one;
                    // it goes straight into the uniform buffer where the layout allows.
                    u32 lo = 0, hi = 96;
                    if (vs_bank_valid) {
                        lo = std::min<u32>(pica.vs_setup.sync_f_lo, 96);
                        hi = std::min<u32>(pica.vs_setup.sync_f_hi, 96);
                    } else {
                        for (u32 n = 96 * 4; n < 128 * 4; n++) {
                            vs_bank[n] = 1.0f;
                        }
                        vs_bank_valid = true;
                    }
                    for (u32 n = lo; n < hi; n++) {
                        vs_bank[n * 4 + 0] = uniforms.f[n].x.ToFloat32();
                        vs_bank[n * 4 + 1] = uniforms.f[n].y.ToFloat32();
                        vs_bank[n * 4 + 2] = uniforms.f[n].z.ToFloat32();
                        vs_bank[n * 4 + 3] = uniforms.f[n].w.ToFloat32();
                    }
                    if (lo < 4) {
                        std::memcpy(&vs_bank[128 * 4], &vs_bank[0], 4 * 4 * sizeof(float));
                    }
                    pica.vs_setup.sync_f_lo = 96;
                    pica.vs_setup.sync_f_hi = 0;
                    if (vsu->f.direct && vsu->f.capacity >= 132 * 4) {
                        std::memcpy(static_cast<float*>(vbuf) + vsu->f.offset, vs_bank.data(),
                                    sizeof(vs_bank));
                    } else {
                        set(vsu->f, vs_bank.data(), 132 * 4);
                    }
                }
                if (vsu->i) {
                    float i[4 * 4];
                    for (u32 n = 0; n < 4; n++) {
                        i[n * 4 + 0] = static_cast<float>(uniforms.i[n].x);
                        i[n * 4 + 1] = static_cast<float>(uniforms.i[n].y);
                        i[n * 4 + 2] = static_cast<float>(uniforms.i[n].z);
                        i[n * 4 + 3] = static_cast<float>(uniforms.i[n].w);
                    }
                    set(vsu->i, i, 16);
                }
                if (vsu->b) {
                    float b[16];
                    for (u32 n = 0; n < 16; n++) {
                        b[n] = uniforms.b[n] ? 1.0f : 0.0f;
                    }
                    set(vsu->b, b, 16);
                }
                if (vsu->any_default) {
                    // The uniform is named by register, the value by attribute index: the
                    // fixed default attribute mapped to the register, else (0, 0, 0, 1).
                    const auto& va = regs.pipeline.vertex_attributes;
                    for (u32 reg = 0; reg < 16; reg++) {
                        if (!vsu->default_regs[reg]) {
                            continue;
                        }
                        float value[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                        for (u32 attr = 0; attr < 16; attr++) {
                            if (va.IsDefaultAttribute(attr) &&
                                regs.vs.GetRegisterForAttribute(attr) == reg) {
                                const auto& v = pica.input_default_attributes[attr];
                                value[0] = v.x.ToFloat32();
                                value[1] = v.y.ToFloat32();
                                value[2] = v.z.ToFloat32();
                                value[3] = v.w.ToFloat32();
                            }
                        }
                        set(vsu->default_regs[reg], value, 4);
                    }
                    uploaded_default_attrs = pica.input_default_attributes;
                }
                pica.vs_setup.uniforms_dirty = false;
            }
        }
        vs_data_dirty = false;
    }
    if (!fragment_dirty) {
        return;
    }
    tex_dims_dirty = false;
    uniform_fb_dims = {fb_dims[0], fb_dims[1]};

    void* fbuf = nullptr;
    if (sceGxmReserveFragmentDefaultUniformBuffer(context, &fbuf) < 0 || fbuf == nullptr) {
        return;
    }
    const FsUniforms& u = UniformsOf(fs);
    const auto set = [&](const UniformSlot& slot, const float* data, u32 count) {
        SetUniformF(fbuf, slot, data, count, "fs");
    };
    const auto set_int = [&](const UniformSlot& slot, int value) {
        const float f = static_cast<float>(value);
        set(slot, &f, 1);
    };
    const auto& d = fs_data;
    set_int(u.alphatest_ref, d.alphatest_ref);
    set(u.depth_scale, &d.depth_scale, 1);
    set(u.depth_offset, &d.depth_offset, 1);
    set_int(u.scissor_x1, d.scissor_x1);
    set_int(u.scissor_y1, d.scissor_y1);
    set_int(u.scissor_x2, d.scissor_x2);
    set_int(u.scissor_y2, d.scissor_y2);
    set_int(u.fog_lut_offset, d.fog_lut_offset);
    set_int(u.proctex_noise_lut_offset, d.proctex_noise_lut_offset);
    set_int(u.proctex_color_map_offset, d.proctex_color_map_offset);
    set_int(u.proctex_alpha_map_offset, d.proctex_alpha_map_offset);
    set_int(u.proctex_lut_offset, d.proctex_lut_offset);
    set_int(u.proctex_diff_lut_offset, d.proctex_diff_lut_offset);
    set(u.proctex_bias, &d.proctex_bias, 1);
    if (u.lighting_lut_offset) {
        float lut[24];
        for (int i = 0; i < 6; i++) {
            lut[i * 4 + 0] = static_cast<float>(d.lighting_lut_offset[i].x);
            lut[i * 4 + 1] = static_cast<float>(d.lighting_lut_offset[i].y);
            lut[i * 4 + 2] = static_cast<float>(d.lighting_lut_offset[i].z);
            lut[i * 4 + 3] = static_cast<float>(d.lighting_lut_offset[i].w);
        }
        set(u.lighting_lut_offset, lut, 24);
    }
    set(u.fog_color, d.fog_color.AsArray(), 3);
    set(u.proctex_noise_f, d.proctex_noise_f.AsArray(), 2);
    set(u.proctex_noise_a, d.proctex_noise_a.AsArray(), 2);
    set(u.proctex_noise_p, d.proctex_noise_p.AsArray(), 2);
    set(u.lighting_global_ambient, d.lighting_global_ambient.AsArray(), 3);
    for (u32 i = 0; i < FsUniforms::Lights; i++) {
        const auto& l = d.light_src[i];
        const auto& p = u.light_src[i];
        set(p.specular_0, l.specular_0.AsArray(), 3);
        set(p.specular_1, l.specular_1.AsArray(), 3);
        set(p.diffuse, l.diffuse.AsArray(), 3);
        set(p.ambient, l.ambient.AsArray(), 3);
        set(p.position, l.position.AsArray(), 3);
        set(p.spot_direction, l.spot_direction.AsArray(), 3);
        set(p.dist_atten_bias, &l.dist_atten_bias, 1);
        set(p.dist_atten_scale, &l.dist_atten_scale, 1);
    }
    if (u.const_color) {
        float cc[24];
        for (int i = 0; i < 6; i++) {
            std::memcpy(&cc[i * 4], d.const_color[i].AsArray(), 16);
        }
        set(u.const_color, cc, 24);
    }
    set(u.tev_combiner_buffer_color, d.tev_combiner_buffer_color.AsArray(), 4);
    set(u.tex_lod_bias, d.tex_lod_bias.AsArray(), 3);
    if (u.tex_border_color) {
        float bc[12];
        for (int i = 0; i < 3; i++) {
            std::memcpy(&bc[i * 4], d.tex_border_color[i].AsArray(), 16);
        }
        set(u.tex_border_color, bc, 12);
    }
    set(u.blend_color, d.blend_color.AsArray(), 4);
    set(u.tex_dims, tex_dims.data(), 6);
    set(u.fb_dims, fb_dims, 2);
    fs_data_dirty = false;
}

void RasterizerGxm::BindFragmentTexture(u32 unit, const SceGxmTexture& texture) {
    if (texture_binding_valid[unit] &&
        std::memcmp(&texture_bindings[unit], &texture, sizeof(texture)) == 0) {
        return;
    }
    // Remember only successful binds; a refused descriptor must be retried next draw.
    texture_binding_valid[unit] =
        sceGxmSetFragmentTexture(Device().context, unit, &texture) >= 0;
    if (texture_binding_valid[unit]) {
        texture_bindings[unit] = texture;
    }
}

void RasterizerGxm::SyncTextureUnits(const Framebuffer* framebuffer) {
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
    std::array<SceGxmTexture, 3> textures{white_texture, white_texture, white_texture};
    std::array<float, 6> dimensions{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    user_config = {};
    bound_surfaces = {};
    const auto pica_textures = regs.texturing.GetTextures();
    for (u32 unit = 0; unit < pica_textures.size(); unit++) {
        const auto& texture = pica_textures[unit];
        if (!texture.enabled) {
            continue;
        }
        if (unit == 0) {
            switch (texture.config.type.Value()) {
            case TextureType::Shadow2D:
            case TextureType::ShadowCube:
            case TextureType::TextureCube:
                // P5: cube maps and shadow textures.
                continue;
            default:
                break;
            }
        }
        // The id, not just the reference: the surface is given this scene's serial after
        // BeginScene, and a SurfaceId survives the cache moving its slots.
        const auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config,
                                                                      texture.format);
        const u32 max_level =
            VideoCore::MipLevels(info.width, info.height, texture.config.lod.max_level) - 1;
        const VideoCore::SurfaceId surface_id = res_cache.GetTextureSurface(info, max_level);
        if (!surface_id) {
            stat_white_nosurface++;
            continue;
        }
        Surface& surface = res_cache.GetSurface(surface_id);
        if (!surface.HasMemory()) {
            stat_white_nomem++;
            continue;
        }
        // Sampling the target being drawn is a feedback loop GXM cannot express on a
        // linear surface within the scene; the GL backend copies the texture. Bind it
        // as-is (the tiler reads what was stored before this scene) rather than white.
        (void)framebuffer;
        if (surface.HasPendingClear()) {
            runtime.FlushPendingClear(surface);
        }
        SceGxmTexture bound = *surface.Texture();
        res_cache.GetSampler(texture.config).Apply(bound);
        textures[unit] = bound;
        // The scene that will sample this has not begun yet (a cache miss here uploads, and
        // an upload may end the open scene), so the surface is noted and given the serial in
        // DrawTriangles once BeginScene has claimed one.
        bound_surfaces[unit] = surface_id;
        dimensions[unit * 2] = static_cast<float>(surface.width);
        dimensions[unit * 2 + 1] = static_cast<float>(surface.height);
    }
    // A cache lookup or pending clear can blit through unit 0. Bind only after all
    // lookups have finished so a later unit's upload cannot overwrite an earlier one.
    for (u32 unit = 0; unit < textures.size(); unit++) {
        BindFragmentTexture(unit, textures[unit]);
    }
    if (tex_dims != dimensions) {
        tex_dims = dimensions;
        tex_dims_dirty = true;
    }
    // Units 3 to 7 (the LUTs and the framebuffer-fetch unit) are bound once a scene, in
    // BeginScene: they are white until P5 and a texture binding persists indefinitely.
}

bool RasterizerGxm::BeginDraw(DrawSetup& setup, const VertexLayout& layout,
                              u32 vertex_count) {
    SyncDrawUniforms();

    const auto& fb_regs = regs.framebuffer;
    const auto& om = fb_regs.output_merger;
    // Alpha rejection precedes depth/stencil updates. Leave the framebuffer helper
    // uncreated so a fully rejected draw neither uploads textures nor invalidates surfaces.
    using CompareFunc = Pica::FramebufferRegs::CompareFunc;
    const auto& alpha = om.alpha_test;
    if (alpha.enable &&
        (alpha.func == CompareFunc::Never ||
         (alpha.func == CompareFunc::LessThan && alpha.ref == 0) ||
         (alpha.func == CompareFunc::GreaterThan && alpha.ref == 255))) {
        return false;
    }
    if (fb_regs.IsShadowRendering()) {
        // No image load/store on this backend: shadow maps are dropped (P5).
        return false;
    }
    const bool has_stencil = fb_regs.HasStencil();
    const bool write_color_fb =
        fb_regs.framebuffer.allow_color_write != 0 &&
        (om.red_enable || om.green_enable || om.blue_enable || om.alpha_enable);
    // PICA writes depth whenever the write is enabled, test or no test: a title's sky
    // pass (test off, write on) is what refreshes the depth buffer each frame, and SM3DL
    // never fills it otherwise. SyncState tests with ALWAYS for that case, as the GL
    // renderer does.
    const bool depth_write = om.depth_write_enable &&
                             fb_regs.framebuffer.allow_depth_stencil_write != 0;
    const bool stencil_write = has_stencil && om.stencil_test.enable &&
                               om.stencil_test.write_mask != 0 &&
                               fb_regs.framebuffer.allow_depth_stencil_write != 0;
    const bool using_color_fb =
        fb_regs.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb = fb_regs.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
                                (depth_write || stencil_write || om.depth_test_enable != 0 ||
                                 (has_stencil && om.stencil_test.enable));

    {
        const PhaseTimer timer{time_draws, times.framebuffer};
        setup.helper.emplace(res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb));
    }
    Framebuffer* fb = setup.helper->Framebuffer();
    if (fb->RenderTarget() == nullptr) {
        return false;
    }
    // Texture uploads the cache does for this draw may wait for the GPU (the sync hook ends
    // the scene), and so may a LUT ring wrap; both before the scene begins.
    {
        const PhaseTimer timer{time_draws, times.textures};
        SyncAndUploadLUTs();
        SyncTextureUnits(fb);
    }
    {
        const PhaseTimer timer{time_draws, times.scene};
        BeginScene(fb);
    }
    if (!in_scene) {
        return false;
    }
    // The scene now has a serial: the surfaces it will sample are read for as long as its
    // fragment processing runs, so nothing may write them until the GPU passes it.
    for (const VideoCore::SurfaceId id : bound_surfaces) {
        if (id) {
            res_cache.GetSurface(id).gpu_serial = runtime.CurrentEpoch();
        }
    }
    {
        const PhaseTimer timer{time_draws, times.state};
        SyncState(*setup.helper);
    }
    DrawTrace("state synced");
    {
        const PhaseTimer timer{time_draws, times.programs};
        if (!BindPrograms(layout, vertex_count)) {
            stat_skipped++;
            return false;
        }
    }
    DrawTrace("programs bound");
    // No second texture bind here: the SDK is explicit that sceGxmSetFragmentTexture "may be
    // called at any time" and that the binding "persists indefinitely", so the one above is
    // enough - and a second one would be a hazard, because a cache lookup that misses uploads
    // through the sync hook, which ends the scene this draw is inside. Nothing between
    // BeginScene and the draw may reach the cache for that reason.
    if (!in_scene) {
        // Something between BeginScene and here synced. The draw would be silently refused
        // with SCE_GXM_ERROR_NOT_WITHIN_SCENE, so say so instead of losing it quietly.
        LOG_ERROR(Render, "GXM: the scene was ended under a draw");
        return false;
    }
    return true;
}

void RasterizerGxm::ReleaseDraw(DrawSetup& setup) {
    DrawTrace("release draw");
    // The helper's destructor invalidates the drawn region: cache work worth its own line.
    const PhaseTimer timer{time_draws, times.invalidate};
    setup.helper.reset();
    DrawTrace("release draw done");
}

void RasterizerGxm::DrawTriangles() {
    // The triangles are in the ring, written there by the shipping facade or by the mirror's
    // software shader; vertex_batch is only ever used when there is no ring at all.
    if (pending_ring_ranges.empty() && vertex_batch.empty()) {
        return;
    }
    if (!EnsureResources()) {
        DropDraw();
        return;
    }
    const PhaseTimer total_timer{time_draws, times.total};
    u32 vertex_count = static_cast<u32>(vertex_batch.size());
    for (const auto& range : pending_ring_ranges) {
        vertex_count += range.count;
    }
    DrawSetup setup;
    if (!BeginDraw(setup, fixed_layout, vertex_count)) {
        ReleaseDraw(setup);
        DropDraw();
        return;
    }
    static const bool log_draw = GxmFlag("logdraw");
    if (log_draw && stat_draws == 0 && !pending_ring_ranges.empty()) {
        const auto& v = *reinterpret_cast<const Pica::OutputVertex*>(
            vertex_ring.base + static_cast<std::size_t>(pending_ring_ranges[0].first) *
                                   sizeof(Pica::OutputVertex));
        LOG_INFO(Render, "draw: v0 pos ({}, {}, {}, {}) colour ({}, {}, {}, {}) uv0 ({}, {}) verts {}",
                 v.pos.x.ToFloat32(), v.pos.y.ToFloat32(), v.pos.z.ToFloat32(),
                 v.pos.w.ToFloat32(), v.color.x.ToFloat32(), v.color.y.ToFloat32(),
                 v.color.z.ToFloat32(), v.color.w.ToFloat32(), v.tc0.x.ToFloat32(),
                 v.tc0.y.ToFloat32(), vertex_count);
    }
    stat_draws++;

    SceGxmContext* context = Device().context;
    const PhaseTimer draw_timer{time_draws, times.draw};
    // Each range is a run of whole triangles already sitting in GPU memory: the stream base
    // moves to the run and the indices always start at zero, so no index offset is needed and
    // a run may be longer than a u16 index could reach.
    for (const auto& range : pending_ring_ranges) {
        u32 first = range.first;
        u32 left = range.count;
        while (left > 0) {
            const u32 count = std::min(left, MaxVerticesPerDraw);
            sceGxmSetVertexStream(
                context, 0, vertex_ring.base + static_cast<std::size_t>(first) *
                                                   sizeof(Pica::OutputVertex));
            sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                       identity_indices, count);
            first += count;
            left -= count;
        }
        stat_vertices += range.count;
        ring_scene_pos = std::max(ring_scene_pos, range.end_pos);
        // How much of the ring is outstanding right now: measured here rather than at the
        // frame's end, where the wait has just retired all of it and the answer is zero.
        stat_ring_peak = std::max(
            stat_ring_peak,
            static_cast<u32>(std::min<u64>(
                range.end_pos - vertex_ring.retired.load(std::memory_order_relaxed), ring_size)));
    }
    vertex_batch.clear();
    pending_ring_ranges.clear();
    ReleaseDraw(setup);
}

Usse::VsRequest RasterizerGxm::MakeVsRequest(
    const Pica::Shader::Generator::Cg::VSExtra& extra) const {
    const Pica::Shader::Generator::PicaVSConfig config{regs, pica.vs_setup};
    const auto& state = config.state;
    Usse::VsRequest request{};
    request.code = pica.vs_setup.GetProgramCode();
    request.swizzle = pica.vs_setup.GetSwizzleData();
    request.code_size =
        std::min<u32>(pica.vs_setup.GetBiggestProgramSize(), Pica::MAX_PROGRAM_CODE_LENGTH);
    request.main_offset = state.main_offset;
    request.num_outputs = state.num_outputs;
    request.output_map = state.output_map;
    const auto maps = state.gs_state.GetSemanticMaps();
    for (u32 i = 0; i < maps.size(); i++) {
        request.semantics[i] = {static_cast<u8>(std::min<u32>(maps[i].attribute_index, 255)),
                                static_cast<u8>(maps[i].component_index & 3)};
    }
    request.default_regs = extra.default_regs;
    request.bools = extra.bools;
    request.bool_mask = extra.bool_mask;
    request.reg_components = extra.reg_components;
    return request;
}

void RasterizerGxm::ConsiderTier2(Shader* shader, u64 key, const Pica::Shader::FSConfig* fs,
                                  const Usse::VsRequest* vs) {
    if (tier2 == nullptr || shader == nullptr || shader->tier2_queued) {
        return;
    }
    if (fs != nullptr && ++shader->draws < tier2_threshold) {
        return;
    }
    shader->tier2_queued = true;
    if (fs != nullptr) {
        tier2->QueueFragment(key, *fs);
    } else if (vs != nullptr) {
        tier2->QueueVertex(key, *vs);
    }
}

void RasterizerGxm::InstallTier2() {
    if (tier2 == nullptr) {
        return;
    }
    static const bool dump = GxmFlag("usse_dump");
    for (ShaderTier2::Result& result : tier2->Take()) {
        if (result.program.empty()) {
            stat_tier2_refused++;
            continue;
        }
        const std::string name =
            fmt::format("{}_{:016x}", result.vertex ? "tv" : "tf", result.key);
        Shader* shader =
            pipeline_cache->Replace(result.key, name, result.program, result.input_regs);
        if (shader == nullptr) {
            stat_tier2_refused++;
            continue;
        }
        if (result.vertex) {
            generated_vs[result.key] = shader;
        }
        stat_tier2++;
        if (dump) {
            sceIoMkdir("ux0:data/azahar/usse", 0777);
            const std::string path = fmt::format("ux0:data/azahar/usse/{}.gxp", name);
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(result.program.data(), 1, result.program.size(), f);
                std::fclose(f);
            }
        }
    }
}

Shader* RasterizerGxm::VertexShaderFor(u16 default_regs,
                                       const std::array<u8, 16>& reg_components) {
    using namespace Pica::Shader::Generator;
    const PicaVSConfig config{regs, pica.vs_setup};
    Cg::VSExtra extra{};
    extra.default_regs = default_regs;
    extra.reg_components = reg_components;
    // The bools the program branches on are baked in (see VSExtra). Which ones those are is
    // a property of the program, found once per program.
    const auto used_it = used_bools.find(config.state.program_hash);
    if (used_it != used_bools.end()) {
        extra.bool_mask = used_it->second;
    } else {
        extra.bool_mask = Cg::UsedBoolUniforms(pica.vs_setup);
        used_bools.emplace(config.state.program_hash, extra.bool_mask);
    }
    for (u32 n = 0; n < 16; n++) {
        extra.bools |= static_cast<u16>((pica.vs_setup.uniforms.b[n] ? 1u : 0u) << n);
    }
    extra.bools &= extra.bool_mask;
    const u64 key = Cg::VertexShaderKey(config, extra);
    const auto it = generated_vs.find(key);
    if (it != generated_vs.end()) {
        if (it->second != nullptr && tier2 != nullptr && !it->second->tier2_queued &&
            ++it->second->draws >= tier2_threshold) {
            // The request again, for the second tier; the setup is this program's (the
            // key holds its hash).
            Usse::VsRequest request = MakeVsRequest(extra);
            ConsiderTier2(it->second, key, nullptr, &request);
        }
        return it->second;
    }
    // Bisect list: ux0:data/azahar/hwvs_refuse.txt names vertex keys (hex, one per line)
    // whose programs go to the software shader, so a wrong picture can be pinned on one.
    static const std::vector<u64> refuse_keys = [] {
        std::vector<u64> keys;
        if (FILE* f = std::fopen("ux0:data/azahar/hwvs_refuse.txt", "rb")) {
            char line[64];
            while (std::fgets(line, sizeof(line), f) != nullptr) {
                keys.push_back(std::strtoull(line, nullptr, 16));
            }
            std::fclose(f);
        }
        return keys;
    }();
    const bool listed =
        std::find(refuse_keys.begin(), refuse_keys.end(), key) != refuse_keys.end();
    if (listed) {
        LOG_INFO(Render, "vs_{:016x} refused by hwvs_refuse.txt", key);
    }
    Shader* vs = nullptr;
    u32 input_regs = 0;
    // The emitter is the only path: what it refuses shades in software (nothing compiles on
    // the console any more).
    if (!listed) {
        const Usse::VsRequest request = MakeVsRequest(extra);
        std::string refusal;
        const std::vector<u8> program = Usse::EmitVertexProgram(request, input_regs, &refusal);
        if (!program.empty()) {
            vs = pipeline_cache->UseBuiltinShader(key, fmt::format("uv_{:016x}", key), program,
                                                  input_regs);
            stat_vs_emitted++;
            static const bool dump = GxmFlag("usse_dump");
            if (dump) {
                sceIoMkdir("ux0:data/azahar/usse", 0777);
                const std::string stem = fmt::format("ux0:data/azahar/usse/uv_{:016x}", key);
                if (FILE* f = std::fopen((stem + ".gxp").c_str(), "wb")) {
                    std::fwrite(program.data(), 1, program.size(), f);
                    std::fclose(f);
                }
                // The request itself, so the emitter can be run on the host at any level
                // (the host emitter harness reads it).
                if (FILE* f = std::fopen((stem + ".vq").c_str(), "wb")) {
                    const u32 head[4] = {request.main_offset, request.num_outputs,
                                         request.code_size,
                                         static_cast<u32>(request.swizzle.size())};
                    std::fwrite(head, sizeof(u32), 4, f);
                    std::fwrite(request.output_map.data(), sizeof(u32), 16, f);
                    std::fwrite(request.semantics.data(), sizeof(Usse::VsSemantic), 24, f);
                    const u16 extra16[3] = {request.default_regs, request.bools, request.bool_mask};
                    std::fwrite(extra16, sizeof(u16), 3, f);
                    std::fwrite(request.reg_components.data(), 1, 16, f);
                    std::fwrite(request.code.data(), sizeof(u32), request.code_size, f);
                    std::fwrite(request.swizzle.data(), sizeof(u32), request.swizzle.size(), f);
                    std::fclose(f);
                }
                // The same program as Cg, so psp2cgc can compile the twin off the console.
                u32 cg_inputs = 0;
                const std::string cg =
                    Cg::GenerateVertexShader(pica.vs_setup, config, extra, cg_inputs);
                if (FILE* f = std::fopen((stem + ".cg").c_str(), "wb")) {
                    std::fwrite(cg.data(), 1, cg.size(), f);
                    std::fclose(f);
                }
            }
        } else {
            LOG_WARNING(Render, "vertex emitter refused vs_{:016x}: {}", key, refusal);
            stat_vs_refused++;
        }
    }
    if (vs == nullptr) {
        stat_hw_refused++;
    }
    generated_vs.emplace(key, vs);
    return vs;
}

bool RasterizerGxm::AccelerateShippedDraw(const Pica::DrawPayload& payload) {
    using Topology = Pica::PipelineRegs::TriangleTopology;
    static const bool software_only = GxmFlag("nohwvs");
    if (software_only || regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No ||
        payload.immediate) {
        return false;
    }
    if (!EnsureResources()) {
        return false;
    }
    const PhaseTimer total_timer{time_draws, times.total};

    SceGxmPrimitiveType primitive = SCE_GXM_PRIMITIVE_TRIANGLES;
    switch (regs.pipeline.triangle_topology) {
    case Topology::Strip:
        primitive = SCE_GXM_PRIMITIVE_TRIANGLE_STRIP;
        break;
    case Topology::Fan:
        primitive = SCE_GXM_PRIMITIVE_TRIANGLE_FAN;
        break;
    default:
        break;
    }
    // The stream is indexed with 16-bit values relative to the arena's first vertex, and the
    // SDK keeps them below 64000; a non-indexed list past the identity table is drawn in
    // pieces, which a strip or fan cannot be.
    const u32 vertex_range = payload.vertex_max - payload.vertex_min + 1;
    if (vertex_range > 64000 || payload.vertex_max < payload.vertex_min) {
        return false;
    }
    if (!payload.is_indexed && payload.num_vertices > MaxVerticesPerDraw &&
        primitive != SCE_GXM_PRIMITIVE_TRIANGLES) {
        return false;
    }

    // The streams: one per vertex loader that owns arena bytes, walked exactly as the GL
    // backend and the software loader walk them, so the offsets agree with the arena.
    static constexpr const char* AttributeNames[16] = {
        "vs_in_attr0",  "vs_in_attr1",  "vs_in_attr2",  "vs_in_attr3",
        "vs_in_attr4",  "vs_in_attr5",  "vs_in_attr6",  "vs_in_attr7",
        "vs_in_attr8",  "vs_in_attr9",  "vs_in_attr10", "vs_in_attr11",
        "vs_in_attr12", "vs_in_attr13", "vs_in_attr14", "vs_in_attr15",
    };
    struct Candidate {
        u32 reg;
        VertexAttribute attribute;
    };
    std::array<Candidate, 16> candidates{};
    u32 candidate_count = 0;
    std::array<u32, PipelineInfo::MaxStreams> stream_arena_offset{};
    std::array<u32, PipelineInfo::MaxStreams> stream_bytes{};
    VertexLayout layout{};
    u32 enabled_regs = 0;
    const auto& va = regs.pipeline.vertex_attributes;
    for (u32 loader_index = 0; loader_index < 12; loader_index++) {
        const auto& loader = va.attribute_loaders[loader_index];
        if (loader.component_count == 0 || loader.byte_count == 0 ||
            (payload.layout.used_mask & (1u << loader_index)) == 0) {
            continue;
        }
        const u32 stream = layout.stream_count;
        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; ++comp) {
            const u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index < 12) {
                if (va.GetNumElements(attribute_index) == 0) {
                    continue;
                }
                const u32 element_size = va.GetElementSizeInBytes(attribute_index);
                offset = Common::AlignUp(offset, element_size);
                const u32 reg = regs.vs.GetRegisterForAttribute(attribute_index);
                SceGxmAttributeFormat format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
                switch (va.GetFormat(attribute_index)) {
                case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
                    format = SCE_GXM_ATTRIBUTE_FORMAT_S8;
                    break;
                case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
                    format = SCE_GXM_ATTRIBUTE_FORMAT_U8;
                    break;
                case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
                    format = SCE_GXM_ATTRIBUTE_FORMAT_S16;
                    break;
                default:
                    break;
                }
                if (candidate_count == candidates.size()) {
                    return false;
                }
                candidates[candidate_count++] = {
                    reg, VertexAttribute{AttributeNames[reg], static_cast<u16>(offset), format,
                                         static_cast<u8>(va.GetNumElements(attribute_index)),
                                         static_cast<u8>(stream)}};
                enabled_regs |= 1u << reg;
                offset += va.GetStride(attribute_index);
            } else {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings.
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            }
        }
        layout.strides[stream] = static_cast<u16>(loader.byte_count);
        stream_arena_offset[stream] = payload.layout.loader_offset[loader_index];
        stream_bytes[stream] = loader.byte_count * vertex_range;
        layout.stream_count++;
    }
    if (layout.stream_count == 0) {
        return false;
    }
    // Every register no loader feeds is a uniform to the program (the emitter only
    // declares the ones it reads): the fixed default attribute's value where the guest
    // set one, (0, 0, 0, 1) otherwise, as GL's disabled generic attribute. Declaring such
    // a register as an attribute made the patcher refuse the program with INVALID_VALUE
    // (Pokémon Alpha Sapphire reads five input registers nothing feeds, 2026-09-06).
    const u16 default_regs = static_cast<u16>(~enabled_regs & 0xffffu);
    std::array<u8, 16> reg_components{};
    for (u32 n = 0; n < candidate_count; n++) {
        reg_components[candidates[n].reg] = candidates[n].attribute.components;
    }
    Shader* vs = VertexShaderFor(default_regs, reg_components);
    if (vs == nullptr || !vs->IsDone() || vs->failed) {
        // No program yet, or none possible: the software shader on the mirror takes the
        // draw. Slow, but the picture is there while the compile runs - a generated vertex
        // program is twenty seconds on this CPU - and the disk cache makes it a first-run
        // cost.
        return false;
    }
    layout.shader = vs;
    // Only the attributes the program reads: the patcher refuses one it does not name. Two
    // loaders feeding the same register is a guest bug; the last wins, as in GL, and the
    // earlier one must not stay in the list (the patcher refuses two attributes on one
    // register with INVALID_VALUE: Pokémon Alpha Sapphire, 2026-09-06).
    for (u32 n = 0; n < candidate_count; n++) {
        if ((vs->input_regs & (1u << candidates[n].reg)) == 0) {
            continue;
        }
        bool later = false;
        for (u32 m = n + 1; m < candidate_count; m++) {
            later |= candidates[m].reg == candidates[n].reg;
        }
        if (!later) {
            layout.attributes[layout.attribute_count++] = candidates[n].attribute;
        }
    }
    static const bool dump_layout = GxmFlag("usse_dump");
    if (dump_layout) {
        std::string what;
        for (u32 n = 0; n < layout.attribute_count; n++) {
            const auto& a = layout.attributes[n];
            what += fmt::format(" {}:s{}+{} f{:#x} x{}", a.name, a.stream, a.offset,
                                static_cast<u32>(a.format), a.components);
        }
        for (u32 stream = 0; stream < layout.stream_count; stream++) {
            what += fmt::format(" stride{}={}", stream, layout.strides[stream]);
        }
        LOG_INFO(Render, "layout for {}:{}", vs->name, what);
    }

    std::array<u32, PipelineInfo::MaxStreams> stream_ring_offset{};
    const u8* base = nullptr;
    const u16* indices = identity_indices;
    if (payload.in_ring) {
        // The emulation thread wrote the draw into this ring already, in this layout: the
        // streams are the loader blocks where the arena layout put them, the indices 16-bit
        // and rebased. Nothing to copy; the ring position is retired with the scene.
        hw_ring_end = std::max(hw_ring_end, payload.ring_end);
        base = vertex_ring.base + payload.ring_offset;
        for (u32 stream = 0; stream < layout.stream_count; stream++) {
            stream_ring_offset[stream] = stream_arena_offset[stream];
        }
        if (payload.is_indexed) {
            indices = reinterpret_cast<const u16*>(base + payload.layout.index_offset);
        }
    } else {
        // An arena draw (the ring was full when it shipped): ring space for the streams and
        // the indices, in one allocation padded to the software path's vertex stride so its
        // first-index arithmetic stays exact.
        u32 total = 0;
        for (u32 stream = 0; stream < layout.stream_count; stream++) {
            stream_ring_offset[stream] = total;
            total += Common::AlignUp(stream_bytes[stream], 16);
        }
        const u32 index_ring_offset = total;
        const u32 index_bytes = payload.is_indexed ? payload.num_vertices * 2 : 0;
        total += Common::AlignUp(index_bytes, 16);
        total = Common::AlignUp<u32>(std::max<u32>(total, 1), sizeof(Pica::OutputVertex));
        if (total > ring_size / 2) {
            return false;
        }
        // Under the ring's mutex: the emulation thread allocates from it too.
        u32 off;
        {
            std::scoped_lock lock{vertex_ring.mutex};
            off = vertex_ring.TryAlloc(total);
        }
        for (u32 attempt = 0; off == VideoCore::VertexRing::FULL; attempt++) {
            if (attempt == 64) {
                // Nothing in flight is holding the space and nothing is being freed.
                LOG_WARNING(Render, "GXM: no ring space for a {} byte hardware draw", total);
                return false;
            }
            RingRetireBlocking();
            std::scoped_lock lock{vertex_ring.mutex};
            off = vertex_ring.TryAlloc(total);
        }
        hw_ring_end = std::max(hw_ring_end, vertex_ring.cursor);
        u8* const dst_base = vertex_ring.base + off;
        for (u32 stream = 0; stream < layout.stream_count; stream++) {
            std::memcpy(dst_base + stream_ring_offset[stream],
                        payload.arena + stream_arena_offset[stream], stream_bytes[stream]);
        }
        if (payload.is_indexed) {
            // Rebased to the arena's first vertex, and widened: GXM indexes with 16 bits.
            Pica::VertexLoader::WriteRingIndices(dst_base + index_ring_offset,
                                                 payload.arena + payload.layout.index_offset,
                                                 payload.index_u16, payload.num_vertices,
                                                 payload.vertex_min);
            indices = reinterpret_cast<const u16*>(dst_base + index_ring_offset);
        }
        base = dst_base;
    }

    DrawSetup setup;
    if (!BeginDraw(setup, layout, payload.num_vertices)) {
        // Consumed, not drawn: a program still compiling, a target the cache has no surface
        // for. The software shader would only reach the same refusal after shading.
        ReleaseDraw(setup);
        DropDraw();
        return true;
    }
    stat_draws++;
    stat_hw_draws++;
    if (scenes_this_frame > 0 && scenes_this_frame <= MaxSceneRecords) {
        scene_records[scenes_this_frame - 1].draws++;
    }

    SceGxmContext* context = Device().context;
    const PhaseTimer draw_timer{time_draws, times.draw};
    if (payload.is_indexed || payload.num_vertices <= MaxVerticesPerDraw) {
        for (u32 stream = 0; stream < layout.stream_count; stream++) {
            sceGxmSetVertexStream(context, stream, base + stream_ring_offset[stream]);
        }
        DrawTrace("hw draw");
        {
            static const bool on = GxmFlag("logdraw");
            // The first draws of each vertex program, so one program's draws can be read
            // against another's rather than only the frame's first few.
            static std::map<const Shader*, u32> per_program;
            const bool show = on && layout.shader != nullptr && per_program[layout.shader]++ < 6;
            if (show) {
                std::string attrs;
                for (u32 n = 0; n < layout.attribute_count; n++) {
                    const auto& a = layout.attributes[n];
                    attrs += fmt::format(" {}@{}:fmt{}x{}", a.name, a.offset,
                                         static_cast<int>(a.format), a.components);
                }
                std::string v0;
                for (u32 n = 0; n < std::min<u32>(layout.strides[0], 32); n++) {
                    v0 += fmt::format("{:02x}", base[stream_ring_offset[0] + n]);
                }
                LOG_INFO(Render,
                         "hw draw: {} verts, indexed {} (min {} max {}), {} streams, stride0 {} "
                         "bytes0 {}, stride1 {} bytes1 {}, {} attributes, primitive {}, vs {},"
                         "{} v0 {}",
                         payload.num_vertices, payload.is_indexed, payload.vertex_min,
                         payload.vertex_max, layout.stream_count, layout.strides[0],
                         stream_bytes[0], layout.strides[1], stream_bytes[1],
                         layout.attribute_count, static_cast<int>(primitive),
                         layout.shader != nullptr ? layout.shader->name : "none", attrs, v0);
            }
        }
        {
            // vsdump: the first draws of the vertex program whose name contains the word in
            // ux0:data/azahar/vsdump.txt, with the default uniform buffer and stream 0 as
            // the GPU sees them, to ux0:data/azahar/vsdump_NN.bin (header: vertex count,
            // stride, stream bytes, uniform bytes; then the two blocks), so the program's
            // translation can be run off the console on the same data.
            static const bool vsdump = GxmFlag("vsdump");
            static u32 dumped = 0;
            if (vsdump && layout.shader != nullptr && last_vertex_uniforms != nullptr &&
                dumped < 32) {
                // The file holds the key and, optionally, the first frame to dump from.
                static const std::pair<std::string, u32> key_and_frame = [] {
                    std::string k;
                    u32 from = 0;
                    if (FILE* f = std::fopen("ux0:data/azahar/vsdump.txt", "rb")) {
                        char buf[64]{};
                        std::fread(buf, 1, sizeof(buf) - 1, f);
                        std::fclose(f);
                        char word[64]{};
                        if (std::sscanf(buf, "%63s %u", word, &from) >= 1) {
                            k = word;
                        }
                    }
                    return std::make_pair(k, from);
                }();
                const std::string& key = key_and_frame.first;
                if (!key.empty() && frame_index >= key_and_frame.second &&
                    layout.shader->name.find(key) != std::string::npos) {
                    const u32 ubytes =
                        sceGxmProgramGetDefaultUniformBufferSize(layout.shader->Program());
                    char path[64];
                    std::snprintf(path, sizeof(path), "ux0:data/azahar/vsdump_%02u.bin",
                                  dumped++);
                    if (FILE* f = std::fopen(path, "wb")) {
                        const u32 header[4] = {payload.num_vertices, layout.strides[0],
                                               stream_bytes[0], ubytes};
                        std::fwrite(header, sizeof(header), 1, f);
                        std::fwrite(last_vertex_uniforms, 1, ubytes, f);
                        std::fwrite(base + stream_ring_offset[0], 1, stream_bytes[0], f);
                        std::fclose(f);
                    }
                }
            }
        }
        GXM_PHASE_NAME("gpu:draw");
        const int err =
            sceGxmDraw(context, primitive, SCE_GXM_INDEX_FORMAT_U16, indices, payload.num_vertices);
        GXM_PHASE_NAME("gpu:drawn");
        DrawTrace("hw draw done");
        if (err < 0) {
            LOG_ERROR(Render, "GXM: hardware draw refused: {} ({:#x})", GxmErrorName(err),
                      static_cast<u32>(err));
        }
    } else {
        // A non-indexed list longer than the identity table: move every stream's base along
        // with the vertices.
        u32 first = 0;
        u32 left = payload.num_vertices;
        while (left > 0) {
            const u32 count = std::min(left, MaxVerticesPerDraw);
            for (u32 stream = 0; stream < layout.stream_count; stream++) {
                sceGxmSetVertexStream(context, stream,
                                      base + stream_ring_offset[stream] +
                                          static_cast<std::size_t>(first) * layout.strides[stream]);
            }
            GXM_PHASE_NAME("gpu:draw");
            sceGxmDraw(context, primitive, SCE_GXM_INDEX_FORMAT_U16, identity_indices, count);
            GXM_PHASE_NAME("gpu:drawn");
            first += count;
            left -= count;
        }
    }
    stat_vertices += payload.num_vertices;
    ring_scene_pos = std::max(ring_scene_pos, hw_ring_end);
    stat_ring_peak = std::max(
        stat_ring_peak,
        static_cast<u32>(std::min<u64>(
            hw_ring_end - vertex_ring.retired.load(std::memory_order_relaxed), ring_size)));
    hw_ring_end = 0;
    ReleaseDraw(setup);
    DrawTrace("hw draw released");
    return true;
}

void RasterizerGxm::FlushAll() {
    DrawTrace("flush all");
    const PhaseTimer timer{time_draws, times.flush};
    res_cache.FlushAll();
}

void RasterizerGxm::FlushRegion(PAddr addr, u32 size) {
    DrawTrace("flush region");
    const PhaseTimer timer{time_draws, times.flush};
    stat_flushes++;
    res_cache.FlushRegion(addr, size);
}

void RasterizerGxm::InvalidateRegion(PAddr addr, u32 size) {
    DrawTrace("invalidate region");
    const PhaseTimer timer{time_draws, times.invalidate_region};
    stat_invalidates++;
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerGxm::InvalidateGuestFlushedRegion(PAddr addr, u32 size) {
    DrawTrace("invalidate guest-flushed region");
    const PhaseTimer timer{time_draws, times.invalidate_region};
    stat_invalidates++;
    // logdraw: the ranges the guest flushes, to see what the hash checks they cause cover.
    static const bool log_flushes = GxmFlag("logdraw");
    if (log_flushes) {
        static u32 shown = 0;
        if (shown++ < 400) {
            LOG_INFO(Render, "guest flush {:08X}+{:X}", addr, size);
        }
    }
    res_cache.InvalidateRegionUnlessDirty(addr, size);
}

void RasterizerGxm::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    DrawTrace("flush+invalidate");
    const PhaseTimer timer{time_draws, times.flush};
    stat_flushes++;
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerGxm::NoteGuestWrite(PAddr addr, u32 size) {
    DrawTrace("guest write");
    const PhaseTimer timer{time_draws, times.guest_writes};
    stat_guest_writes++;
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerGxm::ClearAll(bool flush) {
    res_cache.ClearAll(flush);
}

bool RasterizerGxm::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    DrawTrace("display transfer");
    const PhaseTimer timer{time_draws, times.transfers};
    const bool done = EnsureResources() && res_cache.AccelerateDisplayTransfer(config);
    // Bring-up switch: a transfer the cache refuses is done by the CPU against memory the
    // surfaces do not know about, so the screen and the surface stop agreeing.
    static const bool log_transfers = GxmFlag("logdraw");
    if (log_transfers) {
        static u32 shown = 0;
        if (shown++ % 400 == 0) {
            LOG_INFO(Render,
                     "transfer {}: {:08X} {}x{} fmt {} -> {:08X} {}x{} fmt {}, flip {} scale {}",
                     done ? "accelerated" : "REFUSED", config.GetPhysicalInputAddress(),
                     config.input_width.Value(), config.input_height.Value(),
                     static_cast<u32>(config.input_format.Value()),
                     config.GetPhysicalOutputAddress(), config.output_width.Value(),
                     config.output_height.Value(), static_cast<u32>(config.output_format.Value()),
                     config.flip_vertically.Value(), config.scaling.Value());
        }
    }
    return done;
}

bool RasterizerGxm::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    DrawTrace("texture copy");
    const PhaseTimer timer{time_draws, times.transfers};
    return EnsureResources() && res_cache.AccelerateTextureCopy(config);
}

bool RasterizerGxm::AccelerateFill(const Pica::MemoryFillConfig& config) {
    DrawTrace("fill");
    const PhaseTimer timer{time_draws, times.transfers};
    const bool done = EnsureResources() && res_cache.AccelerateFill(config);
    static const bool log_fills = GxmFlag("logdraw");
    if (log_fills) {
        static u32 shown = 0;
        if (shown < 64 || shown % 400 == 0) {
            LOG_INFO(Render, "fill {}: {:08X}..{:08X} value {:08X} width {}",
                     done ? "accelerated" : "REFUSED", config.GetStartAddress(),
                     config.GetEndAddress(), config.value_32bit,
                     static_cast<u32>(config.fill_32bit.Value()));
        }
        shown++;
    }
    return done;
}

const SceGxmTexture* RasterizerGxm::AccelerateDisplay(int screen, PAddr addr, u32 width, u32 height,
                                                      u32 pixel_stride, Pica::PixelFormat format,
                                                      Common::Rectangle<u32>* out_rect) {
    if (addr == 0 || !resources_ready) {
        return nullptr;
    }
    VideoCore::SurfaceParams src_params;
    src_params.addr = addr;
    src_params.width = std::min(width, pixel_stride);
    src_params.height = height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(format);
    src_params.UpdateParams();
    const auto [surface_id, rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);
    if (!surface_id) {
        static const bool log_draw = GxmFlag("logdraw");
        if (log_draw && frame_index % 120 == 0) {
            LOG_INFO(Render,
                     "display {}: no surface covers {:08X} {}x{} stride {} {}; guest memory "
                     "presented instead",
                     screen, src_params.addr, src_params.width, src_params.height,
                     src_params.stride, VideoCore::PixelFormatAsString(src_params.pixel_format));
        }
        return nullptr;
    }
    const Surface& surface = res_cache.GetSurface(surface_id);
    // A surface with this framebuffer's format and stride that holds it whole: the
    // framebuffer itself, or the larger buffer a title renders and transfers (NSMB2's
    // 256x416 at 18000000 holding its 240x400 from 18002000, sixteen rows in; the frontend
    // samples that part). The stride and the fit keep a buffer a title recycles between
    // screens from showing the other screen's shape.
    const u32 first_pixel =
        surface.addr <= src_params.addr ? surface.PixelsInBytes(src_params.addr - surface.addr) : 0;
    const u32 x0 = surface.stride != 0 ? first_pixel % surface.stride : 0;
    const u32 y0 = surface.stride != 0 ? first_pixel / surface.stride : 0;
    if (surface.addr > src_params.addr || surface.stride != src_params.stride ||
        x0 + src_params.width > surface.width || y0 + src_params.height > surface.height ||
        surface.pixel_format != src_params.pixel_format || !surface.HasMemory()) {
        static const bool log_draw = GxmFlag("logdraw");
        if (log_draw && frame_index % 120 == 0) {
            LOG_INFO(Render,
                     "display {}: wanted {:08X} {}x{} {}, cache holds {:08X} {}x{} {}; guest "
                     "memory presented instead",
                     screen, src_params.addr, src_params.width, src_params.height,
                     VideoCore::PixelFormatAsString(src_params.pixel_format), surface.addr,
                     surface.width, surface.height,
                     VideoCore::PixelFormatAsString(surface.pixel_format));
        }
        return nullptr;
    }
    // Presentation happens after this frame's wait, so the surface is complete by then - but
    // the frontend's scene samples it and runs on into the next frame's draws, and that scene
    // carries no serial of ours. Flag it so anything touching it waits, until EndFrame's
    // sceGxmFinish proves the frontend is done.
    static const bool log_display = GxmFlag("logdraw");
    if (log_display) {
        static u32 shown = 0;
        if (shown++ < 8) {
            LOG_INFO(Render, "display {}: wants {:08X} {}x{}, got surface {} at {:08X} {}x{}",
                     screen, src_params.addr, src_params.width, src_params.height,
                     surface_id.index, surface.addr, surface.width, surface.height);
        }
    }
    if (Surface& shown = res_cache.GetSurface(surface_id); shown.HasPendingClear()) {
        runtime.FlushPendingClear(shown);
    }
    res_cache.GetSurface(surface_id).presented = true;
    if (out_rect != nullptr) {
        *out_rect = {x0, y0, x0 + src_params.width, y0 + src_params.height};
    }
    display_surfaces[screen & 1] = surface_id;
    display_textures[screen & 1] = *surface.Texture();
    return &display_textures[screen & 1];
}

void RasterizerGxm::EndFrame() {
    DrawTrace("end frame");
    if (!resources_ready) {
        return;
    }
    const PhaseTimer end_frame_timer{time_draws, times.end_frame};
    EndScene();
    InstallTier2();
    // No finish: the GPU keeps this frame's scenes while the next frame is being submitted.
    // Razor showed the GPU with jobs for a quarter of each frame and the render thread 55%
    // of its time inside the finish that was here (2026-09-05): the two ran in turn. What a
    // frame must not touch until the GPU is done with it (ring space, freed surfaces, the
    // blit ring's half) waits for that frame's last scene by notification instead.
    frame_serial[frame_index & 1] = runtime.CurrentEpoch();
    RetireRing();
    {
        // The half of the blit ring the next frame writes was last read by the frame before
        // this one; the wait is what the GPU still owes of that frame, usually nothing.
        const PhaseTimer timer{time_draws, times.finish};
        const u32 next_half = (frame_index + 1) & 1;
        runtime.WaitForEpoch(frame_serial[next_half]);
        blit_ring_used = next_half * (BlitRingSize / 2);
        blit_ring_limit = blit_ring_used + BlitRingSize / 2;
    }
    for (const VideoCore::SurfaceId id : display_surfaces) {
        if (id) {
            res_cache.GetSurface(id).presented = false;
        }
    }
    display_surfaces = {};
    last_frame_scenes = scene_records;
    last_frame_scene_count = std::min<u32>(scenes_this_frame, MaxSceneRecords);
    scenes_this_frame = 0;
    scene_records = {};
    frame_index++;
    stat_frames++;
    // The frontend presents through this same context and sets its own cull mode, depth
    // state, viewport and region clip while doing it.
    state_cache = CachedState{};
    texture_binding_valid.reset();
    runtime.Finish();
    TrimSurfaceCache();
    // dumpsurfaces: what the pause menu's button does, once, 300 frames in, for a headless
    // run. The presentation layer runs the dump after this frame's present, as for the button.
    static const bool dump_surfaces_once = GxmFlag("dumpsurfaces");
    if (dump_surfaces_once && frame_index == 300) {
        RequestSurfaceDump();
    }
    res_cache.TickFrame();
    // Report on wall time, not on a frame count: at the frame rates this phase produces, a
    // 120-frame report can be two minutes apart, which is no use while finding out why.
    const u64 now = static_cast<u64>(Common::Timer::GetTimeMs().count());
    if (report_at_ms == 0) {
        report_at_ms = now + ReportIntervalMs;
    } else if (now >= report_at_ms) {
        report_at_ms = now + ReportIntervalMs;
        LOG_INFO(Render,
                 "gxm {} frames: {} draws ({} hardware shaded, {} refused; {} "
                 "vertices, {} KiB ring peak), {} skipped, {} without a scissor kill, {} "
                 "scenes, {} fs emitted / {} "
                 "refused, {} vs emitted / {} refused, {} tier 2 ({} refused), "
                 "{} dropped, {} waits ({} broke a scene, {} avoided, {} for LUT rows), {} scene "
                 "deps, "
                 "{} uniform writes "
                 "({} skipped), "
                 "{} uploads ({} KiB, {} renamed), {} downloads ({} KiB), {} transfers, {} gpu blits "
                 "({} merged), "
                 "{} cpu blits, {} fills ({} handed to a scene), {} shaders; "
                 "pools LPDDR {}/{} KiB "
                 "CDRAM {}/{} KiB",
                 stat_frames, stat_draws, stat_hw_draws, stat_hw_refused,
                 stat_vertices,
                 stat_ring_peak / 1024, stat_skipped, stat_scissor_free,
                 stat_scenes, stat_usse_emitted, stat_usse_refused, stat_vs_emitted,
                 stat_vs_refused, stat_tier2, stat_tier2_refused, stat_dropped_draws, stat_syncs,
                 stat_scene_breaks,
                 runtime.stat_syncs_avoided, stat_lut_waits, stat_scene_waits,
                 stat_uniform_uploads, stat_uniform_skips,
                 runtime.stat_uploads,
                 runtime.stat_upload_bytes / 1024, runtime.stat_renames, runtime.stat_downloads,
                 runtime.stat_download_bytes / 1024, runtime.stat_transfers,
                 runtime.stat_gpu_blits, runtime.stat_gpu_blit_merges, runtime.stat_cpu_blits,
                 runtime.stat_fills,
                 runtime.stat_deferred_clears,
                 pipeline_cache->ShaderCount(),
                 runtime.MappedPool().Used() / 1024, runtime.MappedPool().Reserved() / 1024,
                 runtime.CdramPool().Used() / 1024, runtime.CdramPool().Reserved() / 1024);
        {
            // What the surface cache is holding: the pools say how much memory is out, this
            // says how many surfaces it is spread over. Target capability is a format
            // property and includes ordinary sampled RGB/RGBA textures.
            u32 surfaces = 0, targets = 0;
            u64 surface_bytes = 0, target_bytes = 0;
            res_cache.ForEachSurface([&](VideoCore::SurfaceId, Surface& surface) {
                surfaces++;
                surface_bytes += surface.AllocSize();
                if (surface.IsColorTarget() || surface.IsDepthTarget()) {
                    targets++;
                    target_bytes += surface.AllocSize();
                }
            });
            LOG_INFO(Render, "gxm surfaces: {} holding {} KiB, of which {} target-capable surfaces "
                             "holding {} KiB, {} sentenced, {} given up, white binds {} "
                             "(no surface) {} (no memory), {} guest flushes needed no upload",
                     surfaces, surface_bytes / 1024, targets, target_bytes / 1024,
                     res_cache.SentencedCount(), stat_trimmed, stat_white_nosurface,
                     stat_white_nomem, res_cache.guest_flush_hash_skips);
            res_cache.guest_flush_hash_skips = 0;
        }
        {
            const auto memory = pipeline_cache->MemoryStats();
            LOG_INFO(Render, "gxm shader cache: {} shaders + {} retired, {} pipelines, "
                             "{} KiB GXP storage, {} pending surface frees",
                     pipeline_cache->ShaderCount(), memory.retired_shaders, memory.pipelines,
                     memory.program_bytes / 1024, runtime.PendingFreeCount());
#ifdef __vita__
            const auto* patcher = Device().patcher;
            const auto heap = mallinfo();
            LOG_INFO(Render, "gxm memory: heap {}/{} KiB, patcher host {} KiB, buffer {} KiB, "
                             "vertex USSE {} KiB, fragment USSE {} KiB",
                     heap.uordblks / 1024, heap.arena / 1024,
                     sceGxmShaderPatcherGetHostMemAllocated(patcher) / 1024,
                     sceGxmShaderPatcherGetBufferMemAllocated(patcher) / 1024,
                     sceGxmShaderPatcherGetVertexUsseMemAllocated(patcher) / 1024,
                     sceGxmShaderPatcherGetFragmentUsseMemAllocated(patcher) / 1024);
#endif
        }
        {
            // Which target each scene of a frame drew into, in the order Razor GPU Live
            // indexes them, so its per-scene fragment time can be read against a target.
            std::string scenes;
            for (u32 i = 0; i < last_frame_scene_count; i++) {
                const auto& r = last_frame_scenes[i];
                scenes += fmt::format("{}{}: {:08X} {}x{} {} draws{}", i == 0 ? "" : ", ", i,
                                      r.addr, r.width, r.height, r.draws, r.blit ? " (blit)" : "");
            }
            LOG_INFO(Render, "gxm scenes of the last frame: {}", scenes);
        }
        if (time_draws) {
            // What the GPU thread charged to itself over the same window, so the phases can
            // be read against the thread's own busy time and not only against the wall. From
            // the running total: the per-second stats line zeroes the windowed counter, so
            // reading that here only ever saw the residue since the last stats line - a
            // saturated thread reported as a quarter busy.
            const u64 busy_total =
                Common::PipelineStats::gpu_busy_total_us.load(std::memory_order_relaxed);
            const u64 busy_us = busy_total - busy_total_at_report;
            busy_total_at_report = busy_total;
            const u64 window = ReportIntervalMs * 1000;
            const auto pct = [&](u64 us) { return window == 0 ? 0u : u32(us * 100 / window); };
            const u64 accounted = times.total + times.end_frame + times.guest_writes +
                                  times.invalidate_region + times.flush + times.transfers +
                                  times.retire + times.sync + times.present;
            const u64 unaccounted = busy_us > accounted ? busy_us - accounted : 0;
            LOG_INFO(Render,
                     "gxm time (us over {} ms, {}%% of it): framebuffer {} ({}%%), textures {} "
                     "({}%%), scene {} ({}%%) + end scene {} ({}%%), state {} ({}%%), programs {} ({}%%), uniforms {} "
                     "({}%%), draw {} ({}%%), invalidate {} ({}%%); end frame {} ({}%%) of "
                     "which the GPU wait is {} ({}%%); outside a draw: {} guest writes {} "
                     "({}%%), {} invalidates {} ({}%%), {} flushes {} ({}%%), transfers {} "
                     "({}%%), ring retire {} ({}%%), sync {} ({}%%), present {} ({}%%); the "
                     "render thread was busy {} ({}%%), {} ({}%%) of it in none of the above",
                     ReportIntervalMs, pct(times.total + times.end_frame), times.framebuffer,
                     pct(times.framebuffer), times.textures, pct(times.textures), times.scene,
                     pct(times.scene), times.scene_end, pct(times.scene_end), times.state,
                     pct(times.state), times.programs,
                     pct(times.programs), times.uniforms, pct(times.uniforms), times.draw,
                     pct(times.draw), times.invalidate, pct(times.invalidate), times.end_frame,
                     pct(times.end_frame), times.finish, pct(times.finish), stat_guest_writes,
                     times.guest_writes, pct(times.guest_writes), stat_invalidates,
                     times.invalidate_region, pct(times.invalidate_region), stat_flushes,
                     times.flush, pct(times.flush), times.transfers, pct(times.transfers),
                     times.retire, pct(times.retire), times.sync, pct(times.sync),
                     times.present, pct(times.present), busy_us, pct(busy_us), unaccounted,
                     pct(unaccounted));
            times = DrawTimes{};
            stat_guest_writes = stat_invalidates = stat_flushes = 0;
        }
        stat_hw_draws = stat_hw_refused = stat_scissor_free = 0;
        stat_lut_waits = 0;
        stat_frames = stat_draws = stat_vertices = stat_scenes = stat_skipped = stat_finishes =
            stat_syncs = stat_scene_breaks = stat_ring_peak = stat_uniform_uploads =
                stat_uniform_skips = stat_dropped_draws = stat_scene_waits = 0;
        runtime.stat_uploads = runtime.stat_upload_bytes = runtime.stat_downloads =
            runtime.stat_download_bytes = runtime.stat_transfers = runtime.stat_cpu_blits =
                runtime.stat_renames = runtime.stat_deferred_clears = runtime.stat_fills = runtime.stat_syncs =
                    runtime.stat_syncs_avoided =
                    runtime.stat_gpu_blits = runtime.stat_gpu_blit_merges = 0;
    }
}

void RasterizerGxm::DumpSurfaces() {
    // Level 0 of every surface with memory, read back the way a download is (through the
    // transfer unit, which is the only way a rendered surface's pixels reach the CPU under
    // Vita3K), as a PPM of its colour or depth plus a PGM of its alpha or stencil where the
    // format has one. Rows are stored bottom-up; the files are written top-down.
    sceIoMkdir("ux0:data/azahar/dump", 0777);
    const u32 frame = surface_dump_index++ % 20;
    const std::string index_path = fmt::format("ux0:data/azahar/dump/f{:02}_index.txt", frame);
    FILE* index = std::fopen(index_path.c_str(), "wb");
    u32 count = 0;
    res_cache.ForEachSurface([&](VideoCore::SurfaceId, Surface& surface) {
        using VideoCore::PixelFormat;
        if (!surface.HasMemory() || surface.width == 0 || surface.height == 0) {
            return;
        }
        const u32 n = count++;
        const u32 w = surface.width;
        const u32 h = surface.height;
        const PixelFormat pf = surface.pixel_format;
        const u32 bpp = pf == PixelFormat::D16 ? 2u : surface.Format().bytes_per_pixel;
        const u32 bytes = w * h * bpp;
        const auto staging = runtime.FindStaging(bytes, false);
        surface.Download(VideoCore::BufferTextureCopy{.buffer_offset = 0,
                                                      .buffer_size = bytes,
                                                      .texture_rect = {0, h, w, 0},
                                                      .texture_level = 0},
                         staging);
        static constexpr const char* TypeNames[] = {"color", "texture", "depth", "depthstencil",
                                                    "fill", "invalid"};
        const char* type = TypeNames[std::min<u32>(static_cast<u32>(surface.type), 5)];
        const std::string stem =
            fmt::format("ux0:data/azahar/dump/f{:02}_{:03}_{}_{:08x}_{}x{}_{}", frame, n, type,
                        surface.addr, w, h, VideoCore::PixelFormatAsString(pf));
        if (index != nullptr) {
            const std::string line = fmt::format(
                "{:03} {} addr {:08x}-{:08x} {}x{} stride {} levels {} {}{}{}{} serial {}\n", n,
                type, surface.addr, surface.end, w, h, surface.stride, surface.levels,
                VideoCore::PixelFormatAsString(pf), surface.IsColorTarget() ? " colour-target" : "",
                surface.IsDepthTarget() ? " depth-target" : "",
                surface.presented ? " presented" : "", surface.gpu_serial);
            std::fwrite(line.data(), 1, line.size(), index);
        }
        const bool depth_word = pf == PixelFormat::D24S8;
        const bool depth_f32 = pf == PixelFormat::D24;
        const bool depth_16 = pf == PixelFormat::D16;
        const bool has_alpha = !depth_word && !depth_f32 && !depth_16 &&
                               pf != PixelFormat::RGB565 && pf != PixelFormat::RGB8;
        const std::string ppm_path = stem + ".ppm";
        const std::string pgm_path = stem + (depth_word ? "_s.pgm" : "_a.pgm");
        FILE* ppm = std::fopen(ppm_path.c_str(), "wb");
        FILE* pgm = (has_alpha || depth_word) ? std::fopen(pgm_path.c_str(), "wb") : nullptr;
        if (ppm == nullptr) {
            if (pgm != nullptr) {
                std::fclose(pgm);
            }
            return;
        }
        std::string header = fmt::format("P6\n{} {}\n255\n", w, h);
        std::fwrite(header.data(), 1, header.size(), ppm);
        if (pgm != nullptr) {
            header = fmt::format("P5\n{} {}\n255\n", w, h);
            std::fwrite(header.data(), 1, header.size(), pgm);
        }
        std::vector<u8> rgb(w * 3);
        std::vector<u8> extra(w);
        for (u32 row = 0; row < h; row++) {
            const u8* src = staging.mapped.data() + static_cast<std::size_t>(h - 1 - row) * w * bpp;
            for (u32 x = 0; x < w; x++) {
                const u8* px = src + x * bpp;
                u8 r = 0, g = 0, b = 0, a = 255;
                switch (pf) {
                case PixelFormat::RGB5A1: {
                    u16 v;
                    std::memcpy(&v, px, 2);
                    r = static_cast<u8>(((v >> 11) & 31) * 255 / 31);
                    g = static_cast<u8>(((v >> 6) & 31) * 255 / 31);
                    b = static_cast<u8>(((v >> 1) & 31) * 255 / 31);
                    a = (v & 1) ? 255 : 0;
                    break;
                }
                case PixelFormat::RGB565: {
                    u16 v;
                    std::memcpy(&v, px, 2);
                    r = static_cast<u8>(((v >> 11) & 31) * 255 / 31);
                    g = static_cast<u8>(((v >> 5) & 63) * 255 / 63);
                    b = static_cast<u8>((v & 31) * 255 / 31);
                    break;
                }
                case PixelFormat::RGBA4: {
                    u16 v;
                    std::memcpy(&v, px, 2);
                    r = static_cast<u8>(((v >> 12) & 15) * 17);
                    g = static_cast<u8>(((v >> 8) & 15) * 17);
                    b = static_cast<u8>(((v >> 4) & 15) * 17);
                    a = static_cast<u8>((v & 15) * 17);
                    break;
                }
                case PixelFormat::D16: {
                    u16 v;
                    std::memcpy(&v, px, 2);
                    r = g = b = static_cast<u8>(v >> 8);
                    break;
                }
                case PixelFormat::D24: {
                    float f;
                    std::memcpy(&f, px, 4);
                    r = g = b = static_cast<u8>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
                    break;
                }
                case PixelFormat::D24S8: {
                    // GL order after Download: depth in the top 24 bits, stencil below.
                    u32 v;
                    std::memcpy(&v, px, 4);
                    r = g = b = static_cast<u8>(v >> 24);
                    a = static_cast<u8>(v & 0xff);
                    break;
                }
                default:
                    // Everything else is stored as bytes R,G,B,A.
                    r = px[0];
                    g = px[1];
                    b = px[2];
                    a = px[3];
                    break;
                }
                rgb[x * 3 + 0] = r;
                rgb[x * 3 + 1] = g;
                rgb[x * 3 + 2] = b;
                extra[x] = a;
            }
            std::fwrite(rgb.data(), 1, rgb.size(), ppm);
            if (pgm != nullptr) {
                std::fwrite(extra.data(), 1, extra.size(), pgm);
            }
        }
        std::fclose(ppm);
        if (pgm != nullptr) {
            std::fclose(pgm);
        }
    });
    if (index != nullptr) {
        std::fclose(index);
    }
    LOG_INFO(Render, "surface dump f{:02}: {} surfaces under ux0:data/azahar/dump", frame, count);
}


void RasterizerGxm::OpenSceneTrace(u32 addr, u16 width, u16 height, bool blit) {
    scene_open = SceneTrace{};
    scene_open.addr = addr;
    scene_open.width = width;
    scene_open.height = height;
    scene_open.blit = blit;
}

void RasterizerGxm::NoteScenePipeline(const char* vs, const char* fs) {
    scene_open.draws++;
    for (u8 i = 0; i < scene_open.names; i++) {
        if (scene_open.vs[i] == vs && scene_open.fs[i] == fs) {
            return;
        }
    }
    if (scene_open.names < scene_open.vs.size()) {
        scene_open.vs[scene_open.names] = vs;
        scene_open.fs[scene_open.names] = fs;
        scene_open.names++;
    }
}

void RasterizerGxm::CloseSceneTrace() {
    scene_open.serial = runtime.CurrentEpoch();
    scene_trace[scene_trace_next % SceneTraceLen] = scene_open;
    scene_trace_next++;
}

void RasterizerGxm::ReportSceneTrace() {
    // Read from the emulation thread while the render thread is parked in its wait: the ring
    // is not changing then, and a torn line is worth more than none.
    const u64 completed = runtime.CompletedEpoch();
    LOG_CRITICAL(Render, "gxm last scenes, GPU completed epoch {}, newest submitted {}:", completed,
                 runtime.CurrentEpoch());
    const u32 count = std::min(scene_trace_next, SceneTraceLen);
    for (u32 k = 0; k < count; k++) {
        const SceneTrace& t = scene_trace[(scene_trace_next - count + k) % SceneTraceLen];
        std::string names;
        for (u8 i = 0; i < t.names; i++) {
            names += fmt::format(" {}+{}", t.vs[i] ? t.vs[i] : "?", t.fs[i] ? t.fs[i] : "?");
        }
        LOG_CRITICAL(Render, "  {} {}: {:08X} {}x{} {} draws{}{}", t.serial,
                     t.serial <= completed ? "done" : "PENDING", t.addr, t.width, t.height,
                     t.draws, t.blit ? " (blit)" : "", names);
    }
}

} // namespace GxmRenderer
