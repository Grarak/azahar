// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include <psp2/display.h>
#include <psp2/gxm.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/razor_capture.h>
#include <psp2/razor_hud.h>
// In librazorhud and its stub, not yet in vitasdk's header.
extern "C" int sceRazorHudReloadSettings();
extern "C" int sceRazorGpuPerfGetMode();
extern "C" int sceRazorHudSetDisplayEnabled(int enabled);
#include <psp2/sysmodule.h>

#include "citra_vita/gxm_present.h"
#include "citra_vita/gxm_shaders.gen.h"
#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "video_core/pica/regs_external.h"
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_flags.h"

namespace VitaFrontend::GxmPresent {

namespace {

// Two display buffers and two frame arenas: AddEntry blocks once two flips are pending, and
// the per-slot fragment notification below proves the GPU is done with a slot's arena before
// it is reused, so two is enough and CDRAM is tight.
// Three display buffers: with two, the scene that draws into the back buffer waits (in the
// GPU's in-order fragment queue, so everything behind it waits too) until the buffer it
// replaces has flipped off the screen, a vblank-quantised bubble every frame - Razor showed
// the GPU busy 75-80% of the frame with the render thread waiting on it. The third buffer
// is the one that flipped two frames ago, off screen already. `doublebuf` restores two.
constexpr u32 MaxDisplayBuffers = 3;
u32 display_buffers = MaxDisplayBuffers;
constexpr u32 FrameArenaSize = 512 * 1024;
// The SDK's default. It was 2 MB while CDRAM was tight; a 3D scene here is ~30k triangles
// over four scenes, and a parameter buffer that fills makes the firmware render partially
// and reload every tile, which Razor showed as fragment jobs with the USSE a third busy.
constexpr u32 ParameterBufferSize = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE;

enum class MapKind { Regular, VertexUsse, FragmentUsse };

struct MappedBlock {
    SceUID uid; ///< memblock we allocated (guest memory, which we only mapped, has none)
    void* addr;
    u32 size;
    MapKind kind;
};


/// One memblock taken at init and mapped once, handed out first-fit: what vitaGL does with
/// the whole of free CDRAM and RAM at vglInit, so the budget is claimed up front and a
/// boot that does not fit fails at boot, not on the first level that needs a texture.
class MemPool {
public:
    bool Init(const char* pool_name, SceKernelMemBlockType type, u32 pool_size, u32 align) {
        name = pool_name;
        block_align = align;
        size = (pool_size + align - 1) & ~(align - 1);
        uid = sceKernelAllocMemBlock(name, type, size, nullptr);
        if (uid < 0) {
            return false;
        }
        void* addr = nullptr;
        sceKernelGetMemBlockBase(uid, &addr);
        if (sceGxmMapMemory(addr, size,
                            static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                                 SCE_GXM_MEMORY_ATTRIB_WRITE)) < 0) {
            sceKernelFreeMemBlock(uid);
            uid = -1;
            return false;
        }
        base = static_cast<u8*>(addr);
        spans = {{0, size, true}};
        return true;
    }
    [[nodiscard]] bool Ready() const noexcept {
        return base != nullptr;
    }
    /// The largest allocation this pool could still satisfy, counting the alignment an
    /// allocation would lose at the front of a span.
    [[nodiscard]] u32 LargestFree() const noexcept {
        u32 best = 0;
        for (const Span& s : spans) {
            if (!s.free) {
                continue;
            }
            const u32 start = (s.offset + block_align - 1) & ~(block_align - 1);
            const u32 end = s.offset + s.size;
            if (end > start) {
                best = std::max(best, end - start);
            }
        }
        return best;
    }
    void* Alloc(u32 want) {
        const u32 need = (want + 0xFFFu) & ~0xFFFu;
        for (std::size_t i = 0; i < spans.size(); i++) {
            Span& s = spans[i];
            if (!s.free) {
                continue;
            }
            const u32 start = (s.offset + block_align - 1) & ~(block_align - 1);
            if (start + need > s.offset + s.size) {
                continue;
            }
            const u32 lead = start - s.offset;
            const u32 tail = s.offset + s.size - (start + need);
            std::vector<Span> pieces;
            if (lead != 0) {
                pieces.push_back({s.offset, lead, true});
            }
            pieces.push_back({start, need, false});
            if (tail != 0) {
                pieces.push_back({start + need, tail, true});
            }
            spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(i));
            spans.insert(spans.begin() + static_cast<std::ptrdiff_t>(i), pieces.begin(),
                         pieces.end());
            used += need;
            return base + start;
        }
        return nullptr;
    }
    [[nodiscard]] bool Owns(const void* p) const noexcept {
        return base != nullptr && p >= base && p < base + size;
    }
    void Free(void* p) {
        const u32 offset = static_cast<u32>(static_cast<u8*>(p) - base);
        for (std::size_t i = 0; i < spans.size(); i++) {
            if (spans[i].offset != offset || spans[i].free) {
                continue;
            }
            spans[i].free = true;
            used -= spans[i].size;
            if (i + 1 < spans.size() && spans[i + 1].free) {
                spans[i].size += spans[i + 1].size;
                spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(i + 1));
            }
            if (i > 0 && spans[i - 1].free) {
                spans[i - 1].size += spans[i].size;
                spans.erase(spans.begin() + static_cast<std::ptrdiff_t>(i));
            }
            return;
        }
    }
    void Release() {
        if (base != nullptr) {
            sceGxmUnmapMemory(base);
            sceKernelFreeMemBlock(uid);
            base = nullptr;
        }
    }
    const char* name = "";
    u8* base{};
    u32 size{}, used{}, block_align = 4096;
    SceUID uid = -1;

private:
    struct Span {
        u32 offset, size;
        bool free;
    };
    std::vector<Span> spans;
};

struct State {
    bool initialized{};

    // Long-lived GPU memory we allocated (rings, display buffers, arenas, AllocMapped).
    std::vector<MappedBlock> owned;
    MemPool pool_cdram, pool_lpddr; ///< CreatePools: CDRAM and uncached LPDDR, up front
    // Foreign memory (guest RAM) we mapped but do not own.
    std::vector<MappedBlock> guest;

    void* context_host_mem{};
    SceGxmContext* context{};
    SceGxmRenderTarget* render_target{};

    void* display_mem[MaxDisplayBuffers]{};
    SceGxmColorSurface display_surface[MaxDisplayBuffers]{};
    SceGxmSyncObject* sync[MaxDisplayBuffers]{};

    SceGxmShaderPatcher* patcher{};

    // The fixed pipelines.
    SceGxmShaderPatcherId quad_v_id{}, quad_f_id{}, color_v_id{}, color_f_id{};
    SceGxmVertexProgram* quad_vp{};
    SceGxmFragmentProgram* quad_fp{};
    SceGxmVertexProgram* color_vp{};
    SceGxmFragmentProgram* color_fp{};
    const SceGxmProgramParameter* color_param{};

    void* quad_indices{}; ///< the one index buffer every quad uses

    // Frame state.
    u32 frame_index{};
    u32 back{}; ///< display buffer / arena slot being rendered
    u8* arena[MaxDisplayBuffers]{};
    u32 arena_used[MaxDisplayBuffers]{};
    volatile u32* notification[MaxDisplayBuffers]{};
    u32 notification_value[MaxDisplayBuffers]{};
    u32 notification_counter{};
    u32 notifications_taken{}; ///< how far into libgxm's notification region we have allocated
    bool in_scene{};
};

State g;

bool razor_loaded = false;
u32 razor_captures = 0;

// Every kernel memblock this file takes, with the running total per type, so a console log
// shows the partition being spent: retail units have ~365 MB in all and refuse a boot
// that asks for more, and the newlib heap (citra_vita_main.cpp) is the other big taker.
void LogMemBlock(const char* name, SceKernelMemBlockType type, SceSize size, SceUID uid) {
    static u64 totals[5]{};
    const char* kind = "other";
    int slot = 4;
    switch (type) {
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_RW: kind = "USER_RW"; slot = 0; break;
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE: kind = "USER_RW_UNCACHE"; slot = 1; break;
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW: kind = "CDRAM_RW"; slot = 2; break;
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW:
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW: kind = "PHYCONT"; slot = 3; break;
    default: break;
    }
    if (uid >= 0) {
        totals[slot] += size;
    }
    if (uid >= 0) {
        LOG_INFO(Frontend, "memblock {}: {} KiB {} -> {:#x}; {} so far {} KiB", name, size >> 10,
                 kind, static_cast<u32>(uid), kind, totals[slot] >> 10);
    }
}

// Razor GPU Live: the GPU's own job timings per frame, the way vitaGL reads them. Two
// result buffers, one per display buffer; a [gpu] line once a second.
//
// The hardware reports one counter group at a time, so a run picks one and the fields of
// the others stay zero. The gxm flags are gpulive0 (parameter buffer: peak usage, partial
// renders, paused vertex jobs), gpulive for the USSE overview, gpulive2 (pixels before and
// after hidden surface removal, vertex and primitive counts) and gpulive3 (tiling and ISP
// parameter fetch bytes). gpuhud draws the same numbers over the game and selects the USSE
// overview unless one of the others asked for a group.
struct GpuLive {
    bool on = false;
    void* buffer[MaxDisplayBuffers]{};
    static constexpr u32 BufferSize = 256 * 1024;
    u64 report_at_us = 0;
    u32 frames = 0, frame_us = 0, busy_us = 0, vertex_us = 0, fragment_us = 0, firmware_us = 0;
    u32 vertex_jobs = 0, fragment_jobs = 0;
    float usse_vertex = 0, usse_fragment = 0, dep_tex = 0, nondep_tex = 0;
    u32 px_before_hsr = 0, px_out = 0, vdm_verts = 0, mte_prims = 0;
    u32 tiling_writes = 0, isp_reads = 0, pbuf_peak = 0, partial = 0;
    /// A vertex job paused for want of parameter buffer space, and the jobs that were
    /// themselves partial renders. The SDK ranks the ways a scene can be split: a paused
    /// vertex job is the cheap one, a partial render the middling one, and a fragment ring
    /// buffer that runs out is the expensive one, since it processes every tile again.
    u32 vertex_paused = 0, partial_jobs = 0;
    u32 entries = 0, set_failures = 0;
    int last_set_error = 0;
    /// Fragment time, jobs and pixels by the scene's index within its frame, so the render
    /// targets can be told apart: the renderer opens its scenes in order and the
    /// presentation layer's is the last of each frame.
    static constexpr u32 Scenes = 12;
    u32 scene_fragment_us[Scenes]{};
    u32 scene_jobs[Scenes]{};
    u32 scene_pixels[Scenes]{};
    /// Where the GPU had nothing to do: before the frame's first job, between jobs, and
    /// after the last one. Every job carries a start and an end, so the gaps are exact.
    u64 idle_before_us = 0, idle_between_us = 0, idle_after_us = 0;
    u32 max_gap_us = 0, max_gap_scene = 0, gaps = 0;
} live;
/// The last second's summary for the on-screen HUD (gxm flag gpuhud), rewritten with the
/// [gpu] line; read on the presenting thread only, like everything here.
std::string live_text;

/// One GPU job's span, for the gap arithmetic below.
struct JobSpan {
    u64 start, end;
    u16 scene;
};

#ifdef VITA_DIAGNOSTICS
// Reads the GPU's own counters for the last second and prints them. Diagnostic only:
// the release build neither starts the counters nor carries the strings.
void GpuLiveFrame() {
    SceRazorGpuLiveResultInfo res{};
    JobSpan spans[128];
    u32 span_count = 0;
    u64 frame_start = 0, frame_end = 0;
    const int set = sceRazorGpuLiveSetBuffer(live.buffer[g.frame_index % MaxDisplayBuffers],
                                             GpuLive::BufferSize, &res);
    if (set < 0 || res.result_data == nullptr) {
        // Reported with the next line rather than silently: which of the two it is says
        // whether the module is there at all or just has nothing for this process.
        live.set_failures++;
        live.last_set_error = set;
        res.entry_count = 0;
    }
    live.entries += res.entry_count;
    const SceUID pid = sceKernelGetProcessId();
    const u8* ptr = static_cast<const u8*>(res.result_data);
    for (u32 i = 0; i < res.entry_count; i++) {
        const auto* header = reinterpret_cast<const SceRazorGpuLiveEntryHeader*>(ptr);
        if (header->entry_size == 0) {
            break;
        }
        switch (header->entry_type) {
        case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_JOB: {
            const auto* job = reinterpret_cast<const SceRazorGpuLiveEntryJob*>(ptr);
            const u32 us = static_cast<u32>(job->end_time - job->start_time);
            if (span_count < std::size(spans) && job->end_time > job->start_time) {
                spans[span_count++] = {job->start_time, job->end_time, job->scene_index};
            }
            // vitasdk names the last word of the entry `unk`; the SDK header has it as a
            // draw count it never fills in and a flag word, whose first bit marks a job that
            // was a partial render. The struct sizes agree, so the halves are these.
            if ((static_cast<u32>(job->unk) >> 16) & 1u) {
                live.partial_jobs++;
            }
            if (job->type == SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FIRMWARE) {
                live.firmware_us += us;
            } else if (job->process_id == static_cast<u32>(pid)) {
                switch (job->type) {
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX0:
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX1:
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX2:
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX3:
                    live.vertex_jobs++;
                    live.vertex_us += us;
                    break;
                default:
                    live.fragment_jobs++;
                    live.fragment_us += us;
                    if (job->scene_index < GpuLive::Scenes) {
                        live.scene_fragment_us[job->scene_index] += us;
                        live.scene_jobs[job->scene_index]++;
                    }
                    break;
                }
                const auto& v = job->job_values;
                switch (job->type) {
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX1:
                    live.usse_vertex += v.vertex_values_type1.usse_vertex_processing_percent;
                    break;
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT1:
                    live.usse_fragment += v.fragment_values_type1.usse_fragment_processing_percent;
                    live.dep_tex += v.fragment_values_type1.usse_dependent_texture_reads_percent;
                    live.nondep_tex +=
                        v.fragment_values_type1.usse_non_dependent_texture_reads_percent;
                    break;
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX2:
                    live.vdm_verts += v.vertex_values_type2.vdm_vertices_input_num;
                    live.mte_prims += v.vertex_values_type2.mte_primitives_output_num;
                    break;
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT2:
                    live.px_before_hsr += v.fragment_values_type2.rasterized_pixels_before_hsr_num;
                    live.px_out += v.fragment_values_type2.rasterized_output_pixels_num;
                    if (job->scene_index < GpuLive::Scenes) {
                        live.scene_pixels[job->scene_index] +=
                            v.fragment_values_type2.rasterized_output_pixels_num;
                    }
                    break;
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX3:
                    live.tiling_writes += v.vertex_values_type3.tiling_accelerated_mem_writes;
                    break;
                case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT3:
                    live.isp_reads += v.fragment_values_type3.isp_parameter_fetches_mem_reads;
                    break;
                default:
                    break;
                }
            }
            break;
        }
        case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_PARAMETER_BUFFER: {
            const auto* pb = reinterpret_cast<const SceRazorGpuLiveEntryParameterBuffer*>(ptr);
            live.pbuf_peak = std::max(live.pbuf_peak, pb->peak_usage_value);
            live.partial += pb->partial_render;
            live.vertex_paused += pb->vertex_job_paused;
            break;
        }
        case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_FRAME: {
            const auto* frame = reinterpret_cast<const SceRazorGpuLiveEntryFrame*>(ptr);
            frame_start = frame->start_time;
            frame_end = frame->start_time + frame->duration;
            live.frames++;
            live.frame_us += frame->duration;
            live.busy_us += frame->gpu_activity_duration_time;
            break;
        }
        default:
            break;
        }
        ptr += header->entry_size;
    }
    // The gaps in the GPU's work: sort the jobs by start, walk them keeping the furthest
    // end seen, and every jump forward is time the GPU had nothing queued. The scene of the
    // job that ends the gap says what it was waiting for.
    if (span_count > 0) {
        std::sort(spans, spans + span_count,
                  [](const JobSpan& a, const JobSpan& b) { return a.start < b.start; });
        u64 cursor = spans[0].start;
        for (u32 i = 0; i < span_count; i++) {
            if (spans[i].start > cursor) {
                const u64 gap = spans[i].start - cursor;
                live.idle_between_us += gap;
                live.gaps++;
                if (gap > live.max_gap_us) {
                    live.max_gap_us = static_cast<u32>(gap);
                    live.max_gap_scene = spans[i].scene;
                }
            }
            cursor = std::max(cursor, spans[i].end);
        }
        if (frame_start != 0 && spans[0].start > frame_start) {
            live.idle_before_us += spans[0].start - frame_start;
        }
        if (frame_end > cursor) {
            live.idle_after_us += frame_end - cursor;
        }
    }

    const u64 now = sceKernelGetProcessTimeWide();
    if (now < live.report_at_us) {
        return;
    }
    live.report_at_us = now + 1000000;
    const u32 n = std::max(live.frames, 1u);
    const u32 fj = std::max(live.fragment_jobs, 1u);
    const u32 vj = std::max(live.vertex_jobs, 1u);
    if (live.frames == 0) {
        sceClibPrintf("[gpu] no frame entries: %u entries in all, set buffer failed %u times "
                      "(last %#x)\n",
                      live.entries, live.set_failures, static_cast<unsigned>(live.last_set_error));
    }
    const u32 frame_us = live.frame_us / n;
    const u32 busy_us = live.busy_us / n;
    sceClibPrintf("[gpu] %u frames: per frame %u us, gpu busy %u us, vertex %u us (%u jobs), "
                  "fragment %u us (%u jobs), firmware %u us | usse vertex %u%% fragment %u%% "
                  "dependent tex %u%% non-dependent %u%% | pixels before hsr %u after %u, "
                  "vertices %u, prims %u | tiling writes %u KiB, isp reads %u KiB, "
                  "pbuf peak %u KiB, partial renders %u (%u jobs), vertex jobs paused %u\n",
                  live.frames, frame_us, busy_us, live.vertex_us / n, live.vertex_jobs,
                  live.fragment_us / n, live.fragment_jobs, live.firmware_us / n,
                  static_cast<u32>(live.usse_vertex / vj), static_cast<u32>(live.usse_fragment / fj),
                  static_cast<u32>(live.dep_tex / fj), static_cast<u32>(live.nondep_tex / fj),
                  live.px_before_hsr / n, live.px_out / n, live.vdm_verts / n, live.mte_prims / n,
                  live.tiling_writes / n / 1024, live.isp_reads / n / 1024, live.pbuf_peak / 1024,
                  live.partial, live.partial_jobs, live.vertex_paused);
    // Where the fragment time goes, by the scene's place in its frame, in the order the
    // renderer opened them: a scene that costs most of the frame can be found this way.
    char scenes[256];
    int at = 0;
    for (u32 i = 0; i < GpuLive::Scenes && at < static_cast<int>(sizeof(scenes)) - 40; i++) {
        if (live.scene_jobs[i] == 0) {
            continue;
        }
        at += std::snprintf(scenes + at, sizeof(scenes) - at, " %u: %u us %u px,", i,
                            live.scene_fragment_us[i] / n, live.scene_pixels[i] / n);
    }
    if (at > 0) {
        scenes[at - 1] = '\0';
        sceClibPrintf("[gpu] fragment us and pixels per frame by scene:%s\n", scenes);
    }
    sceClibPrintf("[gpu] idle per frame: %llu us before the first job, %llu us in %u gaps "
                  "between jobs, %llu us after the last; the biggest gap was %u us, ending "
                  "at a job of scene %u\n",
                  live.idle_before_us / n, live.idle_between_us / n, live.gaps / n,
                  live.idle_after_us / n, live.max_gap_us, live.max_gap_scene);
    live_text = fmt::format(
        "GPU {} frames/s: {} us per frame, busy {} us ({}%)\n"
        "vertex {} us ({} jobs)  fragment {} us ({} jobs)  firmware {} us\n"
        "usse vertex {}%  fragment {}%  tex dependent {}%  non-dependent {}%\n"
        "pixels {} before hsr, {} out  vertices {}  prims {}\n"
        "tiling {} KiB  isp {} KiB  pbuf peak {} KiB\n"
        "partial renders {} ({} jobs)  vertex jobs paused {}",
        live.frames, frame_us, busy_us, frame_us ? busy_us * 100 / frame_us : 0,
        live.vertex_us / n, live.vertex_jobs, live.fragment_us / n, live.fragment_jobs,
        live.firmware_us / n, static_cast<u32>(live.usse_vertex / vj),
        static_cast<u32>(live.usse_fragment / fj), static_cast<u32>(live.dep_tex / fj),
        static_cast<u32>(live.nondep_tex / fj), live.px_before_hsr / n, live.px_out / n,
        live.vdm_verts / n, live.mte_prims / n, live.tiling_writes / n / 1024,
        live.isp_reads / n / 1024, live.pbuf_peak / 1024, live.partial, live.partial_jobs,
        live.vertex_paused);
    if (live.frames == 0) {
        live_text = fmt::format("GPU live: no frame entries ({} entries, set buffer failed {} "
                                "times, last {:#x})",
                                live.entries, live.set_failures,
                                static_cast<u32>(live.last_set_error));
    }
    const bool on = live.on;
    void* buffers[MaxDisplayBuffers];
    std::memcpy(buffers, live.buffer, sizeof(buffers));
    live = GpuLive{};
    live.on = on;
    std::memcpy(live.buffer, buffers, sizeof(buffers));
    live.report_at_us = now + 1000000;
}
#else
void GpuLiveFrame() {}
#endif

u32 GpuLiveGroup() {
#ifndef VITA_DIAGNOSTICS
    // No counters in the release build, so the module is never loaded either.
    return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_NUM;
#else
    // One group per run, so the first flag named here wins. gpuhud comes last because it is
    // about drawing the numbers rather than about which ones to collect.
    if (GxmRenderer::GxmFlag("gpulive0")) {
        return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_PBUFFER_USAGE;
    }
    if (GxmRenderer::GxmFlag("gpulive2")) {
        return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_OVERVIEW_2;
    }
    if (GxmRenderer::GxmFlag("gpulive3")) {
        return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_OVERVIEW_3;
    }
    if (GxmRenderer::GxmFlag("gpulive") || GxmRenderer::GxmFlag("gpuhud")) {
        return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_OVERVIEW_1;
    }
    return SCE_RAZOR_GPU_LIVE_METRICS_GROUP_NUM;
#endif
}

/// The HUD module hooks libgxm when libgxm initialises, so it has to be loaded before
/// sceGxmInitialize (loaded after, every call answers NOT_INITIALIZED, perf mode 0); the
/// live metrics are started after, as vitaGL orders both.
int hud_module = -1;
void LoadGpuLiveModule() {
    if (GpuLiveGroup() != SCE_RAZOR_GPU_LIVE_METRICS_GROUP_NUM) {
        hud_module = sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_HUD);
    }
}

void StartGpuLive() {
    const u32 group = GpuLiveGroup();
    if (group == SCE_RAZOR_GPU_LIVE_METRICS_GROUP_NUM) {
        return;
    }
    const int hud = hud_module;
    for (u32 i = 0; i < MaxDisplayBuffers; i++) {
        const SceUID uid = sceKernelAllocMemBlock("azahar-gpulive", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
                                                  GpuLive::BufferSize, nullptr);
        LogMemBlock("azahar-gpulive", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, GpuLive::BufferSize, uid);
        if (uid < 0 || sceKernelGetMemBlockBase(uid, &live.buffer[i]) < 0) {
            LOG_WARNING(Frontend, "GPU Live: no result buffer ({:#x})", static_cast<u32>(uid));
            return;
        }
        sceGxmMapMemory(live.buffer[i], GpuLive::BufferSize,
                        static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                             SCE_GXM_MEMORY_ATTRIB_WRITE));
    }
    // The perf mode comes from app0:hud_settings.ini (LIVE=1 selects live metrics, mode
    // 3); it is read when the module starts, and reloaded here in case that was too early.
    const int mode_before = sceRazorGpuPerfGetMode();
    int reload = sceRazorHudReloadSettings();
    if (reload == static_cast<int>(0x8008c008)) { // SCE_RAZOR_GPU_ERROR_SETTINGS_NOT_FOUND
        // app0:hud_settings.ini is not where this firmware's module looks. Probe: the same
        // file at each candidate, a reload after each, the first that is found is logged.
        static const char* const candidates[] = {
            "ux0:data/hud_settings.ini",  "ux0:hud_settings.ini",
            "ux0:data/azahar/hud_settings.ini", "ur0:data/hud_settings.ini",
            "ux0:temp/hud_settings.ini",  "host0:hud_settings.ini",
            "ur0:hud_settings.ini",       "ux0:app/hud_settings.ini",
        };
        // The module's own on-screen HUD (GPU_HUD = 1) never drew anything on this
        // firmware; the gpuhud flag draws the live counters with imgui instead (vita_ui).
        const std::string settings = "ENABLE = 1\nLIVE = 1\nGPU_HUD = 0\nCPU_HUD = 0\n"
                                     "HUD_DISPLAY_FRAMES = 2\nHUD_FIRMWARE = 1\n";
        for (const char* path : candidates) {
            FILE* f = std::fopen(path, "wb");
            if (f == nullptr) {
                LOG_WARNING(Frontend, "GPU Live: cannot write {}", path);
                continue;
            }
            std::fwrite(settings.data(), 1, settings.size(), f);
            std::fclose(f);
            reload = sceRazorHudReloadSettings();
            LOG_WARNING(Frontend, "GPU Live: settings at {} -> reload {:#x} mode {}", path,
                        static_cast<u32>(reload), sceRazorGpuPerfGetMode());
            if (reload >= 0) {
                break;
            }
        }
    }
    const int mode = sceRazorGpuPerfGetMode();
    const int set = sceRazorGpuLiveSetMetricsGroup(group);
    const int start = sceRazorGpuLiveStart();
    live.on = set >= 0 && start >= 0;
    LOG_WARNING(Frontend,
                "GPU Live: hud {:#x} perf mode {} -> {} after reload ({:#x}), group {} set {:#x} "
                "start {:#x} -> {}",
                static_cast<u32>(hud), mode_before, mode, static_cast<u32>(reload), group,
                static_cast<u32>(set), static_cast<u32>(start), live.on ? "on" : "off");
}
/// What a presented frame costs this thread, reported once a second: the wait for the
/// display slot the frame is about to draw into (the GPU must be done with the buffer that
/// was on screen two flips ago), the two scene calls, and handing the frame to the display
/// queue. The vertical blank the queue callback waits for is on libgxm's own thread and is
/// counted apart.
struct PresentTimes {
    u64 report_at_us = 0;
    u32 frames = 0, slot_waits = 0, callbacks = 0;
    u64 slot_wait_us = 0, begin_scene_us = 0, end_scene_us = 0, queue_us = 0, callback_us = 0;
} ptimes;

/// Microseconds from here to the destructor, into `into`.
struct PresentTimer {
    u64& into;
    u64 start = sceKernelGetProcessTimeWide();
    explicit PresentTimer(u64& t) : into{t} {}
    ~PresentTimer() {
        into += sceKernelGetProcessTimeWide() - start;
    }
};

bool dump_requested = false; ///< DumpNextFrame: the next EndFrame writes its picture

// The frame just drawn as a PPM, after waiting for the GPU. A frame is 1.5 MB.
void WriteDisplayPpm(const char* path) {
    sceGxmFinish(g.context);
    FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        return;
    }
    std::fprintf(f, "P6\n%u %u\n255\n", DisplayWidth, DisplayHeight);
    const u8* px = static_cast<const u8*>(g.display_mem[g.back]);
    std::vector<u8> row(DisplayWidth * 3);
    for (u32 y = 0; y < DisplayHeight; y++) {
        const u8* src = px + static_cast<std::size_t>(y) * DisplayWidth * 4;
        for (u32 x = 0; x < DisplayWidth; x++) {
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

/// The Razor GPU capture module, the way vitaGL loads it: the system modules on a devkit,
/// the user-installed suprx otherwise; whichever succeeds. Must precede sceGxmInitialize.
void LoadRazor() {
    if (!GxmRenderer::GxmFlag("razor")) {
        return;
    }
    const int hud = sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_HUD);
    const int cap = sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_CAPTURE);
    SceUID mod = -1;
    if (cap < 0) {
        mod = sceKernelLoadStartModule("ur0:data/librazorcapture_es4.suprx", 0, nullptr, 0,
                                       nullptr, nullptr);
    }
    razor_loaded = cap >= 0 || mod >= 0;
    LOG_WARNING(Frontend, "Razor: hud {:#x} capture {:#x} suprx {:#x} -> {}",
                static_cast<u32>(hud), static_cast<u32>(cap), static_cast<u32>(mod),
                razor_loaded ? "loaded" : "unavailable");
    if (razor_loaded) {
        sceRazorGpuCaptureEnableSalvage("ux0:data/azahar/gpucrash.sgx");
    }
}


struct DisplayData {
    void* addr;
};

void DisplayCallback(const void* data) {
    const auto* display = static_cast<const DisplayData*>(data);
    SceDisplayFrameBuf fb{};
    fb.size = sizeof(fb);
    fb.base = display->addr;
    fb.pitch = DisplayWidth;
    fb.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
    fb.width = DisplayWidth;
    fb.height = DisplayHeight;
    const PresentTimer timer{ptimes.callback_us};
    ptimes.callbacks++;
    sceDisplaySetFrameBuf(&fb, SCE_DISPLAY_SETBUF_NEXTFRAME);
    // Returning before the flip lands would let the GPU write the buffer still on screen -
    // the SDK is explicit that this wait is correctness, not politeness.
    sceDisplayWaitVblankStart();
}

/// What the pool behind alloc_cdram / alloc_mapped could still hand out in one piece.
u32 LargestFree(bool cdram) {
    const MemPool& pool = cdram ? g.pool_cdram : g.pool_lpddr;
    return pool.Ready() ? pool.LargestFree() : 0;
}

void* GpuAlloc(SceKernelMemBlockType type, u32 size, SceGxmMemoryAttribFlags attribs,
               SceUID* out_uid) {
    // The pools first; a memblock of its own only when the pool is out (or was never made).
    MemPool* pool = type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW        ? &g.pool_cdram
                    : type == SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE ? &g.pool_lpddr
                                                                        : nullptr;
    if (pool != nullptr && pool->Ready()) {
        if (void* addr = pool->Alloc(size)) {
            *out_uid = pool->uid;
            LOG_INFO(Frontend, "pool {}: {} KiB -> {} of {} KiB used", pool->name, size >> 10,
                     pool->used >> 10, pool->size >> 10);
            return addr;
        }
        // The caller halves its ask and comes back, and once a pool is full it stays full,
        // so this is said a few times and then only occasionally.
        static u64 pool_out_reports = 0;
        if (++pool_out_reports <= 4 || pool_out_reports % 1024 == 0) {
            LOG_WARNING(Frontend, "pool {} is out: {} KiB wanted, {} of {} KiB used, {} times",
                        pool->name, size >> 10, pool->used >> 10, pool->size >> 10,
                        pool_out_reports);
        }
    }
    const u32 align = type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW ? 256 * 1024
                      : type == SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW ? 1024 * 1024
                                                                                : 4 * 1024;
    size = (size + align - 1) & ~(align - 1);
    SceUID uid = sceKernelAllocMemBlock("azahar-gxm", type, size, nullptr);
    LogMemBlock("azahar-gxm", type, size, uid);
    if (uid < 0) {
        static u64 refusals = 0;
        if (++refusals <= 4 || refusals % 1024 == 0) {
            LOG_WARNING(Frontend, "GXM memblock of {} bytes refused: {:#x}, {} times", size,
                        static_cast<u32>(uid), refusals);
        }
        return nullptr;
    }
    void* addr = nullptr;
    sceKernelGetMemBlockBase(uid, &addr);
    if (sceGxmMapMemory(addr, size, attribs) < 0) {
        sceKernelFreeMemBlock(uid);
        return nullptr;
    }
    *out_uid = uid;
    g.owned.push_back({uid, addr, size, MapKind::Regular});
    return addr;
}

/// One 32-bit slot from libgxm's notification region, handed out in order so the renderer
/// (through GxmDevice::alloc_notification) cannot land on a slot this file is using. The
/// region holds SCE_GXM_NOTIFICATION_COUNT of them and we need a handful.
volatile u32* AllocNotification() {
    constexpr u32 NotificationCount = 512;
    if (g.notifications_taken >= NotificationCount) {
        LOG_ERROR(Frontend, "GXM: the notification region is exhausted");
        return nullptr;
    }
    return sceGxmGetNotificationRegion() + g.notifications_taken++;
}

/// vitaGL's rule at vglInit: everything free, less a margin, into pools. CDRAM entire (128 MB
/// the game partition does not count, less a little for libgxm); uncached LPDDR everything
/// the partition still has less what threads, modules and the vertex compiler need.
void CreatePools() {
    SceKernelFreeMemorySizeInfo info{};
    info.size = sizeof(info);
    sceKernelGetFreeMemorySize(&info);
    constexpr u32 CdramMargin = 2 * 1024 * 1024;
    constexpr u32 LpddrMargin = 32 * 1024 * 1024;
    const auto make = [](MemPool& pool, const char* name, SceKernelMemBlockType type, u32 want,
                         u32 align) {
        // A refusal halves the ask: what the kernel reports free and what one block can be
        // are not the same thing.
        for (u32 size = want & ~(align - 1); size >= 8 * 1024 * 1024; size /= 2) {
            if (pool.Init(name, type, size, align)) {
                return;
            }
        }
    };
    if (info.size_cdram > CdramMargin) {
        make(g.pool_cdram, "azahar-pool-cdram", SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
             static_cast<u32>(info.size_cdram) - CdramMargin, 256 * 1024);
    }
    if (info.size_user > LpddrMargin) {
        make(g.pool_lpddr, "azahar-pool-lpddr", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
             static_cast<u32>(info.size_user) - LpddrMargin, 4096);
    }
    LOG_INFO(Frontend,
             "GPU pools: CDRAM {} MiB of {} free, uncached LPDDR {} MiB of {} free user "
             "({} MiB phycont free, untouched)",
             g.pool_cdram.size >> 20, info.size_cdram >> 20, g.pool_lpddr.size >> 20,
             info.size_user >> 20, info.size_phycont >> 20);
}

void* GpuAllocUncached(u32 size) {
    SceUID uid;
    return GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size,
                    static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                         SCE_GXM_MEMORY_ATTRIB_WRITE),
                    &uid);
}

/// Memory the GPU works and the CPU only fills now and then (libgxm's rings, surfaces):
/// CDRAM first, which is 128 MB the game partition does not count, as vitaGL places them;
/// then uncached LPDDR; then the physically contiguous pool, which is another 26 MB outside
/// the partition. `nocdram` keeps such memory in LPDDR, for measuring the CPU's writes.
void* GpuAllocForGpu(u32 size) {
    constexpr auto rw = static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                             SCE_GXM_MEMORY_ATTRIB_WRITE);
    SceUID uid;
    void* addr = nullptr;
    {
        addr = GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, size, rw, &uid);
    }
    if (addr == nullptr) {
        addr = GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, rw, &uid);
    }
    if (addr == nullptr) {
        addr = GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW,
                        (size + 0xFFFFFu) & ~0xFFFFFu, rw, &uid);
    }
    return addr;
}

// Patcher host memory is plain heap.
void* PatcherHostAlloc(void*, u32 size) {
    return std::malloc(size);
}
void PatcherHostFree(void*, void* mem) {
    std::free(mem);
}

/**
 * The patcher's GPU-side heaps, grown on demand.
 *
 * A static region can be handed to sceGxmShaderPatcherCreate instead, and that is what this
 * did: 64 KiB of fragment USSE. It is enough for the handful of fixed programs and nothing
 * else - a title with many PICA configurations exhausts it and every later
 * sceGxmShaderPatcherCreateFragmentProgram returns SCE_GXM_ERROR_OUT_OF_FRAGMENT_USSE_MEMORY,
 * which is a draw that never appears. So the patcher gets callbacks and this allocator gets
 * a new memblock whenever the live ones cannot fit a program.
 *
 * USSE offsets are linear within one mapping (that is the premise the patcher's own internal
 * heap rests on), so a block at `base + n` is addressed at `usse_offset + n`. Chunks are
 * mapped once, whole, and never unmapped until shutdown; only the sub-allocations come and go.
 */
class PatcherHeap {
public:
    enum class Kind { Buffer, VertexUsse, FragmentUsse };

    PatcherHeap(Kind kind_, u32 chunk_size_) : kind{kind_}, chunk_size{chunk_size_} {}

    void* Alloc(u32 size, u32* usse_offset) {
        size = (size + 63u) & ~63u;
        for (int attempt = 0; attempt < 2; attempt++) {
            for (Chunk& chunk : chunks) {
                for (auto it = chunk.free.begin(); it != chunk.free.end(); ++it) {
                    if (it->second < size) {
                        continue;
                    }
                    const u32 start = it->first;
                    const u32 rest = it->second - size;
                    chunk.free.erase(it);
                    if (rest > 0) {
                        chunk.free.emplace(start + size, rest);
                    }
                    u8* ptr = chunk.base + start;
                    live.emplace(ptr, std::make_pair(&chunk - chunks.data(), size));
                    used += size;
                    if (usse_offset != nullptr) {
                        *usse_offset = chunk.usse_offset + start;
                    }
                    return ptr;
                }
            }
            if (!AddChunk(size)) {
                break;
            }
        }
        LOG_ERROR(Frontend, "GXM patcher {} heap: {} bytes refused, {} KiB live in {} chunks",
                  Name(), size, used / 1024, chunks.size());
        return nullptr;
    }

    void Free(void* ptr) {
        const auto it = live.find(static_cast<u8*>(ptr));
        if (it == live.end()) {
            return;
        }
        Chunk& chunk = chunks[it->second.first];
        const u32 size = it->second.second;
        u32 start = static_cast<u32>(it->first - chunk.base);
        live.erase(it);
        used -= size;
        // Coalesce with the neighbours so a long run of build-and-release does not shred
        // the chunk into unusable slivers.
        u32 end = start + size;
        auto next = chunk.free.lower_bound(start);
        if (next != chunk.free.end() && next->first == end) {
            end += next->second;
            next = chunk.free.erase(next);
        }
        if (next != chunk.free.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == start) {
                start = prev->first;
                chunk.free.erase(prev);
            }
        }
        chunk.free.emplace(start, end - start);
    }

    void Release() {
        // The chunks themselves are in g.owned and are unmapped and freed with the rest.
        chunks.clear();
        live.clear();
        used = 0;
    }

    [[nodiscard]] u32 Reserved() const noexcept {
        u32 total = 0;
        for (const Chunk& chunk : chunks) {
            total += chunk.size;
        }
        return total;
    }

private:
    struct Chunk {
        u8* base{};
        u32 size{};
        u32 usse_offset{};
        std::map<u32, u32> free; ///< offset -> size
    };

    [[nodiscard]] const char* Name() const noexcept {
        switch (kind) {
        case Kind::Buffer:
            return "buffer";
        case Kind::VertexUsse:
            return "vertex USSE";
        default:
            return "fragment USSE";
        }
    }

    bool AddChunk(u32 min_size) {
        const u32 size = (std::max(chunk_size, min_size) + 0xFFFu) & ~0xFFFu;
        Chunk chunk;
        chunk.size = size;
        if (kind == Kind::Buffer) {
            chunk.base = static_cast<u8*>(GpuAllocUncached(size));
            if (chunk.base == nullptr) {
                return false;
            }
        } else {
            const SceUID uid = sceKernelAllocMemBlock(
                "azahar-gxm-usse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, nullptr);
            LogMemBlock("azahar-gxm-usse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, uid);
            if (uid < 0) {
                return false;
            }
            void* addr = nullptr;
            sceKernelGetMemBlockBase(uid, &addr);
            const MapKind map_kind =
                kind == Kind::VertexUsse ? MapKind::VertexUsse : MapKind::FragmentUsse;
            const int err = map_kind == MapKind::VertexUsse
                                ? sceGxmMapVertexUsseMemory(addr, size, &chunk.usse_offset)
                                : sceGxmMapFragmentUsseMemory(addr, size, &chunk.usse_offset);
            if (err < 0) {
                sceKernelFreeMemBlock(uid);
                return false;
            }
            chunk.base = static_cast<u8*>(addr);
            g.owned.push_back({uid, addr, size, map_kind});
        }
        chunk.free.emplace(0u, size);
        // Growing the vector moves the chunks; the live map holds indices, not pointers, so
        // only the Chunk& taken during Alloc matters and that call is already finished.
        chunks.push_back(std::move(chunk));
        LOG_INFO(Frontend, "GXM patcher {} heap grew to {} KiB in {} chunks ({} KiB live)",
                 Name(), Reserved() / 1024, chunks.size(), used / 1024);
        return true;
    }

    Kind kind;
    u32 chunk_size;
    u32 used{};
    std::vector<Chunk> chunks;
    std::map<u8*, std::pair<std::size_t, u32>> live; ///< ptr -> (chunk index, size)
};

PatcherHeap patcher_buffer_heap{PatcherHeap::Kind::Buffer, 256 * 1024};
PatcherHeap patcher_vertex_usse_heap{PatcherHeap::Kind::VertexUsse, 128 * 1024};
PatcherHeap patcher_fragment_usse_heap{PatcherHeap::Kind::FragmentUsse, 256 * 1024};

void* PatcherBufferAlloc(void*, u32 size) {
    return patcher_buffer_heap.Alloc(size, nullptr);
}
void PatcherBufferFree(void*, void* mem) {
    patcher_buffer_heap.Free(mem);
}
void* PatcherVertexUsseAlloc(void*, u32 size, u32* usse_offset) {
    return patcher_vertex_usse_heap.Alloc(size, usse_offset);
}
void PatcherVertexUsseFree(void*, void* mem) {
    patcher_vertex_usse_heap.Free(mem);
}
void* PatcherFragmentUsseAlloc(void*, u32 size, u32* usse_offset) {
    return patcher_fragment_usse_heap.Alloc(size, usse_offset);
}
void PatcherFragmentUsseFree(void*, void* mem) {
    patcher_fragment_usse_heap.Free(mem);
}

/// A vendored program, rejected loudly if it is not a GXP the firmware will take. These are
/// compiled offline by the SDK's own psp2cgc (shaders/compile.sh), so a failure here means a
/// corrupt or mis-generated blob, not anything the console did.
const SceGxmProgram* FixedProgram(const char* name, const SceGxmProgram* program) {
    if (sceGxmProgramCheck(program) != 0) {
        LOG_CRITICAL(Frontend, "the vendored '{}' program is not a valid GXP", name);
        return nullptr;
    }
    return program;
}

/// The clip-space transform every quad uses: display pixels in, clip coordinates out.
/// (If bring-up shows the picture upside down, this y sign is the one to flip.)
inline float ClipX(float px) {
    return px * (2.0f / DisplayWidth) - 1.0f;
}
inline float ClipY(float py) {
    return 1.0f - py * (2.0f / DisplayHeight);
}

struct QuadVertex {
    float x, y, u, v;
};

bool BuildFixedPipelines() {
    const SceGxmProgram* quad_v = FixedProgram("quad_v", GxmShader_quad_v());
    const SceGxmProgram* quad_f = FixedProgram("quad_f", GxmShader_quad_f());
    const SceGxmProgram* color_v = FixedProgram("color_v", GxmShader_color_v());
    const SceGxmProgram* color_f = FixedProgram("color_f", GxmShader_color_f());
    if (!quad_v || !quad_f || !color_v || !color_f) {
        return false;
    }

    sceGxmShaderPatcherRegisterProgram(g.patcher, quad_v, &g.quad_v_id);
    sceGxmShaderPatcherRegisterProgram(g.patcher, quad_f, &g.quad_f_id);
    sceGxmShaderPatcherRegisterProgram(g.patcher, color_v, &g.color_v_id);
    sceGxmShaderPatcherRegisterProgram(g.patcher, color_f, &g.color_f_id);

    const auto attribute = [](const SceGxmProgram* program, const char* name, u16 offset,
                              SceGxmAttributeFormat format, u8 components) {
        const SceGxmProgramParameter* param = sceGxmProgramFindParameterByName(program, name);
        SceGxmVertexAttribute attr{};
        attr.streamIndex = 0;
        attr.offset = offset;
        attr.format = static_cast<u8>(format);
        attr.componentCount = components;
        attr.regIndex = sceGxmProgramParameterGetResourceIndex(param);
        return attr;
    };

    // Textured quad: x, y, u, v as floats.
    {
        SceGxmVertexAttribute attrs[2] = {
            attribute(quad_v, "aPosition", 0, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2),
            attribute(quad_v, "aTexcoord", 8, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2),
        };
        SceGxmVertexStream stream{};
        stream.stride = sizeof(QuadVertex);
        stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
        if (sceGxmShaderPatcherCreateVertexProgram(g.patcher, g.quad_v_id, attrs, 2, &stream, 1,
                                                   &g.quad_vp) < 0) {
            return false;
        }
        if (sceGxmShaderPatcherCreateFragmentProgram(
                g.patcher, g.quad_f_id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                SCE_GXM_MULTISAMPLE_NONE, nullptr, quad_v, &g.quad_fp) < 0) {
            return false;
        }
    }

    // Solid color: x, y as floats; the color is a fragment uniform.
    {
        SceGxmVertexAttribute attr =
            attribute(color_v, "aPosition", 0, SCE_GXM_ATTRIBUTE_FORMAT_F32, 2);
        SceGxmVertexStream stream{};
        stream.stride = sizeof(float) * 2;
        stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
        if (sceGxmShaderPatcherCreateVertexProgram(g.patcher, g.color_v_id, &attr, 1, &stream,
                                                   1, &g.color_vp) < 0) {
            return false;
        }
        if (sceGxmShaderPatcherCreateFragmentProgram(
                g.patcher, g.color_f_id, SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                SCE_GXM_MULTISAMPLE_NONE, nullptr, color_v, &g.color_fp) < 0) {
            return false;
        }
        g.color_param = sceGxmProgramFindParameterByName(color_f, "uColor");
    }

    // Every quad is the same two triangles.
    auto* indices = static_cast<u16*>(GpuAllocUncached(6 * sizeof(u16)));
    if (indices == nullptr) {
        return false;
    }
    const u16 quad[6] = {0, 1, 2, 2, 1, 3};
    std::memcpy(indices, quad, sizeof(quad));
    g.quad_indices = indices;
    return true;
}

/// The GXM texture format for a guest framebuffer format, byte-order-matched to
/// Common::Color::Decode*. Verified against the software decode at bring-up.
bool GuestTextureFormat(Pica::PixelFormat format, SceGxmTextureFormat* out) {
    switch (format) {
    case Pica::PixelFormat::RGBA8: // bytes A,B,G,R => R in the word's top byte
        *out = SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_RGBA;
        return true;
    case Pica::PixelFormat::RGB8: // bytes B,G,R
        *out = SCE_GXM_TEXTURE_FORMAT_U8U8U8_RGB;
        return true;
    case Pica::PixelFormat::RGB565:
        *out = SCE_GXM_TEXTURE_FORMAT_U5U6U5_RGB;
        return true;
    case Pica::PixelFormat::RGB5A1:
        *out = SCE_GXM_TEXTURE_FORMAT_U5U5U5U1_RGBA;
        return true;
    case Pica::PixelFormat::RGBA4:
        *out = SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_RGBA;
        return true;
    default:
        return false;
    }
}

void DrawQuad(const QuadVertex corners[4]) {
    auto* verts = static_cast<QuadVertex*>(FrameAlloc(4 * sizeof(QuadVertex)));
    if (verts == nullptr) {
        return;
    }
    std::memcpy(verts, corners, 4 * sizeof(QuadVertex));
    sceGxmSetVertexStream(g.context, 0, verts);
    sceGxmDraw(g.context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
               g.quad_indices, 6);
}

} // Anonymous namespace

bool Initialize() {
    if (g.initialized) {
        return true;
    }

    LoadRazor();
    LoadGpuLiveModule();

    SceGxmInitializeParams init{};
    init.flags = 0;
    display_buffers = MaxDisplayBuffers;
    init.displayQueueMaxPendingCount = display_buffers;
    init.displayQueueCallback = DisplayCallback;
    init.displayQueueCallbackDataSize = sizeof(DisplayData);
    init.parameterBufferSize = ParameterBufferSize;
    if (sceGxmInitialize(&init) < 0) {
        LOG_CRITICAL(Frontend, "sceGxmInitialize failed");
        return false;
    }
    // Razor GPU Live wants libgxm up, as vitaGL orders it.
    StartGpuLive();
    CreatePools();

    // The immediate context and its ring buffers. Both hold default uniform buffers, which a
    // reservation takes from and the GPU frees only once the draw that read them is done - so
    // a ring that wraps mid-frame stalls until it is. A generated vertex program's block is
    // the whole PICA float bank, two kilobytes, against a dozen bytes for the fixed one, and
    // a fragment block is hundreds of floats; at a few hundred draws a frame the SDK defaults
    // wrap every few frames. These are a fraction of what the texture cache takes.
    constexpr u32 VertexRingSize = 16 * 1024 * 1024;
    constexpr u32 FragmentRingSize = 8 * 1024 * 1024;
    g.context_host_mem = std::malloc(SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE);
    void* vdm_ring = GpuAllocForGpu(SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE);
    void* vertex_ring = GpuAllocForGpu(VertexRingSize);
    void* fragment_ring = GpuAllocForGpu(FragmentRingSize);
    SceUID fragment_usse_uid;
    void* fragment_usse_ring = nullptr;
    {
        const u32 size = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE;
        SceUID uid = sceKernelAllocMemBlock("azahar-gxm-fusse",
                                            SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
                                            (size + 0xFFFu) & ~0xFFFu, nullptr);
        LogMemBlock("azahar-gxm-fusse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
                    (size + 0xFFFu) & ~0xFFFu, uid);
        if (uid >= 0) {
            sceKernelGetMemBlockBase(uid, &fragment_usse_ring);
            u32 offset = 0;
            if (sceGxmMapFragmentUsseMemory(fragment_usse_ring, (size + 0xFFFu) & ~0xFFFu,
                                            &offset) < 0) {
                sceKernelFreeMemBlock(uid);
                fragment_usse_ring = nullptr;
            } else {
                g.owned.push_back(
                    {uid, fragment_usse_ring, (size + 0xFFFu) & ~0xFFFu, MapKind::FragmentUsse});
                SceGxmContextParams params{};
                params.hostMem = g.context_host_mem;
                params.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
                params.vdmRingBufferMem = vdm_ring;
                params.vdmRingBufferMemSize = SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE;
                params.vertexRingBufferMem = vertex_ring;
                params.vertexRingBufferMemSize = VertexRingSize;
                params.fragmentRingBufferMem = fragment_ring;
                params.fragmentRingBufferMemSize = FragmentRingSize;
                params.fragmentUsseRingBufferMem = fragment_usse_ring;
                params.fragmentUsseRingBufferMemSize =
                    SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE;
                params.fragmentUsseRingBufferOffset = offset;
                if (sceGxmCreateContext(&params, &g.context) < 0) {
                    g.context = nullptr;
                }
                (void)fragment_usse_uid;
            }
        }
    }
    if (g.context == nullptr || vdm_ring == nullptr || vertex_ring == nullptr ||
        fragment_ring == nullptr || fragment_usse_ring == nullptr) {
        LOG_CRITICAL(Frontend, "GXM context creation failed");
        Shutdown();
        return false;
    }

    // Render target: created once - the SDK calls this costly.
    SceGxmRenderTargetParams rt{};
    rt.flags = 0;
    rt.width = DisplayWidth;
    rt.height = DisplayHeight;
    rt.scenesPerFrame = 1;
    rt.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
    rt.multisampleLocations = 0;
    rt.driverMemBlock = static_cast<SceUID>(-1); // let sceGxm allocate
    if (sceGxmCreateRenderTarget(&rt, &g.render_target) < 0) {
        LOG_CRITICAL(Frontend, "GXM render target creation failed");
        Shutdown();
        return false;
    }

    // Display buffers: what scans out must be CDRAM. Everything else in this file is LPDDR.
    for (u32 i = 0; i < display_buffers; i++) {
        SceUID uid;
        g.display_mem[i] =
            GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, DisplayWidth * DisplayHeight * 4,
                     static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                          SCE_GXM_MEMORY_ATTRIB_WRITE),
                     &uid);
        if (g.display_mem[i] == nullptr ||
            sceGxmColorSurfaceInit(&g.display_surface[i], SCE_GXM_COLOR_FORMAT_A8B8G8R8,
                                   SCE_GXM_COLOR_SURFACE_LINEAR, SCE_GXM_COLOR_SURFACE_SCALE_NONE,
                                   SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT, DisplayWidth,
                                   DisplayHeight, DisplayWidth, g.display_mem[i]) < 0 ||
            sceGxmSyncObjectCreate(&g.sync[i]) < 0) {
            LOG_CRITICAL(Frontend, "GXM display surface {} failed", i);
            Shutdown();
            return false;
        }
        std::memset(g.display_mem[i], 0, DisplayWidth * DisplayHeight * 4);
        g.arena[i] = static_cast<u8*>(GpuAllocUncached(FrameArenaSize));
        g.notification[i] = AllocNotification();
    }

    // The shader patcher, over heaps that grow: the renderer creates one fragment program per
    // PICA configuration a title uses, which no fixed region sizes up front.
    {
        SceGxmShaderPatcherParams params{};
        params.userData = nullptr;
        params.hostAllocCallback = PatcherHostAlloc;
        params.hostFreeCallback = PatcherHostFree;
        params.bufferAllocCallback = PatcherBufferAlloc;
        params.bufferFreeCallback = PatcherBufferFree;
        params.vertexUsseAllocCallback = PatcherVertexUsseAlloc;
        params.vertexUsseFreeCallback = PatcherVertexUsseFree;
        params.fragmentUsseAllocCallback = PatcherFragmentUsseAlloc;
        params.fragmentUsseFreeCallback = PatcherFragmentUsseFree;
        if (sceGxmShaderPatcherCreate(&params, &g.patcher) < 0) {
            LOG_CRITICAL(Frontend, "GXM shader patcher creation failed");
            Shutdown();
            return false;
        }
    }

    if (!BuildFixedPipelines()) {
        Shutdown();
        return false;
    }

    g.initialized = true;
    // The renderer draws through the same context and patcher.
    GxmRenderer::SetDevice(GxmRenderer::GxmDevice{
        .context = g.context,
        .patcher = g.patcher,
        .alloc_mapped = AllocMapped,
        .alloc_mapped_cached = AllocMappedCached,
        .alloc_cdram = AllocCdram,
        .largest_free = LargestFree,
        .frame_alloc = FrameAlloc,
        .free_block = FreeBlock,
        .alloc_notification = AllocNotification,
        .razor = razor_loaded,
    });
    LOG_INFO(Frontend, "GXM presentation layer up: {} display buffers, {} KiB frame arenas",
             display_buffers, FrameArenaSize / 1024);
    return true;
}

bool IsInitialized() {
    return g.initialized;
}

void Shutdown() {
    GxmRenderer::SetDevice(GxmRenderer::GxmDevice{});
    if (g.context != nullptr) {
        sceGxmDisplayQueueFinish();
        sceGxmFinish(g.context);
    }
    if (g.patcher != nullptr) {
        if (g.quad_vp != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(g.patcher, g.quad_vp);
        }
        if (g.quad_fp != nullptr) {
            sceGxmShaderPatcherReleaseFragmentProgram(g.patcher, g.quad_fp);
        }
        if (g.color_vp != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(g.patcher, g.color_vp);
        }
        if (g.color_fp != nullptr) {
            sceGxmShaderPatcherReleaseFragmentProgram(g.patcher, g.color_fp);
        }
        if (g.quad_v_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(g.patcher, g.quad_v_id);
        }
        if (g.quad_f_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(g.patcher, g.quad_f_id);
        }
        if (g.color_v_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(g.patcher, g.color_v_id);
        }
        if (g.color_f_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(g.patcher, g.color_f_id);
        }
        sceGxmShaderPatcherDestroy(g.patcher);
    }
    // The patcher freed every program's memory through the callbacks above; the chunks the
    // heaps sit on are in g.owned and are unmapped with the rest below.
    patcher_buffer_heap.Release();
    patcher_vertex_usse_heap.Release();
    patcher_fragment_usse_heap.Release();
    for (u32 i = 0; i < display_buffers; i++) {
        if (g.sync[i] != nullptr) {
            sceGxmSyncObjectDestroy(g.sync[i]);
        }
    }
    if (g.render_target != nullptr) {
        sceGxmDestroyRenderTarget(g.render_target);
    }
    if (g.context != nullptr) {
        sceGxmDestroyContext(g.context);
    }
    UnmapAllGuestMemory();
    for (const auto& block : g.owned) {
        // A mapping is undone with the unmap that matches how it was made.
        switch (block.kind) {
        case MapKind::Regular:
            sceGxmUnmapMemory(block.addr);
            break;
        case MapKind::VertexUsse:
            sceGxmUnmapVertexUsseMemory(block.addr);
            break;
        case MapKind::FragmentUsse:
            sceGxmUnmapFragmentUsseMemory(block.addr);
            break;
        }
        sceKernelFreeMemBlock(block.uid);
    }
    g.pool_cdram.Release();
    g.pool_lpddr.Release();
    std::free(g.context_host_mem);
    if (g.initialized || g.context != nullptr) {
        sceGxmTerminate();
    }
    g = State{};
}

void MapGuestMemory(void* base, u32 size) {
    for (const auto& block : g.guest) {
        if (block.addr == base) {
            return;
        }
    }
    if (sceGxmMapMemory(base, size, SCE_GXM_MEMORY_ATTRIB_READ) < 0) {
        LOG_ERROR(Frontend, "could not map {} bytes of guest memory for the GPU", size);
        return;
    }
    g.guest.push_back({0, base, size, MapKind::Regular});
    LOG_INFO(Frontend, "guest memory mapped for zero-copy presentation: {} MiB",
             size / (1024 * 1024));
}

void UnmapAllGuestMemory() {
    if (g.guest.empty()) {
        return;
    }
    if (g.context != nullptr) {
        sceGxmFinish(g.context);
    }
    for (const auto& block : g.guest) {
        sceGxmUnmapMemory(block.addr);
    }
    g.guest.clear();
}

SceGxmContext* Context() {
    return g.context;
}

SceGxmShaderPatcher* Patcher() {
    return g.patcher;
}

void* AllocMapped(u32 size) {
    return GpuAllocUncached(size);
}

void* AllocMappedCached(u32 size) {
    // SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, the ordinary cached kind. Cached memory is
    // GPU-snoop-coherent on this hardware, so no flush is needed either way; the choice is
    // whether the CPU's writes or the GPU's reads pay. For a buffer written once by the CPU
    // and read once, sequentially, by the vertex fetch, that trade may well go the other way
    // from the textures the uncached rule was written for.
    SceUID uid;
    return GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size,
                    static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                         SCE_GXM_MEMORY_ATTRIB_WRITE),
                    &uid);
}

void* AllocCdram(u32 size) {
    SceUID uid;
    return GpuAlloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, size,
                    static_cast<SceGxmMemoryAttribFlags>(SCE_GXM_MEMORY_ATTRIB_READ |
                                                         SCE_GXM_MEMORY_ATTRIB_WRITE),
                    &uid);
}

void FreeBlock(void* block) {
    for (MemPool* pool : {&g.pool_cdram, &g.pool_lpddr}) {
        if (pool->Owns(block)) {
            pool->Free(block);
            return;
        }
    }
    for (auto it = g.owned.begin(); it != g.owned.end(); ++it) {
        if (it->addr == block && it->kind == MapKind::Regular) {
            sceGxmUnmapMemory(it->addr);
            sceKernelFreeMemBlock(it->uid);
            g.owned.erase(it);
            return;
        }
    }
}

const std::string& GpuLiveText() {
    return live_text;
}

bool HasRazor() {
    return razor_loaded;
}

void CaptureNextFrame() {
    if (!razor_loaded) {
        LOG_WARNING(Frontend, "GPU capture asked for, but Razor is not loaded (gxm_flags.txt: razor)");
        return;
    }
    static char path[64];
    std::snprintf(path, sizeof(path), "ux0:data/azahar/capture_%u.sgx", razor_captures++);
    sceRazorGpuCaptureSetTriggerNextFrame(path);
    LOG_WARNING(Frontend, "Razor: capturing the next frame to {}", path);
}

void DumpNextFrame() {
    dump_requested = true;
}

bool DumpPending() {
    return dump_requested;
}

void DrawTexture(const SceGxmTexture* texture, float x, float y, float w, float h) {
    if (!g.in_scene || texture == nullptr) {
        return;
    }
    sceGxmSetVertexProgram(g.context, g.quad_vp);
    sceGxmSetFragmentProgram(g.context, g.quad_fp);
    sceGxmSetFragmentTexture(g.context, 0, texture);
    const float x0 = ClipX(x), y0 = ClipY(y), x1 = ClipX(x + w), y1 = ClipY(y + h);
    const QuadVertex corners[4] = {
        {x0, y0, 0.0f, 0.0f},
        {x1, y0, 1.0f, 0.0f},
        {x0, y1, 0.0f, 1.0f},
        {x1, y1, 1.0f, 1.0f},
    };
    DrawQuad(corners);
}

void* FrameAlloc(u32 size, u32 align) {
    u32 offset = (g.arena_used[g.back] + align - 1) & ~(align - 1);
    if (offset + size > FrameArenaSize) {
        LOG_ERROR(Frontend, "frame arena exhausted ({} + {} bytes)", offset, size);
        return nullptr;
    }
    g.arena_used[g.back] = offset + size;
    return g.arena[g.back] + offset;
}

void BeginFrame(u8 clear_r, u8 clear_g, u8 clear_b) {
    if (!g.initialized) {
        return;
    }
    g.back = g.frame_index % display_buffers;

    // The GPU must be done with this slot's arena and display buffer before either is
    // touched: the notification written at this slot's last EndScene proves it.
    if (g.notification_value[g.back] != 0) {
        SceGxmNotification wait{};
        wait.address = const_cast<u32*>(g.notification[g.back]);
        wait.value = g.notification_value[g.back];
        const PresentTimer timer{ptimes.slot_wait_us};
        ptimes.slot_waits++;
        GXM_PHASE("present:slot-wait");
        sceGxmNotificationWait(&wait);
    }
    g.arena_used[g.back] = 0;

    // The screens are sampled from surfaces the renderer's own scenes wrote, and scenes on
    // one context are not ordered against each other unless they say so. The renderer sets
    // FRAGMENT_SET_DEPENDENCY on every scene it ends, so waiting here is what pairs with it.
    {
        const PresentTimer timer{ptimes.begin_scene_us};
        GXM_PHASE("present:begin-scene");
        sceGxmBeginScene(g.context,
                         SCE_GXM_SCENE_FRAGMENT_SET_DEPENDENCY |
                             SCE_GXM_SCENE_VERTEX_WAIT_FOR_DEPENDENCY,
                         g.render_target, nullptr, nullptr, g.sync[g.back],
                         &g.display_surface[g.back], nullptr);
    }
    g.in_scene = true;

    sceGxmSetDefaultRegionClipAndViewport(g.context, DisplayWidth - 1, DisplayHeight - 1);
    sceGxmSetCullMode(g.context, SCE_GXM_CULL_NONE);
    sceGxmSetFrontDepthFunc(g.context, SCE_GXM_DEPTH_FUNC_ALWAYS);
    sceGxmSetFrontDepthWriteEnable(g.context, SCE_GXM_DEPTH_WRITE_DISABLED);

    DrawSolidRect(0.0f, 0.0f, static_cast<float>(DisplayWidth),
                  static_cast<float>(DisplayHeight), clear_r, clear_g, clear_b, 255);
}

void DrawSolidRect(float x, float y, float w, float h, u8 r, u8 gc, u8 b, u8 a) {
    if (!g.in_scene) {
        return;
    }
    sceGxmSetVertexProgram(g.context, g.color_vp);
    sceGxmSetFragmentProgram(g.context, g.color_fp);

    void* uniform_buffer = nullptr;
    sceGxmReserveFragmentDefaultUniformBuffer(g.context, &uniform_buffer);
    const float color[4] = {r / 255.0f, gc / 255.0f, b / 255.0f, a / 255.0f};
    sceGxmSetUniformDataF(uniform_buffer, g.color_param, 0, 4, color);

    auto* verts = static_cast<float*>(FrameAlloc(4 * 2 * sizeof(float)));
    if (verts == nullptr) {
        return;
    }
    const float x0 = ClipX(x), y0 = ClipY(y), x1 = ClipX(x + w), y1 = ClipY(y + h);
    const float quad[8] = {x0, y0, x1, y0, x0, y1, x1, y1};
    std::memcpy(verts, quad, sizeof(quad));
    sceGxmSetVertexStream(g.context, 0, verts);
    sceGxmDraw(g.context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16, g.quad_indices,
               6);
}

void DrawGuestScreen(const GuestScreen& screen, float x, float y, float w, float h) {
    if (!g.in_scene) {
        return;
    }
    if (screen.fill_enabled) {
        DrawSolidRect(x, y, w, h, screen.fill_r, screen.fill_g, screen.fill_b, 255);
        return;
    }
    if ((screen.pixels == nullptr && screen.texture == nullptr) || screen.height == 0) {
        return;
    }

    const auto format = static_cast<Pica::PixelFormat>(screen.pica_format);
    SceGxmTextureFormat tex_format;
    if (!GuestTextureFormat(format, &tex_format)) {
        return;
    }
    const u32 bpp = Pica::BytesPerPixel(format);
    // The guest framebuffer is 240-pixel columns, one per scanline. 240 is a multiple of 8,
    // so a packed buffer takes the plain linear init; an unusual stride takes the strided
    // variant (which cannot express 24-bit formats - those keep the last frame instead).
    // A renderer-provided texture (the cached surface) has the same layout already.
    SceGxmTexture texture;
    if (screen.texture != nullptr) {
        texture = *screen.texture;
    } else if (screen.stride_bytes == 240 * bpp) {
        if (sceGxmTextureInitLinear(&texture, screen.pixels, tex_format, 240, screen.height,
                                    1) < 0) {
            return;
        }
    } else {
        if (format == Pica::PixelFormat::RGB8 ||
            sceGxmTextureInitLinearStrided(&texture, screen.pixels, tex_format, 240,
                                           screen.height, screen.stride_bytes) < 0) {
            return;
        }
    }
    // Bring-up switch: the guest framebuffer as it stands in memory, before this quad samples
    // it. A picture that is already wrong here was drawn wrong; one that is right here and
    // wrong on screen is sampled wrong. Header: width, height, stride in bytes, bytes per
    // pixel, then the rows exactly as the guest wrote them.
    static const bool dump_frames_fb = GxmRenderer::GxmFlag("dumpframe");
    if (dump_frames_fb) {
        const bool top = screen.height >= 400;
        static u64 last_fb_us[2]{};
        u64& last = last_fb_us[top ? 0 : 1];
        const u64 now_us = sceKernelGetProcessTimeWide();
        if (now_us - last >= 5000000ull) {
            last = now_us;
            // Whatever the quad is about to sample, which is the renderer's own surface when
            // it has one and the guest's framebuffer otherwise.
            const void* pixels = sceGxmTextureGetData(&texture);
            const u32 tex_width = sceGxmTextureGetWidth(&texture);
            const u32 tex_height = sceGxmTextureGetHeight(&texture);
            const u32 tex_stride = screen.texture != nullptr
                                       ? ((tex_width + 7u) & ~7u) * bpp
                                       : screen.stride_bytes;
            char path[64];
            std::snprintf(path, sizeof(path), "ux0:data/azahar/fb_%s.raw", top ? "top" : "bot");
            if (FILE* f = std::fopen(path, "wb")) {
                const u32 header[4] = {tex_width, tex_height, tex_stride, bpp};
                std::fwrite(header, sizeof(header), 1, f);
                std::fwrite(pixels, tex_stride, tex_height, f);
                std::fclose(f);
            }
            // And the guest's own memory for the same screen, whatever the quad samples.
            if (screen.pixels != nullptr) {
                std::snprintf(path, sizeof(path), "ux0:data/azahar/fb_%s_mem.raw",
                              top ? "top" : "bot");
                if (FILE* f = std::fopen(path, "wb")) {
                    const u32 header[4] = {240, screen.height, screen.stride_bytes, bpp};
                    std::fwrite(header, sizeof(header), 1, f);
                    std::fwrite(screen.pixels, screen.stride_bytes, screen.height, f);
                    std::fclose(f);
                }
            }
        }
    }

    // Bring-up switch: what the screen quad actually samples. A texture whose stride is not
    // its width, or whose size is not the framebuffer's, shears the picture across the quad,
    // which is what a wrong one looks like on screen.
    static const bool log_quad = GxmRenderer::GxmFlag("logdraw");
    if (log_quad) {
        static u32 logged = 0;
        if (logged++ % 600 == 0) {
            LOG_INFO(Render,
                     "screen quad: {} texture {}x{}, type {}, stride {}, format {:#x}; guest "
                     "stride {} bytes, height {}, bpp {}",
                     screen.texture != nullptr ? "renderer" : "guest memory",
                     sceGxmTextureGetWidth(&texture), sceGxmTextureGetHeight(&texture),
                     static_cast<u32>(sceGxmTextureGetType(&texture)),
                     sceGxmTextureGetStride(&texture),
                     static_cast<u32>(sceGxmTextureGetFormat(&texture)), screen.stride_bytes,
                     screen.height, bpp);
        }
    }

    sceGxmTextureSetMagFilter(&texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    // A strided texture has no minification filter to set - GXM refuses the call - so a
    // framebuffer whose stride is not the implicit pitch is point-sampled when it shrinks.
    if (sceGxmTextureGetType(&texture) != SCE_GXM_TEXTURE_LINEAR_STRIDED) {
        sceGxmTextureSetMinFilter(&texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    }
    sceGxmTextureSetUAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);
    sceGxmTextureSetVAddrMode(&texture, SCE_GXM_TEXTURE_ADDR_CLAMP);

    sceGxmSetVertexProgram(g.context, g.quad_vp);
    sceGxmSetFragmentProgram(g.context, g.quad_fp);
    sceGxmSetFragmentTexture(g.context, 0, &texture);

    // The rotation lives here, in the texture coordinates: the texture is 240 wide (u runs
    // along a scanline column, bottom of the upright picture at u=0) by height rows (v runs
    // left to right across the upright picture). So the screen-space vertical axis maps to
    // u, reversed, and the horizontal axis maps to v.
    // A renderer texture may be the larger surface the framebuffer sits in (NSMB2 renders
    // 256x416 and shows 240x400 of it, 16 rows in): only the framebuffer's part is sampled.
    float u0 = 0.0f, u1 = 1.0f, v0 = 0.0f, v1 = 1.0f;
    if (screen.texture != nullptr && screen.tex_width != 0 && screen.tex_height != 0) {
        const float tex_width = static_cast<float>(sceGxmTextureGetWidth(&texture));
        const float tex_height = static_cast<float>(sceGxmTextureGetHeight(&texture));
        u0 = static_cast<float>(screen.tex_x) / tex_width;
        u1 = static_cast<float>(screen.tex_x + screen.tex_width) / tex_width;
        v0 = static_cast<float>(screen.tex_y) / tex_height;
        v1 = static_cast<float>(screen.tex_y + screen.tex_height) / tex_height;
    }
    const float x0 = ClipX(x), y0 = ClipY(y), x1 = ClipX(x + w), y1 = ClipY(y + h);
    const QuadVertex corners[4] = {
        {x0, y0, u1, v0}, // top-left shows the top of the picture: u at its far end
        {x1, y0, u1, v1},
        {x0, y1, u0, v0},
        {x1, y1, u0, v1},
    };
    DrawQuad(corners);
}

void EndFrame() {
    if (!g.initialized || !g.in_scene) {
        return;
    }
    g.notification_counter++;
    SceGxmNotification done{};
    done.address = const_cast<u32*>(g.notification[g.back]);
    done.value = g.notification_counter;
    g.notification_value[g.back] = g.notification_counter;
    {
        const PresentTimer timer{ptimes.end_scene_us};
        GXM_PHASE("present:end-scene");
        sceGxmEndScene(g.context, nullptr, &done);
    }
    g.in_scene = false;

    // Bring-up switch: every five seconds, the frame just drawn as a PPM under
    // ux0:data/azahar, so what the GPU produced can be looked at off the console. Costs a
    // full GPU wait and a 1.5 MB write each time, which is why it is a switch.
    static const bool dump_frames = GxmRenderer::GxmFlag("dumpframe");
    static u64 last_dump_us = 0;
    static u32 dump_index = 0;
    const u64 now_us = sceKernelGetProcessTimeWide();
    if (dump_requested || (dump_frames && now_us - last_dump_us >= 5000000ull)) {
        dump_requested = false;
        last_dump_us = now_us;
        char path[64];
        std::snprintf(path, sizeof(path), "ux0:data/azahar/frame_%02u.ppm", dump_index++ % 20);
        WriteDisplayPpm(path);
        LOG_INFO(Frontend, "wrote {}", path);
    }
    // The surface dump the button (or the dumpsurfaces flag) asked for: here, after the
    // frame is done, whether or not the emulation is running.
    if (GxmRenderer::SurfaceDumpPending()) {
        GXM_PHASE("present:dump-finish");
        sceGxmFinish(g.context);
        GxmRenderer::RunSurfaceDumpIfRequested();
    }

    const u32 front = (g.frame_index + display_buffers - 1) % display_buffers;
    DisplayData data{g.display_mem[g.back]};
    // Razor's frame marker, as vitaGL sends it before every flip: what the capture library
    // and the HUD use to know a frame's end and which surface the HUD is drawn over.
    if (razor_loaded || live.on || hud_module >= 0) {
        sceGxmPadHeartbeat(&g.display_surface[g.back], g.sync[g.back]);
    }
    {
        const PresentTimer timer{ptimes.queue_us};
        GXM_PHASE("present:queue-add");
        sceGxmDisplayQueueAddEntry(g.sync[front], g.sync[g.back], &data);
    }
    g.frame_index++;
    ptimes.frames++;
#ifdef VITA_DIAGNOSTICS
    if (const u64 now = sceKernelGetProcessTimeWide(); now >= ptimes.report_at_us) {
        if (ptimes.report_at_us != 0 && ptimes.frames != 0) {
            const u32 n = ptimes.frames;
            sceClibPrintf("[present] %u frames: slot wait %llu us over %u waits, begin scene "
                          "%llu us, end scene %llu us, queue %llu us (all per frame); the "
                          "display thread's flip and vblank %llu us over %u callbacks\n",
                          n, ptimes.slot_wait_us / n, ptimes.slot_waits, ptimes.begin_scene_us / n,
                          ptimes.end_scene_us / n, ptimes.queue_us / n,
                          ptimes.callbacks ? ptimes.callback_us / ptimes.callbacks : 0,
                          ptimes.callbacks);
        }
        ptimes = PresentTimes{};
        ptimes.report_at_us = now + 1000000;
    }
#endif
    if (live.on) {
        GpuLiveFrame();
    }
    Common::PipelineStats::presents.fetch_add(1, std::memory_order_relaxed);
    GXM_PHASE("present:done");
}

} // namespace VitaFrontend::GxmPresent
