// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <psp2/gxm.h>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"

namespace VideoCore {
struct Material;
class RendererBase;
} // namespace VideoCore

namespace GxmRenderer {

/**
 * The GXM instantiation of the shared rasterizer cache: what the cache asks
 * of a backend, on libgxm.
 *
 * Every surface is one linear allocation the GPU can sample and, for the colour and depth
 * types, render into: a SceGxmTexture over the same memory as the SceGxmColorSurface or
 * SceGxmDepthStencilSurface, rows stored bottom-up in the GL convention the codec and the
 * cache's rectangles use (row index = GL y). Memory comes from two sub-allocated pools:
 * CDRAM for the big surfaces the GPU alone touches (render targets), uncached LPDDR for
 * everything the CPU uploads (GXM_FACTS: GPU reads of cached memory are slow, CPU reads of
 * CDRAM slower still).
 *
 * Copies, blits and fills go through the PTLA transfer engine where its rules allow
 * (linear, matching texel size; a vertical flip is a negative stride; the fixed 50% box
 * downscale), and through a CPU path otherwise. Both are synchronous at this phase: the
 * rasterizer's sync hook ends the open scene and waits for the GPU before any transfer or
 * upload touches a surface it may be drawing or sampling, and every transfer is waited for
 * before the next command. P6 replaces the waits with notifications.
 */

struct GxmFormat {
    SceGxmTextureFormat texture;
    /// The colour and depth-stencil forms, meaningful for a colour and a depth surface type
    /// respectively (and left zero for the rest). Neither is a sentinel: zero is a real
    /// colour format - SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR is exactly zero.
    SceGxmColorFormat color;
    SceGxmDepthStencilFormat depth;
    SceGxmTransferFormat transfer;  ///< the PTLA's view: a real format or RAW by size
    u32 bytes_per_pixel;            ///< in surface memory
    bool converted;                 ///< the codec converts to RGBA8 (or F32) for it
    bool native_transfer;           ///< `transfer` is a real colour format (converts)
};

/// The GXM view of a cache pixel format.
[[nodiscard]] const GxmFormat& FormatOf(VideoCore::PixelFormat format);

/// Sub-allocator over big GPU-mapped blocks: first fit, coalescing, per pool kind.
class GpuPool {
public:
    enum class Kind { Mapped, Cdram };
    GpuPool(Kind kind, u32 chunk_size, u32 budget);
    ~GpuPool();

    [[nodiscard]] void* Alloc(u32 size, u32 align = 64);
    void Free(void* ptr);

    [[nodiscard]] u32 Used() const noexcept {
        return used;
    }
    [[nodiscard]] u32 Reserved() const noexcept {
        return reserved;
    }
    /// The most this pool may ever reserve.
    [[nodiscard]] u32 Budget() const noexcept {
        return budget;
    }

private:
    struct Chunk {
        u8* base{};
        u32 size{};
        std::map<u32, u32> free; ///< offset -> size
    };
    bool AddChunk(u32 min_size);

    Kind kind;
    u32 chunk_size;
    u32 budget;
    u32 used{};
    u32 reserved{};
    std::vector<Chunk> chunks;
    std::map<u8*, std::pair<std::size_t, u32>> live; ///< ptr -> (chunk index, size)
    u64 exhausted_reports = 0;
};

class Surface;
class Framebuffer;

class TextureRuntime {
    friend class Surface;
    friend class Framebuffer;

public:
    explicit TextureRuntime(VideoCore::RendererBase& renderer);
    ~TextureRuntime();

    /// Called before any GPU-side transfer or CPU write that may race the open scene: the
    /// rasterizer ends the scene and waits for the GPU. Set by the rasterizer.
    /// Called with a colour target that has clears pending; returns true when the caller
    /// drew them into the scene open on that surface, so nothing stays pending.
    void SetColourClearHook(std::function<bool(Surface&)> hook) {
        colour_clear_hook = std::move(hook);
    }
    /// The hook ends the open scene and waits for the GPU: for the serial it is given, or
    /// for everything when that is 0 (a surface the presentation layer's own scene reads
    /// carries no serial of ours).
    void SetSyncHook(std::function<void(u64)> hook) {
        sync_hook = std::move(hook);
    }

    /**
     * Waits for the GPU only if it may still be touching these surfaces.
     *
     * Every scene carries a serial, and every surface it draws into or samples records that
     * serial; a fragment notification writes the serial into the notification region once the
     * GPU has finished the scene, so how far the GPU has got is a plain load. A surface whose
     * serial the GPU has already passed needs no wait at all, which is the common case: a
     * texture being uploaded on a cache miss has never been in a scene, and one uploaded again
     * has usually not been in the last few either.
     *
     * A null argument means "wait unconditionally". Passing both surfaces of a copy is the
     * same as passing each in turn.
     */
    void Sync(Surface* a = nullptr, Surface* b = nullptr, const char* why = "?");
    /// Writes a deferred whole-surface depth clear into memory, the way a partial clear
    /// is done, when the memory is about to be read or written outside a scene.
    void FlushPendingClear(Surface& surface);

    /**
     * Converts one surface into another with a draw, or declines.
     *
     * The PTLA cannot convert between the guest's pixel formats - its only 16-bit formats are
     * BGR-ordered and the guest's are RGB-ordered - so a blit that changes format had no
     * hardware path and fell to a CPU pass over every texel. The fragment pipeline has one:
     * a textured quad into the destination's colour surface converts on output, in the pixel
     * back end, for nothing. Set by the rasterizer, which owns the programs and the context.
     */
    void SetBlitHook(std::function<bool(Surface&, Surface&, const VideoCore::TextureBlit&)> hook) {
        blit_hook = std::move(hook);
    }

    /// Claims the serial for a scene about to begin and fills in the fragment notification
    /// that reports its completion. The rasterizer calls this from sceGxmBeginScene and hands
    /// the notification to sceGxmEndScene.
    u64 BeginSceneEpoch(SceGxmNotification* notification);
    /// The highest scene serial the GPU has finished, read from the notification region.
    [[nodiscard]] u64 CompletedEpoch() const;
    /// Records that the scene just claimed has been handed to sceGxmEndScene, so a wait for
    /// it can finish. Only a submitted scene is ever waited for: see WaitForEpoch.
    void SceneSubmitted() {
        submitted_serial = scene_serial;
    }
    /// Blocks until the GPU has passed scene `serial`, clamped to the newest *submitted* one.
    /// Cheaper than a finish when only an old scene is waited for: nothing after it is
    /// drained.
    void WaitForEpoch(u64 serial);
    /// A surface's memory goes back to its pool only once the GPU has passed the last scene
    /// that used it: frames are no longer finished on the CPU, so a freed surface may still
    /// be in flight. ReapFreed returns what is done (all of it, after waiting, when `wait`).
    void DeferFree(void* data, bool cdram, u64 serial);
    void ReapFreed(bool wait);
    /// Whether a scene the GPU has not passed may still read or write the surface.
    [[nodiscard]] bool InFlight(const Surface& surface) const {
        return NeedsWait(&surface);
    }
    u32 stat_renames{}; ///< uploads that took a fresh block instead of waiting for the GPU
    /// The serial of the scene being recorded, for marking the surfaces it uses.
    [[nodiscard]] u64 CurrentEpoch() const noexcept {
        return scene_serial;
    }
    /// Forgets that anything is outstanding: called after a full sceGxmFinish. The floor goes
    /// to the newest *submitted* scene, not the newest claimed one - a claimed scene whose
    /// sceGxmBeginScene failed is never written by anyone, and a floor above what the
    /// notification word will ever hold makes CompletedEpoch's 32-bit reconstruction decide
    /// the counter has wrapped, after which it returns a value four billion too high forever.
    void EpochDrained() {
        completed_floor = submitted_serial;
    }

    /// Opaque tick for the cache's garbage collector: a surface sentenced at tick t is freed
    /// once the tick moves past it. The rasterizer advances it after each frame's wait.
    u64 GetResourceTick();
    void Finish();

    /// True when the codec must convert this surface's format on the way in and out.
    bool NeedsConversion(const Surface& surface) const;

    /**
     * Whether a readback of this surface may be answered from what the last one left in guest
     * memory instead of by draining the GPU for fresh pixels. Only while the GPU has not
     * passed the surface, which is when the drain would cost something.
     *
     * Off unless `staleread` asks for it, because the titles measured so far read back the
     * very buffers they display: Mario Kart 7 reads 240x400 of each of four 256x416 targets,
     * 0x2000 in, 140 times a second (2026-09-09), which is its two screens double-buffered.
     * Stale pixels there are a stale picture, and at two and a half reads a frame the lag
     * varies from frame to frame, which reads as judder. A title that reads a target back for
     * something the screen does not show is the case this was written for, and none has been
     * measured yet.
     */
    [[nodiscard]] bool CanSkipDownload(const Surface& surface) const {
        static const bool on = GxmFlag("staleread");
        return on && NeedsWait(&surface);
    }

    /// A CPU staging buffer for pixel uploads/downloads.
    VideoCore::StagingData FindStaging(u32 size, bool upload);

    bool Reinterpret(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);
    void ClearTexture(Surface& surface, const VideoCore::TextureClear& clear);
    bool CopyTextures(Surface& source, Surface& dest,
                      std::span<const VideoCore::TextureCopy> copies);
    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy) {
        return CopyTextures(source, dest, std::array{copy});
    }
    bool BlitTextures(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);
    void GenerateMipmaps(Surface& surface);

    [[nodiscard]] GpuPool& MappedPool() {
        return mapped_pool;
    }
    /**
     * How much video memory the surface cache could still get: what its CDRAM pool has spare
     * plus the largest block the device would still hand it. This is the number that matters,
     * not the pool's nominal budget - the pool grows in chunks until the device refuses, and
     * on the console the device refuses long before the budget is reached.
     */
    [[nodiscard]] u32 CdramHeadroom() const;

    [[nodiscard]] GpuPool& CdramPool() {
        return cdram_pool;
    }

    /// Counters, reset by the rasterizer's frame report.
    u32 stat_uploads{}, stat_upload_bytes{}, stat_downloads{}, stat_download_bytes{};
    u32 stat_transfers{}, stat_cpu_blits{}, stat_gpu_blits{}, stat_fills{};
    /// Of the GPU blits, the ones that went into a scene already open on their destination
    /// instead of opening one of their own. A scene is the expensive unit on this GPU.
    u32 stat_gpu_blit_merges{};
    u32 stat_deferred_clears{}; ///< whole-surface depth clears handed to the next scene
    u32 stat_syncs{}, stat_syncs_avoided{};

private:
    /// Waits for the PTLA; every transfer is followed by one at this phase.
    void TransferWait();
    void ClearNow(Surface& surface, const VideoCore::TextureClear& clear);
    bool CpuBlit(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);

    [[nodiscard]] bool NeedsWait(const Surface* surface) const;

    std::function<void(u64)> sync_hook;
    std::function<bool(Surface&)> colour_clear_hook;
    std::function<bool(Surface&, Surface&, const VideoCore::TextureBlit&)> blit_hook;
    u64 current_resource_tick{};
    /// Scene serials. `scene_serial` is the last one claimed; `completed_floor` is what a
    /// full wait has already proved finished, which is what makes the notification's 32-bit
    /// value enough to carry a 64-bit serial.
    u64 scene_serial{};
    /// The newest serial actually handed to sceGxmEndScene. Never below what the GPU can
    /// write, and never above it either, which is the point.
    u64 submitted_serial{};
    u64 completed_floor{};
    struct PendingFree {
        void* data;
        bool cdram;
        u64 serial;
    };
    std::vector<PendingFree> pending_frees;
    /// One notification word per in-flight scene, scene s writing epoch_words[s % EpochSlots].
    /// sceGxmNotificationWait returns only when the word EQUALS the value, and with a single
    /// word for every scene the GPU overtook the waited-for serial while the thread was
    /// asleep and the wait never returned (the whole-console freeze, 2026-09-08: waited for
    /// 25546, GPU already at 25550). A slot is only ever written by s and s + EpochSlots, and
    /// s + EpochSlots is claimed by the waiting thread itself, so it cannot be submitted
    /// before the wait for s begins.
    static constexpr u32 EpochSlots = 32;
    std::array<volatile u32*, EpochSlots> epoch_words{};
    bool epoch_words_ready = false;
    [[nodiscard]] volatile u32* EpochWord(u64 serial) const {
        return epoch_words[serial % EpochSlots];
    }
    std::vector<u8> staging_buffer;
    /**
     * Where a download lands before the CPU reads it: the GPU's transfer unit copies the
     * surface here, and only then does the CPU touch it. On the console this makes the
     * readback a DMA instead of the CPU walking GPU-written memory; under Vita3K it is what
     * makes the readback see the picture at all, since its Vulkan backend writes a rendered
     * surface back to guest memory when it becomes the source of a transfer and not when
     * the CPU reads it.
     */
    u8* download_mem{};
    static constexpr u32 DownloadMemSize = 1024 * 1024;
    GpuPool mapped_pool;
    GpuPool cdram_pool;
};

class Surface : public VideoCore::SurfaceBase {
public:
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params,
                     const VideoCore::SurfaceFlagBits& initial_flag_bits = {});
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceBase& surface,
                     const VideoCore::Material* material);
    ~Surface();

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&& o) noexcept;
    Surface& operator=(Surface&& o) noexcept;

    /// The sampling view; copy it, apply the sampler, bind the copy.
    [[nodiscard]] const SceGxmTexture* Texture() const noexcept {
        return &texture;
    }
    [[nodiscard]] bool HasMemory() const noexcept {
        return data != nullptr;
    }
    [[nodiscard]] u8* Data() const noexcept {
        return data;
    }
    /// Bytes reserved for this surface, for deciding whether two of them overlap.
    [[nodiscard]] u32 AllocSize() const noexcept {
        return alloc_size;
    }
    /// Row pitch in texels of level 0 (rows are GL-ordered: row index = y from the bottom).
    [[nodiscard]] u32 StrideTexels() const noexcept {
        return stride_texels;
    }
    [[nodiscard]] u32 StrideBytes() const noexcept {
        return stride_texels * format.bytes_per_pixel;
    }
    [[nodiscard]] const GxmFormat& Format() const noexcept {
        return format;
    }
    /// Byte offset of a mip level's first row.
    [[nodiscard]] u32 LevelOffset(u32 level) const noexcept {
        return level_offsets[std::min<u32>(level, VideoCore::MAX_PICA_LEVELS - 1)];
    }
    [[nodiscard]] u32 LevelStrideTexels(u32 level) const noexcept;
    [[nodiscard]] u8* RowPointer(u32 level, u32 y, u32 x = 0) const noexcept;

    /// The serial of the last scene that drew into or sampled this surface, and whether the
    /// presentation layer was handed it to sample from a scene this renderer did not record.
    u64 gpu_serial{};
    bool presented{};
    /// A whole-surface depth clear waiting for the next scene on this surface, which
    /// starts from the value instead of loading memory (BeginScene applies it). Anything
    /// that reads or writes the memory before then lands it first (FlushPendingClear).
    bool pending_clear{};
    float pending_depth{};
    u8 pending_stencil{};
    /// Colour clears waiting to be drawn as quads into the next scene on this surface (or
    /// the open one), in order; landed in memory by FlushPendingClear otherwise.
    struct PendingColourClear {
        Common::Rectangle<u32> rect;
        Common::Vec4f color;
    };
    std::vector<PendingColourClear> pending_colour;
    [[nodiscard]] bool HasPendingClear() const noexcept {
        return pending_clear || !pending_colour.empty();
    }

    /// Render-target views, valid when the surface is a colour or depth type with memory.
    [[nodiscard]] SceGxmColorSurface* ColorSurface() noexcept {
        return &color_surface;
    }
    [[nodiscard]] SceGxmDepthStencilSurface* DepthStencilSurface() noexcept {
        return &ds_surface;
    }
    [[nodiscard]] bool IsColorTarget() const noexcept {
        return is_color_target;
    }
    [[nodiscard]] bool IsDepthTarget() const noexcept {
        return is_depth_target;
    }

    void Upload(const VideoCore::BufferTextureCopy& upload, const VideoCore::StagingData& staging);

    /**
     * Where the decoded rows of `upload` can be written directly, when the destination is one
     * contiguous run (full-width rows of a level, a format stored as decoded): the same
     * sync-or-rename as Upload, then the texture memory itself, `bytes` long. Empty when a
     * staging pass through Upload is needed. Skips the staging copy, which was every texture
     * byte written twice, the second time into uncached video memory.
     */
    std::span<u8> DirectUploadSpan(const VideoCore::BufferTextureCopy& upload, u32 bytes);
    /// Moves the surface to a fresh block of the same pool, the old one going back once the
    /// GPU is done with it: an upload of the whole surface then need not wait. Textures
    /// only (a target's descriptors live in framebuffers too). False when the pool is out.
    bool Rename();
    void UploadCustom(const VideoCore::Material* material, u32 level);
    void Download(const VideoCore::BufferTextureCopy& download,
                  const VideoCore::StagingData& staging);
    void ScaleUp(u32 new_scale);
    u32 GetInternalBytesPerPixel() const;

private:
    void Allocate();
    void Release();

    TextureRuntime* runtime{};
    GxmFormat format{};
    u8* data{};
    bool in_cdram{};
    u32 stride_texels{};
    u32 alloc_size{};
    std::array<u32, VideoCore::MAX_PICA_LEVELS> level_offsets{};
    SceGxmTexture texture{};
    SceGxmColorSurface color_surface{};
    SceGxmDepthStencilSurface ds_surface{};
    bool is_color_target{};
    bool is_depth_target{};
};

class Framebuffer : public VideoCore::FramebufferParams {
public:
    explicit Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                         Surface* color, Surface* depth_stencil);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& o) noexcept;
    Framebuffer& operator=(Framebuffer&& o) noexcept;

    [[nodiscard]] u32 Scale() const noexcept {
        return 1;
    }
    [[nodiscard]] SceGxmRenderTarget* RenderTarget() const noexcept {
        return render_target;
    }
    /// The colour surface to begin the scene with: the attachment's, or a disabled one.
    [[nodiscard]] const SceGxmColorSurface* ColorSurface() const noexcept {
        return &color_surface;
    }
    [[nodiscard]] const SceGxmDepthStencilSurface* DepthStencilSurface() const noexcept {
        return has_depth ? &ds_surface : nullptr;
    }
    /// The same, for the per-scene load mode and background value.
    [[nodiscard]] SceGxmDepthStencilSurface* MutableDepthStencilSurface() noexcept {
        return has_depth ? &ds_surface : nullptr;
    }
    [[nodiscard]] bool HasColor() const noexcept {
        return has_color;
    }
    [[nodiscard]] bool HasDepth() const noexcept {
        return has_depth;
    }
    [[nodiscard]] u32 Width() const noexcept {
        return width;
    }
    [[nodiscard]] u32 Height() const noexcept {
        return height;
    }
    /// The colour surface's memory, to recognise a texture that samples the target it draws.
    [[nodiscard]] const u8* ColorData() const noexcept {
        return color_data;
    }

private:
    SceGxmRenderTarget* render_target{};
    SceGxmColorSurface color_surface{};
    SceGxmDepthStencilSurface ds_surface{};
    const u8* color_data{};
    u32 width{}, height{};
    bool has_color{}, has_depth{};
};

/// GXM has no sampler objects: the filters and address modes live in the texture control
/// word, so a Sampler is the set of values applied to a copy of the surface's texture at bind.
class Sampler {
public:
    explicit Sampler(TextureRuntime&, VideoCore::SamplerParams params);
    ~Sampler() = default;

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    void Apply(SceGxmTexture& texture) const;

private:
    SceGxmTextureFilter mag_filter{};
    SceGxmTextureFilter min_filter{};
    SceGxmTextureAddrMode wrap_s{};
    SceGxmTextureAddrMode wrap_t{};
};

class DebugScope {
public:
    template <typename... T>
    explicit DebugScope(TextureRuntime&, Common::Vec4f, fmt::format_string<T...>, T...) {}
    explicit DebugScope(TextureRuntime&, Common::Vec4f, std::string_view) {}
    ~DebugScope() = default;
};

struct Traits {
    using Runtime = GxmRenderer::TextureRuntime;
    using Sampler = GxmRenderer::Sampler;
    using Surface = GxmRenderer::Surface;
    using Framebuffer = GxmRenderer::Framebuffer;
    using DebugScope = GxmRenderer::DebugScope;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace GxmRenderer
