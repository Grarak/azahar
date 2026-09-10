// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <list>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <array>
#include <boost/icl/interval_map.hpp>
#include <tsl/robin_map.h>

#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/sampler_params.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/texture_cube.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct RegsInternal;
struct DisplayTransferConfig;
struct MemoryFillConfig;
} // namespace Pica

namespace Pica::Texture {
struct TextureInfo;
}

namespace Settings {
enum class TextureFilter : u32;
}

namespace VideoCore {

enum class ScaleMatch {
    Exact,   ///< Only accept same res scale
    Upscale, ///< Only allow higher scale than params
    Ignore   ///< Accept every scaled res
};

enum class MatchFlags {
    Exact = 1 << 0,       ///< Surface perfectly matches params
    SubRect = 1 << 1,     ///< Surface encompasses params
    Copy = 1 << 2,        ///< Surface that can be used as a copy source
    TexCopy = 1 << 3,     ///< Surface that will match a display transfer "texture copy" parameters
    Reinterpret = 1 << 4, ///< Surface might have different pixel format.
};

DECLARE_ENUM_FLAG_OPERATORS(MatchFlags);

class CustomTexManager;
class RendererBase;
enum class SurfaceFlagBits : u32;

template <class T>
class RasterizerCache {
    /// Address shift for caching surfaces into a hash table
    static constexpr u64 CITRA_PAGEBITS = 18;

    using Runtime = typename T::Runtime;
    using Sampler = typename T::Sampler;
    using Surface = typename T::Surface;
    using Framebuffer = typename T::Framebuffer;
    using DebugScope = typename T::DebugScope;

    using SurfaceMap = boost::icl::interval_map<PAddr, SurfaceId, boost::icl::partial_absorber,
                                                std::less, boost::icl::inplace_plus,
                                                boost::icl::inter_section, SurfaceInterval>;

    using SurfaceRect_Tuple = std::pair<SurfaceId, Common::Rectangle<u32>>;
    using PageMap = boost::icl::interval_map<u32, int>;

public:
    explicit RasterizerCache(Memory::MemorySystem& memory, CustomTexManager& custom_tex_manager,
                             Runtime& runtime, Pica::RegsInternal& regs, RendererBase& renderer);
    ~RasterizerCache();

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Perform hardware accelerated texture copy according to the provided configuration
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config);

    /// Perform hardware accelerated display transfer according to the provided configuration
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config);

    /// Perform hardware accelerated memory fill according to the provided configuration
    bool AccelerateFill(const Pica::MemoryFillConfig& config);

    /// Returns a reference to the surface object assigned to surface_id
    Surface& GetSurface(SurfaceId surface_id);

    /// Calls func(surface_id, surface) for every live surface, for a debug dump.
    template <typename Func>
    void ForEachSurface(Func&& func) {
        slot_surfaces.ForEach(std::forward<Func>(func));
    }

    /// Every cached surface to <dir> as an image plus an index, for looking at what the
    /// renderer holds. Render thread only: it reads back through the runtime.
    void DumpSurfaces(const std::string& dir, u32 frame);

    /// The surface whose validation is forcing a flush, for the readback log. Render thread.
    const Surface* validating_for{};

    /// Surfaces waiting for the garbage collector: unregistered, still holding their memory.
    [[nodiscard]] std::size_t SentencedCount() const {
        return sentenced.size();
    }

    /**
     * Gives up the surfaces the runtime nominates: each is unregistered and sentenced, and the
     * garbage collector frees it once the runtime's tick has moved past.
     *
     * Nothing else evicts. A surface leaves the cache when the guest writes over its memory,
     * and this port never sees the guest's writes - the render thread cannot flip page
     * attributes under the emulation thread, and the native CPU backend traps no stores at
     * all - so the cache keeps every surface a title ever makes. A desktop driver absorbs
     * that; a fixed pool of video memory does not.
     *
     * The runtime chooses, because only it knows what a surface costs and whether the GPU
     * still holds it. Call with no scene open and nothing bound.
     */
    void SentenceSurfaces(std::span<const SurfaceId> ids);

    /**
     * Gives up least recently used surfaces until about `want_bytes` of them are sentenced.
     * `cost(surface)` says what a surface holds, in the runtime's own bytes; `skip(surface)`
     * names the ones the runtime still needs (targets in flight, what presentation samples).
     * A surface that owns a dirty region is never a candidate: its pixels exist nowhere
     * else. Returns how many were sentenced. Call with no scene open and nothing bound.
     */
    template <typename CostFn, typename SkipFn>
    u32 TrimSurfaces(u64 want_bytes, CostFn&& cost, SkipFn&& skip);

    /// Whether `surface_id` owns any dirty region, i.e. holds rendered pixels the guest's
    /// memory has not received.
    [[nodiscard]] bool OwnsDirtyRegion(SurfaceId surface_id) const;

    /// Uploads a guest cache flush would have caused that the hash of guest memory proved
    /// unnecessary. For the backend's stats line, which resets it.
    u32 guest_flush_hash_skips = 0;
    /// What the current validation serves (texture, framebuffer): CopySurface logging.
    const char* validate_reason = "?";

    /// Trace-probe builds: log every surface given up and every reload of one, for a backend
    /// with no environment to read AZAHAR_TRIM_LOG from.
    void SetProbeTrimLog([[maybe_unused]] bool enabled) {
#ifdef CITRA_TRACE_PROBES
        probe_trim_log = enabled;
#endif
    }

    /// Returns a reference to the sampler object matching the provided configuration
    Sampler& GetSampler(const Pica::TexturingRegs::TextureConfig& config);
    Sampler& GetSampler(SamplerId sampler_id);

    /// Copy one surface's region to another
    void CopySurface(Surface& src_surface, Surface& dst_surface, SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    SurfaceId GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                         bool load_if_create, const SurfaceFlagBits& create_initial_flags = {});

    /// Attempt to find a subrect (resolution scaled) of a surface, otherwise loads a texture from
    /// 3DS memory to OpenGL and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create,
                                        const SurfaceFlagBits& create_initial_flags = {});

    /// Get a surface based on the texture configuration
    Surface& GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    SurfaceId GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get a texture cube based on the texture configuration
    Surface& GetTextureCube(const TextureCubeConfig& config);

    /// Get the color and depth surfaces based on the framebuffer configuration
    FramebufferHelper<T> GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty)
    void FlushRegion(PAddr addr, u32 size, SurfaceId flush_surface = {});

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner = {});

    /// Invalidates the parts of [addr, addr+size) that no surface holds unflushed rendered
    /// pixels for, byte-exactly. What a guest cache flush of CPU-written data means to a
    /// renderer that never writes its output back to guest memory.
    void InvalidateRegionUnlessDirty(PAddr addr, u32 size);

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

    /// Clear all cached resources tracked by this cache manager
    void ClearAll(bool flush);

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    void ForEachPage(PAddr addr, std::size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> CITRA_PAGEBITS;
        for (u64 page = addr >> CITRA_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Iterates over all the surfaces in a region calling func
    template <typename Func>
    void ForEachSurfaceInRegion(PAddr addr, std::size_t size, Func&& func);

    /// Get the best surface match (and its match type) for the given flags
    template <MatchFlags find_flags>
    SurfaceId FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                        std::optional<SurfaceInterval> validate_interval = std::nullopt);

    /// Unregisters sentenced surfaces that have surpassed the destruction threshold.
    void RunGarbageCollector();

    /// Removes any framebuffers that reference the provided surface_id.
    void RemoveFramebuffers(SurfaceId surface_id);

    /// Removes any references of the provided surface id from cached texture cubes.
    void RemoveTextureCubeFace(SurfaceId surface_id);

    /// Computes the hash of the provided texture data.
    u64 ComputeHash(const SurfaceParams& load_info, std::span<u8> upload_data);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(SurfaceId surface, PAddr addr, u32 size);

    /// Copies pixel data in interval from the guest VRAM to the host GPU surface
    void UploadSurface(Surface& surface, SurfaceInterval interval);

    /// Uploads a custom texture identified with hash to the target surface
    bool UploadCustomSurface(SurfaceId surface_id, SurfaceInterval interval);

    /// Copies pixel data in interval from the host GPU surface to the guest VRAM
    void DownloadSurface(Surface& surface, SurfaceInterval interval);

    /// Downloads a fill surface to guest VRAM
    void DownloadFillSurface(Surface& surface, SurfaceInterval interval);

    /// Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(Surface& surface, SurfaceParams params,
                                    const SurfaceInterval& interval);

    /// Create a new surface
    SurfaceId CreateSurface(const SurfaceParams& params, const SurfaceFlagBits& initial_flags = {});

    /// Register surface into the cache
    void RegisterSurface(SurfaceId surface);

    /// Remove surface from the cache
    void UnregisterSurface(SurfaceId surface);

    /// Unregisters all surfaces from the cache
    void UnregisterAll();

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta);

private:
    Memory::MemorySystem& memory;
    CustomTexManager& custom_tex_manager;
    Runtime& runtime;
    Pica::RegsInternal& regs;
    RendererBase& renderer;
    std::unordered_map<TextureCubeConfig, TextureCube> texture_cube_cache;
    tsl::robin_pg_map<u64, std::vector<SurfaceId>, Common::IdentityHash<u64>> page_table;
    std::unordered_map<FramebufferParams, FramebufferId> framebuffers;
    std::unordered_map<SamplerParams, SamplerId> samplers;
    std::list<std::pair<SurfaceId, u64>> sentenced;
    Common::SlotVector<Surface> slot_surfaces;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    /// The framebuffer the last draw used, so a change of render target can be counted
    /// (Common::PipelineStats::fb_switches).
    FramebufferId last_framebuffer{};
    SurfaceMap dirty_regions;
    PageMap cached_pages;
    u32 resolution_scale_factor;
    FramebufferParams fb_params;
    Settings::TextureFilter filter;
    bool dump_textures;
    bool use_custom_textures;
#ifdef CITRA_TRACE_PROBES
    /// AZAHAR_SURFACE_BUDGET_MB: trim the cache to this many (guest-scaled) bytes each frame,
    /// on any backend, so eviction can be exercised where memory is not short.
    u64 probe_budget_bytes = 0;
    /// AZAHAR_TRIM_LOG: log every surface given up and every reload of an address that was
    /// given up recently, with what guest memory held at the time.
    bool probe_trim_log = false;
    std::array<PAddr, 128> recent_evictions{};
    u32 recent_eviction_next = 0;
#endif
};

} // namespace VideoCore
