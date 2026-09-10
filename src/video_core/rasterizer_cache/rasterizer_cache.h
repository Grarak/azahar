// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <boost/container/small_vector.hpp>
#include <boost/range/iterator_range.hpp>
#include "common/alignment.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/pipeline_stats.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_base.h"
#include "video_core/texture/texture_decode.h"

namespace VideoCore {

MICROPROFILE_DECLARE(RasterizerCache_CopySurface);
MICROPROFILE_DECLARE(RasterizerCache_UploadSurface);
MICROPROFILE_DECLARE(RasterizerCache_DownloadSurface);
MICROPROFILE_DECLARE(RasterizerCache_Invalidation);

constexpr auto RangeFromInterval(const auto& map, const auto& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

template <class T>
RasterizerCache<T>::RasterizerCache(Memory::MemorySystem& memory_,
                                    CustomTexManager& custom_tex_manager_, Runtime& runtime_,
                                    Pica::RegsInternal& regs_, RendererBase& renderer_)
    : memory{memory_}, custom_tex_manager{custom_tex_manager_}, runtime{runtime_}, regs{regs_},
      renderer{renderer_}, resolution_scale_factor{renderer.GetResolutionScaleFactor()},
      filter{Settings::values.texture_filter.GetValue()},
      dump_textures{Settings::values.dump_textures.GetValue()},
      use_custom_textures{Settings::values.custom_textures.GetValue()} {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;
#ifdef CITRA_TRACE_PROBES
    if (const char* budget = std::getenv("AZAHAR_SURFACE_BUDGET_MB")) {
        probe_budget_bytes = static_cast<u64>(std::strtoul(budget, nullptr, 10)) << 20;
    }
    probe_trim_log = std::getenv("AZAHAR_TRIM_LOG") != nullptr;
#endif

    // Create null handles for all cached resources
    void(slot_surfaces.insert(runtime, SurfaceParams{
                                           .width = 1,
                                           .height = 1,
                                           .stride = 1,
                                           .texture_type = VideoCore::TextureType::Texture2D,
                                           .pixel_format = VideoCore::PixelFormat::RGBA8,
                                           .type = VideoCore::SurfaceType::Color,
                                       }));
    void(slot_surfaces.insert(runtime, SurfaceParams{
                                           .width = 1,
                                           .height = 1,
                                           .stride = 1,
                                           .texture_type = TextureType::CubeMap,
                                           .pixel_format = PixelFormat::RGBA8,
                                           .type = SurfaceType::Color,
                                       }));
    void(slot_samplers.insert(runtime, SamplerParams{
                                           .mag_filter = TextureConfig::TextureFilter::Linear,
                                           .min_filter = TextureConfig::TextureFilter::Linear,
                                           .mip_filter = TextureConfig::TextureFilter::Linear,
                                           .wrap_s = TextureConfig::WrapMode::ClampToBorder,
                                           .wrap_t = TextureConfig::WrapMode::ClampToBorder,
                                       }));

    auto& null_surface = slot_surfaces[NULL_SURFACE_ID];
    runtime.ClearTexture(null_surface, {
                                           .texture_level = 0,
                                           .texture_layer = 0,
                                           .texture_rect = null_surface.GetScaledRect(),
                                           .value =
                                               {
                                                   .color = {0.f, 0.f, 0.f, 0.f},
                                               },
                                       });

    auto& null_surface_cube = slot_surfaces[NULL_SURFACE_CUBE_ID];
    // Clear all cubemap faces
    for (u32 cube_layer_index = 0; cube_layer_index < 6; ++cube_layer_index) {
        runtime.ClearTexture(null_surface_cube,
                             {
                                 .texture_level = 0,
                                 .texture_layer = cube_layer_index,
                                 .texture_rect = null_surface_cube.GetScaledRect(),
                                 .value =
                                     {
                                         .color = {0.f, 0.f, 0.f, 0.f},
                                     },
                             });
    }
}

template <class T>
RasterizerCache<T>::~RasterizerCache() {
    ClearAll(false);
}

template <class T>
void RasterizerCache<T>::TickFrame() {
    custom_tex_manager.TickFrame();
    RunGarbageCollector();
#ifdef CITRA_TRACE_PROBES
    if (probe_budget_bytes != 0) {
        // A backend-neutral cost: the guest bytes scaled by the resolution factor. Close
        // enough to drive eviction on a machine that is not short of video memory.
        const auto cost = [](const Surface& surface) -> u64 {
            return static_cast<u64>(surface.size) * surface.res_scale * surface.res_scale;
        };
        u64 resident = 0;
        slot_surfaces.ForEach([&](SurfaceId id, const Surface& surface) {
            if (True(surface.flags & SurfaceFlagBits::Registered)) {
                const u64 bytes = cost(surface);
                if (bytes > (64u << 20) && probe_trim_log) {
                    LOG_INFO(HW_GPU, "trim: surface {} at {:08X} costs {} bytes ({}x{} stride {} "
                             "scale {} size {} type {})",
                             id.index, surface.addr, bytes, surface.width, surface.height,
                             surface.stride, surface.res_scale, surface.size,
                             static_cast<u32>(surface.type));
                }
                resident += bytes;
            }
        });
        u32 trimmed = 0;
        if (resident > probe_budget_bytes) {
            trimmed = TrimSurfaces(resident - probe_budget_bytes, cost,
                                   [](const Surface&) { return false; });
        }
        if (probe_trim_log && runtime.GetResourceTick() % 120 == 0) {
            LOG_INFO(HW_GPU, "trim: tick {} resident {} KiB budget {} KiB, {} sentenced, {} trimmed now",
                     runtime.GetResourceTick(), resident >> 10, probe_budget_bytes >> 10,
                     sentenced.size(), trimmed);
        }
    }
#endif

    const auto new_filter = Settings::values.texture_filter.GetValue();
    if (filter != new_filter) [[unlikely]] {
        filter = new_filter;
        UnregisterAll();
    }

    const u32 scale_factor = renderer.GetResolutionScaleFactor();
    const bool resolution_scale_changed = resolution_scale_factor != scale_factor;
    const bool use_custom_texture_changed =
        Settings::values.custom_textures.GetValue() != use_custom_textures;

    if (resolution_scale_changed || use_custom_texture_changed) {
        resolution_scale_factor = scale_factor;
        use_custom_textures = Settings::values.custom_textures.GetValue();
        if (use_custom_textures) {
            custom_tex_manager.FindCustomTextures();
        }
        UnregisterAll();
    }
}

template <class T>
void RasterizerCache<T>::SentenceSurfaces(std::span<const SurfaceId> ids) {
    for (const SurfaceId id : ids) {
        const Surface& surface = slot_surfaces[id];
        if (False(surface.flags & SurfaceFlagBits::Registered)) {
            continue;
        }
        // A surface that owns a dirty region holds pixels that exist nowhere else - the
        // guest's memory has not been written back - so dropping it would lose them. This is
        // what makes eviction safe: everything else can be uploaded again from guest memory.
        if (OwnsDirtyRegion(id)) {
            continue;
        }
#ifdef CITRA_TRACE_PROBES
        if (probe_trim_log) {
            LOG_INFO(HW_GPU,
                     "trim: give up surface {} at {:08X} size {} {}x{} stride {} {} type {} "
                     "flags {:#x} scale {} last used tick {} of {}",
                     id.index, surface.addr, surface.size, surface.width, surface.height,
                     surface.stride, PixelFormatAsString(surface.pixel_format),
                     static_cast<u32>(surface.type), static_cast<u32>(surface.flags),
                     surface.res_scale, surface.last_use_tick, runtime.GetResourceTick());
        }
        recent_evictions[recent_eviction_next] = surface.addr;
        recent_eviction_next = (recent_eviction_next + 1) % recent_evictions.size();
#endif
        UnregisterSurface(id);
    }
}

template <class T>
bool RasterizerCache<T>::OwnsDirtyRegion(SurfaceId surface_id) const {
    const Surface& surface = slot_surfaces[surface_id];
    for (const auto& pair : RangeFromInterval(dirty_regions, surface.GetInterval())) {
        if (pair.second == surface_id) {
            return true;
        }
    }
    return false;
}

template <class T>
template <typename CostFn, typename SkipFn>
u32 RasterizerCache<T>::TrimSurfaces(u64 want_bytes, CostFn&& cost, SkipFn&& skip) {
    struct Candidate {
        SurfaceId id;
        u64 last_use;
        u64 bytes;
    };
    boost::container::small_vector<Candidate, 64> candidates;
    slot_surfaces.ForEach([&](SurfaceId id, Surface& surface) {
        if (False(surface.flags & SurfaceFlagBits::Registered) || skip(surface)) {
            return;
        }
        // Never the surfaces the current frame is drawing with, whatever the runtime says:
        // the trim runs between frames and anything used this tick is bound somewhere.
        if (surface.last_use_tick >= runtime.GetResourceTick()) {
            return;
        }
        if (OwnsDirtyRegion(id)) {
            return;
        }
        candidates.push_back({id, surface.last_use_tick, cost(surface)});
    });
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.last_use < b.last_use; });
    boost::container::small_vector<SurfaceId, 64> give_up;
    u64 freed = 0;
    for (const Candidate& c : candidates) {
        if (freed >= want_bytes) {
            break;
        }
        give_up.push_back(c.id);
        freed += c.bytes;
    }
    if (give_up.empty()) {
        return 0;
    }
    SentenceSurfaces({give_up.data(), give_up.size()});
    return static_cast<u32>(give_up.size());
}

template <class T>
void RasterizerCache<T>::RunGarbageCollector() {
    // A sentenced surface is freed once the runtime's tick has moved a few steps past the
    // value recorded when it was sentenced. The tick says how far the GPU has got: frames
    // finished for OpenGL and GXM, the master semaphore's known GPU tick for Vulkan. The
    // margin stands in for the old RemoveThreshold, which was the swap chain's length.
    //
    // The test used to read `remove_tick >= resource_tick`, and both values come from the
    // same counter, which never goes backwards - so it was true every time, every sentenced
    // surface was skipped, and slot_surfaces.erase was unreachable. Nothing was ever
    // collected and no Surface destructor ever ran.
    //
    // On a desktop driver that only wastes memory. On the Vita the surface cache owns a
    // fixed pool of CDRAM and gets a block back only from ~Surface, so it filled
    // monotonically: Super Smash Bros reached 419 surfaces holding 51.5 MB of 56 MB after
    // two minutes of a fight, and the allocation that could not be satisfied then parked the
    // render thread for good. It also mints an SceGxmRenderTarget per surface pair and the
    // hardware allows 48.
    constexpr u64 RemoveThreshold = 4;
    const u64 remove_tick = runtime.GetResourceTick();
    for (auto it = sentenced.begin(); it != sentenced.end();) {
        const auto [surface_id, resource_tick] = *it;
        if (remove_tick < resource_tick + RemoveThreshold) {
            // Still close enough to the sentencing that the GPU may not be done with it.
            it++;
            continue;
        }
        RemoveFramebuffers(surface_id);
        slot_surfaces.erase(surface_id);
        it = sentenced.erase(it);
    }
}

template <class T>
void RasterizerCache<T>::RemoveFramebuffers(SurfaceId surface_id) {
    for (auto it = framebuffers.begin(); it != framebuffers.end();) {
        const auto& params = it->first;
        if (params.color_id == surface_id || params.depth_id == surface_id) {
            slot_framebuffers.erase(it->second);
            it = framebuffers.erase(it);
        } else {
            it++;
        }
    }
}

template <class T>
void RasterizerCache<T>::RemoveTextureCubeFace(SurfaceId surface_id) {
    if (False(slot_surfaces[surface_id].flags & SurfaceFlagBits::Tracked)) {
        return;
    }

    for (auto it = texture_cube_cache.begin(); it != texture_cube_cache.end();) {
        TextureCube& cube = it->second;
        for (SurfaceId& face_id : cube.face_ids) {
            if (face_id == surface_id) {
                face_id = SurfaceId{};
            }
        }
        if (std::none_of(cube.face_ids.begin(), cube.face_ids.end(),
                         [](SurfaceId id) { return id; })) {
            sentenced.emplace_back(cube.surface_id, runtime.GetResourceTick());
            it = texture_cube_cache.erase(it);
        } else {
            it++;
        }
    }
}

template <class T>
bool RasterizerCache<T>::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateTextureCopy ({})", config.DebugName()};

    // Texture copy size is aligned to 16 byte units
    const u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (input_width == 0 && input_gap != 0) {
        return false;
    }
    if (input_gap == 0 || input_width >= copy_size) {
        input_width = copy_size;
        input_gap = 0;
    }
    if (copy_size % input_width != 0) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (output_width == 0 && output_gap != 0) {
        return false;
    }
    if (output_gap == 0 || output_width >= copy_size) {
        output_width = copy_size;
        output_gap = 0;
    }
    if (copy_size % output_width != 0) {
        return false;
    }

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.stride = input_width + input_gap; // stride in bytes
    src_params.width = input_width;              // width in bytes
    src_params.height = copy_size / input_width;
    src_params.size = ((src_params.height - 1) * src_params.stride) + src_params.width;
    src_params.end = src_params.addr + src_params.size;

    const auto [src_surface_id, src_rect] = GetTexCopySurface(src_params);
    if (!src_surface_id) {
        return false;
    }

    const SurfaceParams src_info = slot_surfaces[src_surface_id];
    if (output_gap != 0 &&
        (output_width != src_info.BytesInPixels(src_rect.GetWidth() / src_info.res_scale) *
                             (src_info.is_tiled ? 8 : 1) ||
         output_gap % src_info.BytesInPixels(src_info.is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    SurfaceParams dst_params = src_info;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = src_rect.GetWidth() / src_info.res_scale;
    dst_params.stride =
        dst_params.width + src_info.PixelsInBytes(src_info.is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = src_rect.GetHeight() / src_info.res_scale;
    dst_params.res_scale = src_info.res_scale;
    dst_params.UpdateParams();

    // Since we are going to invalidate the gap if there is one, we will have to load it first
    const bool load_gap = output_gap != 0;
    const auto [dst_surface_id, dst_rect] =
        GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, load_gap);
    if (!dst_surface_id) {
        return false;
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    Surface& dst_surface = slot_surfaces[dst_surface_id];

    if (dst_surface.type == SurfaceType::Texture ||
        !CheckFormatsBlittable(src_surface.pixel_format, dst_surface.pixel_format)) {
        return false;
    }

    if (src_rect.GetWidth() != dst_rect.GetWidth()) {
        LOG_ERROR(
            HW_GPU,
            "Surface source and destination width mismatch, skipping... src_width={}, dst_width={}",
            src_rect.GetWidth(), dst_rect.GetHeight());
        return false;
    }

    const TextureCopy texture_copy = {
        .src_level = src_surface.LevelOf(src_params.addr),
        .dst_level = dst_surface.LevelOf(dst_params.addr),
        .src_offset = {src_rect.left, src_rect.bottom},
        .dst_offset = {dst_rect.left, dst_rect.bottom},
        .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
    };
    runtime.CopyTextures(src_surface, dst_surface, texture_copy);

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateDisplayTransfer ({})", config.DebugName()};

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.width = config.output_width;
    src_params.stride = config.input_width;
    src_params.height = config.output_height;
    src_params.is_tiled = !config.input_linear;
    src_params.pixel_format = PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.UpdateParams();

    // Using flip_vertically alongside crop_input_lines produces skewed output on hardware.
    // We have to emulate this because some games rely on this behaviour to render correctly.
    if (config.flip_vertically && config.crop_input_lines) {
        dst_params.addr += (config.input_width - config.output_width) * (config.output_height - 1) *
                           Pica::BytesPerPixel(config.output_format);
    }

    auto [src_surface_id, src_rect] = GetSurfaceSubRect(src_params, ScaleMatch::Ignore, true);
    if (!src_surface_id) {
        return false;
    }
    // What the transfer asked for against what the cache handed back. A source that does not
    // land inside the surface the guest just rendered into means a new surface was made and
    // validated from guest memory, which downloads the render target: Mario Kart 7's readback
    // storm (2026-09-09). AZAHAR_BLIT_LOG names the first few.
    static const bool log_xfers = std::getenv("AZAHAR_BLIT_LOG") != nullptr;
    if (log_xfers) {
        static u32 logged = 0;
        if (logged < 10) {
            logged++;
            const Surface& got = slot_surfaces[src_surface_id];
            LOG_INFO(HW_GPU,
                     "xfer {}: want src {:08x} {}x{} stride {} {} tiled {} -> got {:08x} {}x{} "
                     "stride {} {} tiled {} | want dst {:08x} {}x{} stride {} {} tiled {}",
                     logged, src_params.addr, src_params.width, src_params.height,
                     src_params.stride, PixelFormatAsString(src_params.pixel_format),
                     src_params.is_tiled, got.addr, got.width, got.height, got.stride,
                     PixelFormatAsString(got.pixel_format), got.is_tiled, dst_params.addr,
                     dst_params.width, dst_params.height, dst_params.stride,
                     PixelFormatAsString(dst_params.pixel_format), dst_params.is_tiled);
        }
    }

    dst_params.res_scale = slot_surfaces[src_surface_id].res_scale;

    const auto [dst_surface_id, dst_rect] =
        GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, false);
    if (!dst_surface_id) {
        return false;
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    Surface& dst_surface = slot_surfaces[dst_surface_id];

    if (src_surface.is_tiled != dst_surface.is_tiled) {
        std::swap(src_rect.top, src_rect.bottom);
    }
    if (config.flip_vertically) {
        std::swap(src_rect.top, src_rect.bottom);
    }

    if (!CheckFormatsBlittable(src_surface.pixel_format, dst_surface.pixel_format)) {
        return false;
    }

    const TextureBlit texture_blit = {
        .src_level = src_surface.LevelOf(src_params.addr),
        .dst_level = dst_surface.LevelOf(dst_params.addr),
        .src_rect = src_rect,
        .dst_rect = dst_rect,
    };
    static const bool log_blits = std::getenv("AZAHAR_BLIT_LOG") != nullptr;
    if (log_blits) {
        static u32 logged = 0;
        if (logged < 40) {
            logged++;
            LOG_INFO(HW_GPU, "blit {}: {:08x} {}x{} {} [{},{}-{},{}] -> {:08x} {}x{} {} "
                             "[{},{}-{},{}]",
                     logged, src_surface.addr, src_surface.width, src_surface.height,
                     PixelFormatAsString(src_surface.pixel_format), src_rect.left, src_rect.bottom,
                     src_rect.right, src_rect.top, dst_surface.addr, dst_surface.width,
                     dst_surface.height, PixelFormatAsString(dst_surface.pixel_format),
                     dst_rect.left, dst_rect.bottom, dst_rect.right, dst_rect.top);
        }
    }
    runtime.BlitTextures(src_surface, dst_surface, texture_blit);

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateFill(const Pica::MemoryFillConfig& config) {
    const DebugScope scope{runtime, Common::Vec4f{1.f, 0.f, 1.f, 1.f},
                           "RasterizerCache::AccelerateFill ({})", config.DebugName()};

    SurfaceParams params;
    params.addr = config.GetStartAddress();
    params.end = config.GetEndAddress();
    params.size = params.end - params.addr;
    params.type = SurfaceType::Fill;
    params.res_scale = std::numeric_limits<u16>::max();

    SurfaceId fill_surface_id = slot_surfaces.insert(runtime, params);
    Surface& fill_surface = slot_surfaces[fill_surface_id];

    std::memcpy(&fill_surface.fill_data[0], &config.value_32bit, sizeof(u32));
    if (config.fill_32bit) {
        fill_surface.fill_size = 4;
    } else if (config.fill_24bit) {
        fill_surface.fill_size = 3;
    } else {
        fill_surface.fill_size = 2;
    }

    RegisterSurface(fill_surface_id);
    InvalidateRegion(fill_surface.addr, fill_surface.size, fill_surface_id);
    return true;
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetSurface(SurfaceId surface_id) {
    return slot_surfaces[surface_id];
}

template <class T>
typename T::Sampler& RasterizerCache<T>::GetSampler(SamplerId sampler_id) {
    return slot_samplers[sampler_id];
}

template <class T>
typename T::Sampler& RasterizerCache<T>::GetSampler(
    const Pica::TexturingRegs::TextureConfig& config) {
    using TextureFilter = Pica::TexturingRegs::TextureConfig::TextureFilter;

    const auto get_filter = [](TextureFilter filter) {
        switch (Settings::values.texture_sampling.GetValue()) {
        case Settings::TextureSampling::GameControlled:
            return filter;
        case Settings::TextureSampling::NearestNeighbor:
            return TextureFilter::Nearest;
        case Settings::TextureSampling::Linear:
            return TextureFilter::Linear;
        default:
            return filter;
        }
    };

    const SamplerParams params = {
        .mag_filter = get_filter(config.mag_filter),
        .min_filter = get_filter(config.min_filter),
        .mip_filter = get_filter(config.mip_filter),
        .wrap_s = config.wrap_s,
        .wrap_t = config.wrap_t,
        .border_color = config.border_color.raw,
        .lod_min = config.lod.min_level,
        .lod_max = config.lod.max_level,
        .lod_bias = config.lod.bias,
    };

    auto [it, is_new] = samplers.try_emplace(params);
    if (is_new) {
        it->second = slot_samplers.insert(runtime, params);
    }

    return slot_samplers[it->second];
}

template <class T>
void RasterizerCache<T>::CopySurface(Surface& src_surface, Surface& dst_surface,
                                     SurfaceInterval copy_interval) {
    MICROPROFILE_SCOPE(RasterizerCache_CopySurface);
    const PAddr copy_addr = copy_interval.lower();
    const SurfaceParams subrect_params = dst_surface.FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);

    if (src_surface.type == SurfaceType::Fill) {
        const TextureClear clear = {
            .texture_level = dst_surface.LevelOf(copy_addr),
            .texture_rect = dst_surface.GetScaledSubRect(subrect_params),
            .value = src_surface.MakeClearValue(copy_addr, dst_surface.pixel_format),
        };
        runtime.ClearTexture(dst_surface, clear);
        return;
    }

    const u32 src_scale = src_surface.res_scale;
    const u32 dst_scale = dst_surface.res_scale;
    if (src_scale > dst_scale) {
        dst_surface.ScaleUp(src_scale);
    }

    const auto src_rect = src_surface.GetScaledSubRect(subrect_params);
    const auto dst_rect = dst_surface.GetScaledSubRect(subrect_params);
    if (src_scale == dst_scale) {
        const TextureCopy copy = {
            .src_level = src_surface.LevelOf(copy_addr),
            .dst_level = dst_surface.LevelOf(copy_addr),
            .src_offset = {src_rect.left, src_rect.bottom},
            .dst_offset = {dst_rect.left, dst_rect.bottom},
            .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
        };
        runtime.CopyTextures(src_surface, dst_surface, copy);
    } else {
        const TextureBlit blit = {
            .src_level = src_surface.LevelOf(copy_addr),
            .dst_level = dst_surface.LevelOf(copy_addr),
            .src_rect = src_rect,
            .dst_rect = dst_rect,
        };
        runtime.BlitTextures(src_surface, dst_surface, blit);
    }
}

template <class T>
SurfaceId RasterizerCache<T>::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                         bool load_if_create,
                                         const SurfaceFlagBits& create_initial_flags) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return {};
    }
    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);
    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    SurfaceId surface_id = FindMatch<MatchFlags::Exact>(params, match_res_scale);

    if (!surface_id) {
        surface_id = CreateSurface(params, create_initial_flags);
        RegisterSurface(surface_id);
    }

    if (load_if_create) {
        ValidateSurface(surface_id, params.addr, params.size);
    }

    return surface_id;
}

template <class T>
typename RasterizerCache<T>::SurfaceRect_Tuple RasterizerCache<T>::GetSurfaceSubRect(
    const SurfaceParams& params, ScaleMatch match_res_scale, bool load_if_create,
    const SurfaceFlagBits& create_initial_flags) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return std::make_pair(SurfaceId{}, Common::Rectangle<u32>{});
    }

    // Attempt to find encompassing surface
    SurfaceId surface_id = FindMatch<MatchFlags::SubRect>(params, match_res_scale);

    // Check if FindMatch failed because of res scaling. If that's the case create a new surface
    // with the dimensions of the lower res_scale surface to suggest it should not be used again.
    if (!surface_id && match_res_scale != ScaleMatch::Ignore) {
        surface_id = FindMatch<MatchFlags::SubRect>(params, ScaleMatch::Ignore);
        if (surface_id) {
            SurfaceParams new_params = slot_surfaces[surface_id];
            new_params.res_scale = params.res_scale;

            surface_id = CreateSurface(new_params, create_initial_flags);
            RegisterSurface(surface_id);
        }
    }

    SurfaceParams aligned_params = params;
    if (params.is_tiled) {
        aligned_params.height = Common::AlignUp(params.height, 8);
        aligned_params.width = Common::AlignUp(params.width, 8);
        aligned_params.stride = Common::AlignUp(params.stride, 8);
        aligned_params.UpdateParams();
    }

    // No subrect found - create and return a new surface
    if (!surface_id) {
        SurfaceParams new_params = aligned_params;
        // Can't have gaps in a surface
        new_params.width = aligned_params.stride;
        new_params.UpdateParams();
        // GetSurface will create the new surface and possibly adjust res_scale if necessary
        surface_id = GetSurface(new_params, match_res_scale, load_if_create, create_initial_flags);
    } else if (load_if_create) {
        ValidateSurface(surface_id, aligned_params.addr, aligned_params.size);
    }

    return std::make_pair(surface_id, slot_surfaces[surface_id].GetScaledSubRect(params));
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetTextureSurface(
    const Pica::TexturingRegs::FullTextureConfig& config) {
    const auto info = Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    const u32 max_level = MipLevels(info.width, info.height, config.config.lod.max_level) - 1;
    const SurfaceId surface_id = GetTextureSurface(info, max_level);
    return slot_surfaces[surface_id];
}

template <class T>
SurfaceId RasterizerCache<T>::GetTextureSurface(const Pica::Texture::TextureInfo& info,
                                                u32 max_level) {
    validate_reason = "texture";
    if (info.physical_address == 0) [[unlikely]] {
        // Can occur when texture addr is null or its memory is unmapped/invalid
        // HACK: In this case, the correct behaviour for the PICA is to use the last
        // rendered colour. But because this would be impractical to implement, the
        // next best alternative is to use a clear texture, essentially skipping
        // the geometry in question.
        // For example: a bug in Pokemon X/Y causes NULL-texture squares to be drawn
        // on the male character's face, which in the OpenGL default appear black.
        return NULL_SURFACE_ID;
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.levels = max_level + 1;
    params.is_tiled = true;
    params.pixel_format = PixelFormatFromTextureFormat(info.format);
    params.res_scale = filter != Settings::TextureFilter::NoFilter ? resolution_scale_factor : 1;
    SurfaceFlagBits initial_flags{};
    if (info.is_shadow_source) {
        initial_flags |= SurfaceFlagBits::ShadowSource;
    }
    params.UpdateParams();

    const u32 min_width = info.width >> max_level;
    const u32 min_height = info.height >> max_level;
    if (min_width % 8 != 0 || min_height % 8 != 0) {
        if (min_width % 4 != 0 || min_height % 4 != 0) {
            LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) is not multiple of 4", min_width,
                         min_height);
            return NULL_SURFACE_ID;
        }
        const auto [src_surface_id, rect] =
            GetSurfaceSubRect(params, ScaleMatch::Ignore, true, initial_flags);
        Surface& src_surface = slot_surfaces[src_surface_id];

        params.res_scale = src_surface.res_scale;
        SurfaceId tmp_surface_id = CreateSurface(params, initial_flags);
        Surface& tmp_surface = slot_surfaces[tmp_surface_id];
        sentenced.emplace_back(tmp_surface_id, runtime.GetResourceTick());

        const TextureBlit blit = {
            .src_level = src_surface.LevelOf(params.addr),
            .dst_level = 0,
            .src_rect = rect,
            .dst_rect = tmp_surface.GetScaledRect(),
        };
        runtime.BlitTextures(src_surface, tmp_surface, blit);
        return tmp_surface_id;
    }
    if (info.width != (min_width << max_level) || info.height != (min_height << max_level)) {
        LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) does not support required mipmap level ({})",
                     params.width, params.height, max_level);
        return NULL_SURFACE_ID;
    }

    SurfaceId surface_id = GetSurface(params, ScaleMatch::Ignore, true, initial_flags);
    return surface_id ? surface_id : NULL_SURFACE_ID;
}

template <class T>
typename T::Surface& RasterizerCache<T>::GetTextureCube(const TextureCubeConfig& config) {
    if (config.width == 0) [[unlikely]] {
        return slot_surfaces[NULL_SURFACE_CUBE_ID];
    }

    auto [it, new_surface] = texture_cube_cache.try_emplace(config);
    TextureCube& cube = it->second;

    const std::array addresses = {config.px, config.nx, config.py, config.ny, config.pz, config.nz};

    if (new_surface) {
        Pica::Texture::TextureInfo info = {
            .width = config.width,
            .height = config.width,
            .format = config.format,
        };
        info.SetDefaultStride();

        u32 res_scale = 1;
        for (u32 i = 0; i < addresses.size(); i++) {
            if (!addresses[i]) {
                continue;
            }

            SurfaceId& face_id = cube.face_ids[i];
            if (!face_id) {
                info.physical_address = addresses[i];
                face_id = GetTextureSurface(info, config.levels - 1);
                Surface& surface = slot_surfaces[face_id];
                ASSERT_MSG(
                    surface.levels >= config.levels,
                    "Texture cube face levels are not enough to validate the levels requested");
                surface.flags |= SurfaceFlagBits::Tracked;
            }
            Surface& surface = slot_surfaces[face_id];
            res_scale = std::max(surface.res_scale, res_scale);
        }

        SurfaceParams cube_params = {
            .addr = config.px,
            .width = config.width,
            .height = config.width,
            .stride = config.width,
            .levels = config.levels,
            .res_scale = res_scale,
            .texture_type = TextureType::CubeMap,
            .pixel_format = PixelFormatFromTextureFormat(config.format),
            .type = SurfaceType::Texture,
        };
        cube_params.UpdateParams();
        cube.surface_id = CreateSurface(cube_params);
    }

    Surface& cube_surface = slot_surfaces[cube.surface_id];
    for (u32 i = 0; i < addresses.size(); i++) {
        const SurfaceId& face_id = cube.face_ids[i];
        if (!addresses[i] || !face_id) {
            continue;
        }
        Surface& surface = slot_surfaces[face_id];
        if (cube.ticks[i] == surface.modification_tick) {
            continue;
        }
        cube.ticks[i] = surface.modification_tick;
        boost::container::small_vector<TextureCopy, 8> upload_copies;
        for (u32 level = 0; level < config.levels; level++) {
            const u32 width_lod = surface.GetScaledWidth() >> level;
            upload_copies.push_back({
                .src_level = level,
                .dst_level = level,
                .src_layer = 0,
                .dst_layer = i,
                .src_offset = {0, 0},
                .dst_offset = {0, 0},
                .extent = {width_lod, width_lod},
            });
        }
        runtime.CopyTextures(surface, cube_surface, upload_copies);
    }

    return slot_surfaces[cube.surface_id];
}

template <class T>
FramebufferHelper<T> RasterizerCache<T>::GetFramebufferSurfaces(bool using_color_fb,
                                                                bool using_depth_fb) {
    validate_reason = "framebuffer";
    const auto& config = regs.framebuffer.framebuffer;

    const s32 framebuffer_width = config.GetWidth();
    const s32 framebuffer_height = config.GetHeight();
    const auto viewport_rect = regs.rasterizer.GetViewportRect();
    const Common::Rectangle<u32> viewport_clamped = {
        static_cast<u32>(std::clamp(viewport_rect.left, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, framebuffer_height)),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.bottom, 0, framebuffer_height)),
    };

    SurfaceParams color_params;
    color_params.is_tiled = true;
    color_params.res_scale = resolution_scale_factor;
    color_params.width = config.GetWidth();
    color_params.height = config.GetHeight();
    SurfaceParams depth_params = color_params;

    color_params.addr = config.GetColorBufferPhysicalAddress();
    color_params.pixel_format = PixelFormatFromColorFormat(config.color_format);
    color_params.UpdateParams();

    depth_params.addr = config.GetDepthBufferPhysicalAddress();
    depth_params.pixel_format = PixelFormatFromDepthFormat(config.depth_format);
    depth_params.UpdateParams();

    auto color_vp_interval = color_params.GetSubRectInterval(viewport_clamped);
    auto depth_vp_interval = depth_params.GetSubRectInterval(viewport_clamped);

    Common::Rectangle<u32> color_rect{};
    SurfaceId color_id{};
    u32 color_level{};
    if (using_color_fb)
        std::tie(color_id, color_rect) = GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> depth_rect{};
    SurfaceId depth_id{};
    u32 depth_level{};
    if (using_depth_fb)
        std::tie(depth_id, depth_rect) = GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> fb_rect{};
    if (color_id && depth_id) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_id = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_id = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = slot_surfaces[color_id].GetScaledRect();
        }
    } else if (color_id) {
        fb_rect = color_rect;
    } else if (depth_id) {
        fb_rect = depth_rect;
    }

    Surface* color_surface = color_id ? &slot_surfaces[color_id] : nullptr;
    Surface* depth_surface = depth_id ? &slot_surfaces[depth_id] : nullptr;

    if (color_id) {
        color_level = color_surface->LevelOf(color_params.addr);
        color_surface->flags |= SurfaceFlagBits::RenderTarget;
        ValidateSurface(color_id, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
    }
    if (depth_id) {
        depth_level = depth_surface->LevelOf(depth_params.addr);
        depth_surface->flags |= SurfaceFlagBits::RenderTarget;
        ValidateSurface(depth_id, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval));
    }

    const FramebufferParams fb_params = {
        .color_id = color_id,
        .depth_id = depth_id,
        .color_level = color_level,
        .depth_level = depth_level,
        .shadow_rendering = regs.framebuffer.IsShadowRendering(),
    };

    auto [it, new_framebuffer] = framebuffers.try_emplace(fb_params);
    if (new_framebuffer) {
        it->second = slot_framebuffers.insert(runtime, fb_params, color_surface, depth_surface);
    }

    Common::PipelineStats::fb_binds.fetch_add(1, std::memory_order_relaxed);
    if (it->second != last_framebuffer) {
        last_framebuffer = it->second;
        Common::PipelineStats::fb_switches.fetch_add(1, std::memory_order_relaxed);
    }

    return FramebufferHelper<T>{this, &slot_framebuffers[it->second],
                                regs.framebuffer.framebuffer.IsFlipped(), regs.rasterizer, fb_rect};
}

template <class T>
typename RasterizerCache<T>::SurfaceRect_Tuple RasterizerCache<T>::GetTexCopySurface(
    const SurfaceParams& params) {
    Common::Rectangle<u32> rect{};

    SurfaceId match_id = FindMatch<MatchFlags::TexCopy>(params, ScaleMatch::Ignore);

    if (match_id) {
        ValidateSurface(match_id, params.addr, params.size);

        SurfaceParams match_subrect;
        Surface& match_surface = slot_surfaces[match_id];
        if (params.width != params.stride) {
            const u32 tiled_size = match_surface.is_tiled ? 8 : 1;
            match_subrect = params;
            match_subrect.width = match_surface.PixelsInBytes(params.width) / tiled_size;
            match_subrect.stride = match_surface.PixelsInBytes(params.stride) / tiled_size;
            match_subrect.height *= tiled_size;
        } else {
            match_subrect = match_surface.FromInterval(params.GetInterval());
            ASSERT(match_subrect.GetInterval() == params.GetInterval());
        }

        rect = match_surface.GetScaledSubRect(match_subrect);
    }

    return std::make_pair(match_id, rect);
}

template <class T>
template <typename Func>
void RasterizerCache<T>::ForEachSurfaceInRegion(PAddr addr, std::size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, SurfaceId, Surface&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<SurfaceId, 8> surfaces;
    ForEachPage(addr, size, [this, &surfaces, addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const SurfaceId surface_id : it->second) {
            Surface& surface = slot_surfaces[surface_id];
            if (True(surface.flags & SurfaceFlagBits::Picked)) {
                continue;
            }
            if (!surface.Overlaps(addr, size)) {
                continue;
            }

            surface.flags |= SurfaceFlagBits::Picked;
            surfaces.push_back(surface_id);
            if constexpr (BOOL_BREAK) {
                if (func(surface_id, surface)) {
                    return true;
                }
            } else {
                func(surface_id, surface);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const SurfaceId surface_id : surfaces) {
        slot_surfaces[surface_id].flags &= ~SurfaceFlagBits::Picked;
    }
}

template <class T>
template <MatchFlags find_flags>
SurfaceId RasterizerCache<T>::FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                                        std::optional<SurfaceInterval> validate_interval) {
    SurfaceId match_id{};
    bool match_valid = false;
    u32 match_scale = 0;
    SurfaceInterval match_interval{};

    ForEachSurfaceInRegion(params.addr, params.size, [&](SurfaceId surface_id, Surface& surface) {
        const bool res_scale_matched = match_scale_type == ScaleMatch::Exact
                                           ? (params.res_scale == surface.res_scale)
                                           : (params.res_scale <= surface.res_scale);
        const bool is_valid =
            True(find_flags & MatchFlags::Copy)
                ? true
                : surface.IsRegionValid(validate_interval.value_or(params.GetInterval()));

        auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
            if (False(find_flags & check_type))
                return;

            bool matched;
            SurfaceInterval surface_interval;
            std::tie(matched, surface_interval) = match_fn();
            if (!matched)
                return;

            if (!res_scale_matched && match_scale_type != ScaleMatch::Ignore &&
                surface.type != SurfaceType::Fill)
                return;

            // Found a match, update only if this is better than the previous one
            auto UpdateMatch = [&] {
                match_id = surface_id;
                match_valid = is_valid;
                match_scale = surface.res_scale;
                match_interval = surface_interval;
            };

            if (surface.res_scale > match_scale) {
                UpdateMatch();
                return;
            } else if (surface.res_scale < match_scale) {
                return;
            }

            if (is_valid && !match_valid) {
                UpdateMatch();
                return;
            } else if (is_valid != match_valid) {
                return;
            }

            if (boost::icl::length(surface_interval) > boost::icl::length(match_interval)) {
                UpdateMatch();
            }
        };
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Exact>{}, [&] {
            return std::make_pair(surface.ExactMatch(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::SubRect>{}, [&] {
            return std::make_pair(surface.CanSubRect(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Copy>{}, [&] {
            ASSERT(validate_interval);
            const SurfaceInterval copy_interval =
                surface.GetCopyableInterval(params.FromInterval(*validate_interval));
            const bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                                 surface.CanCopy(params, copy_interval);
            return std::make_pair(matched, copy_interval);
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Reinterpret>{}, [&] {
            ASSERT(validate_interval);
            const SurfaceInterval copy_interval =
                surface.GetCopyableInterval(params.FromInterval(*validate_interval));
            const bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                                 surface.CanReinterpret(params);
            return std::make_pair(matched, copy_interval);
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::TexCopy>{}, [&] {
            return std::make_pair(surface.CanTexCopy(params), surface.GetInterval());
        });
    });
    return match_id;
}

template <class T>
void RasterizerCache<T>::ValidateSurface(SurfaceId surface_id, PAddr addr, u32 size) {
    if (size == 0) [[unlikely]] {
        return;
    }

    Surface& surface = slot_surfaces[surface_id];
    const SurfaceInterval validate_interval(addr, addr + size);
    surface.last_use_tick = runtime.GetResourceTick();

    if (surface.type == SurfaceType::Fill) {
        ASSERT_MSG(surface.IsRegionValid(validate_interval),
                   "Attempted to validate a non-valid fill surface");
        return;
    }

    SurfaceRegions validate_regions = surface.invalid_regions & validate_interval;

    if (validate_regions.empty()) {
        return;
    }

    auto notify_validated = [&](SurfaceInterval interval) {
        surface.MarkValid(interval);
        validate_regions.erase(interval);
    };

    const DebugScope scope{runtime, Common::Vec4f{0.f, 1.f, 0.f, 1.f},
                           "RasterizerCache::ValidateSurface (from {:#x} to {:#x})", addr,
                           addr + size};

    u32 level = surface.LevelOf(addr);
    SurfaceInterval level_interval = surface.LevelInterval(level);
    while (!validate_regions.empty()) {
        // Take an invalid interval from the validation regions and clamp it
        // to the current level interval. If the interval is empty
        // then we have validated the entire level so move to the next.
        const auto interval = *validate_regions.begin() & level_interval;
        if (boost::icl::is_empty(interval)) {
            level_interval = surface.LevelInterval(++level);
            continue;
        }

        // Look for a valid surface to copy from.
        const SurfaceParams params = surface.FromInterval(interval);
        const SurfaceId copy_surface_id =
            FindMatch<MatchFlags::Copy>(params, ScaleMatch::Ignore, interval);
        if (copy_surface_id && copy_surface_id != surface_id) {
            Surface& copy_surface = slot_surfaces[copy_surface_id];
            const SurfaceInterval copy_interval = copy_surface.GetCopyableInterval(params);
            static const bool log_copies = std::getenv("AZAHAR_CACHE_LOG") != nullptr;
            if (log_copies) {
                static u32 shown = 0;
                if (shown++ < 200) {
                    LOG_INFO(HW_GPU, "validate for {}: {:08X} {}x{} from {:08X} {}x{} ({:X}..{:X})",
                             validate_reason, surface.addr, surface.width, surface.height,
                             copy_surface.addr, copy_surface.width, copy_surface.height,
                             copy_interval.lower(), copy_interval.upper());
                }
            }
            CopySurface(copy_surface, surface, copy_interval);
            surface.upload_hash = 0;
            notify_validated(copy_interval);
            continue;
        }

        // Try to find surface in cache with different format
        // that can can be reinterpreted to the requested format.
        if (ValidateByReinterpretation(surface, params, interval)) {
            surface.upload_hash = 0;
            notify_validated(interval);
            continue;
        }

        if (True(surface.flags & SurfaceFlagBits::GuestFlushed) &&
            interval == surface.GetInterval()) {
            // The guest flushed this texture's memory; whether it changed is decided here,
            // once, by the bytes.
            surface.flags &= ~SurfaceFlagBits::GuestFlushed;
            surface.last_hash_tick = runtime.GetResourceTick();
            const MemoryRef source = memory.GetPhysicalRef(surface.addr);
            if (source && source.GetSize() >= surface.size) {
                const u64 hash = Common::ComputeHash64(source.GetPtr(), surface.size);
                if (hash == surface.upload_hash) {
                    guest_flush_hash_skips++;
                    memory.WriteTracker().Arm(source.GetPtr(), surface.size);
                    notify_validated(interval);
                    continue;
                }
            }
        }
        // Whose validation forced the flush, for the readback log below.
        validating_for = &surface;
        FlushRegion(params.addr, params.size);
        validating_for = nullptr;
        if (!use_custom_textures || !UploadCustomSurface(surface_id, interval)) {
            UploadSurface(surface, interval);
        }
        notify_validated(params.GetInterval());
    }

    // Filtered mipmaps often look really bad. We can achieve better quality by
    // generating them from the base level.
    if (surface.res_scale != 1 && level != 0) {
        runtime.GenerateMipmaps(surface);
    }
}

template <class T>
void RasterizerCache<T>::UploadSurface(Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_UploadSurface);

    const SurfaceParams load_info = surface.FromInterval(interval);
    ASSERT(load_info.addr >= surface.addr && load_info.end <= surface.end);

    const auto staging = runtime.FindStaging(
        load_info.width * load_info.height * surface.GetInternalBytesPerPixel(), true);

    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return;
    }

    const auto upload_data = source_ptr.GetWriteBytes(load_info.end - load_info.addr);
    // What was just read from guest memory is what the guest must not change unnoticed.
    memory.WriteTracker().Arm(upload_data.data(), upload_data.size());
    // Whole uploads of textures and colour surfaces are hashed so a later guest flush of the
    // same bytes can skip the upload. Smash's title screen uploads a CPU-written 512x256 RGBA4
    // colour target every frame (256 KiB, the largest per-frame cost on the render thread).
    if (surface.type != SurfaceType::Texture && surface.type != SurfaceType::Color) {
        // Depth data is not compared.
    } else if (load_info.addr == surface.addr && load_info.end == surface.end) {
        surface.upload_hash = Common::ComputeHash64(upload_data.data(), upload_data.size());
    } else {
        surface.upload_hash = 0;
    }
#ifdef CITRA_TRACE_PROBES
    if (probe_trim_log) {
        const bool evicted_before =
            std::find(recent_evictions.begin(), recent_evictions.end(), surface.addr) !=
            recent_evictions.end();
        if (evicted_before) {
            // What guest memory holds for a surface that was given up: all zero and all 0xFF
            // are the two shapes a lost texture takes.
            std::size_t zero = 0, ones = 0;
            for (const u8 byte : upload_data) {
                zero += byte == 0;
                ones += byte == 0xFF;
            }
            LOG_INFO(HW_GPU,
                     "trim: reload {:08X}+{} into surface at {:08X} {}x{} {} type {}: {} of {} "
                     "bytes zero, {} are ff",
                     load_info.addr, load_info.end - load_info.addr, surface.addr, surface.width,
                     surface.height, PixelFormatAsString(surface.pixel_format),
                     static_cast<u32>(surface.type), zero, upload_data.size(), ones);
        }
    }
#endif
    const BufferTextureCopy upload = {
        .buffer_offset = staging.offset,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(load_info),
        .texture_level = surface.LevelOf(load_info.addr),
    };

    const bool should_dump = False(surface.flags & SurfaceFlagBits::Custom) &&
                             False(surface.flags & SurfaceFlagBits::RenderTarget);
    if (dump_textures && should_dump) {
        const u64 hash = ComputeHash(load_info, upload_data);
        const u32 level = surface.LevelOf(load_info.addr);
        custom_tex_manager.DumpTexture(load_info, level, upload_data, hash);
    }

    // A runtime whose texture memory the CPU can write decodes straight into it when the
    // destination is contiguous, and skips the staging copy.
    if constexpr (requires { surface.DirectUploadSpan(upload, staging.size); }) {
        const std::span<u8> direct = surface.DirectUploadSpan(upload, staging.size);
        if (!direct.empty()) {
            DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, direct,
                          runtime.NeedsConversion(surface));
            return;
        }
    }
    DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, staging.mapped,
                  runtime.NeedsConversion(surface));
    surface.Upload(upload, staging);
}

template <class T>
u64 RasterizerCache<T>::ComputeHash(const SurfaceParams& load_info, std::span<u8> upload_data) {
    if (!custom_tex_manager.UseNewHash()) {
        const u32 width = load_info.width;
        const u32 height = load_info.height;
        const u32 bpp = GetFormatBytesPerPixel(load_info.pixel_format);
        auto decoded = std::vector<u8>(width * height * bpp);
        DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, decoded, false);
        return Common::ComputeHash64<Common::HashAlgo64::CityHash>(decoded.data(), decoded.size());
    } else {
        return Common::ComputeHash64<Common::HashAlgo64::CityHash>(upload_data.data(),
                                                                   upload_data.size());
    }
}

template <class T>
bool RasterizerCache<T>::UploadCustomSurface(SurfaceId surface_id, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_UploadSurface);

    Surface& surface = slot_surfaces[surface_id];
    const SurfaceParams load_info = surface.FromInterval(interval);
    ASSERT(load_info.addr >= surface.addr && load_info.end <= surface.end);

    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return false;
    }

    const auto upload_data = source_ptr.GetWriteBytes(load_info.end - load_info.addr);
    const u64 hash = ComputeHash(load_info, upload_data);

    const u32 level = surface.LevelOf(load_info.addr);
    Material* material = custom_tex_manager.GetMaterial(hash);

    if (!material) {
        return surface.IsCustom();
    }
    if (level != 0 && custom_tex_manager.SkipMipmaps()) {
        return true;
    }

    surface.flags |= SurfaceFlagBits::Custom;

    const auto upload = [this, level, surface_id, material]() -> bool {
        ASSERT_MSG(True(slot_surfaces[surface_id].flags & SurfaceFlagBits::Custom),
                   "Surface is not suitable for custom upload, aborting!");
        if (!slot_surfaces[surface_id].IsCustom()) {
            const SurfaceBase old_surface{slot_surfaces[surface_id]};
            const SurfaceId old_id =
                slot_surfaces.swap_and_insert(surface_id, runtime, old_surface, material);
            slot_surfaces[old_id].flags &= ~SurfaceFlagBits::Registered;
            sentenced.emplace_back(old_id, runtime.GetResourceTick());
        }
        Surface& surface = slot_surfaces[surface_id];
        surface.UploadCustom(material, level);
        if (custom_tex_manager.SkipMipmaps()) {
            runtime.GenerateMipmaps(surface);
        }
        return true;
    };
    return custom_tex_manager.Decode(material, std::move(upload));
}

template <class T>
void RasterizerCache<T>::DumpSurfaces(const std::string& dir, u32 frame) {
    // Level 0 of every surface that has memory, read back the way a download is, as a PPM of
    // its colour or depth and a PGM of its alpha or stencil where the format carries one.
    // Rows are stored bottom-up and the files are written top-down.
    FileUtil::CreateFullPath(dir + "/");
    const std::string index_path = fmt::format("{}/f{:02}_index.txt", dir, frame);
    FILE* index = std::fopen(index_path.c_str(), "wb");
    u32 count = 0;
    ForEachSurface([&](SurfaceId, Surface& surface) {
        if (surface.addr == 0 || surface.type == SurfaceType::Invalid || surface.width == 0 ||
            surface.height == 0) {
            return;
        }
        const u32 n = count++;
        const u32 w = surface.width;
        const u32 h = surface.height;
        const PixelFormat pf = surface.pixel_format;
        const u32 bpp = surface.GetInternalBytesPerPixel();
        const u32 bytes = w * h * bpp;
        const auto staging = runtime.FindStaging(bytes, false);
        surface.Download(BufferTextureCopy{.buffer_offset = 0,
                                           .buffer_size = bytes,
                                           .texture_rect = {0, h, w, 0},
                                           .texture_level = 0},
                         staging);
        static constexpr const char* TypeNames[] = {"color", "texture", "depth", "depthstencil",
                                                    "fill", "invalid"};
        const char* type = TypeNames[std::min<u32>(static_cast<u32>(surface.type), 5)];
        const std::string stem =
            fmt::format("{}/f{:02}_{:03}_{}_{:08x}_{}x{}_{}", dir, frame, n, type, surface.addr, w,
                        h, PixelFormatAsString(pf));
        if (index != nullptr) {
            const std::string line =
                fmt::format("{:03} {} addr {:08x}-{:08x} {}x{} stride {} levels {} {}{}{}\n", n,
                            type, surface.addr, surface.end, w, h, surface.stride, surface.levels,
                            PixelFormatAsString(pf),
                            surface.type == SurfaceType::Color ? " colour-target" : "",
                            surface.type == SurfaceType::Depth ||
                                    surface.type == SurfaceType::DepthStencil
                                ? " depth-target"
                                : "");
            std::fwrite(line.data(), 1, line.size(), index);
        }
        const bool depth_word = pf == PixelFormat::D24S8;
        const bool depth_f32 = pf == PixelFormat::D24;
        const bool depth_16 = pf == PixelFormat::D16;
        const bool has_alpha = !depth_word && !depth_f32 && !depth_16 &&
                               pf != PixelFormat::RGB565 && pf != PixelFormat::RGB8;
        FILE* ppm = std::fopen((stem + ".ppm").c_str(), "wb");
        FILE* pgm = (has_alpha || depth_word)
                        ? std::fopen((stem + (depth_word ? "_s.pgm" : "_a.pgm")).c_str(), "wb")
                        : nullptr;
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
                    u32 v;
                    std::memcpy(&v, px, 4);
                    r = g = b = static_cast<u8>(v >> 24);
                    a = static_cast<u8>(v & 0xff);
                    break;
                }
                default:
                    r = px[0];
                    g = bpp > 1 ? px[1] : px[0];
                    b = bpp > 2 ? px[2] : px[0];
                    a = bpp > 3 ? px[3] : 255;
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
    LOG_INFO(HW_GPU, "dumped {} surfaces to {}", count, dir);
}

template <class T>
void RasterizerCache<T>::DownloadSurface(Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_DownloadSurface);

    // A readback the guest asks for while the GPU is still writing the surface costs a full
    // drain of the pipeline before it can be answered, and Mario Kart 7 asks 140 times a
    // second (27 MB, two and a half screenfuls a frame, measured on the pi5 2026-09-09). What
    // the last readback left in guest memory is a frame or so old and free, so it is handed
    // over instead - but only so many times in a row, or a surface read every frame would
    // freeze at its first download and never advance again.
    constexpr u8 MaxStaleReadbacks = 2;
    if (surface.ever_downloaded && surface.download_skips < MaxStaleReadbacks &&
        runtime.CanSkipDownload(surface)) {
        surface.download_skips++;
        Common::PipelineStats::cache_stale_reads.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    surface.download_skips = 0;
    surface.ever_downloaded = true;
    // What the guest actually reads back, for deciding whether it can be answered stale:
    // AZAHAR_READBACK_LOG=1 names the first ranges and their surfaces.
    static const bool log_readbacks = std::getenv("AZAHAR_READBACK_LOG") != nullptr;
    if (log_readbacks) {
        static u32 logged = 0;
        if (logged < 12) {
            logged++;
            LOG_INFO(HW_GPU,
                     "readback {}: {:#010x}+{:#x} surface {:#010x} {}x{} {} {} for {}", logged,
                     boost::icl::first(interval),
                     boost::icl::last_next(interval) - boost::icl::first(interval), surface.addr,
                     surface.width, surface.height,
                     VideoCore::PixelFormatAsString(surface.pixel_format),
                     surface.type == SurfaceType::Color ? "color" : "other",
                     validating_for != nullptr
                         ? fmt::format("validating {:#010x} {}x{} stride {} {} tiled {}",
                                       validating_for->addr, validating_for->width,
                                       validating_for->height, validating_for->stride,
                                       PixelFormatAsString(validating_for->pixel_format),
                                       validating_for->is_tiled)
                         : std::string{"the guest"});
        }
    }

    const SurfaceParams flush_info = surface.FromInterval(interval);
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    const auto staging = runtime.FindStaging(
        flush_info.width * flush_info.height * surface.GetInternalBytesPerPixel(), false);

    const BufferTextureCopy download = {
        .buffer_offset = staging.offset,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(flush_info),
        .texture_level = surface.LevelOf(flush_start),
    };
    surface.Download(download, staging);

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const auto download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);
    Common::PipelineStats::cache_downloads.fetch_add(1, std::memory_order_relaxed);
    Common::PipelineStats::cache_download_kb.fetch_add((flush_end - flush_start) / 1024,
                                                       std::memory_order_relaxed);
    EncodeTexture(flush_info, flush_start, flush_end, staging.mapped, download_dest,
                  runtime.NeedsConversion(surface));
}

template <class T>
void RasterizerCache<T>::DownloadFillSurface(Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    MemoryRef dest_ptr = memory.GetPhysicalRef(surface.addr);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const u32 start_offset = flush_start - surface.addr;
    const u32 end_offset =
        std::clamp(flush_end - surface.addr, 0u, static_cast<u32>(dest_ptr.GetSize()));
    const u32 coarse_start_offset = start_offset - (start_offset % surface.fill_size);
    const u32 backup_bytes = start_offset % surface.fill_size;

    std::array<u8, 4> backup_data;
    if (backup_bytes) {
        std::memcpy(backup_data.data(), &dest_ptr[coarse_start_offset], backup_bytes);
    }

    for (u32 offset = coarse_start_offset; offset < end_offset; offset += surface.fill_size) {
        std::memcpy(&dest_ptr[offset], &surface.fill_data[0],
                    std::min(surface.fill_size, end_offset - offset));
    }

    if (backup_bytes) {
        std::memcpy(&dest_ptr[coarse_start_offset], &backup_data[0], backup_bytes);
    }
}

template <class T>
bool RasterizerCache<T>::ValidateByReinterpretation(Surface& surface, SurfaceParams params,
                                                    const SurfaceInterval& interval) {
    SurfaceId reinterpret_id =
        FindMatch<MatchFlags::Reinterpret>(params, ScaleMatch::Ignore, interval);
    if (reinterpret_id) {
        Surface& src_surface = slot_surfaces[reinterpret_id];
        const SurfaceInterval copy_interval = src_surface.GetCopyableInterval(params);
        if (boost::icl::is_empty(copy_interval & interval)) {
            return false;
        }
        const u32 res_scale = src_surface.res_scale;
        if (res_scale > surface.res_scale) {
            surface.ScaleUp(res_scale);
        }
        const PAddr addr = boost::icl::lower(interval);
        const SurfaceParams copy_params = surface.FromInterval(copy_interval);
        const auto src_rect = src_surface.GetScaledSubRect(copy_params);
        const auto dst_rect = surface.GetScaledSubRect(copy_params);
        const TextureCopy reinterpret = {
            .src_level = src_surface.LevelOf(addr),
            .dst_level = surface.LevelOf(addr),
            .src_offset = {src_rect.left, src_rect.bottom},
            .dst_offset = {dst_rect.left, dst_rect.bottom},
            .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
        };
        return runtime.Reinterpret(src_surface, surface, reinterpret);
    }

    // No surfaces were found in the cache that had a matching bit-width.
    // Before entering the slow path, check if part of the interval is owned
    // by a gpu modified surface with a different stride than ours. This is indicative
    // of texture aliasing by the guest, which for the vast majority of cases we don't
    // need to validate.
    // TODO: While this works for the vast majority of cases, in Fire Emblem: Shadows of Valentia
    // the warping effect when running in dugeons relies on this stride reinterpretation.
    // In the future this transformation should be properly implemented with a GPU shader.
    const auto it = dirty_regions.find(interval);
    return it != dirty_regions.end() && it->second &&
           slot_surfaces[it->second].stride != surface.stride;
}

template <class T>
void RasterizerCache<T>::ClearAll(bool flush) {
    const auto flush_interval = PageMap::interval_type::right_open(0x0, 0xFFFFFFFF);
    // Force flush all surfaces from the cache
    if (flush) {
        FlushRegion(0x0, 0xFFFFFFFF);
    }
    // Unmark all of the marked pages
    for (auto& pair : RangeFromInterval(cached_pages, flush_interval)) {
        const auto interval = pair.first & flush_interval;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
    }

    // Remove the whole cache without really looking at it.
    cached_pages -= flush_interval;
    dirty_regions.clear();
    page_table.clear();
}

template <class T>
void RasterizerCache<T>::FlushRegion(PAddr addr, u32 size, SurfaceId flush_surface_id) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval flush_interval(addr, addr + size);
    SurfaceRegions flushed_intervals;

    for (const auto& [region, surface_id] : RangeFromInterval(dirty_regions, flush_interval)) {
        if (flush_surface_id && surface_id != flush_surface_id) {
            continue;
        }

        // Small sizes imply that this most likely comes from the cpu, flush the entire region
        // the point is to avoid thousands of small writes every frame if the cpu decides to
        // access that region, anything higher than 8 you're guaranteed it comes from a service
        const auto interval = size <= 8 ? region : region & flush_interval;
        Surface& surface = slot_surfaces[surface_id];
        ASSERT_MSG(surface.IsRegionValid(interval), "Region owner has invalid regions");

        const DebugScope scope{runtime, Common::Vec4f{0.f, 0.f, 0.f, 1.f},
                               "RasterizerCache::FlushRegion (from {:#x} to {:#x})",
                               interval.lower(), interval.upper()};

        SCOPE_EXIT({ flushed_intervals += interval; });
        if (surface.type == SurfaceType::Fill) {
            DownloadFillSurface(surface, interval);
            continue;
        }

        // Download each requested level of the surface.
        const u32 start_level = surface.LevelOf(interval.lower());
        const u32 end_level = surface.LevelOf(interval.upper());
        for (u32 level = start_level; level <= end_level; level++) {
            const auto download_interval = interval & surface.LevelInterval(level);
            if (boost::icl::is_empty(download_interval)) {
                continue;
            }
            DownloadSurface(surface, download_interval);
        }
    }

    // Reset dirty regions
    dirty_regions -= flushed_intervals;
}

template <class T>
void RasterizerCache<T>::FlushAll() {
    FlushRegion(0, 0xFFFFFFFF);
}

template <class T>
void RasterizerCache<T>::InvalidateRegionUnlessDirty(PAddr addr, u32 size) {
    if (size == 0) [[unlikely]] {
        return;
    }
    const SurfaceInterval interval(addr, addr + size);
    // Take the dirty regions out of the range; what remains is CPU data the guest may have
    // rewritten. A texture that was uploaded whole is kept, invalid, with the guest-flushed
    // flag: its next use hashes guest memory against what it was uploaded from and skips the
    // upload when nothing changed (Smash flushes the same texture memory every frame;
    // decoding and uploading 20 textures a frame for that measured 780 KiB a frame under
    // Vita3K). Everything else in the range is dropped as InvalidateRegion would.
    boost::container::small_vector<SurfaceInterval, 8> keep;
    for (const auto& [region, surface_id] : RangeFromInterval(dirty_regions, interval)) {
        keep.push_back(region & interval);
    }
    // A blanket flush: Smash flushes its whole FCRAM (0x20000000+0x3DAB000) every frame,
    // which says nothing about which bytes changed. Taken literally it queued a hash of
    // every texture in the cache each frame (11 a frame, 400 MB per 3 s on the Smash title
    // screen; the page-map gate it replaced ignored the flush outright). Under a blanket
    // flush a hashed texture is re-checked at most once every BlanketRecheckTicks resource
    // ticks: the cost spreads out, a stale texture shows for at most that long, and a
    // targeted flush still checks at once.
    constexpr u32 BlanketBytes = 16 * 1024 * 1024;
    constexpr u64 BlanketRecheckTicks = 30;
    const bool blanket = size >= BlanketBytes;
    const u64 tick = runtime.GetResourceTick();
    boost::container::small_vector<SurfaceId, 8> remove_surfaces;
    const auto invalidate_cpu = [&](PAddr start, u32 length) {
        const SurfaceInterval cpu_interval(start, start + length);
        ForEachSurfaceInRegion(start, length, [&](SurfaceId surface_id, Surface& surface) {
            if (blanket && surface.upload_hash != 0 &&
                tick - surface.last_hash_tick < BlanketRecheckTicks &&
                (surface.GetInterval() & cpu_interval) == surface.GetInterval()) {
                return;
            }
            surface.MarkInvalid(surface.GetInterval() & cpu_interval);
            if (!surface.IsFullyInvalid()) {
                return;
            }
            if (surface.upload_hash != 0) {
                surface.flags |= SurfaceFlagBits::GuestFlushed;
                return;
            }
            remove_surfaces.push_back(surface_id);
        });
    };
    PAddr cursor = addr;
    for (const SurfaceInterval& dirty : keep) {
        if (dirty.lower() > cursor) {
            invalidate_cpu(cursor, dirty.lower() - cursor);
        }
        cursor = std::max(cursor, dirty.upper());
    }
    if (cursor < addr + size) {
        invalidate_cpu(cursor, addr + size - cursor);
    }
    for (const SurfaceId surface_id : remove_surfaces) {
        if (True(slot_surfaces[surface_id].flags & SurfaceFlagBits::Registered)) {
            UnregisterSurface(surface_id);
        }
    }
}

template <class T>
void RasterizerCache<T>::InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner_id) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval invalid_interval(addr, addr + size);

    if (region_owner_id) {
        Surface& region_owner = slot_surfaces[region_owner_id];
        ASSERT(region_owner.type != SurfaceType::Texture);
        ASSERT(addr >= region_owner.addr && addr + size <= region_owner.end);
        ASSERT(region_owner.width == region_owner.stride);
        region_owner.MarkValid(invalid_interval);
        // The GPU wrote this surface: its content no longer matches any upload, so a guest
        // flush of the old upload bytes must not be mistaken for "unchanged".
        region_owner.upload_hash = 0;
    }

    boost::container::small_vector<SurfaceId, 4> remove_surfaces;
    ForEachSurfaceInRegion(addr, size, [&](SurfaceId surface_id, Surface& surface) {
        if (surface_id == region_owner_id) {
            return;
        }
        // A small invalidation with no owner is upstream's page-marking write hook: the guest
        // CPU is about to store into a cached page, so the surface is written back first (the
        // pixels the store does not cover must survive) and then dropped to make the pages
        // uncached again. That order only holds while the hook exists. With marking off - the
        // render thread cannot flip page attributes under the emulation thread - the only
        // caller of this shape is the guest's own GSP cache flush, which arrives *after* the
        // stores have landed, so writing the surface back puts dead pixels on top of live
        // data. Mario Kart 7 read a pointer out of a freed 64x64 texture's grave and
        // dereferenced null. Under no marking the range is invalidated and nothing is written.
        if (!region_owner_id && size <= 8 && memory.RasterizerCacheMarkingEnabled()) {
            FlushRegion(surface.addr, surface.size, surface_id);
            remove_surfaces.push_back(surface_id);
            return;
        }
        const auto interval = surface.GetInterval() & invalid_interval;
        surface.MarkInvalid(interval);
        if (!surface.IsFullyInvalid()) {
            return;
        }
        remove_surfaces.push_back(surface_id);
    });

    if (region_owner_id) {
        dirty_regions.set({invalid_interval, region_owner_id});
    } else {
        dirty_regions.erase(invalid_interval);
    }

    for (const SurfaceId surface_id : remove_surfaces) {
        UnregisterSurface(surface_id);
    }
}

template <class T>
SurfaceId RasterizerCache<T>::CreateSurface(const SurfaceParams& params,
                                            const SurfaceFlagBits& initial_flags) {
    const SurfaceId surface_id = [&] {
        const auto it = std::find_if(sentenced.begin(), sentenced.end(), [&](const auto& pair) {
            return slot_surfaces[pair.first] == params;
        });
        if (it == sentenced.end()) {
            return slot_surfaces.insert(runtime, params, initial_flags);
        }
        const SurfaceId surface_id = it->first;
        sentenced.erase(it);
        return surface_id;
    }();
    Surface& surface = slot_surfaces[surface_id];
    if (params.res_scale > surface.res_scale) {
        surface.ScaleUp(params.res_scale);
    }
    surface.MarkInvalid(surface.GetInterval());
    return surface_id;
}

template <class T>
void RasterizerCache<T>::RegisterSurface(SurfaceId surface_id) {
    Surface& surface = slot_surfaces[surface_id];
    ASSERT_MSG(False(surface.flags & SurfaceFlagBits::Registered),
               "Trying to register an already registered surface");

    surface.flags |= SurfaceFlagBits::Registered;
    UpdatePagesCachedCount(surface.addr, surface.size, 1);
    ForEachPage(surface.addr, surface.size,
                [this, surface_id](u64 page) { page_table[page].push_back(surface_id); });
}

template <class T>
void RasterizerCache<T>::UnregisterSurface(SurfaceId surface_id) {
    Surface& surface = slot_surfaces[surface_id];
    ASSERT_MSG(True(surface.flags & SurfaceFlagBits::Registered),
               "Trying to unregister an already unregistered surface");

    surface.flags &= ~SurfaceFlagBits::Registered;
    UpdatePagesCachedCount(surface.addr, surface.size, -1);
    ForEachPage(surface.addr, surface.size, [this, surface_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRA_PAGEBITS);
            return;
        }
        std::vector<SurfaceId>& surfaces = page_it.value();
        const auto vector_it = std::find(surfaces.begin(), surfaces.end(), surface_id);
        if (vector_it == surfaces.end()) {
            ASSERT_MSG(false, "Unregistering unregistered surface in page=0x{:x}",
                       page << CITRA_PAGEBITS);
            return;
        }
        surfaces.erase(vector_it);
    });

    if (surface.type != SurfaceType::Fill) {
        RemoveTextureCubeFace(surface_id);
        sentenced.emplace_back(surface_id, runtime.GetResourceTick());
        return;
    }

    slot_surfaces.erase(surface_id);
}

template <class T>
void RasterizerCache<T>::UnregisterAll() {
    FlushAll();
    for (auto& [page, surfaces] : page_table) {
        while (!surfaces.empty()) {
            UnregisterSurface(surfaces.back());
        }
    }
    runtime.Finish();
    RunGarbageCollector();
}

template <class T>
void RasterizerCache<T>::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 num_pages =
        ((addr + size - 1) >> Memory::CITRA_PAGE_BITS) - (addr >> Memory::CITRA_PAGE_BITS) + 1;
    const u32 page_start = addr >> Memory::CITRA_PAGE_BITS;
    const u32 page_end = page_start + num_pages;

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = PageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0) {
        cached_pages.add({pages_interval, delta});
    }

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta) {
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, true);
        } else if (delta < 0 && count == -delta) {
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
        } else {
            ASSERT(count >= 0);
        }
    }

    if (delta < 0) {
        cached_pages.add({pages_interval, delta});
    }
}

} // namespace VideoCore
