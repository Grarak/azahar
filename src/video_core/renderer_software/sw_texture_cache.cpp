// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include "common/hash.h"
#include "video_core/renderer_software/sw_texture_cache.h"

namespace SwRenderer {

namespace {
u64 EntryKey(PAddr addr, const Pica::Texture::TextureInfo& info) {
    u64 key = addr;
    key ^= static_cast<u64>(info.width) << 32;
    key ^= static_cast<u64>(info.height) << 44;
    key ^= static_cast<u64>(info.format) << 56;
    return key;
}
} // namespace

const u32* SwTextureCache::GetLinear(const u8* source, PAddr addr,
                                     const Pica::Texture::TextureInfo& info) {
    if (source == nullptr || info.width == 0 || info.height == 0) {
        return nullptr;
    }

    const u64 key = EntryKey(addr, info);
    Entry& entry = entries[key];
    const u32 source_bytes = static_cast<u32>(info.stride * (info.height / 8));

    const bool sized = entry.width == info.width && entry.height == info.height &&
                       entry.source_bytes == source_bytes && !entry.rgba.empty();
    // A texture inside a rendered framebuffer changes between draws of the same frame, so the
    // once-per-frame shortcut only holds if nothing has been rendered since it was validated.
    const bool render_current =
        !OverlapsRendered(addr, source_bytes) || entry.validated_render_gen == render_gen;
    if (sized && entry.validated_epoch == batch_epoch && render_current) {
        return entry.rgba.data();
    }

    const u64 hash = Common::ComputeHash64(source, source_bytes);
    if (sized && hash == entry.content_hash) {
        entry.validated_epoch = batch_epoch;
        entry.validated_render_gen = render_gen;
        return entry.rgba.data();
    }

    // (Re)decode the whole texture into a linear RGBA8 plane.
    if (!sized) {
        total_bytes += static_cast<std::size_t>(info.width) * info.height * 4 -
                       entry.rgba.size() * sizeof(u32);
        entry.rgba.resize(static_cast<std::size_t>(info.width) * info.height);
        entry.width = info.width;
        entry.height = info.height;
        entry.source_bytes = source_bytes;
    }
    // Guest code runs natively, so it can rewrite this texture while the decode is reading it.
    // Decode, then re-hash the source: if it moved under us the plane is torn, so decode again.
    // Stamping either hash without this check marks a plane that matches no state as valid.
    u64 decoded_hash = hash;
    // PICA texture dimensions are multiples of the 8x8 tile; the per-texel fallback keeps any
    // degenerate configuration working.
    const bool whole_tiles = (info.width % 8) == 0 && (info.height % 8) == 0;
    const std::size_t tile_bytes = Pica::Texture::CalculateTileSize(info.format);
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (whole_tiles) {
            for (u32 ty = 0; ty < info.height / 8; ty++) {
                const u8* tile_row = source + static_cast<std::size_t>(ty) * info.stride;
                u32* dst_row = entry.rgba.data() + static_cast<std::size_t>(ty) * 8 * info.width;
                for (u32 tx = 0; tx < info.width / 8; tx++) {
                    Pica::Texture::DecodeTileRGBA8(dst_row + tx * 8, info.width,
                                                   tile_row + tx * tile_bytes, info);
                }
            }
#ifdef CITRA_TRACE_PROBES
            // AZAHAR_TILE_VERIFY=1: cross-check the tile decoder against the per-texel
            // reference on a sample of texels.
            static const bool tile_verify = std::getenv("AZAHAR_TILE_VERIFY") != nullptr;
            if (tile_verify) {
                static std::atomic<u64> checked{0}, bad{0};
                for (u32 y = 0; y < info.height; y += 3) {
                    for (u32 x = 0; x < info.width; x += 3) {
                        const auto c = Pica::Texture::LookupTexture(source, x, y, info);
                        const u32 want = static_cast<u32>(c.r()) |
                                         (static_cast<u32>(c.g()) << 8) |
                                         (static_cast<u32>(c.b()) << 16) |
                                         (static_cast<u32>(c.a()) << 24);
                        const u32 got = entry.rgba[static_cast<std::size_t>(y) * info.width + x];
                        checked.fetch_add(1, std::memory_order_relaxed);
                        if (want != got && bad.fetch_add(1, std::memory_order_relaxed) < 8) {
                            LOG_CRITICAL(HW_GPU,
                                         "TILE_VERIFY mismatch fmt={} {}x{} at ({},{}): "
                                         "want={:08x} got={:08x} [checked={}]",
                                         static_cast<u32>(info.format), info.width, info.height,
                                         x, y, want, got, checked.load());
                        }
                    }
                }
            }
#endif // CITRA_TRACE_PROBES
        } else {
            for (u32 y = 0; y < info.height; y++) {
                u32* row = entry.rgba.data() + static_cast<std::size_t>(y) * info.width;
                for (u32 x = 0; x < info.width; x++) {
                    const auto c = Pica::Texture::LookupTexture(source, x, y, info);
                    row[x] = static_cast<u32>(c.r()) | (static_cast<u32>(c.g()) << 8) |
                             (static_cast<u32>(c.b()) << 16) | (static_cast<u32>(c.a()) << 24);
                }
            }
        }
        const u64 after = Common::ComputeHash64(source, source_bytes);
        if (after == decoded_hash) {
            break;
        }
        decoded_hash = after;
    }
    entry.content_hash = decoded_hash;
    entry.validated_epoch = batch_epoch;
    entry.validated_render_gen = render_gen;

    // Past the budget, the planes validated longest ago go first - never one this batch has
    // fetched, whose pointer a batched triangle may still hold. Dropped planes go to the
    // graveyard until the next barrier for the same reason. The budget is what the platform
    // can spare: decoded planes are RGBA8, eight times an ETC1 texture, and on the Vita the
    // whole emulator has about 110 MB beside the guest's FCRAM (2026-09-01).
#ifdef __vita__
    constexpr std::size_t MAX_CACHE_BYTES = 16 * 1024 * 1024;
#else
    constexpr std::size_t MAX_CACHE_BYTES = 32 * 1024 * 1024;
#endif
    while (total_bytes > MAX_CACHE_BYTES) {
        auto victim = entries.end();
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->first == key || it->second.validated_epoch >= batch_epoch ||
                it->second.rgba.empty()) {
                continue;
            }
            if (victim == entries.end() ||
                it->second.validated_epoch < victim->second.validated_epoch) {
                victim = it;
            }
        }
        if (victim == entries.end()) {
            break; // everything left was fetched this batch; the budget yields, not the frame
        }
        total_bytes -= victim->second.rgba.size() * sizeof(u32);
        graveyard.push_back(std::move(victim->second.rgba));
        entries.erase(victim);
    }

    return entries[key].rgba.data();
}

void SwTextureCache::NoteRenderedRange(PAddr addr, u32 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    // Every draw advances the generation: a texture overlapping a render target must re-hash
    // on its next fetch no matter how recently it was validated.
    render_gen++;
    for (const auto& range : rendered_ranges) {
        if (range.addr == addr && range.size == size) {
            return;
        }
    }
    // A frame renders into a handful of buffers at most; if a title somehow exceeds the cap,
    // start the list over rather than growing without bound (costs correctness nothing - the
    // re-noted buffers return on the very next draw).
    if (rendered_ranges.size() >= 16) {
        rendered_ranges.clear();
    }
    rendered_ranges.push_back({addr, size});
}

void SwTextureCache::InvalidateRegion(PAddr addr, u32 size) {
    const PAddr end = addr + size;
    for (auto it = entries.begin(); it != entries.end();) {
        const PAddr entry_addr = static_cast<PAddr>(it->first & 0xFFFFFFFF);
        const PAddr entry_end = entry_addr + it->second.source_bytes;
        if (entry_addr < end && addr < entry_end) {
            total_bytes -= it->second.rgba.size() * sizeof(u32);
            if (!it->second.rgba.empty()) {
                graveyard.push_back(std::move(it->second.rgba));
            }
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace SwRenderer
