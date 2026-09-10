// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/texture/texture_decode.h"

namespace SwRenderer {

/**
 * Decode-on-first-use texture cache for the software renderer.
 *
 * The per-pixel path used to Morton-address and format-decode every texel on every fetch;
 * this cache decodes a texture once into a linear RGBA8 plane and sampling becomes an array
 * read. Everything runs on the render thread (ProcessTriangle and the present both live
 * there), so no locking.
 *
 * Invalidation: CPU-side writes arrive through RasterizerSoftware::InvalidateRegion (the
 * facade routes guest writes there). Fills and transfers invalidate the same way. Entries
 * outside any rendered framebuffer revalidate with a content hash at most once per presented
 * frame (NewFrame). Render targets are different: a game can render a scene and sample it as
 * a texture in the same frame (Smash's post pass does every fight frame), so the rasterizer
 * reports each buffer it renders into via NoteRenderedRange, and entries overlapping one
 * revalidate whenever rendering has happened since they were last checked.
 */
class SwTextureCache {
public:
    /// Returns the decoded RGBA8 plane (width * height, row-major, y=0 = first texture row
    /// in PICA layout) for the texture at addr, decoding or revalidating if needed.
    const u32* GetLinear(const u8* source, PAddr addr, const Pica::Texture::TextureInfo& info);

    /// Drops entries overlapping [addr, addr + size).
    void InvalidateRegion(PAddr addr, u32 size);

    /// The rasterizer calls this with the color/depth buffer of every triangle it draws.
    /// Textures overlapping a noted range lose the once-per-frame revalidation shortcut and
    /// re-hash on fetch until rendering stops touching them.
    void NoteRenderedRange(PAddr addr, u32 size);

    /// Evicted planes are parked, not freed: the rasterizer batches triangles per draw and
    /// their recorded plane pointers must stay valid until the batch has rasterized. The
    /// rasterizer calls this after each batch barrier.
    void CollectGarbage() {
        graveyard.clear();
    }

    /// Advances the revalidation epoch; called once per present.
    void NewFrame() {
        batch_epoch++;
        rendered_ranges.clear();
    }

    /// Whether [addr, addr+size) overlaps memory the rasterizer has rendered into this frame.
    /// A texture there samples the frame's own output, and the caller must flush the surface
    /// cache over the range before decoding: the pixels live host-side until then, and the
    /// decode reads guest memory.
    bool OverlapsRendered(PAddr addr, u32 size) const {
        for (const auto& range : rendered_ranges) {
            if (addr < range.addr + range.size && range.addr < addr + size) {
                return true;
            }
        }
        return false;
    }

    /// Advances the revalidation epoch for a new draw. Guest code executes natively here, so a
    /// write to texture memory traps nothing and this cache can only notice it by hashing. Tying
    /// revalidation to presents was not enough: when the renderer falls behind, one present spans
    /// many guest frames, and a plane validated in the first was reused unchecked through all of
    /// them while the guest kept rewriting it. A draw is the real boundary - hardware cannot see
    /// a texture change inside one either - so validate at most once per draw.
    void NewBatch() {
        batch_epoch++;
    }

private:
    struct Entry {
        std::vector<u32> rgba;
        u32 width = 0;
        u32 height = 0;
        u32 source_bytes = 0;
        u64 content_hash = 0;
        u64 validated_epoch = 0;
        u64 validated_render_gen = 0;
    };

    struct RenderedRange {
        PAddr addr = 0;
        u32 size = 0;
    };



    std::unordered_map<u64, Entry> entries;
    std::vector<RenderedRange> rendered_ranges;
    std::vector<std::vector<u32>> graveyard;
    u64 batch_epoch = 1;
    u64 render_gen = 1;
    std::size_t total_bytes = 0;
};

} // namespace SwRenderer
