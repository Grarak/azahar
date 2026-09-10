// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "core/hle/service/gsp/gsp_interrupt.h"

#include <atomic>
#include <functional>
#include <optional>
#include "common/common_types.h"
#include "video_core/pica/draw_payload.h"
#include "video_core/vertex_ring.h"

namespace Pica {
struct OutputVertex;
}

namespace Pica {
class PicaCore;
struct DisplayTransferConfig;
struct MemoryFillConfig;
} // namespace Pica

namespace VideoCore {

enum class LoadCallbackStage {
    Prepare,
    Preload,
    Decompile,
    Build,
    Complete,
};
using DiskResourceLoadCallback =
    std::function<void(LoadCallbackStage, std::size_t, std::size_t, const std::string&)>;

class RasterizerInterface {
public:
    virtual ~RasterizerInterface() = default;

    /// Queues the primitive formed by the given vertices for rendering
    virtual void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                             const Pica::OutputVertex& v2) = 0;

    /// Draw the current batch of triangles
    virtual void DrawTriangles() = 0;

    /// Notify rasterizer that all caches should be flushed to 3DS memory
    virtual void FlushAll() = 0;

    /// Notify rasterizer that any caches of the specified region should be flushed to 3DS memory
    virtual void FlushRegion(PAddr addr, u32 size) = 0;

    /// Notify rasterizer that any caches of the specified region should be invalidated
    virtual void InvalidateRegion(PAddr addr, u32 size) = 0;

    /// The guest cleaned its CPU data cache over this range, so bytes it wrote with the CPU are
    /// what the GPU must read from now on. A renderer whose rendered output stays host-side
    /// drops cached copies of everything in the range except the pixels that exist nowhere but
    /// in its caches (the real GPU would have written those to RAM long ago); the default is
    /// the plain invalidate.
    virtual void InvalidateGuestFlushedRegion(PAddr addr, u32 size) {
        InvalidateRegion(addr, size);
    }

    /// The software blitter has finished writing every byte of [addr, addr+size) in guest
    /// memory, on the render thread. Cached copies are stale; the default drops them. The
    /// software rasterizer overrides this to absorb the bytes into its live surfaces instead:
    /// dropping a surface forces a full reload of content that is sitting right there, and the
    /// reload traffic was the single largest memcpy consumer in the whole process.
    virtual void NoteGuestWrite(PAddr addr, u32 size) {
        InvalidateRegion(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to 3DS memory
    /// and invalidated
    virtual void FlushAndInvalidateRegion(PAddr addr, u32 size) = 0;

    /// Removes as much state as possible from the rasterizer in preparation for a save/load state
    virtual void ClearAll(bool flush) = 0;

    /// Attempt to use a faster method to perform a display transfer with is_texture_copy = 0
    virtual bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig&) {
        return false;
    }

    /// Attempt to use a faster method to perform a display transfer with is_texture_copy = 1
    virtual bool AccelerateTextureCopy(const Pica::DisplayTransferConfig&) {
        return false;
    }

    /// Attempt to use a faster method to fill a region
    virtual bool AccelerateFill(const Pica::MemoryFillConfig&) {
        return false;
    }

    /// Attempt to draw using hardware shaders
    /// The persistently mapped vertex ring, when this rasterizer can draw straight out of one;
    /// null otherwise. The producer writes HardwareVertex-layout triangles through it and draws
    /// them with DrawRingRange, skipping every intermediate vertex copy.
    /// Software-renderer contract: when true, GPU completion interrupts must be posted via
    /// PostInterruptAfterQueue so they fire only after previously queued render-thread work
    /// (whose output lands in guest RAM) has retired.
    virtual bool DefersInterrupts() const {
        return false;
    }

    virtual void PostInterruptAfterQueue(Service::GSP::InterruptId id, u64 delay_ns) {}

    virtual VertexRing* GetVertexRing() {
        return nullptr;
    }

    /// Draws `count` vertices starting at vertex index `first` of the ring buffer, then fences
    /// the ring up to `ring_end_pos` so Retire can eventually release the space.
    virtual void DrawRingRange([[maybe_unused]] u32 first, [[maybe_unused]] u32 count,
                               [[maybe_unused]] u64 ring_end_pos) {}

    /// Blocks on the oldest outstanding ring fence and retires it. The producer requests this
    /// through the work queue when the ring is full and polling has not caught up.
    virtual void RingRetireBlocking() {}

    /// Opens a shipped draw. nullopt: this rasterizer does not consume shipped draws — shade
    /// locally. A ticket with null data: the draw is skipped entirely (frameskip decided before
    /// any snapshot copy is paid). Otherwise `data` is `bytes` of arena to fill and ship.
    struct DrawArenaTicket {
        u8* data = nullptr;
        void* slot = nullptr;
        /// `data` is inside the vertex ring (ring layout, read in place by the draw).
        bool in_ring = false;
        u32 ring_offset = 0;
        u64 ring_end = 0;
    };
    /// `ring_bytes` is the draw's size in the ring layout; nonzero asks for ring placement,
    /// which the facade grants when the real rasterizer reads ring draws and the ring has the
    /// space right now, and otherwise falls back to an arena of `bytes`.
    virtual std::optional<DrawArenaTicket> BeginShippedDraw([[maybe_unused]] u32 bytes,
                                                            [[maybe_unused]] bool skippable,
                                                            [[maybe_unused]] u32 ring_bytes = 0) {
        return std::nullopt;
    }

    /// Whether AccelerateShippedDraw reads a ring-layout payload in place.
    virtual bool WantsRingDraws() {
        return false;
    }

    /// Ships a draw opened by BeginShippedDraw (or an immediate-mode draw with no arena).
    virtual void ShipDraw([[maybe_unused]] Pica::DrawPayload&& payload,
                          [[maybe_unused]] void* slot) {}

    /// Whether ShipDraw consumes draws at all; immediate-mode draws check this directly since
    /// they carry no arena.
    virtual bool ConsumesShippedDraws() {
        return false;
    }

    /// Executes a hw-flagged shipped draw through the GLSL vertex-shader path, reading vertex
    /// and index data from the payload's arena. False = caller runs the software mirror path.
    virtual bool AccelerateShippedDraw([[maybe_unused]] const Pica::DrawPayload& payload) {
        return false;
    }

    /// Fences everything drawn from the ring so far and retires whatever already completed.
    /// Called at presents — the natural flush points — so fence traffic stays per-frame, not
    /// per-batch.
    virtual void RingFenceTick() {}

    virtual bool AccelerateDrawBatch([[maybe_unused]] bool is_indexed) {
        return false;
    }

    virtual void LoadDefaultDiskResources(
        [[maybe_unused]] const std::atomic_bool& stop_loading,
        [[maybe_unused]] const DiskResourceLoadCallback& callback) {}

    virtual void SwitchDiskResources([[maybe_unused]] u64 title_id) {}

    static void SetSwitchDiskResourcesCallback(const DiskResourceLoadCallback& callback) {
        switch_disk_resources_callback = callback;
    }

    void SetAccurateMul(bool accurate_mul_) {
        accurate_mul = accurate_mul_;
    }

protected:
    bool accurate_mul = false;

    // Rasterizer gets destroyed on reboot, so make the callback
    // static until a better solution is found.
    static DiskResourceLoadCallback switch_disk_resources_callback;
};
} // namespace VideoCore
