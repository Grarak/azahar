// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <deque>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <functional>
#include "core/hle/service/gsp/gsp_interrupt.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/pica_core.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/vertex_ring.h"

namespace VideoCore {

/**
 * Whether the three guest-memory writes the render thread performs on the emulation thread's
 * behalf ride its queue, or are waited for.
 *
 * Ordering the guest can observe is preserved either way - completion interrupts already ride
 * the same queue (PostInterruptAfterQueue) - so the wait looks removable, and removing all of
 * it does make the emulation thread stop waiting entirely. Measured on the Linux tier with
 * Super Mario 3D Land and the software renderer (2026-08-30):
 *
 *   both waited (as before)     23% speed, emulation thread waiting 1007 ms of every second
 *   async fills only            23%,       still 1007 ms - fills are not where the wait is
 *   async transfers only        70%,        735 ms
 *   both async                 100%,          0 ms
 *
 * and the last two do not work: the title stops progressing at its opening screen. Nothing is
 * corrupted - the guest is waiting, not crashing. A display transfer's PPF interrupt is queued
 * behind the transfer, so with the render thread saturated (it is: ~1000 ms of work per second
 * on this title) that interrupt arrives tens of milliseconds late, every frame. Waiting moves
 * from the emulation thread to the guest, which spins through emulated time instead - the run
 * reports 100% speed while the guest gets a quarter as far. The emulator is renderer-throughput
 * bound here, and synchronisation only decides who does the waiting.
 *
 * So what measured correct is what runs: fills, transfers and the invalidate write-back all
 * ride the render-thread queue, with the completion interrupt queued behind the write so the
 * guest is never told a fill finished before it has.
 */

/**
 * Whether a fill or transfer's completion is signalled on the engine's own modelled timing
 * rather than when the render thread reaches the queued write.
 *
 * Riding the queue is the honest ordering, and it is also what throttles the guest: the
 * interrupt sits behind a whole frame of queued draws, measured at 15 ms with transfers waited
 * and 26 ms with them queued, against a 16.7 ms frame. The guest then spends most of its
 * emulated time waiting - GSP command lists per second of emulated time fell from 94 to 24
 * while the run reported 100% speed.
 *
 * Signalling early is what the GL path has always done ("the completion interrupt runs on the
 * fill engine's modelled timing, never on the render thread's backlog"). The write itself still
 * rides the queue, so everything the *renderer* does stays ordered against it - including
 * presents, which are queued ops, so the display never shows a frame whose transfer has not
 * run. What is given up is the guest's own CPU reading or rewriting the region after being told
 * the engine finished, which is the hazard the deferred interrupts were introduced for.
 *
 * No snapshot can cover that here: a display transfer's source is usually a render target whose
 * content exists only in the render thread's surface cache, so copying guest memory at command
 * time would capture the wrong pixels.
 *
 * On by default. Measured on the Linux tier with Super Mario 3D Land and the software
 * renderer: 23% speed and 13.7 fps become 100% and 59.8, the emulation thread's waiting goes
 * from 1007 ms of every second to none, and the frames produced match the ordered path to two
 * pixels in 192000, stable across a 90 s soak. The GL path has always signalled this way -
 * DefersInterrupts is set only for the software renderer - so this makes the two agree.
 *
 * It was briefly demoted to opt-in on the strength of a flicker and a freeze seen while playing
 * on; both turned out to be something else - the flicker was the SDL window surface being
 * driven from the render thread, and the freeze was GL shader compilation in a different run -
 * and 200 consecutive frames captured past the opening dialogs with this on contain no torn
 * frame at all. The hazard named above is real but has not been observed: what would close it
 * is ordering the guest's own accesses against the queue, not delaying the signal again.
 *
 */
/**
 * Whether a fill or transfer runs on the emulation thread rather than being handed to the
 * render thread at all.
 *
 * The write is what the guest is waiting for, so the honest way to acknowledge it promptly is
 * to have performed it. This drains the queue first - everything already handed over lands, and
 * the render thread goes idle, which is what makes touching its surface cache from here safe -
 * then does the blit on this thread and signals immediately. The acknowledgement is accurate:
 * the bytes are in guest memory before the guest is told so, with none of the hazard early
 * completion trades away, and no interrupt sitting in a queue afterwards either.
 *
 * What it does not avoid is the drain. A display transfer usually sources the render target the
 * frame just drew, so the draws have to land first no matter who performs the transfer.
 *
 * Applies to both renderers. GL never had deferred interrupts at all - it has always enqueued
 * the transfer and signalled at once, which is the same acknowledgement-before-the-fact the
 * software path only recently adopted - so this is the first thing that makes GL's completions
 * true as well. The work itself stays on the render thread either way, since that is where the
 * GL context is current.
 */


class GpuThread;

/**
 * The rasterizer PicaCore drives when GPU execution lives on the render thread.
 *
 * PicaCore itself — command lists, vertex shading, primitive assembly, and with it the exact P3D
 * interrupt timing — runs on the emulation thread. This facade collects the resulting triangles
 * and, at each batch end, ships them together with a snapshot of the PICA registers and whatever
 * LUT state went dirty. The render thread applies the snapshot to a mirror PicaCore that the real
 * rasterizer was constructed against and replays the triangles into it. The emulation thread
 * never waits: a batch that arrives while the renderer is skipping frames applies its state and
 * drops its draws.
 *
 * The synchronous paths that remain are the ones that read the GL caches back (flushes) — those
 * round-trip through the render thread and wait.
 */
class ThreadedRasterizer final : public RasterizerInterface {
public:
    explicit ThreadedRasterizer(GpuThread& gpu_thread, Pica::PicaCore& emu_pica,
                                Pica::PicaCore& mirror_pica);

    /// The real rasterizer is recreated with the renderer; rebind after each recreation.
    void SetRealRasterizer(RasterizerInterface* real);

    /// Whole-frame skip, decided at each vblank on the emulation thread: while set, batches ship
    /// their register/LUT state but no triangles, and the frame's present is never queued. Frames
    /// that do render are therefore always complete.
    void SetSkipFrame(bool skip) {
        skip_frame = skip;
    }

    bool IsSkippingFrame() const {
        return skip_frame;
    }

    /// One guest frame has passed: what was drawn into during it counts as drawn into before.
    /// Called from the vblank on the emulation thread, which owns all of the batch state.
    void NoteFrameBoundary() {
        for (std::size_t i = 0; i < repeat_targets.size(); i++) {
            if (repeat_seen_now[i]) {
                repeat_seen_before[i] = true;
                repeat_seen_now[i] = false;
            }
        }
    }

    /// Conservative page map of guest physical ranges the GL may ever have written (render
    /// targets, fill and transfer destinations). Everything that writes GL-side originates on
    /// the emulation thread, so this needs no locking. A clear bit proves a flush is a no-op.
    void MarkGLWritten(PAddr addr, u32 size);
    bool TestGLWritten(PAddr addr, u32 size) const;

    /// Software-renderer mode: rendered output lands in guest RAM on the render thread, so
    /// completion interrupts must not fire until the queued work retires — the guest reuses
    /// memory the moment it believes the GPU is done. GL keeps its output in host caches and
    /// signals at ship time as before.
    void SetDeferredInterrupts(std::function<void(Service::GSP::InterruptId, u64)> poster) {
        deferred_interrupt_poster = std::move(poster);
    }

    /// Where an early completion goes instead of the queue: a zero-cycle timing event on the
    /// emulation thread, so delivery still happens on a clean stack.
    void SetEarlyInterruptPoster(std::function<void(Service::GSP::InterruptId, u64)> poster) {
        early_interrupt_poster = std::move(poster);
    }

    /// True when completion interrupts ride the render-thread queue (software renderer).
    bool DefersInterrupts() const override {
        return static_cast<bool>(deferred_interrupt_poster);
    }

    /// Enqueues an op that posts the interrupt when everything queued before it has executed.
    void PostInterruptAfterQueue(Service::GSP::InterruptId id, u64 delay_ns) override;

    /// Unsafe mode: even true flushes (ranges the GL really wrote) become fire-and-forget.
    /// Readback effects see stale pixels; nothing can crash. For measuring what correctness
    /// costs, and for users who prefer the speed.
    void SetUnsafeAsyncFlush(bool enabled) {
        unsafe_async_flush = enabled;
    }

    /// Whether addr is the base of a recently used top-screen render target (color or depth).
    /// Used to drop the GL side of clears aimed at skipped targets: a clear that runs while the
    /// repaint is skipped turns the cached frame black, which is exactly what frameskip must not
    /// present. Emulation thread only, like the rest of the batch state.
    bool IsTopTarget(u32 addr) const {
        for (const u32 target : top_targets) {
            if (target != 0 && target == addr) {
                return true;
            }
        }
        return false;
    }

    /// Whether addr lies inside a recently used render target rather than at its base. A title
    /// that draws both screens into one target and slices it sources the second screen from
    /// part way in - 160 rows in, for the one this was written against - so an exact-base test
    /// does not see that the pixels being read are the ones a skipped frame never drew.
    bool IsWithinTopTarget(u32 addr, u32 span) const {
        for (const u32 target : top_targets) {
            if (target != 0 && addr >= target && addr < target + span) {
                return true;
            }
        }
        return false;
    }

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override;
    void DrawTriangles() override;

    std::optional<DrawArenaTicket> BeginShippedDraw(u32 bytes, bool skippable,
                                                    u32 ring_bytes) override;
    void ShipDraw(Pica::DrawPayload&& payload, void* slot) override;
    bool ConsumesShippedDraws() override {
        return true;
    }

    /// Called at each vblank on the emulation thread: closes the frame's arena slot and opens a
    /// free one. When both slots are still owned by unfinished render-thread work, the frame
    /// runs without a slot — the vblank skip decision reads that as backpressure.
    void RotateArenaSlots();

    /// True when the last rotation found no free slot: the renderer still owns both arenas, so
    /// the coming frame's skippable draws should be skipped.
    bool ArenaBackpressure() const {
        return current_slot == nullptr;
    }

    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void InvalidateGuestFlushedRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;

    bool AccelerateDrawBatch(bool is_indexed) override {
        // Hardware vertex shaders would read guest attribute memory on the render thread;
        // everything goes through the software vertex pipeline and AddTriangle instead.
        return false;
    }

private:
    GpuThread& gpu_thread;
    Pica::PicaCore& emu_pica;
    Pica::PicaCore& mirror_pica;
    RasterizerInterface* real = nullptr;
    bool skip_frame = false;

    std::vector<Pica::OutputVertex> batch;

    /// Zero-copy mode, active when the real rasterizer exposes a mapped vertex ring: triangles
    /// are converted to HardwareVertex straight into GPU-visible memory at AddTriangle time and
    /// batches ship as ranges (one per batch unless the ring wrapped under it). Null means the
    /// copying batch path above is used.
    bool unsafe_async_flush = false;
    bool early_completion = true;
    std::function<void(Service::GSP::InterruptId, u64)> early_interrupt_poster;
    std::function<void(Service::GSP::InterruptId, u64)> deferred_interrupt_poster;

    VertexRing* ring = nullptr;
    std::vector<VertexRingRange> ranges;
    u64 batch_start_cursor = 0;
    u32 batch_start_offset = 0;
    /// One blocking-retire request in flight at a time; the flood of them was itself what
    /// starved the render thread when the ring first filled.
    std::atomic<bool> retire_kick_pending{false};

    void RingWaitForSpace(u64 bytes);

    /// Ring positions of hardware draws written by the emulation thread and not yet consumed
    /// by the render thread, in queue order (start, end); the front's start is the ring's
    /// consume_floor. Guarded by ring->mutex.
    std::deque<std::pair<u64, u64>> ring_inflight;
    void RingDrawConsumed(u64 ring_end);

    /// Draw-snapshot swapchain: two frame arenas the renderer consumes; chunked so shipped
    /// pointers stay stable while the frame keeps appending.
    struct ArenaSlot {
        std::vector<std::unique_ptr<u8[]>> chunks;
        std::size_t active_chunk = 0;
        u32 chunk_used = 0;
        std::atomic<u32> pending{0};
    };
    std::array<ArenaSlot, 2> arena_slots;
    ArenaSlot* current_slot = nullptr;
    /// Buffer for a draw that must ship while no slot is free (bottom screen, forced frames);
    /// handed to its shipment, which owns it from then on.
    std::unique_ptr<u8[]> emergency_arena;
    u32 emergency_arena_bytes = 0;
    /// Bytes of emergency arenas queued and not yet retired, and how many may be. Every draw
    /// past the two frame slots would otherwise take its own heap buffer for as long as the
    /// render thread is behind - on the Vita, with 110 MB for the whole emulator beside the
    /// guest's FCRAM, a queue a thousand draws deep ended in std::bad_alloc (2026-09-01).
    /// Over the budget the emulation thread waits for the render thread to retire work
    /// instead: the guest slows, the process lives.
    std::atomic<u32> emergency_bytes{0};
    static constexpr u32 EmergencyBudget = 8u << 20;

    static u8* SlotAlloc(ArenaSlot& slot, u32 bytes);

    /// Shader-unit state going to the mirror: uniforms per change (stored inline - they change
    /// every draw and must not cost an allocation), code arrays only on actual change (rare).
    /// Shader program and swizzle data, shipped as the used prefix only: the full arrays are
    /// 16 KB each, titles use a fraction, and code goes dirty on most shipments - copying the
    /// whole arrays four times per trip (collect and apply, both blocks) was a measured chunk
    /// of all memcpy traffic. The high-water sizes travel implicitly as the vector lengths.
    struct CodeSync {
        std::vector<u32> program;
        std::vector<u32> swizzle;
    };

    struct SetupSync {
        bool has_uniforms = false;
        /// Float window actually shipped, [f_lo, f_hi); the rest of `uniforms.f` is whatever an
        /// earlier trip through the pool left there and must not be read.
        u32 f_lo = 0;
        u32 f_hi = 0;
        Pica::Uniforms uniforms;
        std::unique_ptr<CodeSync> code;
    };

    /// Everything the render thread needs to reproduce this batch's PICA state. Instances cycle
    /// through a pool: one per queued op, reset on reuse with vectors keeping their grown
    /// capacity, so steady state allocates nothing per draw.
    struct Shipment {
        std::vector<Pica::OutputVertex> vertices;
        /// Zero-copy mode: ranges of the vertex ring to draw instead of `vertices`.
        std::vector<VertexRingRange> ranges;
        /// Offload mode: the draw itself, executed against the mirror.
        bool has_draw = false;
        Pica::DrawPayload draw{};
        std::unique_ptr<u8[]> owned_arena;
        u32 owned_arena_bytes = 0;
        SetupSync vs;
        SetupSync gs;
        bool has_default_attrs = false;
        Pica::AttributeBuffer default_attrs{};
        /// Primitive-assembler events to replay on the mirror before this batch's draws.
        u32 pa_reconfigure_events = 0;
        u32 pa_reset_events = 0;
        u32 pa_topology = 0;
        /// Sparse register delta: only what went dirty since the last shipment.
        std::vector<std::pair<u16, u32>> reg_delta;
        Pica::DirtyRegs dirty;
        // LUT blocks travel only when their dirty flags fired since the last shipment.
        /// Lighting LUTs: only the ones flagged dirty travel, a kilobyte each, into a buffer
        /// the shipment keeps across reuse (the whole block was 24.5 KB, copied twice per
        /// shipment, 1300 times a second on SM3DL).
        u32 lighting_dirty = 0;
        std::unique_ptr<std::array<std::array<u32, 256>, 24>> lighting_luts;
        std::unique_ptr<Pica::PicaCore::Fog> fog;
        std::unique_ptr<Pica::PicaCore::ProcTex> proctex;

        void Reset() {
            vertices.clear();
            ranges.clear();
            has_draw = false;
            owned_arena.reset();
            owned_arena_bytes = 0;
            vs.has_uniforms = false;
            vs.code.reset();
            gs.has_uniforms = false;
            gs.code.reset();
            has_default_attrs = false;
            pa_reconfigure_events = 0;
            pa_reset_events = 0;
            pa_topology = 0;
            reg_delta.clear();
            lighting_dirty = 0;
            fog.reset();
            proctex.reset();
        }
    };

    Shipment* TakeShipment();
    void ReturnShipment(Shipment* shipment);
    std::vector<std::unique_ptr<Shipment>> shipment_pool;
    std::vector<Shipment*> shipment_free;
    std::mutex shipment_pool_mutex;

    static void ApplySetup(Pica::ShaderSetup& setup, const SetupSync& sync);
    static void ApplyShipment(Pica::PicaCore& mirror, const Shipment& shipment);
    void CollectState(Shipment& shipment);

    /// One bit per 4 KiB page over the window the PICA can actually target
    /// ([0x18000000, 0x28000000): VRAM through N3DS FCRAM) - 8 KiB of bitmap rather than 8 MiB
    /// over the whole physical space. Addresses outside are treated conservatively.
    static constexpr u32 GL_WRITTEN_PAGE_BITS = 12;
    static constexpr PAddr GL_WRITTEN_BASE = 0x18000000;
    static constexpr PAddr GL_WRITTEN_END = 0x28000000;
    std::array<u64, ((GL_WRITTEN_END - GL_WRITTEN_BASE) >> (GL_WRITTEN_PAGE_BITS + 6))>
        gl_written_pages{};
    /// Framebuffer registers rarely change between draws; identical ranges skip re-marking.
    PAddr last_marked_addr = 0;
    u32 last_marked_size = 0;

    /// Ring of top-screen render target base addresses seen at recent batch ends.
    std::array<u32, 16> top_targets{};
    std::size_t top_target_next = 0;

    void RecordTopTarget(u32 addr) {
        if (addr == 0 || IsTopTarget(addr)) {
            return;
        }
        top_targets[top_target_next] = addr;
        top_target_next = (top_target_next + 1) % top_targets.size();
    }

    /// Ring of every render target drawn into, whatever its size, and how the skip decides
    /// about the small off-screen ones. A title that composes its screen through a chain of
    /// small targets puts nearly all of its work there - Smash's victory screen draws into
    /// 256x128 and 128x64 targets - and the height test above, which exists so that a HUD
    /// drawn once into the bottom screen is never dropped, excludes all of it. What actually
    /// separates the two cases is repetition, not size: a target drawn into a second time is
    /// part of the per-frame composite and the next rendered frame will draw it again, while
    /// content drawn exactly once is seen here exactly once and stays.
    std::array<u32, 32> repeat_targets{};
    /// Drawn into since the last frame boundary, and drawn into during an earlier frame. The
    /// two generations are what make the test "a previous frame drew this" rather than "this
    /// has been drawn a few times": a single frame issues many batches into one target, so
    /// counting sightings alone would call a HUD repeated before its only frame was over.
    std::array<bool, 32> repeat_seen_now{};
    std::array<bool, 32> repeat_seen_before{};
    std::size_t repeat_target_next = 0;

    bool IsRepeatTarget(u32 addr) const {
        for (std::size_t i = 0; i < repeat_targets.size(); i++) {
            if (repeat_targets[i] != 0 && repeat_targets[i] == addr) {
                return repeat_seen_before[i];
            }
        }
        return false;
    }

    void RecordDrawTarget(u32 addr) {
        if (addr == 0) {
            return;
        }
        for (std::size_t i = 0; i < repeat_targets.size(); i++) {
            if (repeat_targets[i] == addr) {
                repeat_seen_now[i] = true;
                return;
            }
        }
        repeat_targets[repeat_target_next] = addr;
        repeat_seen_now[repeat_target_next] = true;
        repeat_seen_before[repeat_target_next] = false;
        repeat_target_next = (repeat_target_next + 1) % repeat_targets.size();
    }

    /// Vertex buffers cycle between the threads instead of being reallocated: the emulation side
    /// takes one from here, swaps the batch into it, and the render thread returns it — with its
    /// grown capacity — once the draws are replayed.
    std::vector<std::vector<Pica::OutputVertex>> buffer_pool;
    std::mutex pool_mutex;

    std::vector<Pica::OutputVertex> TakeBuffer();
    void ReturnBuffer(std::vector<Pica::OutputVertex>&& buffer);
};

} // namespace VideoCore
