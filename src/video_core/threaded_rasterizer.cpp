// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/pipeline_stats.h"
#include <cstdlib>
#include <chrono>
#include <memory>
#include <new>
#include "video_core/gpu_thread.h"
#include "video_core/pica/pica_core.h"
#include <cstring>
#include "video_core/guest_watch.h"
#include "video_core/threaded_rasterizer.h"

namespace VideoCore {

void ThreadedRasterizer::ApplySetup(Pica::ShaderSetup& setup, const SetupSync& sync) {
    if (sync.has_uniforms) {
        if (sync.f_lo < sync.f_hi) {
            std::memcpy(&setup.uniforms.f[sync.f_lo], &sync.uniforms.f[sync.f_lo],
                        (sync.f_hi - sync.f_lo) * sizeof(setup.uniforms.f[0]));
        }
        setup.uniforms.b = sync.uniforms.b;
        setup.uniforms.i = sync.uniforms.i;
        setup.uniforms_dirty = true;
        // The mirror's own window: what a renderer keeping a converted copy of the bank has
        // to convert again.
        setup.sync_f_lo = std::min(setup.sync_f_lo, sync.f_lo);
        setup.sync_f_hi = std::max(setup.sync_f_hi, sync.f_hi);
    }
    if (sync.code) {
        // The word-compare in the range updaters doubles as a filter: an upload identical to
        // what the mirror already holds leaves the hashes valid, where the old whole-array
        // assignment forced a 16 KB rehash per shipment.
        setup.UpdateProgramCodeRange(0, sync.code->program.data(),
                                     static_cast<u32>(sync.code->program.size()));
        setup.UpdateSwizzleDataRange(0, sync.code->swizzle.data(),
                                     static_cast<u32>(sync.code->swizzle.size()));
    }
}

/// Applies a shipment's state to the render thread's mirror.
void ThreadedRasterizer::ApplyShipment(Pica::PicaCore& mirror, const Shipment& shipment) {
    for (const auto& [index, value] : shipment.reg_delta) {
        mirror.regs.internal.reg_array[index] = value;
    }
    ApplySetup(mirror.vs_setup, shipment.vs);
    ApplySetup(mirror.gs_setup, shipment.gs);
    if (shipment.pa_reconfigure_events != 0) {
        mirror.GetPrimitiveAssembler().Reconfigure(
            static_cast<Pica::PipelineRegs::TriangleTopology>(shipment.pa_topology));
    }
    if (shipment.pa_reset_events != 0) {
        mirror.GetPrimitiveAssembler().Reset();
    }
    if (shipment.has_default_attrs) {
        mirror.input_default_attributes = shipment.default_attrs;
    }
    mirror.dirty_regs.Or(shipment.dirty);
    if (shipment.lighting_dirty != 0) {
        static_assert(sizeof(mirror.lighting.luts[0]) == sizeof((*shipment.lighting_luts)[0]));
        u32 bits = shipment.lighting_dirty;
        while (bits != 0) {
            const u32 lut = static_cast<u32>(__builtin_ctz(bits));
            bits &= bits - 1;
            std::memcpy(&mirror.lighting.luts[lut], &(*shipment.lighting_luts)[lut],
                        sizeof(mirror.lighting.luts[lut]));
        }
        mirror.lighting.lut_dirty |= shipment.lighting_dirty;
    }
    if (shipment.fog) {
        mirror.fog = *shipment.fog;
        mirror.fog.lut_dirty = true;
    }
    if (shipment.proctex) {
        const u8 dirty = shipment.proctex->table_dirty | mirror.proctex.table_dirty;
        mirror.proctex = *shipment.proctex;
        mirror.proctex.table_dirty = dirty;
    }
}

ThreadedRasterizer::ThreadedRasterizer(GpuThread& gpu_thread_, Pica::PicaCore& emu_pica_,
                                       Pica::PicaCore& mirror_pica_)
    : gpu_thread{gpu_thread_}, emu_pica{emu_pica_}, mirror_pica{mirror_pica_} {}

void ThreadedRasterizer::SetRealRasterizer(RasterizerInterface* real_) {
    real = real_;
    // A recreated rasterizer brings a fresh ring (new mapping, cursor at zero); rebind and reset
    // the batch bookkeeping. This runs with the render thread quiescent.
    ring = real ? real->GetVertexRing() : nullptr;
    ranges.clear();
    batch_start_cursor = ring ? ring->cursor : 0;
    batch_start_offset = ring ? ring->offset : 0;
}

void ThreadedRasterizer::RingDrawConsumed(u64 ring_end) {
    // Render thread, after the draw that read this allocation was submitted (or dropped): the
    // floor moves to the next outstanding hardware draw, or away entirely.
    std::scoped_lock lock{ring->mutex};
    while (!ring_inflight.empty() && ring_inflight.front().second <= ring_end) {
        ring_inflight.pop_front();
    }
    ring->consume_floor.store(ring_inflight.empty() ? ~0ull : ring_inflight.front().first,
                              std::memory_order_release);
}

void ThreadedRasterizer::RingWaitForSpace(u64 bytes) {
    std::unique_lock lock{ring->mutex};
    while (ring->TryAllocWouldFail(static_cast<u32>(bytes))) {
        // The render thread only learns that space ran out from us: hand it a blocking retire,
        // but never more than one at a time — a queue full of blocking waits is a starved render
        // thread. Enqueue is safe here (it does not take ring->mutex), and in inline mode it
        // simply runs the retire on this thread with the context current.
        if (!retire_kick_pending.exchange(true, std::memory_order_acq_rel)) {
            gpu_thread.Enqueue({.run = [this] {
                real->RingRetireBlocking();
                retire_kick_pending.store(false, std::memory_order_release);
            }});
        }
        ring->retired_cv.wait_for(lock, std::chrono::milliseconds(2));
    }
}

std::vector<Pica::OutputVertex> ThreadedRasterizer::TakeBuffer() {
    std::scoped_lock lock{pool_mutex};
    if (buffer_pool.empty()) {
        return {};
    }
    auto buffer = std::move(buffer_pool.back());
    buffer_pool.pop_back();
    return buffer;
}

void ThreadedRasterizer::ReturnBuffer(std::vector<Pica::OutputVertex>&& buffer) {
    buffer.clear();
    std::scoped_lock lock{pool_mutex};
    if (buffer_pool.size() < 64) {
        buffer_pool.push_back(std::move(buffer));
    }
}

namespace {
/// The GL reads the ring in OutputVertex layout, so pushing a vertex is one 96-byte copy plus
/// the per-triangle quaternion sign fix the HardwareVertex conversion used to do: opposite
/// quaternions (negative dot against the provoking vertex) are negated so interpolation takes
/// the short arc. f24 stores a plain float, so the dot works on the raw fields.
float QuatDot(const Common::Vec4<Pica::f24>& qa, const Common::Vec4<Pica::f24>& qb) {
    return qa.x.ToFloat32() * qb.x.ToFloat32() + qa.y.ToFloat32() * qb.y.ToFloat32() +
           qa.z.ToFloat32() * qb.z.ToFloat32() + qa.w.ToFloat32() * qb.w.ToFloat32();
}

void NegateQuat(Pica::OutputVertex& v) {
    v.quat.x = Pica::f24::FromFloat32(-v.quat.x.ToFloat32());
    v.quat.y = Pica::f24::FromFloat32(-v.quat.y.ToFloat32());
    v.quat.z = Pica::f24::FromFloat32(-v.quat.z.ToFloat32());
    v.quat.w = Pica::f24::FromFloat32(-v.quat.w.ToFloat32());
}
} // Anonymous namespace

void ThreadedRasterizer::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                     const Pica::OutputVertex& v2) {
    if (ring) {
        // Zero-copy: the triangle goes straight into the mapped GL vertex buffer in guest
        // layout; no other copy of it will ever exist.
        constexpr u32 tri_bytes = 3 * sizeof(Pica::OutputVertex);
        // Under the ring's mutex: the render thread allocates from the same ring for the
        // mirror's software triangles.
        u32 off;
        {
            std::scoped_lock lock{ring->mutex};
            off = ring->TryAlloc(tri_bytes);
        }
        if (off == VertexRing::FULL) {
            RingWaitForSpace(tri_bytes);
            std::scoped_lock lock{ring->mutex};
            off = ring->TryAlloc(tri_bytes);
        }
        auto* const dst = reinterpret_cast<Pica::OutputVertex*>(ring->base + off);
        std::memcpy(dst + 0, &v0, sizeof(Pica::OutputVertex));
        std::memcpy(dst + 1, &v1, sizeof(Pica::OutputVertex));
        std::memcpy(dst + 2, &v2, sizeof(Pica::OutputVertex));
        if (QuatDot(v0.quat, v1.quat) < 0.0f) {
            NegateQuat(dst[1]);
        }
        if (QuatDot(v0.quat, v2.quat) < 0.0f) {
            NegateQuat(dst[2]);
        }

        const u32 first = off / sizeof(Pica::OutputVertex);
        if (ranges.empty() || ranges.back().first + ranges.back().count != first) {
            ranges.push_back({first, 0, 0});
        }
        ranges.back().count += 3;
        ranges.back().end_pos = ring->cursor;
        return;
    }

    batch.push_back(v0);
    batch.push_back(v1);
    batch.push_back(v2);
}

void ThreadedRasterizer::CollectState(Shipment& shipment) {
    // Ship only the registers that changed, discovered through the dirty bits; the LUTs travel
    // only when PicaCore flagged them since the last batch.
    shipment.dirty = emu_pica.dirty_regs;
    for (u32 word = 0; word < 12; word++) {
        u64 bits = shipment.dirty.qwords[word];
        while (bits != 0) {
            const u32 bit = static_cast<u32>(__builtin_ctzll(bits));
            bits &= bits - 1;
            const u32 index = word * 64 + bit;
            if (index < Pica::RegsInternal::NUM_REGS) {
                shipment.reg_delta.emplace_back(static_cast<u16>(index),
                                                emu_pica.regs.internal.reg_array[index]);
            }
        }
    }
    emu_pica.dirty_regs.Clear();

    if (emu_pica.lighting.lut_dirty != 0) {
        if (!shipment.lighting_luts) {
            shipment.lighting_luts = std::make_unique<std::array<std::array<u32, 256>, 24>>();
        }
        u32 bits = emu_pica.lighting.lut_dirty & Pica::PicaCore::Lighting::LutAllDirty;
        shipment.lighting_dirty = bits;
        while (bits != 0) {
            const u32 lut = static_cast<u32>(__builtin_ctz(bits));
            bits &= bits - 1;
            std::memcpy(&(*shipment.lighting_luts)[lut], &emu_pica.lighting.luts[lut],
                        sizeof(emu_pica.lighting.luts[lut]));
        }
        emu_pica.lighting.lut_dirty = 0;
    }
    if (emu_pica.fog.lut_dirty) {
        shipment.fog = std::make_unique<Pica::PicaCore::Fog>(emu_pica.fog);
        emu_pica.fog.lut_dirty = false;
    }
    if (emu_pica.proctex.table_dirty != 0) {
        shipment.proctex = std::make_unique<Pica::PicaCore::ProcTex>(emu_pica.proctex);
        emu_pica.proctex.table_dirty = 0;
    }

    const auto collect_setup = [](Pica::ShaderSetup& setup, SetupSync& sync) {
        if (setup.uniforms_sync_dirty) {
            sync.has_uniforms = true;
            // Only the float window written since the last collect travels; bools and ints are
            // 32 bytes and go whole. The full-struct copy here was a fixed 1.5 KB per shipment.
            sync.f_lo = std::min<u32>(setup.sync_f_lo, 96);
            sync.f_hi = std::min<u32>(setup.sync_f_hi, 96);
            if (sync.f_lo < sync.f_hi) {
                std::memcpy(&sync.uniforms.f[sync.f_lo], &setup.uniforms.f[sync.f_lo],
                            (sync.f_hi - sync.f_lo) * sizeof(setup.uniforms.f[0]));
            }
            sync.uniforms.b = setup.uniforms.b;
            sync.uniforms.i = setup.uniforms.i;
            setup.uniforms_sync_dirty = false;
            setup.sync_f_lo = 96;
            setup.sync_f_hi = 0;
        }
        if (setup.code_sync_dirty) {
            sync.code = std::make_unique<CodeSync>();
            const auto& program = setup.GetProgramCode();
            const auto& swizzle = setup.GetSwizzleData();
            sync.code->program.assign(program.begin(),
                                      program.begin() + setup.GetBiggestProgramSize());
            sync.code->swizzle.assign(swizzle.begin(),
                                      swizzle.begin() + setup.GetBiggestSwizzleSize());
            setup.code_sync_dirty = false;
        }
    };
    collect_setup(emu_pica.vs_setup, shipment.vs);
    collect_setup(emu_pica.gs_setup, shipment.gs);
    shipment.pa_reconfigure_events = emu_pica.pa_reconfigure_events;
    shipment.pa_reset_events = emu_pica.pa_reset_events;
    shipment.pa_topology =
        static_cast<u32>(emu_pica.regs.internal.pipeline.triangle_topology.Value());
    emu_pica.pa_reconfigure_events = 0;
    emu_pica.pa_reset_events = 0;
    if (emu_pica.default_attributes_sync_dirty) {
        shipment.has_default_attrs = true;
        shipment.default_attrs = emu_pica.input_default_attributes;
        emu_pica.default_attributes_sync_dirty = false;
    }
}

u8* ThreadedRasterizer::SlotAlloc(ArenaSlot& slot, u32 bytes) {
    constexpr u32 CHUNK_SIZE = 512 * 1024;
    const u32 need = std::max(bytes, 1u);
    if (slot.active_chunk >= slot.chunks.size() ||
        slot.chunk_used + need > (slot.chunks[slot.active_chunk] ? CHUNK_SIZE : 0)) {
        // Advance to the next chunk that fits; oversized draws get a dedicated chunk.
        if (slot.active_chunk < slot.chunks.size() && slot.chunk_used != 0) {
            slot.active_chunk++;
        }
        while (slot.active_chunk >= slot.chunks.size()) {
            slot.chunks.push_back(std::make_unique<u8[]>(std::max(need, CHUNK_SIZE)));
        }
        slot.chunk_used = 0;
        if (need > CHUNK_SIZE) {
            slot.chunks[slot.active_chunk] = std::make_unique<u8[]>(need);
        }
    }
    u8* const ptr = slot.chunks[slot.active_chunk].get() + slot.chunk_used;
    slot.chunk_used += need;
    return ptr;
}

void ThreadedRasterizer::RotateArenaSlots() {
    current_slot = nullptr;
    for (auto& slot : arena_slots) {
        if (slot.pending.load(std::memory_order_acquire) == 0) {
            slot.active_chunk = 0;
            slot.chunk_used = 0;
            current_slot = &slot;
            break;
        }
    }
}

std::optional<RasterizerInterface::DrawArenaTicket> ThreadedRasterizer::BeginShippedDraw(
    u32 bytes, bool skippable, u32 ring_bytes) {
    {
        const auto& fb = emu_pica.regs.internal.framebuffer.framebuffer;
        const u32 fb_pixels = (fb.width + 1) * (fb.height + 1);
        MarkGLWritten(fb.GetColorBufferPhysicalAddress(), fb_pixels * 4);
        MarkGLWritten(fb.GetDepthBufferPhysicalAddress(), fb_pixels * 4);
        // The caller's flag is the top screen's own framebuffer, by height. A target an
        // earlier frame already drew into is skippable as well - see repeat_targets - and the
        // recording has to happen for every draw, whatever the skip state, or a target is
        // never learned. This is the gate the hardware renderer's draws pass through;
        // DrawTriangles below is the software one, and the two must agree.
        const u32 color_target = fb.GetColorBufferPhysicalAddress();
        const bool repeat_target = IsRepeatTarget(color_target);
        RecordDrawTarget(color_target);
        if (skippable) {
            RecordTopTarget(fb.GetColorBufferPhysicalAddress());
            RecordTopTarget(fb.GetDepthBufferPhysicalAddress());
        }
        if (skip_frame && (skippable || repeat_target)) {
            return DrawArenaTicket{nullptr, nullptr};
        }
    }
    if (ring_bytes != 0 && ring != nullptr && real != nullptr && real->WantsRingDraws() &&
        ring_bytes <= ring->size / 2) {
        // Straight into the GPU-visible ring, once, in the layout the draw reads. Only when
        // the ring has the room now: waiting here would be the emulation thread waiting on
        // the renderer, so a full ring takes the arena and the render thread's copy instead.
        std::scoped_lock lock{ring->mutex};
        const u32 off = ring->TryAlloc(ring_bytes);
        if (off != VertexRing::FULL) {
            const u64 end = ring->cursor;
            if (ring_inflight.empty()) {
                ring->consume_floor.store(end - ring_bytes, std::memory_order_release);
            }
            ring_inflight.emplace_back(end - ring_bytes, end);
            return DrawArenaTicket{ring->base + off, nullptr, true, off, end};
        }
    }
    if (current_slot == nullptr) {
        // No free frame arena (or none was ever opened): the draw still ships — frames that
        // render must be complete — through a buffer the shipment will own. Bounded: past the
        // budget, wait for the render thread to retire something and look for a freed slot.
        while (emergency_bytes.load(std::memory_order_relaxed) + bytes > EmergencyBudget &&
               gpu_thread.HasWorker()) {
            gpu_thread.WaitForRetire();
            RotateArenaSlots();
            if (current_slot != nullptr) {
                return DrawArenaTicket{SlotAlloc(*current_slot, bytes), current_slot};
            }
        }
        emergency_arena = std::make_unique<u8[]>(std::max(bytes, 1u));
        emergency_arena_bytes = std::max(bytes, 1u);
        emergency_bytes.fetch_add(emergency_arena_bytes, std::memory_order_relaxed);
        return DrawArenaTicket{emergency_arena.get(), nullptr};
    }
    return DrawArenaTicket{SlotAlloc(*current_slot, bytes), current_slot};
}

void ThreadedRasterizer::PostInterruptAfterQueue(Service::GSP::InterruptId id, u64 delay_ns) {
    if (early_completion && early_interrupt_poster) {
        // The engine's own modelled timing rather than the render thread's backlog. The work
        // this acknowledges still rides the queue, and everything the renderer does - presents
        // included - stays ordered behind it.
        early_interrupt_poster(id, delay_ns);
        return;
    }
    // Timed: the gap between enqueueing this and the render thread reaching it is exactly what
    // a guest waiting on the completion has to sit through. See Common::PipelineStats.
    const u64 queued_at = Common::PipelineStats::NowUs();
    gpu_thread.Enqueue({.run = [this, id, delay_ns, queued_at] {
        Common::PipelineStats::interrupt_lag_us.fetch_add(
            Common::PipelineStats::NowUs() - queued_at, std::memory_order_relaxed);
        Common::PipelineStats::interrupt_count.fetch_add(1, std::memory_order_relaxed);
        deferred_interrupt_poster(id, delay_ns);
    }});
}

void ThreadedRasterizer::ShipDraw(Pica::DrawPayload&& payload, void* slot) {
#ifdef CITRA_TRACE_PROBES
    {
        static const bool trace = std::getenv("AZAHAR_DRAW_TRACE") != nullptr;
        static std::atomic<u32> n{0};
        if (trace && n.fetch_add(1) < 3000) {
            LOG_INFO(HW_GPU, "DRAWTRACE ship hw={}", payload.hw);
        }
    }
#endif // CITRA_TRACE_PROBES
    Shipment* const shipment = TakeShipment();
    shipment->has_draw = true;
    shipment->draw = payload;
    shipment->owned_arena = std::move(emergency_arena);
    shipment->owned_arena_bytes = emergency_arena_bytes;
    emergency_arena_bytes = 0;
    CollectState(*shipment);

    auto* const arena_slot = static_cast<ArenaSlot*>(slot);
    if (arena_slot) {
        arena_slot->pending.fetch_add(1, std::memory_order_acq_rel);
    }
    gpu_thread.Enqueue({.run = [this, shipment, arena_slot] {
        ApplyShipment(mirror_pica, *shipment);
        const u32 watch = GuestWatch::Read();
        if (!shipment->draw.hw || !real->AccelerateShippedDraw(shipment->draw)) {
            mirror_pica.RunShippedDraw(shipment->draw);
        }
        if (shipment->draw.in_ring) {
            RingDrawConsumed(shipment->draw.ring_end);
        }
        {
            const auto& fb = mirror_pica.regs.internal.framebuffer.framebuffer;
            GuestWatch::Check(watch, "draw", fb.GetColorBufferPhysicalAddress(),
                              fb.GetDepthBufferPhysicalAddress(),
                              (fb.GetWidth() << 16) | fb.GetHeight());
        }
        if (arena_slot) {
            arena_slot->pending.fetch_sub(1, std::memory_order_acq_rel);
        }
        if (shipment->owned_arena_bytes) {
            emergency_bytes.fetch_sub(shipment->owned_arena_bytes, std::memory_order_relaxed);
        }
        ReturnShipment(shipment);
    }});
}

void ThreadedRasterizer::DrawTriangles() {
    if (ring ? ranges.empty() : batch.empty()) {
        return;
    }

    // Frameskip is cost-based per render target: the bottom screen's framebuffer (240x320
    // against the top's 240x400) carries cheap 2D that titles often draw exactly once — a HUD
    // dropped with its frame stays black forever. Only draws into the expensive, continuously
    // redrawn top-screen targets are skippable.
    const auto& fb = emu_pica.regs.internal.framebuffer.framebuffer;
    const u32 fb_height = fb.height + 1;
    const u32 color_target = fb.GetColorBufferPhysicalAddress();
    // Either the target is the top screen's own framebuffer, or it is one this phase has
    // already drawn into at least twice - see the note on repeat_targets. Recording happens
    // whatever the skip state, so a target that only starts repeating during a skipped run
    // is still recognised.
    const bool repeat_target = IsRepeatTarget(color_target);
    RecordDrawTarget(color_target);
    const bool skippable_target = fb_height > 350 || repeat_target;

    if (fb_height > 350) {
        // Remember the target so clears aimed at it can be dropped along with the draws.
        RecordTopTarget(fb.GetColorBufferPhysicalAddress());
        RecordTopTarget(fb.GetDepthBufferPhysicalAddress());
    }

    if (skip_frame && skippable_target) {
        // Nothing ships for a skipped batch — not even state. The dirty bits keep accumulating
        // on the emulation side and the next shipment carries them all at once; with the guest
        // uncapped, per-batch state ops alone were enough to saturate the render thread. In ring
        // mode the cursor simply rewinds over the batch's vertices: nothing referenced them yet.
        if (ring) {
            ring->Rewind(batch_start_cursor, batch_start_offset);
            ranges.clear();
        } else {
            batch.clear();
        }
        return;
    }

    Shipment* const shipment = TakeShipment();
    if (ring) {
        std::swap(shipment->ranges, ranges);
        ranges.clear();
        batch_start_cursor = ring->cursor;
        batch_start_offset = ring->offset;
    } else {
        // The batch swaps into the shipment and a pooled buffer (with grown capacity) swaps in
        // behind it; the render thread returns the shipment's buffer to the pool afterwards, so
        // no per-batch reallocation ever happens in steady state.
        shipment->vertices = std::move(batch);
        batch = TakeBuffer();
    }

    CollectState(*shipment);

    gpu_thread.Enqueue({.run = [this, shipment] {
        ApplyShipment(mirror_pica, *shipment);
        if (!shipment->ranges.empty()) {
            for (const auto& range : shipment->ranges) {
                real->DrawRingRange(range.first, range.count, range.end_pos);
            }
        } else if (!shipment->vertices.empty()) {
            for (std::size_t i = 0; i + 2 < shipment->vertices.size(); i += 3) {
                real->AddTriangle(shipment->vertices[i], shipment->vertices[i + 1],
                                  shipment->vertices[i + 2]);
            }
            real->DrawTriangles();
            ReturnBuffer(std::move(shipment->vertices));
        }
        ReturnShipment(shipment);
    }});
}

void ThreadedRasterizer::MarkGLWritten(PAddr addr, u32 size) {
    if (size == 0) {
        return;
    }
    if (addr == last_marked_addr && size == last_marked_size) {
        return; // The framebuffer registers rarely move between draws.
    }
    last_marked_addr = addr;
    last_marked_size = size;
    const PAddr begin = std::max(addr, GL_WRITTEN_BASE);
    const u64 end = std::min<u64>(u64(addr) + size, GL_WRITTEN_END);
    if (begin >= end) {
        return;
    }
    const u32 first = (begin - GL_WRITTEN_BASE) >> GL_WRITTEN_PAGE_BITS;
    const u32 last = static_cast<u32>(end - 1 - GL_WRITTEN_BASE) >> GL_WRITTEN_PAGE_BITS;
    // Word-granular fill: transfers mark hundreds of pages per call and this measured 5% of
    // the emulation thread when done bit by bit.
    const u32 first_word = first >> 6;
    const u32 last_word = last >> 6;
    const u64 first_mask = ~0ull << (first & 63);
    const u64 last_mask = ~0ull >> (63 - (last & 63));
    if (first_word == last_word) {
        gl_written_pages[first_word] |= first_mask & last_mask;
        return;
    }
    gl_written_pages[first_word] |= first_mask;
    for (u32 word = first_word + 1; word < last_word; ++word) {
        gl_written_pages[word] = ~0ull;
    }
    gl_written_pages[last_word] |= last_mask;
}

bool ThreadedRasterizer::TestGLWritten(PAddr addr, u32 size) const {
    if (size == 0) {
        return false;
    }
    if (addr < GL_WRITTEN_BASE || u64(addr) + size > GL_WRITTEN_END) {
        return true; // Outside the tracked window: be conservative.
    }
    const u32 first = (addr - GL_WRITTEN_BASE) >> GL_WRITTEN_PAGE_BITS;
    const u32 last = (addr + size - 1 - GL_WRITTEN_BASE) >> GL_WRITTEN_PAGE_BITS;
    const u32 first_word = first >> 6;
    const u32 last_word = last >> 6;
    const u64 first_mask = ~0ull << (first & 63);
    const u64 last_mask = ~0ull >> (63 - (last & 63));
    if (first_word == last_word) {
        return (gl_written_pages[first_word] & first_mask & last_mask) != 0;
    }
    if (gl_written_pages[first_word] & first_mask) {
        return true;
    }
    for (u32 word = first_word + 1; word < last_word; ++word) {
        if (gl_written_pages[word] != 0) {
            return true;
        }
    }
    return (gl_written_pages[last_word] & last_mask) != 0;
}

ThreadedRasterizer::Shipment* ThreadedRasterizer::TakeShipment() {
    {
        std::scoped_lock lock{shipment_pool_mutex};
        if (!shipment_free.empty()) {
            Shipment* shipment = shipment_free.back();
            shipment_free.pop_back();
            shipment->Reset();
            return shipment;
        }
    }
    shipment_pool.push_back(std::make_unique<Shipment>());
    return shipment_pool.back().get();
}

void ThreadedRasterizer::ReturnShipment(Shipment* shipment) {
    std::scoped_lock lock{shipment_pool_mutex};
    shipment_free.push_back(shipment);
}

void ThreadedRasterizer::FlushAll() {
    gpu_thread.RunSync([this] { real->FlushAll(); });
}

void ThreadedRasterizer::FlushRegion(PAddr addr, u32 size) {
    // A flush writes GL-rendered content back to guest memory before the caller reads it. If
    // the GL never wrote inside this range there is nothing to wait for; the page map is
    // conservative, so a clear range is proof.
    if (!TestGLWritten(addr, size)) {
        return;
    }
    if (unsafe_async_flush) {
        // The caller reads stale guest memory for up to the queue latency; see
        // SetUnsafeAsyncFlush.
        gpu_thread.Enqueue({.run = [this, addr, size] { real->FlushRegion(addr, size); }});
        return;
    }
    gpu_thread.RunSync([this, addr, size] { real->FlushRegion(addr, size); });
}

void ThreadedRasterizer::InvalidateRegion(PAddr addr, u32 size) {
    // Invalidation is an ordering requirement, not a synchronous one: the caches must drop
    // stale surfaces before the *next* GL op that could read them, and that op sits behind this
    // one in the same queue. Fire and forget.
    gpu_thread.Enqueue({.run = [this, addr, size] { real->InvalidateRegion(addr, size); }});
}

void ThreadedRasterizer::InvalidateGuestFlushedRegion(PAddr addr, u32 size) {
    // Decided where the answer lives: the cache's dirty regions, on the render thread, say
    // byte-exactly which pixels exist only host-side. Ordered like any invalidate.
    gpu_thread.Enqueue(
        {.run = [this, addr, size] { real->InvalidateGuestFlushedRegion(addr, size); }});
}

void ThreadedRasterizer::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    if (!TestGLWritten(addr, size)) {
        InvalidateRegion(addr, size);
        return;
    }
    if (unsafe_async_flush) {
        gpu_thread.Enqueue(
            {.run = [this, addr, size] { real->FlushAndInvalidateRegion(addr, size); }});
        return;
    }
    gpu_thread.RunSync([this, addr, size] { real->FlushAndInvalidateRegion(addr, size); });
}

void ThreadedRasterizer::ClearAll(bool flush) {
    gpu_thread.RunSync([this, flush] { real->ClearAll(flush); });
}

} // namespace VideoCore
