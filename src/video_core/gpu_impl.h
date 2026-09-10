// Copyright 2023 Citra Emulator Project
// Copyright 2024 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>

#include "common/archives.h"
#include "common/pipeline_stats.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/gpu_thread.h"
#include "video_core/threaded_rasterizer.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {
/// One VBlank copy of a CPU-written framebuffer; see GPU::Impl::snapshot_slots.
struct DisplaySnapshotSlot {
    PAddr addr = 0;
    u32 bytes = 0;
    u8* pixels = nullptr; ///< page-aligned, inside `raw`
    void* raw = nullptr;  ///< what malloc returned, a page of slack included
    std::atomic<u32> in_use{0};
};

struct GPU::Impl {
    Core::Timing& timing;
    Core::System& system;
    Memory::MemorySystem& memory;
    std::shared_ptr<Pica::DebugContext> debug_context;
    Pica::PicaCore pica;
    /// The render thread's view of PICA state, refreshed from shipments; the renderer and real
    /// rasterizer are constructed against this instance, never against `pica`.
    Pica::PicaCore pica_mirror;
    GraphicsDebugger gpu_debugger;
    std::unique_ptr<RendererBase> renderer;
    RasterizerInterface* rasterizer;
    std::unique_ptr<ThreadedRasterizer> shipping_rasterizer;
    std::unique_ptr<SwRenderer::SwBlitter> sw_blitter;
    Core::TimingEventType* vblank_event;
    Service::GSP::InterruptHandler signal_interrupt;
    /// The handler PICA sees: posts through the GPU thread's interrupt mailbox when needed.
    Service::GSP::InterruptHandler pica_interrupt;
    GpuThread gpu_thread;
    Core::TimingEventType* gpu_interrupt_event = nullptr;
    Core::TimingEventType* fill_finished_event = nullptr;
    Frontend::EmuWindow* emu_window = nullptr;
    Frontend::EmuWindow* secondary_window_ptr = nullptr;
    std::atomic<u64> presented_frames{0};
    /// Render pacing: a frame renders only while a token is free, and the token returns when its
    /// present has actually executed. Feedback-paces drawing to what the GL stack really absorbs,
    /// including llvmpipe's internal pipeline that queue depth cannot see.
    std::atomic<s32> render_tokens{2};
    bool unlimited_speed = false;
    /// Destination addresses of transfers handed to the render thread and not yet performed.
    /// A framebuffer in here holds whatever it held before - for a recycled buffer, another
    /// screen's picture - so presenting it is what puts the top screen on the bottom.
    std::mutex pending_fill_mutex;
    /// Display-buffer writes queued but not yet performed, as address ranges: the display
    /// registers may select an address inside a transferred buffer rather than its base
    /// (Mario Kart 7 shows its top frame from 0x2000 into the buffer the transfer writes).
    struct FillRange {
        PAddr addr;
        u32 size;
        [[nodiscard]] bool Contains(PAddr a) const {
            return a >= addr && a < addr + size;
        }
    };
    std::vector<FillRange> pending_fill;
    /// Completion order of blits into display buffers: the address of each write the render
    /// thread has finished, with a rising sequence number. The present path picks whichever of
    /// a screen's two candidate addresses completed most recently. Selection by completion
    /// order is monotonic - the render thread finishes these in queue order - where selection
    /// by the (historical) register snapshot walks backwards through the guest's rotating
    /// buffers whenever the renderer lags.
    /// Sized for what one title actually rotates: SM3DL cycles nine-plus display-buffer
    /// destinations (three per screen pane), so eight slots evicted live entries every frame
    /// and LastFillSeq answered 0 for a buffer written moments ago - the selection then fell
    /// back to whichever slot survived, an older frame.
    struct CompletedFill {
        PAddr addr = 0;
        u32 size = 0;
        u64 seq = 0;
        [[nodiscard]] bool Contains(PAddr a) const {
            return addr != 0 && a >= addr && a < addr + size;
        }
    };
    std::array<CompletedFill, 32> completed_fill{};
    /// AZAHAR_PRESENT_LOG: the picture each pane last showed (PickDisplayAddr).
    struct LastShown {
        PAddr addr = 0;
        u64 seq = 0;
    };
    std::array<LastShown, 3> pane_last_shown{};
    u64 vblank_serial = 0; ///< AZAHAR_PRESENT_LOG: numbers the vblanks in the trace
    u64 completed_fill_seq = 0;
    /// Per-pane display buffer candidates (0 = top-left, 1 = top-right, 2 = bottom). The
    /// framebuffer registers name the pair of buffers the guest is about to write next, so with
    /// a lagging renderer and three rotating buffers, the buffer holding the newest *completed*
    /// frame is outside the register pair on every other present - selection constrained to the
    /// pair alternates between the newest frame and the one before it (measured: the fade's
    /// luminance bouncing new/old per present). The candidates are every address this pane's
    /// registers have named recently; the newest completion among them is monotonic.
    struct PaneCandidates {
        std::array<PAddr, 4> addrs{};
        u32 next = 0;
    };
    std::array<PaneCandidates, 3> pane_candidates{};

    /// The guest-memory writes the render thread performs late: every queued fill and
    /// transfer, with when it was queued and when it actually ran. Read by the fault reporter
    /// so a guest data abort can be matched against a write that landed on memory the guest
    /// had already reused (pending_fill_mutex guards it).
    struct GuestWrite {
        u32 seq = 0;
        u8 kind = 0; // 0 fill, 1 texture copy, 2 display transfer
        PAddr start = 0, end = 0;
        u32 value = 0;
        u64 queued_us = 0, done_us = 0;
    };
    std::array<GuestWrite, 256> guest_writes{};
    u32 guest_write_seq = 0;
    u32 NoteGuestWrite(u8 kind, PAddr start, PAddr end, u32 value) {
        std::scoped_lock lock{pending_fill_mutex};
        const u32 seq = ++guest_write_seq;
        guest_writes[seq % guest_writes.size()] = {seq, kind, start, end, value,
                                                   Common::PipelineStats::NowUs(), 0};
        return seq;
    }
    void NoteGuestWriteDone(u32 seq) {
        std::scoped_lock lock{pending_fill_mutex};
        auto& w = guest_writes[seq % guest_writes.size()];
        if (w.seq == seq) {
            w.done_us = Common::PipelineStats::NowUs();
        }
    }

    void RecordFillDone(PAddr addr, u32 size) {
        std::scoped_lock lock{pending_fill_mutex};
        auto* slot = &completed_fill[0];
        for (auto& entry : completed_fill) {
            if (entry.addr == addr) {
                slot = &entry;
                break;
            }
            if (entry.seq < slot->seq) {
                slot = &entry; // no entry for this address yet: take the oldest
            }
        }
        *slot = {addr, size, ++completed_fill_seq};
    }
    /**
     * VBlank snapshots of CPU-written framebuffers.
     *
     * A present captures the framebuffer registers when it is queued but reads the pixels when
     * it runs, which may be frames later. For a framebuffer the GPU writes that is fine: the
     * writes ride the same queue and the cache holds the pixels. For one the CPU writes -
     * software renderers, video decoders, libctru's console - the late read shows whatever the
     * guest has put there since, so a double-buffered title tears and shows frame n+1 twice
     * while frame n is never seen. The real display scans out at the VBlank; so does this: the
     * emulation thread copies the bytes at the VBlank and the present samples the copy.
     *
     * Slots are taken at the VBlank and released when the present that used them has run; a
     * VBlank that finds none free presents live memory as before. Buffers are page-aligned so
     * the Vita can map them for the GPU to sample in place.
     */
    std::array<std::array<DisplaySnapshotSlot, 4>, 2> snapshot_slots;
    /// What the present now running shows per screen (render thread).
    std::array<const DisplaySnapshotSlot*, 2> display_snapshot{};
    u64 snapshots_taken = 0;
    ~Impl();

    u64 floor_last_presented = 0;
    u32 floor_starved_vblanks = 0;
    u32 skip_flip_pending_vblanks = 0;
    /// The same wait, for arming rather than releasing the skip.
    u32 skip_arm_pending_vblanks = 0;
    u32 skip_streak_vblanks = 0;
    // A duty-cycle un-skip that has been decided but not yet consumed by a frame boundary.
    bool skip_forced_render = false;
    bool frameskip_allowed = true;
    /// Skip state decided at vblank but applied to draws only at the game's own frame boundary
    /// (the top-screen display transfer), so one frame's draws never split across skip states.
    bool skip_pending = false;
    bool skip_current = false;
    /// Last presented top-screen framebuffer addresses; a change means a scene transition, whose
    /// first frames are force-rendered — transition screens are the classic draw-once content
    /// that frameskip would otherwise gamble away.
    u32 last_fb_left1 = 0;
    u32 last_fb_left2 = 0;
    u32 force_render_frames = 0;

    explicit Impl(Core::System& system, Frontend::EmuWindow& emu_window_,
                  Frontend::EmuWindow* secondary_window)
        : timing{system.CoreTiming()}, system{system}, memory{system.Memory()},
          debug_context{Pica::g_debug_context}, pica{memory, debug_context},
          pica_mirror{memory, nullptr},
          renderer{VideoCore::CreateRenderer(emu_window_, secondary_window, pica_mirror, system)},
          rasterizer{renderer->Rasterizer()},
          sw_blitter{std::make_unique<SwRenderer::SwBlitter>(memory, rasterizer)},
          emu_window{&emu_window_}, secondary_window_ptr{secondary_window} {}
};
} // namespace VideoCore