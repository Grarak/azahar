// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/archives.h"
#include "common/hacks/hack_manager.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/frontend/emu_window.h"
#include "core/loader/loader.h"
#include "video_core/debug_utils/debug_utils.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/guest_watch.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef ENABLE_OPENGL
#include <glad/glad.h>
#endif

namespace VideoCore {

/// Presents handed to the render thread and not yet drawn before frames start being skipped.
/// One in flight is the pipelining the queue exists for; beyond that the display is simply
/// falling behind, and what it falls behind by is what the player feels as lag.
constexpr u32 PresentBacklogSkip = 2;

/// How long the queue may take to drain before frames start being skipped. Presents alone do
/// not see this: most of what piles up is draws, and it is the total the display waits behind.
///
/// This used to be a count of ops, 128 of them, chosen because that was a fraction of a second
/// at the rate the software renderer retires them. An op is not a fixed amount of work: the
/// GXM backend ships one draw call per op and a guest frame is a couple of hundred of those,
/// each retiring in tens of microseconds, so 128 ops was four milliseconds and the skip was
/// armed permanently - and the duty-cycle escape below is itself gated on being under the
/// threshold, so the only thing that ever un-skipped was the one-per-second floor rescue. That
/// is what a hardware backend running at two frames a second while nothing was saturated
/// turned out to be. A time reproduces the old behaviour for the software renderer, whose ops
/// are around a millisecond, and reads the fast backend correctly.
constexpr u64 QueueBacklogSkipUs = 120000;


constexpr VAddr VADDR_LCD = 0x1ED02000;
constexpr VAddr VADDR_GPU = 0x1EF00000;

class DelayGenerator {
private:
    DelayGenerator() = default;

    // Average transfer speed based on measurements taken from real
    // hardware. 4 different modes have been taken into consideration:
    // RAM -> RAM, RAM -> VRAM, VRAM -> RAM and VRAM -> VRAM.
    // Furthermore, measurements are split into DMA transfers and tex
    // copies. For simplicity, we will assume fills are as fast as
    // texture copies.

    static constexpr double mibps_to_ns_per_byte(double mib_per_sec) {
        return 1'000'000'000.0 / (mib_per_sec * 1024.0 * 1024.0);
    }

    static constexpr std::array<std::array<double, 4>, 2> speed_mibps = {
        {{
             190.0, // DMA RAMTORAM
             310.0, // DMA RAMTOVRAM
             380.0, // DMA VRAMTORAM
             380.0, // DMA VRAMTOVRAM
         },
         {
             450.0,  // TEX RAMTORAM
             3100.0, // TEX RAMTOVRAM
             5400.0, // TEX VRAMTORAM
             5400.0, // TEX VRAMTOVRAM
         }}};

public:
    enum class CopyMode {
        RAMTORAM,
        RAMTOVRAM,
        VRAMTORAM,
        VRAMTOVRAM,
    };

    static CopyMode GetCopyMode(bool input_vram, bool output_vram) {
        if (!input_vram && !output_vram) {
            return CopyMode::RAMTORAM;
        } else if (!input_vram && output_vram) {
            return CopyMode::RAMTOVRAM;
        } else if (input_vram && !output_vram) {
            return CopyMode::VRAMTORAM;
        } else {
            return CopyMode::VRAMTOVRAM;
        }
    }

    static u64 CalculateDelayNanoseconds(CopyMode mode, bool is_textre, size_t size) {
        double base_ns_per_byte =
            mibps_to_ns_per_byte(speed_mibps[is_textre][static_cast<u32>(mode)]);

        return static_cast<u64>(size * base_ns_per_byte);
    }
};

MICROPROFILE_DEFINE(GPU_DisplayTransfer, "GPU", "DisplayTransfer", MP_RGB(100, 100, 255));

GPU::Impl::~Impl() {
    for (auto& screen : snapshot_slots) {
        for (auto& slot : screen) {
            std::free(slot.raw);
        }
    }
}

/// Emulation thread, at the VBlank: copies the framebuffer the guest shows on `screen` into a
/// free slot when the CPU is what writes it, and returns the slot; null otherwise.
DisplaySnapshotSlot* GPU::TakeDisplaySnapshot(u32 screen) {
    Impl& impl = *this->impl;
    const auto& fb = impl.pica.regs.framebuffer_config[screen];
    const PAddr addr = fb.active_fb == 0 ? fb.address_left1 : fb.address_left2;
    const u32 bytes = fb.stride * fb.height;
    if (addr == 0 || bytes == 0 || bytes > GPU::DisplaySnapshotBytes) {
        return nullptr;
    }
    // Anything a fill, transfer or draw has ever landed on is GPU-written: the pixels are in
    // the surface cache, in queue order, and a snapshot of guest memory would be stale. The
    // map is page-granular and conservative, so a CPU framebuffer sharing a page with GPU
    // output is presented live, as before.
    if (impl.shipping_rasterizer->TestGLWritten(addr, bytes)) {
        return nullptr;
    }
    const u8* source = impl.memory.GetPhysicalPointer(addr);
    if (source == nullptr) {
        return nullptr;
    }
    for (auto& slot : impl.snapshot_slots[screen]) {
        if (slot.in_use.load(std::memory_order_acquire) != 0) {
            continue;
        }
        if (slot.pixels == nullptr) {
            // Page-aligned and a whole number of pages, so the Vita can map it for the GPU.
            // Aligned by hand the way HostSharedMemory does: vitasdk's newlib has no
            // posix_memalign and its aligned_alloc gives malloc alignment.
            constexpr std::uintptr_t page = 4096;
            const std::size_t rounded = (GPU::DisplaySnapshotBytes + page - 1) / page * page;
            slot.raw = std::malloc(rounded + page);
            if (slot.raw == nullptr) {
                return nullptr;
            }
            const auto base = reinterpret_cast<std::uintptr_t>(slot.raw);
            slot.pixels = reinterpret_cast<u8*>((base + page - 1) & ~(page - 1));
        }
        std::memcpy(slot.pixels, source, bytes);
        slot.addr = addr;
        slot.bytes = bytes;
        slot.in_use.store(1, std::memory_order_release);
        impl.snapshots_taken++;
        Common::PipelineStats::display_snapshots.fetch_add(1, std::memory_order_relaxed);
        return &slot;
    }
    return nullptr;
}
MICROPROFILE_DEFINE(GPU_CmdlistProcessing, "GPU", "Cmdlist Processing", MP_RGB(100, 255, 100));

GPU::GPU(Core::System& system, Frontend::EmuWindow& emu_window,
         Frontend::EmuWindow* secondary_window)
    : right_eye_disabler{std::make_unique<RightEyeDisabler>(*this)},
      impl{std::make_unique<Impl>(system, emu_window, secondary_window)} {
    impl->renderer->SetOwnerGPU(this);
    GuestWatch::Init(system.Memory());
    impl->vblank_event = impl->timing.RegisterEvent(
        "GPU::VBlankCallback",
        [this](uintptr_t user_data, s64 cycles_late) { VBlankCallback(user_data, cycles_late); });
    impl->timing.ScheduleEvent(FRAME_TICKS, impl->vblank_event);

    // Interrupts raised while executing on the GPU thread cannot touch the kernel from there;
    // they are posted through this event, which core timing moves onto the emulation thread.
    // user_data packs the interrupt id in the top four bits and the delay in the low 28.
    impl->gpu_interrupt_event = impl->timing.RegisterEvent(
        "GPU::ThreadInterrupt", [this](uintptr_t user_data, s64) {
            const auto id = static_cast<Service::GSP::InterruptId>(user_data >> 28);
            const u64 delay_ns = user_data & 0x0FFFFFFF;
            if (impl->signal_interrupt) {
                impl->signal_interrupt(id, delay_ns);
            }
        });

    // Software-renderer retirement ordering for memory fills: the guest polls the fill
    // engine's `finished` bit, so under the software renderer it may only flip after the
    // fill's memory effect has landed. The render-thread op schedules this event, which sets
    // the bit back on the emulation thread (register words are only ever written there).
    impl->fill_finished_event = impl->timing.RegisterEvent(
        "GPU::FillFinished", [this](uintptr_t user_data, s64) {
            const u32 index = static_cast<u32>(user_data);
            if (index < 2) {
                impl->pica.regs.memory_fill_config[index].finished.Assign(1);
            }
        });

    // The emulation-side PICA drives the shipping facade; the real rasterizer only ever runs on
    // the render thread against the mirror.
    impl->shipping_rasterizer = std::make_unique<ThreadedRasterizer>(
        impl->gpu_thread, impl->pica, impl->pica_mirror);
    impl->shipping_rasterizer->SetRealRasterizer(impl->rasterizer);
    // The software renderer's rendered output lands in guest RAM on the render thread, and its
    // fills and transfers run on the CPU against guest RAM too - GL absorbs both into the host
    // GPU and never writes there. So its completion interrupts ride the render-thread queue:
    // signalling at ship time hands the guest a licence to reuse memory a queued write has not
    // reached yet, which corrupted titles right after their intro.
    if (Settings::GetWorkingGraphicsAPI() == Settings::GraphicsAPI::Software) {
        impl->shipping_rasterizer->SetDeferredInterrupts(
            [this](Service::GSP::InterruptId id, u64 delay_ns) { PostInterrupt(id, delay_ns); });
        impl->shipping_rasterizer->SetEarlyInterruptPoster(
            [this](Service::GSP::InterruptId id, u64 delay_ns) {
                PostInterruptDeferred(id, delay_ns);
            });
    }
    impl->pica.BindRasterizer(impl->shipping_rasterizer.get());
    // Shipped draws execute against the mirror, whose triangles go straight to the real
    // rasterizer on the render thread.
    impl->pica_mirror.BindRasterizer(impl->rasterizer);
}

void GPU::PostInterrupt(Service::GSP::InterruptId id, u64 delay_ns) {
    if (impl->gpu_thread.OnThread()) {
        const uintptr_t packed = (static_cast<uintptr_t>(id) << 28) |
                                 static_cast<uintptr_t>(std::min<u64>(delay_ns, 0x0FFFFFFF));
        impl->timing.ScheduleEvent(0, impl->gpu_interrupt_event, packed,
                                   std::numeric_limits<std::size_t>::max(), true);
    } else if (impl->signal_interrupt) {
        impl->signal_interrupt(id, delay_ns);
    }
}

void GPU::PostInterruptDeferred(Service::GSP::InterruptId id, u64 delay_ns) {
    // Zero-cycle event so delivery happens on a clean stack: these are raised from inside GSP's
    // own command handling, which must not re-enter.
    const uintptr_t packed = (static_cast<uintptr_t>(id) << 28) |
                             static_cast<uintptr_t>(std::min<u64>(delay_ns, 0x0FFFFFFF));
    impl->timing.ScheduleEvent(0, impl->gpu_interrupt_event, packed);
}

void GPU::QueueMemoryFill(u32 index, u32 start, u32 end, u32 value, u32 control, u32 intr_index) {
    // Mirror the register-visible effects on the emulation side immediately: games poll the
    // `finished` bit through ReadHWRegs, and the completion interrupt runs on the fill engine's
    // modelled timing, never on the render thread's backlog. The memory effect lands when the op
    // retires — the same contract the hardware offers software that ignores the interrupt.
    auto& config = impl->pica.regs.memory_fill_config[index];
    config.address_start = start;
    config.address_end = end;
    config.value_32bit = value;
    config.control = control;

    Pica::MemoryFillConfig snapshot = config;

    u64 delay = DelayGenerator::CalculateDelayNanoseconds(
        DelayGenerator::GetCopyMode(true, config.IsVRAM()), true,
        config.GetEndAddress() - config.GetStartAddress());
    const bool defer_completion = impl->shipping_rasterizer->DefersInterrupts();
    // Completion signals on the engine's modelled timing while the write still rides the
    // queue; see the completion note in threaded_rasterizer.h.
    const auto post_psc = [&](bool deferred) {
        if (config.GetStartAddress() == 0) {
            return;
        }
        const auto id = intr_index == 0 ? Service::GSP::InterruptId::PSC0
                                        : Service::GSP::InterruptId::PSC1;
        if (intr_index > 1) {
            return;
        }
        if (deferred) {
            impl->shipping_rasterizer->PostInterruptAfterQueue(id, delay);
        } else {
            PostInterruptDeferred(id, delay);
        }
    };
    post_psc(false);
    config.trigger.Assign(0);
    config.finished.Assign(1);

    impl->shipping_rasterizer->MarkGLWritten(snapshot.GetStartAddress(),
                                             snapshot.GetEndAddress() -
                                                 snapshot.GetStartAddress());

    // A clear aimed at a render target whose repaint is being skipped must be skipped with it:
    // running the clear alone leaves the target holding nothing but the clear colour, and that
    // is what the next present shows - a frame of flat sky with the terrain missing. Dropping
    // both keeps the last completely drawn frame, which is what frameskip means. Matching is by
    // range, not by base: a title that draws both screens into one target clears part way into
    // it, and an exact-base test does not see that. Non-target fills always run.
    constexpr u32 MaxFillTargetSpan = 400 * 240 * 4;
    if (impl->shipping_rasterizer && impl->shipping_rasterizer->IsSkippingFrame() &&
        impl->shipping_rasterizer->IsWithinTopTarget(snapshot.GetStartAddress(),
                                                     MaxFillTargetSpan)) {
        // The completion was posted above, before this branch, and posting it again here sent
        // the guest two PSC interrupts for one fill. GSP counts those, so the extra one is a
        // completion for work that was never requested: the guest consumes it in place of the
        // next fill's, and the thread waiting on that one waits forever - holding whatever it
        // holds, which is how a title ends up with most of its threads parked on locks nobody
        // will release. It only ever fired while skipping, which is why turning frameskip off
        // made the stall go away and suppressing the draws and transfers did not.
        return;
    }

    // The software blitter writes guest memory directly, riding the queue rather than being
    // waited for. GL absorbs these into AccelerateFill and needs no such thing.
    const u32 write_seq = impl->NoteGuestWrite(0, snapshot.GetStartAddress(),
                                               snapshot.GetEndAddress(), snapshot.value_32bit);
    impl->gpu_thread.Enqueue({.run = [this, snapshot, write_seq] {
        const u32 watch = GuestWatch::Read();
        if (!impl->rasterizer->AccelerateFill(snapshot)) {
            impl->sw_blitter->MemoryFill(snapshot);
        }
        GuestWatch::Check(watch, "fill", snapshot.GetStartAddress(), snapshot.GetEndAddress(),
                          snapshot.value_32bit);
        impl->NoteGuestWriteDone(write_seq);
        // A fill into a display buffer is presentable content too (loading screens are
        // often nothing else), so it takes part in completion-order selection.
        impl->RecordFillDone(snapshot.GetStartAddress(),
                             snapshot.GetEndAddress() - snapshot.GetStartAddress());
    }});
}

void GPU::QueueMemoryTransfer(const Pica::DisplayTransferConfig& config_in) {
    Pica::DisplayTransferConfig config = config_in;
    impl->pica.regs.display_transfer_config = config;

    u64 delay{};
    if (config.is_texture_copy) {
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.texture_copy.size);
    } else {
        delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(config.IsInputVRAM(), config.IsOutputVRAM()), true,
            config.input_width * config.input_height * BytesPerPixel(config.input_format));
    }
    const bool defer_completion = impl->shipping_rasterizer->DefersInterrupts();
    PostInterruptDeferred(Service::GSP::InterruptId::PPF, delay);
    impl->pica.regs.display_transfer_config.trigger.Assign(0);

    // Transfers normally run even for skipped frames: unlike draws they are how content persists —
    // HUDs and result screens are often transferred exactly once, and a skipped transfer is a
    // black region forever. The exception is a top-screen display transfer during a skipped frame:
    // its source is the render target whose repaint was just dropped, so running it would push the
    // stale (cleared) target over the last good frame in the display buffer. Top-sized outputs are
    // exactly the continuously redrawn case, so the next unskipped frame refreshes them.
    if (config.is_texture_copy) {
        impl->shipping_rasterizer->MarkGLWritten(config.GetPhysicalOutputAddress(),
                                                 config.texture_copy.size);
    } else {
        impl->shipping_rasterizer->MarkGLWritten(
            config.GetPhysicalOutputAddress(),
            config.output_width * config.output_height * 4);
    }

    const bool top_sized = config.output_width > 350 || config.output_height > 350;
    // A transfer whose source is a render target the skipped frame never drew into carries
    // whatever was left there - and this title draws both screens into one target and slices
    // it, so the bottom screen's transfer sources the very target the top's skipped draws
    // would have filled, and the bottom ends up showing the top's picture. Size alone does not
    // catch it: that transfer is 320 wide and so not "top sized". Confirmed by turning
    // frameskip off, which takes the artifact from 4 frames in 30 to none.
    // A render target is at most the largest screen's worth of pixels; anything sourced within
    // that of a recorded target's base is reading that target.
    constexpr u32 MaxTargetSpan = 400 * 240 * 4;
    const bool from_skipped_target =
        impl->shipping_rasterizer &&
        impl->shipping_rasterizer->IsWithinTopTarget(config.GetPhysicalInputAddress(),
                                                     MaxTargetSpan);
    const bool skipping_this_transfer =
        impl->shipping_rasterizer && impl->shipping_rasterizer->IsSkippingFrame();
    // A title that moves its frame to the display buffer with a texture copy instead of a
    // display transfer (Mario Kart 7) has the same frame boundary in that copy: a top-sized
    // copy whose source is the top render target. Without this the skip state only ever
    // changed through the vblank timeouts, mid-frame, and a skipped frame's copy carried the
    // target's previous picture into the display buffer - the picture stepped back a frame
    // every time the skip fired.
    const bool top_copy = config.is_texture_copy && from_skipped_target &&
                          config.texture_copy.size >= 400 * 240 * 2;
    if ((!config.is_texture_copy && top_sized) || top_copy) {
        // The game just finished composing a top-screen frame: the only moment the skip state
        // may change without splitting one frame's draws between skip states. The transfer
        // itself still honors the state the frame was drawn under.
        if (impl->skip_current != impl->skip_pending) {
            impl->skip_current = impl->skip_pending;
            impl->shipping_rasterizer->SetSkipFrame(impl->skip_current);
        }
        // Boundaries are arriving, so flips happen at frame edges; the vblank timeouts below
        // must only rescue phases with no boundaries at all.
        impl->skip_flip_pending_vblanks = 0;
        impl->skip_arm_pending_vblanks = 0;
    }
    if (skipping_this_transfer &&
        ((!config.is_texture_copy && (top_sized || from_skipped_target)) || top_copy)) {
        return;
    }

    const bool allow = config.is_texture_copy ||
                       right_eye_disabler->ShouldAllowDisplayTransfer(
                           config.GetPhysicalInputAddress(), config.input_height);
    const auto blit_op = [this, config, allow] {
        if (config.is_texture_copy) {
            if (!impl->rasterizer->AccelerateTextureCopy(config)) {
                Common::PipelineStats::copy_soft.fetch_add(1, std::memory_order_relaxed);
                impl->sw_blitter->TextureCopy(config);
            } else {
                Common::PipelineStats::copy_accel.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (allow) {
            if (!impl->rasterizer->AccelerateDisplayTransfer(config)) {
                Common::PipelineStats::xfer_soft.fetch_add(1, std::memory_order_relaxed);
                impl->sw_blitter->DisplayTransfer(config);
            } else {
                Common::PipelineStats::xfer_accel.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    {
        // Note the destination until the write has actually happened: a present of this buffer
        // before then shows whatever it held previously.
        const PAddr out_addr = config.GetPhysicalOutputAddress();
        // Texture copies target textures, not screens, and a disallowed transfer wrote
        // nothing - neither may claim its destination is fresh content.
        const bool wrote_display = !config.is_texture_copy && allow;
        const u32 out_size = config.is_texture_copy
                                 ? config.texture_copy.size
                                 : config.output_width * config.output_height *
                                       BytesPerPixel(config.output_format);
        {
            std::scoped_lock lock{impl->pending_fill_mutex};
            impl->pending_fill.push_back({out_addr, out_size});
        }
        const u32 write_seq = impl->NoteGuestWrite(config.is_texture_copy ? 1 : 2, out_addr,
                                                   out_addr + out_size, allow ? 1 : 0);
        const auto op = [this, blit_op, out_addr, wrote_display, write_seq,
                                          out_size, is_copy = config.is_texture_copy,
                                          in_addr = config.GetPhysicalInputAddress()] {
            const u32 watch = GuestWatch::Read();
            blit_op();
            GuestWatch::Check(watch, is_copy ? "texcopy" : "transfer", out_addr, out_addr + out_size,
                              in_addr);
            impl->NoteGuestWriteDone(write_seq);
            {
                std::scoped_lock lock{impl->pending_fill_mutex};
                const auto it =
                    std::find_if(impl->pending_fill.begin(), impl->pending_fill.end(),
                                 [&](const auto& f) { return f.addr == out_addr; });
                if (it != impl->pending_fill.end()) {
                    impl->pending_fill.erase(it);
                }
            }
            if (wrote_display) {
                impl->RecordFillDone(out_addr, out_size);
            }
        };
        if (config.is_texture_copy) {
            // A texture copy's destination is data the guest reads with its own CPU, and the
            // PPF completion is posted on the engine's modelled timing while the write itself
            // rides the render thread's queue. Mario Kart 7 read one 100 ms before the render
            // thread performed it - the crash dump names the copy and how late it ran - and
            // dereferenced a null pointer out of what it found. So a copy is waited for.
            // Display transfers feed the screen rather than the guest's own reads and stay
            // asynchronous, which is where the speed of this design comes from.
            impl->gpu_thread.RunSync(op);
        } else {
            impl->gpu_thread.Enqueue({.run = op});
        }
    }
}

void GPU::StartRenderThread() {
    ASSERT(!impl->gpu_thread.IsRunning());
    impl->renderer->SetThreadedPresentation(true);
    // See SetRasterizerCacheMarkingEnabled: the cache must not touch page tables from the render
    // thread. The frontend flushed and released the context before calling here.
    impl->memory.SetRasterizerCacheMarkingEnabled(false);
    Frontend::EmuWindow* window = impl->emu_window;
    impl->gpu_thread.Start([window] {
        if (window) {
            window->MakeCurrent();
        }
    });
    LOG_INFO(HW_GPU, "GPU execution moved to the render thread");
}

void GPU::StopRenderThread() {
    if (!impl->gpu_thread.IsRunning()) {
        return;
    }
    Frontend::EmuWindow* window = impl->emu_window;
    impl->gpu_thread.Stop([window] {
        if (window) {
            window->DoneCurrent();
        }
    });
}

u64 GPU::LastFillSeq(PAddr addr) const {
    std::scoped_lock lock{impl->pending_fill_mutex};
    for (const auto& fill : impl->completed_fill) {
        if (fill.Contains(addr)) {
            return fill.seq;
        }
    }
    return 0;
}

PAddr GPU::PickDisplayAddr(u32 pane, PAddr slot1, PAddr slot2, PAddr selected) {
    auto& cand = impl->pane_candidates[pane];
    const auto note = [&](PAddr addr) {
        if (addr == 0) {
            return;
        }
        // A buffer the guest hands to this pane stops being a candidate anywhere else - titles
        // recycle display buffers between screens, and a stale claim is one screen showing the
        // other's picture.
        for (u32 other = 0; other < impl->pane_candidates.size(); other++) {
            if (other == pane) {
                continue;
            }
            for (auto& a : impl->pane_candidates[other].addrs) {
                if (a == addr) {
                    a = 0;
                }
            }
        }
        for (const PAddr a : cand.addrs) {
            if (a == addr) {
                return;
            }
        }
        cand.addrs[cand.next] = addr;
        cand.next = (cand.next + 1) % cand.addrs.size();
    };
    note(slot1);
    note(slot2);

    std::scoped_lock lock{impl->pending_fill_mutex};
    // Only a screen the GPU writes is ordered by completed writes. A buffer the registers
    // select that the GPU has never touched and is not about to (libctru's console draws
    // its screen with the CPU into linear memory) is shown as the registers say; the
    // candidate with the newest write would be the buffer the title left behind.
    // By range, not by base: the registers may select an address inside the buffer the
    // transfer wrote (Mario Kart 7's top screen starts 0x2000 into it), and an exact test
    // left that title on register selection, which after a skipped frame is the buffer
    // whose transfer was dropped - the picture stepped back a frame at every skip.
    const bool selected_is_gpu_driven =
        std::any_of(impl->completed_fill.begin(), impl->completed_fill.end(),
                    [&](const auto& fill) { return fill.Contains(selected); }) ||
        std::any_of(impl->pending_fill.begin(), impl->pending_fill.end(),
                    [&](const auto& fill) { return fill.Contains(selected); });
    if (!selected_is_gpu_driven) {
        static const bool log_presents = std::getenv("AZAHAR_PRESENT_LOG") != nullptr;
        static u64 last_log_us = 0;
        const u64 now_us = Common::PipelineStats::NowUs();
        if (log_presents && now_us - last_log_us > 1'000'000) {
            last_log_us = now_us;
            std::string fills;
            for (const auto& fill : impl->completed_fill) {
                if (fill.addr != 0) {
                    fills += fmt::format(" {:#x}+{:#x}:{}", fill.addr, fill.size, fill.seq);
                }
            }
            LOG_INFO(HW_GPU, "pane {} selected {:#x} (slots {:#x}/{:#x}) has no completion; fills:{}",
                     pane, selected, slot1, slot2, fills);
        }
        return 0;
    }
    PAddr best = 0;
    u64 best_seq = 0;
    for (const PAddr a : cand.addrs) {
        if (a == 0) {
            continue;
        }
        for (const auto& fill : impl->completed_fill) {
            if (fill.Contains(a) && fill.seq > best_seq) {
                best_seq = fill.seq;
                best = a;
            }
        }
    }
    // AZAHAR_PRESENT_LOG=1: every present whose picture is older than the last one this pane
    // showed, by completion order - the picture going back in time on screen.
    static const bool log_presents = std::getenv("AZAHAR_PRESENT_LOG") != nullptr;
    if (log_presents) {
        auto& last = impl->pane_last_shown[pane];
        if (best_seq < last.seq) {
            LOG_WARNING(HW_GPU,
                        "present pane {} went back: {:#x} seq {} after {:#x} seq {} (registers "
                        "{:#x}/{:#x} selected {:#x})",
                        pane, best, best_seq, last.addr, last.seq, slot1, slot2, selected);
        }
        last = {best, best_seq};
    }
    return best;
}

bool GPU::IsFillPending(PAddr addr) const {
    std::scoped_lock lock{impl->pending_fill_mutex};
    return std::any_of(impl->pending_fill.begin(), impl->pending_fill.end(),
                       [&](const auto& fill) { return fill.Contains(addr); });
}

u32 GPU::RenderQueueDepth() const {
    return impl->gpu_thread.Depth();
}

const u8* GPU::DisplaySnapshot(u32 screen, PAddr addr) const {
    if (screen > 1) {
        return nullptr;
    }
    const auto* slot = impl->display_snapshot[screen];
    return slot != nullptr && slot->addr == addr ? slot->pixels : nullptr;
}

void GPU::RunOnRenderThread(std::function<void()> fn) {
    impl->gpu_thread.RunSync(std::move(fn));
}

void GPU::SetRenderThreadLockstep(bool lockstep) {
    impl->gpu_thread.SetLockstep(lockstep);
}


void GPU::SetUnlimitedSpeed(bool unlimited) {
    impl->unlimited_speed = unlimited;
}

void GPU::SetFrameskipAllowed(bool allowed) {
    impl->frameskip_allowed = allowed;
}

void GPU::SetUnsafeAsyncFlush(bool enabled) {
    impl->shipping_rasterizer->SetUnsafeAsyncFlush(enabled);
}

VideoCore::RasterizerInterface* GPU::CacheRasterizer() {
    return impl->shipping_rasterizer.get();
}

void GPU::InvalidateOnGuestFlush(PAddr addr, u32 size) {
    // GSP FlushDataCache: the guest's CPU writes to this range are what the GPU must read from
    // now on, so cached copies of it are stale. But a range the GL has rendered into and not
    // yet flushed holds content that exists nowhere but host-side; dropping that reloads
    // whatever guest RAM happens to contain, which presented a top screen of noise. Real
    // hardware has no such case - rendered pixels are in RAM before the CPU writes near them -
    // so rendered content wins here and only what the GL never wrote is dropped.
    //
    // The test used to be a page map of everything the GL had ever written, set at ship time
    // and never cleared, so a texture the CPU composes on a page that once held a render
    // target, fill or transfer output was never invalidated again. The cache's own dirty
    // regions answer the question byte-exactly and as of now, so the decision is made there,
    // on the render thread, ordered like any other invalidate. The software renderer keeps the
    // plain invalidate: its surface cache writes rendered bytes back (shadow-merged) before
    // letting the range reload.
    // With guest write tracking on, the flush's range says nothing the tracker does not
    // know better: only the pages the guest actually stored into are invalidated. (This
    // branch went missing with the `oldflushgate` flag on 2026-09-08 and tracking armed
    // and faulted without ever invalidating anything until it came back.)
    auto& tracker = impl->memory.WriteTracker();
    if (tracker.Enabled()) {
        tracker.Consume([this](u32 paddr, u32 size) {
            impl->shipping_rasterizer->InvalidateGuestFlushedRegion(paddr, size);
        });
        return;
    }
    impl->shipping_rasterizer->InvalidateGuestFlushedRegion(addr, size);
}

u64 GPU::SkippedFrames() const {
    return impl->gpu_thread.SkippedFrames();
}

u64 GPU::PresentedFrames() const {
    return impl->presented_frames.load(std::memory_order_relaxed);
}

bool GPU::RenderThreadRunning() const {
    return impl->gpu_thread.IsRunning();
}

void GPU::SwitchDiskResources(u64 title_id) {
    // Compiles and links GL programs; must run where the context lives.
    impl->gpu_thread.RunSync(
        [this, title_id] { impl->renderer->Rasterizer()->SwitchDiskResources(title_id); });
}

GPU::~GPU() {
    if (impl->gpu_thread.IsRunning()) {
        impl->gpu_thread.RunSync([this] {
            impl->renderer.reset();
            impl->sw_blitter.reset();
            // The presentation mailboxes hold GL objects but live in the windows, which are
            // destroyed on the main thread after the context is gone. Take them down here,
            // where the context is still current.
            if (impl->emu_window) {
                impl->emu_window->mailbox.reset();
            }
            if (impl->secondary_window_ptr) {
                impl->secondary_window_ptr->mailbox.reset();
            }
        });
        StopRenderThread();
    }
}

std::string GPU::DescribeRecentWrites(std::span<const u32> regs, u32 far) {
    std::scoped_lock lock{impl->pending_fill_mutex};
    const u64 now = Common::PipelineStats::NowUs();
    std::string out;
    const auto covers = [](const Impl::GuestWrite& w, PAddr pa) {
        return pa != 0 && pa >= w.start && pa < w.end;
    };
    // Every write on record that covers the faulting address or anything in a register,
    // newest first; then the eight most recent regardless.
    const u32 last = impl->guest_write_seq;
    for (u32 seq = last; seq > 0 && seq + impl->guest_writes.size() > last; seq--) {
        const auto& w = impl->guest_writes[seq % impl->guest_writes.size()];
        if (w.seq != seq) {
            break;
        }
        std::string hits;
        if (covers(w, VirtualToPhysicalAddress(far))) {
            hits += " far";
        }
        for (std::size_t i = 0; i < regs.size(); i++) {
            if (covers(w, VirtualToPhysicalAddress(regs[i]))) {
                hits += fmt::format(" r{}", i);
            }
        }
        const bool recent = seq + 8 > last;
        if (hits.empty() && !recent) {
            continue;
        }
        static constexpr const char* kinds[] = {"fill", "texcopy", "transfer"};
        out += fmt::format("  write {} {} {:08X}-{:08X} value {:08X} queued {} us ago, {}{}\n",
                           w.seq, kinds[w.kind], w.start, w.end, w.value, now - w.queued_us,
                           w.done_us ? fmt::format("ran {} us ago ({} us late)", now - w.done_us,
                                                   w.done_us - w.queued_us)
                                     : "NOT RUN YET",
                           hits.empty() ? "" : " covers" + hits);
    }
    return out;
}

PAddr GPU::VirtualToPhysicalAddress(VAddr addr) {
    if (addr == 0) {
        return 0;
    }

    if (addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END) {
        return addr - Memory::VRAM_VADDR + Memory::VRAM_PADDR;
    }
    if (addr >= Memory::LINEAR_HEAP_VADDR && addr <= Memory::LINEAR_HEAP_VADDR_END) {
        return addr - Memory::LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::NEW_LINEAR_HEAP_VADDR && addr <= Memory::NEW_LINEAR_HEAP_VADDR_END) {
        return addr - Memory::NEW_LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    PAddr plg_fb_addr;
    if (addr >= Memory::PLUGIN_3GX_FB_VADDR && addr <= Memory::PLUGIN_3GX_FB_VADDR_END &&
        (plg_fb_addr = impl->system.Memory().Plugin3GXFramebufferAddress())) {
        return addr - Memory::PLUGIN_3GX_FB_VADDR + plg_fb_addr;
    }

    LOG_ERROR(HW_Memory, "Unknown virtual address @ 0x{:08X}", addr);
    return addr;
}

void GPU::SetInterruptHandler(Service::GSP::InterruptHandler handler) {
    impl->signal_interrupt = handler;
    impl->pica_interrupt = [this](Service::GSP::InterruptId id, u64 delay_ns) {
        PostInterrupt(id, delay_ns);
    };
    impl->pica.SetInterruptHandler(impl->pica_interrupt);
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    impl->gpu_thread.RunSync([this, addr, size] { impl->rasterizer->FlushRegion(addr, size); });
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    impl->gpu_thread.RunSync(
        [this, addr, size] { impl->rasterizer->InvalidateRegion(addr, size); });
}

void GPU::ClearAll(bool flush) {
    impl->gpu_thread.RunSync([this, flush] { impl->rasterizer->ClearAll(flush); });
}

void GPU::Execute(const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;

#ifdef CITRA_TRACE_PROBES
    // Temporary divergence tracing for the software-renderer boot hang.
    static const bool gx_trace = std::getenv("AZAHAR_GX_TRACE") != nullptr;
    static std::atomic<u32> gx_trace_count{0};
    if (gx_trace && gx_trace_count.fetch_add(1) < 8000) {
        LOG_INFO(HW_GPU, "GXTRACE cmd id={}", static_cast<u32>(command.id.Value()));
    }
#endif // CITRA_TRACE_PROBES

    switch (command.id) {
    case CommandId::RequestDma: {
        impl->system.Memory().RasterizerFlushVirtualRegion(
            command.dma_request.source_address, command.dma_request.size, Memory::FlushMode::Flush);
        impl->system.Memory().RasterizerFlushVirtualRegion(command.dma_request.dest_address,
                                                           command.dma_request.size,
                                                           Memory::FlushMode::Invalidate);

        // TODO(Subv): These memory accesses should not go through the application's memory mapping.
        // They should go through the GSP module's memory mapping.
        const auto process = impl->system.Kernel().GetCurrentProcess();
        impl->memory.CopyBlock(*process, command.dma_request.dest_address,
                               command.dma_request.source_address, command.dma_request.size);

        auto is_vram = [&](u32 addr) {
            return addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END;
        };

        u64 delay = DelayGenerator::CalculateDelayNanoseconds(
            DelayGenerator::GetCopyMode(is_vram(command.dma_request.source_address),
                                        is_vram(command.dma_request.dest_address)),
            false, command.dma_request.size);

        impl->signal_interrupt(Service::GSP::InterruptId::DMA, delay);
        break;
    }
    case CommandId::SubmitCmdList: {
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;

        // Write to the command buffer GPU registers
        cmdbuffer.addr[0].Assign(VirtualToPhysicalAddress(params.address) >> 3);
        cmdbuffer.size[0].Assign(params.size >> 3);
        cmdbuffer.trigger[0] = 1;

        // Processing runs right here on the emulation thread: register state, vertex shading and
        // the P3D interrupt all keep their exact timing, and only the resulting triangles travel
        // to the render thread.
        ExecSubmitCmdList(0);
#ifdef CITRA_TRACE_PROBES
        {
            static const bool gx_trace2 = std::getenv("AZAHAR_GX_TRACE") != nullptr;
            static std::atomic<u32> done_count{0};
            if (gx_trace2 && done_count.fetch_add(1) < 20000) {
                LOG_INFO(HW_GPU, "GXTRACE submit done");
            }
        }
#endif // CITRA_TRACE_PROBES
        break;
    }
    case CommandId::MemoryFill: {
        auto& params = command.memory_fill;

        // If both buffers are set GSP dispatches PSC0 only.
        const bool has_both_bufs = params.start1 != 0 && params.start2 != 0;
        if (params.start1 != 0) {
            QueueMemoryFill(0, VirtualToPhysicalAddress(params.start1) >> 3,
                            VirtualToPhysicalAddress(params.end1) >> 3, params.value1,
                            params.control1,
                            has_both_bufs ? std::numeric_limits<u32>::max() : 0);
        }
        if (params.start2 != 0) {
            QueueMemoryFill(1, VirtualToPhysicalAddress(params.start2) >> 3,
                            VirtualToPhysicalAddress(params.end2) >> 3, params.value2,
                            params.control2, has_both_bufs ? 0 : 1);
        }
        break;
    }
    case CommandId::DisplayTransfer: {
        auto& params = command.display_transfer;
        Pica::DisplayTransferConfig config{};
        config.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        config.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        config.input_size = params.in_buffer_size;
        config.output_size = params.out_buffer_size;
        config.flags = params.flags;
        QueueMemoryTransfer(config);
        break;
    }
    case CommandId::TextureCopy: {
        auto& params = command.texture_copy;
        Pica::DisplayTransferConfig config{};
        config.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        config.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        config.texture_copy.size = params.size;
        config.texture_copy.input_size = params.in_width_gap;
        config.texture_copy.output_size = params.out_width_gap;
        config.flags = params.flags;
        config.is_texture_copy.Assign(1);
        QueueMemoryTransfer(config);
        break;
    }
    case CommandId::CacheFlush: {
        // Rasterizer flushing handled elsewhere in CPU read/write and other GPU handlers
        // Use command.cache_flush.regions to implement this handler
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown command {:#08X}", command.id.Value());
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    const PAddr phys_address_left = VirtualToPhysicalAddress(info.address_left);
    const PAddr phys_address_right = VirtualToPhysicalAddress(info.address_right);
    const auto info_copy = info;

    auto& framebuffer = impl->pica.regs.framebuffer_config[screen_id];
    if (info_copy.active_fb == 0) {
        framebuffer.address_left1 = phys_address_left;
        framebuffer.address_right1 = phys_address_right;
    } else {
        framebuffer.address_left2 = phys_address_left;
        framebuffer.address_right2 = phys_address_right;
    }
    framebuffer.stride = info_copy.stride;
    framebuffer.format = info_copy.format;
    framebuffer.active_fb = info_copy.shown_fb;

    // Notify debugger about the buffer swap.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::BufferSwapped, nullptr);
    }

    if (screen_id == 0) {
        MicroProfileFlip();
        impl->system.perf_stats->EndGameFrame();
        right_eye_disabler->ReportEndFrame();
    }
}

void GPU::SetColorFill(const Pica::ColorFill& fill) {
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
}

u32 GPU::ReadReg(VAddr addr) {
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        return impl->pica.regs_lcd[index];
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        return impl->pica.regs.reg_array[index];
    }
    default:
        UNREACHABLE_MSG("Read from unknown GPU address {:#08X}", addr);
    }
}

void GPU::WriteReg(VAddr addr, u32 data) {
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        impl->pica.regs_lcd[index] = data;
        break;
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);

        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        ExecWriteReg(index, data);
        break;
    }
    default:
        UNREACHABLE_MSG("Write to unknown GPU address {:#08X}", addr);
    }
}

void GPU::ExecWriteReg(u32 index, u32 data) {
    impl->pica.regs.reg_array[index] = data;

    // Handle registers that trigger GPU actions
    switch (index) {
    case GPU_REG_INDEX(memory_fill_config[0].trigger): {
        const auto& config = impl->pica.regs.memory_fill_config[0];
        if (config.trigger) {
            QueueMemoryFill(0, config.address_start, config.address_end, config.value_32bit,
                            config.control, 0);
        }
        break;
    }
    case GPU_REG_INDEX(memory_fill_config[1].trigger): {
        const auto& config = impl->pica.regs.memory_fill_config[1];
        if (config.trigger) {
            QueueMemoryFill(1, config.address_start, config.address_end, config.value_32bit,
                            config.control, 1);
        }
        break;
    }
    case GPU_REG_INDEX(display_transfer_config.trigger): {
        if (impl->pica.regs.display_transfer_config.trigger.Value()) {
            QueueMemoryTransfer(impl->pica.regs.display_transfer_config);
        }
        break;
    }
    case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[0]):
        ExecSubmitCmdList(0);
        break;
    case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[1]):
        ExecSubmitCmdList(1);
        break;
    default:
        break;
    }
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

Pica::PicaCore& GPU::PicaCore() {
    return impl->pica;
}

const Pica::PicaCore& GPU::PicaCore() const {
    return impl->pica;
}

GraphicsDebugger& GPU::Debugger() {
    return impl->gpu_debugger;
}

void GPU::ApplyPerProgramSettings(u64 program_ID) {
    auto hack = Common::Hacks::hack_manager.GetHack(
        Common::Hacks::HackType::ACCURATE_MULTIPLICATION, program_ID);
    bool use_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    if (hack) {
        switch (hack->mode) {
        case Common::Hacks::HackAllowMode::DISALLOW:
            use_accurate_mul = false;
            break;
        case Common::Hacks::HackAllowMode::FORCE:
            use_accurate_mul = true;
            break;
        case Common::Hacks::HackAllowMode::ALLOW:
        default:
            break;
        }
    }
    impl->gpu_thread.RunSync(
        [this, use_accurate_mul] { impl->rasterizer->SetAccurateMul(use_accurate_mul); });
}

void GPU::ExecSubmitCmdList(u32 index) {
    // Check if a command list was triggered.
    auto& config = impl->pica.regs.internal.pipeline.command_buffer;
    if (!config.trigger[index]) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_CmdlistProcessing);

    // Forward command list processing to the PICA core.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);
    impl->pica.ProcessCmdList(addr, size,
                              !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size));
    config.trigger[index] = 0;
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
    // Signal to GSP that GPU interrupt has occurred
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0, 0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1, 0);

    // Whole-frame frameskip, decided here before any of the next frame's work is queued: with a
    // present backlog, the coming frame ships state but no triangles and no present. Frames that
    // do render are always complete.
    const bool threaded = impl->gpu_thread.IsRunning();
    // Whole-frame draw skipping is an emergency valve only, at a depth no lightweight scene ever
    // reaches: screens that are drawn once and then merely presented (boot dialogs, menus between
    // redraws) would otherwise lose their only drawing to a skip and stay black forever. Heavy
    // continuously-redrawn scenes are the ones that push the queue this deep, and they repaint
    // every frame, so dropping their draws is safe. Presents self-regulate by coalescing.
    // A heavy scene can hold the queue over the threshold indefinitely; without a floor the
    // display freezes on the last pre-backlog frame for as long as the fight lasts. Intervene
    // only on real starvation — no present has retired for 60 vblanks — because forcing frames
    // merely for skipping feeds the backlog and locks healthy scenes into it.
    // Skip now runs on snapshot backpressure: the draw-arena swapchain has two frame slots the
    // renderer consumes, and a vblank that finds both still owned by unfinished render work
    // skips its skippable draws.
    impl->shipping_rasterizer->RotateArenaSlots();
    // Frameskip also serves the software renderer, which is permanently behind on 3D scenes:
    // the guest then runs at full speed with a low present rate. The freezes originally blamed
    // on this were the early-ack races (fixed by retirement-ordered interrupts and fill
    // completion) and the boundary-starved un-skip (fixed by the vblank timeout below).
    // Two signals that the renderer is behind. Arena backpressure catches frames piling up in
    // the draw arenas; queued presents catch the case it misses entirely - with the emulation
    // thread no longer waiting on the render thread, the guest runs ahead into the queue while
    // the arenas keep recycling in time. That queue reached thousands of ops deep on an
    // ordinary title screen, five seconds of work between the guest and the picture, and the
    // guest looked frozen for that long after every input while the counters still read 100%
    // speed. Skipping keeps it bounded by dropping draws, which the renderer was never going to
    // show in time; stalling the guest instead only moves the wait.
    const u32 frames_behind = impl->gpu_thread.PendingSwaps();
    const u64 queue_backlog_us = impl->gpu_thread.EstimatedBacklogUs();
    // The time budget and the queue's own op ceiling have to agree, or the brake fires first
    // and stalls the guest where a skip was meant to spare it. The budget is depth times the
    // average op cost, so a scene of many cheap ops reaches 512 ops long before 120 ms of
    // estimated work: Smash's victory screen ships about 380 ops a frame (188 scenes of one
    // or two draws, ping-ponging between two small targets), hit the ceiling twice a second
    // and held the emulation thread at 70% speed with its core idle (2026-09-09). Skipping at
    // half the ceiling leaves the brake as what it was meant to be, the backstop for a render
    // thread that has stopped altogether.
    const u32 queue_depth = impl->gpu_thread.Depth();
    constexpr u32 QueueDepthSkip = GpuThread::MaxDepth / 2;
    bool skip_next = threaded && impl->frameskip_allowed &&
                     (impl->shipping_rasterizer->ArenaBackpressure() ||
                      frames_behind >= PresentBacklogSkip ||
                      queue_backlog_us >= QueueBacklogSkipUs ||
                      queue_depth >= QueueDepthSkip);
    // Duty-cycle the skip: under permanent backpressure (the software renderer's normal state
    // on 3D scenes) an unbounded skip freezes the top screen on its last rendered frame while
    // the game runs on invisibly. Forcing one rendered frame every 15 vblanks keeps ~4 fps of
    // real progress on screen; transient backpressure never accumulates 15 in a row, so the
    // GL path is unaffected.
    // The un-skip has to be held until a frame boundary consumes it. Vblanks decide, but the
    // flip is applied at the game's frame boundary, and a single vblank of pending un-skip
    // rarely coincides with one - the forced render was usually dropped, the screen stayed
    // frozen, and it was the two-second rescue below that eventually unfroze it by flipping
    // mid-frame. That mid-frame flip is the shear.
    if (skip_next) {
        if (impl->skip_forced_render) {
            skip_next = false;
            if (!impl->skip_current) {
                // A boundary applied it and this frame is being drawn; let the streak start
                // again so the next forced render is another 15 vblanks out.
                impl->skip_forced_render = false;
                impl->skip_streak_vblanks = 0;
            }
        } else if (++impl->skip_streak_vblanks >= 15 && queue_backlog_us < QueueBacklogSkipUs &&
                   queue_depth < QueueDepthSkip / 2) {
            // The queue gate is what keeps the floor honest. Fifteen vblanks promise a forced
            // frame every 250 ms, but on a scene the renderer needs longer than that to
            // retire, each forced frame arrives before the last one has drained: the queue
            // then grows without bound until the emergency brake stalls the emulation thread
            // for seconds (and any flush-forced drain costs the whole backlog). A forced
            // frame into a deep queue cannot reach the screen promptly anyway - it waits
            // behind the very backlog it deepens. So under overload the streak simply runs
            // past 15 and the force fires on the first vblank that finds the renderer caught
            // up; the display advances at whatever rate the renderer can truly sustain, and
            // the queue stays bounded near the backlog budget plus one frame's draws.
            impl->skip_forced_render = true;
            skip_next = false;
        }
    } else {
        impl->skip_streak_vblanks = 0;
        impl->skip_forced_render = false;
    }
    // One vblank is one guest frame's worth of drawing, which is the granularity the skip's
    // repeat test needs: a target drawn into on both sides of a boundary is part of the
    // per-frame composite, one drawn into within a single frame only is not.
    if (impl->shipping_rasterizer) {
        impl->shipping_rasterizer->NoteFrameBoundary();
    }
    const u64 presented_now = impl->presented_frames.load(std::memory_order_relaxed);
    if (presented_now != impl->floor_last_presented) {
        impl->floor_last_presented = presented_now;
        impl->floor_starved_vblanks = 0;
    } else if (skip_next && ++impl->floor_starved_vblanks >= 60) {
        skip_next = false;
        impl->floor_starved_vblanks = 0;
        // A full second without a present means the boundary-based flip below is not coming:
        // the guest is not emitting top-sized display transfers in this phase (menus can run
        // entirely on texture copies), so waiting for one keeps the screen frozen forever —
        // the software renderer hit exactly that after its slow boot armed the skip. Force the
        // flip here; a possible one-frame shear beats a permanent freeze.
        impl->skip_current = false;
        impl->skip_pending = false;
        impl->shipping_rasterizer->SetSkipFrame(false);
    }
    // The vblank only decides; the flip is applied at the game's frame boundary (the top
    // display transfer), because the guest's draw stream is not aligned to vblanks and flipping
    // mid-frame ships half a frame — presented frames were sheared mixes of skipped and
    // rendered draws exactly at backpressure transitions.
    impl->skip_pending = skip_next;
    // The flip below normally waits for a top-sized display transfer so one frame's draws all
    // share a skip state. Phases that compose entirely from texture copies never emit one: the
    // un-skip then stays pending forever while presents keep showing the last drawn frame (the
    // software renderer sat on the boot screen this way, with presents flowing). If the
    // *un*-skip has been pending for two seconds of vblanks with no boundary arriving (the
    // counter resets whenever one does), force it — rendering again after a possible one-frame
    // shear beats a frozen screen. Arming the skip is never forced: waiting for the boundary
    // there only costs performance, not correctness.
    if (!impl->skip_pending && impl->skip_current) {
        if (++impl->skip_flip_pending_vblanks >= 120) {
            impl->skip_current = false;
            impl->shipping_rasterizer->SetSkipFrame(false);
            impl->skip_flip_pending_vblanks = 0;
        }
    } else {
        impl->skip_flip_pending_vblanks = 0;
    }
    // Arming waits for the same boundary, and the note above says waiting there costs only
    // performance. On a phase that emits no top-sized display transfer at all it costs all of
    // it: Smash's victory screen composes through texture copies, so the skip stayed pending
    // for as long as the screen lasted, every frame's triangles were drawn in full, and only
    // the present was dropped - the entire cost of rendering with none of its saving. What
    // reached the screen was the starvation floor below, one frame a second (console,
    // 2026-09-09). So arming gets a timeout of its own, and a short one: the frame a forced
    // arming splits is by definition the frame being skipped, so its shear is never seen,
    // where releasing the skip mid-frame does show one.
    if (impl->skip_pending && !impl->skip_current) {
        if (++impl->skip_arm_pending_vblanks >= 8) {
            impl->skip_current = true;
            impl->shipping_rasterizer->SetSkipFrame(true);
            impl->skip_arm_pending_vblanks = 0;
        }
    } else {
        impl->skip_arm_pending_vblanks = 0;
    }
    skip_next = impl->skip_current;

    // AZAHAR_PRESENT_LOG=1: one line per vblank with what became of its present, in
    // microseconds of wall time - queued, executed, swapped - and the completion sequence of
    // the top pane's picture, so a recording's hitches can be matched to the trace.
    static const bool log_presents = std::getenv("AZAHAR_PRESENT_LOG") != nullptr;
    const u64 vblank_serial = ++impl->vblank_serial;
    if (skip_next) {
        if (log_presents) {
            LOG_INFO(HW_GPU, "present vb {} skipped by frameskip at {} us, queue {}", vblank_serial,
                     Common::PipelineStats::NowUs(), impl->gpu_thread.Depth());
        }
        impl->gpu_thread.NoteSkippedFrame();
    } else {
        const u64 queued_us = log_presents ? Common::PipelineStats::NowUs() : 0;
        const u32 queue_depth = impl->gpu_thread.Depth();
        // The presentation registers (framebuffer pointers, LCD fill) travel with the op, since
        // the renderer reads them from the mirror.
        std::array<Pica::FramebufferConfig, 2> fb_config{impl->pica.regs.framebuffer_config[0],
                                                         impl->pica.regs.framebuffer_config[1]};
        const auto lcd_regs = impl->pica.regs_lcd;
        // CPU-written framebuffers are copied now, at the VBlank the real display would scan
        // them out, and the present samples the copy (see Impl::DisplaySnapshotSlot).
        std::array<DisplaySnapshotSlot*, 2> snapshots{};
        if (threaded) {
            snapshots[0] = TakeDisplaySnapshot(0);
            snapshots[1] = TakeDisplaySnapshot(1);
        }
        impl->gpu_thread.Enqueue(
            {.run =
                 [this, threaded, fb_config, lcd_regs, snapshots, vblank_serial, queued_us,
                  queue_depth] {
                     const u64 exec_us = log_presents ? Common::PipelineStats::NowUs() : 0;
                     impl->pica_mirror.regs.framebuffer_config[0] = fb_config[0];
                     impl->pica_mirror.regs.framebuffer_config[1] = fb_config[1];
                     impl->pica_mirror.regs_lcd = lcd_regs;
                     impl->display_snapshot = {snapshots[0], snapshots[1]};
                     // Presents are the flush points of the zero-copy vertex ring: fence what
                     // has been drawn and retire what completed.
                     impl->rasterizer->RingFenceTick();
                     impl->renderer->SwapBuffers();
                     if (threaded) {
#ifdef ENABLE_OPENGL
                         // Under the software renderer there is no GL context and the glad
                         // pointers are null; the bind belongs to the GL present path only.
                         if (Settings::GetWorkingGraphicsAPI() == Settings::GraphicsAPI::OpenGL) {
                             glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                         }
#endif
                         impl->renderer->TryPresent(0, false);
                         if (impl->emu_window) {
                             impl->emu_window->SwapBuffers();
                         }
                     }
                     impl->presented_frames.fetch_add(1, std::memory_order_relaxed);
                     if (log_presents) {
                         LOG_INFO(HW_GPU,
                                  "present vb {} queued {} us (queue {}) executed {} us swapped {} "
                                  "us, top picture seq {} at {:#x}",
                                  vblank_serial, queued_us, queue_depth, exec_us,
                                  Common::PipelineStats::NowUs(), impl->pane_last_shown[0].seq,
                                  impl->pane_last_shown[0].addr);
                     }
                     impl->display_snapshot = {};
                     for (auto* slot : snapshots) {
                         if (slot != nullptr) {
                             slot->in_use.store(0, std::memory_order_release);
                         }
                     }
                     if (threaded) {
                         impl->render_tokens.fetch_add(1, std::memory_order_acq_rel);
                     }
                 },
             .swap_marker = true});
    }

    // Reschedule recurrent event
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);

    // Pace the guest to real time unless unlimited speed was chosen; frameskip and the render
    // tokens handle everything the renderer cannot keep up with either way.
    if (threaded && !impl->unlimited_speed) {
        impl->system.frame_limiter.DoFrameLimiting(impl->timing.GetGlobalTimeUs());
    }
}

void GPU::RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window) {
    impl->gpu_thread.RunSync([this, &emu_window, secondary_window] {
        RecreateRendererLocked(emu_window, secondary_window);
    });
}

void GPU::RecreateRendererLocked(Frontend::EmuWindow& emu_window,
                                 Frontend::EmuWindow* secondary_window) {
    // Reset the renderer (this will destroy OpenGL resources)
    impl->renderer.reset();

    // Create a new renderer against the render thread's mirror state
    impl->renderer =
        VideoCore::CreateRenderer(emu_window, secondary_window, impl->pica_mirror, impl->system);
    impl->renderer->SetOwnerGPU(this);
    impl->rasterizer = impl->renderer->Rasterizer();

    // PICA keeps driving the shipping facade; only the facade's target changes
    impl->shipping_rasterizer->SetRealRasterizer(impl->rasterizer);
    // The software renderer's rendered output lands in guest RAM on the render thread, and its
    // fills and transfers run on the CPU against guest RAM too - GL absorbs both into the host
    // GPU and never writes there. So its completion interrupts ride the render-thread queue:
    // signalling at ship time hands the guest a licence to reuse memory a queued write has not
    // reached yet, which corrupted titles right after their intro.
    if (Settings::GetWorkingGraphicsAPI() == Settings::GraphicsAPI::Software) {
        impl->shipping_rasterizer->SetDeferredInterrupts(
            [this](Service::GSP::InterruptId id, u64 delay_ns) { PostInterrupt(id, delay_ns); });
        impl->shipping_rasterizer->SetEarlyInterruptPoster(
            [this](Service::GSP::InterruptId id, u64 delay_ns) {
                PostInterruptDeferred(id, delay_ns);
            });
    }
    impl->pica.BindRasterizer(impl->shipping_rasterizer.get());
    // Shipped draws execute against the mirror, whose triangles go straight to the real
    // rasterizer on the render thread.
    impl->pica_mirror.BindRasterizer(impl->rasterizer);
    impl->pica_mirror.dirty_regs.SetAllDirty();

    // Update the sw_blitter with the new rasterizer
    impl->sw_blitter = std::make_unique<SwRenderer::SwBlitter>(impl->memory, impl->rasterizer);

    if (impl->gpu_thread.IsRunning()) {
        impl->renderer->SetThreadedPresentation(true);
    }

    // Re-apply per-game configuration and reload disk shader cache
    u64 program_id{};
    impl->system.GetAppLoader().ReadProgramId(program_id);
    ApplyPerProgramSettings(program_id);
    if (Settings::values.use_disk_shader_cache) {
        impl->renderer->Rasterizer()->LoadDefaultDiskResources(false, nullptr);
    }

    // Mark ALL GPU registers as dirty so current state gets uploaded to new renderer
    impl->pica.dirty_regs.SetAllDirty();

    // Also mark shader setups as dirty so uniforms get re-uploaded and
    // stale pointers to the old rasterizer's JIT cache are cleared.
    impl->pica.vs_setup.uniforms_dirty = true;
    impl->pica.vs_setup.cached_shader = nullptr;
    impl->pica.gs_setup.uniforms_dirty = true;
    impl->pica.gs_setup.cached_shader = nullptr;

    // Mark all cached LUT/table state in pica as dirty
    impl->pica.lighting.lut_dirty = impl->pica.lighting.LutAllDirty;
    impl->pica.fog.lut_dirty = true;
    impl->pica.proctex.table_dirty = impl->pica.proctex.TableAllDirty;
}

void GPU::ReleaseRenderer() {
    impl->gpu_thread.RunSync([this] {
        // Just reset the renderer to release OpenGL resources
        // Don't null out rasterizer pointer as it will become dangling
        impl->renderer.reset();
        impl->sw_blitter.reset();
    });
    LOG_INFO(HW_GPU, "Renderer released for context destroy");
}

template <class Archive>
void GPU::serialize(Archive& ar, const u32 file_version) {
    // PICA state must not be mid-mutation on the render thread while it is written out or
    // replaced wholesale.
    impl->gpu_thread.Drain();
    ar & impl->pica;
}

SERIALIZE_IMPL(GPU)

} // namespace VideoCore
