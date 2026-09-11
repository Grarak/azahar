// Copyright Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include "common/common_types.h"

#ifdef __vita__
#include <psp2/kernel/processmgr.h>
#endif

namespace Common::PipelineStats {

/**
 * Where the emulation thread's wall time goes when it is not running the guest.
 *
 * The software renderer is a pipeline of three threads and the emulation thread waits on it in
 * two places: the drains behind RunSync, and - one level down, inside the ops those drains are
 * waiting for - the rasterizer's stripe barrier. Both were invisible in any profile, because a
 * thread asleep on a condition variable spends no cycles and shows up nowhere. These counters
 * are the only instrument that sees them, so they are built on every platform rather than only
 * on the console: the Linux tier exists to rehearse this design and cannot rehearse what it
 * cannot measure.
 *
 * Every counter accumulates until something drains it with exchange(0); a frontend reading them
 * once a second turns them into a per-second breakdown.
 */

/// Microseconds on a clock that does not jump. std::chrono::steady_clock is not usable for this
/// on the console - it read 1178 ms inside a 1000 ms window - so the kernel's process timer,
/// which is microseconds by definition, stands in there.
[[nodiscard]] inline u64 NowUs() {
#ifdef __vita__
    return sceKernelGetProcessTimeWide();
#else
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
#endif
}

/// Ops the render thread retired, and the wall time spent inside their bodies.
inline std::atomic<u32> gpu_ops{0};
inline std::atomic<u64> gpu_busy_us{0};
/// The subset of that time spent in swap-marker ops (a whole frame's present).
inline std::atomic<u64> gpu_swap_us{0};
/// Surface readbacks the guest CPU forced: the count and the kilobytes handed back. A read of
/// a render target drains the GPU before it can answer, so a title that does this every frame
/// pays for it whether or not the pixels are new.
inline std::atomic<u32> cache_downloads{0};
inline std::atomic<u64> cache_download_kb{0};
/// Readbacks answered from the previous one instead of draining the GPU for fresh pixels.
inline std::atomic<u32> cache_stale_reads{0};
/// Guest display transfers and texture copies, split by whether the rasterizer took them or
/// the software blitter did. A software one reads the source through the CPU, which flushes
/// whatever surface owns it: that is where a readback storm comes from.
inline std::atomic<u32> xfer_accel{0};
inline std::atomic<u32> xfer_soft{0};
inline std::atomic<u32> copy_accel{0};
inline std::atomic<u32> copy_soft{0};

/// Draws that asked the cache for a framebuffer, and the subset of those that changed it. A
/// change of render target is a register write on the PICA and free there, but it is what the
/// Vita's renderer has to turn into a new GXM scene, and a scene costs a tile store and a
/// reload however few draws it holds. The guest decides this, so the count is the same number
/// on both tiers: what the pi5 reports here is what the console will be asked to do.
inline std::atomic<u32> fb_binds{0};
inline std::atomic<u32> fb_switches{0};

/// Times the queue hit its depth ceiling and made the emulation thread wait for it to drain,
/// and the wall time those waits cost it. This is the emulator throttling itself to the render
/// thread's rate: while it is braking the guest does not run, so the core looks idle and the
/// speed falls by exactly this much.
inline std::atomic<u32> gpu_brakes{0};

/**
 * Where the render thread is right now, for the case where it stops being anywhere.
 *
 * A hang on the console produces no output at all from the thread that hung, but the
 * emulation thread keeps printing its per-second line for several seconds afterwards -
 * until the queue reaches GpuThread::MaxDepth and the brake parks it too. That window is
 * the only chance to learn where the render thread went, so it leaves a breadcrumb: a
 * pointer to a string literal and the time it was stored. The frontend prints both, and a
 * phase that stops advancing while its age climbs names the call that never returned.
 *
 * The store is one relaxed word, and the call sites are compiled under VITA_RENDER_TRACE,
 * which the Vita build has on by default.
 */
inline std::atomic<const char*> render_phase{"none"};
inline std::atomic<u64> render_phase_us{0};
/// Three numbers the phase may carry, printed beside it. For a wait: what it is waiting for,
/// what has been submitted, and what the GPU has finished - which is the whole question when a
/// wait never returns.
inline std::atomic<u64> render_phase_a{0};
inline std::atomic<u64> render_phase_b{0};
inline std::atomic<u64> render_phase_c{0};

inline void SetRenderPhase(const char* name) {
    render_phase.store(name, std::memory_order_relaxed);
    render_phase_us.store(NowUs(), std::memory_order_relaxed);
}

inline void SetRenderPhase(const char* name, u64 a, u64 b = 0, u64 c = 0) {
    render_phase_a.store(a, std::memory_order_relaxed);
    render_phase_b.store(b, std::memory_order_relaxed);
    render_phase_c.store(c, std::memory_order_relaxed);
    SetRenderPhase(name);
}

/// The name alone, for a site hot enough that the clock read would count (a draw): the age
/// printed beside it is then that of the last timed phase.
inline void SetRenderPhaseName(const char* name) {
    render_phase.store(name, std::memory_order_relaxed);
}

/// What the frontend adds to the stuck-queue log (GpuThread::Enqueue): a line of its own
/// with what only it can read, such as the kernel module's cross-core counters. Called from
/// the emulation thread, once a second while the render thread does not drain.
inline std::atomic<void (*)()> stuck_report{nullptr};
/// The renderer's own addition to the same log: the last scenes it submitted, against the
/// GPU's completed epoch. Read from the emulation thread while the render thread is parked.
inline std::atomic<void (*)()> scene_report{nullptr};

#ifdef VITA_RENDER_TRACE
#define GXM_PHASE(...) ::Common::PipelineStats::SetRenderPhase(__VA_ARGS__)
#define GXM_PHASE_NAME(name) ::Common::PipelineStats::SetRenderPhaseName(name)
#else
#define GXM_PHASE(...) ((void)0)
#define GXM_PHASE_NAME(name) ((void)0)
#endif

/// Frames actually put on the screen. The frontend's "game fps" counts frames the guest
/// finished, which is what the emulated console produced, not what the display showed - the
/// two diverge whenever presents are being dropped, and only this one tracks the latter.
inline std::atomic<u32> presents{0};
/// VBlank snapshots taken of CPU-written framebuffers (GPU::DisplaySnapshot).
inline std::atomic<u32> display_snapshots{0};

/// Wall time spent inside the frontend's present call (GL swap, or the software path's
/// texture upload + copy + present). When the display connection itself is the bottleneck -
/// vsync against a slow display, or an X server on the far side of an ssh tunnel - this is
/// where the render thread blocks, and this number approaches a full second per second.
inline std::atomic<u64> present_us{0};

/// The emulation thread waiting for the render thread's queue to empty (every RunSync).
inline std::atomic<u64> drain_us{0};
inline std::atomic<u32> drain_count{0};

/// How long a completion interrupt spends queued: from the emulation thread enqueueing it
/// behind its fill or transfer, to the render thread actually posting it. This is what the
/// guest waits out when the write it is waiting for rides the render-thread queue.
inline std::atomic<u64> interrupt_lag_us{0};
inline std::atomic<u32> interrupt_count{0};

/// Pixels the software rasterizer actually walked, after the triangle edges have clipped each
/// scanline - not the bounding span, which counts however diagonal the geometry happens to be. Divided into
/// the render thread's busy time this gives nanoseconds per fragment, which is what the
/// rasterizer's speed actually is and does not move with whatever the game happens to be
/// drawing. Wall-clock frame rate on a demo that wanders between light and heavy scenes varies
/// by a third run to run, which is wider than most changes worth making.
inline std::atomic<u64> fragments{0};

/// The same two totals, never drained, so a whole run can be reduced to one figure. A
/// per-second sample covers whatever the game was drawing that second; summing the run removes
/// that, and a fixed run from boot then repeats to well under a percent.
inline std::atomic<u64> fragments_total{0};
inline std::atomic<u64> gpu_busy_total_us{0};

/// The render thread waiting for its stripe worker at the end of a batch (HelpAndWait).
inline std::atomic<u64> barrier_us{0};
inline std::atomic<u32> barrier_count{0};

} // namespace Common::PipelineStats
