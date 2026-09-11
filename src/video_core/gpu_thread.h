// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "common/named_thread.h"
#include "common/wake_flag.h"
#include "common/common_types.h"

namespace VideoCore {

/**
 * The GPU worker thread: owns the GL context and executes everything from the GSP command down.
 *
 * The emulation thread enqueues operations and never blocks on them.
 * When the queue holds too many unpresented frames the thread switches to skimming: an op's
 * `skim` alternative runs instead of `run`, which for command lists applies register writes but
 * suppresses draws, and for presents does nothing. That makes lag cost rendered frames rather
 * than guest speed.
 *
 * The rare operations that need an answer (register reads, cache flushes, savestates) run through
 * RunSync, which drains the queue first.
 */
class GpuThread {
public:
    struct Op {
        std::function<void()> run;
        /// Marks a present; used to measure how far behind the thread is.
        bool swap_marker = false;
    };

    GpuThread() = default;
    ~GpuThread();

    /// Starts the worker. `on_thread_start` runs first on the worker (context acquisition).
    void Start(std::function<void()> on_thread_start);

    /// Stops and joins the worker. `on_thread_exit` runs last on the worker (context release).
    void Stop(std::function<void()> on_thread_exit);

    bool IsRunning() const {
        return thread.joinable();
    }

    /// Whether the caller is the GPU worker thread.
    bool OnThread() const {
        return thread.joinable() && thread.IsCurrentThread();
    }

    /// Queues an operation. Never blocks (in lockstep mode, waits for it to retire).
    /// With no worker running, executes inline — the pre-thread and non-threaded configurations.
    void Enqueue(Op op);

    /// Runs `fn` on the GPU thread after everything queued before it, and waits for it.
    void RunSync(std::function<void()> fn);

    /// Waits until every queued operation has retired.
    void Drain();

    /// Blocks until the worker retires at least one op (returns at once when nothing is
    /// queued or there is no worker). For callers that must not let the queue grow by
    /// another item of their own until something ahead has finished.
    void WaitForRetire();

    [[nodiscard]] bool HasWorker() const {
        return thread.joinable();
    }

    /// Every enqueue behaves like RunSync; used by deterministic mode.
    void SetLockstep(bool enabled) {
        lockstep.store(enabled, std::memory_order_relaxed);
    }

    /// Presents currently queued.
    u32 PendingSwaps() const {
        return pending_swaps.load(std::memory_order_relaxed);
    }

    /// Operations currently queued; the emulation side skips whole frames when this backs up.
    u32 Depth() const {
        return depth.load(std::memory_order_relaxed);
    }

    /**
     * Roughly how long the queue would take to drain, in microseconds.
     *
     * Depth alone does not say how far behind the display is, because an op means different
     * amounts of work per backend: the software renderer ships a batch of rasterization per
     * op, while the GXM backend ships one draw call, and a guest frame is a couple of hundred
     * of those. A depth that is a fraction of a second for one is four milliseconds for the
     * other, so anything deciding to skip frames on a depth threshold reads one of them
     * wrong. This multiplies the depth by what ops have actually been costing.
     */
    u64 EstimatedBacklogUs() const {
        return static_cast<u64>(Depth()) * avg_op_us.load(std::memory_order_relaxed);
    }

    /// Records a frame the emulation side decided not to render at all.
    void NoteSkippedFrame() {
        skipped_frames.fetch_add(1, std::memory_order_relaxed);
    }

    /// Producer-side emergency brake: with no frame limiter the guest can outrun the render
    /// thread by an order of magnitude, and an unbounded queue eventually dies in allocation.
    /// Far above the frameskip threshold, so it only engages at pathological speeds - and
    /// when the render thread stops for good. On the console the heap sits within a
    /// megabyte of its ceiling in a Smash fight, and 4096 queued shipments (about 1 KB
    /// each) ran it out before the brake came on (2026-09-08: the queue at 3759, the heap
    /// full, the process gone with core 2 still taken, the console needed a hard reset).
    /// The brake now bounds the queue at a few hundred kilobytes: a stopped render thread
    /// stalls the game instead of killing the process.
    static constexpr u32 MaxDepth = 512;

    u64 SkippedFrames() const {
        return skipped_frames.load(std::memory_order_relaxed);
    }

private:
    void ThreadLoop(std::function<void()> on_thread_start);

    Common::NamedThread thread;
    std::mutex mutex;
    /// Raised when work arrives or a stop is requested, and when an op retires. Not condition
    /// variables: newlib's are built on kernel semaphores whose count climbs by one on every
    /// wait and is never drained, so after 32766 waits - nine minutes of a thread that sleeps
    /// whenever its queue empties - the post fails with SCE_KERNEL_ERROR_SEMA_OVF, the
    /// pthreads layer discards the error, and the wakeup is lost for good. A kernel event flag
    /// keeps no such count. See Common::WakeFlag.
    Common::WakeFlag enqueue_wake;
    Common::WakeFlag retire_wake;
    /// Arrival order. Presents coalesce at pop time: with more than one pending, older ones are
    /// dropped as skipped frames, so the thread never renders a frame nobody will see. Nothing
    /// guest-visible waits on this queue — every completion the guest can observe is raised on
    /// the emulation side — so ordering fairness is all that matters here.
    std::deque<Op> queue;
    bool stop_requested = false;
    bool idle = true; ///< Worker has nothing queued and nothing in flight

    /// Longest the screen may hold the same picture before a present runs whatever else is
    /// queued. Coalescing is a saving only while the display is actually being updated; past
    /// this it is a freeze, so the next present runs regardless of how many follow it.
    static constexpr u64 PresentStaleUs = 33'000;
    /// When the last present executed (GPU thread only).
    u64 last_present_us = 0;

    /// Exponential moving average of an op's cost, maintained by the worker.
    std::atomic<u64> avg_op_us{0};
    std::atomic<u32> pending_swaps{0};
    std::atomic<u32> depth{0};
    /// Callers parked in WaitForRetire; the worker signals retire_cv per op only while nonzero.
    std::atomic<u32> retire_waiters{0};
    std::atomic<bool> lockstep{false};
    std::atomic<u64> skipped_frames{0};
};

} // namespace VideoCore
