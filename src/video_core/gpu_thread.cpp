// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/thread.h"
#include <cstdlib>
#include "video_core/gpu_thread.h"
#include "video_core/guest_watch.h"
#include "common/pipeline_stats.h"

namespace VideoCore {

MICROPROFILE_DEFINE(GPU_Thread_Op, "GPU", "GPU Thread Op", MP_RGB(255, 150, 60));

GpuThread::~GpuThread() {
    if (thread.joinable()) {
        Stop({});
    }
}

void GpuThread::Start(std::function<void()> on_thread_start) {
    ASSERT(!thread.joinable());
    stop_requested = false;
    thread = Common::NamedThread(Common::ThreadCfg{"GPU thread"},
                                 [this, start = std::move(on_thread_start)]() mutable {
                                     ThreadLoop(std::move(start));
                                 });
}

void GpuThread::Stop(std::function<void()> on_thread_exit) {
    if (!thread.joinable()) {
        return;
    }
    Drain();
    if (on_thread_exit) {
        RunSync(std::move(on_thread_exit));
    }
    {
        std::scoped_lock lock{mutex};
        stop_requested = true;
    }
    enqueue_wake.RaiseAll();
    thread.join();
}

void GpuThread::Enqueue(Op op) {
    if (!thread.joinable() || OnThread()) {
        // No worker, or already on it (an op enqueueing another op): run inline.
        if (op.run) {
            op.run();
        }
        return;
    }
    {
        std::unique_lock lock{mutex};
        if (depth.load(std::memory_order_relaxed) >= MaxDepth) {
            // Emergency brake only: wait for the worker to drain below the ceiling. Said
            // once a second while it lasts: a render thread that never drains is what the
            // log then has to explain (the phase breadcrumb, VITA_RENDER_TRACE builds).
            Common::PipelineStats::gpu_brakes.fetch_add(1, std::memory_order_relaxed);
            while (depth.load(std::memory_order_relaxed) >= MaxDepth / 2) {
                retire_wake.Lower();
                if (depth.load(std::memory_order_relaxed) < MaxDepth / 2) {
                    break;
                }
                lock.unlock();
                retire_wake.Wait();
                lock.lock();
            }
        }
        idle = false;
        if (op.swap_marker) {
            pending_swaps.fetch_add(1, std::memory_order_relaxed);
        }
        depth.fetch_add(1, std::memory_order_relaxed);
        queue.push_back(std::move(op));
    }
    enqueue_wake.Raise();

    if (lockstep.load(std::memory_order_relaxed)) {
        Drain();
    }
}

void GpuThread::RunSync(std::function<void()> fn) {
    if (!thread.joinable() || OnThread()) {
        // No worker (or already on it): degrade to inline execution.
        fn();
        return;
    }
    Enqueue({.run = std::move(fn)});
    Drain();
}

void GpuThread::WaitForRetire() {
    if (!thread.joinable() || OnThread()) {
        return;
    }
    std::unique_lock lock{mutex};
    if (idle) {
        return;
    }
    retire_waiters.fetch_add(1, std::memory_order_relaxed);
    retire_wake.Lower();
    lock.unlock();
    retire_wake.Wait();
    lock.lock();
    retire_waiters.fetch_sub(1, std::memory_order_relaxed);
}

void GpuThread::Drain() {
    if (!thread.joinable() || OnThread()) {
        return;
    }
    const u64 drain_start = Common::PipelineStats::NowUs();
    std::unique_lock lock{mutex};
    while (!idle) {
        retire_wake.Lower();
        if (idle) {
            break;
        }
        lock.unlock();
        retire_wake.Wait();
        lock.lock();
    }
    Common::PipelineStats::drain_us.fetch_add(Common::PipelineStats::NowUs() - drain_start,
                                              std::memory_order_relaxed);
    Common::PipelineStats::drain_count.fetch_add(1, std::memory_order_relaxed);
}

void GpuThread::ThreadLoop(std::function<void()> on_thread_start) {
    Common::SetCurrentThreadRole(Common::ThreadRole::Render);
    Common::SetCurrentThreadName("GpuThread");
    if (on_thread_start) {
        on_thread_start();
    }

    std::unique_lock lock{mutex};
    while (true) {
        if (queue.empty()) {
            idle = true;
            retire_wake.RaiseAll();
            GXM_PHASE("queue:idle");
            enqueue_wake.Lower();
            if (!stop_requested && queue.empty()) {
                lock.unlock();
                enqueue_wake.Wait();
                lock.lock();
            }
            if (stop_requested && queue.empty()) {
                break;
            }
            continue;
        }

        Op op = std::move(queue.front());
        queue.pop_front();
        const u32 new_depth = depth.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (new_depth == MaxDepth / 2) {
            retire_wake.RaiseAll();
        }
        // Coalesce presents, which this loop has always documented and never done. A present
        // captures the framebuffer registers when it is queued but reads the pixels when it
        // runs, so one that runs frames late shows whatever the guest has since put at that
        // address - and with the guest rotating buffers, the display walks backwards through
        // them. Measured on the pi5: 70 presented frames held 27 distinct pictures and 21 of
        // them were an *older* frame than the one before.
        //
        // Dropping every present but the newest costs nothing: the frames dropped here are
        // ones the display would have replaced before anyone saw them, and the survivor
        // carries its own, newer register snapshot.
        //
        // The drop is conditional on the display having been updated recently, and that
        // condition is the whole difference between coalescing and a frozen screen. "A newer
        // present is queued" is true of *every* present once the thread is permanently behind
        // - the guest queues one per vblank and the thread is thousands of ops back - so the
        // unconditional form dropped all of them and nothing ever reached the screen. Measured
        // on the pi5 with the software renderer and the guest uncapped: shown fps sat at 0 for
        // most of a run against a steady 10-18 without the drop, and the queue ran *deeper*
        // (3442 against 696), because the presents being skipped were also what paced it.
        const u64 now_us = Common::PipelineStats::NowUs();
        const bool display_is_current = now_us - last_present_us < PresentStaleUs;
        const bool superseded = op.swap_marker && display_is_current &&
                                pending_swaps.load(std::memory_order_relaxed) > 1;
        if (superseded) {
            static const bool log_presents = std::getenv("AZAHAR_PRESENT_LOG") != nullptr;
            if (log_presents) {
                LOG_INFO(HW_GPU, "present dropped as superseded at {} us, {} pending, depth {}",
                         now_us, pending_swaps.load(std::memory_order_relaxed),
                         depth.load(std::memory_order_relaxed));
            }
            pending_swaps.fetch_sub(1, std::memory_order_relaxed);
            skipped_frames.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        lock.unlock();

        {
            MICROPROFILE_SCOPE(GPU_Thread_Op);
            const u64 op_start = Common::PipelineStats::NowUs();
            const u32 watch = GuestWatch::Read();
            GXM_PHASE("op");
            if (op.run) {
                op.run();
            }
            GXM_PHASE("op:returned");
            GuestWatch::Check(watch, "op", op.swap_marker, new_depth, 0);
            const u64 op_us = Common::PipelineStats::NowUs() - op_start;
            // A slow-moving average of what an op costs, so a queue depth can be read as a
            // time rather than a count. Fifteen-sixteenths keeps it steady across the mix of
            // cheap draws and expensive presents without lagging a real change for long.
            const u64 previous = avg_op_us.load(std::memory_order_relaxed);
            avg_op_us.store((previous * 15 + op_us) / 16, std::memory_order_relaxed);
            Common::PipelineStats::gpu_busy_us.fetch_add(op_us, std::memory_order_relaxed);
            Common::PipelineStats::gpu_busy_total_us.fetch_add(op_us, std::memory_order_relaxed);
            if (op.swap_marker) {
                Common::PipelineStats::gpu_swap_us.fetch_add(op_us, std::memory_order_relaxed);
            }
            Common::PipelineStats::gpu_ops.fetch_add(1, std::memory_order_relaxed);
        }

        if (op.swap_marker) {
            pending_swaps.fetch_sub(1, std::memory_order_relaxed);
            last_present_us = Common::PipelineStats::NowUs();
        }

        lock.lock();
        if (retire_waiters.load(std::memory_order_relaxed) > 0) {
            retire_wake.RaiseAll();
        }
    }
}

} // namespace VideoCore
