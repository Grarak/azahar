// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace Common {

/// Something built on another thread: done once MarkDone runs. Shaders and pipelines of the
/// hardware renderers derive from it, so a draw can ask whether its programs exist yet and
/// skip rather than stall (Vulkan and GXM).
struct AsyncHandle {
public:
    AsyncHandle(bool is_done_ = false) : is_done{is_done_} {}

    [[nodiscard]] bool IsDone() noexcept {
        return is_done.load(std::memory_order::relaxed);
    }

    void WaitDone() noexcept {
        std::unique_lock lock{mutex};
        condvar.wait(lock, [this] { return is_done.load(std::memory_order::relaxed); });
    }

    void MarkDone(bool done = true) noexcept {
        std::scoped_lock lock{mutex};
        is_done = done;
        condvar.notify_all();
    }

private:
    std::condition_variable condvar;
    std::mutex mutex;
    std::atomic_bool is_done{false};
};

} // namespace Common
