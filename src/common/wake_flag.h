// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#ifndef __vita__
#include <condition_variable>
#include <mutex>
#else
#include <cstdint>
#endif

namespace Common {

/// A sticky one-bit wakeup. Raise() marks it and wakes a waiter; the mark persists until
/// Lower() takes it away, so a raise can never be missed by a thread that has not started
/// waiting yet - which is the whole point, and what a condition variable cannot promise
/// without the waiter holding a lock across the wait.
///
/// The contract that makes it correct: a waiter Lowers only while holding the lock that also
/// guards whatever it is waiting for, and only after observing that there is nothing to do.
/// A producer publishes under that same lock and Raises afterwards, so its raise is always
/// ordered after the waiter's lower, and a raise left over from work already taken is
/// discarded rather than causing a spurious block.
///
/// On the Vita this is a kernel event flag. std::condition_variable there is
/// pthread-embedded's, whose wait is a poll loop - pte_osSemaphoreCancellablePend spins on
/// sceKernelWaitSema with a 500 us timeout, checking a cancellation event flag between
/// attempts - so a blocked waiter costs about two thousand syscalls a second, and the
/// semaphore bookkeeping under it has been seen to overflow (SCE_KERNEL_ERROR_SEMA_OVF,
/// which pte discards, losing the wakeup). A kernel event flag blocks properly and keeps no
/// such counts.
class WakeFlag {
public:
    WakeFlag();
    ~WakeFlag();

    WakeFlag(const WakeFlag&) = delete;
    WakeFlag& operator=(const WakeFlag&) = delete;

    /// Marks the flag and wakes one waiter.
    void Raise();
    /// Marks the flag and wakes every waiter.
    void RaiseAll();
    /// Clears the mark. Call it under the lock that guards the waited-for state.
    void Lower();
    /// Blocks until the flag is marked. Does not clear it.
    void Wait();

private:
#ifdef __vita__
    std::int32_t evid;
#else
    std::mutex mutex;
    std::condition_variable condvar;
    bool raised{false};
#endif
};

#ifndef __vita__

inline WakeFlag::WakeFlag() = default;
inline WakeFlag::~WakeFlag() = default;

inline void WakeFlag::Raise() {
    {
        std::lock_guard lock{mutex};
        raised = true;
    }
    condvar.notify_one();
}

inline void WakeFlag::RaiseAll() {
    {
        std::lock_guard lock{mutex};
        raised = true;
    }
    condvar.notify_all();
}

inline void WakeFlag::Lower() {
    std::lock_guard lock{mutex};
    raised = false;
}

inline void WakeFlag::Wait() {
    std::unique_lock lock{mutex};
    condvar.wait(lock, [this] { return raised; });
}

#endif

} // namespace Common
