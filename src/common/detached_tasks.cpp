// Copyright 2018 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/named_thread.h"
#include "common/assert.h"
#include "common/detached_tasks.h"
#include "common/thread.h"

namespace Common {

DetachedTasks* DetachedTasks::instance = nullptr;

DetachedTasks::DetachedTasks() {
    ASSERT(instance == nullptr);
    instance = this;
}

void DetachedTasks::WaitForAllTasks() {
    std::unique_lock lock{mutex};
    cv.wait(lock, [this]() { return count == 0; });
}

DetachedTasks::~DetachedTasks() {
    std::unique_lock lock{mutex};
    ASSERT(count == 0);
    instance = nullptr;
}

void DetachedTasks::AddTask(std::function<void()> task) {
    std::unique_lock lock{instance->mutex};
    ++instance->count;
    // Notification happens inside the callable rather than through
    // std::notify_all_at_thread_exit: the detached NamedThread is a raw SCE thread on the
    // Vita, and pthread exit hooks never run there.
    Common::NamedThread(Common::ThreadCfg{"detached task"}, [task{std::move(task)}]() {
        Common::SetCurrentThreadRole(Common::ThreadRole::Other);
        task();
        std::unique_lock lock{instance->mutex};
        --instance->count;
        instance->cv.notify_all();
    }).detach();
}

} // namespace Common
