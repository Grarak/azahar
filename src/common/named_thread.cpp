// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#ifdef __vita__

#include <cstdlib>

#include <psp2/kernel/threadmgr.h>
#include <vitasdk/utils.h>

#include "common/named_thread.h"

extern "C" {
// newlib's own thread-exit helpers (lib_a-threading.o): clean up the thread's lazily
// created reentrancy state, then sceKernelExitThread / sceKernelExitDeleteThread. Not
// declared in any public header.
[[noreturn]] void vita_exit_thread(int status);
[[noreturn]] void vita_exit_delete_thread(int status);
// pte's per-thread setup for the calling thread (pthread-embedded platform/vita): builds
// the pspThreadData record {entryPoint, argv, cancel event flag} that every cancellable
// wait dereferences. Returns PTE_OS_OK (0) on success.
int pte_osInit(void);
}

namespace Common {

namespace {

enum : std::uint32_t {
    StateRunning = 0,
    StateDetached = 1,
    StateFinished = 2,
};

} // namespace

struct NamedThread::Ctrl {
    std::function<void()> run;
    // Two-party handoff between the owner and the thread: whoever moves second - the
    // thread finishing, or the owner detaching - learns the other's decision and takes the
    // cleanup that falls to it.
    std::atomic<std::uint32_t> state{StateRunning};
};

int NamedThread::Trampoline(unsigned int args, void* argp) {
    auto* ctrl = *static_cast<NamedThread::Ctrl**>(argp);

    // Attach to newlib's pthread layer: pte's cancellable waits (under every std::mutex
    // and condition variable) dereference the pspThreadData record unconditionally, and
    // only pthread_create-made threads get one for free - a bare SCE thread crashed in
    // pte_osSemaphoreCancellablePend without it (2026-08-30). pte_osInit builds the record
    // for the calling thread, exactly as pte_osThreadCreate does on its first-ever call.
    // (pthread_self's implicit registration struct still leaks per thread, as it does for
    // any foreign thread; every NamedThread but the detached tasks lives process-long.)
    void** slot = static_cast<void**>(vitasdk_get_pthread_data(0));
    bool attached = false;
    if (slot != nullptr && *slot == nullptr) {
        attached = pte_osInit() == 0;
    }

    ctrl->run();

    if (attached && *slot != nullptr) {
        // Mirror of pte_osThreadDelete's cleanup: the record is
        // {entryPoint, argv, SceUID evid} (pthread-embedded platform/vita/vita_osal.c).
        void** record = static_cast<void**>(*slot);
        const int evid = static_cast<int>(reinterpret_cast<intptr_t>(record[2]));
        if (evid >= 0) {
            sceKernelDeleteEventFlag(evid);
        }
        *slot = nullptr;
        std::free(record);
    }

    const bool detached = ctrl->state.exchange(StateFinished) == StateDetached;
    if (detached) {
        delete ctrl; // the owner is gone; this side frees and self-deletes
        vita_exit_delete_thread(0);
    }
    vita_exit_thread(0); // the owner joins, deletes the thread, and frees ctrl
}

void NamedThread::Launch(const ThreadCfg& cfg, std::function<void()> f) {
    auto* c = new Ctrl{std::move(f)};
    uid = sceKernelCreateThread(cfg.name, &Trampoline, 0x10000100,
                                static_cast<SceSize>(cfg.stack_size), 0, 0, nullptr);
    if (uid < 0) {
        delete c;
        return;
    }
    ctrl = c;
    sceKernelStartThread(uid, sizeof(c), &c);
}

void NamedThread::join() {
    if (uid < 0) {
        return;
    }
    // Under Vita3K a worker joined right after its stop request (the AM scan worker at a
    // savestate load's shutdown, 2026-09-07) never returns from this wait although the
    // thread's callable has returned and it has entered newlib's exit; the same join on a
    // thread that exited earlier returns at once. Not reproduced on the console.
    sceKernelWaitThreadEnd(uid, nullptr, nullptr);
    sceKernelDeleteThread(uid);
    delete ctrl;
    uid = -1;
    ctrl = nullptr;
}

void NamedThread::detach() {
    if (uid < 0) {
        return;
    }
    if (ctrl->state.exchange(StateDetached) == StateFinished) {
        // The thread already finished as joinable; the cleanup falls here.
        join();
        return;
    }
    uid = -1;
    ctrl = nullptr;
}

bool NamedThread::IsCurrentThread() const noexcept {
    return uid >= 0 && sceKernelGetThreadId() == uid;
}

} // namespace Common

#endif // __vita__
