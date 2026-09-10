// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#ifdef __vita__

#include <psp2/kernel/threadmgr.h>

#include "common/wake_flag.h"

namespace Common {

WakeFlag::WakeFlag() {
    evid = sceKernelCreateEventFlag("azahar-wake", SCE_EVENT_WAITMULTIPLE, 0, nullptr);
}

WakeFlag::~WakeFlag() {
    if (evid >= 0) {
        sceKernelDeleteEventFlag(evid);
    }
}

void WakeFlag::Raise() {
    sceKernelSetEventFlag(evid, 1);
}

void WakeFlag::RaiseAll() {
    // One bit, and the kernel releases every thread the pattern satisfies, so there is no
    // wake-one/wake-all distinction to make here.
    sceKernelSetEventFlag(evid, 1);
}

void WakeFlag::Lower() {
    // Poll-with-clear rather than sceKernelClearEventFlag: it clears exactly the bit waited
    // on, and its "no match" return is the flag already being down, which is not an error.
    sceKernelPollEventFlag(evid, 1, SCE_EVENT_WAITOR | SCE_EVENT_WAITCLEAR_PAT, nullptr);
}

void WakeFlag::Wait() {
    // No clear on wake: Lower is the only thing that takes the mark down, and it runs under
    // the caller's lock so that a raise cannot be lost between the check and the wait.
    sceKernelWaitEventFlag(evid, 1, SCE_EVENT_WAITOR, nullptr, nullptr);
}

} // namespace Common

#endif // __vita__
