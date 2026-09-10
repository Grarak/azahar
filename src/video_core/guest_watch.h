// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <cstdlib>
#include "common/common_types.h"
#include "common/logging/log.h"
#include "core/memory.h"

/// A watch on one guest physical word for the render thread's writers: every fill, transfer
/// and draw reads it before and after running, and the op that changes it is logged with what
/// it was. Armed by AZAHAR_WATCH_PADDR (hex); costs two loads per op when armed, nothing when
/// not. Diagnostic for a guest object being clobbered by the software renderer's guest-memory
/// writes (2026-09-01: a vtable pointer reading FE40FE40).
namespace VideoCore::GuestWatch {

inline const volatile u32* watch_ptr = nullptr;
inline u32 watch_paddr = 0;

inline void Init(Memory::MemorySystem& memory) {
    const char* env = std::getenv("AZAHAR_WATCH_PADDR");
    if (!env) {
        return;
    }
    watch_paddr = static_cast<u32>(std::strtoul(env, nullptr, 16));
    watch_ptr = reinterpret_cast<const volatile u32*>(memory.GetPhysicalPointer(watch_paddr));
    LOG_CRITICAL(HW_GPU, "guest watch armed on paddr {:08X} ({})", watch_paddr,
                 watch_ptr ? "mapped" : "NOT MAPPED");
}

inline u32 Read() {
    return watch_ptr ? *watch_ptr : 0;
}

/// Logs when the watched word changed across an op. `what` names the op, a/b/c describe it.
inline void Check(u32 before, const char* what, u32 a, u32 b, u32 c) {
    if (!watch_ptr) {
        return;
    }
    const u32 after = *watch_ptr;
    if (after != before) {
        LOG_CRITICAL(HW_GPU, "guest watch {:08X}: {:08X} -> {:08X} by {} ({:08X} {:08X} {:08X})",
                     watch_paddr, before, after, what, a, b, c);
    }
}

} // namespace VideoCore::GuestWatch
