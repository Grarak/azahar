// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include "common/common_types.h"

namespace Core {

/// Why guest execution stopped.
enum class NativeExitReason : u32 {
    /// The guest executed an svc. `svc_immediate` holds the service number.
    Svc = 0,
    /// The preemption timer expired; the guest's timeslice is over.
    Preempted = 1,
    /// The guest took a memory fault. `fault_address` holds the address it touched.
    DataAbort = 2,
    /// The guest executed an instruction this host does not implement.
    UndefinedInstruction = 3,
    /// Execution was stopped from outside, by NativeStopExecution.
    Interrupted = 4,
};

/**
 * The guest's register file, in the layout the entry and exit stubs expect.
 *
 * The offsets are hard-coded in native_entry.S, so the static asserts in native_executor.cpp are
 * load-bearing rather than decorative.
 */
struct alignas(8) NativeGuestContext {
    /// r0-r15. Bit 0 of regs[15] carries the Thumb state, as for any interworking branch.
    std::array<u32, 16> regs{};
    u32 cpsr{};
    u32 fpscr{};
    u32 exit_reason{};
    /// Host stack pointer, stored on the way in so the exit stub can get back to it.
    u32 host_sp{};
    std::array<u64, 32> fpregs{};
};

/// Extra detail about the last exit, valid until the next call to NativeRunGuest.
struct NativeExitInfo {
    NativeExitReason reason{};
    u32 svc_immediate{};
    u32 fault_address{};
    /// Address of the instruction that faulted, for the fault reasons.
    u32 fault_pc{};
};

/**
 * Prepares the calling thread to execute guest code.
 *
 * Installs the signal handlers, the alternate signal stack and the seccomp filter that turns a
 * guest svc into a trap back to us rather than a host system call. Safe to call more than once; only
 * the first call per thread does anything.
 *
 * @param guest_base  Start of the address range guest code executes in.
 * @param guest_end   End of that range, exclusive.
 * @returns false if the host refused any of it, in which case guest code must not be run.
 */
bool NativePrepareThread(u32 guest_base, u32 guest_end);

/// Whether NativePrepareThread has succeeded on the calling thread.
bool NativeThreadIsPrepared();

/**
 * Executes guest code until it stops.
 *
 * @param ctx          Guest register file; updated in place with the state at the exit.
 * @param quantum_ns   How long the guest may run before being preempted. Zero means no preemption.
 * @param guest_tls    Value the guest expects to read from TPIDRURO. Installed for the duration of
 *                     guest execution and taken back out afterwards, since the host thread keeps
 *                     its own thread pointer in the same register.
 * @param out          Receives detail about why execution stopped.
 * @returns the elapsed wall-clock time in nanoseconds, which is what guest time is derived from.
 */
u64 NativeRunGuest(NativeGuestContext& ctx, u64 quantum_ns, u32 guest_tls, NativeExitInfo& out);

/**
 * Asks the guest to stop as soon as it can.
 *
 * If the guest is executing, it is interrupted; otherwise the next call to NativeRunGuest returns
 * immediately without entering it. The request stands until NativeClearStopRequest, so a caller
 * that is already between guest entries still has its request honoured.
 */
void NativeStopExecution();

/**
 * Withdraws any outstanding stop request.
 *
 * Call this when starting a fresh timeslice: the scheduler asks the core to stop immediately
 * before handing it its next slice, and that request is about the slice just finished.
 */
void NativeClearStopRequest();

/// Makes instructions written through another mapping visible to instruction fetch at [start, end).
void NativeFlushInstructionCache(void* start, std::size_t length);

} // namespace Core

namespace Memory {
class GuestWriteTracker;
}
namespace Core {
class NativeAddressSpaceMirror;
/// Routes guest stores into armed pages to the write tracker instead of treating them as
/// data aborts. Null pointers turn it off.
void NativeSetWriteTracking(const NativeAddressSpaceMirror* mirror,
                            Memory::GuestWriteTracker* tracker);
} // namespace Core
