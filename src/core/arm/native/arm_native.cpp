// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include "core/arm/native/arm_native.h"

#include <algorithm>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "core/arm/native/native_mirror.h"
#include "core/core.h"
#include "video_core/gpu.h"
#include "core/core_timing.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/thread.h"
#include "core/memory.h"

namespace Core {

SvcRingArray g_svc_ring{};
std::atomic<u32> g_svc_ring_head{};

namespace {

/// Never hand the guest a slice longer than this, so a runaway guest cannot stall the frontend.
constexpr u64 MaxQuantumNs = 16'000'000;
/// Nor one so short that the timer costs more than the work it bounds.
constexpr u64 MinQuantumNs = 250'000;

u64 TicksToNanoseconds(s64 ticks) {
    if (ticks <= 0) {
        return 0;
    }
    return static_cast<u64>(ticks) * 1'000'000'000ull / BASE_CLOCK_RATE_ARM11;
}

u64 NanosecondsToTicks(u64 nanoseconds) {
    return nanoseconds * BASE_CLOCK_RATE_ARM11 / 1'000'000'000ull;
}

/// See SetNativeDeterministicTiming.
bool deterministic_timing = false;

/// Guest ticks charged for one svc when timing deterministically. Roughly what the round trip
/// costs in practice, but the exact figure matters far less than its being fixed.
constexpr u64 TicksPerSvc = 700;

/// Guest ticks charged when the guest is preempted rather than leaving through an svc.
constexpr u64 TicksPerPreemption = 100'000;

/// In deterministic mode the preemption timer is a backstop against a guest that never calls svc,
/// not a scheduling mechanism. If it fires, the run is no longer reproducible.
constexpr u64 DeterministicBackstopNs = 2'000'000'000;

} // Anonymous namespace

void SetNativeDeterministicTiming(bool enabled) {
    deterministic_timing = enabled;
}

bool NativeDeterministicTiming() {
    return deterministic_timing;
}

MICROPROFILE_DEFINE(ARM_Native, "ARM Native", "ARM Native", MP_RGB(64, 255, 128));

std::shared_ptr<NativeAddressSpaceMirror> ARM_Native::CreateMirror(
    Memory::MemorySystem& memory) {
    auto mirror = std::make_shared<NativeAddressSpaceMirror>(memory);
    if (!mirror->Reserve()) {
        return nullptr;
    }
    memory.SetMappingObserver(mirror.get());
    return mirror;
}

ARM_Native::ARM_Native(Core::System& system_, Memory::MemorySystem& memory_, u32 id,
                       std::shared_ptr<Core::Timing::Timer> timer_,
                       std::shared_ptr<NativeAddressSpaceMirror> mirror_)
    : ARM_Interface(id, timer_), system{system_}, memory{memory_}, svc_context{system_},
      mirror{std::move(mirror_)} {
    // Both emulated cores belong to the same process and so share one address space, and Azahar
    // timeslices them onto a single host thread. Every core can therefore execute natively against
    // the one mirror.
    usable = mirror != nullptr && mirror->IsUsable();
    // Guest write tracking (core/guest_write_tracker.h): pages the surface cache arms fault on
    // the guest's first store. Opt-in while it is being proven: AZAHAR_WRITE_TRACK=1.
    static const bool track_writes = std::getenv("AZAHAR_WRITE_TRACK") != nullptr;
    if (usable && track_writes) {
        auto& tracker = memory.WriteTracker();
        std::shared_ptr<NativeAddressSpaceMirror> mirror_ref = mirror;
        tracker.SetProtect([mirror_ref](const u8* host, std::size_t size, bool read_only) {
            mirror_ref->ProtectHostRange(host, size, read_only);
        });
        NativeSetWriteTracking(mirror.get(), &tracker);
        // AZAHAR_WRITE_TRACK_HOT=0 keeps faulting on pages written every frame (A/B).
        if (const char* hot = std::getenv("AZAHAR_WRITE_TRACK_HOT"); hot != nullptr) {
            tracker.SetHotPages(hot[0] != '0');
        }
        tracker.SetEnabled(true);
        LOG_INFO(Core_ARM11, "Native backend: guest write tracking on");
    }
}

ARM_Native::~ARM_Native() {
    // The last core to go takes the observer registration with it.
    if (mirror != nullptr && mirror.use_count() == 1) {
        memory.SetMappingObserver(nullptr);
    }
}

void ARM_Native::Run() {
    if (break_flag) [[unlikely]] {
        return;
    }
    MICROPROFILE_SCOPE(ARM_Native);

    // The scheduler calls PrepareReschedule immediately before handing out a new slice, so any
    // outstanding stop request refers to the slice that just ended. Honouring it here would mean
    // never executing anything at all.
    NativeClearStopRequest();

    const u64 quantum =
        deterministic_timing
            ? DeterministicBackstopNs
            : std::clamp(TicksToNanoseconds(timer->GetDowncount()), MinQuantumNs, MaxQuantumNs);

    // Keep going until the slice is used up or something wants the core back. An svc that returns
    // straight away must not end the whole timeslice, or a guest making frequent system calls would
    // make almost no progress per Run().
    while (RunSlice(quantum)) {
        if (timer->GetDowncount() <= 0 || break_flag) {
            break;
        }
    }
}

void ARM_Native::Step() {
    if (break_flag) [[unlikely]] {
        return;
    }
    // Without hardware single-stepping the closest thing available is the shortest slice the
    // preemption timer can express.
    NativeClearStopRequest();
    RunSlice(MinQuantumNs);
}

bool ARM_Native::RunSlice(u64 quantum_ns) {
    if (!usable) [[unlikely]] {
        // Nothing can be executed, and pretending otherwise would spin forever.
        system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised,
                         "Native ARM backend is not usable on this host");
        return false;
    }

    if (!thread_prepared) {
        thread_prepared = NativePrepareThread(NativeAddressSpaceMirror::GuestBase,
                                              NativeAddressSpaceMirror::GuestEnd);
        if (!thread_prepared) {
            system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised,
                             "Native ARM backend could not prepare the CPU thread");
            usable = false;
            return false;
        }
    }

    NativeExitInfo exit{};
    const u64 elapsed = NativeRunGuest(context, quantum_ns, cp15[CP15_THREAD_URO], exit);
    CreditTime(elapsed, exit.reason);

    switch (exit.reason) {
    case NativeExitReason::Svc: {
        // Observation ring of guest system calls, read through the debug port at a wedge. The
        // slot is filled before CallSVC and the stored r0 updated after it returns, so a poll
        // loop shows up as its repeating (svc, arguments, result) pattern. CallSVC can
        // reschedule, in which case the post-call register file belongs to the next thread and
        // the recorded result is marked untrustworthy.
        auto& rec = g_svc_ring[g_svc_ring_head.fetch_add(1, std::memory_order_relaxed) %
                               g_svc_ring.size()];
        const Kernel::Thread* pre_thread =
            system.Kernel().GetThreadManager(GetID()).GetCurrentThread();
        rec.tid = pre_thread ? pre_thread->thread_id : 0xFFFFFFFF;
        rec.svc = exit.svc_immediate;
        rec.r0 = context.regs[0];
        rec.r1 = context.regs[1];
        rec.r2 = context.regs[2];
        rec.r3 = context.regs[3];
        rec.pc = context.regs[15];
        rec.lr = context.regs[14];
        rec.result = 0xDEADDEAD;
        svc_context.CallSVC(exit.svc_immediate);
        const Kernel::Thread* post_thread =
            system.Kernel().GetThreadManager(GetID()).GetCurrentThread();
        rec.result = (post_thread == pre_thread) ? context.regs[0] : 0xEEEEEEEE;
        // An svc can reschedule, which replaces the register file wholesale.
        return !break_flag;
    }

    case NativeExitReason::Preempted:
        return false;

    case NativeExitReason::Interrupted:
        return false;

    case NativeExitReason::DataAbort:
    case NativeExitReason::UndefinedInstruction:
        ReportFault(exit);
        return false;
    }

    return false;
}

void ARM_Native::CreditTime(u64 elapsed_ns, NativeExitReason reason) {
    u64 ticks;
    if (deterministic_timing) {
        // A fixed charge per exit, so guest time depends only on what the guest did.
        if (reason == NativeExitReason::Preempted) {
            LOG_WARNING(Core_ARM11, "Preemption backstop fired in deterministic mode; this run is "
                                    "no longer reproducible");
            ticks = TicksPerPreemption;
        } else {
            ticks = TicksPerSvc;
        }
    } else {
        // Derived from host time, because native execution has no instruction count to derive it
        // from. The guest advances at real time and simply gets more work done per tick than the
        // hardware would have.
        ticks = NanosecondsToTicks(elapsed_ns);
    }

    if (ticks > 0) {
        timer->AddTicks(ticks);
    }
}

void ARM_Native::ReportFault(const NativeExitInfo& exit) {
    std::string registers;
    for (int i = 0; i < 16; i++) {
        registers += fmt::format("r{:02d} = {:08X}\n", i, context.regs[i]);
    }

    const char* kind = exit.reason == NativeExitReason::UndefinedInstruction
                           ? "undefined instruction"
                           : "data abort";
    const std::string message =
        fmt::format("{}Guest {} at PC {:08X}, touching {:08X}", registers, kind, exit.fault_pc,
                    exit.fault_address);
    LOG_CRITICAL(Core_ARM11, "{}", message);
    std::string code;
    for (u32 a = exit.fault_pc - 16; a <= exit.fault_pc + 12; a += 4) {
        code += fmt::format("{}{:08X}", a == exit.fault_pc ? " *" : " ", memory.Read32(a));
    }
    LOG_CRITICAL(Core_ARM11, "code around PC:{}\nrender-thread writes:\n{}", code,
                 system.GPU().DescribeRecentWrites(std::span<const u32>(context.regs),
                                                   exit.fault_address));
    system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised, message.c_str());
}

void ARM_Native::ClearInstructionCache() {
    // Guest code is written through the emulator's own view of the memory, so instruction fetch at
    // the guest addresses has to be told about it — but only over what is actually mapped.
    if (mirror) {
        mirror->FlushInstructionCache();
    }
}

void ARM_Native::InvalidateCacheRange(u32 start_address, std::size_t length) {
    if (start_address < NativeAddressSpaceMirror::GuestBase ||
        start_address + length > NativeAddressSpaceMirror::GuestEnd) {
        return;
    }
    NativeFlushInstructionCache(reinterpret_cast<void*>(start_address), length);
}

void ARM_Native::ClearExclusiveState() {
    // Guest ldrex/strex run on the host's own exclusive monitor. Dropping the reservation across a
    // guest context switch makes a later strex fail, which every such loop already handles.
#if defined(__arm__)
    asm volatile("clrex" ::: "memory");
#endif
}

void ARM_Native::SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) {
    // No cache maintenance here: this is called on every guest thread switch, and the mirror
    // already flushes each range as it maps it.
    current_page_table = page_table;
}

std::shared_ptr<Memory::PageTable> ARM_Native::GetPageTable() const {
    return current_page_table;
}

void ARM_Native::SetPC(u32 pc) {
    // Preserve the Thumb state already selected by the CPSR; callers pass a plain address.
    const bool thumb = (context.cpsr & 0x20u) != 0;
    context.regs[15] = thumb ? (pc | 1u) : (pc & ~1u);
}

u32 ARM_Native::GetPC() const {
    return context.regs[15] & ~1u;
}

u32 ARM_Native::GetReg(int index) const {
    ASSERT(index >= 0 && index < 16);
    return index == 15 ? GetPC() : context.regs[index];
}

void ARM_Native::SetReg(int index, u32 value) {
    ASSERT(index >= 0 && index < 16);
    if (index == 15) {
        SetPC(value);
    } else {
        context.regs[index] = value;
    }
}

u32 ARM_Native::GetVFPReg(int index) const {
    ASSERT(index >= 0 && index < 64);
    return reinterpret_cast<const u32*>(context.fpregs.data())[index];
}

void ARM_Native::SetVFPReg(int index, u32 value) {
    ASSERT(index >= 0 && index < 64);
    reinterpret_cast<u32*>(context.fpregs.data())[index] = value;
}

u32 ARM_Native::GetVFPSystemReg(VFPSystemRegister reg) const {
    switch (reg) {
    case VFP_FPSCR:
        return context.fpscr;
    case VFP_FPEXC:
        return fpexc;
    default:
        UNREACHABLE_MSG("Unknown VFP system register: {}", reg);
    }
}

void ARM_Native::SetVFPSystemReg(VFPSystemRegister reg, u32 value) {
    switch (reg) {
    case VFP_FPSCR:
        context.fpscr = value;
        return;
    case VFP_FPEXC:
        fpexc = value;
        return;
    default:
        UNREACHABLE_MSG("Unknown VFP system register: {}", reg);
    }
}

u32 ARM_Native::GetCPSR() const {
    return context.cpsr;
}

void ARM_Native::SetCPSR(u32 cpsr) {
    const bool was_thumb = (context.cpsr & 0x20u) != 0;
    const bool now_thumb = (cpsr & 0x20u) != 0;
    context.cpsr = cpsr;
    if (was_thumb != now_thumb) {
        // The instruction set selection lives in bit 0 of the stored PC when entering the guest.
        context.regs[15] = now_thumb ? (context.regs[15] | 1u) : (context.regs[15] & ~1u);
    }
}

u32 ARM_Native::GetCP15Register(CP15Register reg) const {
    return cp15[reg];
}

void ARM_Native::SetCP15Register(CP15Register reg, u32 value) {
    // CP15_THREAD_URO is picked up from here on the way into guest execution rather than written
    // through now: it shares TPIDRURO with the host thread's own pointer.
    cp15[reg] = value;
}

void ARM_Native::SaveContext(ThreadContext& ctx) {
    ctx.cpu_registers = context.regs;
    ctx.cpu_registers[15] = GetPC();
    ctx.cpsr = context.cpsr;
    std::memcpy(ctx.fpu_registers.data(), context.fpregs.data(), sizeof(ctx.fpu_registers));
    ctx.fpscr = context.fpscr;
    ctx.fpexc = fpexc;
}

void ARM_Native::LoadContext(const ThreadContext& ctx) {
    context.regs = ctx.cpu_registers;
    context.cpsr = ctx.cpsr;
    // Re-fold the Thumb bit, which SaveContext stripped from the stored PC.
    const bool thumb = (ctx.cpsr & 0x20u) != 0;
    context.regs[15] = thumb ? (ctx.cpu_registers[15] | 1u) : (ctx.cpu_registers[15] & ~1u);
    std::memcpy(context.fpregs.data(), ctx.fpu_registers.data(), sizeof(ctx.fpu_registers));
    context.fpscr = ctx.fpscr;
    fpexc = ctx.fpexc;

    ClearExclusiveState();
}

void ARM_Native::PrepareReschedule() {
    NativeStopExecution();
}

} // namespace Core
