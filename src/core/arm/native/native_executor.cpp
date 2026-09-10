// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/native/native_executor.h"
#include "core/arm/native/native_mirror.h"
#include "core/guest_write_tracker.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include "common/logging/log.h"

#if defined(__arm__) && defined(__linux__)

#include <csignal>
#include <ctime>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

namespace Core {

static_assert(offsetof(NativeGuestContext, regs) == 0);
static_assert(offsetof(NativeGuestContext, cpsr) == 64);
static_assert(offsetof(NativeGuestContext, fpscr) == 68);
static_assert(offsetof(NativeGuestContext, exit_reason) == 72);
static_assert(offsetof(NativeGuestContext, host_sp) == 76);
static_assert(offsetof(NativeGuestContext, fpregs) == 80);

extern "C" {
/// Enters guest code. Returns the exit reason; see native_entry.S.
u32 NativeEnterGuest(NativeGuestContext* ctx);
/// Resumed into by sigreturn to leave guest code. Never called directly.
void NativeExitGuest();
}

namespace {

constexpr u32 CPSR_THUMB_BIT = 0x20;
/// IT-block state, split across cpsr[26:25] and cpsr[15:10].
constexpr u32 CPSR_IT_MASK = 0x0600FC00;

/// Signal used for the preemption timer. SIGRTMIN is reserved by glibc internals, so take the next.
int PreemptionSignal() {
    return SIGRTMIN + 1;
}

/**
 * State shared between the guest thread and its signal handlers.
 *
 * Deliberately not thread_local. While the guest executes, TPIDRURO holds the *guest's* thread
 * pointer, and that is the register the host's thread-local storage is reached through — so a
 * signal handler that touched a thread_local, or errno, would read whatever the guest happens to
 * have there. Only one thread ever runs guest code (Azahar timeslices both emulated cores onto
 * one host thread), so plain globals are both correct and cheaper.
 */
struct GuestThreadState {
    bool prepared{};
    bool in_guest{};
    u32 guest_base{};
    u32 guest_end{};
    timer_t preemption_timer{};
    bool has_timer{};
    NativeExitInfo* exit_info{};
    NativeGuestContext* context{};
    std::atomic<bool> stop_requested{};
    u8* signal_stack{};
    pid_t owner_tid{};
};

GuestThreadState t_state;

/**
 * Rewrites the interrupted context so that returning from this signal handler leaves guest
 * execution instead of resuming it.
 *
 * The guest's integer state is copied out here; its floating point state is left alone, because
 * sigreturn is about to restore it to the hardware where NativeExitGuest can save it.
 */
void ExitGuest(ucontext_t* uc, NativeExitReason reason) {
    auto& mc = uc->uc_mcontext;
    NativeGuestContext* ctx = t_state.context;

    ctx->regs[0] = mc.arm_r0;
    ctx->regs[1] = mc.arm_r1;
    ctx->regs[2] = mc.arm_r2;
    ctx->regs[3] = mc.arm_r3;
    ctx->regs[4] = mc.arm_r4;
    ctx->regs[5] = mc.arm_r5;
    ctx->regs[6] = mc.arm_r6;
    ctx->regs[7] = mc.arm_r7;
    ctx->regs[8] = mc.arm_r8;
    ctx->regs[9] = mc.arm_r9;
    ctx->regs[10] = mc.arm_r10;
    ctx->regs[11] = mc.arm_fp;
    ctx->regs[12] = mc.arm_ip;
    ctx->regs[13] = mc.arm_sp;
    ctx->regs[14] = mc.arm_lr;

    const bool thumb = (mc.arm_cpsr & CPSR_THUMB_BIT) != 0;
    ctx->regs[15] = mc.arm_pc | (thumb ? 1u : 0u);
    ctx->cpsr = mc.arm_cpsr;
    ctx->exit_reason = static_cast<u32>(reason);

    // Resume in the exit stub on the host stack, with the reason in r0 so it becomes the return
    // value of NativeEnterGuest and the context in r1 so the stub can reach it without naming a
    // global. The stub is ARM code outside any IT block, so the Thumb and IT-state bits have to go.
    mc.arm_pc = reinterpret_cast<u32>(&NativeExitGuest);
    mc.arm_sp = ctx->host_sp;
    mc.arm_r0 = static_cast<u32>(reason);
    mc.arm_r1 = reinterpret_cast<u32>(ctx);
    mc.arm_cpsr &= ~(CPSR_THUMB_BIT | CPSR_IT_MASK);

    t_state.in_guest = false;
}

/**
 * Whether the interrupted context was actually executing guest code.
 *
 * The in_guest flag alone cannot answer this: it is raised before the entry stub has finished
 * handing the CPU to the guest, so a signal landing in that window finds host registers under a
 * raised flag. Treating them as guest state corrupts both the context structure and, through the
 * stale host_sp the exit stub would then unwind with, the host stack. The interrupted PC is exact
 * where the flag is approximate: guest code executes only inside the guest address range, and host
 * code never does.
 */
bool InterruptedInGuest(const ucontext_t* uc) {
    const u32 pc = uc->uc_mcontext.arm_pc;
    return pc >= t_state.guest_base && pc < t_state.guest_end;
}

/// Recovers the service number from the svc instruction the guest just executed.
u32 DecodeSvcImmediate(const ucontext_t* uc) {
    const u32 pc = uc->uc_mcontext.arm_pc;
    if ((uc->uc_mcontext.arm_cpsr & CPSR_THUMB_BIT) != 0) {
        u16 insn;
        std::memcpy(&insn, reinterpret_cast<const void*>(pc - 2), sizeof(insn));
        return insn & 0xFFu;
    }
    u32 insn;
    std::memcpy(&insn, reinterpret_cast<const void*>(pc - 4), sizeof(insn));
    return insn & 0x00FFFFFFu;
}

void HandleSigsys(int, siginfo_t*, void* ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    if (!t_state.in_guest || !InterruptedInGuest(uc)) {
        // A host system call the seccomp filter should have permitted. Nothing sensible to do.
        return;
    }
    t_state.exit_info->svc_immediate = DecodeSvcImmediate(uc);
    ExitGuest(uc, NativeExitReason::Svc);
}

/// The guest write tracker and the mirror it looks up guest pages through; set by
/// NativeSetWriteTracking, read by the fault handler.
Memory::GuestWriteTracker* g_write_tracker = nullptr;
const NativeAddressSpaceMirror* g_write_mirror = nullptr;

void UnprotectGuestPage(void* ctx) {
    const auto page = reinterpret_cast<std::uintptr_t>(ctx) & ~std::uintptr_t{4095};
    mprotect(reinterpret_cast<void*>(page), 4096, PROT_READ | PROT_WRITE | PROT_EXEC);
}

void HandleFault(int sig, siginfo_t* info, void* ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    if (sig == SIGSEGV && t_state.in_guest && g_write_tracker != nullptr &&
        g_write_mirror != nullptr) {
        // A store into a page the surface cache armed: record it, make the page writable and
        // let the store retry. Not a guest exception.
        const auto guest = reinterpret_cast<std::uintptr_t>(info->si_addr);
        const u8* host = g_write_mirror->HostPageOf(static_cast<VAddr>(guest));
        if (host != nullptr && g_write_tracker->OnFault(host, UnprotectGuestPage, info->si_addr)) {
            return;
        }
    }
    if (!t_state.in_guest || !InterruptedInGuest(uc)) {
        // Not the guest's fault: restore the default action and let it crash normally, so the
        // backtrace points at the real culprit instead of at us.
        signal(sig, SIG_DFL);
        return;
    }

    t_state.exit_info->fault_address = reinterpret_cast<u32>(info->si_addr);
    t_state.exit_info->fault_pc = uc->uc_mcontext.arm_pc;
    ExitGuest(uc, sig == SIGILL ? NativeExitReason::UndefinedInstruction
                                : NativeExitReason::DataAbort);
}

void HandlePreemption(int, siginfo_t*, void* ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    if (!t_state.in_guest) {
        return;
    }
    if (!InterruptedInGuest(uc)) {
        // The timer expired after in_guest was raised but before the entry stub finished handing
        // the CPU to the guest, so the registers here are the host's and there is nothing to exit
        // from. Drop the tick; the timer is periodic precisely so that a successor lands once the
        // guest is genuinely running.
        return;
    }
    ExitGuest(uc, t_state.stop_requested.load(std::memory_order_relaxed)
                      ? NativeExitReason::Interrupted
                      : NativeExitReason::Preempted);
}

} // namespace

void NativeSetWriteTracking(const NativeAddressSpaceMirror* mirror,
                            Memory::GuestWriteTracker* tracker) {
    g_write_mirror = mirror;
    g_write_tracker = tracker;
}

namespace {

bool InstallSignalHandlers() {
    // Guest code runs with a guest stack pointer, so signal frames must not be pushed onto it.
    // SIGSTKSZ is a runtime query on current glibc, so this cannot be constexpr.
    const std::size_t stack_size = std::max<std::size_t>(SIGSTKSZ * 4, 65536);
    t_state.signal_stack = static_cast<u8*>(mmap(nullptr, stack_size, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (t_state.signal_stack == MAP_FAILED) {
        t_state.signal_stack = nullptr;
        return false;
    }

    stack_t ss{};
    ss.ss_sp = t_state.signal_stack;
    ss.ss_size = stack_size;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, nullptr) != 0) {
        return false;
    }

    const auto install = [](int sig, void (*handler)(int, siginfo_t*, void*)) {
        struct sigaction sa{};
        sa.sa_sigaction = handler;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
        sigemptyset(&sa.sa_mask);
        // A preemption tick arriving inside another handler would corrupt the exit it is building.
        sigaddset(&sa.sa_mask, PreemptionSignal());
        return sigaction(sig, &sa, nullptr) == 0;
    };

    return install(SIGSYS, HandleSigsys) && install(SIGSEGV, HandleFault) &&
           install(SIGBUS, HandleFault) && install(SIGILL, HandleFault) &&
           install(PreemptionSignal(), HandlePreemption);
}

/**
 * Installs the filter that turns guest svc instructions into SIGSYS.
 *
 * The filter discriminates on the address the syscall was made from, not on the syscall number.
 * A guest service number can be anything at all and will collide with host syscall numbers, but
 * guest code only ever executes inside the guest address range, so the address is exact where the
 * number is a guess.
 */
bool InstallSeccompFilter(u32 guest_base, u32 guest_end) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_ARM, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, instruction_pointer)),
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, guest_base, 0, 2),
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, guest_end, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog{};
    prog.len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0]));
    prog.filter = filter;

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        return false;
    }
    return syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) == 0;
}

bool CreatePreemptionTimer() {
    struct sigevent sev{};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = PreemptionSignal();
    sev.sigev_value.sival_ptr = nullptr;
    // Deliver to this thread specifically: any other thread would be interrupted for nothing.
    sev._sigev_un._tid = static_cast<int>(syscall(SYS_gettid));

    if (timer_create(CLOCK_MONOTONIC, &sev, &t_state.preemption_timer) != 0) {
        return false;
    }
    t_state.has_timer = true;
    return true;
}

void ArmPreemptionTimer(u64 quantum_ns) {
    if (!t_state.has_timer || quantum_ns == 0) {
        return;
    }
    struct itimerspec its{};
    its.it_value.tv_sec = static_cast<time_t>(quantum_ns / 1000000000ull);
    its.it_value.tv_nsec = static_cast<long>(quantum_ns % 1000000000ull);
    // Periodic rather than one-shot: a tick that lands before the entry stub has handed the CPU to
    // the guest is deliberately dropped by the handler, and only a successor tick can end the
    // guest's slice then. The slice overshoots by at most one quantum.
    its.it_interval = its.it_value;
    timer_settime(t_state.preemption_timer, 0, &its, nullptr);
}

void DisarmPreemptionTimer() {
    if (!t_state.has_timer) {
        return;
    }
    struct itimerspec its{};
    timer_settime(t_state.preemption_timer, 0, &its, nullptr);
}

/**
 * The guest's TLS register.
 *
 * The guest reads its thread pointer straight out of TPIDRURO, which is readable but not writable
 * at user level, so setting it needs the kernel's ARM-private set_tls call. glibc keeps the host
 * thread's own pointer in that same register, so the guest's value can only be installed for as
 * long as the guest is running and must be taken back out before any host code runs again.
 */
u32 ReadThreadPointer() {
    u32 value;
    asm volatile("mrc p15, 0, %0, c13, c0, 3" : "=r"(value));
    return value;
}

void WriteThreadPointer(u32 value) {
    // ARM-private syscall __ARM_NR_set_tls, whose number goes in r7 like any other.
    register u32 r0 asm("r0") = value;
    register u32 r7 asm("r7") = 0x0F0005;
    asm volatile("svc #0" : "+r"(r0) : "r"(r7) : "memory");
}

u64 MonotonicNanoseconds() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<u64>(ts.tv_sec) * 1000000000ull + static_cast<u64>(ts.tv_nsec);
}

} // Anonymous namespace

bool NativePrepareThread(u32 guest_base, u32 guest_end) {
    if (t_state.prepared) {
        if (static_cast<pid_t>(syscall(SYS_gettid)) != t_state.owner_tid) {
            // The seccomp filter and the alternate signal stack are per-thread, and the state the
            // handlers read is a single global. Guest code has to stay on the thread that set it up.
            LOG_ERROR(Core_ARM11, "Native backend: guest code moved to a different host thread");
            return false;
        }
        return true;
    }

    t_state.guest_base = guest_base;
    t_state.guest_end = guest_end;

    if (!InstallSignalHandlers()) {
        LOG_ERROR(Core_ARM11, "Native backend: could not install signal handlers");
        return false;
    }
    if (!CreatePreemptionTimer()) {
        LOG_ERROR(Core_ARM11, "Native backend: could not create the preemption timer");
        return false;
    }
    if (!InstallSeccompFilter(guest_base, guest_end)) {
        LOG_ERROR(Core_ARM11, "Native backend: could not install the seccomp filter; a guest svc "
                            "would enter the host kernel as a system call");
        return false;
    }

    t_state.prepared = true;
    t_state.owner_tid = static_cast<pid_t>(syscall(SYS_gettid));
    LOG_INFO(Core_ARM11, "Native backend ready: guest code executes at {:#010x}-{:#010x}", guest_base,
             guest_end);
    return true;
}

bool NativeThreadIsPrepared() {
    return t_state.prepared;
}

u64 NativeRunGuest(NativeGuestContext& ctx, u64 quantum_ns, u32 guest_tls, NativeExitInfo& out) {
    out = {};
    if (t_state.stop_requested.exchange(false, std::memory_order_acq_rel)) {
        out.reason = NativeExitReason::Interrupted;
        return 0;
    }

    t_state.exit_info = &out;
    t_state.context = &ctx;

    const u64 started = MonotonicNanoseconds();
    t_state.in_guest = true;
    ArmPreemptionTimer(quantum_ns);

    // Nothing but assembly may run between these two calls: host thread-locals and errno are
    // reached through the very register being handed to the guest.
    const u32 host_tls = ReadThreadPointer();
    WriteThreadPointer(guest_tls);
    const u32 reason = NativeEnterGuest(&ctx);
    WriteThreadPointer(host_tls);

    DisarmPreemptionTimer();
    t_state.in_guest = false;
    const u64 elapsed = MonotonicNanoseconds() - started;

    out.reason = static_cast<NativeExitReason>(reason);
    return elapsed;
}

void NativeClearStopRequest() {
    t_state.stop_requested.store(false, std::memory_order_relaxed);
}

void NativeStopExecution() {
    if (!t_state.prepared) {
        return;
    }
    t_state.stop_requested.store(true, std::memory_order_relaxed);
    if (t_state.in_guest && t_state.has_timer) {
        // Fire the preemption timer as soon as the kernel will let us, and keep firing: a tick
        // landing before the guest is genuinely entered is dropped by the handler, and without the
        // interval the stop would then wait on the full quantum — or on the multi-second backstop
        // in deterministic mode.
        struct itimerspec its{};
        its.it_value.tv_nsec = 1;
        its.it_interval.tv_nsec = 100000;
        timer_settime(t_state.preemption_timer, 0, &its, nullptr);
    }
}

void NativeFlushInstructionCache(void* start, std::size_t length) {
    auto* begin = static_cast<char*>(start);
    __builtin___clear_cache(begin, begin + length);
}

} // namespace Core

#else

namespace Core {

bool NativePrepareThread(u32, u32) {
    LOG_ERROR(Core_ARM11, "Native execution requires a 32-bit ARM Linux host");
    return false;
}

bool NativeThreadIsPrepared() {
    return false;
}

u64 NativeRunGuest(NativeGuestContext&, u64, u32, NativeExitInfo& out) {
    out = {};
    out.reason = NativeExitReason::Interrupted;
    return 0;
}

void NativeStopExecution() {}

void NativeClearStopRequest() {}

void NativeFlushInstructionCache(void* start, std::size_t length) {
    auto* begin = static_cast<char*>(start);
    __builtin___clear_cache(begin, begin + length);
}

} // namespace Core

#endif
