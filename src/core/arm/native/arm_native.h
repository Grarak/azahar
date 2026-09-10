// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>

#include <memory>
#include "core/arm/arm_interface.h"
#include "core/arm/native/native_executor.h"
#include "core/hle/kernel/svc.h"

namespace Memory {
class MemorySystem;
}

namespace Core {

class System;
class NativeAddressSpaceMirror;

/**
 * Chooses how guest time advances under native execution.
 *
 * By default it tracks the host clock, which is the honest thing to do when the guest is running
 * on real hardware at real speed but makes no two runs alike: the same inputs replayed give a
 * different result, because the guest sees different timings.
 *
 * Set this and guest time instead advances a fixed amount per exit from guest code, so it becomes
 * a function of what the guest itself did. Two runs with the same inputs then agree, which is what
 * an input recording needs to be worth anything. The cost is that guest time no longer bears any
 * relation to real time.
 */
void SetNativeDeterministicTiming(bool enabled);
bool NativeDeterministicTiming();

/// One guest system call as the native backend saw it: filled at svc entry, result patched
/// after the handler returns (0xEEEEEEEE when a reschedule made the result unreadable).
struct SvcRec {
    u32 tid, svc, r0, r1, r2, r3, pc, lr, result;
};
using SvcRingArray = std::array<SvcRec, 8192>;
extern SvcRingArray g_svc_ring;
extern std::atomic<u32> g_svc_ring_head;

/**
 * Runs the guest's ARM code on the host CPU, unmodified and at its own addresses.
 *
 * The 3DS is an ARMv6K machine and this backend only builds for 32-bit ARM hosts, so guest
 * instructions are host instructions: nothing is decoded, recompiled or relocated. Guest pages are
 * mapped at the addresses the guest uses (NativeAddressSpaceMirror), and the guest is entered by
 * loading its register file and branching to its PC.
 *
 * Control comes back on an svc, a fault, or the preemption timer. An svc arrives as a SIGSYS raised
 * by a seccomp filter, so it reaches the HLE kernel instead of the host's system call table.
 *
 * Two things this cannot do, and both are silent.
 *
 * ARMv6K's VFP short vectors: ARMv8's AArch32 makes FPSCR.LEN read-as-zero rather than
 * trapping, so guest code that selects a vector length computes scalar results, with no
 * trap to hang an emulation off.
 *
 * IT-block state does not survive preemption: the entry stub restores the guest's flags
 * with MSR APSR_nzcvq, which writes NZCVQ and nothing else, so the guest is re-entered
 * with ITSTATE clear even though the exit saved it. A guest preempted between an IT and
 * the last instruction it guards resumes with the rest of that block unconditional.
 * Nothing in user mode can put the bits back, writes to the execution-state bits there
 * being UNPREDICTABLE, so the fix is re-entry through rt_sigreturn with a hand-built
 * ucontext_t rather than another instruction in the stub.
 */
class ARM_Native final : public ARM_Interface {
public:
    /**
     * @param mirror The host address space mirror, shared by every core: the guest cores of a
     *               process share one address space, and there is only one host address space to
     *               map it into.
     */
    explicit ARM_Native(Core::System& system, Memory::MemorySystem& memory, u32 id,
                        std::shared_ptr<Core::Timing::Timer> timer,
                        std::shared_ptr<NativeAddressSpaceMirror> mirror);
    ~ARM_Native() override;

    /// Whether the host can actually run guest code this way. If false, the caller must fall back.
    bool IsUsable() const {
        return usable;
    }

    /// Creates the shared address space mirror, or nullptr if the host will not give up the guest's
    /// address range.
    static std::shared_ptr<NativeAddressSpaceMirror> CreateMirror(Memory::MemorySystem& memory);

    void Run() override;
    void Step() override;

    void ClearInstructionCache() override;
    void InvalidateCacheRange(u32 start_address, std::size_t length) override;
    void ClearExclusiveState() override;
    void SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) override;

    void SetPC(u32 pc) override;
    u32 GetPC() const override;
    u32 GetReg(int index) const override;
    void SetReg(int index, u32 value) override;
    u32 GetVFPReg(int index) const override;
    void SetVFPReg(int index, u32 value) override;
    u32 GetVFPSystemReg(VFPSystemRegister reg) const override;
    void SetVFPSystemReg(VFPSystemRegister reg, u32 value) override;
    u32 GetCPSR() const override;
    void SetCPSR(u32 cpsr) override;
    u32 GetCP15Register(CP15Register reg) const override;
    void SetCP15Register(CP15Register reg, u32 value) override;

    void SaveContext(ThreadContext& ctx) override;
    void LoadContext(const ThreadContext& ctx) override;
    void PrepareReschedule() override;

    bool HasSingleInstructionBreakAccuracy() override {
        // Stopping between two instructions needs hardware debug support this backend does not use.
        return false;
    }

protected:
    std::shared_ptr<Memory::PageTable> GetPageTable() const override;

private:
    /// Runs the guest for one slice and handles whatever stopped it. Returns false if the core
    /// should stop running for now.
    bool RunSlice(u64 quantum_ns);

    /// Advances the guest's clock for one trip through guest code.
    void CreditTime(u64 elapsed_ns, NativeExitReason reason);

    void ReportFault(const NativeExitInfo& exit);

    Core::System& system;
    Memory::MemorySystem& memory;
    Kernel::SVCContext svc_context;

    NativeGuestContext context{};
    std::shared_ptr<NativeAddressSpaceMirror> mirror;
    std::shared_ptr<Memory::PageTable> current_page_table;
    std::array<u32, CP15_REGISTER_COUNT> cp15{};
    u32 fpexc{};
    bool usable{};
    bool thread_prepared{};
};

} // namespace Core
