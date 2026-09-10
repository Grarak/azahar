// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <vector>
#include "core/arm/arm_interface.h"
#include "azahar_native.h"
#include "core/hle/kernel/svc.h"
#include "core/memory.h"

namespace Core {

class System;

/**
 * Keeps the taken core's translation table matching the emulated process's page table.
 *
 * The PS Vita build runs guest code on a CPU core taken away from the Vita's kernel, under a
 * translation table the kernel module authors (vita_native/src/native.c). The guest
 * therefore sees memory at its own addresses, but those addresses are not this process's: what
 * this process reaches through a MemoryRef pointer, the guest reaches through a descriptor the
 * module writes for the same physical pages. This observer turns each mapping change of the
 * emulated process into a azaharMap / azaharEdit call.
 *
 * Until the table is installed (the first Run), mappings are only recorded; Install() then
 * installs an empty table and adds them one by one.
 */
class VitaGuestMap final : public Memory::MappingObserver {
public:
    explicit VitaGuestMap(Memory::MemorySystem& memory);
    ~VitaGuestMap() override;

    void OnMapped(VAddr base, u32 size, MemoryRef target) override;
    void OnUnmapped(VAddr base, u32 size) override;
    void OnPageTableChanged() override;

    /// Installs the table on the taken core with everything recorded so far. Idempotent.
    bool Install();

    /// Which PageTable the recorded mappings mirror. Cleared on OnPageTableChanged; the backend
    /// compares against it to re-sync only when the kernel actually hands a core a new table.
    const void* MirroredTable() const {
        return mirrored;
    }
    void SetMirroredTable(const void* table) {
        mirrored = table;
    }
    bool IsInstalled() const {
        return installed;
    }

    /// The app wrote instructions into [base, base+size) of guest memory through its own
    /// pointers: clean them out to memory and have the taken core drop its instruction cache.
    void SyncCode(VAddr base, u32 size);
    void SyncAllCode();

    /// The guest write tracker's primitive: every guest alias of the app's pages [host, host +
    /// size) gets read-only or read-write (AZAHAR_EDIT_PROTECT; the taken core drops its TLB at
    /// the next azaharRun). Emulation thread, between slices.
    void ProtectHostRange(const u8* host, std::size_t size, bool read_only);
    /// The app's page the guest page holding `guest` is mapped from, or null.
    [[nodiscard]] const u8* HostPageOf(VAddr guest) const;
    void ProtectGuestPage(VAddr guest_page, bool read_only);

    /// Highest guest address the module's table can hold, exclusive.
    static constexpr u32 GuestEnd = 0x40000000;

private:
    struct Mapping {
        u32 size;
        u8* host;
        bool operator==(const Mapping&) const = default;
    };

    /// Drops [base, base+size) from the bookkeeping, trimming or splitting entries that reach
    /// into it, so the map only ever describes disjoint ranges. With unmap set the guest side is
    /// cleared too.
    void ForgetRange(VAddr base, u32 size, bool unmap);
    bool Edit(u32 op, u32 user_va, u32 guest_va, u32 size, u32 perm) const;

    Memory::MemorySystem& memory;
    bool installed{};
    const void* mirrored{};
    /// Guest ranges currently mapped (or recorded for the install), keyed by guest base, disjoint.
    std::map<VAddr, Mapping> live_mappings;
};

/**
 * Runs the guest's ARM code natively on a PS Vita core taken away from the Vita's kernel.
 *
 * The counterpart of ARM_Native (the Linux backend) with every host mechanism replaced by a call
 * into the azahar-native kernel module: the guest is entered with azaharRun, which returns when the guest
 * takes an svc, an undefined instruction, a prefetch or data abort, or when its quantum on the
 * core's private timer expires. There are no signals and no address-space mirror; the module
 * owns the core's translation table (VitaGuestMap keeps it in step) and the guest runs at PL0
 * under vectors the module installed. See DEVELOPMENT.md.
 *
 * Both emulated cores share the one taken core and the one host thread, as on Linux.
 */
class ARM_VitaNative final : public ARM_Interface {
public:
    explicit ARM_VitaNative(Core::System& system, Memory::MemorySystem& memory, u32 id,
                            std::shared_ptr<Core::Timing::Timer> timer,
                            std::shared_ptr<VitaGuestMap> map);
    ~ARM_VitaNative() override;

    /**
     * Takes the core and registers the mapping observer, or returns nullptr if the module is
     * absent or the core cannot be taken — the caller then falls back to interpretation. Once
     * per process: the core stays taken until the console reboots.
     */
    static std::shared_ptr<VitaGuestMap> CreateMap(Memory::MemorySystem& memory);

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
        return false;
    }

protected:
    std::shared_ptr<Memory::PageTable> GetPageTable() const override;

private:
    /// One trip into the guest. Returns false if the core should stop running for now.
    bool RunSlice(u64 quantum_ns);
    void CreditTime(const AzaharRunRequest& run, u64 quantum_ns);
    void ReportFault(const AzaharRunRequest& run);

    Core::System& system;
    Memory::MemorySystem& memory;
    Kernel::SVCContext svc_context;

    /// The guest register file as the module expects it. The PC is kept without the Thumb bit;
    /// the instruction set lives in CPSR.T alone.
    AzaharContext context{};
    std::shared_ptr<VitaGuestMap> map;
    std::shared_ptr<Memory::PageTable> current_page_table;
    std::array<u32, CP15_REGISTER_COUNT> cp15{};
    u32 fpexc{};
    std::atomic<bool> stop_requested{};
    bool usable{};
};

} // namespace Core
