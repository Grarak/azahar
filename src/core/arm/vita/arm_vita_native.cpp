// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/vita/arm_vita_native.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "core/arm/vita/native_stats.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <psp2/io/fcntl.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "video_core/gpu.h"
#include "core/core_timing.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/process.h"
#include "core/hle/kernel/vm_manager.h"
#include "core/hle/kernel/thread.h"
#include "core/memory.h"

namespace Core {

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

/// Every guest page is mapped read, write and execute. The emulated process's own permissions
/// are not reported through the mapping observer; W^X per section is a later refinement.
constexpr u32 GuestPerm = AZAHAR_PERM_R | AZAHAR_PERM_W | AZAHAR_PERM_X;

/// The core the module takes. The emulation thread must not be pinned there, or the detach
/// leaves it with nowhere to run: it moves to core 0, where the rasterizer's first worker lives
/// and where it mostly waits inside azaharRun anyway.
constexpr int TakenCore = 2;

const char* ReasonName(u32 reason) {
    switch (reason) {
    case AZAHAR_EXIT_SVC:
        return "svc";
    case AZAHAR_EXIT_UND:
        return "undefined instruction";
    case AZAHAR_EXIT_PABT:
        return "prefetch abort";
    case AZAHAR_EXIT_DABT:
        return "data abort";
    case AZAHAR_EXIT_TIMER:
        return "preemption";
    default:
        return "unknown exit";
    }
}

} // Anonymous namespace

// ---------------------------------------------------------------------------- VitaGuestMap

VitaGuestMap::VitaGuestMap(Memory::MemorySystem& memory_) : memory{memory_} {}

VitaGuestMap::~VitaGuestMap() {
    if (installed) {
        azaharRelease();
    }
}

void VitaGuestMap::ProtectHostRange(const u8* host, std::size_t size, bool read_only) {
    if (!installed) {
        return;
    }
    const u32 perm = read_only ? (AZAHAR_PERM_R | AZAHAR_PERM_X) : GuestPerm;
    for (const auto& [base, mapping] : live_mappings) {
        const u8* lo = std::max<const u8*>(mapping.host, host);
        const u8* hi = std::min<const u8*>(mapping.host + mapping.size, host + size);
        if (lo >= hi) {
            continue;
        }
        Edit(AZAHAR_EDIT_PROTECT, 0, base + static_cast<u32>(lo - mapping.host),
             static_cast<u32>(hi - lo), perm);
    }
}

void VitaGuestMap::ProtectGuestPage(VAddr guest_page, bool read_only) {
    Edit(AZAHAR_EDIT_PROTECT, 0, guest_page, 4096, read_only ? (AZAHAR_PERM_R | AZAHAR_PERM_X) : GuestPerm);
}

const u8* VitaGuestMap::HostPageOf(VAddr guest) const {
    auto it = live_mappings.upper_bound(guest);
    if (it == live_mappings.begin()) {
        return nullptr;
    }
    --it;
    if (guest >= it->first + it->second.size) {
        return nullptr;
    }
    return it->second.host + ((guest - it->first) & ~VAddr{4095});
}

bool VitaGuestMap::Edit(u32 op, u32 user_va, u32 guest_va, u32 size, u32 perm) const {
    AzaharEditRequest req{};
    req.op = op;
    req.range = AzaharRange{user_va, guest_va, size, perm};
    const int ret = azaharEdit(&req);
    if (ret < 0) {
        LOG_ERROR(Core_ARM11, "azaharEdit op {} guest {:#010x}+{:#x} failed: {}", op, guest_va, size,
                  ret);
        return false;
    }
    return true;
}

void VitaGuestMap::OnMapped(VAddr base, u32 size, MemoryRef target) {
    if (base + size > GuestEnd || base + size < base) {
        return;
    }
    u8* host = target.GetPtr();
    if (host == nullptr) {
        OnUnmapped(base, size);
        return;
    }
    if ((reinterpret_cast<uintptr_t>(host) & 0xFFFu) != 0) {
        // The module maps whole pages; backing that does not start on one cannot be described.
        LOG_WARNING(Core_ARM11, "Guest {:#010x}+{:#x}: backing {} is not page aligned", base, size,
                    fmt::ptr(host));
        OnUnmapped(base, size);
        return;
    }

    const Mapping wanted{size, host};
    const auto existing = live_mappings.find(base);
    if (existing != live_mappings.end() && existing->second == wanted) {
        return;
    }

    // Whatever was there is replaced, on the guest side too when the table is live.
    ForgetRange(base, size, installed);
    live_mappings.emplace(base, wanted);
    if (installed) {
        Edit(AZAHAR_EDIT_MAP, reinterpret_cast<u32>(host), base, size, GuestPerm);
    }
}

void VitaGuestMap::OnUnmapped(VAddr base, u32 size) {
    if (base + size > GuestEnd || base + size < base) {
        return;
    }
    ForgetRange(base, size, installed);
}

void VitaGuestMap::ForgetRange(VAddr base, u32 size, bool unmap) {
    if (size == 0) {
        return;
    }
    const VAddr end = base + size;

    auto it = live_mappings.lower_bound(base);
    if (it != live_mappings.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.size > base) {
            it = prev;
        }
    }

    while (it != live_mappings.end() && it->first < end) {
        const VAddr entry_base = it->first;
        const Mapping entry = it->second;
        const VAddr entry_end = entry_base + entry.size;
        const VAddr overlap_base = std::max(entry_base, base);
        const VAddr overlap_end = std::min(entry_end, end);

        it = live_mappings.erase(it);

        if (unmap) {
            Edit(AZAHAR_EDIT_UNMAP, 0, overlap_base, overlap_end - overlap_base, 0);
        }

        if (entry_base < overlap_base) {
            live_mappings.emplace(entry_base, Mapping{overlap_base - entry_base, entry.host});
        }
        if (overlap_end < entry_end) {
            live_mappings.emplace(overlap_end, Mapping{entry_end - overlap_end,
                                                       entry.host + (overlap_end - entry_base)});
        }
    }
}

void VitaGuestMap::OnPageTableChanged() {
    mirrored = nullptr;
    if (installed) {
        for (const auto& [base, mapping] : live_mappings) {
            Edit(AZAHAR_EDIT_UNMAP, 0, base, mapping.size, 0);
        }
    }
    live_mappings.clear();
}

bool VitaGuestMap::Install() {
    if (installed) {
        return true;
    }
    AzaharMapRequest req{};
    req.count = 0;
    const int ret = azaharMap(&req);
    if (ret < 0) {
        LOG_ERROR(Core_ARM11, "azaharMap failed: {}", ret);
        return false;
    }
    installed = true;
    // Guest write tracking (core/guest_write_tracker.h) over AZAHAR_EDIT_PROTECT, armed between
    // slices. `nowritetrack` in gxm_flags.txt turns it off.
    {
        auto& tracker = memory.WriteTracker();
        tracker.SetProtect([this](const u8* host, std::size_t size, bool read_only) {
            ProtectHostRange(host, size, read_only);
        });
        tracker.SetDeferred(true);
        tracker.SetHotPages(!GxmRenderer::GxmFlag("nohotpages"));
        tracker.SetEnabled(!GxmRenderer::GxmFlag("nowritetrack"));
    }
    unsigned added = 0;
    for (const auto& [base, mapping] : live_mappings) {
        if (Edit(AZAHAR_EDIT_MAP, reinterpret_cast<u32>(mapping.host), base, mapping.size,
                 GuestPerm)) {
            added++;
        }
    }
    LOG_INFO(Core_ARM11, "Guest table installed on the taken core with {} range(s)", added);
    return true;
}

void VitaGuestMap::SyncCode(VAddr base, u32 size) {
    if (!installed || size == 0) {
        return;
    }
    const VAddr start = base & ~0xFFFu;
    const VAddr end = (base + size + 0xFFFu) & ~0xFFFu;
    auto it = live_mappings.lower_bound(start);
    if (it != live_mappings.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.size > start) {
            it = prev;
        }
    }
    for (; it != live_mappings.end() && it->first < end; ++it) {
        const VAddr lo = std::max(it->first, start);
        const VAddr hi = std::min(it->first + it->second.size, end);
        if (lo < hi) {
            Edit(AZAHAR_EDIT_SYNC_CODE, reinterpret_cast<u32>(it->second.host + (lo - it->first)),
                 lo, hi - lo, 0);
        }
    }
}

void VitaGuestMap::SyncAllCode() {
    if (!installed) {
        return;
    }
    for (const auto& [base, mapping] : live_mappings) {
        Edit(AZAHAR_EDIT_SYNC_CODE, reinterpret_cast<u32>(mapping.host), base, mapping.size, 0);
    }
}

// ---------------------------------------------------------------------------- ARM_VitaNative

std::shared_ptr<VitaGuestMap> ARM_VitaNative::CreateMap(Memory::MemorySystem& memory) {
    // Under the Vita3K emulator there is no kernel to take a core from, and an unresolved import
    // there returns zero rather than an error - which azaharTakeCore cannot tell from success. So
    // that target says so with a file instead, and the caller falls back to the interpreter.
    if (SceUID fd = sceIoOpen("ux0:data/azahar/cpu_interpreter", SCE_O_RDONLY, 0); fd >= 0) {
        sceIoClose(fd);
        LOG_WARNING(Core_ARM11, "ux0:data/azahar/cpu_interpreter present: not taking a core");
        return nullptr;
    }

    // The module holds a core only in its resident design, and a thread pinned to a held core
    // never runs again - so this thread steps aside before the take, and steps back if the
    // module answers that it folded the guest into its caller instead. Folded, the partition's
    // choice is the one that matters: the slice runs on this thread's core, so the emulation
    // thread belongs on the core the frontend gave it and the guest goes there with it.
    const SceUID self = sceKernelGetThreadId();
    const int affinity = sceKernelGetThreadCpuAffinityMask(self);
    if (affinity == (SCE_KERNEL_CPU_MASK_USER_0 << TakenCore) || affinity == 0) {
        sceKernelChangeThreadCpuAffinityMask(self, SCE_KERNEL_CPU_MASK_USER_0);
    }

    sceClibPrintf("[azahar] CreateMap: calling azaharTakeCore\n");
    const int ret = azaharTakeCore();
    sceClibPrintf("[azahar] CreateMap: azaharTakeCore -> %d\n", ret);
    if (ret < 0) {
        LOG_WARNING(Core_ARM11, "azaharTakeCore failed ({}): native execution unavailable", ret);
        return nullptr;
    }
    if (ret == AZAHAR_TAKE_FOLDED && affinity != 0) {
        const int back = sceKernelChangeThreadCpuAffinityMask(self, affinity);
        LOG_INFO(Core_ARM11, "folded native execution: emulation thread back on affinity {:#x} "
                             "({:#010x})", affinity, back);
    }
    auto map = std::make_shared<VitaGuestMap>(memory);
    memory.SetMappingObserver(map.get());
    return map;
}

ARM_VitaNative::ARM_VitaNative(Core::System& system_, Memory::MemorySystem& memory_, u32 id,
                               std::shared_ptr<Core::Timing::Timer> timer_,
                               std::shared_ptr<VitaGuestMap> map_)
    : ARM_Interface(id, timer_), system{system_}, memory{memory_}, svc_context{system_},
      map{std::move(map_)} {
    usable = map != nullptr;
    context.cpsr = 0x10;
}

ARM_VitaNative::~ARM_VitaNative() {
    if (map != nullptr && map.use_count() == 1) {
        memory.SetMappingObserver(nullptr);
    }
}

void ARM_VitaNative::Run() {
    if (break_flag) [[unlikely]] {
        return;
    }
    // The scheduler asks for a stop immediately before handing out a new slice; that request is
    // about the slice that just ended.
    stop_requested = false;

    const u64 quantum = std::clamp(TicksToNanoseconds(timer->GetDowncount()), MinQuantumNs,
                                   MaxQuantumNs);
    while (RunSlice(quantum)) {
        if (timer->GetDowncount() <= 0 || break_flag || stop_requested) {
            break;
        }
    }
}

void ARM_VitaNative::Step() {
    if (break_flag) [[unlikely]] {
        return;
    }
    stop_requested = false;
    RunSlice(MinQuantumNs);
}

bool ARM_VitaNative::RunSlice(u64 quantum_ns) {
    if (!usable) [[unlikely]] {
        system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised,
                         "Vita native backend is not usable");
        return false;
    }
    if (!map->IsInstalled() && !map->Install()) {
        system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised,
                         "Vita native backend could not install the guest table");
        usable = false;
        return false;
    }

    // Pages the surface cache wants armed since the last slice: protected now, while the
    // guest is stopped, so the claim and the protection take effect together at this entry.
    memory.WriteTracker().ApplyPending();

    AzaharRunRequest run{};
    run.ctx = context;
    run.ctx.tpidruro = cp15[CP15_THREAD_URO];
    run.ctx.tpidrurw = cp15[CP15_THREAD_UPRW];
    run.quantum_us = static_cast<u32>(quantum_ns / 1000);

    const SceUInt64 vanrun_start = sceKernelGetProcessTimeWide();
    const int ret = azaharRun(&run);
    NativeStats::vanrun_us.fetch_add(sceKernelGetProcessTimeWide() - vanrun_start,
                                     std::memory_order_relaxed);
    if (ret < 0) {
        system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised,
                         fmt::format("azaharRun failed: {}", ret).c_str());
        usable = false;
        return false;
    }

    context = run.ctx;
    cp15[CP15_THREAD_UPRW] = run.ctx.tpidrurw;
    NativeStats::slices.fetch_add(1, std::memory_order_relaxed);
    if (run.reason == AZAHAR_EXIT_SVC) {
        NativeStats::svcs.fetch_add(1, std::memory_order_relaxed);
    }
    CreditTime(run, quantum_ns);

    switch (run.reason) {
    case AZAHAR_EXIT_SVC: {
        const SceUInt64 svc_start = sceKernelGetProcessTimeWide();
        svc_context.CallSVC(run.svc_number);
        NativeStats::svc_us.fetch_add(sceKernelGetProcessTimeWide() - svc_start,
                                      std::memory_order_relaxed);
    }
        // An svc can reschedule, which replaces the register file wholesale.
        return !break_flag && !stop_requested;

    case AZAHAR_EXIT_TIMER:
        return false;

    case AZAHAR_EXIT_DABT: {
        // A store into a page the surface cache armed: mark it, make it writable again and
        // resume at the store (r15 is the faulting instruction). A fault on any other page
        // is the guest's own.
        auto& tracker = memory.WriteTracker();
        const u8* host = tracker.Enabled() ? map->HostPageOf(run.far) : nullptr;
        if (host != nullptr) {
            struct Unprotect {
                VitaGuestMap* map;
                u32 guest_page;
            } ctx{map.get(), run.far & ~u32{4095}};
            const bool ours = tracker.OnFault(
                host,
                [](void* raw) {
                    auto* u = static_cast<Unprotect*>(raw);
                    u->map->ProtectGuestPage(u->guest_page, false);
                },
                &ctx);
            if (ours) {
                NativeStats::write_faults.fetch_add(1, std::memory_order_relaxed);
                return !break_flag && !stop_requested;
            }
        }
        ReportFault(run);
        return false;
    }
    case AZAHAR_EXIT_UND:
    case AZAHAR_EXIT_PABT:
    default:
        ReportFault(run);
        return false;
    }
}

void ARM_VitaNative::CreditTime(const AzaharRunRequest& run, u64 quantum_ns) {
    // The core's private timer counted the slice down; what it has left is what the guest did not
    // use. Native execution has no instruction count, so guest time is real time.
    u64 elapsed_ns = quantum_ns;
    if (run.timer_hz != 0) {
        const u64 quantum_ticks = quantum_ns / 1000 * run.timer_hz / 1'000'000ull;
        const u64 spent = quantum_ticks > run.timer_left ? quantum_ticks - run.timer_left : 0;
        elapsed_ns = spent * 1'000'000'000ull / run.timer_hz;
    }
    NativeStats::credited_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    const u64 ticks = NanosecondsToTicks(elapsed_ns);
    if (ticks > 0) {
        timer->AddTicks(ticks);
    }
}

void ARM_VitaNative::ReportFault(const AzaharRunRequest& run) {
    std::string registers;
    for (int i = 0; i < 16; i++) {
        registers += fmt::format("r{:02d} = {:08X}\n", i, context.r[i]);
    }
    const std::string message =
        fmt::format("{}Guest {} at PC {:08X} (cpsr {:08X}), fsr {:08X} far {:08X}", registers,
                    ReasonName(run.reason), context.r[15], context.cpsr, run.fsr, run.far);
    LOG_CRITICAL(Core_ARM11, "{}", message);
    std::string code;
    for (u32 a = context.r[15] - 16; a <= context.r[15] + 12; a += 4) {
        code += fmt::format("{}{:08X}", a == context.r[15] ? " *" : " ", memory.Read32(a));
    }
    LOG_CRITICAL(Core_ARM11, "code around PC:{}\nrender-thread writes:\n{}", code,
                 system.GPU().DescribeRecentWrites(std::span<const u32>(context.r),
                                                   run.far));
    system.SetStatus(System::ResultStatus::ErrorCoreExceptionRaised, message.c_str());
}

void ARM_VitaNative::ClearInstructionCache() {
    if (map) {
        map->SyncAllCode();
    }
}

void ARM_VitaNative::InvalidateCacheRange(u32 start_address, std::size_t length) {
    if (!map) {
        return;
    }
    // svcInvalidateInstructionCacheRange, which some titles issue every frame over a page
    // of their data. A sync is a kernel round trip that cleans the page and drops core 2's
    // icache, and it is only owed where something can execute: the process's own view of
    // the range says whether it can.
    if (const auto process = system.Kernel().GetCurrentProcess()) {
        const auto& vm = process->vm_manager;
        bool executable = false;
        const VAddr end = start_address + static_cast<VAddr>(length);
        for (VAddr at = start_address; at < end;) {
            const auto vma = vm.FindVMA(at);
            if (vma == vm.vma_map.end()) {
                break;
            }
            if ((static_cast<u8>(vma->second.permissions) &
                 static_cast<u8>(Kernel::VMAPermission::Execute)) != 0) {
                executable = true;
                break;
            }
            at = vma->second.base + vma->second.size;
        }
        if (!executable) {
            static u32 shown = 0;
            if (shown++ < 4) {
                LOG_INFO(Core_ARM11, "icache invalidate of {:08X}+{:X} skipped: nothing executes there",
                         start_address, length);
            }
            return;
        }
    }
    map->SyncCode(start_address, static_cast<u32>(length));
}

void ARM_VitaNative::ClearExclusiveState() {
    // Nothing to do here, but not for the reason this used to give. Whether an exception return
    // clears the local monitor is IMPLEMENTATION DEFINED on ARMv7-A, so entry.S now executes
    // CLREX explicitly before every guest entry - which covers this and every other path back
    // into the guest, including the ones that never call through here.
}

void ARM_VitaNative::SetPageTable(const std::shared_ptr<Memory::PageTable>& page_table) {
    current_page_table = page_table;
    // The one call that reaches every core: the kernel updates the memory system's current
    // table only for the core it considers running, so a table given to any other core would
    // never reach the mapping observer at all - and the guest table would install empty.
    if (map && page_table && map->MirroredTable() != page_table.get()) {
        memory.SyncMappingObserver(*page_table);
        map->SetMirroredTable(page_table.get());
    }
}

std::shared_ptr<Memory::PageTable> ARM_VitaNative::GetPageTable() const {
    return current_page_table;
}

void ARM_VitaNative::SetPC(u32 pc) {
    context.r[15] = pc & ~1u;
}

u32 ARM_VitaNative::GetPC() const {
    return context.r[15] & ~1u;
}

u32 ARM_VitaNative::GetReg(int index) const {
    ASSERT(index >= 0 && index < 16);
    return index == 15 ? GetPC() : context.r[index];
}

void ARM_VitaNative::SetReg(int index, u32 value) {
    ASSERT(index >= 0 && index < 16);
    if (index == 15) {
        SetPC(value);
    } else {
        context.r[index] = value;
    }
}

u32 ARM_VitaNative::GetVFPReg(int index) const {
    ASSERT(index >= 0 && index < 64);
    return context.fpregs[index];
}

void ARM_VitaNative::SetVFPReg(int index, u32 value) {
    ASSERT(index >= 0 && index < 64);
    context.fpregs[index] = value;
}

u32 ARM_VitaNative::GetVFPSystemReg(VFPSystemRegister reg) const {
    switch (reg) {
    case VFP_FPSCR:
        return context.fpscr;
    case VFP_FPEXC:
        return fpexc;
    default:
        UNREACHABLE_MSG("Unknown VFP system register: {}", reg);
    }
}

void ARM_VitaNative::SetVFPSystemReg(VFPSystemRegister reg, u32 value) {
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

u32 ARM_VitaNative::GetCPSR() const {
    return context.cpsr;
}

void ARM_VitaNative::SetCPSR(u32 cpsr) {
    context.cpsr = cpsr;
}

u32 ARM_VitaNative::GetCP15Register(CP15Register reg) const {
    return cp15[reg];
}

void ARM_VitaNative::SetCP15Register(CP15Register reg, u32 value) {
    cp15[reg] = value;
}

void ARM_VitaNative::SaveContext(ThreadContext& ctx) {
    std::memcpy(ctx.cpu_registers.data(), context.r, sizeof(context.r));
    ctx.cpu_registers[15] = GetPC();
    ctx.cpsr = context.cpsr;
    std::memcpy(ctx.fpu_registers.data(), context.fpregs, sizeof(context.fpregs));
    ctx.fpscr = context.fpscr;
    ctx.fpexc = fpexc;
}

void ARM_VitaNative::LoadContext(const ThreadContext& ctx) {
    std::memcpy(context.r, ctx.cpu_registers.data(), sizeof(context.r));
    context.r[15] = ctx.cpu_registers[15] & ~1u;
    context.cpsr = ctx.cpsr;
    std::memcpy(context.fpregs, ctx.fpu_registers.data(), sizeof(context.fpregs));
    context.fpscr = ctx.fpscr;
    fpexc = ctx.fpexc;
}

void ARM_VitaNative::PrepareReschedule() {
    // Called from the HLE side while the guest is stopped at an exit, or from another host thread
    // while it runs. The former ends the slice loop; the latter is bounded by the quantum.
    stop_requested = true;
}

} // namespace Core
