// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <vector>
#include "common/common_types.h"
#include "common/host_shared_memory.h"
#include "core/memory.h"

namespace Core {

/**
 * Keeps the host address space matching the guest's.
 *
 * Guest code executes natively at its own addresses, so a guest pointer has to be a valid host
 * pointer meaning the same thing. This observer watches the emulated process's mappings and mirrors
 * each one: the same backing pages the emulator reaches through its own pointer are mapped a second
 * time at the guest address the guest uses. Writes through either view are the same memory.
 *
 * Guest addresses the observer has not mapped stay unmapped on the host, so a guest access to them
 * faults rather than silently reading whatever the emulator happens to have there.
 */
class NativeAddressSpaceMirror final : public Memory::MappingObserver {
public:
    explicit NativeAddressSpaceMirror(Memory::MemorySystem& memory);
    ~NativeAddressSpaceMirror() override;

    /// Reserves the guest address range, so no host allocation can land inside it. Idempotent.
    bool Reserve();

    void OnMapped(VAddr base, u32 size, MemoryRef target) override;
    void OnUnmapped(VAddr base, u32 size) override;
    void OnPageTableChanged() override;

    /// Whether the whole guest range was successfully reserved.
    bool IsUsable() const {
        return usable;
    }

    /// Makes instructions written through the emulator's own view visible to guest instruction
    /// fetch, across every range currently mirrored.
    void FlushInstructionCache() const;

    /// Flips the protection of every guest alias of the emulator's host pages [host, host +
    /// size): the guest write tracker's primitive. Any thread.
    void ProtectHostRange(const u8* host, std::size_t size, bool read_only);
    /// The emulator's host page the guest page holding `guest` is mirrored onto, or null.
    /// Lock-free, for the fault handler.
    [[nodiscard]] const u8* HostPageOf(VAddr guest) const;

    /// Lowest guest address, matching PROCESS_IMAGE_VADDR. Starting here rather than at page zero
    /// keeps the reservation clear of wherever the host loaded this process, and leaves guest null
    /// dereferences faulting.
    static constexpr u32 GuestBase = 0x00100000;
    /// End of the guest address range, exclusive: the top of NEW_LINEAR_HEAP.
    static constexpr u32 GuestEnd = 0x40000000;

private:
    void UnmapAll();

    /**
     * Drops [base, base+size) from the bookkeeping, trimming or splitting whatever entries reach
     * into it, so the map only ever describes disjoint ranges. The observer is told about ranges
     * at whatever granularity the page table happens to coalesce into, and the same bytes can be
     * reported as one large run on one call and as single pages on the next; keying by base alone
     * left the finer entries behind, and they then answered for bytes they no longer owned.
     *
     * With unmap set, the part of each entry that falls inside the range is also handed back to
     * the PROT_NONE reservation. A caller that is about to map over the range itself wants this
     * off: MAP_FIXED replaces the old mapping in the same step.
     */
    void ForgetRange(VAddr base, u32 size, bool unmap);

    /// What a guest range is currently mirrored onto, so an unchanged mapping can be left alone.
    struct Mapping {
        u32 size;
        const Common::HostSharedMemory* block;
        std::size_t offset;

        bool operator==(const Mapping&) const = default;
    };

    Memory::MemorySystem& memory;
    bool usable{};
    /// Guest ranges currently mirrored, keyed by guest base address, and kept disjoint.
    std::map<VAddr, Mapping> live_mappings;
    /// Guards live_mappings between the emulation thread (mapping changes) and whoever arms
    /// pages (the render thread, through ProtectHostRange).
    mutable std::recursive_mutex mappings_mutex;
    /// One entry per guest page of the reservation: the emulator's host page it mirrors, or
    /// null. Kept alongside live_mappings so the fault handler can look up without a lock.
    std::vector<std::atomic<const u8*>> page_to_host;
    void SetPageMap(VAddr base, u32 size, const u8* host);
};

/**
 * Claims the guest's address range for the lifetime of the process.
 *
 * Worth calling as the first thing a frontend does. The range starts at 0x00100000, which on a
 * 32-bit host is low enough that the heap can grow into it: by the time a title is loaded the
 * range may already be occupied, and then guest code has nowhere to live. Reserving at startup,
 * before anything has had a chance to allocate, makes that failure impossible rather than
 * unlikely. Calling it more than once is harmless, and not calling it at all only means the
 * reservation is attempted later, when it may fail.
 */
bool NativeReserveGuestAddressSpace();

} // namespace Core
