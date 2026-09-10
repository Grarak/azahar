// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/arm/native/native_mirror.h"

#include <algorithm>
#include <iterator>

#include "common/host_shared_memory.h"
#include "common/logging/log.h"
#include "core/arm/native/native_executor.h"

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace Core {

NativeAddressSpaceMirror::NativeAddressSpaceMirror(Memory::MemorySystem& memory_)
    : memory{memory_} {}

NativeAddressSpaceMirror::~NativeAddressSpaceMirror() {
    // The mapped pages go, but the reservation itself is kept for the life of the process: giving
    // it back would let the heap take the range, and a second title could then not be loaded.
    if (usable) {
        UnmapAll();
    }
}

namespace {
/// Whether the process-wide reservation has been made, and whether it succeeded.
bool reservation_attempted = false;
bool reservation_held = false;
} // Anonymous namespace

bool NativeReserveGuestAddressSpace() {
#ifdef __linux__
    if (reservation_attempted) {
        return reservation_held;
    }
    reservation_attempted = true;

    // One PROT_NONE reservation over the whole guest range. Individual pages are mapped over it as
    // the guest maps them; the rest stays unreadable, so a stray guest access faults instead of
    // hitting an unrelated host allocation.
    constexpr u32 base = NativeAddressSpaceMirror::GuestBase;
    constexpr u32 end = NativeAddressSpaceMirror::GuestEnd;
    void* wanted = reinterpret_cast<void*>(base);
    void* got = mmap(wanted, end - base, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (got != wanted) {
        if (got != MAP_FAILED) {
            munmap(got, end - base);
        }
        LOG_ERROR(Core_ARM11,
                  "Native backend: could not reserve the guest address range {:#010x}-{:#010x}; "
                  "something else in this process already occupies it",
                  base, end);
        return false;
    }
    reservation_held = true;
    LOG_INFO(Core_ARM11, "Native backend: reserved guest address range {:#010x}-{:#010x}", base,
             end);
    return true;
#else
    return false;
#endif
}

bool NativeAddressSpaceMirror::Reserve() {
    usable = NativeReserveGuestAddressSpace();
    return usable;
}

void NativeAddressSpaceMirror::OnMapped(VAddr base, u32 size, MemoryRef target) {
    if (!usable || base < GuestBase || base + size > GuestEnd) {
        return;
    }

    if (target.GetPtr() == nullptr) {
        OnUnmapped(base, size);
        return;
    }

    std::size_t offset{};
    const Common::HostSharedMemory* block = target.AliasableBlock(offset);
    if (block == nullptr) {
        // Backing memory that cannot be aliased cannot be reached by natively executing guest
        // code. Leave it unmapped so the guest faults on it, which is a far better failure than
        // executing against memory that is not really the guest's.
        LOG_WARNING(Core_ARM11, "Native backend: {:#010x}+{:#x} has no aliasable backing", base,
                    size);
        OnUnmapped(base, size);
        return;
    }

    // Remapping a range to what it already points at costs a syscall and an instruction cache
    // flush for nothing, and the page table is re-reported far more often than it changes.
    const Mapping wanted{size, block, offset};
    const auto existing = live_mappings.find(base);
    if (existing != live_mappings.end() && existing->second == wanted) {
        return;
    }

    std::scoped_lock lock{mappings_mutex};
    void* guest_address = reinterpret_cast<void*>(base);
    if (!block->MapAt(guest_address, offset, size, true)) {
        LOG_ERROR(Core_ARM11, "Native backend: failed to mirror {:#010x}+{:#x}", base, size);
        OnUnmapped(base, size);
        return;
    }

    // The emulator wrote this memory through its own mapping, so anything the guest is about to
    // execute from it may still be sitting in the data cache.
    NativeFlushInstructionCache(guest_address, size);

    // MAP_FIXED has already replaced every host mapping inside this range in one step, so the
    // entries describing them are stale. Only what they held outside the range is still real.
    ForgetRange(base, size, false);
    live_mappings.emplace(base, wanted);
    SetPageMap(base, size, block->Data() + offset);
}

void NativeAddressSpaceMirror::OnUnmapped(VAddr base, u32 size) {
    if (!usable || base < GuestBase || base + size > GuestEnd) {
        return;
    }
    std::scoped_lock lock{mappings_mutex};
    ForgetRange(base, size, true);
    SetPageMap(base, size, nullptr);
}

void NativeAddressSpaceMirror::SetPageMap(VAddr base, u32 size, const u8* host) {
    if (page_to_host.empty()) {
        page_to_host = std::vector<std::atomic<const u8*>>((GuestEnd - GuestBase) / 4096);
    }
    const VAddr first = base & ~VAddr{4095};
    for (VAddr page = first; page < base + size; page += 4096) {
        const std::size_t index = (page - GuestBase) / 4096;
        if (index < page_to_host.size()) {
            page_to_host[index].store(host != nullptr ? host + (page - first) : nullptr,
                                      std::memory_order_release);
        }
    }
}

const u8* NativeAddressSpaceMirror::HostPageOf(VAddr guest) const {
    if (guest < GuestBase || guest >= GuestEnd || page_to_host.empty()) {
        return nullptr;
    }
    return page_to_host[(guest - GuestBase) / 4096].load(std::memory_order_acquire);
}

void NativeAddressSpaceMirror::ProtectHostRange(const u8* host, std::size_t size, bool read_only) {
    std::scoped_lock lock{mappings_mutex};
    const int prot = read_only ? (PROT_READ | PROT_EXEC) : (PROT_READ | PROT_WRITE | PROT_EXEC);
    for (const auto& [base, mapping] : live_mappings) {
        const u8* mapped = mapping.block->Data() + mapping.offset;
        const u8* lo = std::max(mapped, host);
        const u8* hi = std::min(mapped + mapping.size, host + size);
        if (lo >= hi) {
            continue;
        }
        void* guest = reinterpret_cast<void*>(base + static_cast<u32>(lo - mapped));
        if (mprotect(guest, static_cast<std::size_t>(hi - lo), prot) != 0) {
            LOG_ERROR(Core_ARM11, "Native backend: mprotect of {} failed", guest);
        }
    }
}

void NativeAddressSpaceMirror::ForgetRange(VAddr base, u32 size, bool unmap) {
    if (size == 0) {
        return;
    }
    const VAddr end = base + size;

    // The entry covering base can start before it, and std::map can only find it by its own base,
    // so the walk starts one entry earlier when that one still reaches into the range.
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
            Common::HostSharedMemory::UnmapAt(reinterpret_cast<void*>(overlap_base),
                                              overlap_end - overlap_base);
        }

        // What lies outside the range is untouched host mapping and has to stay described, so the
        // entry is put back as the pieces that survived. A block offset is contiguous across an
        // entry, so a piece's offset moves with its base.
        if (entry_base < overlap_base) {
            live_mappings.emplace(entry_base,
                                  Mapping{overlap_base - entry_base, entry.block, entry.offset});
        }
        if (overlap_end < entry_end) {
            live_mappings.emplace(overlap_end, Mapping{entry_end - overlap_end, entry.block,
                                                       entry.offset + (overlap_end - entry_base)});
        }
    }
}

void NativeAddressSpaceMirror::OnPageTableChanged() {
    UnmapAll();
}

void NativeAddressSpaceMirror::FlushInstructionCache() const {
    // Only what is actually mapped. Sweeping the whole reserved range would hand the kernel a
    // gigabyte to walk, which measured at over a millisecond a call.
    for (const auto& [base, mapping] : live_mappings) {
        NativeFlushInstructionCache(reinterpret_cast<void*>(base), mapping.size);
    }
}

void NativeAddressSpaceMirror::UnmapAll() {
    std::scoped_lock lock{mappings_mutex};
    SetPageMap(GuestBase, GuestEnd - GuestBase, nullptr);
    for (const auto& [base, mapping] : live_mappings) {
        Common::HostSharedMemory::UnmapAt(reinterpret_cast<void*>(base), mapping.size);
    }
    live_mappings.clear();
}

} // namespace Core
