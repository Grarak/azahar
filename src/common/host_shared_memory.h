// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <memory>
#include "common/common_types.h"

namespace Common {

/**
 * A block of memory that can be mapped into the host address space more than once.
 *
 * Ordinary allocations cannot do this, and the native ARM backend needs it: guest memory has to be
 * reachable both through the emulator's own pointer and at the address the guest itself uses, with
 * writes through either visible to the other. On Linux the block is a memfd, so a second mapping is
 * a plain mmap of the same file. Elsewhere it degrades to a single anonymous allocation and
 * MapAt() fails, which leaves every existing backend working exactly as before.
 */
class HostSharedMemory {
public:
    /**
     * Where a block should come from, on a console where that is a real choice.
     *
     * The Vita's physically contiguous pool is a separate ~26 MB the game partition's heap
     * cannot reach. It is a scarce, named resource, not a faster heap, so a block only asks
     * for it when the whole point of the block is to live there - which today means the
     * title's RomFS cache and nothing else. Guest memory does not: FCRAM is 128 MB and could
     * never fit, and asking anyway made the kernel print an allocation failure at every boot.
     * A block that asks and cannot be served falls back to the heap.
     */
    enum class Placement {
        Heap,
        PhysicallyContiguous,
    };

    explicit HostSharedMemory(std::size_t size, const char* name,
                              Placement placement = Placement::Heap);
    ~HostSharedMemory();

    HostSharedMemory(const HostSharedMemory&) = delete;
    HostSharedMemory& operator=(const HostSharedMemory&) = delete;

    /// The emulator's own view of the block. Always valid.
    u8* Data() const noexcept {
        return base;
    }

    std::size_t Size() const noexcept {
        return size;
    }

    /// Whether this block can be mapped at a second host address.
    bool SupportsAliasing() const noexcept {
        return fd >= 0;
    }

    /**
     * Maps part of this block at a fixed host address, replacing whatever is mapped there.
     * @param host_address Address to map at. Must be page aligned.
     * @param offset       Offset into this block. Must be page aligned.
     * @param length       Bytes to map. Must be page aligned.
     * @param executable   Whether the mapping should permit instruction fetch.
     * @returns true if the mapping was established at exactly host_address.
     */
    bool MapAt(void* host_address, std::size_t offset, std::size_t length, bool executable) const;

    /// Removes a mapping previously made by MapAt, leaving the address unmapped.
    static bool UnmapAt(void* host_address, std::size_t length);

private:
    u8* base{};
    void* raw{}; // the unaligned allocation base holding it, where the alignment is done by hand
    int memblock{-1}; // Vita: the kernel block holding it instead, when one could be had
    std::size_t size{};
    int fd{-1};
    bool owns_mapping{};
};

} // namespace Common
