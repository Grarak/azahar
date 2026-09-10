// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include <cstring>
#include "common/host_shared_memory.h"
#include "common/logging/log.h"
#ifdef __vita__
#include <psp2/kernel/sysmem.h>
#endif

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#endif

namespace Common {

#ifdef __linux__

static int CreateMemfd(const char* name, std::size_t size) {
#ifdef SYS_memfd_create
    int fd = static_cast<int>(syscall(SYS_memfd_create, name, MFD_CLOEXEC));
#else
    int fd = -1;
#endif
    if (fd < 0) {
        return -1;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

HostSharedMemory::HostSharedMemory(std::size_t size_, const char* name,
                                   [[maybe_unused]] Placement placement)
    : size{size_} {
    fd = CreateMemfd(name, size);
    if (fd >= 0) {
        void* mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapping != MAP_FAILED) {
            base = static_cast<u8*>(mapping);
            owns_mapping = true;
            return;
        }
        LOG_WARNING(Common, "memfd mapping for {} failed, falling back to a plain allocation",
                    name);
        close(fd);
        fd = -1;
    }

    // No aliasing available; every existing backend still works from this pointer.
    if (size == 0) {
        // A region the emulated console does not have. Nothing may be mapped through it, and a
        // zero-byte allocation is not required to return anything usable.
        base = nullptr;
        owns_mapping = false;
        return;
    }
    base = static_cast<u8*>(std::calloc(1, size));
    owns_mapping = false;
}

HostSharedMemory::~HostSharedMemory() {
    if (owns_mapping) {
        munmap(base, size);
    } else {
        std::free(base);
    }
    if (fd >= 0) {
        close(fd);
    }
}

bool HostSharedMemory::MapAt(void* host_address, std::size_t offset, std::size_t length,
                             bool executable) const {
    if (fd < 0 || offset + length > size) {
        return false;
    }
    const int prot = PROT_READ | PROT_WRITE | (executable ? PROT_EXEC : 0);
    void* result = mmap(host_address, length, prot, MAP_SHARED | MAP_FIXED, fd,
                        static_cast<off_t>(offset));
    return result == host_address;
}

bool HostSharedMemory::UnmapAt(void* host_address, std::size_t length) {
    // A PROT_NONE anonymous mapping rather than munmap: it keeps the address reserved, so a later
    // host allocation cannot wander into the guest's address space.
    void* result = mmap(host_address, length, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    return result == host_address;
}

#else

HostSharedMemory::HostSharedMemory(std::size_t size_, [[maybe_unused]] const char* name,
                                   [[maybe_unused]] Placement placement)
    : size{size_} {
    if (size == 0) {
        return;
    }
#if defined(__vita__)
    // The PS Vita's native backend maps these pages into the guest's own translation table by
    // physical address, one 4 KB page at a time, so the block has to start on a page and be a
    // whole number of them. Not aligned_alloc: vitasdk's newlib hands back ordinary malloc
    // alignment from it, so the alignment is done by hand - allocate a page of slack, keep the
    // raw pointer for the free, and point Data() at the first page boundary inside it.
    const std::size_t rounded = (size + 0xFFFu) & ~static_cast<std::size_t>(0xFFFu);
    // Only a block that asked for the physically contiguous pool goes there. It used to be
    // taken by anything a megabyte or over, which meant guest FCRAM asked for 128 MB of a
    // ~26 MB pool at every boot and the kernel printed its refusal three times before the
    // fallback; the pool is for the RomFS cache, which is sized to it.
    if (placement == Placement::PhysicallyContiguous) {
        const std::size_t mb_rounded = (rounded + 0xFFFFFu) & ~static_cast<std::size_t>(0xFFFFFu);
        const SceUID uid = sceKernelAllocMemBlock(
            name ? name : "azahar-shared", SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
            static_cast<SceSize>(mb_rounded), nullptr);
        void* addr = nullptr;
        if (uid >= 0 && sceKernelGetMemBlockBase(uid, &addr) >= 0 && addr != nullptr) {
            memblock = uid;
            base = static_cast<u8*>(addr);
            std::memset(base, 0, rounded);
            LOG_INFO(Common, "memblock {}: {} KiB PHYCONT_RW -> {:#x} (the guest's memory)",
                     name ? name : "?", mb_rounded >> 10, static_cast<u32>(uid));
            return;
        }
        if (uid >= 0) {
            sceKernelFreeMemBlock(uid);
        }
    }
    raw = std::malloc(rounded + 0x1000);
    if (raw != nullptr) {
        base = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(raw) + 0xFFFu) &
                                     ~static_cast<uintptr_t>(0xFFFu));
        std::memset(base, 0, rounded);
    }
#else
    base = static_cast<u8*>(std::calloc(1, size));
#endif
}

HostSharedMemory::~HostSharedMemory() {
#ifdef __vita__
    if (memblock >= 0) {
        sceKernelFreeMemBlock(memblock);
        return;
    }
#endif
    std::free(raw != nullptr ? raw : base);
}

bool HostSharedMemory::MapAt(void*, std::size_t, std::size_t, bool) const {
    return false;
}

bool HostSharedMemory::UnmapAt(void*, std::size_t) {
    return false;
}

#endif

} // namespace Common
