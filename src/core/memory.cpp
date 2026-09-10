// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <csignal>
#include <cstring>
#include <boost/serialization/array.hpp>
#include <boost/serialization/binary_object.hpp>
#include "audio_core/dsp_interface.h"
#include "common/archives.h"
#include "common/assert.h"
#include "common/atomic_ops.h"
#include "common/common_types.h"
#include "common/host_shared_memory.h"
#include "common/logging/log.h"
#include "common/optional_helper.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#ifdef ENABLE_GDBSTUB
#include "core/gdbstub/gdbstub.h"
#endif
#include "core/global.h"
#include "core/hle/kernel/process.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"

SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::FCRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::VRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::DSP>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::N3DS>)

#ifndef SIGTRAP
constexpr u32 SIGTRAP = 5;
#endif

#ifndef SIGSEGV
constexpr u32 SIGSEGV = 11;
#endif

namespace Memory {

MemoryRef PageTable::Pointers::Ref(std::size_t idx) const {
    const u32 page = static_cast<u32>(idx);
    auto it = runs.upper_bound(page);
    if (it == runs.begin()) {
        return {};
    }
    --it;
    if (it->first + it->second.pages <= page) {
        return {};
    }
    return MemoryRef(it->second.ref.GetBackingMem(),
                     it->second.ref.GetOffset() +
                         static_cast<u64>(page - it->first) * CITRA_PAGE_SIZE);
}

void PageTable::Pointers::SetRef(std::size_t idx, MemoryRef value) {
    const u32 page = static_cast<u32>(idx);
    // Whatever run covers this page loses it: split around it.
    auto it = runs.upper_bound(page);
    if (it != runs.begin()) {
        --it;
        const u32 start = it->first;
        if (start + it->second.pages > page) {
            const Run covering = it->second;
            const u32 before = page - start;
            const u32 after = start + covering.pages - page - 1;
            runs.erase(it);
            if (before) {
                runs.emplace(start, Run{before, covering.ref});
            }
            if (after) {
                runs.emplace(page + 1,
                             Run{after, MemoryRef(covering.ref.GetBackingMem(),
                                                  covering.ref.GetOffset() +
                                                      static_cast<u64>(before + 1) *
                                                          CITRA_PAGE_SIZE)});
            }
        }
    }
    if (!value) {
        return;
    }
    const auto same_block = [&](const MemoryRef& a, const MemoryRef& b, u64 a_to_b_bytes) {
        return a.GetBackingMem() == b.GetBackingMem() && a.GetOffset() + a_to_b_bytes == b.GetOffset();
    };
    auto succ = runs.find(page + 1);
    const bool join_succ =
        succ != runs.end() && same_block(value, succ->second.ref, CITRA_PAGE_SIZE);
    auto after_page = runs.lower_bound(page);
    if (after_page != runs.begin()) {
        auto pred = std::prev(after_page);
        if (pred->first + pred->second.pages == page &&
            same_block(pred->second.ref, value,
                       static_cast<u64>(pred->second.pages) * CITRA_PAGE_SIZE)) {
            pred->second.pages++;
            if (join_succ) {
                pred->second.pages += succ->second.pages;
                runs.erase(succ);
            }
            return;
        }
    }
    if (join_succ) {
        Run joined{succ->second.pages + 1, std::move(value)};
        runs.erase(succ);
        runs.emplace(page, std::move(joined));
        return;
    }
    runs.emplace(page, Run{1, std::move(value)});
}

void PageTable::Clear() {
    pointers.Clear();
    attributes.fill(PageType::Unmapped);
}

class RasterizerCacheMarker {
public:
    void Mark(VAddr addr, bool cached) {
        bool* p = At(addr);
        if (p)
            *p = cached;
    }

    bool IsCached(VAddr addr) {
        bool* p = At(addr);
        if (p)
            return *p;
        return false;
    }

private:
    bool* At(VAddr addr) {
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return &vram[(addr - VRAM_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return &linear_heap[(addr - LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return &new_linear_heap[(addr - NEW_LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= PLUGIN_3GX_FB_VADDR && addr < PLUGIN_3GX_FB_VADDR_END) {
            return &plugin_fb[(addr - PLUGIN_3GX_FB_VADDR) / CITRA_PAGE_SIZE];
        }
        return nullptr;
    }

    std::array<bool, VRAM_SIZE / CITRA_PAGE_SIZE> vram{};
    std::array<bool, LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> linear_heap{};
    std::array<bool, NEW_LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> new_linear_heap{};
    std::array<bool, PLUGIN_3GX_FB_SIZE / CITRA_PAGE_SIZE> plugin_fb{};

    static_assert(sizeof(bool) == 1);
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & vram;
        ar & linear_heap;
        ar & new_linear_heap;
        ar & plugin_fb;
    }
};

class MemorySystem::Impl {
public:
    // Visual Studio would try to allocate these on compile time
    // if they are std::array which would exceed the memory limit.
    // Aliasable so that the native ARM backend can map these same pages at the addresses the guest
    // uses. Falls back to a plain allocation where the host cannot alias, which every other
    // backend is indifferent to.
    Common::HostSharedMemory fcram{Memory::FCRAM_SIZE, "azahar-fcram"};
    Common::HostSharedMemory vram{Memory::VRAM_SIZE, "azahar-vram"};
    // Only a New 3DS has this, and only an Old 3DS is emulated, so it is never allocated. The
    // block stays declared because the physical-address paths still name the region.
    Common::HostSharedMemory n3ds_extra_ram{0, "azahar-n3ds-ram"};
    Common::HostSharedMemory dsp_ram{Memory::DSP_RAM_SIZE, "azahar-dsp-ram"};

    MappingObserver* mapping_observer{};
    bool rasterizer_marking_enabled = true;

    Core::System& system;
    std::shared_ptr<PageTable> current_page_table = nullptr;
    RasterizerCacheMarker cache_marker;
    std::vector<std::shared_ptr<PageTable>> page_table_list;

    std::shared_ptr<BackingMem> fcram_mem;
    std::shared_ptr<BackingMem> vram_mem;
    std::shared_ptr<BackingMem> n3ds_extra_ram_mem;
    std::shared_ptr<BackingMem> dsp_mem;

    PAddr plugin_fb_address{};

    Impl(Core::System& system_);

    const u8* GetPtr(Region r) const {
        switch (r) {
        case Region::VRAM:
            return vram.Data();
        case Region::DSP:
            return dsp_ram.Data();
        case Region::FCRAM:
            return fcram.Data();
        case Region::N3DS:
            return n3ds_extra_ram.Data();
        default:
            UNREACHABLE();
        }
    }

    u8* GetPtr(Region r) {
        switch (r) {
        case Region::VRAM:
            return vram.Data();
        case Region::DSP:
            return dsp_ram.Data();
        case Region::FCRAM:
            return fcram.Data();
        case Region::N3DS:
            return n3ds_extra_ram.Data();
        default:
            UNREACHABLE();
        }
    }

    const Common::HostSharedMemory& GetBlock(Region r) const {
        switch (r) {
        case Region::VRAM:
            return vram;
        case Region::DSP:
            return dsp_ram;
        case Region::FCRAM:
            return fcram;
        case Region::N3DS:
            return n3ds_extra_ram;
        default:
            UNREACHABLE();
        }
    }

    u32 GetSize(Region r) const {
        // From the block itself: FCRAM and the New 3DS extra RAM are sized to the console being
        // emulated, and a constant here would describe memory that was never allocated.
        return static_cast<u32>(GetBlock(r).Size());
    }

    u32 GetPC() const noexcept {
        return system.GetRunningCore().GetPC();
    }

    template <bool UNSAFE>
    void ReadBlockImpl(const Kernel::Process& process, const VAddr src_addr, void* dest_buffer,
                       const std::size_t size) {
        auto& page_table = *process.vm_manager.page_table;

        std::size_t remaining_size = size;
        std::size_t page_index = src_addr >> CITRA_PAGE_BITS;
        std::size_t page_offset = src_addr & CITRA_PAGE_MASK;

        while (remaining_size > 0) {
            const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
            const VAddr current_vaddr =
                static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

            switch (page_table.attributes[page_index]) {
            case PageType::Unmapped: {
                LOG_ERROR(
                    HW_Memory,
                    "unmapped ReadBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                    "0x{:08X}",
                    current_vaddr, src_addr, size, GetPC());
                std::memset(dest_buffer, 0, copy_amount);
                break;
            }
            case PageType::Memory: {
                DEBUG_ASSERT(page_table.pointers[page_index]);

                const u8* src_ptr = page_table.pointers[page_index] + page_offset;
                std::memcpy(dest_buffer, src_ptr, copy_amount);
                break;
            }
            case PageType::MemoryWatchpoint: {
                auto it = page_table.watchpoint_pages_map.find(page_index);
                ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                           "Missing memory for watchpoint page");

                const u8* src_ptr = it->second.memory.GetPtr() + page_offset;
                std::memcpy(dest_buffer, src_ptr, copy_amount);
                break;
            }
            case PageType::RasterizerCachedMemory:
            case PageType::RasterizerCachedMemoryWatchpoint: {
                if constexpr (!UNSAFE) {
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Flush);
                }
                std::memcpy(dest_buffer, GetPointerForRasterizerCache(current_vaddr), copy_amount);
                break;
            }
            default:
                UNREACHABLE();
            }

            page_index++;
            page_offset = 0;
            dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
            remaining_size -= copy_amount;
        }
    }

    template <bool UNSAFE>
    void WriteBlockImpl(const Kernel::Process& process, const VAddr dest_addr,
                        const void* src_buffer, const std::size_t size) {
        auto& page_table = *process.vm_manager.page_table;
        std::size_t remaining_size = size;
        std::size_t page_index = dest_addr >> CITRA_PAGE_BITS;
        std::size_t page_offset = dest_addr & CITRA_PAGE_MASK;

        while (remaining_size > 0) {
            const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
            const VAddr current_vaddr =
                static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

            switch (page_table.attributes[page_index]) {
            case PageType::Unmapped: {
                LOG_ERROR(
                    HW_Memory,
                    "unmapped WriteBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                    "0x{:08X}",
                    current_vaddr, dest_addr, size, GetPC());
                break;
            }
            case PageType::Memory: {
                DEBUG_ASSERT(page_table.pointers[page_index]);

                u8* dest_ptr = page_table.pointers[page_index] + page_offset;
                std::memcpy(dest_ptr, src_buffer, copy_amount);
                break;
            }
            case PageType::MemoryWatchpoint: {
                auto it = page_table.watchpoint_pages_map.find(page_index);
                ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                           "Missing memory for watchpoint page");

                u8* dest_ptr = it->second.memory.GetPtr() + page_offset;
                std::memcpy(dest_ptr, src_buffer, copy_amount);
                break;
            }
            case PageType::RasterizerCachedMemory:
            case PageType::RasterizerCachedMemoryWatchpoint: {
                if constexpr (!UNSAFE) {
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Invalidate);
                }
                std::memcpy(GetPointerForRasterizerCache(current_vaddr), src_buffer, copy_amount);
                break;
            }
            default:
                UNREACHABLE();
            }

            page_index++;
            page_offset = 0;
            src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
            remaining_size -= copy_amount;
        }
    }

    MemoryRef GetPointerForRasterizerCache(VAddr addr) const {
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - LINEAR_HEAP_VADDR};
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - NEW_LINEAR_HEAP_VADDR};
        }
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return {vram_mem, addr - VRAM_VADDR};
        }
        if (addr >= PLUGIN_3GX_FB_VADDR && addr < PLUGIN_3GX_FB_VADDR_END && plugin_fb_address) {
            return {fcram_mem, addr - PLUGIN_3GX_FB_VADDR + plugin_fb_address - FCRAM_PADDR};
        }

        UNREACHABLE();
        return MemoryRef{};
    }

    void RasterizerFlushVirtualRegion(VAddr start, u32 size, FlushMode mode) {
        const VAddr end = start + size;

        auto CheckRegion = [&](VAddr region_start, VAddr region_end, PAddr paddr_region_start) {
            if (start >= region_end || end <= region_start) {
                // No overlap with region
                return;
            }

            VAddr overlap_start = std::max(start, region_start);
            VAddr overlap_end = std::min(end, region_end);
            PAddr physical_start = paddr_region_start + (overlap_start - region_start);
            u32 overlap_size = overlap_end - overlap_start;

            // Through the threaded facade: the raw GL rasterizer's caches may only be touched
            // on the render thread, and these flushes arrive from the emulation thread (DMA,
            // y2r, applet captures) and even filesystem worker threads.
            auto* rasterizer = system.GPU().CacheRasterizer();
            switch (mode) {
            case FlushMode::Flush:
                rasterizer->FlushRegion(physical_start, overlap_size);
                break;
            case FlushMode::Invalidate:
                rasterizer->InvalidateRegion(physical_start, overlap_size);
                break;
            case FlushMode::FlushAndInvalidate:
                rasterizer->FlushAndInvalidateRegion(physical_start, overlap_size);
                break;
            }
        };

        CheckRegion(LINEAR_HEAP_VADDR, LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
        CheckRegion(NEW_LINEAR_HEAP_VADDR, NEW_LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
        CheckRegion(VRAM_VADDR, VRAM_VADDR_END, VRAM_PADDR);
        if (plugin_fb_address) {
            CheckRegion(PLUGIN_3GX_FB_VADDR, PLUGIN_3GX_FB_VADDR_END, plugin_fb_address);
        }
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        // Kept in the stream so the savestate layout does not move; always false now.
        bool save_n3ds_ram = false;
        ar & save_n3ds_ram;
        ar& boost::serialization::make_binary_object(vram.Data(), Memory::VRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            fcram.Data(), save_n3ds_ram ? Memory::FCRAM_N3DS_SIZE : Memory::FCRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            n3ds_extra_ram.Data(), save_n3ds_ram ? Memory::N3DS_EXTRA_RAM_SIZE : 0);
        ar& boost::serialization::make_binary_object(dsp_ram.Data(), Memory::DSP_RAM_SIZE);
        ar & cache_marker;
        ar & page_table_list;
        // dsp is set from Core::System at startup
        ar & current_page_table;
        ar & fcram_mem;
        ar & vram_mem;
        ar & n3ds_extra_ram_mem;
        ar & dsp_mem;
        ar & plugin_fb_address;
    }
};

// We use this rather than BufferMem because we don't want new objects to be allocated when
// deserializing. This avoids unnecessary memory thrashing.
template <Region R>
class MemorySystem::BackingMemImpl : public BackingMem {
public:
    BackingMemImpl() : impl(*Core::Global<Core::System>().Memory().impl) {}
    explicit BackingMemImpl(MemorySystem::Impl& impl_) : impl(impl_) {}
    u8* GetPtr() override {
        return impl.GetPtr(R);
    }
    const u8* GetPtr() const override {
        return impl.GetPtr(R);
    }
    std::size_t GetSize() const override {
        return impl.GetSize(R);
    }

    const Common::HostSharedMemory* AliasableBlock() const override {
        const Common::HostSharedMemory& block = impl.GetBlock(R);
        return block.SupportsAliasing() ? &block : nullptr;
    }

private:
    MemorySystem::Impl& impl;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<BackingMem>(*this);
    }
    friend class boost::serialization::access;
};

MemorySystem::Impl::Impl(Core::System& system_)
    : system{system_}, fcram_mem(std::make_shared<BackingMemImpl<Region::FCRAM>>(*this)),
      vram_mem(std::make_shared<BackingMemImpl<Region::VRAM>>(*this)),
      n3ds_extra_ram_mem(std::make_shared<BackingMemImpl<Region::N3DS>>(*this)),
      dsp_mem(std::make_shared<BackingMemImpl<Region::DSP>>(*this)) {}

MemorySystem::MemorySystem(Core::System& system)
    : impl(std::make_unique<Impl>(system)), write_tracker(std::make_unique<GuestWriteTracker>()) {
    write_tracker->RegisterBlock(impl->fcram.Data(), impl->fcram.Size(), FCRAM_PADDR);
    write_tracker->RegisterBlock(impl->vram.Data(), impl->vram.Size(), VRAM_PADDR);
}
MemorySystem::~MemorySystem() = default;

template <class Archive>
void MemorySystem::serialize(Archive& ar, const unsigned int file_version) {
    ar&* impl.get();
}

SERIALIZE_IMPL(MemorySystem)

void MemorySystem::SetCurrentPageTable(const std::shared_ptr<PageTable>& page_table) {
    // Called on every guest thread switch, almost always with the table already in force. Rebuilding
    // the observer's view of a million pages each time would cost far more than the emulation.
    const bool changed = impl->current_page_table != page_table;
    impl->current_page_table = page_table;

    if (changed && impl->mapping_observer != nullptr) {
        impl->mapping_observer->OnPageTableChanged();
        if (page_table != nullptr) {
            NotifyMappingObserver(*page_table, 0, PAGE_TABLE_NUM_ENTRIES);
        }
    }
}

std::shared_ptr<PageTable> MemorySystem::GetCurrentPageTable() const {
    return impl->current_page_table;
}

void MemorySystem::RasterizerFlushVirtualRegion(VAddr start, u32 size, FlushMode mode) {
    impl->RasterizerFlushVirtualRegion(start, size, mode);
}

PAddr& Memory::MemorySystem::Plugin3GXFramebufferAddress() {
    return impl->plugin_fb_address;
}

void MemorySystem::RegisterWatchpoint(const Kernel::Process& process, VAddr addr, u32 size) {
    auto& page_table = *process.vm_manager.page_table;

    VAddr current = addr;
    VAddr end = addr + size;

    while (current < end) {
        const VAddr page_base = (current & ~CITRA_PAGE_MASK);
        const VAddr page_index = page_base >> CITRA_PAGE_BITS;

        auto it = page_table.watchpoint_pages_map.find(page_index);
        if (it != page_table.watchpoint_pages_map.end()) {
            // Nothing to do, only increment count.
            it->second.watchpoint_count++;
        } else {
            MemoryRef mem;
            PageType& type = page_table.attributes[page_index];

            switch (type) {
            case PageType::Memory:
                mem = page_table.pointers.Ref(page_index);
                type = PageType::MemoryWatchpoint;
                page_table.pointers[page_index] = nullptr;
                break;
            case PageType::RasterizerCachedMemory:
                mem = GetPointerForRasterizerCache(page_base);
                type = PageType::RasterizerCachedMemoryWatchpoint;
                break;
            default:
                LOG_ERROR(HW_Memory, "Cannot get pointer to register watchpoint for page 0x{:08X}",
                          page_base);
                continue;
            }

            page_table.watchpoint_pages_map.insert(
                {page_index,
                 PageTable::WatchpointPageInfo{.watchpoint_count = 1, .memory = std::move(mem)}});
        }

        current = page_base + CITRA_PAGE_SIZE;
    }
}

void MemorySystem::UnregisterWatchpoint(const Kernel::Process& process, VAddr addr, u32 size) {
    auto& page_table = *process.vm_manager.page_table;

    VAddr current = addr;
    VAddr end = addr + size;

    while (current < end) {
        const VAddr page_base = (current & ~CITRA_PAGE_MASK);
        const VAddr page_index = page_base >> CITRA_PAGE_BITS;

        auto it = page_table.watchpoint_pages_map.find(page_index);
        if (it != page_table.watchpoint_pages_map.end()) {
            if (--it->second.watchpoint_count == 0) {

                PageType& type = page_table.attributes[page_index];

                switch (type) {
                case PageType::MemoryWatchpoint:
                    type = PageType::Memory;
                    page_table.pointers[page_index] = it->second.memory;
                    break;
                case PageType::RasterizerCachedMemoryWatchpoint:
                    type = PageType::RasterizerCachedMemory;
                    break;
                default:
                    LOG_ERROR(HW_Memory, "Invalid watchpoint page type for page 0x{:08X}: {}",
                              page_base, static_cast<u8>(type));
                }

                page_table.watchpoint_pages_map.erase(page_index);
            }
        } else {
            LOG_ERROR(HW_Memory, "No watchpoint found on page 0x{:08X}", page_base);
        }

        current = page_base + CITRA_PAGE_SIZE;
    }
}

void MemorySystem::MapPages(PageTable& page_table, u32 base, u32 size, MemoryRef memory,
                            PageType type) {
    LOG_DEBUG(HW_Memory, "Mapping {} onto {:08X}-{:08X}", (void*)memory.GetPtr(),
              base * CITRA_PAGE_SIZE, (base + size) * CITRA_PAGE_SIZE);

    if (impl->system.IsPoweredOn()) {
        RasterizerFlushVirtualRegion(base << CITRA_PAGE_BITS, size * CITRA_PAGE_SIZE,
                                     FlushMode::FlushAndInvalidate);
    }

    const u32 first = base;
    u32 end = base + size;
    while (base != end) {
        ASSERT_MSG(base < PAGE_TABLE_NUM_ENTRIES, "out of range mapping at {:08X}", base);

        page_table.attributes[base] = type;
        page_table.pointers[base] = memory;

        // If the memory to map is already rasterizer-cached, mark the page
        if (type == PageType::Memory && impl->cache_marker.IsCached(base * CITRA_PAGE_SIZE)) {
            page_table.attributes[base] = PageType::RasterizerCachedMemory;
            page_table.pointers[base] = nullptr;
        }

        base += 1;
        if (memory != nullptr && memory.GetSize() > CITRA_PAGE_SIZE)
            memory += CITRA_PAGE_SIZE;
    }

    if (impl->mapping_observer != nullptr && &page_table == impl->current_page_table.get()) {
        NotifyMappingObserver(page_table, first, end);
    }
}

/**
 * Reports the state of pages [first, end) to the mapping observer, coalescing runs that are
 * contiguous in both guest address and host memory. Without the coalescing a 32 MiB mapping would
 * become thousands of separate host mappings.
 */
void MemorySystem::NotifyMappingObserver(PageTable& page_table, u32 first, u32 end) {
    auto* observer = impl->mapping_observer;

    u32 run_start = first;
    while (run_start < end) {
        const bool run_is_memory = page_table.attributes[run_start] == PageType::Memory &&
                                   page_table.pointers[run_start] != nullptr;

        u32 run_end = run_start + 1;
        if (run_is_memory) {
            // Extend while the host pointers stay as contiguous as the guest addresses.
            const u8* expected = page_table.pointers[run_start] + CITRA_PAGE_SIZE;
            while (run_end < end && page_table.attributes[run_end] == PageType::Memory &&
                   page_table.pointers[run_end] == expected) {
                expected += CITRA_PAGE_SIZE;
                run_end++;
            }
        } else {
            while (run_end < end && !(page_table.attributes[run_end] == PageType::Memory &&
                                      page_table.pointers[run_end] != nullptr)) {
                run_end++;
            }
        }

        const VAddr run_vaddr = run_start << CITRA_PAGE_BITS;
        const u32 run_size = (run_end - run_start) << CITRA_PAGE_BITS;
        if (run_is_memory) {
            observer->OnMapped(run_vaddr, run_size, page_table.pointers.Ref(run_start));
        } else {
            observer->OnUnmapped(run_vaddr, run_size);
        }

        run_start = run_end;
    }
}

void MemorySystem::SyncMappingObserver(PageTable& page_table) {
    if (impl->mapping_observer == nullptr) {
        return;
    }
    impl->mapping_observer->OnPageTableChanged();
    NotifyMappingObserver(page_table, 0, PAGE_TABLE_NUM_ENTRIES);
}

void MemorySystem::RefreshMappingObserver() {
    if (impl->mapping_observer == nullptr) {
        return;
    }
    impl->mapping_observer->OnPageTableChanged();
    if (impl->current_page_table != nullptr) {
        NotifyMappingObserver(*impl->current_page_table, 0, PAGE_TABLE_NUM_ENTRIES);
    }
}

void MemorySystem::SetMappingObserver(MappingObserver* observer) {
    impl->mapping_observer = observer;
    if (observer != nullptr && impl->current_page_table != nullptr) {
        observer->OnPageTableChanged();
        NotifyMappingObserver(*impl->current_page_table, 0, PAGE_TABLE_NUM_ENTRIES);
    }
}


void MemorySystem::MapMemoryRegion(PageTable& page_table, VAddr base, u32 size, MemoryRef target) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, target, PageType::Memory);
}

void MemorySystem::UnmapRegion(PageTable& page_table, VAddr base, u32 size) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, nullptr,
             PageType::Unmapped);
}

MemoryRef MemorySystem::GetPointerForRasterizerCache(VAddr addr) const {
    return impl->GetPointerForRasterizerCache(addr);
}

void MemorySystem::RegisterPageTable(std::shared_ptr<PageTable> page_table) {
    impl->page_table_list.push_back(page_table);
}

void MemorySystem::UnregisterPageTable(std::shared_ptr<PageTable> page_table) {
    auto it = std::find(impl->page_table_list.begin(), impl->page_table_list.end(), page_table);
    if (it != impl->page_table_list.end()) {
        impl->page_table_list.erase(it);
    }
}

template <typename T>
void MemorySystem::UnmappedAccess(const VAddr vaddr, const T value, bool read) {
    const std::string mode = (read ? "Read" : "Write");
    const std::string value_str = read ? std::string("") : fmt::format(" 0x{:08X}", value);
    const std::string message = fmt::format("unmapped {}{}{} @ 0x{:08X} at PC 0x{:08X}", mode,
                                            sizeof(T) * 8, value_str, vaddr, impl->GetPC());
#ifdef ENABLE_GDBSTUB
    if (GDBStub::IsConnected()) {
        GDBStub::Break(SIGSEGV);
    } else
#endif
        if (Settings::values.break_on_unmapped_memory_access) {
        impl->system.SetStatus(Core::System::ResultStatus::ErrorMemoryExceptionRaised,
                               message.c_str());
    }

    LOG_ERROR(HW_Memory, "{}", message);
}

template <typename T>
T MemorySystem::Read(const std::shared_ptr<PageTable>& page_table, const VAddr vaddr) {
    constexpr bool is_optional = is_optional_type<T>;
    using ReadType = optional_inner_or_type<T>;

    constexpr size_t read_size = sizeof(ReadType);

    const u8* page_pointer = page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        ReadType value;
        std::memcpy(&value, &page_pointer[vaddr & CITRA_PAGE_MASK], read_size);
        return value;
    }

    // Custom Luma3ds mapping (bit 31 set) Bypasses page tables for FCRAM/MMIO
    constexpr VAddr LUMA_ALIAS_BIT = 0x80000000u;

    if (vaddr & LUMA_ALIAS_BIT) [[unlikely]] {
        const PAddr paddr = vaddr & ~LUMA_ALIAS_BIT;

        // FCRAM (0x2xxxxxxx)
        if ((paddr & 0xF0000000) == Memory::FCRAM_PADDR) {
            ReadType value;
            std::memcpy(&value, GetFCRAMPointer(paddr - Memory::FCRAM_PADDR), read_size);
            return value;
        }

        // MMIO (0x1xxxxxxx, >= IO_AREA_PADDR) - Strictly 32-bit
        if ((paddr & 0xF0000000) == 0x10000000 && paddr >= Memory::IO_AREA_PADDR) [[unlikely]] {
            return static_cast<ReadType>(impl->system.GPU().ReadReg(
                static_cast<VAddr>(paddr) - Memory::IO_AREA_PADDR + 0x1EC00000));
        }
        // Fallthrough: Standard page table lookup
    }

    PageType type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped: {

        UnmappedAccess<ReadType>(vaddr, 0, true);

        if constexpr (is_optional) {
            return std::nullopt;
        } else {
            return T{};
        }
    }
    case PageType::Memory:
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::MemoryWatchpoint: {
        auto it = page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        ReadType value;
        std::memcpy(&value, it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK), read_size);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, read_size, GDBStub::BreakpointType::Read)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return value;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, read_size, FlushMode::Flush);

        ReadType value;
        std::memcpy(&value, GetPointerForRasterizerCache(vaddr), read_size);
        return value;
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, read_size, FlushMode::Flush);

        ReadType value;
        std::memcpy(&value, GetPointerForRasterizerCache(vaddr), read_size);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, read_size, GDBStub::BreakpointType::Read)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return value;
    }
    default:
        UNREACHABLE();
    }

    if constexpr (is_optional) {
        return std::nullopt;
    } else {
        return T{};
    }
}

template <typename T>
void MemorySystem::Write(const std::shared_ptr<PageTable>& page_table, const VAddr vaddr,
                         const T data) {
    u8* page_pointer = page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        std::memcpy(&page_pointer[vaddr & CITRA_PAGE_MASK], &data, sizeof(T));
        return;
    }

    // Custom Luma3ds mapping (bit 31 set) Bypasses page tables for FCRAM/MMIO
    constexpr VAddr LUMA_ALIAS_BIT = 0x80000000u;

    if (vaddr & LUMA_ALIAS_BIT) [[unlikely]] {
        const PAddr paddr = vaddr & ~LUMA_ALIAS_BIT;

        // FCRAM (0x2xxxxxxx)
        if ((paddr & 0xF0000000) == Memory::FCRAM_PADDR) {
            std::memcpy(GetFCRAMPointer(paddr - Memory::FCRAM_PADDR), &data, sizeof(T));
            return;
        }

        // MMIO (0x1xxxxxxx, >= IO_AREA_PADDR) - Strictly 32-bit
        if ((paddr & 0xF0000000) == 0x10000000 && paddr >= Memory::IO_AREA_PADDR) [[unlikely]] {
            ASSERT(sizeof(data) == sizeof(u32));
            impl->system.GPU().WriteReg(static_cast<VAddr>(paddr) - Memory::IO_AREA_PADDR +
                                            0x1EC00000,
                                        static_cast<u32>(data));
            return;
        }
        // Fallthrough: Standard page table lookup
    }

    PageType type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        (void)UnmappedAccess<T>(vaddr, data, false);
        return;
    case PageType::Memory:
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::MemoryWatchpoint: {
        auto it = page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        std::memcpy(it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK), &data, sizeof(T));

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        break;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        std::memcpy(GetPointerForRasterizerCache(vaddr), &data, sizeof(T));
        break;
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        std::memcpy(GetPointerForRasterizerCache(vaddr), &data, sizeof(T));

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        break;
    }
    default:
        UNREACHABLE();
    }
}

template <typename T>
bool MemorySystem::WriteExclusive(const VAddr vaddr, const T data, const T expected) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];

    if (page_pointer) {
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(&page_pointer[vaddr & CITRA_PAGE_MASK]);
        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }

    PageType type = impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        (void)UnmappedAccess<T>(vaddr, data, false);
        return true;
    case PageType::Memory:
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        return true;
    case PageType::MemoryWatchpoint: {
        auto it = impl->current_page_table->watchpoint_pages_map.find(vaddr >> CITRA_PAGE_BITS);
        ASSERT_MSG(it != impl->current_page_table->watchpoint_pages_map.end(),
                   "Missing memory for watchpoint page");

        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(it->second.memory.GetPtr() + (vaddr & CITRA_PAGE_MASK));

        bool ret = Common::AtomicCompareAndSwap(volatile_pointer, data, expected);

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return ret;
    }
    [[likely]] case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(GetPointerForRasterizerCache(vaddr).GetPtr());
        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }
    case PageType::RasterizerCachedMemoryWatchpoint: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        const auto volatile_pointer =
            reinterpret_cast<volatile T*>(GetPointerForRasterizerCache(vaddr).GetPtr());

#ifdef ENABLE_GDBSTUB
        if (GDBStub::CheckBreakpoint(vaddr, sizeof(T), GDBStub::BreakpointType::Write)) {
            GDBStub::Break(SIGTRAP);
        }
#endif

        return Common::AtomicCompareAndSwap(volatile_pointer, data, expected);
    }
    default:
        UNREACHABLE();
    }
    return true;
}

bool MemorySystem::IsValidVirtualAddress(const Kernel::Process& process, const VAddr vaddr) {
    auto& page_table = *process.vm_manager.page_table;

    auto page_pointer = page_table.pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return true;
    }

    if (page_table.attributes[vaddr >> CITRA_PAGE_BITS] != PageType::Unmapped) {
        return true;
    }

    return false;
}

bool MemorySystem::IsValidPhysicalAddress(const PAddr paddr) {
    return GetPhysicalRef(paddr);
}

u8* MemorySystem::GetPointer(const VAddr vaddr) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemory ||
        impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemoryWatchpoint) {
        return GetPointerForRasterizerCache(vaddr);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x} at PC 0x{:08X}", vaddr, impl->GetPC());
    return nullptr;
}

const u8* MemorySystem::GetPointer(const VAddr vaddr) const {
    const u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemory ||
        impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
            PageType::RasterizerCachedMemoryWatchpoint) {
        return GetPointerForRasterizerCache(vaddr);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x}", vaddr);
    return nullptr;
}

std::string MemorySystem::ReadCString(VAddr vaddr, std::size_t max_length) {
    std::string string;
    string.reserve(max_length);
    for (std::size_t i = 0; i < max_length; ++i) {
        char c = Read8(vaddr);
        if (c == '\0') {
            break;
        }

        string.push_back(c);
        ++vaddr;
    }

    string.shrink_to_fit();
    return string;
}

MemorySystem::PhysMemRegionInfo MemorySystem::GetPhysMemRegionInfo(PAddr address) {
    // This lookup runs concurrently on the emulation thread and the render thread. It used to go
    // through a shared single-entry cache, which one thread could rewrite while the other was
    // mid-read; the torn result paired one region's backing memory with another region's start,
    // yielding a well-formed pointer into the wrong part of guest RAM. The command-list parser
    // reading one chain segment through such a pointer walked stale bytes, never saw the list's
    // irq_request, and the title froze waiting for a P3D that had silently been dropped
    // (Smash at match entry, software renderer, measured as a parser read exactly
    // 0x7a00000 bytes off FCRAM's host base). The scan below is four compares against a
    // constexpr table - it does not need a cache, and it must not have a shared one.
    constexpr std::array memory_areas = {
        std::make_pair(VRAM_PADDR, VRAM_SIZE),
        std::make_pair(DSP_RAM_PADDR, DSP_RAM_SIZE),
        std::make_pair(FCRAM_PADDR, FCRAM_N3DS_SIZE),
        std::make_pair(N3DS_EXTRA_RAM_PADDR, N3DS_EXTRA_RAM_SIZE),
    };

    const auto area = std::find_if(memory_areas.begin(), memory_areas.end(), [&](const auto& area) {
        // Note: the region end check is inclusive because the user can pass in an address that
        // represents an open right bound
        return address >= area.first && address <= area.first + area.second;
    });

    if (area == memory_areas.end()) [[unlikely]] {
        LOG_ERROR(HW_Memory, "Unknown GetPhysMemRegionInfo @ {:#08X} at PC {:#08X}", address,
                  impl->GetPC());
        return PhysMemRegionInfo();
    }

    switch (area->first) {
    case VRAM_PADDR:
        return {&impl->vram_mem, area->first, area->second};
    case DSP_RAM_PADDR:
        return {&impl->dsp_mem, area->first, area->second};
    case FCRAM_PADDR:
        return {&impl->fcram_mem, area->first, area->second};
    case N3DS_EXTRA_RAM_PADDR:
        return {&impl->n3ds_extra_ram_mem, area->first, area->second};
    default:
        UNREACHABLE();
    }
}

u8* MemorySystem::GetPhysicalPointer(PAddr address) {
    auto target_mem = GetPhysMemRegionInfo(address);

    if (!target_mem.valid()) [[unlikely]] {
        return {nullptr};
    }

    u32 offset_into_region = address - target_mem.region_start;
    return target_mem.backing_mem->get()->GetPtr() + offset_into_region;
}

MemoryRef MemorySystem::GetPhysicalRef(PAddr address) {
    const auto& target_mem = GetPhysMemRegionInfo(address);

    if (!target_mem.valid()) [[unlikely]] {
        return {nullptr};
    }

    const u32 offset_into_region = address - target_mem.region_start;
    // A region can be matched while the offset still lands past its backing allocation (e.g.
    // fills or textures running past the end of VRAM). MemoryRef's constructor asserts on
    // that, which turned bad guest addresses into emulator crashes - notably from the
    // software blitter's own validity check. Report them as invalid instead.
    if (offset_into_region > (*target_mem.backing_mem)->GetSize()) [[unlikely]] {
        return {nullptr};
    }
    return {*target_mem.backing_mem, offset_into_region};
}

std::vector<VAddr> MemorySystem::PhysicalToVirtualAddressForRasterizer(PAddr addr) {
    if (addr >= VRAM_PADDR && addr < VRAM_PADDR_END) {
        return {addr - VRAM_PADDR + VRAM_VADDR};
    }
    // NOTE: Order matters here.
    PAddr plg_fb_addr = Plugin3GXFramebufferAddress();
    if (plg_fb_addr && addr >= plg_fb_addr && addr < plg_fb_addr + PLUGIN_3GX_FB_SIZE) {
        return {addr - plg_fb_addr + PLUGIN_3GX_FB_VADDR};
    }
    if (addr >= FCRAM_PADDR && addr < FCRAM_PADDR_END) {
        return {addr - FCRAM_PADDR + LINEAR_HEAP_VADDR, addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    if (addr >= FCRAM_PADDR_END && addr < FCRAM_N3DS_PADDR_END) {
        return {addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    // While the physical <-> virtual mapping is 1:1 for the regions supported by the cache,
    // some games (like Pokemon Super Mystery Dungeon) will try to use textures that go beyond
    // the end address of VRAM, causing the Virtual->Physical translation to fail when flushing
    // parts of the texture.
    LOG_ERROR(HW_Memory,
              "Trying to use invalid physical address for rasterizer: {:08X} at PC 0x{:08X}", addr,
              impl->GetPC());
    return {};
}

void MemorySystem::SetRasterizerCacheMarkingEnabled(bool enabled) {
    impl->rasterizer_marking_enabled = enabled;
}

bool MemorySystem::RasterizerCacheMarkingEnabled() const {
    return impl->rasterizer_marking_enabled;
}

void MemorySystem::RasterizerMarkRegionCached(PAddr start, u32 size, bool cached) {
    if (start == 0 || !impl->rasterizer_marking_enabled) {
        return;
    }

    u32 num_pages = ((start + size - 1) >> CITRA_PAGE_BITS) - (start >> CITRA_PAGE_BITS) + 1;
    PAddr paddr = start;

    for (unsigned i = 0; i < num_pages; ++i, paddr += CITRA_PAGE_SIZE) {
        for (VAddr vaddr : PhysicalToVirtualAddressForRasterizer(paddr)) {
            impl->cache_marker.Mark(vaddr, cached);
            for (auto& page_table : impl->page_table_list) {
                PageType& page_type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];

                if (cached) {
                    // Switch page type to cached if now cached
                    switch (page_type) {
                    case PageType::Unmapped:
                        // It is not necessary for a process to have this region mapped into its
                        // address space, for example, a system module need not have a VRAM mapping.
                        break;
                    case PageType::Memory:
                    case PageType::MemoryWatchpoint:
                        page_type = (page_type == PageType::Memory)
                                        ? PageType::RasterizerCachedMemory
                                        : PageType::RasterizerCachedMemoryWatchpoint;
                        page_table->pointers[vaddr >> CITRA_PAGE_BITS] = nullptr;
                        break;
                    default:
                        UNREACHABLE();
                    }
                } else {
                    // Switch page type to uncached if now uncached
                    switch (page_type) {
                    case PageType::Unmapped:
                        // It is not necessary for a process to have this region mapped into its
                        // address space, for example, a system module need not have a VRAM mapping.
                        break;
                    case PageType::RasterizerCachedMemory:
                    case PageType::RasterizerCachedMemoryWatchpoint: {
                        page_type = (page_type == PageType::RasterizerCachedMemory)
                                        ? PageType::Memory
                                        : PageType::MemoryWatchpoint;

                        if (page_type == PageType::Memory) {
                            page_table->pointers[vaddr >> CITRA_PAGE_BITS] =
                                GetPointerForRasterizerCache(vaddr & ~CITRA_PAGE_MASK);
                        }
                        break;
                    }
                    default:
                        UNREACHABLE();
                    }
                }
            }
        }
    }
}

u8 MemorySystem::Read8(const VAddr addr) {
    return Read<u8>(impl->current_page_table, addr);
}

u8 MemorySystem::Read8(const Kernel::Process& process, VAddr addr) {
    return Read<u8>(process.vm_manager.page_table, addr);
}

u16 MemorySystem::Read16(const VAddr addr) {
    return Read<u16_le>(impl->current_page_table, addr);
}

u16 MemorySystem::Read16(const Kernel::Process& process, VAddr addr) {
    return Read<u16_le>(process.vm_manager.page_table, addr);
}

u32 MemorySystem::Read32(const VAddr addr) {
    return Read<u32_le>(impl->current_page_table, addr);
}

u32 MemorySystem::Read32(const Kernel::Process& process, VAddr addr) {
    return Read<u32_le>(process.vm_manager.page_table, addr);
}

u64 MemorySystem::Read64(const VAddr addr) {
    return Read<u64_le>(impl->current_page_table, addr);
}

u64 MemorySystem::Read64(const Kernel::Process& process, VAddr addr) {
    return Read<u64_le>(process.vm_manager.page_table, addr);
}

std::optional<u32> MemorySystem::Read32OrNullopt(VAddr addr) {
    return Read<std::optional<u32_le>>(impl->current_page_table, addr);
}

std::optional<u32> MemorySystem::Read32OrNullopt(const Kernel::Process& process, VAddr addr) {
    return Read<std::optional<u32_le>>(process.vm_manager.page_table, addr);
}

void MemorySystem::ReadBlock(const Kernel::Process& process, const VAddr src_addr,
                             void* dest_buffer, const std::size_t size) {
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::ReadBlock(VAddr src_addr, void* dest_buffer, std::size_t size) {
    const auto& process = *impl->system.Kernel().GetCurrentProcess();
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::Write8(const VAddr addr, const u8 data) {
    Write<u8>(impl->current_page_table, addr, data);
}

void MemorySystem::Write8(const Kernel::Process& process, const VAddr addr, const u8 data) {
    Write<u8>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write16(const VAddr addr, const u16 data) {
    Write<u16_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write16(const Kernel::Process& process, const VAddr addr, const u16 data) {
    Write<u16_le>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write32(const VAddr addr, const u32 data) {
    Write<u32_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write32(const Kernel::Process& process, const VAddr addr, const u32 data) {
    Write<u32_le>(process.vm_manager.page_table, addr, data);
}

void MemorySystem::Write64(const VAddr addr, const u64 data) {
    Write<u64_le>(impl->current_page_table, addr, data);
}

void MemorySystem::Write64(const Kernel::Process& process, const VAddr addr, const u64 data) {
    Write<u64_le>(process.vm_manager.page_table, addr, data);
}

bool MemorySystem::WriteExclusive8(const VAddr addr, const u8 data, const u8 expected) {
    return WriteExclusive<u8>(addr, data, expected);
}

bool MemorySystem::WriteExclusive16(const VAddr addr, const u16 data, const u16 expected) {
    return WriteExclusive<u16_le>(addr, data, expected);
}

bool MemorySystem::WriteExclusive32(const VAddr addr, const u32 data, const u32 expected) {
    return WriteExclusive<u32_le>(addr, data, expected);
}

bool MemorySystem::WriteExclusive64(const VAddr addr, const u64 data, const u64 expected) {
    return WriteExclusive<u64_le>(addr, data, expected);
}

void MemorySystem::WriteBlock(const Kernel::Process& process, const VAddr dest_addr,
                              const void* src_buffer, const std::size_t size) {
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::WriteBlock(const VAddr dest_addr, const void* src_buffer,
                              const std::size_t size) {
    auto& process = *impl->system.Kernel().GetCurrentProcess();
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::ZeroBlock(const Kernel::Process& process, const VAddr dest_addr,
                             const std::size_t size) {
    auto& page_table = *process.vm_manager.page_table;
    std::size_t remaining_size = size;
    std::size_t page_index = dest_addr >> CITRA_PAGE_BITS;
    std::size_t page_offset = dest_addr & CITRA_PAGE_MASK;

    while (remaining_size > 0) {
        const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
        const VAddr current_vaddr =
            static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

        switch (page_table.attributes[page_index]) {
        case PageType::Unmapped: {
            LOG_ERROR(HW_Memory,
                      "unmapped ZeroBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                      "0x{:08X}",
                      current_vaddr, dest_addr, size, impl->GetPC());
            break;
        }
        case PageType::Memory: {
            DEBUG_ASSERT(page_table.pointers[page_index]);

            u8* dest_ptr = page_table.pointers[page_index] + page_offset;
            std::memset(dest_ptr, 0, copy_amount);
            break;
        }
        case PageType::MemoryWatchpoint: {
            auto it = page_table.watchpoint_pages_map.find(page_index);
            ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                       "Missing memory for watchpoint page");

            u8* dest_ptr = it->second.memory.GetPtr() + page_offset;
            std::memset(dest_ptr, 0, copy_amount);
            break;
        }
        case PageType::RasterizerCachedMemory:
        case PageType::RasterizerCachedMemoryWatchpoint: {
            RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                         FlushMode::Invalidate);
            std::memset(GetPointerForRasterizerCache(current_vaddr), 0, copy_amount);
            break;
        }
        default:
            UNREACHABLE();
        }

        page_index++;
        page_offset = 0;
        remaining_size -= copy_amount;
    }
}

void MemorySystem::CopyBlock(const Kernel::Process& process, VAddr dest_addr, VAddr src_addr,
                             const std::size_t size) {
    CopyBlock(process, process, dest_addr, src_addr, size);
}

void MemorySystem::CopyBlock(const Kernel::Process& dest_process,
                             const Kernel::Process& src_process, VAddr dest_addr, VAddr src_addr,
                             std::size_t size) {
    auto& page_table = *src_process.vm_manager.page_table;
    std::size_t remaining_size = size;
    std::size_t page_index = src_addr >> CITRA_PAGE_BITS;
    std::size_t page_offset = src_addr & CITRA_PAGE_MASK;

    while (remaining_size > 0) {
        const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
        const VAddr current_vaddr =
            static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

        switch (page_table.attributes[page_index]) {
        case PageType::Unmapped: {
            LOG_ERROR(HW_Memory,
                      "unmapped CopyBlock @ 0x{:08X} (start address = 0x{:08X}, size = {}) at PC "
                      "0x{:08X}",
                      current_vaddr, src_addr, size, impl->GetPC());
            ZeroBlock(dest_process, dest_addr, copy_amount);
            break;
        }
        case PageType::Memory: {
            DEBUG_ASSERT(page_table.pointers[page_index]);
            const u8* src_ptr = page_table.pointers[page_index] + page_offset;
            WriteBlock(dest_process, dest_addr, src_ptr, copy_amount);
            break;
        }
        case PageType::MemoryWatchpoint: {
            auto it = page_table.watchpoint_pages_map.find(page_index);
            ASSERT_MSG(it != page_table.watchpoint_pages_map.end(),
                       "Missing memory for watchpoint page");

            const u8* src_ptr = it->second.memory.GetPtr() + page_offset;
            WriteBlock(dest_process, dest_addr, src_ptr, copy_amount);
            break;
        }
        case PageType::RasterizerCachedMemory:
        case PageType::RasterizerCachedMemoryWatchpoint: {
            RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                         FlushMode::Flush);
            WriteBlock(dest_process, dest_addr, GetPointerForRasterizerCache(current_vaddr),
                       copy_amount);
            break;
        }
        default:
            UNREACHABLE();
        }

        page_index++;
        page_offset = 0;
        dest_addr += static_cast<VAddr>(copy_amount);
        src_addr += static_cast<VAddr>(copy_amount);
        remaining_size -= copy_amount;
    }
}

u32 MemorySystem::GetFCRAMOffset(const u8* pointer) const {
    ASSERT(pointer >= impl->fcram.Data() &&
           pointer <= impl->fcram.Data() + impl->fcram.Size());
    return static_cast<u32>(pointer - impl->fcram.Data());
}

u8* MemorySystem::GetFCRAMPointer(std::size_t offset) {
    ASSERT(offset <= impl->fcram.Size());
    return impl->fcram.Data() + offset;
}

const u8* MemorySystem::GetFCRAMPointer(std::size_t offset) const {
    ASSERT(offset <= impl->fcram.Size());
    return impl->fcram.Data() + offset;
}

MemoryRef MemorySystem::GetFCRAMRef(std::size_t offset) const {
    ASSERT(offset <= impl->fcram.Size());
    return MemoryRef(impl->fcram_mem, offset);
}

u8* MemorySystem::GetDspMemory(std::size_t offset) const {
    ASSERT(offset <= Memory::DSP_RAM_SIZE);
    return impl->dsp_ram.Data() + offset;
}

} // namespace Memory
