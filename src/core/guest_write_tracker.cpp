// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/logging/log.h"
#include "core/guest_write_tracker.h"

namespace Memory {

void GuestWriteTracker::RegisterBlock(const u8* host_base, std::size_t size, u32 physical_base) {
    if (host_base == nullptr || size == 0) {
        return;
    }
    const std::size_t pages = (size + PageSize - 1) / PageSize;
    Block block{host_base, size, physical_base, state.size()};
    // std::atomic is not movable, so the table is grown by reconstruction; it only grows at
    // construction of the memory system, before anything is tracked.
    std::vector<std::atomic<u32>> grown(state.size() + pages);
    for (std::size_t i = 0; i < state.size(); i++) {
        grown[i].store(state[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    state.swap(grown);
    std::vector<std::atomic<u32>> words((state.size() + 31) / 32);
    for (std::size_t i = 0; i < dirty_words.size(); i++) {
        words[i].store(dirty_words[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    dirty_words.swap(words);
    streak.resize(state.size(), 0);
    blocks.push_back(block);
}

void GuestWriteTracker::SetProtect(ProtectFn fn) {
    protect = std::move(fn);
}

void GuestWriteTracker::SetEnabled(bool on) {
    enabled.store(on && protect != nullptr, std::memory_order_relaxed);
}

const GuestWriteTracker::Block* GuestWriteTracker::FindBlock(const u8* host) const {
    for (const Block& block : blocks) {
        if (host >= block.host_base && host < block.host_base + block.size) {
            return &block;
        }
    }
    return nullptr;
}

void GuestWriteTracker::SetDeferred(bool on) {
    deferred.store(on, std::memory_order_relaxed);
}

void GuestWriteTracker::MarkDirty(std::size_t index) {
    dirty_words[index / 32].fetch_or(1u << (index % 32), std::memory_order_release);
    any_dirty.store(true, std::memory_order_release);
}

void GuestWriteTracker::SetHotPages(bool on) {
    hot_pages.store(on, std::memory_order_relaxed);
}

void GuestWriteTracker::Arm(const u8* host, std::size_t size) {
    if (!Enabled() || size == 0) {
        return;
    }
    if (deferred.load(std::memory_order_relaxed)) {
        std::scoped_lock lock{pending_mutex};
        pending.push_back({host, size});
        return;
    }
    ArmNow(host, size);
}

void GuestWriteTracker::ApplyPending() {
    std::vector<Pending> batch;
    {
        std::scoped_lock lock{pending_mutex};
        batch.swap(pending);
    }
    for (const Pending& p : batch) {
        ArmNow(p.host, p.size);
    }
}

void GuestWriteTracker::ArmNow(const u8* host, std::size_t size) {
    const Block* block = FindBlock(host);
    if (block == nullptr) {
        return;
    }
    const u8* end = std::min(host + size, block->host_base + block->size);
    const u8* first = reinterpret_cast<const u8*>(reinterpret_cast<std::uintptr_t>(host) &
                                                   ~std::uintptr_t{PageSize - 1});
    // Consecutive pages are protected in one call. The platform's primitive is not always
    // cheap: on the Vita it is a syscall that edits the taken core's descriptors and drops its
    // TLB, so a 512 KiB surface armed page by page was a hundred and twenty-eight of them.
    // Protect first and claim after, for the whole run at once: a store landing between the
    // two faults, bumps the page's epoch and leaves it writable and dirty, and that page's
    // claim then fails while its neighbours' succeed.
    // Both ends stay on page boundaries. The range handed in does not: it is a surface's byte
    // extent, so its last page is partial, and passing that length to the platform's primitive
    // asked the Vita's kernel to protect four and a bit pages (0x1180 bytes) and was refused
    // outright (2026-09-09).
    const u8* run_start = nullptr;
    const u8* run_end = nullptr;
    const auto flush_run = [&]() {
        if (run_start == nullptr) {
            return;
        }
        protect(run_start, static_cast<std::size_t>(run_end - run_start), true);
        for (const u8* page = run_start; page < run_end; page += PageSize) {
            const std::size_t index = block->first_page + (page - block->host_base) / PageSize;
            std::atomic<u32>& word = state[index];
            u32 seen = word.load(std::memory_order_acquire);
            if ((seen & ArmedBit) == 0 &&
                word.compare_exchange_strong(seen, seen | ArmedBit, std::memory_order_acq_rel)) {
                stat_arms.fetch_add(1, std::memory_order_relaxed);
            }
        }
        run_start = nullptr;
        run_end = nullptr;
    };
    for (const u8* page = first; page < end; page += PageSize) {
        const std::size_t index = block->first_page + (page - block->host_base) / PageSize;
        std::atomic<u32>& word = state[index];
        const u32 seen = word.load(std::memory_order_acquire);
        if (seen & ArmedBit) {
            flush_run();
            continue;
        }
        if ((seen & HotBit) && hot_pages.load(std::memory_order_relaxed)) {
            // Written every frame: answer dirty now, no protection, no fault.
            flush_run();
            if (!(seen & DirtyBit)) {
                word.fetch_or(DirtyBit, std::memory_order_acq_rel);
                MarkDirty(index);
            }
            stat_hot_skips.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (run_start == nullptr) {
            run_start = page;
        }
        run_end = page + PageSize;
    }
    flush_run();
}

bool GuestWriteTracker::OnFault(const u8* host_page, void (*unprotect)(void*),
                                void* unprotect_ctx) {
    const Block* block = FindBlock(host_page);
    if (block == nullptr) {
        return false;
    }
    const std::size_t index = block->first_page + (host_page - block->host_base) / PageSize;
    std::atomic<u32>& word = state[index];
    u32 seen = word.load(std::memory_order_acquire);
    if (!(seen & ArmedBit) && (seen & DirtyBit)) {
        // A second alias of a page already faulted through another guest address: make this
        // alias writable too, nothing else to record.
        unprotect(unprotect_ctx);
        return true;
    }
    // Disarm and dirty under a new epoch, then make the page writable so the store retries.
    while (!word.compare_exchange_weak(seen, ((seen | DirtyBit) & ~ArmedBit) + EpochOne,
                                       std::memory_order_acq_rel)) {
    }
    unprotect(unprotect_ctx);
    MarkDirty(index);
    stat_faults.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void GuestWriteTracker::Consume(const std::function<void(u32, u32)>& fn) {
    if (!any_dirty.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    consume_tick++;
    const bool probe = consume_tick % HotProbeTicks == 0;
    const bool hot_on = hot_pages.load(std::memory_order_relaxed);
    // The pages dirty since the last consume, from the bitmap. Each has its dirty bit
    // taken and its streak advanced; the list is sorted for the run merge below.
    std::vector<u32> dirty_now;
    for (std::size_t w = 0; w < dirty_words.size(); w++) {
        u32 bits = dirty_words[w].exchange(0, std::memory_order_acq_rel);
        while (bits != 0) {
            const u32 index = static_cast<u32>(w * 32 + static_cast<u32>(__builtin_ctz(bits)));
            bits &= bits - 1;
            std::atomic<u32>& word = state[index];
            u32 seen = word.load(std::memory_order_acquire);
            while ((seen & DirtyBit) &&
                   !word.compare_exchange_weak(seen, seen & ~DirtyBit, std::memory_order_acq_rel)) {
            }
            if (!(seen & DirtyBit)) {
                continue;
            }
            dirty_now.push_back(index);
            u8& run = streak[index];
            if (run == 0) {
                aging.push_back(index);
            }
            if (run < 255) {
                run++;
            }
            if (probe && (seen & HotBit)) {
                // The periodic real arm: the next Arm protects. One fault brings the page
                // straight back to hot.
                word.fetch_and(~HotBit, std::memory_order_acq_rel);
                run = HotAfter - 1;
            } else if (hot_on && run >= HotAfter && !(seen & HotBit)) {
                word.fetch_or(HotBit, std::memory_order_acq_rel);
            }
        }
    }
    std::sort(dirty_now.begin(), dirty_now.end());
    // Runs of consecutive pages, never across a block boundary.
    for (std::size_t k = 0; k < dirty_now.size();) {
        const u32 first = dirty_now[k];
        const Block* block = nullptr;
        for (const Block& b : blocks) {
            const std::size_t pages = (b.size + PageSize - 1) / PageSize;
            if (first >= b.first_page && first < b.first_page + pages) {
                block = &b;
                break;
            }
        }
        const std::size_t block_end =
            block != nullptr ? block->first_page + (block->size + PageSize - 1) / PageSize : first + 1;
        std::size_t n = 1;
        while (k + n < dirty_now.size() && dirty_now[k + n] == first + n && first + n < block_end) {
            n++;
        }
        if (block != nullptr) {
            stat_consumed.fetch_add(n, std::memory_order_relaxed);
            fn(block->physical_base + static_cast<u32>(first - block->first_page) * PageSize,
               static_cast<u32>(n * PageSize));
        }
        k += n;
    }
    // A page with a streak that was not dirty this time cools; the list keeps the warm ones.
    std::size_t kept = 0;
    for (const u32 index : aging) {
        if (std::binary_search(dirty_now.begin(), dirty_now.end(), index)) {
            aging[kept++] = index;
            continue;
        }
        streak[index] = 0;
        std::atomic<u32>& word = state[index];
        if (word.load(std::memory_order_acquire) & HotBit) {
            word.fetch_and(~HotBit, std::memory_order_acq_rel);
        }
    }
    aging.resize(kept);
}

GuestWriteTracker::Stats GuestWriteTracker::TakeStats() {
    return Stats{stat_faults.exchange(0, std::memory_order_relaxed),
                 stat_arms.exchange(0, std::memory_order_relaxed),
                 stat_consumed.exchange(0, std::memory_order_relaxed),
                 stat_hot_skips.exchange(0, std::memory_order_relaxed)};
}

} // namespace Memory
