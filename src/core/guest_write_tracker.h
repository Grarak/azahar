// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>
#include "common/common_types.h"

namespace Memory {

/**
 * Which pages of guest memory the guest's CPU has written since the surface cache last looked.
 *
 * The guest runs natively, so nothing sees its stores. What can be seen is a store into a page
 * that is not writable: the page backing a cached surface is made read-only in the guest's own
 * view of memory (a second mapping of the same bytes, so the emulator's view stays writable),
 * the first store faults, the fault handler marks the page dirty, makes it writable again and
 * lets the store retry. One fault per page per arming; every further store is free. The
 * emulation thread later consumes the dirty pages and invalidates what they back; the cache
 * arms a page again when it next validates a surface from it.
 *
 * Pages are keyed by the emulator's host address of the bytes (the physical layout: one block
 * per 3DS memory region, registered with its physical base). The platform supplies Protect,
 * which flips a host page's protection in the guest view (mprotect on Linux over the native
 * mirror; the taken core's translation table on the Vita), and calls OnFault from its fault
 * handler with the host page the guest hit.
 *
 * The state word of a page is [epoch:29][hot:1][armed:1][dirty:1]. Arming reads the word,
 * protects, then sets armed with a compare-exchange: a fault in between bumps the epoch, the
 * exchange fails and the page stays writable and dirty, which the next consume sees. The fault
 * handler only ever bumps the epoch and clears armed; it never blocks.
 *
 * A page the guest writes every time it is armed (a CPU-composed framebuffer, a texture the
 * game rewrites each frame) would cost a protect, a fault and an unprotect per frame for an
 * answer that is always "dirty". Consume counts consecutive dirty consumes per page; past
 * HotAfter the page is hot, and arming it marks it dirty at once instead of protecting it,
 * so the cache reloads it every frame as before but no fault is taken. Every HotProbeTicks
 * consumes a hot page is armed for real once: a page that has stopped being written comes
 * back clean and cools, one that is still written faults once and is hot again.
 */
class GuestWriteTracker {
public:
    static constexpr u32 PageBits = 12;
    static constexpr u32 PageSize = 1u << PageBits;

    /// Flips the protection of the guest view of the host pages [host, host + size).
    using ProtectFn = std::function<void(const u8* host, std::size_t size, bool read_only)>;

    /// A block of guest memory, registered once at construction of the memory system.
    void RegisterBlock(const u8* host_base, std::size_t size, u32 physical_base);
    /// Installs the platform's protection primitive. Tracking is off until this is called.
    void SetProtect(ProtectFn fn);
    void SetEnabled(bool on);
    [[nodiscard]] bool Enabled() const {
        return enabled.load(std::memory_order_relaxed);
    }

    /// Makes the pages of [host, host + size) fault on the guest's next write. Any thread.
    /// In deferred mode the request is queued and ApplyPending performs it.
    void Arm(const u8* host, std::size_t size);

    /**
     * Deferred mode: Arm only queues, and ApplyPending, called by the thread that runs the
     * guest between two slices, protects and claims the pages then. Where the protection
     * primitive cannot take effect while the guest runs (the Vita: the taken core drops its
     * TLB only at the next azaharRun) this closes the window in which a claimed page is still
     * writable.
     */
    void SetDeferred(bool on);
    void ApplyPending();

    /// Whether pages written every frame skip the protect-fault-unprotect cycle (on by default).
    void SetHotPages(bool on);

    /**
     * From the fault handler: the guest stored into `host_page`'s guest alias. Returns false
     * if the page is not tracked memory at all (a real guest fault). Otherwise marks the page
     * dirty, disarms it, makes that alias writable through `unprotect` and returns true.
     * Async-signal-safe: no locks, no allocation.
     */
    bool OnFault(const u8* host_page, void (*unprotect)(void*), void* unprotect_ctx);

    /// Hands every dirty run to `fn(physical_address, size)` and clears the dirty bits. Runs
    /// of consecutive pages are merged. Emulation thread.
    void Consume(const std::function<void(u32 paddr, u32 size)>& fn);

    struct Stats {
        u64 faults = 0;
        u64 arms = 0;
        u64 consumed_pages = 0;
        u64 hot_skips = 0; ///< arms answered "dirty" without protecting, the page being hot
    };
    Stats TakeStats();

private:
    struct Block {
        const u8* host_base;
        std::size_t size;
        u32 physical_base;
        std::size_t first_page; ///< index of its first page in `state`
    };
    static constexpr u32 DirtyBit = 1u;
    static constexpr u32 ArmedBit = 2u;
    static constexpr u32 HotBit = 4u;
    static constexpr u32 EpochOne = 8u;
    static constexpr u8 HotAfter = 4;        ///< consecutive dirty consumes that make a page hot
    static constexpr u32 HotProbeTicks = 64; ///< consumes between real arms of a hot page

    [[nodiscard]] const Block* FindBlock(const u8* host) const;

    void ArmNow(const u8* host, std::size_t size);
    void MarkDirty(std::size_t index);

    struct Pending {
        const u8* host;
        std::size_t size;
    };
    std::vector<Block> blocks;
    std::vector<std::atomic<u32>> state;
    /// One bit per page, set together with the page's dirty bit: what Consume walks instead
    /// of every page (a title flushing at every frame made the full walk half the emulation
    /// thread on the console).
    std::vector<std::atomic<u32>> dirty_words;
    std::vector<u8> streak; ///< consecutive dirty consumes per page; the consuming thread's only
    std::vector<u32> aging; ///< pages with a non-zero streak, the consuming thread's only
    u32 consume_tick = 0;
    std::atomic<bool> hot_pages{true};
    std::atomic<bool> deferred{false};
    std::mutex pending_mutex;
    std::vector<Pending> pending;
    std::atomic<bool> any_dirty{false};
    std::atomic<bool> enabled{false};
    ProtectFn protect;
    std::atomic<u64> stat_faults{0};
    std::atomic<u64> stat_arms{0};
    std::atomic<u64> stat_consumed{0};
    std::atomic<u64> stat_hot_skips{0};
};

} // namespace Memory
