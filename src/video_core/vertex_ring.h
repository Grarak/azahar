// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include "common/common_types.h"

namespace VideoCore {

/**
 * A ring of GPU-visible vertex memory shared between the emulation and render threads: the
 * persistently mapped vertex stream buffer, produced into directly by the emulation thread and
 * consumed by glDrawArrays on the render thread.
 *
 * Positions are monotonic byte counters; `pos % size` is the offset inside the mapping. The
 * emulation thread owns `cursor` exclusively. `retired` is advanced by the render thread once a
 * fence proves the GL has finished reading up to that position, and the producer may write only
 * inside [retired, retired + size). Allocations never straddle the wrap point: the tail is padded
 * and both counters stay multiples of the vertex stride, so a byte offset divided by the stride
 * is always a whole first-vertex index.
 */
struct VertexRing {
    u8* base = nullptr;
    u64 size = 0;

    /// Producer position, emulation thread only. `offset` mirrors `cursor % size` so the hot
    /// path never divides — u64 divmod is a libcall on 32-bit ARM and measured whole percents.
    u64 cursor = 0;
    u32 offset = 0;

    /// Everything below this position has been read by the GL; render thread advances it.
    std::atomic<u64> retired{0};
    /// Producer's cached view of `retired`, refreshed (with its acquire barrier) only when the
    /// cached value no longer proves space, instead of once per triangle.
    u64 cached_retired = 0;

    std::mutex mutex;
    std::condition_variable retired_cv;

    /// Start of the oldest allocation the emulation thread made for a hardware draw that the
    /// render thread has not consumed yet (~0 when none). Two threads allocate from this ring
    /// - the emulation thread for shipped hardware draws, the render thread for the mirror's
    /// software triangles - so a position is not proof that everything below it has been
    /// drawn: a scene can retire past a hardware draw's bytes that are still queued behind
    /// it. Retire clamps to this floor. Maintained by the shipping facade under `mutex`.
    std::atomic<u64> consume_floor{~0ull};

    bool Active() const {
        return base != nullptr;
    }

    static constexpr u32 FULL = 0xFFFFFFFF;

    /// Fast-path reservation: returns the byte offset of `bytes` contiguous bytes and advances
    /// the cursor, or FULL when the ring has no proven space. Emulation thread only.
    u32 TryAlloc(u32 bytes) {
        u32 off = offset;
        u64 pos = cursor;
        if (off + bytes > size) {
            // Pad out the tail; allocations never straddle the wrap.
            pos += size - off;
            off = 0;
        }
        if (pos + bytes - cached_retired > size) {
            cached_retired = retired.load(std::memory_order_acquire);
            if (pos + bytes - cached_retired > size) {
                return FULL;
            }
        }
        cursor = pos + bytes;
        offset = off + bytes;
        return off;
    }

    /// Space probe for the wait loop: like TryAlloc but without reserving.
    bool TryAllocWouldFail(u32 bytes) {
        u64 pos = cursor;
        if (offset + bytes > size) {
            pos += size - offset;
        }
        cached_retired = retired.load(std::memory_order_acquire);
        return pos + bytes - cached_retired > size;
    }

    /// Rewind to a previously recorded (cursor, offset) pair; nothing may have shipped since.
    void Rewind(u64 saved_cursor, u32 saved_offset) {
        cursor = saved_cursor;
        offset = saved_offset;
    }

    /// Render thread: everything up to `pos` is done with, except bytes a queued hardware
    /// draw still owns (consume_floor).
    void Retire(u64 pos) {
        pos = std::min(pos, consume_floor.load(std::memory_order_acquire));
        u64 current = retired.load(std::memory_order_relaxed);
        while (pos > current &&
               !retired.compare_exchange_weak(current, pos, std::memory_order_release)) {
        }
        std::scoped_lock lock{mutex};
        retired_cv.notify_all();
    }
};

/// A contiguous run of ring vertices forming whole triangles.
struct VertexRingRange {
    u32 first;
    u32 count;
    u64 end_pos;
};

} // namespace VideoCore
