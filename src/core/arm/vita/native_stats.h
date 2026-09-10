// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include "common/common_types.h"

namespace Core::NativeStats {

/// Written by the native executor, read and reset once a second by the frontend's
/// perf line. Rates tell the slow-speed story apart: many slices with tiny credited
/// time means round-trip bound, few slices means the emu thread is busy elsewhere.
inline std::atomic<u32> slices{0};
/// Guest stores into pages the surface cache armed (core/guest_write_tracker.h).
inline std::atomic<u32> write_faults{0};
inline std::atomic<u32> svcs{0};
inline std::atomic<u64> credited_ns{0};

// Where the emulation thread's wall time goes, microseconds per stats window: inside the
// azaharRun syscall (spin/sleep included), inside System::RunLoop as a whole, and inside the
// present path. RunLoop minus azaharRun is HLE and scheduling; the second minus everything is
// time blocked outside all three.
inline std::atomic<u64> vanrun_us{0};
inline std::atomic<u64> runloop_us{0};
inline std::atomic<u64> present_us{0};

// Wall time inside CallSVC: the HLE half of a slice. The rest of the pipeline's waiting is
// measured platform-independently in common/pipeline_stats.h.
inline std::atomic<u64> svc_us{0};


} // namespace Core::NativeStats
