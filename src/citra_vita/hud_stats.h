// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include "common/common_types.h"

/**
 * The numbers the on-screen stats window shows, published once a second by the emulation
 * thread from the same figures the [azahar] console line prints, read by the render thread
 * when it draws the interface. Plain atomics, so neither side waits on the other.
 */
namespace VitaFrontend::HudStats {

inline std::atomic<u32> speed_percent{0};   ///< emulated time over wall time
inline std::atomic<u32> game_fps10{0};      ///< the guest's frame rate, tenths
inline std::atomic<u32> shown_fps{0};       ///< frames presented in the last second
inline std::atomic<u32> skipped_frames{0};  ///< frames whose draws frameskip dropped
inline std::atomic<u32> guest_percent{0};   ///< core 2 running guest code natively
inline std::atomic<u32> emu_percent{0};     ///< the emulation thread inside RunLoop
inline std::atomic<u32> render_percent{0};  ///< the render thread executing queued ops
inline std::atomic<u32> queue_depth{0};     ///< ops queued to the render thread
inline std::atomic<bool> valid{false};

} // namespace VitaFrontend::HudStats
