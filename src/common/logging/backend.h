// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string_view>
#include "common/logging/filter.h"
#ifdef HAVE_LIBRETRO
#include "libretro.h"
#endif

namespace Common::Log {

class Filter;

/// Initializes the logging system. This should be the first thing called in main.
void Initialize(std::string_view log_file = "");
#ifdef HAVE_LIBRETRO
void LibRetroStart(retro_log_printf_t callback);
#endif

/// Flushes and closes the backends. Messages are written as they are logged, so this is only
/// needed to make sure what has been logged has reached the disk before the process ends.
void Stop();

void DisableLoggingInTests();

/**
 * The global filter will prevent any messages from even being processed if they are filtered.
 */
void SetGlobalFilter(const Filter& filter);

/**
 * Only allow messages that match the specified regex. The regex is matched against the final log
 * text.
 */
bool SetRegexFilter(const std::string& regex);

void SetColorConsoleBackendEnabled(bool enabled);
} // namespace Common::Log
