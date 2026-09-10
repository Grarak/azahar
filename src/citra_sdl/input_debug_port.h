// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/common_types.h"

class EmuWindow_SDL2;

/**
 * A line-based TCP port for driving the emulator's input from outside the process.
 *
 * This port exists because the machine this frontend targets is usually driven over ssh, where
 * there is no keyboard to press: a title that stops at a confirmation dialog stops for good, and
 * "does it boot" cannot be answered past the first prompt. Anything that can open a socket can
 * drive it, so a boot test is a shell script:
 *
 *     printf 'tap a\nscreenshot /tmp/menu.ppm\n' | nc -q1 127.0.0.1 5000
 *
 * It binds to the loopback interface only and has no authentication, because it can press buttons
 * in a game and nothing more. Off by default.
 */
class InputDebugPort {
public:
    InputDebugPort();
    ~InputDebugPort();

    InputDebugPort(const InputDebugPort&) = delete;
    InputDebugPort& operator=(const InputDebugPort&) = delete;

    /// Starts listening on 127.0.0.1:port. Returns false if the port could not be opened.
    bool Open(u16 port);

    bool IsOpen() const {
        return listen_fd >= 0;
    }

    /**
     * Accepts connections and runs whatever commands have arrived. Never blocks.
     * @param window The window commands act on.
     * @returns false if a client asked the emulator to quit.
     */
    bool Poll(EmuWindow_SDL2& window);

    /// Sets the text `status` reports.
    void SetStatus(std::string text) {
        status = std::move(text);
    }

private:
    /// Runs one command line and returns the reply to send back.
    std::string Execute(const std::string& line, EmuWindow_SDL2& window);

    /// Saves or loads a savestate slot. Handled by the emulator at its next safe point.
    std::string SaveOrLoadState(bool save, int slot);

    /// Releases any buttons whose `tap` has run its course.
    void ExpireTaps();

    int listen_fd{-1};
    int client_fd{-1};
    std::string pending_input;
    std::string status{"idle"};
    bool quit_requested{};

    /// Scancodes held by a `tap`, and when each should be released.
    std::unordered_map<int, std::chrono::steady_clock::time_point> tapped_keys;
};

/// Installs the keyboard bindings this frontend uses. Without these no input reaches the guest.
void SetDefaultInputBindings();
