// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_base.h"

namespace Core {
class System;
}

namespace VitaFrontend {

class Ui;

/**
 * The Vita's window, such as it is: a fixed 960x544 display with no window system above it.
 *
 * Presentation is zero-copy: the GPU samples the guest's own framebuffers straight out of
 * emulator memory (GxmPresent maps guest RAM for the GPU; the renderer says where this
 * frame's framebuffers are), which is what the real console's display controller does too.
 * No pixel is copied on the CPU between the rasterizer's writes and the screen.
 *
 * Every GXM call happens on the render thread, which owns the presentation layer exactly as
 * the Linux tier's render thread owns its target. A finished frame is presented from
 * SwapBuffers, where it arrives, so the emulation thread never waits for a present.
 *
 * The interface is drawn there too: ImGui's frame is built and rendered inside that same call,
 * so the whole of it lives on one thread, and the menu's decision is published back for the
 * emulation thread to read. While a menu is up emulation is paused and no frames arrive, so
 * that path asks the render thread for a present of its own; before a title is running there is
 * no render thread at all and the caller draws directly, being then the only thread there is.
 */
class EmuWindowVita final : public Frontend::EmuWindow {
public:
    explicit EmuWindowVita(Core::System& system);
    ~EmuWindowVita() override;

    /// Reads the pad and the touchscreen. Called on the emulation thread once per iteration.
    void PollEvents() override;

    /// Called on the render thread with a finished frame: presents it.
    void SwapBuffers() override;

    /// Draws the guest's framebuffers, plus whatever the UI wants over them, and presents.
    /// Runs wherever it is called from, so callers must be the one thread presenting: the
    /// render thread once a title is running, the calling thread before that. `ui` may be
    /// null, in which case only the emulated screens are drawn.
    void Present(Ui* ui);

    /// Hands the interface to the render thread, which builds and draws it inside SwapBuffers
    /// from then on. Null detaches it again.
    void SetUi(Ui* ui) {
        render_ui.store(ui, std::memory_order_release);
    }

    /// The last decision the interface reached on the render thread, taken by the emulation
    /// thread and cleared.
    [[nodiscard]] int TakeUiAction();

    /// True once the user has asked to leave.
    [[nodiscard]] bool IsQuitRequested() const {
        return quit_requested;
    }

    void RequestQuit() {
        quit_requested = true;
    }

    /// Pad state as of the last PollEvents, as a mask of SCE_CTRL_* bits.
    [[nodiscard]] u32 ButtonsHeld() const {
        return buttons_held.load(std::memory_order_relaxed);
    }

    /// Left stick, each axis in [-1, 1] with y positive upwards.
    [[nodiscard]] std::pair<float, float> LeftStick() const;

    /// Stops feeding the guest its pad state, for while a menu owns the input.
    void SetInputToGuest(bool enabled) {
        input_to_guest.store(enabled, std::memory_order_relaxed);
    }

    void UpdateCurrentFramebufferLayout();

private:
    void DrawScreens();
    void UpdateTouch();

    Core::System& system;

    /// The interface, when the render thread is the one driving it.
    std::atomic<Ui*> render_ui{nullptr};
    std::atomic<int> pending_ui_action{0};

    std::atomic<u32> buttons_held{};
    std::atomic<s32> stick_x{}, stick_y{};
    std::atomic<bool> input_to_guest{true};
    std::atomic<bool> quit_requested{};
    bool touch_down{};
};

} // namespace VitaFrontend
