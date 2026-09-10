// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>

#include <psp2/ctrl.h>

#include "citra_vita/emu_window_vita.h"
#include "citra_vita/input_factory_vita.h"
#include "common/param_package.h"
#include "common/settings.h"
#include "core/frontend/input.h"

namespace VitaFrontend {

namespace {

constexpr char EngineName[] = "vita";

EmuWindowVita* g_window = nullptr;

/// A guest button, answered from the frontend's most recent pad sample.
class VitaButton final : public Input::ButtonDevice {
public:
    explicit VitaButton(u32 mask_) : mask{mask_} {}

    bool GetStatus() const override {
        if (g_window == nullptr || mask == 0) {
            return false;
        }
        return (g_window->ButtonsHeld() & mask) != 0;
    }

private:
    u32 mask;
};

class VitaButtonFactory final : public Input::Factory<Input::ButtonDevice> {
public:
    std::unique_ptr<Input::ButtonDevice> Create(const Common::ParamPackage& params) override {
        return std::make_unique<VitaButton>(static_cast<u32>(params.Get("mask", 0)));
    }
};

/// The circle pad, from the left stick.
class VitaAnalog final : public Input::AnalogDevice {
public:
    std::tuple<float, float> GetStatus() const override {
        if (g_window == nullptr) {
            return {0.0f, 0.0f};
        }
        return g_window->LeftStick();
    }
};

class VitaAnalogFactory final : public Input::Factory<Input::AnalogDevice> {
public:
    std::unique_ptr<Input::AnalogDevice> Create(const Common::ParamPackage&) override {
        return std::make_unique<VitaAnalog>();
    }
};

/**
 * Vita button per guest button, in NativeButton order.
 *
 * The face buttons are laid out by position rather than by name: the 3DS's A is where the Vita's
 * circle is. L and R serve the guest's shoulders, and the guest's ZL/ZR and its debug pads have
 * nowhere to go on a console with one pair - they stay unbound rather than stealing something a
 * game needs. Select doubles as the pause menu's release, so it is bound but also watched by the
 * frontend.
 */
constexpr std::array<u32, Settings::NativeButton::NumButtons> ButtonMasks = {
    SCE_CTRL_CIRCLE,   // A
    SCE_CTRL_CROSS,    // B
    SCE_CTRL_TRIANGLE, // X
    SCE_CTRL_SQUARE,   // Y
    SCE_CTRL_UP,       // Up
    SCE_CTRL_DOWN,     // Down
    SCE_CTRL_LEFT,     // Left
    SCE_CTRL_RIGHT,    // Right
    SCE_CTRL_LTRIGGER, // L
    SCE_CTRL_RTRIGGER, // R
    SCE_CTRL_START,    // Start
    SCE_CTRL_SELECT,   // Select
    0,                 // Debug
    0,                 // Gpio14
    0,                 // ZL
    0,                 // ZR
    0,                 // Home
    0,                 // Power
};

} // Anonymous namespace

void InitInput(EmuWindowVita& window) {
    g_window = &window;
    Input::RegisterFactory<Input::ButtonDevice>(EngineName, std::make_shared<VitaButtonFactory>());
    Input::RegisterFactory<Input::AnalogDevice>(EngineName, std::make_shared<VitaAnalogFactory>());

    auto& profile = Settings::values.current_input_profile;
    for (std::size_t i = 0; i < ButtonMasks.size(); i++) {
        Common::ParamPackage param;
        param.Set("engine", EngineName);
        param.Set("mask", static_cast<int>(ButtonMasks[i]));
        profile.buttons[i] = param.Serialize();
    }
    {
        Common::ParamPackage param;
        param.Set("engine", EngineName);
        profile.analogs[Settings::NativeAnalog::CirclePad] = param.Serialize();
    }
    // Nothing on this console can drive the C-stick, and a device that always reads centred is
    // what the guest expects from a 3DS without one.
    profile.analogs[Settings::NativeAnalog::CStick].clear();
    // The touchscreen is the window's own device, fed from the front panel.
    profile.touch_device = "engine:emu_window";
}

void ShutdownInput() {
    Input::UnregisterFactory<Input::ButtonDevice>(EngineName);
    Input::UnregisterFactory<Input::AnalogDevice>(EngineName);
    g_window = nullptr;
}

} // namespace VitaFrontend
