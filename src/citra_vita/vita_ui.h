// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VitaFrontend {

class EmuWindowVita;

/**
 * The on-screen interface: a game list before anything is running and a pause menu over a
 * running title, drawn with Dear ImGui through the raw-GXM backend (imgui_impl_gxm).
 *
 * Structure follows DSVita's, which is the working reference for this shape of interface on this
 * console: one menu that owns the loop while it is open, a settings body shared between the
 * pre-launch and in-game paths with the in-game one showing only what can be changed while a
 * title runs, and gamepad navigation as the primary input with touch alongside it.
 *
 * Every method here runs on the emulation thread, which is the thread that owns GXM.
 */
class Ui {
public:
    /// What the UI is asking the frontend to do, returned from Update.
    enum class Action {
        None,
        LaunchGame,
        ResumeGame,
        QuitGame,
        ExitApplication,
        SaveState,
        LoadState,
    };

    explicit Ui(Core::System& system, EmuWindowVita& window);
    ~Ui();

    /// Builds this frame's interface and returns what the user asked for. Call once per frame,
    /// before Draw.
    Action Update();

    /// Issues the built frame's draw commands. Must be called inside a GxmPresent frame.
    void Draw();

    /// True while a menu owns the input and the emulation should not advance.
    [[nodiscard]] bool MenuOpen() const {
        return state != State::InGame;
    }

    /// The game the user chose, valid after Action::LaunchGame.
    [[nodiscard]] const std::string& SelectedGamePath() const {
        return selected_game_path;
    }

    /// The savestate slot the user chose, valid after Action::SaveState / Action::LoadState.
    [[nodiscard]] u32 SelectedSlot() const {
        return selected_slot;
    }

    /// Switches to the in-game state, where nothing is drawn over the title.
    void EnterGame();

    /// Opens the pause menu over a running title.
    void OpenPauseMenu();

    /// Shows a full-screen message instead of a menu, for the stretches where the frontend is
    /// busy and the display would otherwise sit on a stale frame.
    void SetStatusMessage(std::string message);
    void ClearStatusMessage();

private:
    enum class State {
        GameList,
        GameSettings,
        InGame,
        Pause,
        PauseSettings,
        Savestates,
    };

    /// One row of the settings body. The value is read and written through closures so a row can
    /// name any setting without this file knowing the shape of the settings structure.
    struct Setting {
        enum class Kind { Toggle, Choice, Slider };

        const char* title;
        const char* description;
        Kind kind;
        /// Whether the setting can be changed while a title is running. The pause menu shows
        /// only these; the rest would need a reboot to take effect and pretending otherwise
        /// is how a menu comes to lie about what the emulator is doing.
        bool runtime;
        std::function<bool()> get_bool;
        std::function<void(bool)> set_bool;
        std::function<int()> get_choice;
        std::function<void(int)> set_choice;
        std::vector<const char*> choices;
        std::function<int()> get_slider;
        std::function<void(int)> set_slider;
        int slider_min;
        int slider_max;
    };

    struct GameEntry {
        std::string path;
        std::string name;
    };

    void BuildSettings();
    void ScanGames();

    Action DrawGameList();
    Action DrawGameSettings();
    Action DrawPauseMenu();
    Action DrawPauseSettings();
    Action DrawSavestates();
    void DrawStatusMessage();
    /// The GPU counters over the game (gxm flag gpuhud), drawn every frame like vitaGL's
    /// debugger window.
    void DrawGpuHud();
    void DrawStatsHud();

    /// Draws the settings body, showing only runtime-changeable rows when asked.
    void DrawSettingsBody(bool runtime_only);
    void DrawSetting(Setting& setting, std::size_t index);

    /// True when the user pressed cancel and this overlay was the thing focused, which is what
    /// separates "close the menu" from "leave the combo box the cancel already closed" - ImGui
    /// has consumed the press by the time this runs.
    [[nodiscard]] bool BackPressed() const;

    Core::System& system;
    EmuWindowVita& window;

    State state{State::GameList};
    std::string selected_game_path;
    u32 selected_slot{};
    std::string status_message;

    std::vector<GameEntry> games;
    int selected_game{};
    std::vector<Setting> settings;
    bool overlay_focused{};
};

} // namespace VitaFrontend
