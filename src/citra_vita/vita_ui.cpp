// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>

#include <imgui_vita2d/imgui.h>

#include "citra_vita/imgui_impl_gxm.h"

#include "citra_vita/emu_window_vita.h"
#include "citra_vita/gxm_present.h"
#include "citra_vita/hud_stats.h"
#include "citra_vita/vita_ui.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/renderer_gxm/gxm_flags.h"

namespace VitaFrontend {

namespace {
constexpr float ScreenWidth = 960.0f;
constexpr float ScreenHeight = 544.0f;

/// Where titles are looked for. One directory, no recursion: a console file manager is enough
/// work to use that a deep tree is worse than a flat one.
constexpr char GameDirectory[] = "ux0:/data/azahar/roms";

bool HasRomExtension(const std::string& name) {
    static constexpr const char* extensions[] = {".3ds", ".3dsx", ".cci", ".cxi",
                                                 ".app", ".elf",  ".axf"};
    const auto dot = name.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string extension = name.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::any_of(std::begin(extensions), std::end(extensions),
                       [&](const char* candidate) { return extension == candidate; });
}

/// A full-screen window with no decoration, which is what every screen here wants: the display
/// is fixed and there is nothing to arrange windows against.
bool BeginFullscreen(const char* id) {
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(ScreenWidth, ScreenHeight));
    return ImGui::Begin(id, nullptr,
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                            ImGuiWindowFlags_NoBringToFrontOnFocus);
}

/// A centred panel over whatever is behind it, for the pause menu.
bool BeginCentered(const char* id, float width, float height) {
    ImGui::SetNextWindowPos(ImVec2((ScreenWidth - width) * 0.5f, (ScreenHeight - height) * 0.5f));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    return ImGui::Begin(id, nullptr,
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
}
} // Anonymous namespace

Ui::Ui(Core::System& system_, EmuWindowVita& window_) : system{system_}, window{window_} {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.WindowPadding = ImVec2(16.0f, 16.0f);
    style.ItemSpacing = ImVec2(10.0f, 10.0f);
    // The console is held at arm's length, so everything is a size up from a desktop default.
    ImGui::GetIO().FontGlobalScale = 1.2f;

    ImGui_ImplGxm_Init();
    // The backend has uploaded the font atlas to a GXM texture by now, and the pixels it was
    // built from are 4 MB of heap nothing reads again. ImGui rebuilds them on demand if anything
    // ever does.
    ImGui::GetIO().Fonts->ClearTexData();
    ImGui_ImplGxm_TouchUsage(true);
    ImGui_ImplGxm_GamepadUsage(true);
    ImGui_ImplGxm_MouseStickUsage(true);

    BuildSettings();
    ScanGames();
}

Ui::~Ui() {
    ImGui_ImplGxm_Shutdown();
    ImGui::DestroyContext();
}

void Ui::ScanGames() {
    games.clear();
    FileUtil::FSTEntry root;
    if (!FileUtil::ScanDirectoryTree(GameDirectory, root)) {
        LOG_WARNING(Frontend, "No game directory at {}", GameDirectory);
        return;
    }
    for (const auto& entry : root.children) {
        if (entry.isDirectory || !HasRomExtension(entry.virtualName)) {
            continue;
        }
        games.push_back({entry.physicalName, entry.virtualName});
    }
    std::sort(games.begin(), games.end(),
              [](const GameEntry& a, const GameEntry& b) { return a.name < b.name; });
    LOG_INFO(Frontend, "Found {} titles in {}", games.size(), GameDirectory);
}

void Ui::BuildSettings() {
    const auto toggle = [&](const char* title, const char* description, bool runtime,
                            std::function<bool()> get, std::function<void(bool)> set) {
        Setting setting{};
        setting.title = title;
        setting.description = description;
        setting.kind = Setting::Kind::Toggle;
        setting.runtime = runtime;
        setting.get_bool = std::move(get);
        setting.set_bool = std::move(set);
        settings.push_back(std::move(setting));
    };
    const auto choice = [&](const char* title, const char* description, bool runtime,
                            std::vector<const char*> choices, std::function<int()> get,
                            std::function<void(int)> set) {
        Setting setting{};
        setting.title = title;
        setting.description = description;
        setting.kind = Setting::Kind::Choice;
        setting.runtime = runtime;
        setting.choices = std::move(choices);
        setting.get_choice = std::move(get);
        setting.set_choice = std::move(set);
        settings.push_back(std::move(setting));
    };
    const auto slider = [&](const char* title, const char* description, bool runtime, int min,
                            int max, std::function<int()> get, std::function<void(int)> set) {
        Setting setting{};
        setting.title = title;
        setting.description = description;
        setting.kind = Setting::Kind::Slider;
        setting.runtime = runtime;
        setting.slider_min = min;
        setting.slider_max = max;
        setting.get_slider = std::move(get);
        setting.set_slider = std::move(set);
        settings.push_back(std::move(setting));
    };

    slider("Speed limit", "Percentage of full speed to aim for. Lower it to save battery.", true,
           0, 200, [] { return static_cast<int>(Settings::values.frame_limit.GetValue()); },
           [](int value) { Settings::values.frame_limit = static_cast<double>(value); });
    slider("Volume", "Output volume.", true, 0, 100,
           [] { return static_cast<int>(Settings::values.volume.GetValue() * 100.0f); },
           [](int value) { Settings::values.volume = static_cast<float>(value) / 100.0f; });
    toggle("Stretch audio", "Keeps audio continuous when the emulator runs below full speed.",
           true, [] { return Settings::values.enable_audio_stretching.GetValue(); },
           [](bool value) { Settings::values.enable_audio_stretching = value; });
    choice("Screen layout", "How the two screens are arranged on the display.", true,
           {"Default", "Single screen", "Large screen", "Side by side", "Hybrid"},
           [] { return static_cast<int>(Settings::values.layout_option.GetValue()); },
           [](int value) {
               Settings::values.layout_option = static_cast<Settings::LayoutOption>(value);
           });
    toggle("Swap screens", "Puts the touch screen where the top screen would be.", true,
           [] { return Settings::values.swap_screen.GetValue(); },
           [](bool value) { Settings::values.swap_screen = value; });
    choice("Region", "The console region reported to the title.", false,
           {"Auto", "Japan", "USA", "Europe", "Australia", "China", "Korea", "Taiwan"},
           [] { return Settings::values.region_value.GetValue() + 1; },
           [](int value) { Settings::values.region_value = value - 1; });
}

bool Ui::BackPressed() const {
    // ImGui has already handled the press by the time this is asked, so a cancel that popped a
    // combo or left a child must not also close the overlay. It only counts when the overlay
    // itself held focus at the end of the previous frame.
    //
    // This version of ImGui has no key enum for a pad button; navigation arrives as the
    // NavInputs array, and a duration of exactly zero is the frame the press landed on.
    const ImGuiIO& io = ImGui::GetIO();
    const bool cancel_pressed = io.NavInputsDownDuration[ImGuiNavInput_Cancel] == 0.0f;
    return overlay_focused && cancel_pressed;
}

void Ui::EnterGame() {
    state = State::InGame;
    window.SetInputToGuest(true);
}

void Ui::OpenPauseMenu() {
    state = State::Pause;
    window.SetInputToGuest(false);
}

void Ui::SetStatusMessage(std::string message) {
    status_message = std::move(message);
}

void Ui::ClearStatusMessage() {
    status_message.clear();
}

void Ui::DrawSetting(Setting& setting, std::size_t index) {
    ImGui::PushID(static_cast<int>(index));
    ImGui::TextUnformatted(setting.title);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", setting.description);

    switch (setting.kind) {
    case Setting::Kind::Toggle: {
        bool value = setting.get_bool();
        if (ImGui::Checkbox("##toggle", &value)) {
            setting.set_bool(value);
        }
        break;
    }
    case Setting::Kind::Choice: {
        int value = setting.get_choice();
        value = std::clamp(value, 0, static_cast<int>(setting.choices.size()) - 1);
        ImGui::SetNextItemWidth(320.0f);
        if (ImGui::BeginCombo("##choice", setting.choices[value])) {
            for (int i = 0; i < static_cast<int>(setting.choices.size()); i++) {
                if (ImGui::Selectable(setting.choices[i], i == value)) {
                    setting.set_choice(i);
                }
            }
            ImGui::EndCombo();
        }
        break;
    }
    case Setting::Kind::Slider: {
        int value = setting.get_slider();
        ImGui::SetNextItemWidth(320.0f);
        if (ImGui::SliderInt("##slider", &value, setting.slider_min, setting.slider_max)) {
            setting.set_slider(value);
        }
        break;
    }
    }

    ImGui::Separator();
    ImGui::PopID();
}

void Ui::DrawSettingsBody(bool runtime_only) {
    for (std::size_t i = 0; i < settings.size(); i++) {
        if (runtime_only && !settings[i].runtime) {
            continue;
        }
        DrawSetting(settings[i], i);
    }
}

Ui::Action Ui::DrawGameList() {
    Action action = Action::None;
    BeginFullscreen("##gamelist");

    ImGui::TextUnformatted("Azahar");
    ImGui::SameLine();
    // %zu would be a crash here, not a cosmetic problem: vitasdk's newlib is built without
    // _WANT_IO_C99_FORMATS, so it does not know the z modifier, prints it literally and consumes
    // no argument - and the %s that follows then reads the count instead of the string. With an
    // empty list that count is zero and the formatter walks into strlen(nullptr).
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  %u titles in %s",
                       static_cast<unsigned>(games.size()), GameDirectory);
    ImGui::Separator();

    if (games.empty()) {
        ImGui::TextWrapped("No titles found. Copy .3ds or .cci files into %s and choose Rescan.",
                           GameDirectory);
    }

    ImGui::BeginChild("##list", ImVec2(0.0f, 380.0f));
    for (int i = 0; i < static_cast<int>(games.size()); i++) {
        if (ImGui::Selectable(games[i].name.c_str(), i == selected_game)) {
            selected_game = i;
            selected_game_path = games[i].path;
            action = Action::LaunchGame;
        }
        if (ImGui::IsItemFocused()) {
            selected_game = i;
        }
    }
    ImGui::EndChild();

    ImGui::Separator();
    if (ImGui::Button("Settings", ImVec2(200.0f, 48.0f))) {
        state = State::GameSettings;
        overlay_focused = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Rescan", ImVec2(200.0f, 48.0f))) {
        ScanGames();
    }
    ImGui::SameLine();
    if (ImGui::Button("Exit", ImVec2(200.0f, 48.0f))) {
        action = Action::ExitApplication;
    }

    ImGui::End();
    return action;
}

Ui::Action Ui::DrawGameSettings() {
    BeginFullscreen("##gamesettings");
    ImGui::TextUnformatted("Settings");
    ImGui::Separator();

    ImGui::BeginChild("##settings_scroll", ImVec2(0.0f, 420.0f));
    DrawSettingsBody(false);
    ImGui::EndChild();

    ImGui::Separator();
    const bool back = ImGui::Button("Back", ImVec2(200.0f, 48.0f)) || BackPressed();
    overlay_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGui::End();

    if (back) {
        state = State::GameList;
        overlay_focused = false;
    }
    return Action::None;
}

Ui::Action Ui::DrawPauseMenu() {
    Action action = Action::None;
    BeginCentered("##pause", 360.0f, 420.0f);
    ImGui::TextUnformatted("Paused");
    ImGui::Separator();

    const ImVec2 button_size{320.0f, 48.0f};
    if (ImGui::Button("Resume", button_size)) {
        action = Action::ResumeGame;
    }
    if (ImGui::Button("Settings", button_size)) {
        state = State::PauseSettings;
        overlay_focused = false;
    }
    if (ImGui::Button("Save states", button_size)) {
        state = State::Savestates;
        overlay_focused = false;
    }
    if (ImGui::Button("Dump frame", button_size)) {
        // The next presented frame, screens only, to ux0:data/azahar/frame_NN.ppm, and every
        // surface the cache holds at that frame's end to ux0:data/azahar/dump/.
        GxmPresent::DumpNextFrame();
        GxmRenderer::RequestSurfaceDump();
    }
    if (ImGui::Button("Quit game", button_size)) {
        action = Action::QuitGame;
    }
    if (ImGui::Button("Exit emulator", button_size)) {
        action = Action::ExitApplication;
    }

    if (BackPressed()) {
        action = Action::ResumeGame;
    }
    overlay_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGui::End();
    return action;
}

Ui::Action Ui::DrawPauseSettings() {
    BeginCentered("##pausesettings", 700.0f, 460.0f);
    ImGui::TextUnformatted("Settings");
    ImGui::Separator();

    ImGui::BeginChild("##pause_settings_scroll", ImVec2(0.0f, 320.0f));
    // Only what can be changed under a running title; the rest would need a reboot.
    DrawSettingsBody(true);
    ImGui::EndChild();

    ImGui::Separator();
    const bool back = ImGui::Button("Back", ImVec2(200.0f, 48.0f)) || BackPressed();
    overlay_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGui::End();

    if (back) {
        state = State::Pause;
        overlay_focused = false;
    }
    return Action::None;
}

Ui::Action Ui::DrawSavestates() {
    Action action = Action::None;
    BeginCentered("##savestates", 560.0f, 460.0f);
    ImGui::TextUnformatted("Save states");
    ImGui::Separator();

    ImGui::BeginChild("##slots", ImVec2(0.0f, 300.0f));
    for (u32 slot = 1; slot <= 9; slot++) {
        ImGui::PushID(static_cast<int>(slot));
        ImGui::Text("Slot %u", slot);
        ImGui::SameLine(140.0f);
        if (ImGui::Button("Save", ImVec2(140.0f, 40.0f))) {
            selected_slot = slot;
            action = Action::SaveState;
        }
        ImGui::SameLine();
        if (ImGui::Button("Load", ImVec2(140.0f, 40.0f))) {
            selected_slot = slot;
            action = Action::LoadState;
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::Separator();
    const bool back = ImGui::Button("Back", ImVec2(200.0f, 48.0f)) || BackPressed();
    overlay_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGui::End();

    if (back) {
        state = State::Pause;
        overlay_focused = false;
    }
    return action;
}

void Ui::DrawStatusMessage() {
    BeginCentered("##status", 560.0f, 140.0f);
    ImGui::TextWrapped("%s", status_message.c_str());
    ImGui::End();
}

void Ui::DrawGpuHud() {
    static const bool on = GxmRenderer::GxmFlag("gpuhud");
    if (!on) {
        return;
    }
    const std::string& text = GxmPresent::GpuLiveText();
    ImGui::SetNextWindowPos(ImVec2(8.0f, 8.0f));
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##gpuhud", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
    ImGui::TextUnformatted(text.empty() ? "GPU live: waiting for the first second" : text.c_str());
    ImGui::End();
}

void Ui::DrawStatsHud() {
#ifndef VITA_DIAGNOSTICS
    // A measurement overlay, so the release build does not draw one.
    return;
#else
    // Top right, every frame while a title runs: what the console line prints once a second.
    // A dozen text lines of ImGui, so a few hundred vertices; `nostats` hides it.
    static const bool off = GxmRenderer::GxmFlag("nostats");
    if (off || !HudStats::valid.load(std::memory_order_acquire)) {
        return;
    }
    using namespace HudStats;
    const u32 shown = shown_fps.load(std::memory_order_relaxed);
    const u32 fps10 = game_fps10.load(std::memory_order_relaxed);
    ImGui::SetNextWindowPos(ImVec2(960.0f - 8.0f, 8.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##stats", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
    // Three bare lines: speed/game fps, frame time/shown fps, guest core 0/core 1 busy.
    ImGui::Text("%u%%/%u", speed_percent.load(std::memory_order_relaxed), (fps10 + 5) / 10);
    ImGui::Text("%.1fms/%ufps", shown != 0 ? 1000.0f / static_cast<float>(shown) : 0.0f, shown);
    ImGui::End();
#endif // VITA_DIAGNOSTICS
}

Ui::Action Ui::Update() {
    ImGui_ImplGxm_NewFrame();

    Action action = Action::None;
    if (!status_message.empty()) {
        DrawStatusMessage();
    } else {
        switch (state) {
        case State::GameList:
            action = DrawGameList();
            break;
        case State::GameSettings:
            action = DrawGameSettings();
            break;
        case State::InGame:
            DrawGpuHud();
            DrawStatsHud();
            break;
        case State::Pause:
            action = DrawPauseMenu();
            break;
        case State::PauseSettings:
            action = DrawPauseSettings();
            break;
        case State::Savestates:
            action = DrawSavestates();
            break;
        }
    }

    if (action == Action::ResumeGame) {
        EnterGame();
    } else if (action == Action::QuitGame) {
        state = State::GameList;
        overlay_focused = false;
    }

    ImGui::Render();
    return action;
}

void Ui::Draw() {
    ImGui_ImplGxm_RenderDrawData(ImGui::GetDrawData());
}

} // namespace VitaFrontend
