// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// Dear ImGui backend over the frontend's own GXM presentation layer (gxm_present). Renders
// ImDrawData between GxmPresent::BeginFrame and EndFrame on the presenting thread, and polls
// the pad, the front touchscreen, and a stick-driven cursor itself, like the vita2d backend
// this replaces.

struct ImDrawData;

bool ImGui_ImplGxm_Init();
void ImGui_ImplGxm_Shutdown();
void ImGui_ImplGxm_NewFrame();
void ImGui_ImplGxm_RenderDrawData(ImDrawData* draw_data);

// Input configuration, same shape as the vita2d backend's.
void ImGui_ImplGxm_TouchUsage(bool enabled);
void ImGui_ImplGxm_GamepadUsage(bool enabled);
void ImGui_ImplGxm_MouseStickUsage(bool enabled);
