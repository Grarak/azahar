// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cmath>
#include <cstring>

#include <imgui_vita2d/imgui.h>
#include <psp2/ctrl.h>
#include <psp2/gxm.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/touch.h>

#include "citra_vita/gxm_present.h"
#include "citra_vita/gxm_shaders.gen.h"
#include "citra_vita/imgui_impl_gxm.h"
#include "common/logging/log.h"

namespace {

using VitaFrontend::GxmPresent::Context;
using VitaFrontend::GxmPresent::DisplayHeight;
using VitaFrontend::GxmPresent::DisplayWidth;

struct BackendState {
    SceGxmShaderPatcherId vertex_id{}, fragment_id{};
    SceGxmVertexProgram* vertex_program{};
    SceGxmFragmentProgram* fragment_program{};
    const SceGxmProgramParameter* scale_param{};
    const SceGxmProgramParameter* offset_param{};

    void* font_pixels{};
    SceGxmTexture font_texture{};

    bool touch_enabled{true};
    bool gamepad_enabled{true};
    bool mouse_stick_enabled{true};
    float cursor_x{DisplayWidth / 2.0f};
    float cursor_y{DisplayHeight / 2.0f};
    u64 last_time{};
};

BackendState s;

} // Anonymous namespace

bool ImGui_ImplGxm_Init() {
    auto* patcher = VitaFrontend::GxmPresent::Patcher();
    if (patcher == nullptr) {
        return false;
    }

    const SceGxmProgram* vertex = GxmShader_imgui_v();
    const SceGxmProgram* fragment = GxmShader_imgui_f();
    if (sceGxmProgramCheck(vertex) != 0 || sceGxmProgramCheck(fragment) != 0) {
        LOG_CRITICAL(Frontend, "the vendored imgui programs are not valid GXP");
        return false;
    }
    sceGxmShaderPatcherRegisterProgram(patcher, vertex, &s.vertex_id);
    sceGxmShaderPatcherRegisterProgram(patcher, fragment, &s.fragment_id);

    const auto attribute = [&](const char* name, u16 offset, SceGxmAttributeFormat format,
                               u8 components) {
        const SceGxmProgramParameter* param = sceGxmProgramFindParameterByName(vertex, name);
        SceGxmVertexAttribute attr{};
        attr.streamIndex = 0;
        attr.offset = offset;
        attr.format = static_cast<u8>(format);
        attr.componentCount = components;
        attr.regIndex = sceGxmProgramParameterGetResourceIndex(param);
        return attr;
    };
    static_assert(sizeof(ImDrawVert) == 20, "ImDrawVert layout changed");
    SceGxmVertexAttribute attrs[3] = {
        attribute("aPosition", IM_OFFSETOF(ImDrawVert, pos), SCE_GXM_ATTRIBUTE_FORMAT_F32, 2),
        attribute("aTexcoord", IM_OFFSETOF(ImDrawVert, uv), SCE_GXM_ATTRIBUTE_FORMAT_F32, 2),
        attribute("aColor", IM_OFFSETOF(ImDrawVert, col), SCE_GXM_ATTRIBUTE_FORMAT_U8N, 4),
    };
    SceGxmVertexStream stream{};
    stream.stride = sizeof(ImDrawVert);
    stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    if (sceGxmShaderPatcherCreateVertexProgram(patcher, s.vertex_id, attrs, 3, &stream, 1,
                                               &s.vertex_program) < 0) {
        return false;
    }

    // The UI is the one thing that blends.
    SceGxmBlendInfo blend{};
    blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
    blend.colorFunc = SCE_GXM_BLEND_FUNC_ADD;
    blend.alphaFunc = SCE_GXM_BLEND_FUNC_ADD;
    blend.colorSrc = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
    blend.colorDst = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ONE;
    blend.alphaDst = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    if (sceGxmShaderPatcherCreateFragmentProgram(patcher, s.fragment_id,
                                                 SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
                                                 SCE_GXM_MULTISAMPLE_NONE, &blend, vertex,
                                                 &s.fragment_program) < 0) {
        return false;
    }
    s.scale_param = sceGxmProgramFindParameterByName(vertex, "uScale");
    s.offset_param = sceGxmProgramFindParameterByName(vertex, "uOffset");

    // Font atlas into a GPU-visible linear texture; ImGui's copy can be dropped after.
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    s.font_pixels = VitaFrontend::GxmPresent::AllocMapped(static_cast<u32>(width * height * 4));
    if (s.font_pixels == nullptr) {
        return false;
    }
    std::memcpy(s.font_pixels, pixels, static_cast<size_t>(width) * height * 4);
    // ImGui hands the atlas over as RGBA byte order, R first - the ABGR swizzle, like the
    // display surface.
    sceGxmTextureInitLinear(&s.font_texture, s.font_pixels, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR,
                            static_cast<u32>(width), static_cast<u32>(height), 1);
    sceGxmTextureSetMinFilter(&s.font_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    sceGxmTextureSetMagFilter(&s.font_texture, SCE_GXM_TEXTURE_FILTER_LINEAR);
    io.Fonts->TexID = &s.font_texture;

    io.DisplaySize = ImVec2(DisplayWidth, DisplayHeight);
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.BackendPlatformName = "imgui_impl_gxm";
    io.BackendRendererName = "imgui_impl_gxm";
    return true;
}

void ImGui_ImplGxm_Shutdown() {
    auto* patcher = VitaFrontend::GxmPresent::Patcher();
    if (patcher != nullptr) {
        if (s.vertex_program != nullptr) {
            sceGxmShaderPatcherReleaseVertexProgram(patcher, s.vertex_program);
        }
        if (s.fragment_program != nullptr) {
            sceGxmShaderPatcherReleaseFragmentProgram(patcher, s.fragment_program);
        }
        if (s.vertex_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(patcher, s.vertex_id);
        }
        if (s.fragment_id != nullptr) {
            sceGxmShaderPatcherUnregisterProgram(patcher, s.fragment_id);
        }
    }
    s = BackendState{};
}

void ImGui_ImplGxm_TouchUsage(bool enabled) {
    s.touch_enabled = enabled;
}
void ImGui_ImplGxm_GamepadUsage(bool enabled) {
    s.gamepad_enabled = enabled;
}
void ImGui_ImplGxm_MouseStickUsage(bool enabled) {
    s.mouse_stick_enabled = enabled;
}

void ImGui_ImplGxm_NewFrame() {
    ImGuiIO& io = ImGui::GetIO();

    const u64 now = sceKernelGetProcessTimeWide();
    io.DeltaTime = s.last_time != 0 ? (now - s.last_time) / 1e6f : 1.0f / 60.0f;
    s.last_time = now;

    if (s.gamepad_enabled) {
        SceCtrlData pad{};
        sceCtrlPeekBufferPositive(0, &pad, 1);
        std::memset(io.NavInputs, 0, sizeof(io.NavInputs));
        const auto nav = [&](ImGuiNavInput_ input, u32 button) {
            if (pad.buttons & button) {
                io.NavInputs[input] = 1.0f;
            }
        };
        nav(ImGuiNavInput_Activate, SCE_CTRL_CROSS);
        nav(ImGuiNavInput_Cancel, SCE_CTRL_CIRCLE);
        nav(ImGuiNavInput_Menu, SCE_CTRL_SQUARE);
        nav(ImGuiNavInput_Input, SCE_CTRL_TRIANGLE);
        nav(ImGuiNavInput_DpadUp, SCE_CTRL_UP);
        nav(ImGuiNavInput_DpadDown, SCE_CTRL_DOWN);
        nav(ImGuiNavInput_DpadLeft, SCE_CTRL_LEFT);
        nav(ImGuiNavInput_DpadRight, SCE_CTRL_RIGHT);
        nav(ImGuiNavInput_FocusPrev, SCE_CTRL_LTRIGGER);
        nav(ImGuiNavInput_FocusNext, SCE_CTRL_RTRIGGER);

        if (s.mouse_stick_enabled) {
            // The right stick drives a cursor, for the odd widget navigation cannot reach.
            const auto axis = [](u8 raw) {
                const float value = (raw - 128.0f) / 128.0f;
                return std::abs(value) < 0.25f ? 0.0f : value;
            };
            s.cursor_x += axis(pad.rx) * 700.0f * io.DeltaTime;
            s.cursor_y += axis(pad.ry) * 700.0f * io.DeltaTime;
            s.cursor_x = std::min(std::max(s.cursor_x, 0.0f), static_cast<float>(DisplayWidth));
            s.cursor_y = std::min(std::max(s.cursor_y, 0.0f), static_cast<float>(DisplayHeight));
        }
    }

    bool touched = false;
    if (s.touch_enabled) {
        SceTouchData touch{};
        if (sceTouchPeek(SCE_TOUCH_PORT_FRONT, &touch, 1) > 0 && touch.reportNum > 0) {
            // Panel coordinates are twice the display in each axis.
            s.cursor_x = touch.report[0].x / 2.0f;
            s.cursor_y = touch.report[0].y / 2.0f;
            touched = true;
        }
    }
    io.MousePos = ImVec2(s.cursor_x, s.cursor_y);
    io.MouseDown[0] = touched;

    ImGui::NewFrame();
}

void ImGui_ImplGxm_RenderDrawData(ImDrawData* draw_data) {
    if (draw_data == nullptr || draw_data->CmdListsCount == 0) {
        return;
    }
    auto* context = Context();
    if (context == nullptr) {
        return;
    }

    sceGxmSetVertexProgram(context, s.vertex_program);
    sceGxmSetFragmentProgram(context, s.fragment_program);

    void* uniform_buffer = nullptr;
    sceGxmReserveVertexDefaultUniformBuffer(context, &uniform_buffer);
    const float scale[2] = {2.0f / DisplayWidth, -2.0f / DisplayHeight};
    const float offset[2] = {-1.0f, 1.0f};
    sceGxmSetUniformDataF(uniform_buffer, s.scale_param, 0, 2, scale);
    sceGxmSetUniformDataF(uniform_buffer, s.offset_param, 0, 2, offset);

    for (int n = 0; n < draw_data->CmdListsCount; n++) {
        const ImDrawList* list = draw_data->CmdLists[n];

        auto* verts = static_cast<ImDrawVert*>(VitaFrontend::GxmPresent::FrameAlloc(
            static_cast<u32>(list->VtxBuffer.Size) * sizeof(ImDrawVert)));
        auto* indices = static_cast<ImDrawIdx*>(VitaFrontend::GxmPresent::FrameAlloc(
            static_cast<u32>(list->IdxBuffer.Size) * sizeof(ImDrawIdx), 2));
        if (verts == nullptr || indices == nullptr) {
            break;
        }
        std::memcpy(verts, list->VtxBuffer.Data,
                    static_cast<size_t>(list->VtxBuffer.Size) * sizeof(ImDrawVert));
        std::memcpy(indices, list->IdxBuffer.Data,
                    static_cast<size_t>(list->IdxBuffer.Size) * sizeof(ImDrawIdx));
        sceGxmSetVertexStream(context, 0, verts);

        for (int c = 0; c < list->CmdBuffer.Size; c++) {
            const ImDrawCmd& cmd = list->CmdBuffer[c];
            if (cmd.UserCallback != nullptr) {
                cmd.UserCallback(list, &cmd);
                continue;
            }
            auto* texture = static_cast<SceGxmTexture*>(cmd.TextureId);
            if (texture != nullptr) {
                sceGxmSetFragmentTexture(context, 0, texture);
            }
            // Region clip is tile-granular (32 px); imgui's scissor mostly bounds scrolling
            // regions, where a slightly loose clip is invisible.
            const u32 cx0 = static_cast<u32>(std::max(cmd.ClipRect.x, 0.0f));
            const u32 cy0 = static_cast<u32>(std::max(cmd.ClipRect.y, 0.0f));
            const u32 cx1 = static_cast<u32>(std::min(cmd.ClipRect.z, ImGui::GetIO().DisplaySize.x));
            const u32 cy1 = static_cast<u32>(std::min(cmd.ClipRect.w, ImGui::GetIO().DisplaySize.y));
            if (cx1 <= cx0 || cy1 <= cy0) {
                continue;
            }
            sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE, cx0, cy0, cx1 - 1,
                                cy1 - 1);
            sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES, SCE_GXM_INDEX_FORMAT_U16,
                       indices + cmd.IdxOffset, cmd.ElemCount);
        }
    }
    // Back to the full screen for whoever draws next.
    sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, DisplayWidth - 1,
                        DisplayHeight - 1);
}
