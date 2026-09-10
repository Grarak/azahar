// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/pica/regs_external.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_rasterizer.h"

namespace Core {
class System;
}

namespace SwRenderer {

struct ScreenInfo {
    u32 width;
    u32 height;
    std::vector<u8> pixels;
};

class RendererSoftware : public VideoCore::RendererBase {
public:
    explicit RendererSoftware(Core::System& system, Pica::PicaCore& pica,
                              Frontend::EmuWindow& window);
    ~RendererSoftware() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] const ScreenInfo& Screen(VideoCore::ScreenId id) const noexcept {
        return screen_infos[static_cast<u32>(id)];
    }

    void SwapBuffers() override;
    void TryPresent(int timeout_ms, bool is_secondary) override {}

    [[nodiscard]] std::optional<VideoCore::GuestFramebuffer> GetGuestFramebuffer(
        int screen) override {
        return picker.Get(owner_gpu, screen);
    }

    /// A frontend that presents straight out of guest memory (the Vita) has no use for the
    /// screen_infos conversion; with this set SwapBuffers skips it entirely.
    void SetFrontendPresentsGuestMemory(bool value) override {
        frontend_presents_guest_memory = value;
    }

private:
    void PrepareRenderTarget();
    void LoadFBToScreenInfo(int i, const Pica::ColorFill& color_fill);

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    RasterizerSoftware rasterizer;
    VideoCore::GuestFramebufferPicker picker;
    std::array<ScreenInfo, 3> screen_infos{};
    bool frontend_presents_guest_memory{};
};

} // namespace SwRenderer
