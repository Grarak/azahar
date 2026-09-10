// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/renderer_base.h"
#include "video_core/renderer_gxm/gxm_rasterizer.h"

namespace Core {
class System;
}

namespace GxmRenderer {

/**
 * The GXM renderer. Presentation is the frontend's: the Vita samples the guest
 * framebuffers in place through its own GXM layer, and asks this renderer where they are.
 * From P4 the answer is the cached surface the display transfer wrote, as a texture, and
 * guest memory only when the cache holds nothing for that framebuffer.
 */
class RendererGxm : public VideoCore::RendererBase {
public:
    explicit RendererGxm(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window);
    ~RendererGxm() override;

    [[nodiscard]] VideoCore::RasterizerInterface* Rasterizer() override {
        return &rasterizer;
    }

    void SwapBuffers() override;
    void TryPresent(int timeout_ms, bool is_secondary) override {}

    [[nodiscard]] std::optional<VideoCore::GuestFramebuffer> GetGuestFramebuffer(
        int screen) override;

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    RasterizerGxm rasterizer;
    VideoCore::GuestFramebufferPicker picker;
};

} // namespace GxmRenderer
