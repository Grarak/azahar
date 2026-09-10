// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include "common/color.h"
#include "core/core.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_software/renderer_software.h"

namespace SwRenderer {

RendererSoftware::RendererSoftware(Core::System& system, Pica::PicaCore& pica_,
                                   Frontend::EmuWindow& window)
    : VideoCore::RendererBase{system, window, nullptr}, memory{system.Memory()}, pica{pica_},
      rasterizer{memory, pica}, picker{pica, memory} {}

RendererSoftware::~RendererSoftware() = default;

void RendererSoftware::SwapBuffers() {
    rasterizer.NewTextureFrame();
    system.perf_stats->StartSwap();
    if (!frontend_presents_guest_memory) {
        PrepareRenderTarget();
    }
    system.perf_stats->EndSwap();
    EndFrame();
}


void RendererSoftware::PrepareRenderTarget() {
    const auto& regs_lcd = pica.regs_lcd;
    for (u32 i = 0; i < 3; i++) {
        const u32 fb_id = i == 2 ? 1 : 0;

        const auto color_fill = fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
        LoadFBToScreenInfo(i, color_fill);
    }
}

void RendererSoftware::LoadFBToScreenInfo(int i, const Pica::ColorFill& color_fill) {
    const u32 fb_id = i == 2 ? 1 : 0;
    const auto& framebuffer = pica.regs.framebuffer_config[fb_id];
    auto& info = screen_infos[i];

    // PickFramebufferAddr carries both policies: completion-order selection when blits target
    // the display buffers, and the register selection with the fill-hold when the title draws
    // into them directly. (owner_gpu is null during shutdown while the render thread is still
    // presenting; the helper degrades to plain register selection there.)
    const PAddr framebuffer_addr = picker.PickFramebufferAddr(owner_gpu, fb_id);
    const s32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
    const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);

    const s32 pixel_stride = framebuffer.stride / bpp;
    info.height = framebuffer.height;
    info.width = pixel_stride;
    info.pixels.resize(info.width * info.height * 4);

    // This conversion runs on the GPU thread for every present, so it uses per-format tight
    // loops instead of a per-pixel switch through a lambda.
    if (color_fill.is_enabled) {
        const std::array<u8, 4> fill{static_cast<u8>(color_fill.color_r),
                                     static_cast<u8>(color_fill.color_g),
                                     static_cast<u8>(color_fill.color_b), 255};
        for (u32 p = 0; p < info.width * info.height; p++) {
            std::memcpy(info.pixels.data() + p * 4, fill.data(), 4);
        }
        return;
    }

    const auto convert_rows = [&](auto decode) {
        for (u32 y = 0; y < info.height; y++) {
            const u8* row = framebuffer_data + (y * pixel_stride + pixel_stride) * bpp;
            u8* dest_col = info.pixels.data() + y * 4;
            for (u32 x = 0; x < info.width; x++) {
                const Common::Vec4<u8> color = decode(row);
                row -= bpp;
                std::memcpy(dest_col, color.AsArray(), 4);
                dest_col += info.height * 4;
            }
        }
    };

    switch (framebuffer.color_format) {
    case Pica::PixelFormat::RGBA8:
        convert_rows([](const u8* p) { return Common::Color::DecodeRGBA8(p); });
        break;
    case Pica::PixelFormat::RGB8:
        convert_rows([](const u8* p) { return Common::Color::DecodeRGB8(p); });
        break;
    case Pica::PixelFormat::RGB565:
        convert_rows([](const u8* p) { return Common::Color::DecodeRGB565(p); });
        break;
    case Pica::PixelFormat::RGB5A1:
        convert_rows([](const u8* p) { return Common::Color::DecodeRGB5A1(p); });
        break;
    case Pica::PixelFormat::RGBA4:
        convert_rows([](const u8* p) { return Common::Color::DecodeRGBA4(p); });
        break;
    default:
        UNREACHABLE();
    }
}

} // namespace SwRenderer
