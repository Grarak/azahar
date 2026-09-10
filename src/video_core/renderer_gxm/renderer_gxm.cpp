// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "core/core.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_gxm/renderer_gxm.h"
#include "video_core/renderer_gxm/gxm_flags.h"

namespace GxmRenderer {

RendererGxm::RendererGxm(Core::System& system, Pica::PicaCore& pica_,
                         Frontend::EmuWindow& window)
    : VideoCore::RendererBase{system, window, nullptr}, memory{system.Memory()}, pica{pica_},
      rasterizer{memory, pica, system.CustomTexManager(), *this}, picker{pica, memory} {
    LOG_INFO(Render, "GXM renderer");
}

std::optional<VideoCore::GuestFramebuffer> RendererGxm::GetGuestFramebuffer(int screen) {
    VideoCore::GuestFramebuffer fb = picker.Get(owner_gpu, screen);
    if (!fb.fill_enabled && fb.pixels != nullptr && !fb.snapshot) {
        const auto& config = pica.regs.framebuffer_config[screen == 0 ? 0 : 1];
        const u32 pixel_stride = fb.stride / Pica::BytesPerPixel(fb.format);
        static const bool log_screens = GxmRenderer::GxmFlag("logdraw");
        if (log_screens) {
            static u32 shown[2] = {0, 0};
            if (shown[screen == 0 ? 0 : 1]++ % 400 == 0) {
                // Whether either of the guest's two buffers for this screen holds anything,
                // for a screen the CPU is supposed to be writing (libctru's console).
                const auto has_content = [&](PAddr addr) {
                    const u8* p = memory.GetPhysicalPointer(addr);
                    if (p == nullptr) {
                        return -1;
                    }
                    const u32 bytes = fb.stride * fb.height;
                    for (u32 i = 0; i < bytes; i++) {
                        if (p[i] != 0) {
                            return 1;
                        }
                    }
                    return 0;
                };
                const PAddr left1 = config.address_left1;
                const PAddr left2 = config.address_left2;
                LOG_INFO(Render,
                         "screen {}: addr {:08X} (left1 {:08X} content {}, left2 {:08X} content "
                         "{}, active_fb {}) stride {} format {} bpp {} -> {} px, config {}x{}, "
                         "fb height {}",
                         screen, fb.addr, left1, has_content(left1), left2, has_content(left2),
                         config.active_fb, fb.stride, static_cast<u32>(fb.format),
                         Pica::BytesPerPixel(fb.format), pixel_stride, config.width.Value(),
                         config.height.Value(), fb.height);
            }
        }
        Common::Rectangle<u32> rect{};
        fb.texture = rasterizer.AccelerateDisplay(screen, fb.addr, config.width, fb.height,
                                                  pixel_stride, fb.format, &rect);
        fb.tex_x = rect.left;
        fb.tex_y = rect.top;
        fb.tex_width = rect.right - rect.left;
        fb.tex_height = rect.bottom - rect.top;
    }
    return fb;
}

RendererGxm::~RendererGxm() = default;

void RendererGxm::SwapBuffers() {
    system.perf_stats->StartSwap();
    rasterizer.EndFrame();
    system.perf_stats->EndSwap();
    const u64 start = rasterizer.TimingDraws() ? Common::PipelineStats::NowUs() : 0;
    EndFrame();
    if (rasterizer.TimingDraws()) {
        rasterizer.NotePresentTime(Common::PipelineStats::NowUs() - start);
    }
}

} // namespace GxmRenderer
