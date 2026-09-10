// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include "common/common_types.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/guest_framebuffer.h"
#include "video_core/rasterizer_interface.h"

namespace Frontend {
class EmuWindow;
}

namespace Core {
class System;
}

namespace VideoCore {

class GPU;

enum class ScreenId : u32 {
    TopLeft,
    TopRight,
    Bottom,
};

struct RendererSettings {
    // Screenshot
    std::atomic_bool screenshot_requested{false};
    void* screenshot_bits{};
    std::function<void(bool)> screenshot_complete_callback;
    Layout::FramebufferLayout screenshot_framebuffer_layout;
    // Renderer
    std::atomic_bool bg_color_update_requested{false};
    std::atomic_bool shader_update_requested{false};
};

class RendererBase : NonCopyable {
public:
    explicit RendererBase(Core::System& system, Frontend::EmuWindow& window,
                          Frontend::EmuWindow* secondary_window);
    virtual ~RendererBase();

    /// Returns the rasterizer owned by the renderer
    virtual VideoCore::RasterizerInterface* Rasterizer() = 0;

    /// Finalize rendering the guest frame and draw into the presentation texture
    virtual void SwapBuffers() = 0;

    /// Draws the latest frame to the window waiting timeout_ms for a frame to arrive (Renderer
    /// specific implementation)
    virtual void TryPresent(int timeout_ms, bool is_secondary) = 0;

    /// Composes the screens into an offscreen buffer at the given layout and returns them as
    /// RGBA8, top-down. This is the frame dumper's "what the renderer produced" half, to be
    /// diffed against the frontend's readback of what was actually presented. Renderers that
    /// cannot compose off-screen return false and only the presented half is dumped.
    virtual bool ComposeOffscreen(const Layout::FramebufferLayout& layout, std::vector<u8>& out) {
        return false;
    }
    virtual void TryPresent(int timeout_ms) {
        TryPresent(timeout_ms, false);
    }

    /// Prepares for video dumping (e.g. create necessary buffers, etc)
    virtual void PrepareVideoDumping() {}

    /// Cleans up after video dumping is ended
    virtual void CleanupVideoDumping() {}

    /// This is called to notify the rendering backend of a surface change
    // if second == true then it is the second screen
    virtual void NotifySurfaceChanged(bool second) {}

    /// For a frontend that samples guest memory itself (the Vita): where this frame's
    /// framebuffer for `screen` (0 top, 1 bottom) is. Renderers whose output does not live in
    /// guest memory answer nothing, and the frontend draws nothing.
    virtual std::optional<GuestFramebuffer> GetGuestFramebuffer(int screen) {
        return std::nullopt;
    }

    /// Told by such a frontend, so a renderer can skip whatever conversion it would otherwise
    /// do for a presentation it is not asked for.
    virtual void SetFrontendPresentsGuestMemory(bool value) {}

    /// Returns the resolution scale factor relative to the native 3DS screen resolution
    u32 GetResolutionScaleFactor();

    /// Updates the framebuffer layout of the contained render window handle.
    void UpdateCurrentFramebufferLayout(bool is_portrait_mode = {});

    /// Ends the current frame
    void EndFrame();

    f32 GetCurrentFPS() const {
        return current_fps;
    }

    s32 GetCurrentFrame() const {
        return current_frame;
    }

    Frontend::EmuWindow& GetRenderWindow() {
        return render_window;
    }

    const Frontend::EmuWindow& GetRenderWindow() const {
        return render_window;
    }

    [[nodiscard]] RendererSettings& Settings() {
        return settings;
    }

    [[nodiscard]] const RendererSettings& Settings() const {
        return settings;
    }

    /// Returns true if a screenshot is being processed
    [[nodiscard]] bool IsScreenshotPending() const;

    /// Request a screenshot of the next frame
    void RequestScreenshot(void* data, std::function<void(bool)> callback,
                           const Layout::FramebufferLayout& layout);

    /// When presentation runs on a GPU thread, EndFrame must neither pace the frame (the
    /// emulation thread does that) nor poll window events (main-thread work).
    void SetThreadedPresentation(bool threaded) {
        threaded_presentation = threaded;
    }

    /// The GPU that owns this renderer. Renderers must reach the GPU through this rather than
    /// Core::System::GPU(): the system's pointer is already null while the GPU destructor is
    /// still draining the render thread, and a present that lands in that window would
    /// dereference it (the software renderer crashed at exit exactly this way).
    void SetOwnerGPU(GPU* gpu) {
        owner_gpu = gpu;
    }

protected:
    Core::System& system;
    GPU* owner_gpu{};
    RendererSettings settings;
    Frontend::EmuWindow& render_window;    /// Reference to the render window handle.
    Frontend::EmuWindow* secondary_window; /// Reference to the secondary render window handle.

protected:
    f32 current_fps = 0.0f; /// Current framerate, should be set by the renderer
    s32 current_frame = 0;  /// Current frame, should be set by the renderer
    bool threaded_presentation = false;
};

} // namespace VideoCore
