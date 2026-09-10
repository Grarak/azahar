// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <mutex>
#include <vector>
#include <string>
#include "core/frontend/emu_window.h"

struct SDL_Window;
struct SDL_Renderer;
struct SDL_Texture;

namespace Core {
class System;
}

/**
 * A window carrying the OpenGL context the GL renderer draws through.
 *
 * The context comes from a ladder: the native driver first, then Mesa's llvmpipe when the native
 * driver cannot reach GL 4.3 / GLES 3.2 (the Pi 5's V3D stops at 3.1).
 *
 * For devices with no usable GL at all there is a software-renderer mode (--software, or the
 * automatic fallback when the ladder fails): no GL context is created, and presentation blits
 * the composed frame onto a plain SDL window surface.
 */
class EmuWindow_SDL2 final : public Frontend::EmuWindow {
public:
    explicit EmuWindow_SDL2(Core::System& system, bool software = false);
    ~EmuWindow_SDL2() override;

    void PollEvents() override;

    // Frontend::GraphicsContext.
    void MakeCurrent() override;
    void DoneCurrent() override;
    void SwapBuffers() override;
    bool IsGLES() override {
        return is_gles;
    }
    std::unique_ptr<Frontend::GraphicsContext> CreateSharedContext() const override;

    /// Whether the context ladder produced a usable GL context.
    bool UsingOpenGL() const {
        return use_gl;
    }

    /// Switches this window to software-renderer presentation: no GL context, frames are
    /// blitted to a plain window surface. Also used as the fallback when the GL context
    /// ladder fails entirely.
    void UseSoftwareRendering();

    /// Presents the frame the renderer pushed into the mailbox.
    void Present();

    /// Whether the user has asked to quit.
    bool IsQuitRequested() const {
        return quit_requested;
    }

    /// Writes both emulated screens to `path` as a PPM image.
    bool SaveScreenshot(const std::string& path);

    /// Writes the next `count` presented frames to `prefix`NNNNN.ppm, from the render thread
    /// where the screen buffers are owned. Consecutive frames, so a one-frame artifact that a
    /// periodic screenshot would step over is visible.
    /// Writes the next `count` presented frames as PPMs under `prefix`: the window readback
    /// and the renderer's own composition, or only the window when `window_only`. Recording
    /// footage wants only the window, and the composition costs an offscreen pass and a
    /// second file per frame, which roughly halves the rate a capture runs at.
    void DumpFrames(const std::string& prefix, int count, bool window_only = false);

private:
    /// The composed frame handed from the render thread to the thread that owns the window.
    std::mutex present_mutex;
    std::vector<u8> present_canvas;
    bool present_ready = false;

public:

    /// Touches the bottom screen at the given framebuffer coordinates, for the debug input port.
    /// Returns false if the point is not on the touch screen.
    bool TouchAt(int x, int y);

    /// Lifts any touch.
    void ReleaseTouch();

private:
    void UpdateLayout();

    /// Works down the context ladder. True if a usable GL context is current afterwards.
    bool InitializeGL();
    /// One rung: a window plus context of the requested flavour, glad-loaded and version-checked.
    bool TryCreateGLContext(bool want_gles, int gles_minor = 2);

    void HandleKey(int sdl_scancode, bool pressed);
    void HandleMouse(int x, int y, bool pressed);

    Core::System& system;
    bool quit_requested{};

    SDL_Window* window{};
    /// Software presentation goes through a streaming texture rather than the window surface:
    /// the surface path scales on the CPU and uploads through the driver's CPU tiling code,
    /// which measured ~72% of the render thread and starved the rasterizer workers.
    SDL_Renderer* sw_renderer{};
    SDL_Texture* sw_texture{};
    void* gl_context{}; // SDL_GLContext
    bool use_gl{};
    bool software_mode{};
    bool is_gles{};

    int window_width{};
    int window_height{};
};
