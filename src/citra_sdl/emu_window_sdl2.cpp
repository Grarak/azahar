// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <mutex>
#include "citra_sdl/emu_window_sdl2.h"
#include "common/pipeline_stats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <SDL.h>
#ifdef ENABLE_OPENGL
#include <glad/glad.h>
#endif
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/framebuffer_layout.h"
#include "input_common/keyboard.h"
#include "input_common/main.h"
#include "video_core/gpu.h"
#ifdef ENABLE_SOFTWARE_RENDERER
#include "video_core/renderer_software/renderer_software.h"
#endif
#include "video_core/renderer_base.h"

namespace {

/// 400x240 over 320x240 at 2x, which fits comfortably on any display this runs on.
constexpr int DefaultWindowWidth = 800;
constexpr int DefaultWindowHeight = 960;

/// Frame dumps render at the 3DS's native stacked size, so a screenshot comes out at native
/// resolution whatever the window is doing.
constexpr int ScreenshotWidth = 400;
constexpr int ScreenshotHeight = 480;

#ifdef ENABLE_OPENGL
/**
 * A context sharing the main window's GL objects, for the renderer's worker threads.
 *
 * SDL wants context creation on the thread that owns the window, so these are made on demand from
 * CreateSharedContext (called on the main thread) and only made current elsewhere.
 */
class SharedContext_SDL2 : public Frontend::GraphicsContext {
public:
    SharedContext_SDL2(SDL_Window* window_, bool is_gles_) : window{window_}, is_gles{is_gles_} {
        // Creating a context makes it current, displacing whatever the caller had; put that back,
        // since the caller decides where the new context runs.
        SDL_Window* previous_window = SDL_GL_GetCurrentWindow();
        SDL_GLContext previous = SDL_GL_GetCurrentContext();
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
        context = SDL_GL_CreateContext(window);
        SDL_GL_MakeCurrent(previous_window, previous);
    }

    ~SharedContext_SDL2() override {
        if (context != nullptr) {
            SDL_GL_DeleteContext(context);
        }
    }

    bool IsGLES() override {
        return is_gles;
    }

    void MakeCurrent() override {
        SDL_GL_MakeCurrent(window, context);
    }

    void DoneCurrent() override {
        SDL_GL_MakeCurrent(window, nullptr);
    }

private:
    SDL_Window* window;
    SDL_GLContext context{};
    bool is_gles;
};
#endif

} // Anonymous namespace

EmuWindow_SDL2::EmuWindow_SDL2(Core::System& system_, bool software_) : system{system_} {
    window_width = DefaultWindowWidth;
    window_height = DefaultWindowHeight;

    // Let the shader disk cache precompile on the main context rather than a shared one -
    // the same road Qt takes on wayland.
    strict_context_required = true;

    if (software_) {
        UseSoftwareRendering();
    } else {
        use_gl = InitializeGL();
    }

    UpdateLayout();
}

void EmuWindow_SDL2::UseSoftwareRendering() {
    software_mode = true;

    if (SDL_WasInit(SDL_INIT_VIDEO) == 0) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            LOG_ERROR(Frontend, "SDL_Init for software rendering failed: {}", SDL_GetError());
            return;
        }
    }

    // A plain window without a GL flag; the frame is blitted onto its surface at present time.
    if (window == nullptr) {
        window = SDL_CreateWindow("Azahar (software renderer)", SDL_WINDOWPOS_UNDEFINED,
                                  SDL_WINDOWPOS_UNDEFINED, window_width, window_height,
                                  SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if (window == nullptr) {
            LOG_ERROR(Frontend, "SDL_CreateWindow for software rendering failed: {}",
                      SDL_GetError());
        }
    }
}

EmuWindow_SDL2::~EmuWindow_SDL2() {
#ifdef ENABLE_OPENGL
    if (gl_context != nullptr) {
        SDL_GL_DeleteContext(static_cast<SDL_GLContext>(gl_context));
    }
#endif
    if (window != nullptr) {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
}

#ifdef ENABLE_OPENGL

// GLES 3.1 carries the renderer when these extensions fill the 3.2 gaps. Texture buffers are
// not on the list: the LUTs read from plain 2D textures on every GLES context.
static bool HasRequiredGLES31Extensions() {
    const auto has = [](const char* name) {
        return SDL_GL_ExtensionSupported(name) == SDL_TRUE;
    };
    return has("GL_EXT_copy_image") && has("GL_EXT_draw_elements_base_vertex") &&
           has("GL_OES_draw_buffers_indexed") && has("GL_EXT_texture_border_clamp");
}

// The floor for GLES 3.0: copy_image and base_vertex must exist (OES variants require only
// ES 3.0). Shadow rendering is skipped on such contexts; everything else has fallbacks.
static bool HasRequiredGLES30Extensions() {
    const auto has = [](const char* name) {
        return SDL_GL_ExtensionSupported(name) == SDL_TRUE;
    };
    return (has("GL_EXT_copy_image") || has("GL_OES_copy_image")) &&
           (has("GL_EXT_draw_elements_base_vertex") || has("GL_OES_draw_elements_base_vertex"));
}

/// GLES 3.1 exposes several 3.2-core entry points only under extension suffixes, which this
/// glad build does not load. Resolve them through SDL and alias onto the core names the
/// renderer calls.
static void AliasGLES31EntryPoints() {
    const auto alias = [](auto& core_ptr, const char* suffixed_name) {
        if (core_ptr == nullptr) {
            core_ptr = reinterpret_cast<std::remove_reference_t<decltype(core_ptr)>>(
                SDL_GL_GetProcAddress(suffixed_name));
        }
    };
    alias(glad_glCopyImageSubData, "glCopyImageSubDataEXT");
    alias(glad_glCopyImageSubData, "glCopyImageSubDataOES");
    alias(glad_glDrawRangeElementsBaseVertex, "glDrawRangeElementsBaseVertexEXT");
    alias(glad_glDrawRangeElementsBaseVertex, "glDrawRangeElementsBaseVertexOES");
    alias(glad_glDrawElementsBaseVertex, "glDrawElementsBaseVertexEXT");
    alias(glad_glDrawElementsBaseVertex, "glDrawElementsBaseVertexOES");
    alias(glad_glEnablei, "glEnableiOES");
    alias(glad_glDisablei, "glDisableiOES");
    alias(glad_glColorMaski, "glColorMaskiOES");
    alias(glad_glBlendFuncSeparatei, "glBlendFuncSeparateiOES");
    alias(glad_glBlendEquationSeparatei, "glBlendEquationSeparateiOES");
    alias(glad_glTexParameterIiv, "glTexParameterIivEXT");
    alias(glad_glDebugMessageCallback, "glDebugMessageCallbackKHR");
    alias(glad_glFramebufferTexture, "glFramebufferTextureEXT");
    alias(glad_glTexBuffer, "glTexBufferEXT");
    alias(glad_glObjectLabel, "glObjectLabelKHR");
    alias(glad_glPushDebugGroup, "glPushDebugGroupKHR");
    alias(glad_glPopDebugGroup, "glPopDebugGroupKHR");
    alias(glad_glDebugMessageInsert, "glDebugMessageInsertKHR");
    alias(glad_glDebugMessageControl, "glDebugMessageControlKHR");
    alias(glad_glPatchParameteri, "glPatchParameteriEXT");
    alias(glad_glMinSampleShading, "glMinSampleShadingOES");
    alias(glad_glBlendBarrier, "glBlendBarrierKHR");
}

bool EmuWindow_SDL2::TryCreateGLContext(bool want_gles, int gles_minor) {
    SDL_GL_ResetAttributes();
    if (want_gles) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, gles_minor);
    } else {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    }
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    window = SDL_CreateWindow("Azahar", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              window_width, window_height,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (window == nullptr) {
        LOG_INFO(Frontend, "GL window ({}) failed: {}",
                 want_gles ? (gles_minor == 2   ? "GLES 3.2"
                              : gles_minor == 1 ? "GLES 3.1"
                                                : "GLES 3.0")
                           : "GL 4.3",
                 SDL_GetError());
        return false;
    }

    gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        LOG_INFO(Frontend, "GL context ({}) failed: {}",
                 want_gles ? (gles_minor == 2   ? "GLES 3.2"
                              : gles_minor == 1 ? "GLES 3.1"
                                                : "GLES 3.0")
                           : "GL 4.3",
                 SDL_GetError());
        SDL_DestroyWindow(window);
        window = nullptr;
        return false;
    }

    SDL_GL_MakeCurrent(window, static_cast<SDL_GLContext>(gl_context));

    const auto load = want_gles ? gladLoadGLES2Loader : gladLoadGLLoader;
    if (!load(static_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
        LOG_INFO(Frontend, "glad could not load the {} context", want_gles ? "GLES" : "GL");
    } else if ((want_gles && (GLAD_GL_ES_VERSION_3_2 ||
                              (gles_minor == 1 && GLAD_GL_ES_VERSION_3_1 &&
                               HasRequiredGLES31Extensions()) ||
                              (gles_minor == 0 && GLAD_GL_ES_VERSION_3_0 &&
                               HasRequiredGLES30Extensions()))) ||
               (!want_gles && GLAD_GL_VERSION_4_3)) {
        if (want_gles && !GLAD_GL_ES_VERSION_3_2) {
            AliasGLES31EntryPoints();
        }
#ifdef CITRA_TRACE_PROBES
        if (std::getenv("AZAHAR_GL_DEBUG") != nullptr && glDebugMessageCallback != nullptr) {
            glEnable(GL_DEBUG_OUTPUT);
            glDebugMessageCallback(
                [](GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* message,
                   const void*) {
                    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION) {
                        LOG_ERROR(Frontend, "GL[{}]: {}", type, message);
                    }
                },
                nullptr);
        }
#endif // CITRA_TRACE_PROBES
        is_gles = want_gles;
        const auto* gl_version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const auto* gl_renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const auto* gl_vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        LOG_INFO(Frontend, "Using {} context: {} on {}",
                 want_gles ? (GLAD_GL_ES_VERSION_3_2   ? "GLES 3.2"
                              : GLAD_GL_ES_VERSION_3_1 ? "GLES 3.1+ext"
                                                       : "GLES 3.0+ext")
                           : "GL 4.3",
                 gl_version, gl_renderer);
        // On stdout, not just the log: which driver got picked decides everything about GL
        // performance, and a remote X display silently swaps the GPU for a software rasterizer.
        std::printf("[azahar] GL context: %s | %s | %s\n", gl_version ? gl_version : "?",
                    gl_renderer ? gl_renderer : "?", gl_vendor ? gl_vendor : "?");
        if (gl_renderer != nullptr &&
            (std::strstr(gl_renderer, "llvmpipe") != nullptr ||
             std::strstr(gl_renderer, "softpipe") != nullptr ||
             std::strstr(gl_renderer, "swrast") != nullptr ||
             std::strstr(gl_renderer, "Software Rasterizer") != nullptr)) {
            LOG_WARNING(Frontend, "GL is running on a software rasterizer ({})", gl_renderer);
            std::printf("[azahar] WARNING: this is a software GL rasterizer, not the GPU - "
                        "typical with a forwarded/remote X display. Expect very low "
                        "performance and constant shader recompilation; the GPU's shader "
                        "cache does not apply to this driver.\n");
        }
        return true;
    } else {
        LOG_INFO(Frontend, "Context created but only reaches {}, not enough for the GL renderer",
                 reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    }

    SDL_GL_DeleteContext(static_cast<SDL_GLContext>(gl_context));
    gl_context = nullptr;
    SDL_DestroyWindow(window);
    window = nullptr;
    return false;
}

bool EmuWindow_SDL2::InitializeGL() {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        LOG_ERROR(Frontend, "SDL_Init for GL failed: {}", SDL_GetError());
        return false;
    }

    // The ladder: the native driver in either flavour, then llvmpipe. The driver override is only
    // read when the GL driver is first loaded, so switching to llvmpipe needs the video subsystem
    // torn down and brought back up. kms_swrast rather than LIBGL_ALWAYS_SOFTWARE: the latter is
    // refused when EGL has picked a hardware device, and Mesa's pure-swrast screen crashes under a
    // surfaceless display; kms_swrast is llvmpipe on a render node and works from either.
    for (int use_llvmpipe = 0; use_llvmpipe <= 1; use_llvmpipe++) {
        if (use_llvmpipe) {
            LOG_INFO(Frontend, "Native GL driver is not sufficient; retrying with llvmpipe");
            setenv("MESA_LOADER_DRIVER_OVERRIDE", "kms_swrast", 1);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
            if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
                LOG_ERROR(Frontend, "SDL video re-init for llvmpipe failed: {}", SDL_GetError());
                break;
            }
        }
        // GLES 3.2 first — it is the 3DS-shaped API and the renderer's primary target on small
        // devices; desktop GL second for drivers that only get there.
        if (TryCreateGLContext(true, 2) || TryCreateGLContext(true, 1) ||
            TryCreateGLContext(true, 0) || TryCreateGLContext(false)) {
            return true;
        }
    }

    LOG_CRITICAL(Frontend, "No usable OpenGL context on this machine");
    return false;
}

#else

bool EmuWindow_SDL2::TryCreateGLContext(bool, int) {
    return false;
}

bool EmuWindow_SDL2::InitializeGL() {
    LOG_CRITICAL(Frontend, "This build has no OpenGL renderer (ENABLE_OPENGL was off)");
    return false;
}

#endif

void EmuWindow_SDL2::MakeCurrent() {
#ifdef ENABLE_OPENGL
    if (use_gl) {
        SDL_GL_MakeCurrent(window, static_cast<SDL_GLContext>(gl_context));
    }
#endif
}

void EmuWindow_SDL2::DoneCurrent() {
#ifdef ENABLE_OPENGL
    if (use_gl) {
        SDL_GL_MakeCurrent(window, nullptr);
    }
#endif
}

#ifdef ENABLE_SOFTWARE_RENDERER

// Composes both emulated screens into a 400x480 RGBA canvas: top screen in rows 0..239,
// bottom screen centered in rows 240..479. The software renderer already stores each screen
// upright (its LoadFBToScreenInfo transposes), with the row length in ScreenInfo::height.
static bool ComposeSoftwareFrame(Core::System& system, std::vector<u8>& canvas) {
    constexpr u32 canvas_w = 400;
    constexpr u32 canvas_h = 480;
    if (Settings::GetWorkingGraphicsAPI() != Settings::GraphicsAPI::Software) {
        // Nothing to compose: under the stub-GXM profile this cleared 768 KB per present on
        // the render thread for a frame that was then thrown away.
        return false;
    }
    canvas.assign(canvas_w * canvas_h * 4, 0);
    const auto* renderer =
        static_cast<const SwRenderer::RendererSoftware*>(&system.GPU().Renderer());

    const auto blit_screen = [&](VideoCore::ScreenId id, u32 dst_x, u32 dst_y) {
        const auto& info = renderer->Screen(id);
        const u32 src_w = info.height; // upright width (stored transposed)
        const u32 src_h = info.width;  // upright height
        if (info.pixels.size() < static_cast<std::size_t>(src_w) * src_h * 4) {
            return;
        }
        for (u32 y = 0; y < src_h && dst_y + y < canvas_h; y++) {
            const u32 copy_w = std::min(src_w, canvas_w - dst_x);
            std::memcpy(canvas.data() + ((dst_y + y) * canvas_w + dst_x) * 4,
                        info.pixels.data() + static_cast<std::size_t>(y) * src_w * 4,
                        static_cast<std::size_t>(copy_w) * 4);
        }
    };

    blit_screen(VideoCore::ScreenId::TopLeft, 0, 0);
    blit_screen(VideoCore::ScreenId::Bottom, 40, 240);
    return true;
}

#endif // ENABLE_SOFTWARE_RENDERER

namespace {
/// Frame dumping state; written by the render thread inside SwapBuffers.
std::mutex dump_mutex;
std::string dump_prefix;
int dump_remaining = 0;
int dump_index = 0;
bool dump_window_only = false;

void WritePpm(const std::string& path, const std::vector<u8>& canvas, u32 w, u32 h) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return;
    }
    std::fprintf(file, "P6\n%u %u\n255\n", w, h);
    std::vector<u8> row(static_cast<std::size_t>(w) * 3);
    for (u32 y = 0; y < h; y++) {
        const u8* source = canvas.data() + static_cast<std::size_t>(y) * w * 4;
        for (u32 x = 0; x < w; x++) {
            row[x * 3 + 0] = source[x * 4 + 0];
            row[x * 3 + 1] = source[x * 4 + 1];
            row[x * 3 + 2] = source[x * 4 + 2];
        }
        std::fwrite(row.data(), 1, row.size(), file);
    }
    std::fclose(file);
}
} // namespace

void EmuWindow_SDL2::DumpFrames(const std::string& prefix, int count, bool window_only) {
    std::scoped_lock lock{dump_mutex};
    dump_prefix = prefix;
    dump_remaining = count;
    dump_index = 0;
    dump_window_only = window_only;
}

void EmuWindow_SDL2::SwapBuffers() {
#ifdef ENABLE_OPENGL
    if (use_gl) {
        // Dump before the swap, while the finished frame is still the back buffer. On the render
        // thread, where the context is current - the screenshot path cannot serve this because
        // it pumps the run loop, and the debug port is serviced from the emulation thread.
        {
            std::scoped_lock lock{dump_mutex};
            if (dump_remaining > 0) {
                // The renderer's own composition, at the 3DS's native stacked size, so it can
                // be diffed against the window readback below: anything the two disagree about
                // was introduced by presenting rather than by rendering. Same pair, and the
                // same names, as the software path writes.
                if (!dump_window_only && system.IsPoweredOn()) {
                    std::vector<u8> canvas;
                    const auto canvas_layout = Layout::DefaultFrameLayout(
                        ScreenshotWidth, ScreenshotHeight, false, false);
                    if (system.GPU().Renderer().ComposeOffscreen(canvas_layout, canvas)) {
                        char pre[64];
                        std::snprintf(pre, sizeof(pre), "%05d_canvas.ppm", dump_index);
                        WritePpm(dump_prefix + pre, canvas, canvas_layout.width,
                                 canvas_layout.height);
                    }
                }
                const u32 w = static_cast<u32>(window_width);
                const u32 h = static_cast<u32>(window_height);
                std::vector<u8> rgba(static_cast<std::size_t>(w) * h * 4);
                glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
                // glReadPixels returns bottom-up; PPM is top-down.
                std::vector<u8> flipped(rgba.size());
                for (u32 y = 0; y < h; y++) {
                    std::memcpy(flipped.data() + static_cast<std::size_t>(y) * w * 4,
                                rgba.data() + static_cast<std::size_t>(h - 1 - y) * w * 4,
                                static_cast<std::size_t>(w) * 4);
                }
                char name[64];
                std::snprintf(name, sizeof(name), "%05d_window.ppm", dump_index++);
                WritePpm(dump_prefix + name, flipped, w, h);
                dump_remaining--;
            }
        }
        const auto swap_start = std::chrono::steady_clock::now();
        SDL_GL_SwapWindow(window);
        Common::PipelineStats::present_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                  swap_start)
                .count(),
            std::memory_order_relaxed);
        Common::PipelineStats::presents.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#endif
#ifdef ENABLE_SOFTWARE_RENDERER
    // Software presentation, on the render thread: it owns the renderer's screen buffers and,
    // like the GL path, it owns the presentation target too. The main loop only pumps events
    // and never touches the window surface, so there is a single owner rather than a handoff.
    if (software_mode && window != nullptr && system.IsPoweredOn()) {
        std::scoped_lock lock{present_mutex};
        if (!ComposeSoftwareFrame(system, present_canvas)) {
            return;
        }
        if (sw_renderer == nullptr) {
            // Created here rather than at construction because it must belong to the thread that
            // presents, and never mixed with SDL_GetWindowSurface on the same window.
            sw_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
            if (sw_renderer == nullptr) {
                sw_renderer = SDL_CreateRenderer(window, -1, 0);
            }
            if (sw_renderer != nullptr) {
                sw_texture = SDL_CreateTexture(sw_renderer, SDL_PIXELFORMAT_RGBA32,
                                               SDL_TEXTUREACCESS_STREAMING, 400, 480);
                SDL_SetTextureScaleMode(sw_texture, SDL_ScaleModeNearest);
            }
            if (sw_renderer == nullptr || sw_texture == nullptr) {
                LOG_ERROR(Frontend, "Could not create the software presentation renderer");
                return;
            }
            // Same reasoning as the GL context line: which backend SDL picked decides what a
            // present costs, and a remote X display swaps the GPU path for a software one.
            SDL_RendererInfo info{};
            if (SDL_GetRendererInfo(sw_renderer, &info) == 0) {
                LOG_INFO(Frontend, "Software frames present through SDL '{}' ({}accelerated)",
                         info.name, (info.flags & SDL_RENDERER_ACCELERATED) ? "" : "not ");
                std::printf("[azahar] present backend: SDL '%s'%s\n", info.name,
                            (info.flags & SDL_RENDERER_ACCELERATED) ? ""
                                                                    : " (not GPU-accelerated)");
            }
        }
        {
            const auto present_start = std::chrono::steady_clock::now();
            SDL_UpdateTexture(sw_texture, nullptr, present_canvas.data(), 400 * 4);
            SDL_RenderClear(sw_renderer);
            SDL_RenderCopy(sw_renderer, sw_texture, nullptr, nullptr);
            // Dump what was actually put on the window, not a fresh compose of the renderer's
            // buffers: an artifact introduced by the presentation itself is invisible to a
            // capture that does not go through it, which is why every clean capture so far
            // proved less than it appeared to.
            std::scoped_lock dump_lock{dump_mutex};
            if (dump_remaining > 0 && !dump_window_only) {
                // The composed canvas as it went in, so it can be diffed against what came out
                // on the window: anything the two disagree about was introduced by presenting.
                char pre[64];
                std::snprintf(pre, sizeof(pre), "%05d_canvas.ppm", dump_index);
                WritePpm(dump_prefix + pre, present_canvas, 400, 480);
            }
            if (dump_remaining > 0) {
                // Read back what was drawn for this frame, so a dump still shows what the
                // presentation produced rather than a fresh compose of the same buffers.
                int rw = 0, rh = 0;
                SDL_GetRendererOutputSize(sw_renderer, &rw, &rh);
                const u32 w = static_cast<u32>(rw);
                const u32 h = static_cast<u32>(rh);
                std::vector<u8> rgba(static_cast<std::size_t>(w) * h * 4);
                SDL_RenderReadPixels(sw_renderer, nullptr, SDL_PIXELFORMAT_RGBA32, rgba.data(),
                                     static_cast<int>(w) * 4);
                char name[64];
                std::snprintf(name, sizeof(name), "%05d_window.ppm", dump_index++);
                WritePpm(dump_prefix + name, rgba, w, h);
                dump_remaining--;
            }
            SDL_RenderPresent(sw_renderer);
            Common::PipelineStats::present_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - present_start)
                    .count(),
                std::memory_order_relaxed);
            Common::PipelineStats::presents.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif
}

std::unique_ptr<Frontend::GraphicsContext> EmuWindow_SDL2::CreateSharedContext() const {
#ifdef ENABLE_OPENGL
    if (use_gl) {
        return std::make_unique<SharedContext_SDL2>(window, is_gles);
    }
#endif
    return nullptr;
}

void EmuWindow_SDL2::UpdateLayout() {
    NotifyFramebufferLayoutChanged(Layout::DefaultFrameLayout(
        static_cast<u32>(window_width), static_cast<u32>(window_height),
        Settings::values.swap_screen.GetValue(), Settings::values.upright_screen.GetValue()));
}

void EmuWindow_SDL2::PollEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_QUIT:
            quit_requested = true;
            break;
        case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                window_width = event.window.data1;
                window_height = event.window.data2;
                UpdateLayout();
            } else if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                quit_requested = true;
            }
            break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) {
                quit_requested = true;
                break;
            }
            HandleKey(event.key.keysym.scancode, event.type == SDL_KEYDOWN);
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                HandleMouse(event.button.x, event.button.y, event.type == SDL_MOUSEBUTTONDOWN);
            }
            break;
        case SDL_MOUSEMOTION:
            if ((event.motion.state & SDL_BUTTON_LMASK) != 0) {
                TouchMoved(static_cast<unsigned>(event.motion.x),
                           static_cast<unsigned>(event.motion.y));
            }
            break;
        default:
            break;
        }
    }
}

void EmuWindow_SDL2::HandleKey(int sdl_scancode, bool pressed) {
    auto* keyboard = InputCommon::GetKeyboard();
    if (keyboard == nullptr) {
        return;
    }
    if (pressed) {
        keyboard->PressKey(sdl_scancode);
    } else {
        keyboard->ReleaseKey(sdl_scancode);
    }
}

void EmuWindow_SDL2::HandleMouse(int x, int y, bool pressed) {
    if (pressed) {
        TouchPressed(static_cast<unsigned>(x), static_cast<unsigned>(y));
    } else {
        TouchReleased();
    }
}

bool EmuWindow_SDL2::TouchAt(int x, int y) {
    if (x < 0 || y < 0) {
        return false;
    }
    return TouchPressed(static_cast<unsigned>(x), static_cast<unsigned>(y));
}

void EmuWindow_SDL2::ReleaseTouch() {
    TouchReleased();
}

void EmuWindow_SDL2::Present() {
    // The render thread owns the presentation target under both renderers and presents after
    // each SwapBuffers op; there is nothing left for the main loop to do here.
}

bool EmuWindow_SDL2::SaveScreenshot(const std::string& path) {
#ifdef ENABLE_SOFTWARE_RENDERER
    if (software_mode) {
        if (!system.IsPoweredOn()) {
            return false;
        }
        // The renderer's screen buffers belong to the render thread, which rewrites them (and
        // reallocates them - ScreenInfo::pixels is resized per frame) inside every present.
        // Composing them from this thread reads a vector mid-reallocation: the frame comes out
        // sheared, with one screen's rows carrying the other screen's pixels. Compose where the
        // buffers live instead.
        std::vector<u8> canvas;
        bool composed = false;
        system.GPU().RunOnRenderThread(
            [&] { composed = ComposeSoftwareFrame(system, canvas); });
        if (!composed) {
            return false;
        }
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            return false;
        }
        std::fprintf(file, "P6\n400 480\n255\n");
        std::vector<u8> row(400 * 3);
        for (u32 y = 0; y < 480; y++) {
            const u8* source = canvas.data() + static_cast<std::size_t>(y) * 400 * 4;
            for (u32 x = 0; x < 400; x++) {
                row[x * 3 + 0] = source[x * 4 + 0];
                row[x * 3 + 1] = source[x * 4 + 1];
                row[x * 3 + 2] = source[x * 4 + 2];
            }
            std::fwrite(row.data(), 1, row.size(), file);
        }
        std::fclose(file);
        return true;
    }
#endif
#ifdef ENABLE_OPENGL
    if (!use_gl || !system.IsPoweredOn()) {
        return false;
    }

    const auto layout = Layout::DefaultFrameLayout(ScreenshotWidth, ScreenshotHeight,
                                                   Settings::values.swap_screen.GetValue(),
                                                   Settings::values.upright_screen.GetValue());
    std::vector<u32> pixels(static_cast<std::size_t>(layout.width) * layout.height);

    // The renderer fills the buffer during a later SwapBuffers — on the render thread — so run
    // the emulation until the callback fires. The guest keeps producing frames.
    std::atomic<bool> done{false};
    std::atomic<bool> ok{false};
    system.GPU().Renderer().RequestScreenshot(
        pixels.data(),
        [&](bool success) {
            ok.store(success);
            done.store(true);
        },
        layout);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        if (system.RunLoop() != Core::System::ResultStatus::Success) {
            break;
        }
    }
    if (!done.load() || !ok.load()) {
        LOG_ERROR(Frontend, "GL screenshot did not complete");
        return false;
    }

    // glReadPixels delivered BGRA rows bottom-up; the PPM wants RGB top-down.
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }
    std::fprintf(file, "P6\n%u %u\n255\n", layout.width, layout.height);
    std::vector<u8> row(static_cast<std::size_t>(layout.width) * 3);
    for (u32 y = 0; y < layout.height; y++) {
        const u32* source =
            pixels.data() + static_cast<std::size_t>(layout.height - 1 - y) * layout.width;
        for (u32 x = 0; x < layout.width; x++) {
            const u32 bgra = source[x];
            row[x * 3 + 0] = static_cast<u8>((bgra >> 16) & 0xFF);
            row[x * 3 + 1] = static_cast<u8>((bgra >> 8) & 0xFF);
            row[x * 3 + 2] = static_cast<u8>(bgra & 0xFF);
        }
        std::fwrite(row.data(), 1, row.size(), file);
    }
    std::fclose(file);
    return true;
#else
    return false;
#endif
}
