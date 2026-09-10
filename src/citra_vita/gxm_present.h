// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string>
#include <psp2/gxm.h>
#include "common/common_types.h"

namespace VitaFrontend::GxmPresent {

inline constexpr u32 DisplayWidth = 960;
inline constexpr u32 DisplayHeight = 544;

/// Brings up the whole GXM stack: library, context, render target, display surfaces, shader
/// patcher, and the fixed shaders. Returns false, with the reason logged, when the console
/// cannot present.
bool Initialize();
void Shutdown();
[[nodiscard]] bool IsInitialized();

/// Makes a block of emulator memory visible to the GPU, so textures can point straight into
/// it. Base and size must be 4 KiB aligned (guest FCRAM/VRAM already are). Re-registering a
/// mapped base is a no-op; the registry exists so UnmapAll can run before the memory dies.
void MapGuestMemory(void* base, u32 size);
/// Unmaps everything MapGuestMemory registered, after waiting for the GPU. Must run before
/// the emulator frees the memory - unmapping later than the free is a kernel fault.
void UnmapAllGuestMemory();

/// One guest screen for DrawGuestScreen: a pointer into guest memory as the 3DS scans it out
/// (240-pixel columns), or a solid LCD fill.
struct GuestScreen {
    const u8* pixels;
    const SceGxmTexture* texture; ///< when set, sampled instead of pixels (same layout)
    u32 tex_x, tex_y, tex_width, tex_height; ///< the framebuffer's part of `texture`, texels
    u32 stride_bytes; ///< bytes per 240-pixel column
    u32 height;       ///< 400 or 320
    int pica_format;  ///< Pica::PixelFormat as int, to keep video_core out of this header
    bool fill_enabled;
    u8 fill_r, fill_g, fill_b;
};

/// Frame lifecycle. All of these run on whichever single thread presents (the render thread
/// in-game, the main thread in the menus) - never two at once, which is the same discipline
/// GXM's immediate context demands.
void BeginFrame(u8 clear_r, u8 clear_g, u8 clear_b);
void DrawGuestScreen(const GuestScreen& screen, float x, float y, float w, float h);
void DrawSolidRect(float x, float y, float w, float h, u8 r, u8 g, u8 b, u8 a);
void EndFrame();

// --- for the imgui backend, which speaks GXM itself ---

[[nodiscard]] SceGxmContext* Context();
[[nodiscard]] SceGxmShaderPatcher* Patcher();

/// GPU-visible memory that lives until Shutdown (uncached LPDDR). For rings and the font
/// atlas - not a general allocator.
[[nodiscard]] void* AllocMapped(u32 size);
/// The same lifetime, in ordinary cached memory. Both kinds are coherent with the GPU here;
/// this one puts the cost on the GPU's reads rather than on the CPU's stores.
[[nodiscard]] void* AllocMappedCached(u32 size);
/// CDRAM for render targets the GPU alone touches; lives until FreeBlock or Shutdown.
[[nodiscard]] void* AllocCdram(u32 size);
/// Releases a block from AllocMapped or AllocCdram; the GPU must be done with it.
void FreeBlock(void* block);

/// Draws any GXM texture unrotated into a display rectangle (the renderer's debug overlay).
void DrawTexture(const SceGxmTexture* texture, float x, float y, float w, float h);

/// True when the Razor modules loaded at Initialize, so the host tool can see this process.
[[nodiscard]] bool HasRazor();
/// Razor GPU Live's last second, as lines for the on-screen HUD (gxm flag gpuhud); empty
/// until the first second is in.
[[nodiscard]] const std::string& GpuLiveText();

/// The next EndFrame writes its picture to ux0:data/azahar/frame_NN.ppm (the pause menu's
/// "Dump frame"; the dumpframe flag does the same every five seconds). DumpPending is true
/// until that frame is written, so the caller can leave the overlay out of it.
void DumpNextFrame();
[[nodiscard]] bool DumpPending();

/// Per-frame transient GPU-visible memory (vertices, indices). Valid for the frame being
/// built; recycled two frames later, after the GPU is provably done with it.
[[nodiscard]] void* FrameAlloc(u32 size, u32 align = 16);

} // namespace VitaFrontend::GxmPresent
