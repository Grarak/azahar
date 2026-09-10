// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>

namespace Frontend {
class EmuWindow;
}

namespace Core {
class System;
}

namespace Pica {
class PicaCore;
}

namespace VideoCore {

class RendererBase;

/// Writes every cached surface to a directory as PPM (colour or depth) plus PGM (alpha or
/// stencil) and an index.txt, for looking at what the renderer is holding. Registered by
/// whichever rasterizer is live; the frontends' debug commands call it on the render thread.
extern void (*surface_dump_hook)(void* user, const char* dir);
extern void* surface_dump_user;

std::unique_ptr<RendererBase> CreateRenderer(Frontend::EmuWindow& emu_window,
                                             Frontend::EmuWindow* secondary_window,
                                             Pica::PicaCore& pica, Core::System& system);

} // namespace VideoCore
