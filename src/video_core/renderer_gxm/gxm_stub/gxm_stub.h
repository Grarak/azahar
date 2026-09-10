// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * A libgxm with no GPU behind it, for profiling the GXM renderer's CPU work off the console
 * (ENABLE_GXM_STUB). Every sceGxm* call the renderer makes is here: state
 * setters and draws do nothing, scenes complete the moment they end (the notification is
 * written in sceGxmEndScene), programs are real GXP blobs from the USSE emitters and their
 * parameter tables are parsed for real, memory comes from malloc. The render thread runs
 * every line it runs on the Vita except the ones inside libgxm, so a profiler on the host
 * sees the state translation, the caches, the uniform packing and the emitters with symbols.
 * Nothing is presented.
 */
namespace GxmStub {

/// Installs a GxmDevice backed by this stub (GxmRenderer::SetDevice). Call once before the
/// renderer is created.
void InstallDevice();

} // namespace GxmStub
