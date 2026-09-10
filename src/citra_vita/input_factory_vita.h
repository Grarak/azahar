// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

namespace VitaFrontend {

class EmuWindowVita;

/**
 * Registers the "vita" input engine and points the current input profile at it.
 *
 * The engine reads the pad the frontend last sampled rather than keeping any state of its own:
 * HID asks each device for its status from a core timing event, so a device that answers from
 * the latest sample is both simpler and more current than one fed press and release edges.
 */
void InitInput(EmuWindowVita& window);
void ShutdownInput();

} // namespace VitaFrontend
