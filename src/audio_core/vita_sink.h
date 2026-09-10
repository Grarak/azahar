// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "audio_core/sink.h"

namespace AudioCore {

/**
 * The PS Vita's audio output (sceAudioOut), the shape DSVita gives it: a BGM port at 48 kHz
 * stereo fed 1024-frame buffers from a thread of its own, paced by sceAudioOutOutput, which
 * returns when the port has room for the next buffer. The thread lives on core 3 (the
 * miscellaneous core, ThreadRole::Other), so the DSP mix, the time stretcher (SoundTouch)
 * and the resampling from the 3DS's 32728 Hz run there and never on the emulation or the
 * render core.
 */
class VitaSink final : public Sink {
public:
    explicit VitaSink(std::string_view device_id);
    ~VitaSink() override;

    unsigned int GetNativeSampleRate() const override;
    void SetCallback(std::function<void(s16*, std::size_t)> cb) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

std::vector<std::string> ListVitaSinkDevices();

} // namespace AudioCore
