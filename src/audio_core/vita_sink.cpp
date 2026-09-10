// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <psp2/audioout.h>
#include <psp2/kernel/threadmgr.h>
#include "audio_core/audio_types.h"
#include "audio_core/vita_sink.h"
#include "common/logging/log.h"
#include "common/named_thread.h"
#include "common/thread.h"

namespace AudioCore {

struct VitaSink::Impl {
    static constexpr int OutputRate = 48000;
    /// Frames per sceAudioOutOutput: 21 ms at 48 kHz, DSVita's size. The port double-buffers,
    /// so the call blocks while the previous buffer is still playing.
    static constexpr std::size_t OutputFrames = 1024;
    /// Input frames one output buffer consumes, rounded up, plus the two the interpolation
    /// looks ahead by.
    static constexpr std::size_t InputFrames =
        (OutputFrames * native_sample_rate + OutputRate - 1) / OutputRate + 2;

    int port = -1;
    std::function<void(s16*, std::size_t)> cb;
    std::mutex cb_mutex;
    Common::NamedThread thread;

    // The resampler's state: `input` holds frames pulled from the callback, `pos` the
    // fractional read position in it (frames).
    std::vector<s16> input;
    double pos = 0.0;
    std::array<s16, OutputFrames * 2> output{};

    void Loop(std::stop_token stop);
    void Fill();
};

VitaSink::VitaSink(std::string_view) : impl(std::make_unique<Impl>()) {
    impl->port = sceAudioOutOpenPort(SCE_AUDIO_OUT_PORT_TYPE_BGM, Impl::OutputFrames,
                                     Impl::OutputRate, SCE_AUDIO_OUT_MODE_STEREO);
    if (impl->port < 0) {
        LOG_CRITICAL(Audio_Sink, "sceAudioOutOpenPort failed: {:#x}", static_cast<u32>(impl->port));
        return;
    }
    impl->thread = Common::NamedThread{Common::ThreadCfg{"audio", 256 * 1024},
                                       [this](std::stop_token stop) { impl->Loop(stop); }};
}

VitaSink::~VitaSink() {
    if (impl->thread.joinable()) {
        impl->thread.request_stop();
        impl->thread.join();
    }
    if (impl->port >= 0) {
        sceAudioOutReleasePort(impl->port);
    }
}

unsigned int VitaSink::GetNativeSampleRate() const {
    // What the callback is asked for: the DSP's rate. The resampling to the port's rate is
    // this sink's own, as SDL's device conversion was on the desktop.
    return native_sample_rate;
}

void VitaSink::SetCallback(std::function<void(s16*, std::size_t)> cb) {
    std::scoped_lock lock{impl->cb_mutex};
    impl->cb = std::move(cb);
}

void VitaSink::Impl::Fill() {
    // Pull enough input for one output buffer: the read position advances by
    // OutputFrames * ratio, and the interpolation reads the frame after the last one it lands
    // on.
    constexpr double ratio = static_cast<double>(native_sample_rate) / OutputRate;
    const std::size_t have = input.size() / 2;
    const std::size_t need = static_cast<std::size_t>(pos + OutputFrames * ratio) + 2;
    if (need > have) {
        const std::size_t pull = need - have;
        input.resize((have + pull) * 2);
        s16* dst = input.data() + have * 2;
        std::scoped_lock lock{cb_mutex};
        if (cb) {
            cb(dst, pull);
        } else {
            std::memset(dst, 0, pull * 2 * sizeof(s16));
        }
    }
    for (std::size_t i = 0; i < OutputFrames; i++) {
        const std::size_t base = static_cast<std::size_t>(pos);
        const float frac = static_cast<float>(pos - static_cast<double>(base));
        const s16* a = input.data() + base * 2;
        const s16* b = a + 2;
        output[i * 2 + 0] = static_cast<s16>(a[0] + (b[0] - a[0]) * frac);
        output[i * 2 + 1] = static_cast<s16>(a[1] + (b[1] - a[1]) * frac);
        pos += ratio;
    }
    // Drop the frames the position has passed; keep the fraction.
    const std::size_t consumed = static_cast<std::size_t>(pos);
    input.erase(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(consumed * 2));
    pos -= static_cast<double>(consumed);
}

void VitaSink::Impl::Loop(std::stop_token stop) {
    Common::SetCurrentThreadRole(Common::ThreadRole::AudioSink);
    input.reserve(InputFrames * 4);
    while (!stop.stop_requested()) {
        Fill();
        // Returns once the port has taken the buffer, which is when the previous one is
        // playing: the port's own double buffering is the clock this thread runs on.
        const int r = sceAudioOutOutput(port, output.data());
        if (r < 0) {
            LOG_ERROR(Audio_Sink, "sceAudioOutOutput failed: {:#x}", static_cast<u32>(r));
            sceKernelDelayThread(OutputFrames * 1000000 / OutputRate);
        }
    }
    // Let the port drain rather than cut the last buffer.
    sceAudioOutOutput(port, nullptr);
}

std::vector<std::string> ListVitaSinkDevices() {
    return {"Vita"};
}

} // namespace AudioCore
