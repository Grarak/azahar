// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#if defined(__ARM_FEATURE_SAT)
#include <arm_acle.h>
#endif
#include "audio_core/audio_types.h"
#include "audio_core/codec.h"
#include "common/assert.h"
#include "common/common_types.h"

namespace AudioCore::Codec {

StereoBuffer16 DecodeADPCM(const u8* const data, const std::size_t sample_count,
                           const std::array<s16, 16>& adpcm_coeff, ADPCMState& state) {
    // GC-ADPCM with scale factor and variable coefficients.
    // Frames are 8 bytes long containing 14 samples each.
    // Samples are 4 bits (one nibble) long.

    constexpr std::size_t FRAME_LEN = 8;
    constexpr std::size_t SAMPLES_PER_FRAME = 14;
    static constexpr std::array<int, 16> SIGNED_NIBBLES{
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    };

    // Whole nibble pairs are decoded, so an odd count produces one sample more, and the
    // filter state advances over it too (the DSP decodes bytes, not samples).
    const std::size_t decoded = sample_count % 2 == 0 ? sample_count : sample_count + 1;
    StereoBuffer16 ret(decoded);
    if (decoded == 0) {
        return ret;
    }

    int yn1 = state.yn1, yn2 = state.yn2;
    s16* out = ret.data()->data();
    // Everything in 11-bit fixed point, the second order filter, then back. 0x400 == 0.5 in
    // 11-bit fixed point. Filter: y[n] = x[n] + 0.5 + c1 * y[n-1] + c2 * y[n-2]. The result
    // saturates to 16 bits, one instruction where the target has it: the feedback makes this
    // a serial chain, and the clamp's compares were the longest link.
    const auto decode = [&](int nibble, int shift, int coef1, int coef2) {
        const int xn = SIGNED_NIBBLES[nibble] << shift;
        const int raw = ((xn << 11) + 0x400 + coef1 * yn1 + coef2 * yn2) >> 11;
#if defined(__ARM_FEATURE_SAT)
        const int val = __ssat(raw, 16);
#else
        const int val = std::clamp(raw, -32768, 32767);
#endif
        yn2 = yn1;
        yn1 = val;
        out[0] = static_cast<s16>(val);
        out[1] = static_cast<s16>(val);
        out += 2;
    };
    const u8* frame = data;
    for (std::size_t remaining = decoded; remaining != 0; frame += FRAME_LEN) {
        const int frame_header = frame[0];
        // The scale is a power of two: multiplying a nibble by it is a shift.
        const int shift = frame_header & 0xF;
        const int idx = (frame_header >> 4) & 0x7;
        // Coefficients are fixed point with 11 bits fractional part.
        const int coef1 = adpcm_coeff[idx * 2 + 0];
        const int coef2 = adpcm_coeff[idx * 2 + 1];
        const std::size_t count = std::min(SAMPLES_PER_FRAME, remaining);
        remaining -= count;
        const u8* nibbles = frame + 1;
        const u8* const nibbles_end = nibbles + count / 2;
        for (; nibbles != nibbles_end; nibbles++) {
            const int byte = *nibbles;
            decode(byte >> 4, shift, coef1, coef2);
            decode(byte & 0xF, shift, coef1, coef2);
        }
    }

    state.yn1 = static_cast<s16>(yn1);
    state.yn2 = static_cast<s16>(yn2);

    return ret;
}

StereoBuffer16 DecodePCM8(const unsigned num_channels, const u8* const data,
                          const std::size_t sample_count) {
    ASSERT(num_channels == 1 || num_channels == 2);

    const auto decode_sample = [](u8 sample) {
        return static_cast<s16>(static_cast<u16>(sample) << 8);
    };

    StereoBuffer16 ret(sample_count);

    if (num_channels == 1) {
        for (std::size_t i = 0; i < sample_count; i++) {
            ret[i].fill(decode_sample(data[i]));
        }
    } else {
        for (std::size_t i = 0; i < sample_count; i++) {
            ret[i][0] = decode_sample(data[i * 2 + 0]);
            ret[i][1] = decode_sample(data[i * 2 + 1]);
        }
    }

    return ret;
}

StereoBuffer16 DecodePCM16(const unsigned num_channels, const u8* const data,
                           const std::size_t sample_count) {
    ASSERT(num_channels == 1 || num_channels == 2);

    StereoBuffer16 ret(sample_count);

    if (num_channels == 1) {
        for (std::size_t i = 0; i < sample_count; i++) {
            s16 sample;
            std::memcpy(&sample, data + i * sizeof(s16), sizeof(s16));
            ret[i].fill(sample);
        }
    } else {
        for (std::size_t i = 0; i < sample_count; ++i) {
            std::memcpy(&ret[i], data + i * sizeof(s16) * 2, 2 * sizeof(s16));
        }
    }

    return ret;
}
} // namespace AudioCore::Codec
