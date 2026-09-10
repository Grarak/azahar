// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define AUDIO_FILTER_NEON 1
#endif
#include "audio_core/hle/common.h"
#include "audio_core/hle/filter.h"
#include "audio_core/hle/shared_memory.h"
#include "common/common_types.h"

namespace AudioCore::HLE {

void SourceFilters::Reset() {
    Enable(false, false);
}

void SourceFilters::Enable(bool simple, bool biquad) {
    simple_filter_enabled = simple;
    biquad_filter_enabled = biquad;

    if (!simple)
        simple_filter.Reset();
    if (!biquad)
        biquad_filter.Reset();
}

void SourceFilters::Configure(SourceConfiguration::Configuration::SimpleFilter config) {
    simple_filter.Configure(config);
}

void SourceFilters::Configure(SourceConfiguration::Configuration::BiquadFilter config) {
    biquad_filter.Configure(config);
}

void SourceFilters::ProcessFrame(StereoFrame16& frame) {
    if (!simple_filter_enabled && !biquad_filter_enabled)
        return;

    if (simple_filter_enabled) {
        simple_filter.ProcessFrame(frame);
    }

    if (biquad_filter_enabled) {
        biquad_filter.ProcessFrame(frame);
    }
}

// SimpleFilter

void SourceFilters::SimpleFilter::Reset() {
    y1.fill(0);
    // Configure as passthrough.
    a1 = 0;
    b0 = 1 << 15;
}

void SourceFilters::SimpleFilter::Configure(
    SourceConfiguration::Configuration::SimpleFilter config) {

    a1 = config.a1;
    b0 = config.b0;
}

std::array<s16, 2> SourceFilters::SimpleFilter::ProcessSample(const std::array<s16, 2>& x0) {
    std::array<s16, 2> y0;
    for (std::size_t i = 0; i < 2; i++) {
        const s32 tmp = (b0 * x0[i] + a1 * y1[i]) >> 15;
        y0[i] = std::clamp(tmp, -32768, 32767);
    }

    y1 = y0;

    return y0;
}

#ifdef AUDIO_FILTER_NEON
namespace {
// One stereo sample as the two low lanes of a vector, and back. The lanes are the channels;
// the recursion is along time, so a sample's two channels are all the parallelism there is.
inline int32x2_t LoadSample(const std::array<s16, 2>& sample) {
    u32 packed;
    std::memcpy(&packed, sample.data(), sizeof(packed));
    return vget_low_s32(vmovl_s16(vreinterpret_s16_u32(vdup_n_u32(packed))));
}
// Shifts the accumulator down, saturates to 16 bits (the std::clamp of the scalar path) and
// stores; returns the stored sample widened again for the feedback.
inline int32x2_t StoreSample(std::array<s16, 2>& sample, int32x2_t acc, int shift) {
    const int32x2_t shifted = shift == 15 ? vshr_n_s32(acc, 15) : vshr_n_s32(acc, 14);
    const int16x4_t narrow = vqmovn_s32(vcombine_s32(shifted, shifted));
    const u32 packed = vget_lane_u32(vreinterpret_u32_s16(narrow), 0);
    std::memcpy(sample.data(), &packed, sizeof(packed));
    return vget_low_s32(vmovl_s16(narrow));
}
inline std::array<s16, 2> ToArray(int32x2_t v) {
    return {static_cast<s16>(vget_lane_s32(v, 0)), static_cast<s16>(vget_lane_s32(v, 1))};
}
} // namespace
#endif

void SourceFilters::SimpleFilter::ProcessFrame(StereoFrame16& frame) {
#ifdef AUDIO_FILTER_NEON
    const int32x2_t b0v = vdup_n_s32(b0);
    const int32x2_t a1v = vdup_n_s32(a1);
    int32x2_t y1v = LoadSample(y1);
    for (auto& sample : frame) {
        int32x2_t acc = vmul_s32(b0v, LoadSample(sample));
        acc = vmla_s32(acc, a1v, y1v);
        y1v = StoreSample(sample, acc, 15);
    }
    y1 = ToArray(y1v);
#else
    s32 l1 = y1[0], r1 = y1[1];
    for (auto& sample : frame) {
        l1 = std::clamp((b0 * sample[0] + a1 * l1) >> 15, -32768, 32767);
        r1 = std::clamp((b0 * sample[1] + a1 * r1) >> 15, -32768, 32767);
        sample[0] = static_cast<s16>(l1);
        sample[1] = static_cast<s16>(r1);
    }
    y1 = {static_cast<s16>(l1), static_cast<s16>(r1)};
#endif
}

// BiquadFilter

void SourceFilters::BiquadFilter::Reset() {
    x1.fill(0);
    x2.fill(0);
    y1.fill(0);
    y2.fill(0);
    // Configure as passthrough.
    a1 = a2 = b1 = b2 = 0;
    b0 = 1 << 14;
}

void SourceFilters::BiquadFilter::Configure(
    SourceConfiguration::Configuration::BiquadFilter config) {

    a1 = config.a1;
    a2 = config.a2;
    b0 = config.b0;
    b1 = config.b1;
    b2 = config.b2;
}

std::array<s16, 2> SourceFilters::BiquadFilter::ProcessSample(const std::array<s16, 2>& x0) {
    std::array<s16, 2> y0;
    for (std::size_t i = 0; i < 2; i++) {
        const s32 tmp = (b0 * x0[i] + b1 * x1[i] + b2 * x2[i] + a1 * y1[i] + a2 * y2[i]) >> 14;
        y0[i] = std::clamp(tmp, -32768, 32767);
    }

    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;

    return y0;
}

void SourceFilters::BiquadFilter::ProcessFrame(StereoFrame16& frame) {
#ifdef AUDIO_FILTER_NEON
    const int32x2_t b0v = vdup_n_s32(b0), b1v = vdup_n_s32(b1), b2v = vdup_n_s32(b2);
    const int32x2_t a1v = vdup_n_s32(a1), a2v = vdup_n_s32(a2);
    int32x2_t x1v = LoadSample(x1), x2v = LoadSample(x2);
    int32x2_t y1v = LoadSample(y1), y2v = LoadSample(y2);
    for (auto& sample : frame) {
        const int32x2_t x0v = LoadSample(sample);
        int32x2_t acc = vmul_s32(b0v, x0v);
        acc = vmla_s32(acc, b1v, x1v);
        acc = vmla_s32(acc, b2v, x2v);
        acc = vmla_s32(acc, a1v, y1v);
        acc = vmla_s32(acc, a2v, y2v);
        x2v = x1v;
        x1v = x0v;
        y2v = y1v;
        y1v = StoreSample(sample, acc, 14);
    }
    x1 = ToArray(x1v);
    x2 = ToArray(x2v);
    y1 = ToArray(y1v);
    y2 = ToArray(y2v);
#else
    s32 lx1 = x1[0], rx1 = x1[1], lx2 = x2[0], rx2 = x2[1];
    s32 ly1 = y1[0], ry1 = y1[1], ly2 = y2[0], ry2 = y2[1];
    for (auto& sample : frame) {
        const s32 lx0 = sample[0], rx0 = sample[1];
        const s32 ly0 =
            std::clamp((b0 * lx0 + b1 * lx1 + b2 * lx2 + a1 * ly1 + a2 * ly2) >> 14, -32768, 32767);
        const s32 ry0 =
            std::clamp((b0 * rx0 + b1 * rx1 + b2 * rx2 + a1 * ry1 + a2 * ry2) >> 14, -32768, 32767);
        lx2 = lx1; rx2 = rx1;
        lx1 = lx0; rx1 = rx0;
        ly2 = ly1; ry2 = ry1;
        ly1 = ly0; ry1 = ry0;
        sample[0] = static_cast<s16>(ly0);
        sample[1] = static_cast<s16>(ry0);
    }
    x1 = {static_cast<s16>(lx1), static_cast<s16>(rx1)};
    x2 = {static_cast<s16>(lx2), static_cast<s16>(rx2)};
    y1 = {static_cast<s16>(ly1), static_cast<s16>(ry1)};
    y2 = {static_cast<s16>(ly2), static_cast<s16>(ry2)};
#endif
}

} // namespace AudioCore::HLE
