// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <algorithm>
#include <deque>
#include <vector>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/split_member.hpp>
#include "common/common_types.h"

namespace AudioCore {

/// Samples per second which the 3DS's audio hardware natively outputs at
constexpr int native_sample_rate = 32728; // Hz

/// Samples per audio frame at native sample rate
constexpr int samples_per_frame = 160;

/// The final output to the speakers is stereo. Preprocessing output in Source is also stereo.
using StereoFrame16 = std::array<std::array<s16, 2>, samples_per_frame>;

/// The DSP is quadraphonic internally.
using QuadFrame32 = std::array<std::array<s32, 4>, samples_per_frame>;

/**
 * A variable length buffer of signed PCM16 stereo samples: a decoded guest buffer, consumed
 * from the front by the interpolator, which also pushes its two history samples in front of
 * it every tick. Was a std::deque, whose per-element chunk arithmetic was most of the
 * interpolator's and the decoders' time on the emulation thread (2026-09-07, pi5: the audio
 * HLE was 7% of it). A flat vector with a read head and a little slack in front: a pop is a
 * head move, the two history samples fit in the slack, the decoders and the interpolator
 * walk plain pointers. The archive form stays the deque's, so savestates carry over.
 */
class StereoBuffer16 {
public:
    using Sample = std::array<s16, 2>;
    static constexpr std::size_t FrontSlack = 2;

    StereoBuffer16() = default;
    explicit StereoBuffer16(std::size_t count) : samples(FrontSlack + count), head(FrontSlack) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return samples.size() - head;
    }
    [[nodiscard]] bool empty() const noexcept {
        return head == samples.size();
    }
    void clear() {
        samples.clear();
        head = 0;
    }
    Sample& operator[](std::size_t i) {
        return samples[head + i];
    }
    const Sample& operator[](std::size_t i) const {
        return samples[head + i];
    }
    [[nodiscard]] Sample* data() noexcept {
        return samples.data() + head;
    }
    [[nodiscard]] const Sample* data() const noexcept {
        return samples.data() + head;
    }
    /// Drops the first `count` samples.
    void pop_front(std::size_t count) {
        head += std::min(count, size());
        if (head == samples.size()) {
            samples.clear();
            head = 0;
        }
    }
    /// Puts a sample in front of the first: the interpolator's history. Cheap while the
    /// slack lasts; a full vector shift otherwise.
    void push_front(const Sample& sample) {
        if (head == 0) {
            samples.insert(samples.begin(), FrontSlack, Sample{});
            head = FrontSlack;
        }
        samples[--head] = sample;
    }

    template <class Archive>
    void save(Archive& ar, const unsigned int) const {
        // The deque's own archive layout, so states written before the change load.
        std::deque<Sample> as_deque(samples.begin() + static_cast<std::ptrdiff_t>(head),
                                    samples.end());
        ar << as_deque;
    }
    template <class Archive>
    void load(Archive& ar, const unsigned int) {
        std::deque<Sample> as_deque;
        ar >> as_deque;
        samples.assign(FrontSlack, Sample{});
        samples.insert(samples.end(), as_deque.begin(), as_deque.end());
        head = FrontSlack;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    std::vector<Sample> samples;
    std::size_t head = 0;
};

constexpr std::size_t num_dsp_pipe = 8;
enum class DspPipe {
    Debug = 0,
    Dma = 1,
    Audio = 2,
    Binary = 3,
};

enum class DspState {
    Off,
    On,
    Sleeping,
};

} // namespace AudioCore

// Exactly the deque's bytes on the wire: no class version or object tracking around it, as
// boost writes none for a std::deque. Without this a state saved before the change loaded as
// garbage (a kernel memory assertion right after the DSP section, 2026-09-08).
BOOST_CLASS_IMPLEMENTATION(AudioCore::StereoBuffer16, boost::serialization::object_serializable)
BOOST_CLASS_TRACKING(AudioCore::StereoBuffer16, boost::serialization::track_never)
