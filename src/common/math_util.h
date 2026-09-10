// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// Copyright 2013 Dolphin Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstdlib>
#include <span>
#include <type_traits>
#include <utility>
#include "common_types.h"

namespace Common {

/**
 * floor() without the library call.
 *
 * ARMv7's VFP has no round-toward-negative instruction - vrintm arrived with ARMv8 - so
 * std::floor on that target is a call into libm, and the software rasterizer makes it per
 * fragment: it was 2.1% of a gameplay profile. Converting through int truncates toward zero,
 * which is floor already for anything non-negative, and the correction below costs a compare
 * and a subtract for the rest.
 *
 * Only valid where the value fits in an s32, which every user here is: texture coordinates,
 * lighting lookups and fog indices are all small and bounded.
 */
[[nodiscard]] inline float FastFloor(float x) {
    const float truncated = static_cast<float>(static_cast<s32>(x));
    return truncated > x ? truncated - 1.0f : truncated;
}

constexpr float PI = 3.14159265f;

template <class T>
struct Rectangle {
    T left{};
    T top{};
    T right{};
    T bottom{};

    constexpr Rectangle() = default;

    constexpr Rectangle(T left, T top, T right, T bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    template <typename U>
    operator Rectangle<U>() const {
        return Rectangle<U>{static_cast<U>(left), static_cast<U>(top), static_cast<U>(right),
                            static_cast<U>(bottom)};
    }

    [[nodiscard]] constexpr bool operator==(const Rectangle<T>& rhs) const {
        return (left == rhs.left) && (top == rhs.top) && (right == rhs.right) &&
               (bottom == rhs.bottom);
    }

    [[nodiscard]] constexpr bool operator!=(const Rectangle<T>& rhs) const {
        return !operator==(rhs);
    }

    [[nodiscard]] constexpr Rectangle<T> operator*(const T value) const {
        return Rectangle{left * value, top * value, right * value, bottom * value};
    }
    [[nodiscard]] constexpr Rectangle<T> operator/(const T value) const {
        return Rectangle{left / value, top / value, right / value, bottom / value};
    }

    [[nodiscard]] T GetWidth() const {
        return std::abs(static_cast<std::make_signed_t<T>>(right - left));
    }
    [[nodiscard]] T GetHeight() const {
        return std::abs(static_cast<std::make_signed_t<T>>(bottom - top));
    }
    [[nodiscard]] Rectangle<T> TranslateX(const T x) const {
        return Rectangle{left + x, top, right + x, bottom};
    }
    [[nodiscard]] Rectangle<T> TranslateY(const T y) const {
        return Rectangle{left, top + y, right, bottom + y};
    }
    [[nodiscard]] Rectangle<T> Scale(const float s) const {
        return Rectangle{left, top, static_cast<T>(left + GetWidth() * s),
                         static_cast<T>(top + GetHeight() * s)};
    }
    [[nodiscard]] Rectangle<T> VerticalMirror(T ref_height) const {
        return Rectangle{left, ref_height - bottom, right, ref_height - top};
    }
};

template <typename T>
Rectangle(T, T, T, T) -> Rectangle<T>;

std::pair<u8, u8> FindMinMax(const std::span<const u8>& data);
std::pair<u16, u16> FindMinMax(const std::span<const u16>& data);

} // namespace Common
