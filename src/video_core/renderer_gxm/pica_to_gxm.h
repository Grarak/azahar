// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <psp2/gxm.h>
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_rasterizer.h"

namespace PicaToGxm {

inline SceGxmBlendFunc BlendEquation(Pica::FramebufferRegs::BlendEquation equation) {
    using Eq = Pica::FramebufferRegs::BlendEquation;
    switch (equation) {
    case Eq::Add:
        return SCE_GXM_BLEND_FUNC_ADD;
    case Eq::Subtract:
        return SCE_GXM_BLEND_FUNC_SUBTRACT;
    case Eq::ReverseSubtract:
        return SCE_GXM_BLEND_FUNC_REVERSE_SUBTRACT;
    case Eq::Min:
        return SCE_GXM_BLEND_FUNC_MIN;
    case Eq::Max:
        return SCE_GXM_BLEND_FUNC_MAX;
    default:
        return SCE_GXM_BLEND_FUNC_ADD;
    }
}

inline SceGxmBlendFactor BlendFactor(Pica::FramebufferRegs::BlendFactor factor) {
    using F = Pica::FramebufferRegs::BlendFactor;
    switch (factor) {
    case F::Zero:
        return SCE_GXM_BLEND_FACTOR_ZERO;
    case F::One:
        return SCE_GXM_BLEND_FACTOR_ONE;
    case F::SourceColor:
        return SCE_GXM_BLEND_FACTOR_SRC_COLOR;
    case F::OneMinusSourceColor:
        return SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case F::DestColor:
        return SCE_GXM_BLEND_FACTOR_DST_COLOR;
    case F::OneMinusDestColor:
        return SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case F::SourceAlpha:
        return SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
    case F::OneMinusSourceAlpha:
        return SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case F::DestAlpha:
        return SCE_GXM_BLEND_FACTOR_DST_ALPHA;
    case F::OneMinusDestAlpha:
        return SCE_GXM_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case F::SourceAlphaSaturate:
        return SCE_GXM_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    // GXM has no constant-color factors; the blend colour is folded in by the fragment
    // program in a later phase. Until then these approximate.
    case F::ConstantColor:
    case F::ConstantAlpha:
        return SCE_GXM_BLEND_FACTOR_ONE;
    case F::OneMinusConstantColor:
    case F::OneMinusConstantAlpha:
        return SCE_GXM_BLEND_FACTOR_ZERO;
    default:
        return SCE_GXM_BLEND_FACTOR_ONE;
    }
}

inline SceGxmDepthFunc CompareFunc(Pica::FramebufferRegs::CompareFunc func) {
    using C = Pica::FramebufferRegs::CompareFunc;
    switch (func) {
    case C::Never:
        return SCE_GXM_DEPTH_FUNC_NEVER;
    case C::Always:
        return SCE_GXM_DEPTH_FUNC_ALWAYS;
    case C::Equal:
        return SCE_GXM_DEPTH_FUNC_EQUAL;
    case C::NotEqual:
        return SCE_GXM_DEPTH_FUNC_NOT_EQUAL;
    case C::LessThan:
        return SCE_GXM_DEPTH_FUNC_LESS;
    case C::LessThanOrEqual:
        return SCE_GXM_DEPTH_FUNC_LESS_EQUAL;
    case C::GreaterThan:
        return SCE_GXM_DEPTH_FUNC_GREATER;
    case C::GreaterThanOrEqual:
        return SCE_GXM_DEPTH_FUNC_GREATER_EQUAL;
    default:
        return SCE_GXM_DEPTH_FUNC_ALWAYS;
    }
}

inline SceGxmStencilFunc StencilFunc(Pica::FramebufferRegs::CompareFunc func) {
    using C = Pica::FramebufferRegs::CompareFunc;
    switch (func) {
    case C::Never:
        return SCE_GXM_STENCIL_FUNC_NEVER;
    case C::Always:
        return SCE_GXM_STENCIL_FUNC_ALWAYS;
    case C::Equal:
        return SCE_GXM_STENCIL_FUNC_EQUAL;
    case C::NotEqual:
        return SCE_GXM_STENCIL_FUNC_NOT_EQUAL;
    case C::LessThan:
        return SCE_GXM_STENCIL_FUNC_LESS;
    case C::LessThanOrEqual:
        return SCE_GXM_STENCIL_FUNC_LESS_EQUAL;
    case C::GreaterThan:
        return SCE_GXM_STENCIL_FUNC_GREATER;
    case C::GreaterThanOrEqual:
        return SCE_GXM_STENCIL_FUNC_GREATER_EQUAL;
    default:
        return SCE_GXM_STENCIL_FUNC_ALWAYS;
    }
}

inline SceGxmStencilOp StencilOp(Pica::FramebufferRegs::StencilAction action) {
    using A = Pica::FramebufferRegs::StencilAction;
    switch (action) {
    case A::Keep:
        return SCE_GXM_STENCIL_OP_KEEP;
    case A::Zero:
        return SCE_GXM_STENCIL_OP_ZERO;
    case A::Replace:
        return SCE_GXM_STENCIL_OP_REPLACE;
    case A::Increment:
        return SCE_GXM_STENCIL_OP_INCR;
    case A::Decrement:
        return SCE_GXM_STENCIL_OP_DECR;
    case A::Invert:
        return SCE_GXM_STENCIL_OP_INVERT;
    case A::IncrementWrap:
        return SCE_GXM_STENCIL_OP_INCR_WRAP;
    case A::DecrementWrap:
        return SCE_GXM_STENCIL_OP_DECR_WRAP;
    default:
        return SCE_GXM_STENCIL_OP_KEEP;
    }
}

} // namespace PicaToGxm
