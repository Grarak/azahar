// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <boost/predef.h>

#define CITRA_ARCH(NAME) (CITRA_ARCH_##NAME)

#define CITRA_ARCH_x86_64 BOOST_ARCH_X86_64
#define CITRA_ARCH_arm64                                                                           \
    (BOOST_ARCH_ARM >= BOOST_VERSION_NUMBER(8, 0, 0) && BOOST_ARCH_WORD_BITS == 64)
#define CITRA_ARCH_arm32                                                                           \
    (BOOST_ARCH_ARM >= BOOST_VERSION_NUMBER(7, 0, 0) && BOOST_ARCH_WORD_BITS == 32)

/**
 * Whether a shader JIT exists for this target. The three architectures with a compiler have one,
 * except on the PS Vita: mapping executable memory there needs a kernel plugin this build does
 * not require, so the shader interpreter is used instead.
 */
#if (CITRA_ARCH(x86_64) || CITRA_ARCH(arm64) || CITRA_ARCH(arm32)) && !defined(__vita__)
#define CITRA_HAS_SHADER_JIT 1
#else
#define CITRA_HAS_SHADER_JIT 0
#endif
