// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/arch.h"
#if CITRA_HAS_SHADER_JIT

#include <chrono>

#include "common/assert.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "video_core/shader/shader.h"
#include "video_core/shader/shader_jit.h"
#if CITRA_ARCH(arm64)
#include "video_core/shader/shader_jit_a64_compiler.h"
#endif
#if CITRA_ARCH(x86_64)
#include "video_core/shader/shader_jit_x64_compiler.h"
#endif
#if CITRA_ARCH(arm32)
#include "video_core/shader/shader_jit_a32_compiler.h"
#endif

namespace Pica::Shader {

JitEngine::JitEngine() = default;
JitEngine::~JitEngine() = default;

void JitEngine::SetupBatch(ShaderSetup& setup, u32 entry_point) {
    ASSERT(entry_point < MAX_PROGRAM_CODE_LENGTH);
    setup.entry_point = entry_point;

    setup.DoProgramCodeFixup();
    const u64 code_hash = setup.GetProgramCodeHash();
    const u64 swizzle_hash = setup.GetSwizzleDataHash();

    const u64 cache_key = Common::HashCombine(code_hash, swizzle_hash);
    auto iter = cache.find(cache_key);
    if (iter != cache.end()) {
        setup.cached_shader = iter->second.get();
    } else {
        // Every miss here is a guest shader being translated on whichever thread is running
        // the command list, which is a stall on that thread. Logged so a hitch can be matched
        // against a compile rather than guessed at.
        auto shader = std::make_unique<JitShader>();
        const auto start = std::chrono::steady_clock::now();
        shader->Compile(&setup.GetProgramCode(), &setup.GetSwizzleData());
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        LOG_INFO(HW_GPU, "shader JIT compiled {:016x} entry {} in {}.{:03} ms (cached {})",
                 cache_key, entry_point, us / 1000, us % 1000, cache.size() + 1);
        setup.cached_shader = shader.get();
        cache.emplace_hint(iter, cache_key, std::move(shader));
    }
}

MICROPROFILE_DECLARE(GPU_Shader);

void JitEngine::Run(const ShaderSetup& setup, ShaderUnit& state) const {
    ASSERT(setup.cached_shader != nullptr);

    MICROPROFILE_SCOPE(GPU_Shader);

    const JitShader* shader = static_cast<const JitShader*>(setup.cached_shader);
    shader->Run(setup, state, setup.entry_point);
}

} // namespace Pica::Shader

#endif // CITRA_HAS_SHADER_JIT
