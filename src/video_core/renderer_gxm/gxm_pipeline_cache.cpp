// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#if defined(__vita__)
#include <psp2/kernel/threadmgr.h>
#endif
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "video_core/renderer_gxm/gxm_device.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "video_core/renderer_gxm/gxm_pipeline_cache.h"

namespace GxmRenderer {

u64 PipelineInfo::Hash() const {
    struct {
        const void* vertex;
        const void* fragment;
        u32 stream_count;
        u32 attribute_count;
        u32 blend_enabled;
        u32 output_format;
        SceGxmBlendInfo blend;
        std::array<u16, PipelineInfo::MaxStreams> strides;
        std::array<u64, PipelineInfo::MaxAttributes> attrs;
    } key{};
    key.vertex = vertex;
    key.fragment = fragment;
    key.stream_count = stream_count;
    key.attribute_count = attribute_count;
    key.blend_enabled = blend_enabled ? 1u : 0u;
    key.output_format = static_cast<u32>(output_format);
    key.blend = blend;
    key.strides = strides;
    for (u32 i = 0; i < attribute_count; i++) {
        const auto& a = attributes[i];
        key.attrs[i] = (static_cast<u64>(a.offset) << 32) | (static_cast<u64>(a.stream) << 16) |
                       (static_cast<u64>(a.format) << 8) | a.components;
        // The name is part of the vertex shader, which is already keyed by pointer.
    }
    return Common::ComputeHash64(&key, sizeof(key));
}

Pipeline::Pipeline(const PipelineInfo& info_) : info{info_} {}

Pipeline::~Pipeline() {
    if (!HasDevice()) {
        return;
    }
    SceGxmShaderPatcher* patcher = Device().patcher;
    if (fragment_program != nullptr) {
        sceGxmShaderPatcherReleaseFragmentProgram(patcher, fragment_program);
    }
    if (vertex_program != nullptr) {
        sceGxmShaderPatcherReleaseVertexProgram(patcher, vertex_program);
    }
}

bool Pipeline::TryBuild() {
    if (IsDone()) {
        return !failed;
    }
    Shader* const stages[2] = {info.vertex, info.fragment};
    for (Shader* shader : stages) {
        if (shader == nullptr) {
            continue;
        }
        if (!shader->IsDone()) {
            // Programs are emitted on this thread before they are ever handed out, so one that
            // is not finished here was never built at all. There is nothing to wait for.
            return false;
        }
        if (shader->failed) {
            failed = true;
            MarkDone();
            return false;
        }
    }
    failed = !Build();
    MarkDone();
    return !failed;
}

bool Pipeline::Build() {
    SceGxmShaderPatcher* patcher = Device().patcher;
    for (Shader* shader : {info.vertex, info.fragment}) {
        if (shader->patcher_id != nullptr) {
            continue;
        }
        const int err =
            sceGxmShaderPatcherRegisterProgram(patcher, shader->Program(), &shader->patcher_id);
        if (err < 0) {
            LOG_ERROR(Render, "the patcher refused shader '{}': {} ({:#x})", shader->name,
                      GxmErrorName(err), static_cast<u32>(err));
            return false;
        }
    }
    const SceGxmProgram* vertex = info.vertex->Program();
    std::array<SceGxmVertexAttribute, PipelineInfo::MaxAttributes> attrs{};
    for (u32 i = 0; i < info.attribute_count; i++) {
        const auto& desc = info.attributes[i];
        const SceGxmProgramParameter* param =
            sceGxmProgramFindParameterByName(vertex, desc.name);
        if (param == nullptr) {
            LOG_ERROR(Render, "vertex shader '{}' has no attribute '{}'", info.vertex->name,
                      desc.name);
            return false;
        }
        attrs[i].streamIndex = desc.stream;
        attrs[i].offset = desc.offset;
        attrs[i].format = static_cast<u8>(desc.format);
        attrs[i].componentCount = desc.components;
        attrs[i].regIndex = sceGxmProgramParameterGetResourceIndex(param);
    }
    std::array<SceGxmVertexStream, PipelineInfo::MaxStreams> streams{};
    for (u32 i = 0; i < info.stream_count; i++) {
        streams[i].stride = info.strides[i];
        streams[i].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
    }
    const int vertex_err = sceGxmShaderPatcherCreateVertexProgram(
        patcher, info.vertex->patcher_id, attrs.data(), info.attribute_count, streams.data(),
        info.stream_count, &vertex_program);
    if (vertex_err < 0) {
        std::string what;
        for (u32 i = 0; i < info.attribute_count; i++) {
            what += fmt::format(" {}:s{}+{} f{:#x} x{} r{}", info.attributes[i].name,
                                attrs[i].streamIndex, attrs[i].offset, attrs[i].format,
                                attrs[i].componentCount, attrs[i].regIndex);
        }
        for (u32 i = 0; i < info.stream_count; i++) {
            what += fmt::format(" stride{}={}", i, streams[i].stride);
        }
        LOG_ERROR(Render, "vertex program creation failed for '{}': {} ({:#x}):{}",
                  info.vertex->name, GxmErrorName(vertex_err), static_cast<u32>(vertex_err),
                  what);
        return false;
    }
    // Blend state is baked into the fragment program: this is GXM's one departure from
    // dynamic state, and why the pipeline is keyed on it.
    const int fragment_err = sceGxmShaderPatcherCreateFragmentProgram(
        patcher, info.fragment->patcher_id, info.output_format, SCE_GXM_MULTISAMPLE_NONE,
        info.blend_enabled ? &info.blend : nullptr, vertex, &fragment_program);
    if (fragment_err < 0) {
        const auto& blend = info.blend;
        LOG_ERROR(Render,
                  "fragment program creation failed for '{}': {} ({:#x}); blend {} mask {:#x} "
                  "colour {}/{}/{} alpha {}/{}/{}",
                  info.fragment->name, GxmErrorName(fragment_err),
                  static_cast<u32>(fragment_err), info.blend_enabled, blend.colorMask,
                  static_cast<int>(blend.colorFunc), static_cast<int>(blend.colorSrc),
                  static_cast<int>(blend.colorDst), static_cast<int>(blend.alphaFunc),
                  static_cast<int>(blend.alphaSrc), static_cast<int>(blend.alphaDst));
        return false;
    }
    return true;
}

PipelineCache::PipelineCache() = default;

PipelineCache::~PipelineCache() {
    if (HasDevice() && Device().context != nullptr) {
        // Releasing a program needs the GPU done with it and the context no longer binding it.
        sceGxmFinish(Device().context);
    }
    pipelines.clear();
    if (HasDevice()) {
        for (auto& [key, shader] : shaders) {
            if (shader->patcher_id != nullptr) {
                sceGxmShaderPatcherUnregisterProgram(Device().patcher, shader->patcher_id);
            }
        }
        for (auto& shader : retired) {
            if (shader->patcher_id != nullptr) {
                sceGxmShaderPatcherUnregisterProgram(Device().patcher, shader->patcher_id);
            }
        }
    }
}

Shader* PipelineCache::FindShader(u64 key) const {
    const auto it = shaders.find(key);
    return it == shaders.end() ? nullptr : it->second.get();
}

Shader* PipelineCache::UseBuiltinShader(u64 key, std::string name, std::span<const u8> program,
                                        u32 input_regs) {
    const auto [it, is_new] = shaders.try_emplace(key);
    if (!is_new) {
        return it->second.get();
    }
    it->second = std::make_unique<Shader>();
    Shader* shader = it->second.get();
    shader->name = std::move(name);
    shader->input_regs = input_regs;
    shader->program.assign(program.begin(), program.end());
    const int err = sceGxmProgramCheck(shader->Program());
    if (err < 0) {
        LOG_ERROR(Render, "built-in program '{}' failed sceGxmProgramCheck: {} ({:#x})",
                  shader->name, GxmErrorName(err), static_cast<u32>(err));
        shader->program.clear();
        shader->failed = true;
    }
    shader->MarkDone();
    return shader;
}

Shader* PipelineCache::Replace(u64 key, std::string name, std::span<const u8> program,
                               u32 input_regs) {
    auto replaced = std::make_unique<Shader>();
    Shader* shader = replaced.get();
    shader->name = std::move(name);
    shader->input_regs = input_regs;
    shader->tier = 2;
    shader->tier2_queued = true;
    shader->program.assign(program.begin(), program.end());
    const int err = sceGxmProgramCheck(shader->Program());
    if (err < 0) {
        LOG_ERROR(Render, "tier 2 program '{}' failed sceGxmProgramCheck: {} ({:#x})",
                  shader->name, GxmErrorName(err), static_cast<u32>(err));
        return nullptr;
    }
    shader->MarkDone();
    auto& slot = shaders[key];
    if (slot) {
        retired.push_back(std::move(slot));
    }
    slot = std::move(replaced);
    return shader;
}

bool PipelineCache::BindPipeline(const PipelineInfo& info) {
    const u64 hash = info.Hash();
    const auto [it, is_new] = pipelines.try_emplace(hash);
    if (is_new) {
        it->second = std::make_unique<Pipeline>(info);
    }
    Pipeline* pipeline = it->second.get();
    if (!pipeline->TryBuild()) {
        return false;
    }
    if (pipeline != current_pipeline) {
        SceGxmContext* context = Device().context;
        sceGxmSetVertexProgram(context, pipeline->VertexProgram());
        sceGxmSetFragmentProgram(context, pipeline->FragmentProgram());
        current_pipeline = pipeline;
    }
    return true;
}

} // namespace GxmRenderer
