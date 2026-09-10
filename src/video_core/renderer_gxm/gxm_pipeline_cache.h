// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <psp2/gxm.h>
#include "common/common_types.h"
#include "video_core/renderer_gxm/gxm_shader_util.h"

namespace GxmRenderer {

/**
 * The shader path, in the Vulkan backend's shape (vk_pipeline_cache) minus the worker: a
 * Shader is a program the USSE emitters made the moment a configuration was first seen
 * (microseconds, so it is ready for that draw), and a Pipeline pairs shaders with a vertex
 * layout and blend state through the patcher, cheap enough to run on the render thread at
 * bind time - which also keeps every patcher call on one thread.
 * SceShaccCg, the console's Cg compiler, is no longer used at all.
 */
/**
 * Whether a program or a pipeline has been built.
 *
 * The Vulkan backend compiles its shaders on worker threads, so its Shader and Pipeline derive
 * from Common::AsyncHandle and a draw that arrives early can wait on one. Nothing here is
 * asynchronous. The USSE emitters run in microseconds on the render thread the moment a
 * configuration is first seen, and the second tier hands its programs back through
 * PipelineCache::Replace on that same thread, so every object is already finished before
 * anything else can look at it and the wait never waited.
 *
 * Carrying the machinery anyway was not free. On this target pte builds a
 * std::condition_variable out of a mutex and a sem_t, and the sem_t carries a mutex of its own,
 * so one costs three kernel semaphores; the std::mutex beside it costs a fourth. The caches
 * never evict, so every program and every pipeline held four process UIDs for the rest of the
 * session, and UIDs are bounded. That is what ran the console out of them - Super Mario 3D Land
 * after nine minutes, Super Smash Bros after three - and running out is silent: the refused
 * pthread_mutex_init leaves a lock that never locks, and the thread waiting on it stops for
 * good.
 */
struct BuildFlag {
    [[nodiscard]] bool IsDone() const noexcept {
        return is_done;
    }
    void MarkDone(bool done = true) noexcept {
        is_done = done;
    }

private:
    bool is_done = false;
};

struct Shader : public BuildFlag {
    std::string name;
    /// The GXP. Owned here for as long as the patcher holds a registration of it.
    std::vector<u8> program;
    bool failed = false;
    /// Registered with the patcher the first time a pipeline binds it (render thread).
    SceGxmShaderPatcherId patcher_id{};
    /// Vertex programs generated from PICA bytecode: the input registers the program reads,
    /// which are the only attributes a pipeline may name (the patcher refuses the rest).
    u32 input_regs{};
    /// Draws bound with this program, and whether its second tier has been asked for.
    u32 draws{};
    bool tier2_queued = false;
    int tier = 1;

    [[nodiscard]] const SceGxmProgram* Program() const {
        return program.empty() ? nullptr : reinterpret_cast<const SceGxmProgram*>(program.data());
    }

    /// A uniform or attribute by name, looked up once (render thread).
    [[nodiscard]] const SceGxmProgramParameter* Param(const char* name) {
        const auto it = params.find(name);
        if (it != params.end()) {
            return it->second;
        }
        const SceGxmProgramParameter* p =
            Program() != nullptr ? sceGxmProgramFindParameterByName(Program(), name) : nullptr;
        params.emplace(name, p);
        return p;
    }

private:
    std::unordered_map<std::string, const SceGxmProgramParameter*> params;
};

struct VertexAttribute {
    const char* name; ///< the Cg parameter name, resolved to a register at build time
    u16 offset;
    SceGxmAttributeFormat format;
    u8 components;
    u8 stream; ///< which of the pipeline's vertex streams carries it
};

struct PipelineInfo {
    static constexpr u32 MaxAttributes = SCE_GXM_MAX_VERTEX_ATTRIBUTES;
    static constexpr u32 MaxStreams = 12; ///< one per PICA vertex loader at most
    Shader* vertex{};
    Shader* fragment{};
    u32 stream_count{};
    std::array<u16, MaxStreams> strides{};
    u32 attribute_count{};
    std::array<VertexAttribute, MaxAttributes> attributes{};
    bool blend_enabled{};
    SceGxmBlendInfo blend{};
    SceGxmOutputRegisterFormat output_format = SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4;

    [[nodiscard]] u64 Hash() const;
};

class Pipeline : public BuildFlag {
public:
    explicit Pipeline(const PipelineInfo& info);
    ~Pipeline();

    /// True when the pipeline exists and may be bound. Everything it needs is built on the
    /// calling (render) thread, so this only ever fails for a program the emitters refused.
    bool TryBuild();

    [[nodiscard]] SceGxmVertexProgram* VertexProgram() const {
        return vertex_program;
    }
    [[nodiscard]] SceGxmFragmentProgram* FragmentProgram() const {
        return fragment_program;
    }

private:
    bool Build();

    PipelineInfo info;
    SceGxmVertexProgram* vertex_program{};
    SceGxmFragmentProgram* fragment_program{};
    bool failed = false;
};

class PipelineCache {
public:
    PipelineCache();
    ~PipelineCache();

    /**
     * The shader for `key` if one already exists, null otherwise.
     *
     * Worth having separately from UseBuiltinShader because building a configuration and
     * emitting its program are not free, and its arguments would be evaluated whether or not
     * the shader turns out to be new. Only the first draw of a configuration should pay.
     */
    [[nodiscard]] Shader* FindShader(u64 key) const;

    /// A shader from a program already in memory (the USSE emitters'): done the moment it
    /// is made, never on disk. `input_regs`: the input registers a generated vertex program
    /// reads (see Shader).
    Shader* UseBuiltinShader(u64 key, std::string name, std::span<const u8> program,
                             u32 input_regs = 0);

    /// A better program for `key` (the second tier's): a new Shader takes the key, the old
    /// one stays alive for the pipelines that hold it. Returns the new Shader.
    Shader* Replace(u64 key, std::string name, std::span<const u8> program, u32 input_regs);

    /// Binds the pipeline for `info` on the context, building it if needed. False means the
    /// draw must be skipped: a shader failed the program check. Render thread only.
    bool BindPipeline(const PipelineInfo& info);

    /// The context forgets bound programs across scenes; call at each scene start so the
    /// next bind is not skipped as redundant.
    void InvalidateBinding() {
        current_pipeline = nullptr;
    }

    /// The pipeline BindPipeline last put on the context. A default uniform buffer belongs to
    /// the fragment program it was reserved against, and a blend variant is a different
    /// fragment program, so this is what a cached reservation has to be keyed on.
    [[nodiscard]] const Pipeline* CurrentPipeline() const noexcept {
        return current_pipeline;
    }

    [[nodiscard]] std::size_t ShaderCount() const {
        return shaders.size();
    }
private:
    std::unordered_map<u64, std::unique_ptr<Shader>> shaders;
    std::vector<std::unique_ptr<Shader>> retired; ///< replaced programs, still bound by pipelines
    std::unordered_map<u64, std::unique_ptr<Pipeline>> pipelines;
    Pipeline* current_pipeline{};
};

} // namespace GxmRenderer
