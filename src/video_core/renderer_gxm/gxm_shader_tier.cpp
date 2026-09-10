// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <utility>
#ifdef __vita__
#include <psp2/kernel/threadmgr.h>
#endif
#include "common/logging/log.h"
#include "common/pipeline_stats.h"
#include "common/thread.h"
#include "video_core/renderer_gxm/gxm_shader_tier.h"
#include "video_core/renderer_gxm/usse/fs_usse_gen.h"

namespace GxmRenderer {

ShaderTier2::ShaderTier2()
    : thread{Common::ThreadCfg{"GxmTier2"}, [this](std::stop_token stop) { Loop(stop); }} {}

ShaderTier2::~ShaderTier2() {
    thread.request_stop();
    wake.RaiseAll();
    thread.join();
}

void ShaderTier2::QueueFragment(u64 key, const Pica::Shader::FSConfig& config) {
    {
        std::scoped_lock lock{mutex};
        Job job{};
        job.key = key;
        job.vertex = false;
        job.fs = config;
        jobs.push_back(std::move(job));
    }
    wake.Raise();
}

void ShaderTier2::QueueVertex(u64 key, const Usse::VsRequest& request) {
    {
        std::scoped_lock lock{mutex};
        Job job{};
        job.key = key;
        job.vertex = true;
        job.vs = request;
        job.code.assign(request.code.begin(), request.code.begin() + request.code_size);
        job.swizzle.assign(request.swizzle.begin(), request.swizzle.end());
        jobs.push_back(std::move(job));
    }
    wake.Raise();
}

std::vector<ShaderTier2::Result> ShaderTier2::Take() {
    std::scoped_lock lock{mutex};
    std::vector<Result> out;
    out.swap(results);
    return out;
}

std::size_t ShaderTier2::Pending() const {
    std::scoped_lock lock{mutex};
    return jobs.size();
}

void ShaderTier2::Loop(std::stop_token stop) {
    // Core 3, the system core where libgxm's threads live too, at the lowest user
    // priority: the worker has the core only while nothing else wants it.
    Common::SetCurrentThreadRole(Common::ThreadRole::ShaderCompiler);
    std::stop_callback wake_on_stop(stop, [this] { wake.RaiseAll(); });
    while (true) {
        Job job;
        {
            std::unique_lock lock{mutex};
            if (jobs.empty()) {
                wake.Lower();
                if (stop.stop_requested()) {
                    return;
                }
                lock.unlock();
                wake.Wait();
                continue;
            }
            if (stop.stop_requested()) {
                return;
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        const u64 start_us = Common::PipelineStats::NowUs();
        Result result{};
        result.key = job.key;
        result.vertex = job.vertex;
        if (job.vertex) {
            job.vs.code = job.code;
            job.vs.swizzle = job.swizzle;
            result.program =
                Usse::EmitVertexProgram(job.vs, result.input_regs, &result.refusal, 2);
        } else {
            result.program = Usse::EmitFragmentProgram(*job.fs, &result.refusal, 2);
        }
        const u64 took_us = Common::PipelineStats::NowUs() - start_us;
        if (!result.refusal.empty()) {
            LOG_WARNING(Render, "tier 2 {} {:016x} refused after {} us: {}",
                        job.vertex ? "vs" : "fs", job.key, took_us, result.refusal);
        } else {
            LOG_DEBUG(Render, "tier 2 {} {:016x}: {} bytes in {} us", job.vertex ? "vs" : "fs",
                      job.key, result.program.size(), took_us);
        }
        std::scoped_lock lock{mutex};
        results.push_back(std::move(result));
    }
}

} // namespace GxmRenderer
