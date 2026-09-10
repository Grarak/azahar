// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "common/named_thread.h"
#include "common/wake_flag.h"
#include "video_core/renderer_gxm/usse/vs_usse_gen.h"
#include "video_core/shader/generator/pica_fs_config.h"

namespace GxmRenderer {

/**
 * The second tier: programs that draw a lot are emitted again with the IR's optimising
 * passes (usse_ir level 2) on a worker pinned to core 3 at the lowest priority, and the
 * render thread swaps the better program in at the end of a frame. The first tier's
 * program serves until then, so no draw ever waits.
 *
 * The worker is where SceShaccCg used to run, and that stalled the GPU for seconds at a
 * time; an emission takes milliseconds, and it only runs for programs past the draw
 * threshold, so the core stays mostly libgxm's.
 */
class ShaderTier2 {
public:
    ShaderTier2();
    ~ShaderTier2();

    void QueueFragment(u64 key, const Pica::Shader::FSConfig& config);
    /// Copies the program code and swizzle data the request points at.
    void QueueVertex(u64 key, const Usse::VsRequest& request);

    struct Result {
        u64 key;
        bool vertex;
        std::vector<u8> program; ///< empty when level 2 refused (the first tier stays)
        u32 input_regs;
        std::string refusal;
    };
    /// The programs finished since the last call. Render thread.
    std::vector<Result> Take();

    [[nodiscard]] std::size_t Pending() const;

private:
    struct Job {
        u64 key;
        bool vertex;
        std::optional<Pica::Shader::FSConfig> fs;
        Usse::VsRequest vs;
        std::vector<u32> code;
        std::vector<u32> swizzle;
    };
    void Loop(std::stop_token stop);

    mutable std::mutex mutex;
    std::deque<Job> jobs;
    std::vector<Result> results;
    Common::WakeFlag wake;
    Common::NamedThread thread;
};

} // namespace GxmRenderer
