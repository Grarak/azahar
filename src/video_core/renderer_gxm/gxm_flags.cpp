// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include "common/logging/log.h"
#include "video_core/renderer_gxm/gxm_flags.h"

namespace GxmRenderer {

namespace {

const std::vector<std::string>& Flags() {
    static const std::vector<std::string> flags = [] {
        std::vector<std::string> out;
        std::string text;
        if (const char* env = std::getenv("AZAHAR_GXM_FLAGS")) {
            // Off the console (the stub libgxm build) the flags come from the environment.
            text = env;
        }
        if (FILE* f = std::fopen("ux0:data/azahar/gxm_flags.txt", "rb")) {
            char buf[256];
            while (std::size_t n = std::fread(buf, 1, sizeof(buf), f)) {
                text.append(buf, n);
            }
            std::fclose(f);
        }
        {
            std::istringstream in(text);
            std::string word;
            while (in >> word) {
                out.push_back(word);
            }
            std::string joined;
            for (const auto& w : out) {
                joined += w + ' ';
            }
            LOG_WARNING(Render, "gxm_flags.txt: {}", joined);
        }
        return out;
    }();
    return flags;
}

} // Anonymous namespace

namespace {
std::atomic<bool> surface_dump_requested{false};
}

void RequestSurfaceDump() {
    surface_dump_requested.store(true, std::memory_order_relaxed);
}

bool SurfaceDumpPending() {
    return surface_dump_requested.load(std::memory_order_relaxed);
}

bool TakeSurfaceDumpRequest() {
    return surface_dump_requested.exchange(false, std::memory_order_relaxed);
}

namespace {
void (*surface_dump_hook)(void*) = nullptr;
void* surface_dump_user = nullptr;
} // namespace

void SetSurfaceDumpHook(void (*hook)(void*), void* user) {
    surface_dump_hook = hook;
    surface_dump_user = user;
}

void RunSurfaceDumpIfRequested() {
    if (surface_dump_hook != nullptr && TakeSurfaceDumpRequest()) {
        surface_dump_hook(surface_dump_user);
    }
}

bool GxmFlag(const char* name) {
    for (const auto& flag : Flags()) {
        if (flag == name) {
            return true;
        }
    }
    return false;
}

} // namespace GxmRenderer
