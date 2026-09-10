// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <stop_token>
#include <thread>
#include <utility>

#include "common/thread.h"

namespace Common {

/// Thread configuration: the name shows up in any thread listing (psvshell and friends) on
/// the Vita, where every pthread is otherwise created as an anonymous "pthread".
struct ThreadCfg {
    const char* name;
    std::size_t stack_size = 1024 * 1024;
};

#ifdef __vita__

/// A std::jthread-alike on a raw sceKernelCreateThread thread, so the kernel knows the
/// thread by name. Before the callable runs, the thread attaches itself to newlib's
/// pthread layer (the per-thread record pte's cancellable waits dereference - a bare SCE
/// thread crashed in pte_osSemaphoreCancellablePend without it, 2026-08-30), and it exits
/// through newlib's vita_exit_thread/vita_exit_delete_thread so the lazily created
/// reentrancy state is cleaned up. Supports plain callables and callables taking a
/// std::stop_token, join-on-destruction, request_stop, and detach.
class NamedThread {
public:
    NamedThread() noexcept = default;

    NamedThread(ThreadCfg cfg, std::function<void()> f) {
        Launch(cfg, std::move(f));
    }

    NamedThread(ThreadCfg cfg, std::function<void(std::stop_token)> f) {
        Launch(cfg, [fn = std::move(f), token = ssource.get_token()]() { fn(token); });
    }

    NamedThread(const NamedThread&) = delete;
    NamedThread& operator=(const NamedThread&) = delete;

    NamedThread(NamedThread&& other) noexcept
        : uid{std::exchange(other.uid, -1)}, ctrl{std::exchange(other.ctrl, nullptr)},
          ssource{std::move(other.ssource)} {}

    NamedThread& operator=(NamedThread&& other) noexcept {
        if (this != &other) {
            Finish();
            uid = std::exchange(other.uid, -1);
            ctrl = std::exchange(other.ctrl, nullptr);
            ssource = std::move(other.ssource);
        }
        return *this;
    }

    ~NamedThread() {
        Finish();
    }

    [[nodiscard]] bool joinable() const noexcept {
        return uid >= 0;
    }
    void join();
    void detach();
    bool request_stop() noexcept {
        return ssource.request_stop();
    }
    [[nodiscard]] std::stop_token get_stop_token() const noexcept {
        return ssource.get_token();
    }
    [[nodiscard]] bool IsCurrentThread() const noexcept;

private:
    struct Ctrl;

    static int Trampoline(unsigned int args, void* argp);
    void Launch(const ThreadCfg& cfg, std::function<void()> f);
    void Finish() {
        if (uid >= 0) {
            request_stop();
            join();
        }
    }

    std::int32_t uid = -1;
    Ctrl* ctrl = nullptr;
    std::stop_source ssource;
};

#else

/// Elsewhere the wrapper is a std::jthread that names itself first.
class NamedThread {
public:
    NamedThread() noexcept = default;

    NamedThread(ThreadCfg cfg, std::function<void()> f)
        : inner{[name = cfg.name, fn = std::move(f)]() {
              SetCurrentThreadName(name);
              fn();
          }} {}

    NamedThread(ThreadCfg cfg, std::function<void(std::stop_token)> f)
        : inner{[name = cfg.name, fn = std::move(f)](std::stop_token token) {
              SetCurrentThreadName(name);
              fn(token);
          }} {}

    [[nodiscard]] bool joinable() const noexcept {
        return inner.joinable();
    }
    void join() {
        inner.join();
    }
    void detach() {
        inner.detach();
    }
    bool request_stop() noexcept {
        return inner.request_stop();
    }
    [[nodiscard]] std::stop_token get_stop_token() const noexcept {
        return inner.get_stop_token();
    }
    [[nodiscard]] bool IsCurrentThread() const noexcept {
        return inner.get_id() == std::this_thread::get_id();
    }

private:
    std::jthread inner;
};

#endif

} // namespace Common
