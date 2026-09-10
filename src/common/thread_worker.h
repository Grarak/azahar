// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <queue>

#include "common/logging/log.h"
#include "common/named_thread.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "common/unique_function.h"
#include "common/wake_flag.h"

namespace Common {

template <class StateType = void>
class StatefulThreadWorker {
    static constexpr bool with_state = !std::is_same_v<StateType, void>;

    struct DummyCallable {
        int operator()(std::size_t) const noexcept {
            return 0;
        }
    };

    using Task =
        std::conditional_t<with_state, UniqueFunction<void, StateType*>, UniqueFunction<void>>;
    using StateMaker =
        std::conditional_t<with_state, std::function<StateType(std::size_t)>, DummyCallable>;

public:
    explicit StatefulThreadWorker(std::size_t num_workers, std::string_view name,
                                  StateMaker func = {}, ThreadRole role = ThreadRole::Other)
        : workers_queued{num_workers}, thread_name{name} {
        const auto lambda = [this, func, role](std::stop_token stop_token, std::size_t index) {
            Common::SetCurrentThreadName(thread_name.data());
            Common::SetCurrentThreadRole(role, index);
            // Registered once for the thread's whole life rather than around each wait: a stop
            // request has to break the wait below, and re-arming a callback every time round
            // the loop was pure overhead in the idle case this loop spends its time in.
            std::stop_callback wake_on_stop(stop_token, [this] { work_available.RaiseAll(); });
            {
                [[maybe_unused]] std::conditional_t<with_state, StateType, int> state{func(index)};
                while (!stop_token.stop_requested()) {
                    Task task;
                    {
                        std::unique_lock lock{queue_mutex};
                        if (requests.empty()) {
                            // Lower under the mutex, having just seen the queue empty: a
                            // producer enqueues under this same mutex and raises afterwards,
                            // so its raise cannot land before this lower and be discarded,
                            // while a raise left over from work already taken is.
                            work_available.Lower();
                            drained.RaiseAll();
                            if (stop_token.stop_requested()) {
                                break;
                            }
                            lock.unlock();
                            work_available.Wait();
                            continue;
                        }
                        task = std::move(requests.front());
                        requests.pop();
                    }
                    if constexpr (with_state) {
                        task(&state);
                    } else {
                        task();
                    }
                    ++work_done;
                    drained.RaiseAll();
                }
            }
            ++workers_stopped;
            drained.RaiseAll();
            LOG_DEBUG(Common, "worker {} exiting", thread_name);
        };
        threads.reserve(num_workers);
        for (std::size_t i = 0; i < num_workers; ++i) {
            threads.emplace_back(ThreadCfg{thread_name.data()},
                                 [lambda, i](std::stop_token token) { lambda(token, i); });
        }
    }

    StatefulThreadWorker& operator=(const StatefulThreadWorker&) = delete;
    StatefulThreadWorker(const StatefulThreadWorker&) = delete;

    StatefulThreadWorker& operator=(StatefulThreadWorker&&) = delete;
    StatefulThreadWorker(StatefulThreadWorker&&) = delete;

    void QueueWork(Task work) {
        {
            std::unique_lock lock{queue_mutex};
            requests.emplace(std::move(work));
            ++work_scheduled;
        }
        work_available.Raise();
    }

    /// Runs queued tasks on the calling thread until the queue drains, then waits for the
    /// workers' in-flight tasks. Lets the queueing thread act as an extra worker instead of
    /// sleeping through the batch.
    void HelpAndWait(std::stop_token stop_token = {}) {
        if constexpr (!with_state) {
            while (true) {
                Task task;
                {
                    std::unique_lock lock{queue_mutex};
                    if (requests.empty()) {
                        break;
                    }
                    task = std::move(requests.front());
                    requests.pop();
                }
                task();
                ++work_done;
            }
        }
        WaitForRequests(stop_token);
    }

    void WaitForRequests(std::stop_token stop_token = {}) {
        std::stop_callback callback(stop_token, [this] {
            for (auto& thread : threads) {
                thread.request_stop();
            }
        });
        // Both counters only ever climb, so the condition is monotone: lower, re-read, and
        // only then block. A worker raises after incrementing, so a completion that lands in
        // the gap either shows up in the re-read or leaves the flag up for the wait to find.
        const auto satisfied = [this] {
            return workers_stopped >= workers_queued || work_done >= work_scheduled;
        };
        while (!satisfied()) {
            drained.Lower();
            if (satisfied()) {
                break;
            }
            drained.Wait();
        }
    }

    const std::size_t NumWorkers() const noexcept {
        return threads.size();
    }

private:
    std::queue<Task> requests;
    std::mutex queue_mutex;
    /// Raised by QueueWork, waited on by idle workers.
    WakeFlag work_available;
    /// Raised whenever a task finishes or a worker goes idle or stops; WaitForRequests waits
    /// on it.
    WakeFlag drained;
    std::atomic<std::size_t> work_scheduled{};
    std::atomic<std::size_t> work_done{};
    std::atomic<std::size_t> workers_stopped{};
    std::atomic<std::size_t> workers_queued{};
    std::string_view thread_name;
    std::vector<NamedThread> threads;
};

using ThreadWorker = StatefulThreadWorker<>;

} // namespace Common
