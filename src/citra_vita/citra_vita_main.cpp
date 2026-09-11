// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstddef>
#include <exception>
#include <memory>
#include <pthread.h>
#include <string>
#include <vector>

#include <psp2/appmgr.h>
#include <psp2/ctrl.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/power.h>
#include <psp2/shellutil.h>
#include "citra_vita/gxm_present.h"

#include "citra_vita/emu_window_vita.h"
#include "citra_vita/input_factory_vita.h"

#include "common/named_thread.h"
#include "common/pipeline_stats.h"
#include "core/arm/vita/native_stats.h"
#include "azahar_native.h"

#include <cstdint>
#include <cstdio>
#include <malloc.h>
#include <psp2/io/stat.h>

namespace {

// One record per second, appended to ux0:data/azahar/perf.bin; read the file back with
// src/citra_vita/tools/read_perf.py. The console keeps only the speed line - everything
// else lives here, where a run's whole history survives instead of scrolling away.
#pragma pack(push, 1)
struct PerfFileHeader {
    char magic[4]; // "AZPF"
    std::uint32_t version;
    std::uint32_t record_size;
};
struct PerfRecord {
    std::uint64_t time_us;
    float speed;
    float fps;
    std::uint32_t slices;
    std::uint32_t svcs;
    std::uint64_t guest_ns;
    std::uint32_t gpu_ops;
    std::uint32_t gpu_brakes;
    std::uint64_t gpu_busy_ns;
    std::uint64_t runloop_us;
    /// Wall time inside the azaharRun syscall. Minus guest_ns it is what entering and leaving
    /// the guest costs: the syscall itself, the two world switches and their TLB flushes.
    std::uint64_t vanrun_us;
    std::uint64_t present_us;
    AzaharPmuStats pmu[4]; // zeros when the build carries no CITRA_VITA_PMU
    // Version 2: the emulation thread's HLE and RunSync-drain time, and the GPU thread's
    // swap-op and stripe-barrier time - the second-level decomposition of the two "busy
    // asleep" totals above it.
    std::uint64_t svc_us;
    std::uint64_t drain_us;
    std::uint32_t drain_count;
    std::uint32_t barrier_count;
    std::uint64_t gpu_swap_ns;
    std::uint64_t barrier_ns;
    // Version 4: how long a completion interrupt sits in the render thread's queue before it
    // is posted - what a guest waiting on a fill or transfer has to sit through.
    std::uint64_t irq_lag_us;
    std::uint32_t irq_count;
    std::uint32_t pad4;
};
#pragma pack(pop)
// The reader (tools/read_perf.py) parses this layout by hand; keep the two in lockstep.
static_assert(sizeof(PerfFileHeader) == 12);
static_assert(sizeof(PerfRecord) == 384);

FILE* OpenPerfFile() {
    sceIoMkdir("ux0:data/azahar", 0777);
    FILE* f = std::fopen("ux0:data/azahar/perf.bin", "wb");
    if (f != nullptr) {
        const PerfFileHeader hdr{{'A', 'Z', 'P', 'F'}, 5u, sizeof(PerfRecord)};
        std::fwrite(&hdr, sizeof(hdr), 1, f);
        std::fflush(f);
    }
    return f;
}

} // namespace
#include "citra_vita/hud_stats.h"
#include "citra_vita/vita_ui.h"
#include "common/file_util.h"
#include "common/logging/backend.h"
#include "common/logging/filter.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/applets/default_applets.h"
#include "core/frontend/image_interface.h"
#include "video_core/gpu.h"
#include "video_core/renderer_gxm/gxm_flags.h"
#include "pte_sema.h"
#include "core/hle/service/service.h"
#include "core/movie.h"

// The default heap is nowhere near enough for a 3DS: FCRAM alone is 128 MB, and the caches and
// the surface storage sit on top of it. Newlib reads this at startup.
extern "C" {
// Old-3DS FCRAM is 128 MB (memory.cpp), so the runtime fits in ~200 MB and this leaves room
// in the game partition - about 238 MB of USER_RW, measured with sceKernelGetFreeMemorySize -
// for the thread stacks and the GPU allocations that come out of it alongside this.
// Sized against the extended-memory budget (ATTRIBUTE2=12 in the param.sfo, +109 MB). The
// guest's 128 MB FCRAM and 6 MB VRAM are ordinary mallocs inside this heap and the emulator's
// working set sits on top, which measured a 207 MB peak - so the old 210 MB ran out, and a
// level load's transient took it over into std::bad_alloc.
//
// Not more than this, though: newlib takes the whole heap as one memblock at startup and the
// kernel commits real pages for it, so every megabyte here is a megabyte the *system* cannot
// have. At 300 MB, 100 MB of it never allocated, the console had trouble the moment it wanted
// to start its own services. 240 MB clears the measured peak by ~33 MB and leaves the rest.
unsigned int _newlib_heap_size_user = 240 * 1024 * 1024;

// Default stack for every pthread, and so for every std::thread: the GPU thread, the two
// rasterizer workers, the service threads. vitasdk's built-in default is 32 KB (the fallback
// constant in libpthread's create.o), which the renderer and the savestate serializer overrun.
// std::thread offers no way to ask for a bigger stack, so this weak global is the only lever;
// the emulation thread sets its own, larger stack through pthread_attr_setstacksize in main().
unsigned int _pthread_stack_default_user = 1024 * 1024;

// newlib's lazy network bring-up (libc/sys/vita/vitanet.c). Not declared in any header, but a
// global symbol, and guarded by its own atomic once-flag, so calling it twice costs nothing.
int _vita_net_init(void);
}

namespace {

/// The process's own arguments, kept for LaunchParameter.
std::vector<std::string> g_arguments;

// Startup trace, to the kernel's printf channel (visible over the module logger / a PC). The
// app was failing to launch with no on-screen detail; this says how far it got.
#define VTRACE(...) do { sceClibPrintf("[azahar] " __VA_ARGS__); sceClibPrintf("\n"); } while (0)

#ifdef CITRA_VITA_PMU
/// The kernel module's counters as one string: SGI service windows, the last pending mask,
/// windows that timed out, what pends now, the resident loop's heartbeat, and the core-2
/// eviction sweep (threads moved off the core, threads pinned to it alone, threads last run
/// on it). Empty when the module does not answer.
void FormatSgiStats(char* out, std::size_t size) {
    out[0] = '\0';
    AzaharSgiStats st{};
    if (azaharSgiStats(&st) == 0) {
        sceClibSnprintf(out, size, "  guest cpu%u  sgi w%u p%04x s%u pend%04x hb%u", st.core,
                        st.windows, st.last_pend, st.stuck, st.pending_now, st.heartbeat);
    }
}

/// Appended to the render-queue-stuck log (GpuThread::Enqueue), once a second while it lasts:
/// the emulation thread is parked at the brake then and its own per-second line stops.
void StuckReport() {
    char sgi[128];
    FormatSgiStats(sgi, sizeof(sgi));
    const struct mallinfo heap = mallinfo();
    sceClibPrintf("[azahar] stuck: heap %u/%u KiB%s\n", static_cast<unsigned>(heap.uordblks / 1024),
                  static_cast<unsigned>(heap.arena / 1024), sgi);
    // Where every thread is, into native.txt: the render thread's pc is the question.
    static int dumps = 0;
    if (dumps < 3) {
        dumps++;
        sceClibPrintf("[azahar] stuck: azaharThreadDump wrote %d threads to native.txt\n", azaharThreadDump());
    }
}
#endif

#ifdef VITA_ALLOC_TRACE
extern "C" void VitaAllocTraceDump(void);
extern "C" void VitaAllocTraceWatchThread(void);
#else
// So the call sites do not each need a guard: with the tracer off this folds away to nothing.
inline void VitaAllocTraceDump() {}
inline void VitaAllocTraceWatchThread() {}
#endif

constexpr char UserDirectory[] = "ux0:/data/azahar/";

/// Opens the pause menu alongside the PS button: a fallback for the case where the shell
/// refuses the lock and the PS button therefore never reaches the pad data. Select plus start
/// costs the guest nothing because no title needs both at once.
constexpr u32 MenuCombo = SCE_CTRL_SELECT | SCE_CTRL_START;

void SetupPaths() {
    // Must happen before anything asks for a path: the first request bootstraps the whole table
    // from whatever the platform default is, and on this console there is no such thing.
    FileUtil::SetUserPath(UserDirectory);
}

/// Names the exception behind an abort. std::terminate goes to newlib's abort(), which raises
/// SIGABRT - the crash dump then points at _kill_r and says nothing about the cause, and azahar's
/// own ASSERT is not it (that traps). std::bad_alloc and the std::system_error that a failing
/// pthread call turns into both arrive here.
void SetupTerminateHandler() {
    std::set_terminate([] {
        std::string what = "no active exception";
        if (std::current_exception()) {
            try {
                std::rethrow_exception(std::current_exception());
            } catch (const std::exception& e) {
                what = e.what();
            } catch (...) {
                what = "an exception not derived from std::exception";
            }
        }
        // Straight to the kernel channel as well as the log: whatever broke may have been the
        // logging thread, and this is the last thing the process does.
        sceClibPrintf("[azahar] std::terminate: %s\n", what.c_str());
        LOG_CRITICAL(Frontend, "std::terminate: {}", what);
        Common::Log::Stop();
        sceKernelExitProcess(1);
    });
}

/// Brings the network stack up now rather than leaving it to the first guest socket call.
///
/// newlib initialises it lazily: socket(), inet_pton(), gethostbyname() and friends all route
/// through _vita_net_init, which loads SCE_SYSMODULE_NET, takes ~140 KB for a net memory pool
/// and calls sceNetInit/sceNetCtlInit. Loading that sysmodule drags in SceNet, SceLibNetCtl,
/// SceLibHttp, SceLibSsl, SceNpManager, SceNpCommon, ScePaf, SceSqliteVsh, SceDbutil and
/// SceDrmPsmKdc - about ten system modules, arriving in the middle of a title the moment its
/// own soc:U or http:C service first touches a host socket, which is exactly where a level
/// load was seen to fall over. Paying for it here means it happens once, before the guest
/// exists, at a moment when nothing is timing-critical and the heap is nearly empty.
void SetupNetwork() {
    const int ret = _vita_net_init();
    if (ret < 0) {
        // Not fatal: a console with no network configured still emulates fine, and the guest's
        // network services will fail the same way they would have anyway.
        LOG_WARNING(Frontend, "network stack unavailable ({:#010x}); guest sockets will fail",
                    static_cast<u32>(ret));
    }
}

void SetupLogging() {
#ifdef VITA_DIAGNOSTICS
    Common::Log::Initialize();
    Common::Log::Filter filter;
    filter.ParseFilterString("*:Info");
    Common::Log::SetGlobalFilter(filter);
#endif
    // The release build starts no logging backend at all, so every LOG_* macro reaches a
    // filter that discards it before it formats anything.
}

/// The Vita boots at a conservative clock; a software-rendered 3DS needs every megahertz.
void SetupClocks() {
    scePowerSetArmClockFrequency(444);
    scePowerSetBusClockFrequency(222);
    scePowerSetGpuClockFrequency(222);
    scePowerSetGpuXbarClockFrequency(166);
}

std::string DescribeLoadError(Core::System::ResultStatus status) {
    switch (status) {
    case Core::System::ResultStatus::ErrorGetLoader:
        return "no loader for this file";
    case Core::System::ResultStatus::ErrorSystemMode:
        return "could not read the system mode";
    case Core::System::ResultStatus::ErrorLoader_ErrorEncrypted:
        return "the title is encrypted";
    case Core::System::ResultStatus::ErrorLoader_ErrorInvalidFormat:
        return "unsupported file format";
    case Core::System::ResultStatus::ErrorLoader_ErrorGbaTitle:
        return "GBA titles are not supported";
    case Core::System::ResultStatus::ErrorNotInitialized:
        return "the CPU could not start: load the azahar-native kernel module (ux0:tai/config.txt) "
               "and reboot";
    default:
        return "unknown error";
    }
}

/// Stops the render thread, releases the GPU's view of guest memory, and shuts the system
/// down - in that order, because the unmap must run while guest RAM is still allocated and
/// while no other thread is inside the GXM context.
void ShutdownSystem(Core::System& system) {
    system.GPU().StopRenderThread();
    VitaFrontend::GxmPresent::UnmapAllGuestMemory();
    system.Shutdown();
}

/// Runs one title to completion. Returns false when the user wants the application to close
/// rather than to return to the game list.
bool RunTitle(Core::System& system, VitaFrontend::EmuWindowVita& window, VitaFrontend::Ui& ui,
              const std::string& path) {
    VTRACE("RunTitle: %s", path.c_str());
    ui.SetStatusMessage("Loading...");
    ui.Update();
    window.Present(&ui);

    VTRACE("RunTitle: system.Load");
    const auto load_result = system.Load(window, path);
    VTRACE("RunTitle: system.Load -> %d", static_cast<int>(load_result));
    VitaAllocTraceDump(); // what the load itself allocated
    if (load_result != Core::System::ResultStatus::Success) {
        LOG_CRITICAL(Frontend, "Could not load {}: {}", path, DescribeLoadError(load_result));
        ui.SetStatusMessage("Could not load this title: " + DescribeLoadError(load_result));
        for (int i = 0; i < 240; i++) {
            window.PollEvents();
            ui.Update();
            window.Present(&ui);
        }
        ui.ClearStatusMessage();
        return true;
    }
    ui.ClearStatusMessage();

    // The window presents straight out of guest memory, so the renderer can skip its
    // screen_infos conversion entirely.
    system.GPU().Renderer().SetFrontendPresentsGuestMemory(true);
    // Bring-up switch: frameskip drops top-screen draws when the render thread is behind,
    // which is indistinguishable from a renderer that draws nothing until it is turned off.
    system.GPU().SetFrameskipAllowed(!GxmRenderer::GxmFlag("noskip"));
    // Bring-up switch: every operation the emulation thread queues is finished before it goes
    // on, so nothing the guest writes can change under a draw the render thread has not run
    // yet. A picture that comes right this way was being raced.
    system.GPU().SetRenderThreadLockstep(GxmRenderer::GxmFlag("lockstep"));

    // ux0:data/azahar/autoload.txt names a savestate slot to restore before the render thread
    // starts, the one moment a load is accepted (saves and loads are refused once the thread
    // runs). The signal executes at a quiescent point inside RunLoop, so it gets a few slices.
    // States are the same boost archive on every 32-bit build, so a state the pi5 harness
    // saved in a fight, a results screen or a level lands here without the menus.
    if (FILE* f = std::fopen("ux0:data/azahar/autoload.txt", "rb")) {
        char buf[16]{};
        std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        const int slot = std::atoi(buf);
        // The states come from the pi5 harness, saved at whatever commit it was built from;
        // the archive layout is what has to match, not the revision.
        system.SetLoadStateAnyBuild(true);
        if (slot >= 0 && system.SendSignal(Core::System::Signal::Load, static_cast<u32>(slot))) {
            LOG_INFO(Frontend, "autoload.txt: restoring savestate slot {}", slot);
            // The load runs inside RunLoop once the kernel has no asynchronous HLE request in
            // flight; at this point in the boot that is immediate. Requests are run inline
            // while the load is pending so a title that streams from its ROM (Smash) cannot
            // keep one in flight, and a failed load is reported rather than played over.
            const bool async_before = Settings::values.deterministic_async_operations.GetValue();
            Settings::values.deterministic_async_operations = true;
            // The signal becomes a pending request inside the first RunLoop, so the loop
            // always runs at least once (a pending-only condition ran zero times, and the
            // load was then refused under the render thread).
            for (int i = 0; i < 200000; i++) {
                const auto status = system.RunLoop();
                if (status != Core::System::ResultStatus::Success) {
                    LOG_ERROR(Frontend, "autoload.txt: load of slot {} failed: {} ({})", slot,
                              static_cast<int>(status), system.GetStatusDetails());
                    break;
                }
                if (!system.IsSaveStateRequestPending()) {
                    break;
                }
            }
            Settings::values.deterministic_async_operations = async_before;
            if (system.IsSaveStateRequestPending()) {
                LOG_WARNING(Frontend, "autoload.txt: slot {} never became loadable", slot);
            }
        } else {
            LOG_WARNING(Frontend, "autoload.txt: could not queue a load of slot {}", slot);
        }
    }

    VTRACE("RunTitle: StartRenderThread");
    system.GPU().StartRenderThread();
    ui.EnterGame();
    // From here the GXM presentation layer belongs to the render thread, which draws the
    // interface along with the frame it has just finished.
    window.SetUi(&ui);
    VTRACE("RunTitle: entering run loop");

    bool keep_running = true;
    bool combo_held = false;
    while (!window.IsQuitRequested()) {
        window.PollEvents();

        // Edge triggered, and the guest must not see the frame in which the combination is
        // pressed - it holds two buttons a title may well be reading. The PS button, locked
        // away from the shell below, arrives as an ordinary pad bit that nothing binds for
        // the guest; it is the same button DSVita pauses on.
        const u32 buttons = window.ButtonsHeld();
        const bool combo_now =
            (buttons & SCE_CTRL_PSBUTTON) != 0 || (buttons & MenuCombo) == MenuCombo;
        if (combo_now && !combo_held && !ui.MenuOpen()) {
            ui.OpenPauseMenu();
        }
        combo_held = combo_now;

        if (ui.MenuOpen()) {
            // Emulation does not advance while a menu is up, so no frames arrive to carry a
            // present: ask the render thread, which owns presentation, for one of its own. Waiting
            // for it costs nothing here, because nothing is running to be held up.
            system.GPU().RunOnRenderThread([&] { window.Present(&ui); });
            const auto action = static_cast<VitaFrontend::Ui::Action>(window.TakeUiAction());
            switch (action) {
            case VitaFrontend::Ui::Action::QuitGame:
                keep_running = true;
                window.SetInputToGuest(true);
                // The render thread is about to stop; take the interface back first, so nothing
                // there reaches for it while it goes.
                window.SetUi(nullptr);
                ShutdownSystem(system);
                return keep_running;
            case VitaFrontend::Ui::Action::ExitApplication:
                window.SetUi(nullptr);
                ShutdownSystem(system);
                return false;
            case VitaFrontend::Ui::Action::SaveState:
                system.SendSignal(Core::System::Signal::Save, ui.SelectedSlot());
                ui.EnterGame();
                break;
            case VitaFrontend::Ui::Action::LoadState:
                system.SendSignal(Core::System::Signal::Load, ui.SelectedSlot());
                ui.EnterGame();
                break;
            default:
                break;
            }
            continue;
        }

        // Periodic snapshot on the emulation thread, never from the allocation path.
        static unsigned van_trace_tick = 0;
        if ((++van_trace_tick % 600u) == 0u) {
            VitaAllocTraceDump();
        }
        const SceUInt64 runloop_start = sceKernelGetProcessTimeWide();
        const auto status = system.RunLoop();
        Core::NativeStats::runloop_us.fetch_add(sceKernelGetProcessTimeWide() - runloop_start,
                                                std::memory_order_relaxed);
        if (status == Core::System::ResultStatus::ErrorSavestate) {
            // A refused save or load is not fatal; the title keeps running.
            LOG_ERROR(Frontend, "Savestate operation failed");
        } else if (status != Core::System::ResultStatus::Success) {
            LOG_CRITICAL(Frontend, "Emulation stopped: {}", static_cast<int>(status));
            break;
        }

        // Once a second: the speed line to the console, the full record - GPU health, the
        // emulation thread's time split, all four cores' PMU counters - to the perf blob.
        // Everything measured once a second, and where it is printed. Compiled in only
        // for the diagnostic build: the release build takes no clock readings, drains no
        // counters, and prints nothing.
        {
            static SceUInt64 last_perf_us = 0;
            const SceUInt64 now_us = sceKernelGetProcessTimeWide();
            if (now_us - last_perf_us >= 1000000ull) {
                last_perf_us = now_us;
                const auto perf = system.GetAndResetPerfStats();
                const int speed = static_cast<int>(perf.emulation_speed * 100.0 + 0.5);
                const int fps10 = static_cast<int>(perf.game_fps * 10.0 + 0.5);

                const unsigned presented =
                                    Common::PipelineStats::presents.exchange(0, std::memory_order_relaxed);

#ifdef VITA_DIAGNOSTICS
                // Frames the guest ran whose draws were dropped: the difference between
                // this and the game's frame rate is what the renderer actually drew, which
                // is the number to read against the presented rate.
                static u64 skipped_total = 0;
                const u64 skipped_now = system.GPU().SkippedFrames();
                const unsigned skipped = static_cast<unsigned>(skipped_now - skipped_total);
                skipped_total = skipped_now;
                static FILE* const perf_file = OpenPerfFile();

                PerfRecord rec{};
                rec.time_us = now_us;
                rec.speed = static_cast<float>(perf.emulation_speed);
                rec.fps = static_cast<float>(perf.game_fps);
                rec.slices = Core::NativeStats::slices.exchange(0);
                rec.svcs = Core::NativeStats::svcs.exchange(0);
                rec.guest_ns = Core::NativeStats::credited_ns.exchange(0);
                rec.gpu_ops = Common::PipelineStats::gpu_ops.exchange(0);
                rec.gpu_brakes = Common::PipelineStats::gpu_brakes.exchange(0);
                rec.gpu_busy_ns = Common::PipelineStats::gpu_busy_us.exchange(0) * 1000ull;
                rec.runloop_us = Core::NativeStats::runloop_us.exchange(0);
                rec.vanrun_us = Core::NativeStats::vanrun_us.exchange(0);
                rec.present_us = Core::NativeStats::present_us.exchange(0);
                rec.svc_us = Core::NativeStats::svc_us.exchange(0);
                rec.drain_us = Common::PipelineStats::drain_us.exchange(0);
                rec.drain_count = Common::PipelineStats::drain_count.exchange(0);
                rec.gpu_swap_ns = Common::PipelineStats::gpu_swap_us.exchange(0) * 1000ull;
                rec.barrier_ns = Common::PipelineStats::barrier_us.exchange(0) * 1000ull;
                rec.barrier_count = Common::PipelineStats::barrier_count.exchange(0);
                rec.irq_lag_us = Common::PipelineStats::interrupt_lag_us.exchange(0);
                rec.irq_count = Common::PipelineStats::interrupt_count.exchange(0);
#ifdef CITRA_VITA_PMU
                {
                    AzaharPmuAll pmu;
                    if (azaharPmuReadAll(&pmu) == 0) {
                        for (unsigned c = 0; c < 4; c++) {
                            rec.pmu[c] = pmu.core[c];
                        }
                    }
                }
#endif
                // sceClibPrintf has no reliable float formatting, so fixed-point by hand.
                // Live heap, because a run that lasts long enough dies in std::bad_alloc and
                // the rate of climb here is the only thing that says what is leaking. uordblks
                // is what newlib has handed out and not got back; arena is what it took from
                // the kernel, which only grows.
                const struct mallinfo heap = mallinfo();
                // Both empty unless the Vita tracing is built in.
                char sema[64] = {};
                char phase[128] = {};
                // azahar-native's SGI service windows and core 2's heartbeat: a freeze log's last
                // second says whether core 2 was still polling between commands, whether a
                // cross-core call was pending, and whether a window ever timed out.
                // Like azaharPmuReadAll, an import only a module new enough exports, so it sits
                // behind the same VITA_PMU option (under Vita3K, which has no module, the
                // import resolves to a stub that returns 0 and the fields print as zeros).
                char sgi[128] = {};
#ifdef CITRA_VITA_PMU
                FormatSgiStats(sgi, sizeof(sgi));
#endif
#ifdef VITA_ALLOC_TRACE
                unsigned sema_created = 0, sema_deleted = 0, sema_refused = 0;
                VitaSemaTotals(&sema_created, &sema_deleted, &sema_refused);
                sceClibSnprintf(sema, sizeof(sema), "  sema %d/%d (%u-%u)", VitaSemaLive(),
                                VitaSemaPeak(), sema_created, sema_deleted);
#endif
#ifdef VITA_RENDER_TRACE
                // The render thread's breadcrumb. It hangs without printing anything itself,
                // and this line keeps coming for the few seconds before the queue reaches its
                // ceiling and the brake parks the emulation thread too - so a phase that stops
                // advancing while its age climbs names the call that never returned.
                {
                    const char* const where =
                        Common::PipelineStats::render_phase.load(std::memory_order_relaxed);
                    const u64 since =
                        Common::PipelineStats::render_phase_us.load(std::memory_order_relaxed);
                    const u64 now = Common::PipelineStats::NowUs();
                    sceClibSnprintf(phase, sizeof(phase), "  render %s %llu us [%llu %llu %llu]",
                                    where, since != 0 && now > since ? now - since : 0,
                                    Common::PipelineStats::render_phase_a.load(
                                        std::memory_order_relaxed),
                                    Common::PipelineStats::render_phase_b.load(
                                        std::memory_order_relaxed),
                                    Common::PipelineStats::render_phase_c.load(
                                        std::memory_order_relaxed));
                }
#endif
                const auto wt_stats = system.Memory().WriteTracker().TakeStats();
                sceClibPrintf("[azahar] speed %d%%  game %d.%d fps  shown %u fps  skipped %u  "
                              "slices %u (svc %u)  guest %u us  in %u us  wf %u wa %u wh %u  "
                              "queue %u brake %u/%u us  heap %u/%u KiB%s%s%s\n",
                              speed, fps10 / 10, fps10 % 10, presented, skipped, rec.slices,
                              rec.svcs, static_cast<unsigned>(rec.guest_ns / 1000),
                              static_cast<unsigned>(rec.vanrun_us),
                              Core::NativeStats::write_faults.exchange(0, std::memory_order_relaxed),
                              static_cast<unsigned>(wt_stats.arms),
                              static_cast<unsigned>(wt_stats.hot_skips),
                              system.GPU().RenderQueueDepth(), rec.gpu_brakes,
                              static_cast<unsigned>(rec.gpu_brake_us),
                              static_cast<unsigned>(heap.uordblks / 1024),
                              static_cast<unsigned>(heap.arena / 1024), sema, phase, sgi);
                if (perf_file != nullptr) {
                    std::fwrite(&rec, sizeof(rec), 1, perf_file);
                    std::fflush(perf_file);
                }
#endif // VITA_DIAGNOSTICS
                // The stats window reads these on the render thread.
                {
                    using namespace VitaFrontend::HudStats;
                    speed_percent.store(static_cast<u32>(std::max(speed, 0)), std::memory_order_relaxed);
                    game_fps10.store(static_cast<u32>(std::max(fps10, 0)), std::memory_order_relaxed);
                    shown_fps.store(presented, std::memory_order_relaxed);
                    valid.store(true, std::memory_order_release);
                }
            }
        }
        // Nothing to present from here: a finished frame is presented by the render thread
        // where it arrives (EmuWindowVita::SwapBuffers), so this thread never waits on one.
    }

    window.SetUi(nullptr);
    ShutdownSystem(system);
    return keep_running;
}

/**
 * The title to start without going through the list, if the launcher named one.
 *
 * Two ways in, because there are two launchers. A LiveArea bubble or another application
 * starts this one with a `psgm:play?titleid=...&param=<path>` string, which arrives through
 * sceAppMgrGetAppParam. The Vita3K emulator, where the rendering is debugged off-console,
 * passes its --app-args as the process's own arguments instead. Empty when neither did:
 * then the list comes up as always.
 */
std::string LaunchParameter() {
    if (!g_arguments.empty()) {
        return g_arguments.front();
    }
    char param[1024]{};
    if (sceAppMgrGetAppParam(param) != 0) {
        return {};
    }
    const std::string_view text{param};
    if (text.find("psgm:play") == std::string_view::npos) {
        return {};
    }
    const std::size_t at = text.find("&param=");
    if (at == std::string_view::npos) {
        return {};
    }
    return std::string{text.substr(at + 7)};
}

/// The emulator proper. Runs on a thread of its own; see main.
int EmulatorMain() {
    // Register this stack with the allocation tracer's overflow watch; see alloc_trace.c.
    VitaAllocTraceWatchThread();
    // This thread drives the guest (which itself runs on the detached core 2, outside the
    // system scheduler) and feeds the render thread; RoleCore in common/thread.cpp lays out
    // which of the three schedulable cores each host thread gets.
    Common::SetCurrentThreadRole(Common::ThreadRole::Emulation);

    VTRACE("EmulatorMain: clocks");
    SetupClocks();
    VTRACE("EmulatorMain: paths");
    SetupPaths();
    VTRACE("EmulatorMain: logging");
    SetupLogging();
    VTRACE("EmulatorMain: network");
    SetupNetwork();

    VTRACE("EmulatorMain: GxmPresent::Initialize");
    if (!VitaFrontend::GxmPresent::Initialize()) {
        VTRACE("EmulatorMain: GxmPresent::Initialize FAILED");
        LOG_CRITICAL(Frontend, "Could not bring up GXM");
        return -1;
    }
    // Presents never block this thread in a vblank wait: the display queue's own thread
    // does the flip and the wait, and submission only blocks past two pending flips - which
    // the run loop's own pacing keeps from happening.
    // The PS button would otherwise hand the display to the home menu, which this process does
    // not survive once a core is taken; locked, its presses appear in the pad data instead, and
    // the run loop opens the pause menu on them. The quick menu lock keeps the power slider out
    // from under a running title. Both are held for the whole session, because the core, once
    // taken, is taken for the whole session.
    sceShellUtilInitEvents(0);
    sceShellUtilLock(static_cast<SceShellUtilLockType>(SCE_SHELL_UTIL_LOCK_TYPE_PS_BTN |
                                                       SCE_SHELL_UTIL_LOCK_TYPE_PS_BTN_2 |
                                                       SCE_SHELL_UTIL_LOCK_TYPE_QUICK_MENU));

    VTRACE("EmulatorMain: shell lock");
    auto& system = Core::System::GetInstance();
    VTRACE("EmulatorMain: got System instance");

    // The Vita has neither the memory nor the drivers for the LLE service modules. GXM is the
    // only renderer in this build: the software renderer was too slow on this
    // CPU to be worth carrying and is no longer compiled for the console.
    for (const auto& service_module : Service::service_module_map) {
        Settings::values.lle_modules.emplace(service_module.name, false);
    }
    Settings::values.graphics_api = Settings::GraphicsAPI::Gxm;
    Settings::values.use_cpu_jit = false;
    Settings::values.disable_right_eye_render = true;
    VTRACE("EmulatorMain: applets");
    Frontend::RegisterDefaultApplets(system);
    system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());

    VTRACE("EmulatorMain: EmuWindowVita");
    VitaFrontend::EmuWindowVita window(system);
    VTRACE("EmulatorMain: InitInput");
    // Not InputCommon::Init(): every factory it registers is one this frontend does not use.
    // The buttons and the circle pad come from the "vita" engine above, the touchscreen from
    // EmuWindow's own "emu_window" factory, and nothing here sets a motion device. What it would
    // add is a Cemuhook UDP client - a socket and a thread of its own, talking to a desktop that
    // is not there - and factories for a keyboard and a GameCube adapter this console has not
    // got. SDL is a null state in this build either way.
    VitaFrontend::InitInput(window);

    VTRACE("EmulatorMain: Ui");
    VitaFrontend::Ui ui(system, window);
    VTRACE("EmulatorMain: entering main loop");

    VitaAllocTraceDump();

    bool running = true;
    if (const std::string launch = LaunchParameter(); !launch.empty()) {
        LOG_INFO(Frontend, "launch parameter names {}", launch);
        running = RunTitle(system, window, ui, launch);
    }
    while (running && !window.IsQuitRequested()) {
        window.PollEvents();
        const auto action = ui.Update();
        window.Present(&ui);

        switch (action) {
        case VitaFrontend::Ui::Action::LaunchGame:
            running = RunTitle(system, window, ui, ui.SelectedGamePath());
            break;
        case VitaFrontend::Ui::Action::ExitApplication:
            running = false;
            break;
        default:
            break;
        }
    }

    VitaFrontend::ShutdownInput();
    Common::Log::Stop();

    sceShellUtilUnlock(static_cast<SceShellUtilLockType>(SCE_SHELL_UTIL_LOCK_TYPE_PS_BTN |
                                                         SCE_SHELL_UTIL_LOCK_TYPE_PS_BTN_2 |
                                                         SCE_SHELL_UTIL_LOCK_TYPE_QUICK_MENU));
    VitaFrontend::GxmPresent::Shutdown();
    return 0;
}

} // Anonymous namespace

int main(int argc, char* argv[]) {
#ifdef CITRA_VITA_PMU
    Common::PipelineStats::stuck_report.store(&StuckReport, std::memory_order_relaxed);
#endif
    // What the partition has left after newlib took the heap. The extended-memory mode
    // (ATTRIBUTE2=12, ~365 MB) is only granted with every system app closed; without it the
    // 240 MB heap overruns the plain partition and newlib silently has no heap at all.
    {
        SceKernelFreeMemorySizeInfo info{};
        info.size = sizeof(info);
        sceKernelGetFreeMemorySize(&info);
        sceClibPrintf("[azahar] free memory after the heap: user %u KiB, cdram %u KiB, phycont %u KiB\n",
                      static_cast<unsigned>(info.size_user / 1024),
                      static_cast<unsigned>(info.size_cdram / 1024),
                      static_cast<unsigned>(info.size_phycont / 1024));
        void* probe = std::malloc(64);
        if (probe == nullptr) {
            sceClibPrintf("[azahar] the %u MB heap could not be allocated. The extended memory "
                          "mode (ATTRIBUTE2=12, 365 MB) is granted only with every system app "
                          "closed: browser, store, mail, content manager, a PSP bubble. Close "
                          "them and start again.\n",
                          static_cast<unsigned>(_newlib_heap_size_user / (1024 * 1024)));
            sceKernelExitProcess(0);
        }
        std::free(probe);
    }
    for (int i = 1; i < argc; ++i) {
        if (argv[i] != nullptr && argv[i][0] != '\0') {
            g_arguments.emplace_back(argv[i]);
        }
    }
    SetupTerminateHandler();
    Common::SetCurrentThreadRole(Common::ThreadRole::Other);
    // The process's own thread gets a stack sized for a homebrew launcher, and the global that
    // is supposed to change that does not work. Everything therefore runs on a thread created
    // here with a stack the emulator can actually use - service handling and the savestate
    // serializer both go deep.
    // A thread stack comes out of the game partition, which measures about 238 MB in total and
    // is mostly spoken for by the newlib heap - so this is worth no more than it needs. Eight
    // megabytes still leaves the savestate serializer, which is the deepest thing that runs
    // here, a wide margin over the megabyte every other thread gets.
    constexpr std::size_t EmulationStackSize = 4 * 1024 * 1024;

    {
        Common::NamedThread emu_thread{Common::ThreadCfg{"emulation", EmulationStackSize},
                                       []() { EmulatorMain(); }};
        emu_thread.join();
    }

    sceKernelExitProcess(0);
    return 0;
}
