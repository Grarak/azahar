// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <span>
#include <string>
#include <memory>
#include <boost/serialization/access.hpp>

#include "core/hle/service/gsp/gsp_interrupt.h"

namespace Service::GSP {
struct Command;
struct FrameBufferInfo;
} // namespace Service::GSP

namespace Core {
class System;
}

namespace Pica {
class DebugContext;
class PicaCore;
struct RegsLcd;
union ColorFill;
struct DisplayTransferConfig;
} // namespace Pica

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

class RasterizerInterface;

/// Measured on hardware to be 2240568 timer cycles or 4481136 ARM11 cycles
constexpr u64 FRAME_TICKS = 4481136ull;

class GraphicsDebugger;
class RendererBase;
class RightEyeDisabler;
struct DisplaySnapshotSlot;

/**
 * The GPU class is the high level interface to the video_core for core services.
 */
class GPU {
public:
    explicit GPU(Core::System& system, Frontend::EmuWindow& emu_window,
                 Frontend::EmuWindow* secondary_window);
    ~GPU();

    /// Sets the function to call for signalling GSP interrupts.
    void SetInterruptHandler(Service::GSP::InterruptHandler handler);

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(PAddr addr, u32 size);

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(PAddr addr, u32 size);

    /// Flushes and invalidates all memory in the rasterizer cache and removes any leftover state.
    void ClearAll(bool flush);

    /// Executes the provided GSP command.
    void Execute(const Service::GSP::Command& command);

    /// Updates GPU display framebuffer configuration using the specified parameters.
    void SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info);

    /// Sets the LCD color fill configuration for the top and bottom screens.
    void SetColorFill(const Pica::ColorFill& fill);

    /// Reads a word from the GPU virtual address.
    u32 ReadReg(VAddr addr);

    /// Writes the provided value to the GPU virtual address.
    void WriteReg(VAddr addr, u32 data);

    /// Returns a mutable reference to the renderer.
    [[nodiscard]] VideoCore::RendererBase& Renderer();

    /// Returns a mutable reference to the PICA GPU.
    [[nodiscard]] Pica::PicaCore& PicaCore();

    /// Returns an immutable reference to the PICA GPU.
    [[nodiscard]] const Pica::PicaCore& PicaCore() const;

    /// Returns a mutable reference to the GSP command debugger.
    [[nodiscard]] GraphicsDebugger& Debugger();

    RightEyeDisabler& GetRightEyeDisabler() {
        return *right_eye_disabler;
    }

    void ApplyPerProgramSettings(u64 program_ID);

    /// Recreates the renderer (for GL context reset in libretro)
    void RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window);

    /**
     * Moves GPU execution onto a dedicated thread that takes ownership of the window's GL
     * context. The caller must have released the context (DoneCurrent) first. Until this is
     * called every operation runs inline on the emulation thread, exactly as before.
     */
    void StartRenderThread();

    /// Drains and stops the render thread, handing the GL context back released.
    void StopRenderThread();

    /// Runs fn on the render thread and waits for it. For readers of render-thread-owned state
    /// (the renderer's screen buffers) that would otherwise race the frame being produced.
    /// Whether a transfer into this framebuffer is queued and has not run. Presenting such a
    /// buffer shows whatever it held before - for a buffer the game has recycled, that is the
    /// other screen's picture.
    [[nodiscard]] bool IsFillPending(PAddr addr) const;

    /// Sequence number of the most recent completed blit into addr, 0 if none on record.
    /// The software present path uses it to pick a screen's freshest candidate buffer.
    [[nodiscard]] u64 LastFillSeq(PAddr addr) const;

    /// Present-path framebuffer selection for one pane (0 = top-left, 1 = top-right,
    /// 2 = bottom): tracks the addresses this pane's registers have named and returns the one
    /// whose blit completed most recently, or 0 when none has a completion on record (the
    /// caller then falls back to register selection). Render thread only.
    [[nodiscard]] PAddr PickDisplayAddr(u32 pane, PAddr slot1, PAddr slot2, PAddr selected);

    /// Ops handed to the render thread and not yet performed. This is display latency: the
    /// guest is this far ahead of what is on screen.
    [[nodiscard]] u32 RenderQueueDepth() const;

    /// Largest framebuffer a display snapshot holds (400 columns of 240 pixels at 32 bpp).
    static constexpr u32 DisplaySnapshotBytes = 400 * 240 * 4;

    /// For a screen whose framebuffer the CPU writes (no fill, transfer or draw ever landed
    /// there), the bytes the guest had on show at the VBlank that queued the present now
    /// running; null when the framebuffer is GPU-written or the present is not from a VBlank.
    /// The pointer is page-aligned and DisplaySnapshotBytes long. Render thread, during a
    /// present.
    [[nodiscard]] const u8* DisplaySnapshot(u32 screen, PAddr addr) const;

    /// For a guest fault report: the queued fills and transfers that cover the faulting address
    /// or any register value, with how late each ran. A guest is told a fill finished on the
    /// engine's modelled timing while the write itself rides the render queue, so a write that
    /// ran long after the guest reused the memory is the suspect this names.
    [[nodiscard]] std::string DescribeRecentWrites(std::span<const u32> regs, u32 far);

    void RunOnRenderThread(std::function<void()> fn);

    /// Deterministic mode: every queued operation retires before the emulation thread continues.
    void SetRenderThreadLockstep(bool lockstep);

    /// Uncapped mode: no frame pacing at all; the emulation thread runs as fast as it can and
    /// frameskip absorbs the difference. Menus become hostile to interactive input at 10x speed,
    /// so this is a choice, not the default.
    void SetUnlimitedSpeed(bool unlimited);

    /// Unsafe: fire-and-forget even for flushes of GL-written memory (stale readbacks).
    void SetUnsafeAsyncFlush(bool enabled);

    /// Whether the render thread's guest-memory writes (fills, transfers, surface write-back)

    /// Runtime frameskip override: false forces every frame to render (debug port control).
    void SetFrameskipAllowed(bool allowed);

    /// The rasterizer that cache-maintenance callers (memory flushes, DMA, y2r) must use: the
    /// thread-safe facade, never the raw GL rasterizer. Calling the raw one off the render
    /// thread corrupts the surface maps.
    VideoCore::RasterizerInterface* CacheRasterizer();

    /// The guest cleaned its CPU data cache over this physical range: drop cached copies so the
    /// GPU-side reads see the guest's writes - except where the GL holds rendered content it has
    /// never flushed, whose only truth is host-side. See the implementation for the full story.
    void InvalidateOnGuestFlush(PAddr addr, u32 size);

    /// Frames dropped by the render thread's frameskip so far.
    u64 SkippedFrames() const;

    /// Frames actually rendered and presented so far.
    u64 PresentedFrames() const;

    /// Whether GPU execution currently runs on the render thread.
    bool RenderThreadRunning() const;

    /// Raises a GSP interrupt safely from either thread (mailbox when on the render thread).
    void PostInterrupt(Service::GSP::InterruptId id, u64 delay_ns);

    /// Raises a GSP interrupt via a zero-cycle event, for callers already inside GSP.
    void PostInterruptDeferred(Service::GSP::InterruptId id, u64 delay_ns);


    /// Switches the rasterizer's per-title disk resources on the thread that owns the context.
    void SwitchDiskResources(u64 title_id);

    /// Releases the renderer (for GL context destroy in libretro)
    void ReleaseRenderer();

private:
    // Runs on the emulation thread: registers, vertex processing and P3D keep exact timing.
    void ExecSubmitCmdList(u32 index);

    void ExecWriteReg(u32 index, u32 data);

    // Completion (interrupt + status registers) happens here on modelled timing; the memory
    // effect is queued to the render thread.
    void QueueMemoryFill(u32 index, u32 start, u32 end, u32 value, u32 control, u32 intr_index);
    void QueueMemoryTransfer(const Pica::DisplayTransferConfig& config);

    void RecreateRendererLocked(Frontend::EmuWindow& emu_window,
                                Frontend::EmuWindow* secondary_window);

    void VBlankCallback(uintptr_t user_data, s64 cycles_late);

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version);

    std::unique_ptr<RightEyeDisabler> right_eye_disabler;

private:
    friend class RightEyeDisabler;
    struct Impl;
    std::unique_ptr<Impl> impl;

    /// Emulation thread, at the VBlank: copies a CPU-written framebuffer into a free snapshot
    /// slot and returns it, or null when the framebuffer is GPU-written or no slot is free.
    DisplaySnapshotSlot* TakeDisplaySnapshot(u32 screen);

    PAddr VirtualToPhysicalAddress(VAddr addr);
};

} // namespace VideoCore
