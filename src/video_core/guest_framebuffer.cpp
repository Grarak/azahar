// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/guest_framebuffer.h"
#include "video_core/pica/pica_core.h"

namespace VideoCore {

GuestFramebufferPicker::GuestFramebufferPicker(Pica::PicaCore& pica_, Memory::MemorySystem& memory_)
    : pica{pica_}, memory{memory_} {}

PAddr GuestFramebufferPicker::PickFramebufferAddr(GPU* owner_gpu, u32 fb_id) {
    const auto& framebuffer = pica.regs.framebuffer_config[fb_id];
    const PAddr reg_addr = framebuffer.active_fb == 0 ? framebuffer.address_left1
                                                      : framebuffer.address_left2;
    if (owner_gpu == nullptr) {
        return reg_addr;
    }
    // The mirror's register selection is historical - it reflects the flip as of this present's
    // place in the queue - while the pixels are read now. With the guest rotating three buffers
    // any lag walks the display backwards through them (measured: 11 back-jumps in 30 presented
    // frames on a passive scene). PickDisplayAddr remembers every address this screen's
    // registers have named and returns the one whose blit completed most recently, which is
    // monotonic; picking within just the current register pair is not, because the buffer with
    // the newest completed frame rotates out of the pair on every other present.
    const PAddr best = owner_gpu->PickDisplayAddr(fb_id == 0 ? 0 : 2, framebuffer.address_left1,
                                                  framebuffer.address_left2, reg_addr);
    if (best != 0) {
        held_addr[fb_id] = 0;
        return best;
    }
    // No blit on record for either slot: the title draws into the display buffer directly.
    // Register selection with the fill-hold, as before: while a fill into the buffer the guest
    // just flipped to is still queued, keep presenting the previous one - refreshing early is
    // what put the top screen's picture on the bottom (the game recycles these buffers between
    // screens).
    if (held_addr[fb_id] != 0 && owner_gpu->IsFillPending(reg_addr)) {
        return held_addr[fb_id];
    }
    held_addr[fb_id] = reg_addr;
    return reg_addr;
}

GuestFramebuffer GuestFramebufferPicker::Get(GPU* owner_gpu, int screen) {
    const u32 fb_id = screen == 0 ? 0 : 1;
    const auto& framebuffer = pica.regs.framebuffer_config[fb_id];
    const PAddr addr = PickFramebufferAddr(owner_gpu, fb_id);
    const auto fill =
        fb_id == 0 ? pica.regs_lcd.color_fill_top : pica.regs_lcd.color_fill_bottom;
    const u8* snapshot = owner_gpu != nullptr ? owner_gpu->DisplaySnapshot(fb_id, addr) : nullptr;
    return GuestFramebuffer{
        .addr = addr,
        .pixels = snapshot != nullptr ? snapshot : memory.GetPhysicalPointer(addr),
        .snapshot = snapshot != nullptr,
        .texture = nullptr,
        .stride = framebuffer.stride,
        .height = framebuffer.height,
        .format = framebuffer.color_format.Value(),
        .fill_enabled = fill.is_enabled != 0,
        .fill_r = static_cast<u8>(fill.color_r),
        .fill_g = static_cast<u8>(fill.color_g),
        .fill_b = static_cast<u8>(fill.color_b),
    };
}

} // namespace VideoCore
