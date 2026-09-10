// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "common/color.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica_types.h"
#include "common/scope_exit.h"
#include "video_core/guest_watch.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/utils.h"

namespace SwRenderer {

using Pica::f16;
using Pica::FramebufferRegs;

namespace {

/// Decode/Encode for shadow map format. It is similar to D24S8 format,
/// but the depth field is in big-endian.
const Common::Vec2<u32> DecodeD24S8Shadow(const u8* bytes) {
    return {static_cast<u32>((bytes[0] << 16) | (bytes[1] << 8) | bytes[2]), bytes[3]};
}

void EncodeD24X8Shadow(u32 depth, u8* bytes) {
    bytes[2] = depth & 0xFF;
    bytes[1] = (depth >> 8) & 0xFF;
    bytes[0] = (depth >> 16) & 0xFF;
}

void EncodeX24S8Shadow(u8 stencil, u8* bytes) {
    bytes[3] = stencil;
}
} // Anonymous namespace

Framebuffer::Framebuffer(Memory::MemorySystem& memory_, const Pica::FramebufferRegs& regs_)
    : memory{memory_}, regs{regs_}, cache{memory_} {}

Framebuffer::~Framebuffer() = default;

// The buffer pointers can be null when a bind produced no storage (a zero-sized surface from
// degenerate registers). Every accessor below is guarded rather than dereferencing: the address
// comes from the registers the renderer sees, and drawing at a buffer that maps to nothing is
// something the guest does do.
namespace {
/// Byte range shared by [a, a+an) and [b, b+bn), as an offset into the first and a length.
bool Overlap(PAddr a, u32 an, PAddr b, u32 bn, u32& offset, u32& length) {
    const u64 begin = std::max<u64>(a, b);
    const u64 end = std::min<u64>(u64(a) + an, u64(b) + bn);
    if (begin >= end) {
        return false;
    }
    offset = static_cast<u32>(begin - a);
    length = static_cast<u32>(end - begin);
    return true;
}
} // Anonymous namespace

void SurfaceCache::Reload(PAddr addr, Surface& surface) {
    // Zero only what the guest copy below does not overwrite; a fresh vector arrives zeroed,
    // and a reused one keeps no stale byte because every byte is either copied or filled here.
    if (surface.data.size() != surface.size) {
        surface.data.assign(surface.size, 0);
    }
    if (const u8* src = memory.GetPhysicalPointer(addr)) {
        const auto span = memory.GetPhysicalRef(addr).GetWriteBytes(surface.size);
        // One pass, two stores: guest memory is read once and lands in data and shadow
        // together. Copying into data and then duplicating data into shadow re-read the whole
        // surface - reload traffic was the largest memcpy consumer in the process - and both
        // stores must come from the same read anyway: the natively executing guest can move
        // the memory between two passes, leaving data and shadow disagreeing about the
        // baseline.
        const std::size_t n = span.size();
        surface.shadow.resize(n);
        std::size_t i = 0;
#ifdef __ARM_NEON
        for (; i + 16 <= n; i += 16) {
            const uint8x16_t v = vld1q_u8(src + i);
            vst1q_u8(surface.data.data() + i, v);
            vst1q_u8(surface.shadow.data() + i, v);
        }
#endif
        for (; i < n; ++i) {
            surface.data[i] = src[i];
            surface.shadow[i] = src[i];
        }
        std::fill(surface.data.begin() + n, surface.data.end(), 0);
    } else {
        std::fill(surface.data.begin(), surface.data.end(), 0);
        surface.shadow.clear();
    }
    surface.reload = false;
}

void SurfaceCache::UpdateFromGuest(PAddr addr, u32 size) {
    for (auto& [base, surface] : surfaces) {
        u32 offset, length;
        if (!Overlap(base, surface.size, addr, size, offset, length)) {
            continue;
        }
        if (surface.reload || surface.data.empty()) {
            continue; // The pending reload picks these bytes up anyway.
        }
        const auto span = memory.GetPhysicalRef(base + offset).GetWriteBytes(length);
        if (span.empty()) {
            continue;
        }
        const u8* src = span.data();
        u8* data = surface.data.data() + offset;
        // Data absorbs the whole overlap; shadow only inside its loaded span, so the
        // write-back tail keeps treating everything past it as content to hand back.
        const std::size_t shadowed =
            surface.shadow.size() > offset
                ? std::min<std::size_t>(span.size(), surface.shadow.size() - offset)
                : 0;
        u8* shadow = surface.shadow.data() + offset;
        std::size_t i = 0;
#ifdef __ARM_NEON
        for (; i + 16 <= shadowed; i += 16) {
            const uint8x16_t v = vld1q_u8(src + i);
            vst1q_u8(data + i, v);
            vst1q_u8(shadow + i, v);
        }
#endif
        for (; i < shadowed; ++i) {
            data[i] = src[i];
            shadow[i] = src[i];
        }
        if (i < span.size()) {
            std::memcpy(data + i, src + i, span.size() - i);
        }
    }
}

void SurfaceCache::WriteBack(PAddr addr, Surface& surface) {
    if (!surface.dirty || surface.data.empty()) {
        return;
    }
    auto dst = memory.GetPhysicalRef(addr).GetWriteBytes(surface.size);
    if (dst.empty()) {
        return;
    }
    const u32 watch = VideoCore::GuestWatch::Read();
    SCOPE_EXIT({ VideoCore::GuestWatch::Check(watch, "writeback", addr, addr + surface.size, surface.size); });
    // Only the bytes this surface actually changed go back, and only where guest memory still
    // holds what was loaded. Guest code runs natively here, so a guest CPU write to a surface
    // this cache is holding traps nothing and cannot invalidate it; handing the whole buffer
    // back would quietly restore pre-render content over those writes. Comparing the rendered
    // byte against what was loaded leaves every byte the rasterizer never touched alone
    // (measured on Smash: rare, but 55 KB of guest memory per run went back wrong without it).
    // Comparing guest memory against what was loaded leaves alone every byte the guest has
    // rewritten since - the other half of the same rule. Without it, a render target the title
    // freed and reallocated took a late write-back of rendered pixels over the objects now
    // living there: SM3DL's 32x64 offscreen target in the linear heap, handed back after the
    // scene change, put two pixels (FE40FE40) over a vtable pointer, and every tier crashed at
    // the same PC (2026-09-01). The guest wins wherever it has written; rendered content lands
    // only on bytes nobody has touched since the surface was loaded.
    const std::size_t merged = std::min({dst.size(), surface.data.size(), surface.shadow.size()});
    std::size_t i = 0;
#ifdef __ARM_NEON
    // Byte-precision select without the per-byte branches: keep the destination byte wherever
    // data and shadow agree, take the rendered byte where they differ. The store always happens,
    // but it writes back the destination's own value for untouched bytes, so the merge semantics
    // are identical to the scalar loop.
    for (; i + 16 <= merged; i += 16) {
        const uint8x16_t written = vld1q_u8(surface.data.data() + i);
        const uint8x16_t loaded = vld1q_u8(surface.shadow.data() + i);
        const uint8x16_t same = vceqq_u8(written, loaded);
        // Untouched blocks are skipped entirely, like the scalar u64 test did: the store below
        // rewrites the whole block, and rewriting bytes the renderer never changed is a write
        // to memory the guest may be using for something else by now.
        const uint64x2_t diff = vreinterpretq_u64_u8(vmvnq_u8(same));
        if ((vgetq_lane_u64(diff, 0) | vgetq_lane_u64(diff, 1)) == 0) {
            continue;
        }
        const uint8x16_t current = vld1q_u8(&dst[i]);
        // Take the rendered byte only where it changed and the guest has not: elsewhere the
        // destination keeps its own value.
        const uint8x16_t guest_untouched = vceqq_u8(current, loaded);
        const uint8x16_t take_written = vandq_u8(vmvnq_u8(same), guest_untouched);
        vst1q_u8(&dst[i], vbslq_u8(take_written, written, current));
        // The baseline moves up with the flush, block by block: after this, every byte ever
        // painted matches guest memory, so a withheld byte is one where they already agree.
        // Left at load time instead, a byte later repainted to its load-era value would be
        // withheld while guest memory still held an earlier flush of it - which sprinkled
        // single stale bytes through feedback-sampled surfaces as colour speckles. Blocks the
        // compare skipped need nothing: data and shadow already agree there.
        vst1q_u8(surface.shadow.data() + i, written);
    }
#else
    for (; i + sizeof(u64) <= merged; i += sizeof(u64)) {
        u64 written{}, loaded{};
        std::memcpy(&written, surface.data.data() + i, sizeof(u64));
        std::memcpy(&loaded, surface.shadow.data() + i, sizeof(u64));
        if (written == loaded) {
            continue;
        }
        for (std::size_t j = i; j < i + sizeof(u64); ++j) {
            if (surface.data[j] != surface.shadow[j] && dst[j] == surface.shadow[j]) {
                dst[j] = surface.data[j];
            }
        }
        // Baseline moves up with the flush; see the NEON path.
        std::memcpy(surface.shadow.data() + i, surface.data.data() + i, sizeof(u64));
    }
#endif
    for (; i < merged; ++i) {
        if (surface.data[i] != surface.shadow[i]) {
            if (dst[i] == surface.shadow[i]) {
                dst[i] = surface.data[i];
            }
            surface.shadow[i] = surface.data[i];
        }
    }
    // Anything past what was loaded was never guest content to begin with, so it is all ours,
    // and the baseline follows suit.
    const std::size_t tail = std::min(dst.size(), surface.data.size());
    for (std::size_t k = merged; k < tail; ++k) {
        dst[k] = surface.data[k];
    }
    if (merged < tail) {
        if (surface.shadow.size() < tail) {
            surface.shadow.resize(tail);
        }
        std::memcpy(surface.shadow.data() + merged, surface.data.data() + merged, tail - merged);
    }
    surface.dirty = false;
}

u8* SurfaceCache::GetForWrite(PAddr addr, u32 size) {
    // Entries are keyed by their start address, so without this a surface that overlaps another
    // view of the same memory would get its own host copy: two caches of the same bytes, each
    // holding content the other does not, with whichever happened to be written back last
    // silently winning. Games retarget rendering at overlapping regions constantly - a full
    // frame's run here overlapped twenty thousand times - so hand every overlapping entry back
    // to guest memory and drop it. At most one host copy then covers any byte, and the reload
    // below picks up what those entries held.
    EvictOverlapping(addr, size);
    auto& surface = surfaces[addr];
    if (surface.size != size) {
        // A surface that changed shape has nothing worth keeping; its contents were addressed
        // through the old geometry.
        WriteBack(addr, surface);
        surface.size = size;
        surface.reload = true;
    }
    if (surface.reload) {
        Reload(addr, surface);
    }
    surface.dirty = true;
    return surface.data.data();
}

void SurfaceCache::EvictOverlapping(PAddr addr, u32 size) {
    if (size == 0) {
        return;
    }
    for (auto it = surfaces.begin(); it != surfaces.end();) {
        const PAddr base = it->first;
        if (base == addr || base >= addr + size || addr >= base + it->second.size) {
            ++it;
            continue;
        }
        WriteBack(base, it->second);
        it = surfaces.erase(it);
    }
}

void SurfaceCache::Flush(PAddr addr, u32 size) {
    u32 offset{}, length{};
    for (auto& [base, surface] : surfaces) {
        if (Overlap(base, surface.size, addr, size, offset, length)) {
            WriteBack(base, surface);
        }
    }
}

void SurfaceCache::Invalidate(PAddr addr, u32 size) {
    u32 offset{}, length{};
    for (auto& [base, surface] : surfaces) {
        if (!Overlap(base, surface.size, addr, size, offset, length)) {
            continue;
        }
        // The whole surface goes back, not just the overlap: it is reloaded whole, so anything
        // held only here would be lost. Whatever is about to rewrite the range is queued behind
        // this call and wins, as it should.
        WriteBack(base, surface);
        surface.reload = true;
    }
}

u32 Framebuffer::ColorBufferSize() const {
    const auto& framebuffer = regs.framebuffer;
    const u32 bpp = Pica::BytesPerPixel(Pica::PixelFormat(framebuffer.color_format.Value()));
    // Exactly the surface the registers describe, with no slack: rows are `width` pixels apart,
    // which is the stride BlockOffsets uses, and there are height + 1 of them. Titles park data
    // in the bytes immediately after their colour buffer - Smash keeps three command lists in the
    // 1.5 KB following this one - so handing back even a little more than the surface writes
    // pixels over them, which costs a list its `irq_request` and the frame its P3D completion.
    // BlockOffsets cannot reach past this: its largest offset is
    // (height & ~7) * width + (width - 8) * 8 + 63 pixels, which is inside it.
    return (framebuffer.height + 1) * framebuffer.width * bpp;
}

u32 Framebuffer::DepthBufferSize() const {
    const auto& framebuffer = regs.framebuffer;
    const u32 bpp = Pica::FramebufferRegs::BytesPerDepthPixel(framebuffer.depth_format);
    return (framebuffer.height + 1) * framebuffer.width * bpp;
}

void Framebuffer::Bind() {
    color_addr = regs.framebuffer.GetColorBufferPhysicalAddress();
    color_buffer = cache.GetForWrite(color_addr, ColorBufferSize());

    depth_addr = regs.framebuffer.GetDepthBufferPhysicalAddress();
    depth_buffer = cache.GetForWrite(depth_addr, DepthBufferSize());
}

void Framebuffer::GetPixels(const u32* offsets, u32 mask, u8 (*out)[4]) const {
    if (color_buffer == nullptr) [[unlikely]] {
        return;
    }
    if (regs.framebuffer.color_format == FramebufferRegs::ColorFormat::RGBA8) [[likely]] {
        // Stored as a, b, g, r; the caller wants r, g, b, a. One reversed word per pixel.
        for (u32 lane = 0; lane < 8; ++lane) {
            if ((mask >> lane) & 1) {
                const u8* p = color_buffer + offsets[lane];
                out[lane][0] = p[3];
                out[lane][1] = p[2];
                out[lane][2] = p[1];
                out[lane][3] = p[0];
            }
        }
        return;
    }
    for (u32 lane = 0; lane < 8; ++lane) {
        if ((mask >> lane) & 1) {
            const auto c = GetPixelAt(offsets[lane]);
            out[lane][0] = c.r();
            out[lane][1] = c.g();
            out[lane][2] = c.b();
            out[lane][3] = c.a();
        }
    }
}

void Framebuffer::DrawPixels(const u32* offsets, u32 mask, const u8 (*colors)[4]) const {
    if (color_buffer == nullptr) [[unlikely]] {
        return;
    }
    if (regs.framebuffer.color_format == FramebufferRegs::ColorFormat::RGBA8) [[likely]] {
        for (u32 lane = 0; lane < 8; ++lane) {
            if ((mask >> lane) & 1) {
                u8* p = color_buffer + offsets[lane];
                p[3] = colors[lane][0];
                p[2] = colors[lane][1];
                p[1] = colors[lane][2];
                p[0] = colors[lane][3];
            }
        }
        return;
    }
    for (u32 lane = 0; lane < 8; ++lane) {
        if ((mask >> lane) & 1) {
            DrawPixelAt(offsets[lane], {colors[lane][0], colors[lane][1], colors[lane][2],
                                        colors[lane][3]});
        }
    }
}

Common::Vec4<u8> Framebuffer::GetPixelAt(u32 offset) const {
    if (color_buffer == nullptr) [[unlikely]] {
        return {};
    }
    const u8* src_pixel = color_buffer + offset;
    switch (regs.framebuffer.color_format) {
    case FramebufferRegs::ColorFormat::RGBA8:
        return Common::Color::DecodeRGBA8(src_pixel);
    case FramebufferRegs::ColorFormat::RGB8:
        return Common::Color::DecodeRGB8(src_pixel);
    case FramebufferRegs::ColorFormat::RGB5A1:
        return Common::Color::DecodeRGB5A1(src_pixel);
    case FramebufferRegs::ColorFormat::RGB565:
        return Common::Color::DecodeRGB565(src_pixel);
    case FramebufferRegs::ColorFormat::RGBA4:
        return Common::Color::DecodeRGBA4(src_pixel);
    default:
        LOG_CRITICAL(Render_Software, "Unknown framebuffer color format {:x}",
                     static_cast<u32>(regs.framebuffer.color_format.Value()));
        UNIMPLEMENTED();
    }
    return {0, 0, 0, 0};
}

void Framebuffer::DrawPixelAt(u32 offset, const Common::Vec4<u8>& color) const {
    if (color_buffer == nullptr) [[unlikely]] {
        return;
    }
    u8* dst_pixel = color_buffer + offset;
    switch (regs.framebuffer.color_format) {
    case FramebufferRegs::ColorFormat::RGBA8:
        Common::Color::EncodeRGBA8(color, dst_pixel);
        break;
    case FramebufferRegs::ColorFormat::RGB8:
        Common::Color::EncodeRGB8(color, dst_pixel);
        break;
    case FramebufferRegs::ColorFormat::RGB5A1:
        Common::Color::EncodeRGB5A1(color, dst_pixel);
        break;
    case FramebufferRegs::ColorFormat::RGB565:
        Common::Color::EncodeRGB565(color, dst_pixel);
        break;
    case FramebufferRegs::ColorFormat::RGBA4:
        Common::Color::EncodeRGBA4(color, dst_pixel);
        break;
    default:
        LOG_CRITICAL(Render_Software, "Unknown framebuffer color format {:x}",
                     static_cast<u32>(regs.framebuffer.color_format.Value()));
        UNIMPLEMENTED();
    }
}

u32 Framebuffer::GetDepthAt(u32 offset) const {
    if (depth_buffer == nullptr) [[unlikely]] {
        return 0;
    }
    const u8* src_pixel = depth_buffer + offset;
    switch (regs.framebuffer.depth_format) {
    case FramebufferRegs::DepthFormat::D16:
        return Common::Color::DecodeD16(src_pixel);
    case FramebufferRegs::DepthFormat::D24:
        return Common::Color::DecodeD24(src_pixel);
    case FramebufferRegs::DepthFormat::D24S8:
        return Common::Color::DecodeD24S8(src_pixel).x;
    default:
        LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                     static_cast<u32>(regs.framebuffer.depth_format.Value()));
        UNIMPLEMENTED();
        return 0;
    }
}

void Framebuffer::SetDepthAt(u32 offset, u32 value) const {
    if (depth_buffer == nullptr) [[unlikely]] {
        return;
    }
    u8* dst_pixel = depth_buffer + offset;
    switch (regs.framebuffer.depth_format) {
    case FramebufferRegs::DepthFormat::D16:
        Common::Color::EncodeD16(value, dst_pixel);
        break;
    case FramebufferRegs::DepthFormat::D24:
        Common::Color::EncodeD24(value, dst_pixel);
        break;
    case FramebufferRegs::DepthFormat::D24S8:
        Common::Color::EncodeD24X8(value, dst_pixel);
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                     static_cast<u32>(regs.framebuffer.depth_format.Value()));
        UNIMPLEMENTED();
        break;
    }
}

u8 Framebuffer::GetStencilAt(u32 offset) const {
    if (depth_buffer == nullptr) [[unlikely]] {
        return 0;
    }
    const u8* src_pixel = depth_buffer + offset;
    switch (regs.framebuffer.depth_format) {
    case FramebufferRegs::DepthFormat::D24S8:
        return Common::Color::DecodeD24S8(src_pixel).y;
    default:
        LOG_WARNING(
            HW_GPU,
            "GetStencil called for function which doesn't have a stencil component (format {})",
            static_cast<u32>(regs.framebuffer.depth_format.Value()));
        return 0;
    }
}

void Framebuffer::SetStencilAt(u32 offset, u8 value) const {
    if (depth_buffer == nullptr) [[unlikely]] {
        return;
    }
    u8* dst_pixel = depth_buffer + offset;
    switch (regs.framebuffer.depth_format) {
    case Pica::FramebufferRegs::DepthFormat::D16:
    case Pica::FramebufferRegs::DepthFormat::D24:
        // Nothing to do
        break;
    case Pica::FramebufferRegs::DepthFormat::D24S8:
        Common::Color::EncodeX24S8(value, dst_pixel);
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unimplemented depth format {}",
                     static_cast<u32>(regs.framebuffer.depth_format.Value()));
        UNIMPLEMENTED();
        break;
    }
}

// The coordinate-addressed accessors keep the old signatures and pay one block setup each.
void Framebuffer::DrawPixel(u32 x, u32 y, const Common::Vec4<u8>& color) const {
    u32 offset;
    BlockOffsets(x, y, 1, &offset, nullptr);
    DrawPixelAt(offset, color);
}

const Common::Vec4<u8> Framebuffer::GetPixel(u32 x, u32 y) const {
    u32 offset;
    BlockOffsets(x, y, 1, &offset, nullptr);
    return GetPixelAt(offset);
}

u32 Framebuffer::GetDepth(u32 x, u32 y) const {
    u32 offset;
    BlockOffsets(x, y, 1, nullptr, &offset);
    return GetDepthAt(offset);
}

void Framebuffer::SetDepth(u32 x, u32 y, u32 value) const {
    u32 offset;
    BlockOffsets(x, y, 1, nullptr, &offset);
    SetDepthAt(offset, value);
}

u8 Framebuffer::GetStencil(u32 x, u32 y) const {
    u32 offset;
    BlockOffsets(x, y, 1, nullptr, &offset);
    return GetStencilAt(offset);
}

void Framebuffer::SetStencil(u32 x, u32 y, u8 value) const {
    u32 offset;
    BlockOffsets(x, y, 1, nullptr, &offset);
    SetStencilAt(offset, value);
}

void Framebuffer::DrawShadowMapPixel(u32 x, u32 y, u32 depth, u8 stencil) const {
    const auto& framebuffer = regs.framebuffer;
    const auto& shadow = regs.shadow;

    y = framebuffer.height - y;

    const u32 coarse_y = y & ~7;
    u32 bytes_per_pixel = 4;
    u32 dst_offset = VideoCore::GetMortonOffset(x, y, bytes_per_pixel) +
                     coarse_y * framebuffer.width * bytes_per_pixel;
    // The shadow map is the colour surface, so it goes through the same host storage. Reaching
    // for the guest pointer here instead would both write guest memory from the render thread and
    // race the cached copy that a later flush hands back.
    if (color_buffer == nullptr) [[unlikely]] {
        return;
    }
    u8* dst_pixel = color_buffer + dst_offset;

    const auto ref = DecodeD24S8Shadow(dst_pixel);
    const u32 ref_z = ref.x;
    const u32 ref_s = ref.y;

    if (depth >= ref_z) {
        return;
    }

    if (stencil == 0) {
        EncodeD24X8Shadow(depth, dst_pixel);
    } else {
        const f16 constant = f16::FromRaw(shadow.constant);
        const f16 linear = f16::FromRaw(shadow.linear);
        const f16 x_ = f16::FromFloat32(static_cast<float>(depth) / ref_z);
        const f16 stencil_new = f16::FromFloat32(stencil) / (constant + linear * x_);
        stencil = static_cast<u8>(std::clamp(stencil_new.ToFloat32(), 0.0f, 255.0f));

        if (stencil < ref_s) {
            EncodeX24S8Shadow(stencil, dst_pixel);
        }
    }
}

u8 PerformStencilAction(FramebufferRegs::StencilAction action, u8 old_stencil, u8 ref) {
    switch (action) {
    case FramebufferRegs::StencilAction::Keep:
        return old_stencil;
    case FramebufferRegs::StencilAction::Zero:
        return 0;
    case FramebufferRegs::StencilAction::Replace:
        return ref;
    case FramebufferRegs::StencilAction::Increment:
        // Saturated increment
        return std::min<u8>(old_stencil, 254) + 1;
    case FramebufferRegs::StencilAction::Decrement:
        // Saturated decrement
        return std::max<u8>(old_stencil, 1) - 1;
    case FramebufferRegs::StencilAction::Invert:
        return ~old_stencil;
    case FramebufferRegs::StencilAction::IncrementWrap:
        return old_stencil + 1;
    case FramebufferRegs::StencilAction::DecrementWrap:
        return old_stencil - 1;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown stencil action {:x}", static_cast<int>(action));
        UNIMPLEMENTED();
        return 0;
    }
}

Common::Vec4<u8> EvaluateBlendEquation(const Common::Vec4<u8>& src,
                                       const Common::Vec4<u8>& srcfactor,
                                       const Common::Vec4<u8>& dest,
                                       const Common::Vec4<u8>& destfactor,
                                       FramebufferRegs::BlendEquation equation) {
    Common::Vec4i result;

    const auto src_result = (src * srcfactor).Cast<s32>();
    const auto dst_result = (dest * destfactor).Cast<s32>();

    switch (equation) {
    case FramebufferRegs::BlendEquation::Add:
        result = (src_result + dst_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::Subtract:
        result = (src_result - dst_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::ReverseSubtract:
        result = (dst_result - src_result) / 255;
        break;
    case FramebufferRegs::BlendEquation::Min:
        result.r() = std::min(src_result.r(), dst_result.r()) / 255;
        result.g() = std::min(src_result.g(), dst_result.g()) / 255;
        result.b() = std::min(src_result.b(), dst_result.b()) / 255;
        result.a() = std::min(src_result.a(), dst_result.a()) / 255;
        break;
    case FramebufferRegs::BlendEquation::Max:
        result.r() = std::max(src_result.r(), dst_result.r()) / 255;
        result.g() = std::max(src_result.g(), dst_result.g()) / 255;
        result.b() = std::max(src_result.b(), dst_result.b()) / 255;
        result.a() = std::max(src_result.a(), dst_result.a()) / 255;
        break;
    default:
        LOG_CRITICAL(HW_GPU, "Unknown RGB blend equation 0x{:x}", equation);
        UNIMPLEMENTED();
    }

    return Common::Vec4<u8>(std::clamp(result.r(), 0, 255), std::clamp(result.g(), 0, 255),
                            std::clamp(result.b(), 0, 255), std::clamp(result.a(), 0, 255));
};

u8 LogicOp(u8 src, u8 dest, FramebufferRegs::LogicOp op) {
    switch (op) {
    case FramebufferRegs::LogicOp::Clear:
        return 0;
    case FramebufferRegs::LogicOp::And:
        return src & dest;
    case FramebufferRegs::LogicOp::AndReverse:
        return src & ~dest;
    case FramebufferRegs::LogicOp::Copy:
        return src;
    case FramebufferRegs::LogicOp::Set:
        return 255;
    case FramebufferRegs::LogicOp::CopyInverted:
        return ~src;
    case FramebufferRegs::LogicOp::NoOp:
        return dest;
    case FramebufferRegs::LogicOp::Invert:
        return ~dest;
    case FramebufferRegs::LogicOp::Nand:
        return ~(src & dest);
    case FramebufferRegs::LogicOp::Or:
        return src | dest;
    case FramebufferRegs::LogicOp::Nor:
        return ~(src | dest);
    case FramebufferRegs::LogicOp::Xor:
        return src ^ dest;
    case FramebufferRegs::LogicOp::Equiv:
        return ~(src ^ dest);
    case FramebufferRegs::LogicOp::AndInverted:
        return ~src & dest;
    case FramebufferRegs::LogicOp::OrReverse:
        return src | ~dest;
    case FramebufferRegs::LogicOp::OrInverted:
        return ~src | dest;
    }
    UNREACHABLE();
};

} // namespace SwRenderer
