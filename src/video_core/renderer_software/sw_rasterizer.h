// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <array>
#include <span>
#include <vector>
#include "common/thread_worker.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_software/sw_clipper.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/renderer_software/sw_lighting.h"
#include "video_core/renderer_software/sw_texture_cache.h"

namespace Pica {
struct RegsInternal;
class PicaCore;
} // namespace Pica

namespace SwRenderer {

struct Vertex : Pica::OutputVertex {
    Vertex(const OutputVertex& v) : OutputVertex(v) {}

    /// Attributes used to store intermediate results position after perspective divide.
    Common::Vec3<Pica::f24> screenpos;

    /**
     * Linear interpolation
     * factor: 0=this, 1=vtx
     * Note: This function cannot be called after perspective divide.
     **/
    void Lerp(Pica::f24 factor, const Vertex& vtx) {
        const Pica::f24 one_minus = Pica::f24::One() - factor;
        pos = pos * factor + vtx.pos * one_minus;
        quat = quat * factor + vtx.quat * one_minus;
        color = color * factor + vtx.color * one_minus;
        tc0 = tc0 * factor + vtx.tc0 * one_minus;
        tc1 = tc1 * factor + vtx.tc1 * one_minus;
        tc0_w = tc0_w * factor + vtx.tc0_w * one_minus;
        view = view * factor + vtx.view * one_minus;
        tc2 = tc2 * factor + vtx.tc2 * one_minus;
    }

    /**
     * Linear interpolation
     * factor: 0=v0, 1=v1
     * Note: This function cannot be called after perspective divide.
     **/
    static Vertex Lerp(Pica::f24 factor, const Vertex& v0, const Vertex& v1) {
        Vertex ret = v0;
        ret.Lerp(factor, v1);
        return ret;
    }
};

/**
 * The per-draw state the span loops read, decoded once per batch.
 *
 * PICA registers do not change inside a draw, so decoding the TEV stages and texture units out
 * of their bitfields is batch work, not per-triangle, per-stripe, per-scanline work - which is
 * what it used to be. Plain data only: the texture planes are resolved per triangle and stay
 * with it, and the vector constants are splatted where they are used.
 */
struct SwTevStage {
    u32 cs[3], as[3];   ///< colour and alpha sources, stage 0's Previous already remapped
    u32 cm[3], am[3];   ///< colour and alpha modifiers
    u32 cop, aop;       ///< operations
    u32 cn, an;         ///< inputs the operation reads (1, 2 or 3)
    u32 cshift, ashift; ///< log2 of the output multiplier
    u8 kr, kg, kb, ka;  ///< stage constant
    bool upd_rgb, upd_a;
    bool dot3_rgba;
    u32 index;
};

struct SwTexUnit {
    bool enabled;
    int coord;
    s32 w, h;
    float wf, hf;
    bool s_clamp, t_clamp;
    bool project;   ///< Projection2D on unit 0: divide the coordinates by tc0_w
    bool fast_wrap; ///< wrap is a clamp or a power-of-two mask, so it vectorises
    Pica::TexturingRegs::TextureConfig::WrapMode wrap_s, wrap_t;
    u32 border; ///< border colour in the linear plane's byte order
};

class RasterizerSoftware : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerSoftware(Memory::MemorySystem& memory, Pica::PicaCore& pica);

    void AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                     const Pica::OutputVertex& v2) override;
    void DrawTriangles() override;
    void FlushAll() override {
        FlushBatch();
        fb.FlushRange(0, 0xFFFFFFFFu);
    }
    void FlushRegion(PAddr addr, u32 size) override {
        // Rendered output lives in host storage; this is where the guest gets to see it.
        FlushBatch();
        fb.FlushRange(addr, size);
    }
    void InvalidateRegion(PAddr addr, u32 size) override {
        // Pending triangles must land before the invalidated memory is rewritten (fills,
        // transfers) and before their texture planes can be dropped. Invalidate hands the
        // surface back first, so nothing drawn into it is lost.
        FlushBatch();
        fb.InvalidateRange(addr, size);
        texture_cache.InvalidateRegion(addr, size);
    }
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override {
        FlushBatch();
        fb.InvalidateRange(addr, size);
        texture_cache.InvalidateRegion(addr, size);
    }
    void NoteGuestWrite(PAddr addr, u32 size) override {
        // Same ordering as InvalidateRegion: pending triangles land first. But the surfaces
        // absorb the written bytes instead of being dropped - the per-frame fill/transfer dance
        // was forcing a full surface reload each time. Texture planes still re-decode.
        FlushBatch();
        fb.UpdateFromGuest(addr, size);
        texture_cache.InvalidateRegion(addr, size);
    }

    /// Called once per present by the renderer: advances the texture cache's revalidation epoch.
    void NewTextureFrame() {
        FlushBatch();
        // Hand back every surface still holding content guest memory does not. Rendered output
        // waits in host storage until something flushes or invalidates its range, and for an
        // offscreen target drawn once per scene that was seconds later - after the title had
        // freed the memory and built objects in it, which the hand-back then overwrote with
        // pixels (SM3DL's 32x64 target, 2026-09-01; WriteBack's merge caught most bytes and
        // could not catch one). Once per present bounds how long anything sits here: the
        // screen-sized targets are flushed for the display transfer every frame anyway, so only
        // the stragglers pay, and they are the ones that were dangerous.
        fb.FlushRange(0, 0xFFFFFFFFu);
        texture_cache.NewFrame();
    }
    void ClearAll(bool flush) override {
        FlushBatch();
    }

private:
    /// Computes the screen coordinates of the provided vertex.
    void MakeScreenCoords(Vertex& vtx);

    /// Records the triangle defined by the provided vertices into the draw batch.
    void ProcessTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                         bool reversed = false);

    /// A triangle recorded for the current draw: vertex copies plus the resolved texture
    /// planes (kept alive by the texture cache's graveyard until the batch flushes).
    struct BatchedTriangle {
        Vertex v0, v1, v2;
        bool reversed;
        std::array<const u32*, 3> planes;
        std::array<u32, 3> plane_widths;
    };

    /// Rasterizes the rows of one recorded triangle that belong to the given y-stripe
    /// (stripe = (y >> 4) % 3, absolute, so a row is always owned by the same stripe and
    /// submission order is preserved per pixel). Runs on workers and the render thread.
    void RasterizeTriangleStripe(const BatchedTriangle& tri, int stripe);

    /// Dispatches the recorded batch across the worker stripes and waits for it.
    void FlushBatch();

    /// Decodes the registers into the per-draw tables below. Called once per batch, on the
    /// render thread, before any stripe runs; the tables are read-only after that.
    void DecodeDrawState();

    /// Returns the texture color of the currently processed pixel. The pre-resolved linear
    /// planes come from the batched triangle (resolution happens at record time).
    std::array<Common::Vec4<u8>, 4> TextureColor(
        std::span<const Common::Vec2<f24>, 3> uv,
        std::span<const Pica::TexturingRegs::FullTextureConfig, 3> textures, f24 tc0_w,
        const std::array<const u32*, 3>& tex_planes,
        const std::array<u32, 3>& tex_plane_width) const;

    /// Returns the final pixel color with blending or logic ops applied.
    Common::Vec4<u8> PixelColor(u16 x, u16 y, Common::Vec4<u8> combiner_output) const;

    /// Emulates the TEV configuration and returns the combiner output.
    Common::Vec4<u8> WriteTevConfig(
        std::span<const Common::Vec4<u8>, 4> texture_color,
        std::span<const Pica::TexturingRegs::TevStageConfig, 6> tev_stages,
        Common::Vec4<u8> primary_color, Common::Vec4<u8> primary_fragment_color,
        Common::Vec4<u8> secondary_fragment_color, const std::array<bool, 6>& tev_skip);

    /// Blends fog to the combiner output if enabled.
    void WriteFog(float depth, Common::Vec4<u8>& combiner_output) const;

    /// Performs the alpha test. Returns false if the test failed.
    bool DoAlphaTest(u8 alpha) const;

    /// Performs the depth stencil test. Returns false if the test failed.
    bool DoDepthStencilTest(u16 x, u16 y, float depth) const;

    /**
     * The depth/stencil test for a whole 8-pixel block, for the lanes set in `mask`. Returns
     * the lanes that passed. Same result as calling DoDepthStencilTestAt per lane - the lanes
     * are distinct pixels, so nothing depends on their order - but the register state is read
     * once for the block and a D24S8 pixel is touched once each way instead of up to four
     * times.
     */
    u32 DoDepthStencilTest8(const u32* depth_offsets, const float* depth, u32 mask) const;

    /// The same test against a depth-buffer byte offset from Framebuffer::BlockOffsets.
    bool DoDepthStencilTestAt(u32 depth_offset, float depth) const;

private:
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    Pica::RegsInternal& regs;
    Common::ThreadWorker sw_workers;
    /// Rows are dealt out between the rasterizing threads in bands of 16 by
    /// (y >> 4) % stripe_count. One stripe per thread, so they finish together: with more
    /// stripes than threads one thread draws two bands while another draws one and waits for
    /// it, which is a barrier that costs whatever the extra band cost.
    int stripe_count;
    mutable SwTextureCache texture_cache;
    // Triangles recorded since the last flush. PICA state is constant across a draw, so the
    // stripes re-derive everything else from the vertices; only the resolved texture planes
    // ride along (resolution mutates the cache and must stay on the render thread).
    std::vector<BatchedTriangle> batch;
    /// Registers decoded once per batch (see DecodeDrawState).
    std::array<bool, 6> draw_tev_skip{};
    /// Per-stage constant colours and the combiner buffer's initial colour, decoded once for
    /// the draw. They come out of the register file unchanged for every fragment, and building
    /// them per fragment was up to seven MakeVec-and-cast sequences inside the combiner.
    std::array<Common::Vec4<u8>, 6> draw_tev_const{};
    Common::Vec4<u8> draw_combiner_buffer_color{};
    /// Whether this draw fogs at all, and the fog colour, decoded once. WriteFog is an
    /// out-of-line call made for every fragment that then asks the register file whether fog
    /// is on - for a draw without fog that is a call and a register read per pixel.
    bool draw_fog = false;
    Common::Vec3<u8> draw_fog_color{};
    std::array<SwTexUnit, 3> draw_units{};
    std::array<SwTevStage, 6> draw_stages{};
    u32 draw_num_stages = 0;
    SwLightingState draw_lighting{};
    Framebuffer fb;
};

} // namespace SwRenderer
