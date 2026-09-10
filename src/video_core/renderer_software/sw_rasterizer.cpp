// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/math_util.h"
#include <atomic>
#include <bit>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <boost/container/static_vector.hpp>
#include "common/arch.h"
#if CITRA_ARCH(arm32)
#include <arm_neon.h>
#endif
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/quaternion.h"
#include "common/vector_math.h"
#include "core/memory.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_software/sw_framebuffer.h"
#include "video_core/renderer_software/sw_lighting.h"
#include "video_core/renderer_software/sw_proctex.h"
#include "video_core/renderer_software/sw_rasterizer.h"
#include "common/pipeline_stats.h"
#include "video_core/renderer_software/sw_texturing.h"
#include "video_core/texture/texture_decode.h"

namespace SwRenderer {

using Pica::f24;
using Pica::FramebufferRegs;
using Pica::RasterizerRegs;
using Pica::TexturingRegs;
using Pica::Texture::LookupTexture;
using Pica::Texture::TextureInfo;

// Certain games render 2D elements very close to clip plane 0 resulting in very tiny
// negative/positive z values when computing with f32 precision,
// causing some vertices to get erroneously clipped. To workaround this problem,
// we can use a very small epsilon value for clip plane comparison.
constexpr f32 EPSILON_Z = 0.00000001f;

// Vertex lives in sw_rasterizer.h now: the draw batch stores copies by value.

namespace {

MICROPROFILE_DEFINE(GPU_Rasterization, "GPU", "Rasterization", MP_RGB(50, 50, 240));

struct ClippingEdge {
public:
    constexpr ClippingEdge(Common::Vec4<f24> coeffs,
                           Common::Vec4<f24> bias = Common::Vec4<f24>(f24::Zero(), f24::Zero(),
                                                                      f24::Zero(), f24::Zero()))
        : pos(f24::Zero()), coeffs(coeffs), bias(bias) {}

    bool IsInside(const Vertex& vertex) const {
        return Common::Dot(vertex.pos + bias, coeffs) >= f24::FromFloat32(-EPSILON_Z);
    }

    bool IsOutSide(const Vertex& vertex) const {
        return !IsInside(vertex);
    }

    Vertex GetIntersection(const Vertex& v0, const Vertex& v1) const {
        const f24 dp = Common::Dot(v0.pos + bias, coeffs);
        const f24 dp_prev = Common::Dot(v1.pos + bias, coeffs);
        const f24 factor = dp_prev / (dp_prev - dp);
        return Vertex::Lerp(factor, v0, v1);
    }

private:
    [[maybe_unused]] f24 pos;
    Common::Vec4<f24> coeffs;
    Common::Vec4<f24> bias;
};

} // Anonymous namespace

RasterizerSoftware::RasterizerSoftware(Memory::MemorySystem& memory_, Pica::PicaCore& pica_)
    : memory{memory_}, pica{pica_}, regs{pica.regs.internal},
      // Two rasterizing threads: the render thread takes stripe 0 itself and one worker takes
      // the rest. The count is the renderer's shape, not a host-dependent tunable, and it
      // follows the core partition (common/thread.cpp): the guest owns one of the four cores,
      // the emulation and render threads take one each, and a single worker takes the last.
      // One worker, everywhere: three schedulable cores (the guest owns the fourth) give the
      // emulation and render threads one each, so a single worker takes the last. The stripe
      // interleave is still (y >> 4) % 3 - HelpAndWait has the render thread eat the stripe
      // that no longer has a worker of its own.
      sw_workers{1, "SwRenderer workers", {}, Common::ThreadRole::RasterWorker},
      stripe_count{static_cast<int>(sw_workers.NumWorkers()) + 1},
      fb{memory, regs.framebuffer} {}

void RasterizerSoftware::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                     const Pica::OutputVertex& v2) {
#ifdef CITRA_TRACE_PROBES
    {
        static const bool trace = std::getenv("AZAHAR_DRAW_TRACE") != nullptr;
        static std::atomic<u32> n{0};
        if (trace && n.fetch_add(1) < 20000 && (n.load() & 0xFF) == 1) {
            LOG_INFO(HW_GPU, "DRAWTRACE tri n={}", n.load());
        }
    }
#endif // CITRA_TRACE_PROBES
    /**
     * Clipping a planar n-gon against a plane will remove at least 1 vertex and introduces 2 at
     * the new edge (or less in degenerate cases). As such, we can say that each clipping plane
     * introduces at most 1 new vertex to the polygon. Since we start with a triangle and have a
     * fixed 6 clipping planes, the maximum number of vertices of the clipped polygon is 3 + 6 = 9.
     **/
    static constexpr std::size_t MAX_VERTICES = 9;

    boost::container::static_vector<Vertex, MAX_VERTICES> buffer_a = {v0, v1, v2};
    boost::container::static_vector<Vertex, MAX_VERTICES> buffer_b;

    FlipQuaternionIfOpposite(buffer_a[1].quat, buffer_a[0].quat);
    FlipQuaternionIfOpposite(buffer_a[2].quat, buffer_a[0].quat);

    auto* output_list = &buffer_a;
    auto* input_list = &buffer_b;

    // NOTE: We clip against a w=epsilon plane to guarantee that the output has a positive w value.
    // TODO: Not sure if this is a valid approach. Also should probably instead use the smallest
    //       epsilon possible within f24 accuracy.
    static constexpr f24 EPSILON = f24::FromFloat32(0.00001f);
    static constexpr f24 f0 = f24::Zero();
    static constexpr f24 f1 = f24::One();
    static constexpr std::array<ClippingEdge, 7> clipping_edges = {{
        {Common::MakeVec(-f1, f0, f0, f1)},                                        // x = +w
        {Common::MakeVec(f1, f0, f0, f1)},                                         // x = -w
        {Common::MakeVec(f0, -f1, f0, f1)},                                        // y = +w
        {Common::MakeVec(f0, f1, f0, f1)},                                         // y = -w
        {Common::MakeVec(f0, f0, -f1, f0)},                                        // z =  0
        {Common::MakeVec(f0, f0, f1, f1)},                                         // z = -w
        {Common::MakeVec(f0, f0, f0, f1), Common::Vec4<f24>(f0, f0, f0, EPSILON)}, // w = EPSILON
    }};

    // Simple implementation of the Sutherland-Hodgman clipping algorithm.
    // TODO: Make this less inefficient (currently lots of useless buffering overhead happens here)
    const auto clip = [&](const ClippingEdge& edge) {
        std::swap(input_list, output_list);
        output_list->clear();

        const Vertex* reference_vertex = &input_list->back();
        for (const auto& vertex : *input_list) {
            // NOTE: This algorithm changes vertex order in some cases!
            if (edge.IsInside(vertex)) {
                if (edge.IsOutSide(*reference_vertex)) {
                    output_list->push_back(edge.GetIntersection(vertex, *reference_vertex));
                }
                output_list->push_back(vertex);
            } else if (edge.IsInside(*reference_vertex)) {
                output_list->push_back(edge.GetIntersection(vertex, *reference_vertex));
            }
            reference_vertex = &vertex;
        }
    };

    // Almost every triangle lies fully inside the volume, and for those the seven-plane walk
    // only shuffles the (large) vertices between the two buffers. The same plane tests answer
    // whether any clipping is needed at all, so run them first without copying anything.
    bool fully_inside = !regs.rasterizer.clip_enable;
    for (std::size_t e = 0; fully_inside && e < clipping_edges.size(); ++e) {
        for (const Vertex& vertex : buffer_a) {
            if (clipping_edges[e].IsOutSide(vertex)) {
                fully_inside = false;
                break;
            }
        }
    }

    if (!fully_inside) {
        for (const ClippingEdge& edge : clipping_edges) {
            clip(edge);
            if (output_list->size() < 3) {
                return;
            }
        }

        if (regs.rasterizer.clip_enable) {
            const ClippingEdge custom_edge{regs.rasterizer.GetClipCoef()};
            clip(custom_edge);
            if (output_list->size() < 3) {
                return;
            }
        }
    }

    MakeScreenCoords((*output_list)[0]);
    MakeScreenCoords((*output_list)[1]);

    for (std::size_t i = 0; i < output_list->size() - 2; i++) {
        Vertex& vtx0 = (*output_list)[0];
        Vertex& vtx1 = (*output_list)[i + 1];
        Vertex& vtx2 = (*output_list)[i + 2];

        MakeScreenCoords(vtx2);

        LOG_TRACE(
            Render_Software,
            "Triangle {}/{} at position ({:.3}, {:.3}, {:.3}, {:.3f}), "
            "({:.3}, {:.3}, {:.3}, {:.3}), ({:.3}, {:.3}, {:.3}, {:.3}) and "
            "screen position ({:.2}, {:.2}, {:.2}), ({:.2}, {:.2}, {:.2}), ({:.2}, {:.2}, {:.2})",
            i + 1, output_list->size() - 2, vtx0.pos.x.ToFloat32(), vtx0.pos.y.ToFloat32(),
            vtx0.pos.z.ToFloat32(), vtx0.pos.w.ToFloat32(), vtx1.pos.x.ToFloat32(),
            vtx1.pos.y.ToFloat32(), vtx1.pos.z.ToFloat32(), vtx1.pos.w.ToFloat32(),
            vtx2.pos.x.ToFloat32(), vtx2.pos.y.ToFloat32(), vtx2.pos.z.ToFloat32(),
            vtx2.pos.w.ToFloat32(), vtx0.screenpos.x.ToFloat32(), vtx0.screenpos.y.ToFloat32(),
            vtx0.screenpos.z.ToFloat32(), vtx1.screenpos.x.ToFloat32(),
            vtx1.screenpos.y.ToFloat32(), vtx1.screenpos.z.ToFloat32(),
            vtx2.screenpos.x.ToFloat32(), vtx2.screenpos.y.ToFloat32(),
            vtx2.screenpos.z.ToFloat32());

        ProcessTriangle(vtx0, vtx1, vtx2);
    }
}

void RasterizerSoftware::MakeScreenCoords(Vertex& vtx) {
    Viewport viewport{};
    viewport.halfsize_x = f24::FromRaw(regs.rasterizer.viewport_size_x);
    viewport.halfsize_y = f24::FromRaw(regs.rasterizer.viewport_size_y);
    viewport.offset_x = f24::FromFloat32(static_cast<f32>(regs.rasterizer.viewport_corner.x));
    viewport.offset_y = f24::FromFloat32(static_cast<f32>(regs.rasterizer.viewport_corner.y));

    f24 inv_w = f24::One() / vtx.pos.w;
    vtx.pos.w = inv_w;
    vtx.quat *= inv_w;
    vtx.color *= inv_w;
    vtx.tc0 *= inv_w;
    vtx.tc1 *= inv_w;
    vtx.tc0_w *= inv_w;
    vtx.view *= inv_w;
    vtx.tc2 *= inv_w;

    vtx.screenpos[0] = (vtx.pos.x * inv_w + f24::One()) * viewport.halfsize_x + viewport.offset_x;
    vtx.screenpos[1] = (vtx.pos.y * inv_w + f24::One()) * viewport.halfsize_y + viewport.offset_y;
    vtx.screenpos[2] = vtx.pos.z * inv_w;
}

void RasterizerSoftware::ProcessTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                                         bool reversed) {
    MICROPROFILE_SCOPE(GPU_Rasterization);

    // Vertex positions in rasterizer coordinates
    static auto screen_to_rasterizer_coords = [](const Common::Vec3<f24>& vec) {
        return Common::Vec3{Fix12P4::FromFloat24(vec.x), Fix12P4::FromFloat24(vec.y),
                            Fix12P4::FromFloat24(vec.z)};
    };

    const std::array<Common::Vec3<Fix12P4>, 3> vtxpos = {
        screen_to_rasterizer_coords(v0.screenpos),
        screen_to_rasterizer_coords(v1.screenpos),
        screen_to_rasterizer_coords(v2.screenpos),
    };

    if (regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepAll ||
        regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepAll2) {
        // Make sure we always end up with a triangle wound counter-clockwise
        if (!reversed && SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) <= 0) {
            ProcessTriangle(v0, v2, v1, true);
            return;
        }
    } else {
        if (!reversed && regs.rasterizer.cull_mode == RasterizerRegs::CullMode::KeepClockWise) {
            // Reverse vertex order and use the CCW code path.
            ProcessTriangle(v0, v2, v1, true);
            return;
        }
        // Cull away triangles which are wound clockwise.
        if (SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) <= 0) {
            return;
        }
    }

    // The triangle survived culling. PICA state stays fixed for the rest of the draw, so
    // resolve the texture planes now (the cache may only be touched on the render thread),
    // record the triangle, and rasterize the whole batch in y-stripes when the draw ends
    // (DrawTriangles -> FlushBatch). This costs one barrier per draw instead of one per
    // triangle, which used to put the worker threads' time into the queue, not into pixels.
    {
        const auto& fb_cfg = regs.framebuffer.framebuffer;
        const u32 fb_pixels = static_cast<u32>(fb_cfg.width) * fb_cfg.height;
        texture_cache.NoteRenderedRange(fb_cfg.GetColorBufferPhysicalAddress(), fb_pixels * 4);
        texture_cache.NoteRenderedRange(fb_cfg.GetDepthBufferPhysicalAddress(), fb_pixels * 4);
    }
    BatchedTriangle tri{v0, v1, v2, reversed, {}, {}};
    {
        const auto textures = regs.texturing.GetTextures();
        for (u32 i = 0; i < 3; ++i) {
            const auto& texture = textures[i];
            if (!texture.enabled) {
                continue;
            }
            if (i == 0 && (texture.config.type == TexturingRegs::TextureConfig::TextureCube ||
                           texture.config.type == TexturingRegs::TextureConfig::ShadowCube)) {
                continue; // cube faces change the source address per pixel
            }
            const PAddr addr = texture.config.GetPhysicalAddress();
            const u8* src = memory.GetPhysicalPointer(addr);
            const auto info = TextureInfo::FromPicaRegister(texture.config, texture.format);
            // A texture inside this frame's render output samples the frame itself (stage
            // post-processing does this every frame). The pixels are still in the surface
            // cache's host storage; hand them back to guest memory before the decode below
            // reads it, or the pass composites from whatever the guest wrote there last -
            // Jungle Japes' colour-grade pass turned the whole stage near-black this way.
            const u32 source_bytes = static_cast<u32>(info.stride * (info.height / 8));
            if (texture_cache.OverlapsRendered(addr, source_bytes)) {
                fb.FlushRange(addr, source_bytes);
            }
            tri.planes[i] = texture_cache.GetLinear(src, addr, info);
            tri.plane_widths[i] = info.width;
        }
    }
    batch.push_back(std::move(tri));
}

void RasterizerSoftware::DrawTriangles() {
    FlushBatch();
}

void RasterizerSoftware::DecodeDrawState() {
    const auto textures = regs.texturing.GetTextures();
    const auto tev_stages = regs.texturing.GetTevStages();
    draw_num_stages = 0;
    draw_units = {};
    // Constant for the whole draw; see draw_tev_const.
    for (std::size_t i = 0; i < tev_stages.size(); i++) {
        const auto& stage = tev_stages[i];
        draw_tev_const[i] = Common::MakeVec(stage.const_r.Value(), stage.const_g.Value(),
                                            stage.const_b.Value(), stage.const_a.Value())
                                .Cast<u8>();
    }
    draw_fog = regs.texturing.fog_mode == TexturingRegs::FogMode::Fog;
    draw_fog_color =
        Common::MakeVec(regs.texturing.fog_color.r.Value(), regs.texturing.fog_color.g.Value(),
                        regs.texturing.fog_color.b.Value())
            .Cast<u8>();
    draw_combiner_buffer_color =
        Common::MakeVec(regs.texturing.tev_combiner_buffer_color.r.Value(),
                        regs.texturing.tev_combiner_buffer_color.g.Value(),
                        regs.texturing.tev_combiner_buffer_color.b.Value(),
                        regs.texturing.tev_combiner_buffer_color.a.Value())
            .Cast<u8>();
    if (!regs.lighting.disable) {
        draw_lighting = DecodeLighting(regs.lighting);
    }

// Identity-stage mask for WriteTevConfig: most titles configure one or two real TEV
// stages and leave the rest passing Previous through unchanged, and the combiner loop
// dominated the profile. A stage may only be skipped when nothing reads the combiner
// buffer (skipping also skips the buffer shift) and the stage itself does not update it.
// Stage 0 never skips: its Previous sources remap to source3.
// Requires retirement-ordered completion acks (deferred interrupts + fill finished
// bits): the skip's speedup widened that race before the ordering fixes landed.
bool tev_reads_buffer = false;
{
    using Src = TexturingRegs::TevStageConfig::Source;
    for (const auto& ts : tev_stages) {
        if (ts.color_source1 == Src::PreviousBuffer || ts.color_source2 == Src::PreviousBuffer ||
            ts.color_source3 == Src::PreviousBuffer || ts.alpha_source1 == Src::PreviousBuffer ||
            ts.alpha_source2 == Src::PreviousBuffer || ts.alpha_source3 == Src::PreviousBuffer) {
            tev_reads_buffer = true;
            break;
        }
    }
}
auto& tev_skip = draw_tev_skip;
    tev_skip = {};
if (!tev_reads_buffer) {
    using Op = TexturingRegs::TevStageConfig::Operation;
    using Src = TexturingRegs::TevStageConfig::Source;
    using CMod = TexturingRegs::TevStageConfig::ColorModifier;
    using AMod = TexturingRegs::TevStageConfig::AlphaModifier;
    const auto& buffer_input = regs.texturing.tev_combiner_buffer_input;
    for (u32 i = 1; i < tev_stages.size(); ++i) {
        const auto& ts = tev_stages[i];
        tev_skip[i] =
            ts.color_op == Op::Replace && ts.color_source1 == Src::Previous &&
            ts.color_modifier1 == CMod::SourceColor && ts.GetColorMultiplier() == 1 &&
            ts.alpha_op == Op::Replace && ts.alpha_source1 == Src::Previous &&
            ts.alpha_modifier1 == AMod::SourceAlpha && ts.GetAlphaMultiplier() == 1 &&
            !buffer_input.TevStageUpdatesCombinerBufferColor(i) &&
            !buffer_input.TevStageUpdatesCombinerBufferAlpha(i);
    }
}

for (u32 i = 0; i < 3; ++i) {
    const auto& t = textures[i];
    draw_units[i].enabled = t.enabled;
    if (!t.enabled) {
        continue;
    }
    draw_units[i].coord =
        (i == 2 && regs.texturing.main_config.texture2_use_coord1) ? 1
        : static_cast<int>(i);
            draw_units[i].w = static_cast<s32>(t.config.width);
    draw_units[i].h = static_cast<s32>(t.config.height);
    draw_units[i].wf = static_cast<float>(t.config.width);
    draw_units[i].hf = static_cast<float>(t.config.height);
    draw_units[i].s_clamp =
        t.config.wrap_s == TexturingRegs::TextureConfig::ClampToEdge;
    draw_units[i].t_clamp =
        t.config.wrap_t == TexturingRegs::TextureConfig::ClampToEdge;
    draw_units[i].project =
        i == 0 &&
        t.config.type == TexturingRegs::TextureConfig::Projection2D;
    draw_units[i].wrap_s = t.config.wrap_s;
    draw_units[i].wrap_t = t.config.wrap_t;
    const auto wrap_is_fast =
        [](TexturingRegs::TextureConfig::WrapMode m, u32 dim) {
            if (m == TexturingRegs::TextureConfig::ClampToEdge) {
                return true;
            }
            if (m == TexturingRegs::TextureConfig::Repeat) {
                return (dim & (dim - 1)) == 0 && dim != 0;
            }
            return false;
        };
    draw_units[i].fast_wrap = wrap_is_fast(t.config.wrap_s, t.config.width) &&
                          wrap_is_fast(t.config.wrap_t, t.config.height);
    const auto bc = t.config.border_color;
    draw_units[i].border = static_cast<u32>(bc.r.Value()) |
                       (static_cast<u32>(bc.g.Value()) << 8) |
                       (static_cast<u32>(bc.b.Value()) << 16) |
                       (static_cast<u32>(bc.a.Value()) << 24);
}
using Src = TexturingRegs::TevStageConfig::Source;
const auto& buffer_input = regs.texturing.tev_combiner_buffer_input;
for (u32 i = 0; i < 6; ++i) {
    if (tev_skip[i]) {
        continue;
    }
    const auto& s = tev_stages[i];
    SwTevStage& n = draw_stages[draw_num_stages++];
    n.index = i;
    // Stage 0 remaps Previous color sources to source3 (scalar rule).
    const auto remap = [&](Src src) {
        return (i == 0 && src == Src::Previous) ? s.color_source3.Value()
                                                : src;
    };
    n.cs[0] = static_cast<u32>(remap(s.color_source1)) & 0xF;
    n.cs[1] = static_cast<u32>(remap(s.color_source2)) & 0xF;
    n.cs[2] = static_cast<u32>(s.color_source3.Value()) & 0xF;
    n.as[0] = static_cast<u32>(s.alpha_source1.Value()) & 0xF;
    n.as[1] = static_cast<u32>(s.alpha_source2.Value()) & 0xF;
    n.as[2] = static_cast<u32>(s.alpha_source3.Value()) & 0xF;
    n.cm[0] = static_cast<u32>(s.color_modifier1.Value()) & 0xF;
    n.cm[1] = static_cast<u32>(s.color_modifier2.Value()) & 0xF;
    n.cm[2] = static_cast<u32>(s.color_modifier3.Value()) & 0xF;
    n.am[0] = static_cast<u32>(s.alpha_modifier1.Value()) & 7;
    n.am[1] = static_cast<u32>(s.alpha_modifier2.Value()) & 7;
    n.am[2] = static_cast<u32>(s.alpha_modifier3.Value()) & 7;
    n.cop = static_cast<u32>(s.color_op.Value());
    n.aop = static_cast<u32>(s.alpha_op.Value());
    // Replace reads one input, the three-operand forms read all
    // three, everything else reads two. Building the ones the
    // operation ignores was a third of the interpreter's per-block
    // vector work.
    const auto arity = [](u32 op) -> u32 {
        using Op = TexturingRegs::TevStageConfig::Operation;
        if (op == static_cast<u32>(Op::Replace)) {
            return 1;
        }
        if (op == static_cast<u32>(Op::Lerp) ||
            op == static_cast<u32>(Op::MultiplyThenAdd) ||
            op == static_cast<u32>(Op::AddThenMultiply)) {
            return 3;
        }
        return 2;
    };
    n.cn = arity(n.cop);
    n.an = arity(n.aop);
    n.cshift = static_cast<u32>(std::countr_zero(s.GetColorMultiplier()));
    n.ashift = static_cast<u32>(std::countr_zero(s.GetAlphaMultiplier()));
    n.kr = static_cast<u8>(s.const_r.Value());
    n.kg = static_cast<u8>(s.const_g.Value());
    n.kb = static_cast<u8>(s.const_b.Value());
    n.ka = static_cast<u8>(s.const_a.Value());
    n.upd_rgb = buffer_input.TevStageUpdatesCombinerBufferColor(i);
    n.upd_a = buffer_input.TevStageUpdatesCombinerBufferAlpha(i);
    n.dot3_rgba =
        s.color_op == TexturingRegs::TevStageConfig::Operation::Dot3_RGBA;
}
}

void RasterizerSoftware::FlushBatch() {
    if (batch.empty()) {
        // Still collect: an empty batch means nothing is rasterizing, so no parked plane can
        // still be referenced. Returning early left the graveyard growing across every stretch
        // with invalidations but no draws, and nothing else ever empties it.
        texture_cache.CollectGarbage();
        return;
    }
    texture_cache.NewBatch();
    fb.Bind();
    DecodeDrawState();
    for (int stripe = 1; stripe < stripe_count; stripe++) {
        sw_workers.QueueWork([this, stripe] {
            for (const auto& tri : batch) {
                RasterizeTriangleStripe(tri, stripe);
            }
        });
    }
    for (const auto& tri : batch) {
        RasterizeTriangleStripe(tri, 0);
    }
    // The queued stripes iterate the batch by reference; the barrier keeps it alive.
    const u64 barrier_start = Common::PipelineStats::NowUs();
    sw_workers.HelpAndWait();
    Common::PipelineStats::barrier_us.fetch_add(Common::PipelineStats::NowUs() - barrier_start,
                                                std::memory_order_relaxed);
    Common::PipelineStats::barrier_count.fetch_add(1, std::memory_order_relaxed);
    batch.clear();
    texture_cache.CollectGarbage();
}

void RasterizerSoftware::RasterizeTriangleStripe(const BatchedTriangle& tri, int stripe) {
    // Decoded once per batch by DecodeDrawState; the registers are constant inside a draw.
    const auto& tev_skip = draw_tev_skip;
    const Vertex& v0 = tri.v0;
    const Vertex& v1 = tri.v1;
    const Vertex& v2 = tri.v2;
    const auto& tex_planes = tri.planes;
    const auto& tex_plane_width = tri.plane_widths;

    const auto screen_to_rasterizer_coords = [](const Common::Vec3<f24>& vec) {
        return Common::Vec3<Fix12P4>{Fix12P4::FromFloat24(vec.x), Fix12P4::FromFloat24(vec.y),
                                     Fix12P4::FromFloat24(vec.z)};
    };

    const std::array<Common::Vec3<Fix12P4>, 3> vtxpos = {
        screen_to_rasterizer_coords(v0.screenpos),
        screen_to_rasterizer_coords(v1.screenpos),
        screen_to_rasterizer_coords(v2.screenpos),
    };

    u16 min_x = std::min({vtxpos[0].x, vtxpos[1].x, vtxpos[2].x});
    u16 min_y = std::min({vtxpos[0].y, vtxpos[1].y, vtxpos[2].y});
    u16 max_x = std::max({vtxpos[0].x, vtxpos[1].x, vtxpos[2].x});
    u16 max_y = std::max({vtxpos[0].y, vtxpos[1].y, vtxpos[2].y});

    // Convert the scissor box coordinates to 12.4 fixed point
    const u16 scissor_x1 = static_cast<u16>(regs.rasterizer.scissor_test.x1 << 4);
    const u16 scissor_y1 = static_cast<u16>(regs.rasterizer.scissor_test.y1 << 4);
    // x2,y2 have +1 added to cover the entire sub-pixel area
    const u16 scissor_x2 = static_cast<u16>((regs.rasterizer.scissor_test.x2 + 1) << 4);
    const u16 scissor_y2 = static_cast<u16>((regs.rasterizer.scissor_test.y2 + 1) << 4);

    if (regs.rasterizer.scissor_test.mode == RasterizerRegs::ScissorMode::Include) {
        // Calculate the new bounds
        min_x = std::max(min_x, scissor_x1);
        min_y = std::max(min_y, scissor_y1);
        max_x = std::min(max_x, scissor_x2);
        max_y = std::min(max_y, scissor_y2);
    }

    min_x &= Fix12P4::IntMask();
    min_y &= Fix12P4::IntMask();
    max_x = ((max_x + Fix12P4::FracMask()) & Fix12P4::IntMask());
    max_y = ((max_y + Fix12P4::FracMask()) & Fix12P4::IntMask());

    const int bias0 =
        IsRightSideOrFlatBottomEdge(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy()) ? -1 : 0;
    const int bias1 =
        IsRightSideOrFlatBottomEdge(vtxpos[1].xy(), vtxpos[2].xy(), vtxpos[0].xy()) ? -1 : 0;
    const int bias2 =
        IsRightSideOrFlatBottomEdge(vtxpos[2].xy(), vtxpos[0].xy(), vtxpos[1].xy()) ? -1 : 0;

    const auto w_inverse = Common::MakeVec(v0.pos.w, v1.pos.w, v2.pos.w);

    const auto textures = regs.texturing.GetTextures();
    const auto tev_stages = regs.texturing.GetTevStages();




    // Texture planes were resolved at record time (ProcessTriangle); the framebuffer was
    // bound once for the batch (FlushBatch). Nothing here may touch the texture cache.

    // The edge functions are affine in x, so each scanline computes them once at the first
    // pixel and steps by a constant per pixel; the covered x-interval is solved in exact
    // integer math up front, so the pixel loop runs without per-pixel coverage tests. Values
    // are bit-identical to evaluating SignedArea at every pixel. wsum is the same for every
    // pixel of the triangle (twice its signed area).
    const auto edge_step_x = [](const Common::Vec2<Fix12P4>& a, const Common::Vec2<Fix12P4>& b) {
        // d(SignedArea(a, b, p))/dp.x = -(b.y - a.y); one pixel step is 0x10 in 12.4.
        return -0x10 * (static_cast<s32>(static_cast<u16>(b.y)) -
                        static_cast<s32>(static_cast<u16>(a.y)));
    };
    const s32 w0_dx = edge_step_x(vtxpos[1].xy(), vtxpos[2].xy());
    const s32 w1_dx = edge_step_x(vtxpos[2].xy(), vtxpos[0].xy());
    const s32 w2_dx = edge_step_x(vtxpos[0].xy(), vtxpos[1].xy());
    // w0 + w1 + w2 is twice the triangle's signed area plus the fill biases: constant for
    // every covered pixel. The former per-pixel float divide by it becomes one reciprocal.
    const s32 wsum_tri = bias0 + bias1 + bias2 +
                         SignedArea(vtxpos[1].xy(), vtxpos[2].xy(), vtxpos[0].xy());
    if (wsum_tri == 0) {
        return; // degenerate
    }
    const float inv_wsum = 1.0f / static_cast<float>(wsum_tri);
    const bool scissor_exclude =
        regs.rasterizer.scissor_test.mode == RasterizerRegs::ScissorMode::Exclude;

    // Span-kernel dispatch: the flat-quad kernel handles the
    // census's Replace(PrimaryColor) pattern - untextured Gouraud quads, blended, optional
    // alpha test and unconditional depth write, no depth/stencil test, no fog, no lighting.
    // The kernel body reuses the exact per-pixel helpers, so output is identical; what it
    // removes is the texture sampling, lighting and TEV machinery the pattern never touches.
    enum class SpanKernel {
        None,
        FlatColor,
        ModulateTex0,
        AddModTex0,
        LerpLerpTex0,
        ReplaceTex0,
        PrimAlphaTex0,
        GenericNeon,
    };
    const SpanKernel span_kernel = [&] {
        using Op = TexturingRegs::TevStageConfig::Operation;
        using Src = TexturingRegs::TevStageConfig::Source;
        using CMod = TexturingRegs::TevStageConfig::ColorModifier;
        using AMod = TexturingRegs::TevStageConfig::AlphaModifier;
        const auto& om = regs.framebuffer.output_merger;
        // Stencil is not a gate: the alpha and depth/stencil tests run per lane on the same
        // scalar helper the generic loop uses, in the same ascending order, so the stencil
        // side effects are already the reference implementation's.
        if (scissor_exclude || !regs.lighting.disable ||
            regs.texturing.fog_mode == TexturingRegs::FogMode::Fog ||
            om.fragment_operation_mode != FramebufferRegs::FragmentOperationMode::Default ||
            !om.alphablend_enable) {
            return SpanKernel::None;
        }
        for (u32 i = 2; i < 6; ++i) {
            if (!tev_skip[i]) {
                return SpanKernel::None;
            }
        }
        const bool stage1_active = !tev_skip[1];
        const auto& ts = tev_stages[0];
        const auto& ts1 = tev_stages[1];
        if (ts.GetColorMultiplier() != 1 || ts.GetAlphaMultiplier() != 1 ||
            (stage1_active && (ts1.GetColorMultiplier() != 1 || ts1.GetAlphaMultiplier() != 1))) {
            return SpanKernel::None;
        }
        const bool tex0_2d_ready =
            textures[0].enabled &&
            textures[0].config.type == TexturingRegs::TextureConfig::Texture2D &&
            tex_planes[0] != nullptr;
        // Two-stage fight patterns from the census (exact configs, stage fields strict):
        if (stage1_active && tex0_2d_ready) {
            // K5: s0 color Add(Tex0.rgb, Const.a), alpha Replace(Primary.a);
            //     s1 color Modulate(Const, Previous), alpha Replace(Previous).
            if (ts.color_op == Op::Add && ts.color_source1 == Src::Texture0 &&
                ts.color_source2 == Src::Constant && ts.color_modifier1 == CMod::SourceColor &&
                ts.color_modifier2 == CMod::SourceAlpha && ts.alpha_op == Op::Replace &&
                ts.alpha_source1 == Src::PrimaryColor &&
                ts.alpha_modifier1 == AMod::SourceAlpha && ts1.color_op == Op::Modulate &&
                ts1.color_source1 == Src::Constant && ts1.color_source2 == Src::Previous &&
                ts1.color_modifier1 == CMod::SourceColor &&
                ts1.color_modifier2 == CMod::SourceColor && ts1.alpha_op == Op::Replace &&
                ts1.alpha_source1 == Src::Previous &&
                ts1.alpha_modifier1 == AMod::SourceAlpha) {
                return SpanKernel::AddModTex0;
            }
            // K6: s0 color Lerp(Tex0, Const, f=Tex0), alpha Replace(Const.a);
            //     s1 color Lerp(Tex0, Const, f=Previous), alpha Replace(Previous).
            if (ts.color_op == Op::Lerp && ts.color_source1 == Src::Texture0 &&
                ts.color_source2 == Src::Constant && ts.color_source3 == Src::Texture0 &&
                ts.color_modifier1 == CMod::SourceColor &&
                ts.color_modifier2 == CMod::SourceColor &&
                ts.color_modifier3 == CMod::SourceColor && ts.alpha_op == Op::Replace &&
                ts.alpha_source1 == Src::Constant && ts.alpha_modifier1 == AMod::SourceAlpha &&
                ts1.color_op == Op::Lerp && ts1.color_source1 == Src::Texture0 &&
                ts1.color_source2 == Src::Constant && ts1.color_source3 == Src::Previous &&
                ts1.color_modifier1 == CMod::SourceColor &&
                ts1.color_modifier2 == CMod::SourceColor &&
                ts1.color_modifier3 == CMod::SourceColor && ts1.alpha_op == Op::Replace &&
                ts1.alpha_source1 == Src::Previous &&
                ts1.alpha_modifier1 == AMod::SourceAlpha) {
                return SpanKernel::LerpLerpTex0;
            }
        }
        if (stage1_active) {
            return SpanKernel::None;
        }
        if (ts.color_op == Op::Replace && ts.color_source1 == Src::PrimaryColor &&
            ts.color_modifier1 == CMod::SourceColor && ts.alpha_op == Op::Replace &&
            ts.alpha_source1 == Src::PrimaryColor && ts.alpha_modifier1 == AMod::SourceAlpha) {
            return SpanKernel::FlatColor;
        }
        // Replace(Texture0)/Replace(Texture0.a): plain texture stamp. 26% of fight pixels.
        if (ts.color_op == Op::Replace && ts.color_source1 == Src::Texture0 &&
            ts.color_modifier1 == CMod::SourceColor && ts.alpha_op == Op::Replace &&
            ts.alpha_source1 == Src::Texture0 && ts.alpha_modifier1 == AMod::SourceAlpha &&
            tex0_2d_ready) {
            return SpanKernel::ReplaceTex0;
        }
        // Replace(Primary)/Modulate(Tex0.a, Primary.a): vertex-colored quad whose alpha is
        // shaped by the texture (shadows, glows). 10% of fight pixels.
        if (ts.color_op == Op::Replace && ts.color_source1 == Src::PrimaryColor &&
            ts.color_modifier1 == CMod::SourceColor && ts.alpha_op == Op::Modulate &&
            ts.alpha_modifier1 == AMod::SourceAlpha && ts.alpha_modifier2 == AMod::SourceAlpha &&
            ((ts.alpha_source1 == Src::Texture0 && ts.alpha_source2 == Src::PrimaryColor) ||
             (ts.alpha_source1 == Src::PrimaryColor && ts.alpha_source2 == Src::Texture0)) &&
            tex0_2d_ready) {
            return SpanKernel::PrimAlphaTex0;
        }
        // Modulate(Texture0, PrimaryColor): unit 0 must be a plain 2D texture with its linear
        // plane resolved (nearest sampling matches the generic path, which applies no filter).
        if (ts.color_op == Op::Modulate && ts.alpha_op == Op::Modulate &&
            ts.color_modifier1 == CMod::SourceColor && ts.alpha_modifier1 == AMod::SourceAlpha &&
            ts.color_modifier2 == CMod::SourceColor && ts.alpha_modifier2 == AMod::SourceAlpha &&
            ((ts.color_source1 == Src::Texture0 && ts.color_source2 == Src::PrimaryColor) ||
             (ts.color_source1 == Src::PrimaryColor && ts.color_source2 == Src::Texture0)) &&
            ts.alpha_source1 == ts.color_source1 && ts.alpha_source2 == ts.color_source2 &&
            textures[0].enabled &&
            textures[0].config.type == TexturingRegs::TextureConfig::Texture2D &&
            tex_planes[0] != nullptr) {
            return SpanKernel::ModulateTex0;
        }
        return SpanKernel::None;
    }();

#if CITRA_ARCH(arm32)
    // Why a draw fell off the 8-wide path, for the gate census below. Every early return of
    // the selection lambda records one; with the census compiled out the stores are dead and
    // the compiler drops them.
    enum class NeonReject : u32 {
        None,
        ScissorExclude,
        Lighting,
        Fog,
        Stencil,
        FragmentOp,
        BlendOff,
        TexNot2D,
        TexPlane,
        WrapNpotRepeat,
        WrapOther,
        Dot3,
        Tex3,
        Count,
    };
    [[maybe_unused]] NeonReject neon_reject = NeonReject::None;
    // For TexNot2D/TexPlane/Wrap*: (unit << 4) | texture type, so the census can say which
    // unit and which type paid.
    [[maybe_unused]] u32 neon_reject_detail = 0;

    // Stage patterns the fused kernels miss run through the
    // 8-wide generic TEV interpreter, under the same pipeline gates as the kernels (checked
    // at the top of the selection lambda: no lighting/fog/stencil, plain fragment mode,
    // blending on). Additionally every enabled unit must be a plain 2D texture with a
    // resolved plane and a vectorizable wrap mode, and Dot3/Texture3 stay scalar.
    const SpanKernel effective_kernel = [&] {
        if (span_kernel != SpanKernel::None) {
            return span_kernel;
        }
        using Op = TexturingRegs::TevStageConfig::Operation;
        using Src = TexturingRegs::TevStageConfig::Source;
        const auto& om = regs.framebuffer.output_merger;
        // Fragment lighting is allowed here (P6): the interpreter evaluates it at the span
        // block endpoints and interpolates between them - the plan's accuracy trade. The
        // exact per-pixel path stays available for the diff harness via the env below.
#ifdef CITRA_TRACE_PROBES
        static const bool exact_lighting = std::getenv("AZAHAR_EXACT_LIGHTING") != nullptr;
        static const bool gen_off = std::getenv("AZAHAR_NEON_GEN_OFF") != nullptr;
        if (gen_off) {
            return SpanKernel::None;
        }
#else
        constexpr bool exact_lighting = false;
#endif
        // Fog is no longer a gate: the block applies it in the epilogue, where the scalar
        // loop applies it. It used to reject 31% of this scene's pixel area overall and
        // about half of it in gameplay, and every one of those pixels then paid per-pixel
        // TEV and lighting in the scalar loop.
        if (scissor_exclude || (!regs.lighting.disable && exact_lighting) ||
            om.fragment_operation_mode != FramebufferRegs::FragmentOperationMode::Default ||
            !om.alphablend_enable) {
            neon_reject =
                scissor_exclude ? NeonReject::ScissorExclude
                : (!regs.lighting.disable && exact_lighting)
                    ? NeonReject::Lighting
                : om.fragment_operation_mode != FramebufferRegs::FragmentOperationMode::Default
                    ? NeonReject::FragmentOp
                    : NeonReject::BlendOff;
            return SpanKernel::None;
        }
        for (u32 i = 0; i < 3; ++i) {
            const auto& t = textures[i];
            if (!t.enabled) {
                continue;
            }
            // Only unit 0 respects the texturing type (3DBrew, and TextureColor follows it):
            // units 1 and 2 sample as plain 2D whatever their type field says. Projection2D
            // is handled below by dividing the coordinates by the interpolated tc0_w.
            const bool type_ok =
                i != 0 || t.config.type == TexturingRegs::TextureConfig::Texture2D ||
                t.config.type == TexturingRegs::TextureConfig::Projection2D;
            // Address zero is a texture the scalar path answers with opaque black instead of
            // sampling; the plane would be meaningless.
            // Wrap modes are all handled: the two that map to a vector op stay 8-wide, the
            // rest wrap per lane on the scalar helper and rejoin the block.
            if (!type_ok || t.config.address == 0 || tex_planes[i] == nullptr) {
                neon_reject = !type_ok ? NeonReject::TexNot2D : NeonReject::TexPlane;
                neon_reject_detail = (i << 4) | (static_cast<u32>(t.config.type.Value()) & 0xF);
                return SpanKernel::None;
            }
        }
        for (u32 i = 0; i < 6; ++i) {
            if (tev_skip[i]) {
                continue;
            }
            const auto& s = tev_stages[i];
            if (s.color_op == Op::Dot3_RGB || s.color_op == Op::Dot3_RGBA) {
                neon_reject = NeonReject::Dot3;
                return SpanKernel::None;
            }
            const auto uses_tex3 = [](Src src) { return src == Src::Texture3; };
            if (uses_tex3(s.color_source1) || uses_tex3(s.color_source2) ||
                uses_tex3(s.color_source3) || uses_tex3(s.alpha_source1) ||
                uses_tex3(s.alpha_source2) || uses_tex3(s.alpha_source3)) {
                neon_reject = NeonReject::Tex3;
                return SpanKernel::None;
            }
        }
        return SpanKernel::GenericNeon;
    }();

#ifdef CITRA_TRACE_PROBES
    // Gate census (AZAHAR_SW_GATES=1): pixel area rejected by each gate, so the widening work
    // is aimed at what actually costs. Weighted by triangle area and counted once per
    // triangle (every stripe of a triangle reaches the same verdict).
    {
        static const bool gate_census = std::getenv("AZAHAR_SW_GATES") != nullptr;
        if (gate_census && stripe == 0) {
            static std::array<std::atomic<u64>, static_cast<u32>(NeonReject::Count)> rejected{};
            static std::array<std::atomic<u64>, 64> detail{};
            static std::atomic<u64> accepted_generic{0}, accepted_fused{0}, dump_gate{0};
            const u64 area = static_cast<u64>(
                std::abs(SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), vtxpos[2].xy())) >> 9);
            if (effective_kernel == SpanKernel::GenericNeon) {
                accepted_generic.fetch_add(area, std::memory_order_relaxed);
            } else if (effective_kernel != SpanKernel::None) {
                accepted_fused.fetch_add(area, std::memory_order_relaxed);
            } else {
                rejected[static_cast<u32>(neon_reject)].fetch_add(area,
                                                                  std::memory_order_relaxed);
                if (neon_reject_detail != 0) {
                    detail[neon_reject_detail & 0x3F].fetch_add(area, std::memory_order_relaxed);
                }
            }
            u64 total = accepted_generic.load() + accepted_fused.load();
            for (const auto& r : rejected) {
                total += r.load();
            }
            u64 last = dump_gate.load();
            if (total - last > 20000000 && dump_gate.compare_exchange_strong(last, total)) {
                static constexpr std::array<const char*, static_cast<u32>(NeonReject::Count)>
                    names{"unknown",  "scissor_exclude", "lighting", "fog",
                          "stencil",  "fragment_op",     "blend_off", "tex_not_2d",
                          "tex_plane", "wrap_npot_repeat", "wrap_other", "dot3",
                          "tex3"};
                std::string line;
                for (u32 i = 0; i < static_cast<u32>(NeonReject::Count); ++i) {
                    const u64 v = rejected[i].load();
                    if (v != 0) {
                        line += fmt::format(" {}={}", names[i], v);
                    }
                }
                for (u32 i = 0; i < detail.size(); ++i) {
                    const u64 v = detail[i].load();
                    if (v != 0) {
                        line += fmt::format(" unit{}type{}={}", i >> 4, i & 0xF, v);
                    }
                }
                LOG_INFO(HW_GPU, "SWGATES total={} generic={} fused={}{}", total,
                         accepted_generic.load(), accepted_fused.load(), line);
            }
        }
    }
#endif // CITRA_TRACE_PROBES
#else
    const SpanKernel effective_kernel = span_kernel;
#endif

    // Enter rasterization loop, starting at the center of the topleft bounding box corner.
    {
        // Everything below is constant for the triangle: the register-derived pipeline
        // state, the vertex attributes as floats, and the decoded TEV descriptor. It used to
        // be rebuilt inside process_scanline, once per scanline of every triangle.
        const float depth_scale =
            f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();
        const float depth_offset =
            f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();
        const bool w_buffering = regs.rasterizer.depthmap_enable ==
                                 Pica::RasterizerRegs::DepthBuffering::WBuffering;
        const auto& ts0c = tev_stages[0];
        const auto& ts1c = tev_stages[1];
        const Common::Vec4<u8> const0 = Common::MakeVec(ts0c.const_r.Value(),
                                                        ts0c.const_g.Value(),
                                                        ts0c.const_b.Value(),
                                                        ts0c.const_a.Value())
                                            .Cast<u8>();
        const Common::Vec4<u8> const1 = Common::MakeVec(ts1c.const_r.Value(),
                                                        ts1c.const_g.Value(),
                                                        ts1c.const_b.Value(),
                                                        ts1c.const_a.Value())
                                            .Cast<u8>();
#if CITRA_ARCH(arm32)
        // Lighting reads a texture only for shadow attenuation and bump mapping;
        // without those an endpoint may be evaluated at a pixel the block did not
        // gather texels for.
        // Largest channel gap between a block's lighting endpoints that may be spanned by a
        // straight line. Above it the block subdivides.
        // A block takes its lighting from endpoints sixteen pixels apart and never
        // subdivides. Against evaluating the model per pixel (1.55x the frame time) this moves
        // 1.8% of a fight frame's bytes: a thin highlight running a little brighter along a
        // platform edge, invisible at 1:1.
        constexpr int lit_split_threshold = 256;
        constexpr bool lit_may_split = lit_split_threshold >= 0 && lit_split_threshold <= 255;
        const bool lighting_needs_texture =
            regs.lighting.config0.enable_shadow ||
            regs.lighting.config0.bump_mode !=
                Pica::LightingRegs::LightingBumpMode::None;
        // NEON block path: interpolation, texturing and
        // the combine run 8 pixels at a time; alpha/depth tests and the write stay on
        // the scalar helpers so stencil side effects and blend keep one implementation.
        // The float math mirrors the scalar path operation for operation (f24 carries
        // full f32 precision, so order alone decides the bits); the reciprocal uses a
        // real VFP divide per lane, not vrecpe, for the same reason.
        const bool neon_wrap_ok = [&] {
            if (span_kernel == SpanKernel::FlatColor) {
                return true;
            }
            const auto& cfg = textures[0].config;
            const auto wrap_ok = [](TexturingRegs::TextureConfig::WrapMode m, u32 dim) {
                if (m == TexturingRegs::TextureConfig::ClampToEdge) {
                    return true;
                }
                if (m == TexturingRegs::TextureConfig::Repeat) {
                    return (dim & (dim - 1)) == 0 && dim != 0;
                }
                return false;
            };
            return wrap_ok(cfg.wrap_s, cfg.width) && wrap_ok(cfg.wrap_t, cfg.height);
        }();
        const bool neon_kernel_ok =
            effective_kernel == SpanKernel::GenericNeon ||
            (neon_wrap_ok && (span_kernel == SpanKernel::FlatColor ||
                              span_kernel == SpanKernel::ModulateTex0 ||
                              span_kernel == SpanKernel::ReplaceTex0 ||
                              span_kernel == SpanKernel::PrimAlphaTex0));
#ifdef CITRA_TRACE_PROBES
        static const bool neon_off = std::getenv("AZAHAR_NEON_OFF") != nullptr;
#else
        constexpr bool neon_off = false;
#endif
        const float32_t wi_x = w_inverse.x.ToFloat32();
        const float32_t wi_y = w_inverse.y.ToFloat32();
        const float32_t wi_z = w_inverse.z.ToFloat32();
        // Per-channel vertex attributes as floats (color scaled later, uv raw).
        const float32_t c0r = v0.color.r().ToFloat32(), c1r = v1.color.r().ToFloat32(),
                        c2r = v2.color.r().ToFloat32();
        const float32_t c0g = v0.color.g().ToFloat32(), c1g = v1.color.g().ToFloat32(),
                        c2g = v2.color.g().ToFloat32();
        const float32_t c0b = v0.color.b().ToFloat32(), c1b = v1.color.b().ToFloat32(),
                        c2b = v2.color.b().ToFloat32();
        const float32_t c0a = v0.color.a().ToFloat32(), c1a = v1.color.a().ToFloat32(),
                        c2a = v2.color.a().ToFloat32();
        const float32_t u0 = v0.tc0.u().ToFloat32(), u1 = v1.tc0.u().ToFloat32(),
                        u2 = v2.tc0.u().ToFloat32();
        const float32_t vt0 = v0.tc0.v().ToFloat32(), vt1 = v1.tc0.v().ToFloat32(),
                        vt2 = v2.tc0.v().ToFloat32();
        const float32_t z0 = v0.screenpos[2].ToFloat32(),
                        z1 = v1.screenpos[2].ToFloat32(),
                        z2 = v2.screenpos[2].ToFloat32();
        const auto& tex = textures[0];
        const float32_t twf = static_cast<float32_t>(tex.config.width);
        const float32_t thf = static_cast<float32_t>(tex.config.height);
        const s32 tex_w = static_cast<s32>(tex.config.width);
        const s32 tex_h = static_cast<s32>(tex.config.height);
        const bool s_clamp = tex.config.wrap_s == TexturingRegs::TextureConfig::ClampToEdge;
        const bool t_clamp = tex.config.wrap_t == TexturingRegs::TextureConfig::ClampToEdge;
        const u32* plane = tex_planes[0];
        const u32 plane_w = tex_plane_width[0];

        // ---- P4 generic-interpreter preparation (unused by the fused kernels) ----
        // Vertex uv floats for all three coordinate sets.
        const float32_t uc[3][3] = {
            {v0.tc0.u().ToFloat32(), v1.tc0.u().ToFloat32(), v2.tc0.u().ToFloat32()},
            {v0.tc1.u().ToFloat32(), v1.tc1.u().ToFloat32(), v2.tc1.u().ToFloat32()},
            {v0.tc2.u().ToFloat32(), v1.tc2.u().ToFloat32(), v2.tc2.u().ToFloat32()},
        };
        const float32_t vc[3][3] = {
            {v0.tc0.v().ToFloat32(), v1.tc0.v().ToFloat32(), v2.tc0.v().ToFloat32()},
            {v0.tc1.v().ToFloat32(), v1.tc1.v().ToFloat32(), v2.tc1.v().ToFloat32()},
            {v0.tc2.v().ToFloat32(), v1.tc2.v().ToFloat32(), v2.tc2.v().ToFloat32()},
        };
        // The projective divisor, interpolated like any other attribute.
        const float32_t wq[3] = {v0.tc0_w.ToFloat32(), v1.tc0_w.ToFloat32(),
                                 v2.tc0_w.ToFloat32()};
        const auto& nunits = draw_units;
        const auto& nstages = draw_stages;
        const u32 num_nstages = draw_num_stages;

        const uint8x8_t nbuf_r0 =
            vdup_n_u8(static_cast<u8>(regs.texturing.tev_combiner_buffer_color.r));
        const uint8x8_t nbuf_g0 =
            vdup_n_u8(static_cast<u8>(regs.texturing.tev_combiner_buffer_color.g));
        const uint8x8_t nbuf_b0 =
            vdup_n_u8(static_cast<u8>(regs.texturing.tev_combiner_buffer_color.b));
        const uint8x8_t nbuf_a0 =
            vdup_n_u8(static_cast<u8>(regs.texturing.tev_combiner_buffer_color.a));
        // ---- end P4 preparation ----

        // Left-assoc dot product, one multiply and one add at a time, no fusing:
        // matches ((x*bx + y*by) + z*bz) of the scalar Dot exactly.
        const auto dot3 = [](float32x4_t b0, float32x4_t b1, float32x4_t b2,
                             float32_t s0, float32_t s1, float32_t s2) {
            float32x4_t acc = vmulq_n_f32(b0, s0);
            acc = vaddq_f32(acc, vmulq_n_f32(b1, s1));
            return vaddq_f32(acc, vmulq_n_f32(b2, s2));
        };
        // Reciprocal, the block's only division.
        //
        // The exact form is four scalar VFP divides. On this core that is affordable; on the
        // in-order VFP of a Cortex-A9 - the reason this renderer exists - VDIV.F32 is about
        // fifteen cycles and does not pipeline, so a projected draw's twenty-four divides per
        // eight pixels cost more than the rest of the block put together.
        //
        // The estimate form is vrecpe plus one Newton-Raphson step: four pipelined NEON
        // instructions for all four lanes, about sixteen mantissa bits. That is roughly six
        // bits of subtexel precision at the largest texture size, so it only ever moves a
        // texel that sits on an exact boundary - 79 pixels of a 192000-pixel fight frame.
        // A second `r = vmulq_f32(vrecpsq_f32(d, r), r);` is the middle ground if a title
        // ever shows artefacts.
        const auto recip4 = [&](float32x4_t d) {
            float32x4_t r = vrecpeq_f32(d);
            r = vmulq_f32(vrecpsq_f32(d, r), r);
            return r;
        };
        const auto div4 = [&](float32x4_t a, float32x4_t b) {
            return vmulq_f32(a, recip4(b));
        };

        const int32x4_t w0_dx4 = vdupq_n_s32(w0_dx);
        const int32x4_t w1_dx4 = vdupq_n_s32(w1_dx);
        const int32x4_t w2_dx4 = vdupq_n_s32(w2_dx);
        const int32x4_t lane_idx = {0, 1, 2, 3};
#endif // CITRA_ARCH(arm32)

        u64 shaded_pixels = 0;
        const auto process_scanline = [&](u16 y) {
            const u16 x_first = min_x + 8;
            const s32 w0_row = bias0 + SignedArea(vtxpos[1].xy(), vtxpos[2].xy(), {x_first, y});
            const s32 w1_row = bias1 + SignedArea(vtxpos[2].xy(), vtxpos[0].xy(), {x_first, y});
            const s32 w2_row = bias2 + SignedArea(vtxpos[0].xy(), vtxpos[1].xy(), {x_first, y});

            // Solve w_i(k) = w_row_i + k * w_dx_i >= 0 for the pixel index k.
            s32 k_min = 0;
            s32 k_max = (static_cast<s32>(max_x) - 1 - static_cast<s32>(x_first)) / 0x10;
            const auto restrict_range = [&](s32 w_row, s32 w_dx) {
                if (w_dx > 0) {
                    if (w_row < 0) {
                        // First k with w >= 0: ceil(-w_row / w_dx).
                        k_min = std::max(k_min, (-w_row + w_dx - 1) / w_dx);
                    }
                } else if (w_dx < 0) {
                    if (w_row < 0) {
                        k_max = -1; // starts negative and only decreases
                    } else {
                        // Last k with w >= 0: floor(w_row / -w_dx).
                        k_max = std::min(k_max, w_row / -w_dx);
                    }
                } else if (w_row < 0) {
                    k_max = -1;
                }
            };
            restrict_range(w0_row, w0_dx);
            restrict_range(w1_row, w1_dx);
            restrict_range(w2_row, w2_dx);
            // Pixels this scanline will actually walk, after the edges have clipped it. Counting
            // the bounding span instead made the figure depend on how diagonal the triangles
            // were; see Common::PipelineStats::fragments.
            if (k_max >= k_min) {
                shaded_pixels += static_cast<u64>(k_max - k_min + 1);
            }

            s32 w0 = w0_row + k_min * w0_dx;
            s32 w1 = w1_row + k_min * w1_dx;
            s32 w2 = w2_row + k_min * w2_dx;

#ifdef CITRA_TRACE_PROBES
            // Dual-render arbiter: run the generic loop and the kernel on the same span and
            // compare framebuffer bytes. Immune to run-to-run simulation divergence, which
            // makes it the ground truth for kernel correctness (AZAHAR_KERNEL_VERIFY=1).
            static const bool kernel_verify = std::getenv("AZAHAR_KERNEL_VERIFY") != nullptr;
            // Lit GenericNeon spans interpolate lighting between block endpoints (P6) and
            // diverge from the per-pixel reference by design; they are validated visually
            // (or exactly, with AZAHAR_EXACT_LIGHTING which routes them scalar).
            const bool verify_this_span = kernel_verify &&
                                          effective_kernel != SpanKernel::None &&
                                          regs.lighting.disable && k_max >= k_min;
            // Three words per pixel: colour, depth, stencil. Stencil is snapshotted and
            // restored like the other two because the depth/stencil test writes it, so the
            // generic re-render must start from the same stencil the kernel pass saw. Only
            // D24S8 has one; the other formats warn on every access.
            const bool verify_stencil = regs.framebuffer.framebuffer.depth_format ==
                                        FramebufferRegs::DepthFormat::D24S8;
            std::vector<u32> verify_before, verify_generic;
            if (verify_this_span) {
                verify_before.resize(static_cast<std::size_t>(k_max - k_min + 1) * 3);
                verify_generic.reserve(verify_before.size());
                for (s32 k = k_min; k <= k_max; k++) {
                    const u16 x = static_cast<u16>(x_first + (k << 4));
                    const auto c0 = fb.GetPixel(x >> 4, y >> 4);
                    verify_before[(k - k_min) * 3] = static_cast<u32>(c0.r()) | (c0.g() << 8) |
                                                     (c0.b() << 16) |
                                                     (static_cast<u32>(c0.a()) << 24);
                    verify_before[(k - k_min) * 3 + 1] = fb.GetDepth(x >> 4, y >> 4);
                    verify_before[(k - k_min) * 3 + 2] =
                        verify_stencil ? fb.GetStencil(x >> 4, y >> 4) : 0;
                }
            }
#endif // CITRA_TRACE_PROBES
            if (effective_kernel != SpanKernel::None) {
                s32 k_scalar = k_min;
#if CITRA_ARCH(arm32)
                // Lighting endpoint carried from the previous 8-pixel block of this span.
                // Lighting endpoints carried along this span. Default spacing is sixteen
                // pixels, shared by two blocks; the subdividing modes use eight.
                std::pair<Common::Vec4<u8>, Common::Vec4<u8>> lit_a{}, lit_b{};
                int lit_off = 0;
                bool lit_valid = false;
#endif
                // Whether the NEON path fully rendered the span. A GenericNeon span that the
                // NEON block skipped (kill switch, spans under 8 pixels) must fall through to
                // the generic loop below - the scalar kernel loop has no case for it.
                bool neon_covered = false;
#if CITRA_ARCH(arm32)
                // The fused kernels leave tails to their scalar loop, so they need a full
                // block; the generic interpreter masks the invalid lanes of a partial block
                // and so takes spans of any length. Spans under eight pixels are common - thin
                // triangles, and the ends of every triangle - and were the last thing still
                // reaching the scalar loop.
                if (neon_kernel_ok && !neon_off &&
                    (effective_kernel == SpanKernel::GenericNeon || k_max - k_min + 1 >= 8)) {

                    // The generic interpreter renders partial tail blocks too (invalid lanes
                    // masked in the epilogue); the fused kernels leave tails to their scalar
                    // loop below.
                    while (effective_kernel == SpanKernel::GenericNeon ? k_scalar <= k_max
                                                                       : k_scalar + 7 <= k_max) {
                        const int lanes_valid = std::min<s32>(8, k_max - k_scalar + 1);
                        // Edge weights for lanes k..k+3 and k+4..k+7, exact in s32.
                        int32x4_t w0_lo = vmlaq_s32(vdupq_n_s32(w0), w0_dx4, lane_idx);
                        int32x4_t w1_lo = vmlaq_s32(vdupq_n_s32(w1), w1_dx4, lane_idx);
                        int32x4_t w2_lo = vmlaq_s32(vdupq_n_s32(w2), w2_dx4, lane_idx);
                        int32x4_t w0_hi = vaddq_s32(w0_lo, vshlq_n_s32(w0_dx4, 2));
                        int32x4_t w1_hi = vaddq_s32(w1_lo, vshlq_n_s32(w1_dx4, 2));
                        int32x4_t w2_hi = vaddq_s32(w2_lo, vshlq_n_s32(w2_dx4, 2));

                        alignas(16) u8 comb[8][4];
                        alignas(16) float32_t depth8[8];

                        if (effective_kernel == SpanKernel::GenericNeon) {
                            // ---- P4: 8-wide generic TEV in planar channels ----
                            int32x4_t prh[4][2]; // r,g,b,a x half
                            alignas(16) u32 gather[3][8];
                            for (int half = 0; half < 2; half++) {
                                const float32x4_t b0 =
                                    vcvtq_f32_s32(half == 0 ? w0_lo : w0_hi);
                                const float32x4_t b1 =
                                    vcvtq_f32_s32(half == 0 ? w1_lo : w1_hi);
                                const float32x4_t b2 =
                                    vcvtq_f32_s32(half == 0 ? w2_lo : w2_hi);
                                const float32x4_t denom = dot3(b0, b1, b2, wi_x, wi_y, wi_z);
                                const float32x4_t inv_w = recip4(denom);
                                const auto channel = [&](float32_t a0, float32_t a1,
                                                         float32_t a2) {
                                    float32x4_t n = dot3(b0, b1, b2, a0, a1, a2);
                                    n = vmulq_f32(n, inv_w);
                                    n = vaddq_f32(vmulq_n_f32(n, 255.0f), vdupq_n_f32(0.5f));
                                    return vcvtq_s32_f32(n);
                                };
                                prh[0][half] = channel(c0r, c1r, c2r);
                                prh[1][half] = channel(c0g, c1g, c2g);
                                prh[2][half] = channel(c0b, c1b, c2b);
                                prh[3][half] = channel(c0a, c1a, c2a);
                                // Texture coordinates and gathers per enabled unit.
                                for (u32 ui = 0; ui < 3; ++ui) {
                                    const SwTexUnit& un = nunits[ui];
                                    if (!un.enabled) {
                                        continue;
                                    }
                                    const int c = un.coord;
                                    // A projected coordinate is (Su*b / w) / (Swq*b / w), so
                                    // the perspective divide cancels: one reciprocal of the
                                    // wq numerator replaces the inv_w multiply and the divide,
                                    // for both coordinates. Three reciprocals become one.
                                    const float32x4_t scale =
                                        un.project ? recip4(dot3(b0, b1, b2, wq[0], wq[1], wq[2]))
                                                   : inv_w;
                                    const float32x4_t uu = vmulq_f32(
                                        dot3(b0, b1, b2, uc[c][0], uc[c][1], uc[c][2]), scale);
                                    const float32x4_t vv = vmulq_f32(
                                        dot3(b0, b1, b2, vc[c][0], vc[c][1], vc[c][2]), scale);
                                    int32x4_t ts4 = vcvtq_s32_f32(vmulq_n_f32(uu, un.wf));
                                    int32x4_t tt4 = vcvtq_s32_f32(vmulq_n_f32(vv, un.hf));
                                    if (!un.fast_wrap) [[unlikely]] {
                                        // Border and modulo wrapping have no vector form, so
                                        // these four lanes wrap on the same scalar helper the
                                        // generic loop uses and rejoin the block. Keeping the
                                        // rest of the pipeline 8-wide is worth far more than
                                        // the vector wrap it gives up.
                                        alignas(16) s32 sv[4], tv[4];
                                        vst1q_s32(sv, ts4);
                                        vst1q_s32(tv, tt4);
                                        using WM = TexturingRegs::TextureConfig;
                                        for (int l = 0; l < 4; ++l) {
                                            bool border = false;
                                            if (un.wrap_s == WM::ClampToBorder) {
                                                border |= sv[l] < 0 || sv[l] >= un.w;
                                            } else if (un.wrap_s == WM::ClampToBorder2) {
                                                border |= sv[l] >= un.w;
                                            }
                                            if (un.wrap_t == WM::ClampToBorder) {
                                                border |= tv[l] < 0 || tv[l] >= un.h;
                                            } else if (un.wrap_t == WM::ClampToBorder2) {
                                                border |= tv[l] >= un.h;
                                            }
                                            if (border) {
                                                gather[ui][half * 4 + l] = un.border;
                                                continue;
                                            }
                                            const s32 ws = GetWrappedTexCoord(
                                                un.wrap_s, sv[l], static_cast<u32>(un.w));
                                            const s32 wt =
                                                un.h - 1 -
                                                GetWrappedTexCoord(un.wrap_t, tv[l],
                                                                   static_cast<u32>(un.h));
                                            gather[ui][half * 4 + l] =
                                                tex_planes[ui][static_cast<std::size_t>(wt) *
                                                             tex_plane_width[ui] +
                                                         ws];
                                        }
                                        continue;
                                    }
                                    if (un.s_clamp) {
                                        ts4 = vmaxq_s32(vminq_s32(ts4, vdupq_n_s32(un.w - 1)),
                                                        vdupq_n_s32(0));
                                    } else {
                                        ts4 = vandq_s32(ts4, vdupq_n_s32(un.w - 1));
                                    }
                                    if (un.t_clamp) {
                                        tt4 = vmaxq_s32(vminq_s32(tt4, vdupq_n_s32(un.h - 1)),
                                                        vdupq_n_s32(0));
                                    } else {
                                        tt4 = vandq_s32(tt4, vdupq_n_s32(un.h - 1));
                                    }
                                    tt4 = vsubq_s32(vdupq_n_s32(un.h - 1), tt4);
                                    const int32x4_t idx = vmlaq_s32(
                                        ts4, tt4, vdupq_n_s32(static_cast<s32>(tex_plane_width[ui])));
                                    alignas(16) s32 idx4[4];
                                    vst1q_s32(idx4, idx);
                                    gather[ui][half * 4 + 0] = tex_planes[ui][idx4[0]];
                                    gather[ui][half * 4 + 1] = tex_planes[ui][idx4[1]];
                                    gather[ui][half * 4 + 2] = tex_planes[ui][idx4[2]];
                                    gather[ui][half * 4 + 3] = tex_planes[ui][idx4[3]];
                                }
                                // Depth: same expressions as the scalar path.
                                float32x4_t zow = dot3(b0, b1, b2, z0, z1, z2);
                                zow = vmulq_n_f32(zow, inv_wsum);
                                float32x4_t d = vaddq_f32(vmulq_n_f32(zow, depth_scale),
                                                          vdupq_n_f32(depth_offset));
                                if (w_buffering) {
                                    d = vmulq_f32(d, vmulq_n_f32(inv_w, wsum_tri));
                                }
                                d = vminq_f32(vmaxq_f32(d, vdupq_n_f32(0.0f)),
                                              vdupq_n_f32(1.0f));
                                vst1q_f32(&depth8[half * 4], d);
                            }
                            // Planar channel vectors.
                            const auto narrow8 = [](int32x4_t lo, int32x4_t hi) {
                                return vqmovn_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)));
                            };
                            struct Rgba8 {
                                uint8x8_t r, g, b, a;
                            };
                            const Rgba8 prim{narrow8(prh[0][0], prh[0][1]),
                                             narrow8(prh[1][0], prh[1][1]),
                                             narrow8(prh[2][0], prh[2][1]),
                                             narrow8(prh[3][0], prh[3][1])};
                            const Rgba8 zero8{vdup_n_u8(0), vdup_n_u8(0), vdup_n_u8(0),
                                              vdup_n_u8(0)};
                            Rgba8 texv[3] = {zero8, zero8, zero8};
                            for (u32 ui = 0; ui < 3; ++ui) {
                                if (nunits[ui].enabled) {
                                    const uint8x8x4_t t = vld4_u8(
                                        reinterpret_cast<const u8*>(&gather[ui][0]));
                                    texv[ui] = {t.val[0], t.val[1], t.val[2], t.val[3]};
                                }
                            }
                            // Fragment lighting (P6): evaluate the full scalar model at the
                            // first and last valid lane, interpolate linearly between them.
                            Rgba8 fragprim = zero8;
                            Rgba8 fragsec = zero8;
                            if (!regs.lighting.disable) {
                                const auto lit_at = [&](s32 lane) {
                                    const s32 lw0 = w0 + lane * w0_dx;
                                    const s32 lw1 = w1 + lane * w1_dx;
                                    const s32 lw2 = w2 + lane * w2_dx;
                                    const auto bc =
                                        Common::MakeVec(f24::FromFloat32(static_cast<f32>(lw0)),
                                                        f24::FromFloat32(static_cast<f32>(lw1)),
                                                        f24::FromFloat32(static_cast<f32>(lw2)));
                                    const f24 invw = f24::One() / Common::Dot(w_inverse, bc);
                                    const auto at = [&](f24 a0, f24 a1, f24 a2) {
                                        return (Common::Dot(Common::MakeVec(a0, a1, a2), bc) *
                                                invw)
                                            .ToFloat32();
                                    };
                                    const auto nq =
                                        Common::Quaternion<f32>{
                                            {at(v0.quat.x, v1.quat.x, v2.quat.x),
                                             at(v0.quat.y, v1.quat.y, v2.quat.y),
                                             at(v0.quat.z, v1.quat.z, v2.quat.z)},
                                            at(v0.quat.w, v1.quat.w, v2.quat.w)}
                                            .Normalized();
                                    const Common::Vec3f vw{at(v0.view.x, v1.view.x, v2.view.x),
                                                           at(v0.view.y, v1.view.y, v2.view.y),
                                                           at(v0.view.z, v1.view.z, v2.view.z)};
                                    std::array<Common::Vec4<u8>, 4> tc{};
                                    // Lane 8 is the next block's first pixel, evaluated only
                                    // when lighting ignores texture colours; nothing was
                                    // gathered for it.
                                    for (u32 ui = 0; ui < 3 && lane < 8; ++ui) {
                                        if (nunits[ui].enabled) {
                                            const u32 t = gather[ui][lane];
                                            tc[ui] = {static_cast<u8>(t), static_cast<u8>(t >> 8),
                                                      static_cast<u8>(t >> 16),
                                                      static_cast<u8>(t >> 24)};
                                        }
                                    }
                                    return ComputeFragmentsColors(draw_lighting, pica.lighting,
                                                                  nq, vw, tc);
                                };
                                // Endpoints. Unless lighting samples a texture (shadow or
                                // bump), the second endpoint is the pixel one past the block,
                                // which is the next block's first: carrying it over costs one
                                // evaluation per block instead of two, and makes the
                                // interpolation continuous across block boundaries instead of
                                // stepping at each one.
                                std::pair<Common::Vec4<u8>, Common::Vec4<u8>> ends, endB;
                                int wspan, woff = 0;
                                if (lighting_needs_texture) {
                                    ends = lit_at(0);
                                    endB = lanes_valid > 1 ? lit_at(lanes_valid - 1) : ends;
                                    wspan = lanes_valid - 1;
                                } else if (lit_may_split) {
                                    // Eight-pixel spacing, so the midpoint a split needs is
                                    // lane 4 of this block.
                                    if (!lit_valid) {
                                        lit_a = lit_at(0);
                                        lit_valid = true;
                                    }
                                    ends = lit_a;
                                    endB = lit_at(8);
                                    lit_a = endB;
                                    wspan = 8;
                                } else {
                                    // Sixteen-pixel spacing: a block either evaluates the far
                                    // endpoint or reuses the pair its predecessor built.
                                    if (!lit_valid) {
                                        lit_a = lit_at(0);
                                        lit_b = lit_at(16);
                                        lit_off = 0;
                                        lit_valid = true;
                                    } else if (lit_off == 0) {
                                        lit_off = 8;
                                    } else {
                                        lit_a = lit_b;
                                        lit_b = lit_at(16);
                                        lit_off = 0;
                                    }
                                    ends = lit_a;
                                    endB = lit_b;
                                    wspan = 16;
                                    woff = lit_off;
                                }
                                alignas(8) u8 wj8[8];
                                for (int j = 0; j < 8; j++) {
                                    wj8[j] = wspan > 0
                                                 ? static_cast<u8>(std::min(woff + j, wspan) *
                                                                   255 / wspan)
                                                 : 0;
                                }
                                uint8x8_t wj = vld1_u8(wj8);
                                const auto lerp8 = [&](uint8x8_t a, uint8x8_t b) {
                                    const uint16x8_t p0 = vmull_u8(a, vmvn_u8(wj));
                                    const uint16x8_t p1 = vmull_u8(b, wj);
                                    const uint32x4_t lo =
                                        vaddl_u16(vget_low_u16(p0), vget_low_u16(p1));
                                    const uint32x4_t hi =
                                        vaddl_u16(vget_high_u16(p0), vget_high_u16(p1));
                                    const auto d32 = [](uint32x4_t v) {
                                        uint32x4_t t = vaddq_u32(v, vshrq_n_u32(v, 8));
                                        t = vaddq_u32(t, vdupq_n_u32(1));
                                        return vshrq_n_u32(t, 8);
                                    };
                                    return vqmovn_u16(
                                        vcombine_u16(vqmovn_u32(d32(lo)), vqmovn_u32(d32(hi))));
                                };
                                // Lighting is not linear - a specular highlight can swing
                                // across a few pixels - so interpolating between the block's
                                // ends is only sound while the ends are close. When they are
                                // not, evaluate the middle and interpolate each half: one
                                // extra evaluation, and only on the blocks that need it.
                                const auto spread = [](const Common::Vec4<u8>& a,
                                                       const Common::Vec4<u8>& b) {
                                    return std::max(
                                        {std::abs(a.r() - b.r()), std::abs(a.g() - b.g()),
                                         std::abs(a.b() - b.b()), std::abs(a.a() - b.a())});
                                };
                                // The per-pixel form of the model, kept for the accuracy
                                // comparison; lit_split_threshold selects it at compile time.
                                if (lit_split_threshold < 0 && !lighting_needs_texture) {
                                    alignas(8) u8 pr[8], pg[8], pb[8], pa[8];
                                    alignas(8) u8 sr[8], sg[8], sb[8], sa[8];
                                    for (int j = 0; j < 8; j++) {
                                        const auto l = lit_at(j);
                                        pr[j] = l.first.r();
                                        pg[j] = l.first.g();
                                        pb[j] = l.first.b();
                                        pa[j] = l.first.a();
                                        sr[j] = l.second.r();
                                        sg[j] = l.second.g();
                                        sb[j] = l.second.b();
                                        sa[j] = l.second.a();
                                    }
                                    fragprim = {vld1_u8(pr), vld1_u8(pg), vld1_u8(pb),
                                                vld1_u8(pa)};
                                    fragsec = {vld1_u8(sr), vld1_u8(sg), vld1_u8(sb),
                                               vld1_u8(sa)};
                                } else {
                                    const bool split =
                                        !lighting_needs_texture &&
                                        (spread(ends.first, endB.first) > lit_split_threshold ||
                                         spread(ends.second, endB.second) > lit_split_threshold);
                                    // Lanes 0-3 take the first half's pair, lanes 4-7 the second.
                                    static constexpr uint8x8_t upper_half = {0, 0, 0, 0,
                                                                             255, 255, 255, 255};
                                    const auto pair = [&](u8 lo, u8 mid) {
                                        return vbsl_u8(upper_half, vdup_n_u8(mid), vdup_n_u8(lo));
                                    };
                                    if (split) {
                                        const auto mid = lit_at(4);
                                        for (int j = 0; j < 8; j++) {
                                            wj8[j] = static_cast<u8>((j & 3) * 255 / 4);
                                        }
                                        wj = vld1_u8(wj8);
                                        const auto chan = [&](u8 a, u8 m, u8 b) {
                                            return lerp8(pair(a, m), pair(m, b));
                                        };
                                        fragprim = {chan(ends.first.r(), mid.first.r(), endB.first.r()),
                                                    chan(ends.first.g(), mid.first.g(), endB.first.g()),
                                                    chan(ends.first.b(), mid.first.b(), endB.first.b()),
                                                    chan(ends.first.a(), mid.first.a(), endB.first.a())};
                                        fragsec = {
                                            chan(ends.second.r(), mid.second.r(), endB.second.r()),
                                            chan(ends.second.g(), mid.second.g(), endB.second.g()),
                                            chan(ends.second.b(), mid.second.b(), endB.second.b()),
                                            chan(ends.second.a(), mid.second.a(), endB.second.a())};
                                    } else {
                                        const auto chan = [&](u8 a, u8 b) {
                                            return lerp8(vdup_n_u8(a), vdup_n_u8(b));
                                        };
                                        fragprim = {chan(ends.first.r(), endB.first.r()),
                                                    chan(ends.first.g(), endB.first.g()),
                                                    chan(ends.first.b(), endB.first.b()),
                                                    chan(ends.first.a(), endB.first.a())};
                                        fragsec = {chan(ends.second.r(), endB.second.r()),
                                                   chan(ends.second.g(), endB.second.g()),
                                                   chan(ends.second.b(), endB.second.b()),
                                                   chan(ends.second.a(), endB.second.a())};
                                    }
                                }
                            }
                            Rgba8 prev = zero8;
                            Rgba8 buf_cur = zero8;
                            Rgba8 buf_next{nbuf_r0, nbuf_g0, nbuf_b0, nbuf_a0};
                            // Exact (a*b)/255 truncation on u16 products (max 65025).
                            const auto div255_16 = [](uint16x8_t x) {
                                uint16x8_t t = vaddq_u16(x, vshrq_n_u16(x, 8));
                                t = vaddq_u16(t, vdupq_n_u16(1));
                                return vshrn_n_u16(t, 8);
                            };
                            const auto mul255 = [&](uint8x8_t a, uint8x8_t b) {
                                return div255_16(vmull_u8(a, b));
                            };
                            // Exact trunc /255 for sums up to 130050 (needs 32 bits), with the
                            // final narrow saturating to 255 like the scalar min().
                            const auto div255_wide = [](uint16x8_t x) {
                                const auto d32 = [](uint32x4_t v) {
                                    uint32x4_t t = vaddq_u32(v, vshrq_n_u32(v, 8));
                                    t = vaddq_u32(t, vdupq_n_u32(1));
                                    return vshrq_n_u32(t, 8);
                                };
                                const uint32x4_t lo = d32(vmovl_u16(vget_low_u16(x)));
                                const uint32x4_t hi = d32(vmovl_u16(vget_high_u16(x)));
                                return vqmovn_u16(
                                    vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi)));
                            };
                            const auto combine_ch = [&](u32 op, uint8x8_t a, uint8x8_t b,
                                                        uint8x8_t c) -> uint8x8_t {
                                switch (op) {
                                case 0: // Replace
                                    return a;
                                case 1: // Modulate
                                    return mul255(a, b);
                                case 2: // Add
                                    return vqadd_u8(a, b);
                                case 3: { // AddSigned
                                    const int16x8_t s = vreinterpretq_s16_u16(vaddl_u8(a, b));
                                    return vqmovun_s16(vsubq_s16(s, vdupq_n_s16(128)));
                                }
                                case 4: { // Lerp: (a*c + b*(255-c))/255
                                    const uint16x8_t p0 = vmull_u8(a, c);
                                    const uint16x8_t p1 = vmull_u8(b, vmvn_u8(c));
                                    // Sum can exceed 16 bits; widen inside div255_wide via
                                    // halving trick: split the sum into u32.
                                    const uint32x4_t lo =
                                        vaddl_u16(vget_low_u16(p0), vget_low_u16(p1));
                                    const uint32x4_t hi =
                                        vaddl_u16(vget_high_u16(p0), vget_high_u16(p1));
                                    const auto d32 = [](uint32x4_t v) {
                                        uint32x4_t t = vaddq_u32(v, vshrq_n_u32(v, 8));
                                        t = vaddq_u32(t, vdupq_n_u32(1));
                                        return vshrq_n_u32(t, 8);
                                    };
                                    return vqmovn_u16(
                                        vcombine_u16(vqmovn_u32(d32(lo)), vqmovn_u32(d32(hi))));
                                }
                                case 5: // Subtract
                                    return vqsub_u8(a, b);
                                case 8: { // MultiplyThenAdd: (a*b + 255*c)/255, min 255
                                    const uint16x8_t p0 = vmull_u8(a, b);
                                    const uint16x8_t p1 = vmull_u8(c, vdup_n_u8(255));
                                    const uint32x4_t lo =
                                        vaddl_u16(vget_low_u16(p0), vget_low_u16(p1));
                                    const uint32x4_t hi =
                                        vaddl_u16(vget_high_u16(p0), vget_high_u16(p1));
                                    const auto d32 = [](uint32x4_t v) {
                                        uint32x4_t t = vaddq_u32(v, vshrq_n_u32(v, 8));
                                        t = vaddq_u32(t, vdupq_n_u32(1));
                                        return vshrq_n_u32(t, 8);
                                    };
                                    return vqmovn_u16(
                                        vcombine_u16(vqmovn_u32(d32(lo)), vqmovn_u32(d32(hi))));
                                }
                                case 9: // AddThenMultiply
                                    return mul255(vqadd_u8(a, b), c);
                                default:
                                    return a;
                                }
                            };
                            for (u32 si = 0; si < num_nstages; ++si) {
                                const SwTevStage& st = nstages[si];
                                const auto src_of = [&](u32 id) -> Rgba8 {
                                    switch (id) {
                                    case 0:
                                        return prim;
                                    case 1: // PrimaryFragmentColor (endpoint-lerped lighting)
                                        return fragprim;
                                    case 2: // SecondaryFragmentColor
                                        return fragsec;
                                    case 3:
                                        return texv[0];
                                    case 4:
                                        return texv[1];
                                    case 5:
                                        return texv[2];
                                    case 13:
                                        return buf_cur;
                                    case 14:
                                        return Rgba8{vdup_n_u8(st.kr), vdup_n_u8(st.kg),
                                                     vdup_n_u8(st.kb), vdup_n_u8(st.ka)};
                                    case 15:
                                        return prev;
                                    default:
                                        return zero8;
                                    }
                                };
                                const auto cmod = [&](u32 mod, const Rgba8& s) -> Rgba8 {
                                    switch (mod) {
                                    case 0:
                                        return {s.r, s.g, s.b, s.a};
                                    case 1:
                                        return {vmvn_u8(s.r), vmvn_u8(s.g), vmvn_u8(s.b), s.a};
                                    case 2:
                                        return {s.a, s.a, s.a, s.a};
                                    case 3: {
                                        const uint8x8_t n = vmvn_u8(s.a);
                                        return {n, n, n, n};
                                    }
                                    case 4:
                                        return {s.r, s.r, s.r, s.r};
                                    case 5: {
                                        const uint8x8_t n = vmvn_u8(s.r);
                                        return {n, n, n, n};
                                    }
                                    case 8:
                                        return {s.g, s.g, s.g, s.g};
                                    case 9: {
                                        const uint8x8_t n = vmvn_u8(s.g);
                                        return {n, n, n, n};
                                    }
                                    case 12:
                                        return {s.b, s.b, s.b, s.b};
                                    case 13: {
                                        const uint8x8_t n = vmvn_u8(s.b);
                                        return {n, n, n, n};
                                    }
                                    default:
                                        return {s.r, s.g, s.b, s.a};
                                    }
                                };
                                const auto amod = [&](u32 mod, const Rgba8& s) -> uint8x8_t {
                                    switch (mod) {
                                    case 0:
                                        return s.a;
                                    case 1:
                                        return vmvn_u8(s.a);
                                    case 2:
                                        return s.r;
                                    case 3:
                                        return vmvn_u8(s.r);
                                    case 4:
                                        return s.g;
                                    case 5:
                                        return vmvn_u8(s.g);
                                    case 6:
                                        return s.b;
                                    case 7:
                                        return vmvn_u8(s.b);
                                    default:
                                        return s.a;
                                    }
                                };
                                const Rgba8 in0 = cmod(st.cm[0], src_of(st.cs[0]));
                                const Rgba8 in1 =
                                    st.cn > 1 ? cmod(st.cm[1], src_of(st.cs[1])) : zero8;
                                const Rgba8 in2 =
                                    st.cn > 2 ? cmod(st.cm[2], src_of(st.cs[2])) : zero8;
                                uint8x8_t out_r = combine_ch(st.cop, in0.r, in1.r, in2.r);
                                uint8x8_t out_g = combine_ch(st.cop, in0.g, in1.g, in2.g);
                                uint8x8_t out_b = combine_ch(st.cop, in0.b, in1.b, in2.b);
                                uint8x8_t out_a;
                                {
                                    const uint8x8_t a0 = amod(st.am[0], src_of(st.as[0]));
                                    const uint8x8_t a1 =
                                        st.an > 1 ? amod(st.am[1], src_of(st.as[1]))
                                                  : vdup_n_u8(0);
                                    const uint8x8_t a2 =
                                        st.an > 2 ? amod(st.am[2], src_of(st.as[2]))
                                                  : vdup_n_u8(0);
                                    out_a = combine_ch(st.aop, a0, a1, a2);
                                }
                                // Multiplier: (x * mult) capped at 255.
                                if (st.cshift != 0) {
                                    const int16x8_t sh = vdupq_n_s16(static_cast<s16>(st.cshift));
                                    out_r = vqmovn_u16(vshlq_u16(vmovl_u8(out_r), sh));
                                    out_g = vqmovn_u16(vshlq_u16(vmovl_u8(out_g), sh));
                                    out_b = vqmovn_u16(vshlq_u16(vmovl_u8(out_b), sh));
                                }
                                if (st.ashift != 0) {
                                    const int16x8_t sh = vdupq_n_s16(static_cast<s16>(st.ashift));
                                    out_a = vqmovn_u16(vshlq_u16(vmovl_u8(out_a), sh));
                                }
                                prev = Rgba8{out_r, out_g, out_b, out_a};
                                buf_cur = buf_next;
                                if (st.upd_rgb) {
                                    buf_next.r = prev.r;
                                    buf_next.g = prev.g;
                                    buf_next.b = prev.b;
                                }
                                if (st.upd_a) {
                                    buf_next.a = prev.a;
                                }
                            }
                            // Interleave back for the shared epilogue.
                            uint8x8x4_t out4;
                            out4.val[0] = prev.r;
                            out4.val[1] = prev.g;
                            out4.val[2] = prev.b;
                            out4.val[3] = prev.a;
                            vst4_u8(&comb[0][0], out4);
                        } else
                        for (int half = 0; half < 2; half++) {
                            const float32x4_t b0 =
                                vcvtq_f32_s32(half == 0 ? w0_lo : w0_hi);
                            const float32x4_t b1 =
                                vcvtq_f32_s32(half == 0 ? w1_lo : w1_hi);
                            const float32x4_t b2 =
                                vcvtq_f32_s32(half == 0 ? w2_lo : w2_hi);
                            const float32x4_t denom = dot3(b0, b1, b2, wi_x, wi_y, wi_z);
                            const float32x4_t inv_w = recip4(denom);

                            // Gouraud color: attr = dot(c, bary) * inv_w; then *255 + 0.5,
                            // truncate. Same expression shape as the scalar path.
                            const auto channel = [&](float32_t a0, float32_t a1, float32_t a2) {
                                float32x4_t n = dot3(b0, b1, b2, a0, a1, a2);
                                n = vmulq_f32(n, inv_w);
                                n = vaddq_f32(vmulq_n_f32(n, 255.0f), vdupq_n_f32(0.5f));
                                return vcvtq_s32_f32(n); // truncates toward zero, as the cast
                            };
                            const int32x4_t pr = channel(c0r, c1r, c2r);
                            const int32x4_t pg = channel(c0g, c1g, c2g);
                            const int32x4_t pb = channel(c0b, c1b, c2b);
                            const int32x4_t pa = channel(c0a, c1a, c2a);

                            int32x4_t out_r = pr, out_g = pg, out_b = pb, out_a = pa;
                            if (span_kernel != SpanKernel::FlatColor) {
                                float32x4_t uu = vmulq_f32(dot3(b0, b1, b2, u0, u1, u2), inv_w);
                                float32x4_t vv = vmulq_f32(dot3(b0, b1, b2, vt0, vt1, vt2), inv_w);
                                int32x4_t ts4 = vcvtq_s32_f32(vmulq_n_f32(uu, twf));
                                int32x4_t tt4 = vcvtq_s32_f32(vmulq_n_f32(vv, thf));
                                if (s_clamp) {
                                    ts4 = vmaxq_s32(vminq_s32(ts4, vdupq_n_s32(tex_w - 1)),
                                                    vdupq_n_s32(0));
                                } else {
                                    ts4 = vandq_s32(ts4, vdupq_n_s32(tex_w - 1));
                                }
                                if (t_clamp) {
                                    tt4 = vmaxq_s32(vminq_s32(tt4, vdupq_n_s32(tex_h - 1)),
                                                    vdupq_n_s32(0));
                                } else {
                                    tt4 = vandq_s32(tt4, vdupq_n_s32(tex_h - 1));
                                }
                                // PICA t axis points up: row = height-1 - t.
                                tt4 = vsubq_s32(vdupq_n_s32(tex_h - 1), tt4);
                                const int32x4_t idx =
                                    vmlaq_s32(ts4, tt4, vdupq_n_s32(static_cast<s32>(plane_w)));
                                alignas(16) s32 idx4[4];
                                vst1q_s32(idx4, idx);
                                alignas(16) u32 texel4[4];
                                texel4[0] = plane[idx4[0]];
                                texel4[1] = plane[idx4[1]];
                                texel4[2] = plane[idx4[2]];
                                texel4[3] = plane[idx4[3]];
                                const uint8x16_t tex16 =
                                    vreinterpretq_u8_u32(vld1q_u32(texel4));
                                if (span_kernel == SpanKernel::ReplaceTex0) {
                                    vst1q_u8(&comb[half * 4][0], tex16);
                                } else {
                                // Modulate: (tex * prim) / 255 with exact truncating division:
                                // y = (x + (x >> 8) + 1) >> 8 for x in [0, 255*255].
                                // prim channels to interleaved rgba bytes
                                alignas(16) u8 prim4[16];
                                alignas(16) s32 tmp[4];
                                vst1q_s32(tmp, pr);
                                prim4[0] = static_cast<u8>(tmp[0]); prim4[4] = static_cast<u8>(tmp[1]);
                                prim4[8] = static_cast<u8>(tmp[2]); prim4[12] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, pg);
                                prim4[1] = static_cast<u8>(tmp[0]); prim4[5] = static_cast<u8>(tmp[1]);
                                prim4[9] = static_cast<u8>(tmp[2]); prim4[13] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, pb);
                                prim4[2] = static_cast<u8>(tmp[0]); prim4[6] = static_cast<u8>(tmp[1]);
                                prim4[10] = static_cast<u8>(tmp[2]); prim4[14] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, pa);
                                prim4[3] = static_cast<u8>(tmp[0]); prim4[7] = static_cast<u8>(tmp[1]);
                                prim4[11] = static_cast<u8>(tmp[2]); prim4[15] = static_cast<u8>(tmp[3]);
                                const uint8x16_t prim16 = vld1q_u8(prim4);
                                const uint16x8_t mul_lo =
                                    vmull_u8(vget_low_u8(tex16), vget_low_u8(prim16));
                                const uint16x8_t mul_hi =
                                    vmull_u8(vget_high_u8(tex16), vget_high_u8(prim16));
                                const auto div255 = [](uint16x8_t x) {
                                    uint16x8_t t = vaddq_u16(x, vshrq_n_u16(x, 8));
                                    t = vaddq_u16(t, vdupq_n_u16(1));
                                    return vshrn_n_u16(t, 8);
                                };
                                const uint8x16_t mod =
                                    vcombine_u8(div255(mul_lo), div255(mul_hi));
                                if (span_kernel == SpanKernel::ModulateTex0) {
                                    vst1q_u8(&comb[half * 4][0], mod);
                                } else {
                                    // PrimAlphaTex0: rgb from the vertex color, alpha from
                                    // the modulate's alpha lanes.
                                    const uint8x16_t alpha_lanes =
                                        vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
                                    vst1q_u8(&comb[half * 4][0],
                                             vbslq_u8(alpha_lanes, mod, prim16));
                                }
                                }
                            } else {
                                alignas(16) s32 tmp[4];
                                vst1q_s32(tmp, out_r);
                                comb[half * 4 + 0][0] = static_cast<u8>(tmp[0]);
                                comb[half * 4 + 1][0] = static_cast<u8>(tmp[1]);
                                comb[half * 4 + 2][0] = static_cast<u8>(tmp[2]);
                                comb[half * 4 + 3][0] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, out_g);
                                comb[half * 4 + 0][1] = static_cast<u8>(tmp[0]);
                                comb[half * 4 + 1][1] = static_cast<u8>(tmp[1]);
                                comb[half * 4 + 2][1] = static_cast<u8>(tmp[2]);
                                comb[half * 4 + 3][1] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, out_b);
                                comb[half * 4 + 0][2] = static_cast<u8>(tmp[0]);
                                comb[half * 4 + 1][2] = static_cast<u8>(tmp[1]);
                                comb[half * 4 + 2][2] = static_cast<u8>(tmp[2]);
                                comb[half * 4 + 3][2] = static_cast<u8>(tmp[3]);
                                vst1q_s32(tmp, out_a);
                                comb[half * 4 + 0][3] = static_cast<u8>(tmp[0]);
                                comb[half * 4 + 1][3] = static_cast<u8>(tmp[1]);
                                comb[half * 4 + 2][3] = static_cast<u8>(tmp[2]);
                                comb[half * 4 + 3][3] = static_cast<u8>(tmp[3]);
                            }

                            // Depth: ((z0*w0 + z1*w1) + z2*w2) * inv_wsum, then scale/offset,
                            // optional w-buffer factor, clamp - the scalar expressions verbatim.
                            float32x4_t zow = dot3(b0, b1, b2, z0, z1, z2);
                            zow = vmulq_n_f32(zow, inv_wsum);
                            float32x4_t d = vaddq_f32(vmulq_n_f32(zow, depth_scale),
                                                      vdupq_n_f32(depth_offset));
                            if (w_buffering) {
                                d = vmulq_f32(d, vmulq_n_f32(inv_w, wsum_tri));
                            }
                            d = vminq_f32(vmaxq_f32(d, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
                            vst1q_f32(&depth8[half * 4], d);
                        }

                        // Epilogue: alpha and depth/stencil tests stay scalar (stencil has
                        // side effects), the blend runs 8-wide below. No fog - the kernel
                        // gates exclude it.
                        // The block's eight pixels are consecutive on one row, so their
                        // framebuffer addresses come from one Morton setup instead of one per
                        // access - the tests and the writeout below make up to six each.
                        const u32 block_x0 =
                            static_cast<u32>((x_first + (k_scalar << 4)) >> 4);
                        alignas(16) u32 color_off[8], depth_off[8];
                        fb.BlockOffsets(block_x0, y >> 4, static_cast<u32>(lanes_valid),
                                        color_off, depth_off);
                        // Both tests are register state: hoisted so a disabled test is one
                        // branch for the block instead of a call per lane.
                        const bool alpha_test_on =
                            regs.framebuffer.output_merger.alpha_test.enable != 0;
                        u32 alpha_mask = (1u << lanes_valid) - 1u;
                        if (alpha_test_on) {
                            for (int lane = 0; lane < lanes_valid; lane++) {
                                if (!DoAlphaTest(comb[lane][3])) {
                                    alpha_mask &= ~(1u << lane);
                                }
                            }
                        }
                        // Fog. Only RGB changes, and neither the alpha test above nor the
                        // depth test below reads RGB, so this is the scalar loop's position
                        // for it. Per lane on the same helper, so the result is the scalar
                        // result; the eight-wide win is the TEV and texture work above.
                        // Lanes the alpha test dropped are fogged too and then masked off.
                        if (draw_fog) {
                            for (int lane = 0; lane < lanes_valid; lane++) {
                                Common::Vec4<u8> c{comb[lane][0], comb[lane][1], comb[lane][2],
                                                   comb[lane][3]};
                                WriteFog(depth8[lane], c);
                                comb[lane][0] = c.r();
                                comb[lane][1] = c.g();
                                comb[lane][2] = c.b();
                            }
                        }
                        const u32 pass_mask =
                            alpha_mask == 0
                                ? 0u
                                : DoDepthStencilTest8(depth_off, depth8, alpha_mask);
                        if (pass_mask != 0 &&
                            regs.framebuffer.framebuffer.allow_color_write != 0) {
                            const auto& om = regs.framebuffer.output_merger;
                            // Every heavy draw of the benchmark scene enables blending with
                            // source One, destination Zero and equation Add, which is a copy.
                            // Running the factor selection, the widening multiplies and the
                            // exact division to reproduce the source is most of this block.
                            using BF = FramebufferRegs::BlendFactor;
                            using BE = FramebufferRegs::BlendEquation;
                            const auto& bp = om.alpha_blending;
                            const bool identity_blend =
                                bp.factor_source_rgb == BF::One && bp.factor_dest_rgb == BF::Zero &&
                                bp.factor_source_a == BF::One && bp.factor_dest_a == BF::Zero &&
                                bp.blend_equation_rgb == BE::Add && bp.blend_equation_a == BE::Add;
                            const u32 write_mask = (om.red_enable ? 0x000000FFu : 0) |
                                                   (om.green_enable ? 0x0000FF00u : 0) |
                                                   (om.blue_enable ? 0x00FF0000u : 0) |
                                                   (om.alpha_enable ? 0xFF000000u : 0);
                            if (identity_blend && write_mask == 0xFFFFFFFFu) {
                                fb.DrawPixels(color_off, pass_mask, comb);
                                k_scalar += 8;
                                w0 += 8 * w0_dx;
                                w1 += 8 * w1_dx;
                                w2 += 8 * w2_dx;
                                continue;
                            }
                            // Destination pixels for the whole block (only surviving lanes are
                            // written back, but the blend math runs on all eight).
                            alignas(16) u8 dst8[8][4] = {};
                            fb.GetPixels(color_off, pass_mask, dst8);
                            const uint8x16_t src_lo = vld1q_u8(&comb[0][0]);
                            const uint8x16_t src_hi = vld1q_u8(&comb[4][0]);
                            const uint8x16_t dst_lo = vld1q_u8(&dst8[0][0]);
                            const uint8x16_t dst_hi = vld1q_u8(&dst8[4][0]);
                            const auto params = om.alpha_blending;
                            const u8 cr = static_cast<u8>(om.blend_const.r.Value());
                            const u8 cg = static_cast<u8>(om.blend_const.g.Value());
                            const u8 cb = static_cast<u8>(om.blend_const.b.Value());
                            const u8 ca = static_cast<u8>(om.blend_const.a.Value());
                            // Broadcast a per-pixel alpha byte across that pixel's four lanes.
                            const auto splat_alpha = [](uint8x16_t v) {
                                const uint8x8x2_t t = {vget_low_u8(v), vget_high_u8(v)};
                                const uint8x8_t idx = {3, 3, 3, 3, 7, 7, 7, 7};
                                return vcombine_u8(vtbl2_u8(t, idx),
                                                   vtbl2_u8(t, vadd_u8(idx, vdup_n_u8(8))));
                            };
                            // rgb factor with the alpha-channel factor patched into lane 3 of
                            // each pixel afterwards, exactly like the scalar lookup_factor.
                            const auto factor = [&](FramebufferRegs::BlendFactor f,
                                                    uint8x16_t s, uint8x16_t d,
                                                    bool alpha_channel) -> uint8x16_t {
                                (void)alpha_channel;
                                using BF = FramebufferRegs::BlendFactor;
                                switch (f) {
                                case BF::Zero:
                                    return vdupq_n_u8(0);
                                case BF::One:
                                    return vdupq_n_u8(255);
                                case BF::SourceColor:
                                    return s;
                                case BF::OneMinusSourceColor:
                                    return vmvnq_u8(s);
                                case BF::DestColor:
                                    return d;
                                case BF::OneMinusDestColor:
                                    return vmvnq_u8(d);
                                case BF::SourceAlpha:
                                    return splat_alpha(s);
                                case BF::OneMinusSourceAlpha:
                                    return vmvnq_u8(splat_alpha(s));
                                case BF::DestAlpha:
                                    return splat_alpha(d);
                                case BF::OneMinusDestAlpha:
                                    return vmvnq_u8(splat_alpha(d));
                                case BF::ConstantColor: {
                                    const u8 lane4[4] = {cr, cg, cb, ca};
                                    const uint32x2_t c = vdup_n_u32(
                                        static_cast<u32>(lane4[0]) | (lane4[1] << 8) |
                                        (lane4[2] << 16) | (static_cast<u32>(lane4[3]) << 24));
                                    return vreinterpretq_u8_u32(vcombine_u32(c, c));
                                }
                                case BF::OneMinusConstantColor: {
                                    const u8 lane4[4] = {static_cast<u8>(255 - cr),
                                                         static_cast<u8>(255 - cg),
                                                         static_cast<u8>(255 - cb),
                                                         static_cast<u8>(255 - ca)};
                                    const uint32x2_t c = vdup_n_u32(
                                        static_cast<u32>(lane4[0]) | (lane4[1] << 8) |
                                        (lane4[2] << 16) | (static_cast<u32>(lane4[3]) << 24));
                                    return vreinterpretq_u8_u32(vcombine_u32(c, c));
                                }
                                case BF::ConstantAlpha:
                                    return vdupq_n_u8(ca);
                                case BF::OneMinusConstantAlpha:
                                    return vdupq_n_u8(static_cast<u8>(255 - ca));
                                case BF::SourceAlphaSaturate: {
                                    // rgb: min(src.a, 255 - dst.a); alpha channel: 255.
                                    const uint8x16_t sat =
                                        vminq_u8(splat_alpha(s), vmvnq_u8(splat_alpha(d)));
                                    const uint8x16_t alpha_lanes = vreinterpretq_u8_u32(
                                        vdupq_n_u32(0xFF000000u));
                                    return vorrq_u8(sat, alpha_lanes);
                                }
                                default:
                                    return s; // scalar path logs/asserts; keep the data flowing
                                }
                            };
                            const auto patch_alpha_lanes = [](uint8x16_t rgb, uint8x16_t a) {
                                // lane 3 of each pixel comes from the alpha-factor vector
                                const uint8x16_t mask =
                                    vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
                                return vbslq_u8(mask, a, rgb);
                            };
                            const auto blend_half = [&](uint8x16_t s, uint8x16_t d) {
                                uint8x16_t sf =
                                    patch_alpha_lanes(factor(params.factor_source_rgb, s, d, false),
                                                      factor(params.factor_source_a, s, d, true));
                                uint8x16_t df =
                                    patch_alpha_lanes(factor(params.factor_dest_rgb, s, d, false),
                                                      factor(params.factor_dest_a, s, d, true));
                                const uint16x8_t sp_lo = vmull_u8(vget_low_u8(s), vget_low_u8(sf));
                                const uint16x8_t sp_hi = vmull_u8(vget_high_u8(s), vget_high_u8(sf));
                                const uint16x8_t dp_lo = vmull_u8(vget_low_u8(d), vget_low_u8(df));
                                const uint16x8_t dp_hi = vmull_u8(vget_high_u8(d), vget_high_u8(df));
                                using BE = FramebufferRegs::BlendEquation;
                                const auto eq = [&](uint16x8_t sp, uint16x8_t dp,
                                                    BE equation) -> uint8x8_t {
                                    // Exact trunc-div-255 with clamp; sums need 32 bits.
                                    const auto div255_u32 = [](uint32x4_t x) {
                                        uint32x4_t t = vaddq_u32(x, vshrq_n_u32(x, 8));
                                        t = vaddq_u32(t, vdupq_n_u32(1));
                                        return vshrq_n_u32(t, 8);
                                    };
                                    const auto div255_wide = [&](uint32x4_t lo, uint32x4_t hi) {
                                        const uint16x4_t nlo = vqmovn_u32(div255_u32(lo));
                                        const uint16x4_t nhi = vqmovn_u32(div255_u32(hi));
                                        return vqmovn_u16(vcombine_u16(nlo, nhi));
                                    };
                                    switch (equation) {
                                    case BE::Add: {
                                        const uint32x4_t lo = vaddl_u16(vget_low_u16(sp),
                                                                        vget_low_u16(dp));
                                        const uint32x4_t hi = vaddl_u16(vget_high_u16(sp),
                                                                        vget_high_u16(dp));
                                        return div255_wide(lo, hi);
                                    }
                                    case BE::Subtract: {
                                        // Negative results clamp to zero: saturating u16 sub
                                        // then the exact division reproduces the scalar math.
                                        const uint16x8_t diff = vqsubq_u16(sp, dp);
                                        const uint32x4_t lo = vmovl_u16(vget_low_u16(diff));
                                        const uint32x4_t hi = vmovl_u16(vget_high_u16(diff));
                                        return div255_wide(lo, hi);
                                    }
                                    case BE::ReverseSubtract: {
                                        const uint16x8_t diff = vqsubq_u16(dp, sp);
                                        const uint32x4_t lo = vmovl_u16(vget_low_u16(diff));
                                        const uint32x4_t hi = vmovl_u16(vget_high_u16(diff));
                                        return div255_wide(lo, hi);
                                    }
                                    case BE::Min: {
                                        const uint16x8_t m = vminq_u16(sp, dp);
                                        const uint32x4_t lo = vmovl_u16(vget_low_u16(m));
                                        const uint32x4_t hi = vmovl_u16(vget_high_u16(m));
                                        return div255_wide(lo, hi);
                                    }
                                    case BE::Max: {
                                        const uint16x8_t m = vmaxq_u16(sp, dp);
                                        const uint32x4_t lo = vmovl_u16(vget_low_u16(m));
                                        const uint32x4_t hi = vmovl_u16(vget_high_u16(m));
                                        return div255_wide(lo, hi);
                                    }
                                    default:
                                        return vqmovn_u16(sp);
                                    }
                                };
                                const uint8x8_t rgb_lo =
                                    eq(sp_lo, dp_lo, params.blend_equation_rgb);
                                const uint8x8_t rgb_hi =
                                    eq(sp_hi, dp_hi, params.blend_equation_rgb);
                                const uint8x8_t a_lo = eq(sp_lo, dp_lo, params.blend_equation_a);
                                const uint8x8_t a_hi = eq(sp_hi, dp_hi, params.blend_equation_a);
                                const uint8x16_t alpha_mask =
                                    vreinterpretq_u8_u32(vdupq_n_u32(0xFF000000u));
                                return vbslq_u8(alpha_mask, vcombine_u8(a_lo, a_hi),
                                                vcombine_u8(rgb_lo, rgb_hi));
                            };
                            uint8x16_t out_lo = blend_half(src_lo, dst_lo);
                            uint8x16_t out_hi = blend_half(src_hi, dst_hi);
                            // Per-channel write-enable, constant per draw.
                            const uint8x16_t enable =
                                vreinterpretq_u8_u32(vdupq_n_u32(write_mask));
                            out_lo = vbslq_u8(enable, out_lo, dst_lo);
                            out_hi = vbslq_u8(enable, out_hi, dst_hi);
                            alignas(16) u8 blended[8][4];
                            vst1q_u8(&blended[0][0], out_lo);
                            vst1q_u8(&blended[4][0], out_hi);
                            fb.DrawPixels(color_off, pass_mask, blended);
                        }

                        k_scalar += 8;
                        w0 += 8 * w0_dx;
                        w1 += 8 * w1_dx;
                        w2 += 8 * w2_dx;
                    }
                    neon_covered = k_scalar > k_max;
                }
#endif // CITRA_ARCH(arm32)
                if (span_kernel != SpanKernel::None || neon_covered) {
                for (s32 k = k_scalar; k <= k_max; k++, w0 += w0_dx, w1 += w1_dx, w2 += w2_dx) {
                    const u16 x = static_cast<u16>(x_first + (k << 4));
                    const auto baricentric_coordinates =
                        Common::MakeVec(f24::FromFloat32(static_cast<f32>(w0)),
                                        f24::FromFloat32(static_cast<f32>(w1)),
                                        f24::FromFloat32(static_cast<f32>(w2)));
                    const f24 interpolated_w_inverse =
                        f24::One() / Common::Dot(w_inverse, baricentric_coordinates);
                    const auto attr = [&](f24 a0, f24 a1, f24 a2) {
                        return Common::Dot(Common::MakeVec(a0, a1, a2), baricentric_coordinates) *
                               interpolated_w_inverse;
                    };
                    const Common::Vec4<u8> primary_color{
                        static_cast<u8>(attr(v0.color.r(), v1.color.r(), v2.color.r()).ToFloat32() *
                                            255 +
                                        0.5f),
                        static_cast<u8>(attr(v0.color.g(), v1.color.g(), v2.color.g()).ToFloat32() *
                                            255 +
                                        0.5f),
                        static_cast<u8>(attr(v0.color.b(), v1.color.b(), v2.color.b()).ToFloat32() *
                                            255 +
                                        0.5f),
                        static_cast<u8>(attr(v0.color.a(), v1.color.a(), v2.color.a()).ToFloat32() *
                                            255 +
                                        0.5f),
                    };
                    Common::Vec4<u8> combiner_output = primary_color;
                    if (span_kernel != SpanKernel::FlatColor) {
                        const auto& tex = textures[0];
                        const f24 u = attr(v0.tc0.u(), v1.tc0.u(), v2.tc0.u());
                        const f24 v = attr(v0.tc0.v(), v1.tc0.v(), v2.tc0.v());
                        // The multiply stays in f24 like the generic path: its 16-bit mantissa
                        // rounding decides edge texels, and f32 here would diff.
                        const f24 tw = f24::FromFloat32(static_cast<f32>(tex.config.width));
                        const f24 th = f24::FromFloat32(static_cast<f32>(tex.config.height));
                        const s32 ts_i = static_cast<s32>((u * tw).ToFloat32());
                        const s32 tt_i = static_cast<s32>((v * th).ToFloat32());
                        bool border = false;
                        if (tex.config.wrap_s == TexturingRegs::TextureConfig::ClampToBorder) {
                            border |= ts_i < 0 || ts_i >= static_cast<s32>(tex.config.width);
                        } else if (tex.config.wrap_s ==
                                   TexturingRegs::TextureConfig::ClampToBorder2) {
                            border |= ts_i >= static_cast<s32>(tex.config.width);
                        }
                        if (tex.config.wrap_t == TexturingRegs::TextureConfig::ClampToBorder) {
                            border |= tt_i < 0 || tt_i >= static_cast<s32>(tex.config.height);
                        } else if (tex.config.wrap_t ==
                                   TexturingRegs::TextureConfig::ClampToBorder2) {
                            border |= tt_i >= static_cast<s32>(tex.config.height);
                        }
                        Common::Vec4<u8> tex_color;
                        if (border) {
                            const auto bc = tex.config.border_color;
                            tex_color = Common::MakeVec(bc.r.Value(), bc.g.Value(), bc.b.Value(),
                                                        bc.a.Value())
                                            .Cast<u8>();
                        } else {
                            const s32 ws =
                                GetWrappedTexCoord(tex.config.wrap_s, ts_i, tex.config.width);
                            const s32 wt =
                                static_cast<s32>(tex.config.height) - 1 -
                                GetWrappedTexCoord(tex.config.wrap_t, tt_i, tex.config.height);
                            const u32 texel =
                                tex_planes[0][static_cast<std::size_t>(wt) * tex_plane_width[0] +
                                              ws];
                            tex_color = {static_cast<u8>(texel), static_cast<u8>(texel >> 8),
                                         static_cast<u8>(texel >> 16),
                                         static_cast<u8>(texel >> 24)};
                        }
                        if (span_kernel == SpanKernel::ModulateTex0) {
                            combiner_output[0] =
                                static_cast<u8>(tex_color.r() * primary_color.r() / 255);
                            combiner_output[1] =
                                static_cast<u8>(tex_color.g() * primary_color.g() / 255);
                            combiner_output[2] =
                                static_cast<u8>(tex_color.b() * primary_color.b() / 255);
                            combiner_output[3] =
                                static_cast<u8>(tex_color.a() * primary_color.a() / 255);
                        } else if (span_kernel == SpanKernel::ReplaceTex0) {
                            combiner_output = tex_color;
                        } else if (span_kernel == SpanKernel::PrimAlphaTex0) {
                            combiner_output = primary_color;
                            combiner_output[3] =
                                static_cast<u8>(tex_color.a() * primary_color.a() / 255);
                        } else if (span_kernel == SpanKernel::AddModTex0) {
                            // s0: color Add(Tex0.rgb, Const0.a), alpha Replace(Primary.a);
                            // s1: color Modulate(Const1, Previous), alpha Replace(Previous).
                            const s32 add_a = const0.a();
                            const s32 r0 = std::min(255, tex_color.r() + add_a);
                            const s32 g0 = std::min(255, tex_color.g() + add_a);
                            const s32 b0 = std::min(255, tex_color.b() + add_a);
                            combiner_output[0] = static_cast<u8>(const1.r() * r0 / 255);
                            combiner_output[1] = static_cast<u8>(const1.g() * g0 / 255);
                            combiner_output[2] = static_cast<u8>(const1.b() * b0 / 255);
                            combiner_output[3] = primary_color.a();
                        } else { // LerpLerpTex0
                            // s0: color Lerp(Tex0, Const0, f=Tex0), alpha Replace(Const0.a);
                            // s1: color Lerp(Tex0, Const1, f=Previous), alpha Replace(Previous).
                            const auto lerp = [](s32 a, s32 b, s32 f) {
                                return static_cast<s32>((a * f + b * (255 - f)) / 255);
                            };
                            const s32 r0 = lerp(tex_color.r(), const0.r(), tex_color.r());
                            const s32 g0 = lerp(tex_color.g(), const0.g(), tex_color.g());
                            const s32 b0 = lerp(tex_color.b(), const0.b(), tex_color.b());
                            combiner_output[0] =
                                static_cast<u8>(lerp(tex_color.r(), const1.r(), r0));
                            combiner_output[1] =
                                static_cast<u8>(lerp(tex_color.g(), const1.g(), g0));
                            combiner_output[2] =
                                static_cast<u8>(lerp(tex_color.b(), const1.b(), b0));
                            combiner_output[3] = const0.a();
                        }
                    }
                    if (!DoAlphaTest(combiner_output.a())) {
                        continue;
                    }
                    const float z_over_w =
                        (v0.screenpos[2].ToFloat32() * w0 + v1.screenpos[2].ToFloat32() * w1 +
                         v2.screenpos[2].ToFloat32() * w2) *
                        inv_wsum;
                    float depth = z_over_w * depth_scale + depth_offset;
                    if (w_buffering) {
                        depth *= interpolated_w_inverse.ToFloat32() * wsum_tri;
                    }
                    depth = std::clamp(depth, 0.0f, 1.0f);
                    if (!DoDepthStencilTest(x, y, depth)) {
                        continue;
                    }
                    const auto result = PixelColor(x, y, combiner_output);
                    if (regs.framebuffer.framebuffer.allow_color_write != 0) {
                        fb.DrawPixel(x >> 4, y >> 4, result);
                    }
                }
#ifdef CITRA_TRACE_PROBES
                if (verify_this_span) {
                    // Kernel result is in the framebuffer; snapshot it, restore the original
                    // span, run the generic loop below, then compare and put the kernel's
                    // bytes back (so verified and unverified runs render identically).
                    verify_generic.resize(verify_before.size());
                    for (s32 k = k_min; k <= k_max; k++) {
                        const u16 x = static_cast<u16>(x_first + (k << 4));
                        const auto c0 = fb.GetPixel(x >> 4, y >> 4);
                        verify_generic[(k - k_min) * 3] = static_cast<u32>(c0.r()) |
                                                          (c0.g() << 8) | (c0.b() << 16) |
                                                          (static_cast<u32>(c0.a()) << 24);
                        verify_generic[(k - k_min) * 3 + 1] = fb.GetDepth(x >> 4, y >> 4);
                        verify_generic[(k - k_min) * 3 + 2] =
                            verify_stencil ? fb.GetStencil(x >> 4, y >> 4) : 0;
                        const u32 orig = verify_before[(k - k_min) * 3];
                        fb.DrawPixel(x >> 4, y >> 4,
                                     {static_cast<u8>(orig), static_cast<u8>(orig >> 8),
                                      static_cast<u8>(orig >> 16), static_cast<u8>(orig >> 24)});
                        fb.SetDepth(x >> 4, y >> 4, verify_before[(k - k_min) * 3 + 1]);
                        if (verify_stencil) {
                            fb.SetStencil(x >> 4, y >> 4,
                                          static_cast<u8>(verify_before[(k - k_min) * 3 + 2]));
                        }
                    }
                    // fall through to the generic loop; comparison happens after it. The
                    // kernel loop advanced the edge accumulators past the span - rewind them
                    // or the generic pass evaluates garbage barycentrics (this bug produced
                    // 59k phantom "mismatches" before the inline-TEV probe cleared the kernel).
                    w0 = w0_row + k_min * w0_dx;
                    w1 = w1_row + k_min * w1_dx;
                    w2 = w2_row + k_min * w2_dx;
                } else
#endif // CITRA_TRACE_PROBES
                {
                    return;
                }
                } // span_kernel != None || neon_covered; otherwise the generic loop finishes
            }

            for (s32 k = k_min; k <= k_max; k++, w0 += w0_dx, w1 += w1_dx, w2 += w2_dx) {
                const u16 x = static_cast<u16>(x_first + (k << 4));
                // Do not process the pixel if it's inside the scissor box and the scissor mode
                // is set to Exclude.
                if (scissor_exclude) {
                    if (x >= scissor_x1 && x < scissor_x2 && y >= scissor_y1 && y < scissor_y2) {
                        continue;
                    }
                }
                const s32 wsum = wsum_tri;

                const auto baricentric_coordinates = Common::MakeVec(
                    f24::FromFloat32(static_cast<f32>(w0)), f24::FromFloat32(static_cast<f32>(w1)),
                    f24::FromFloat32(static_cast<f32>(w2)));
                const f24 interpolated_w_inverse =
                    f24::One() / Common::Dot(w_inverse, baricentric_coordinates);

                // interpolated_z = z / w
                const float interpolated_z_over_w =
                    (v0.screenpos[2].ToFloat32() * w0 + v1.screenpos[2].ToFloat32() * w1 +
                     v2.screenpos[2].ToFloat32() * w2) *
                    inv_wsum;

                // Not fully accurate. About 3 bits in precision are missing.
                // Z-Buffer (z / w * scale + offset)
                const float depth_scale =
                    f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();
                const float depth_offset =
                    f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();
                float depth = interpolated_z_over_w * depth_scale + depth_offset;

                // Potentially switch to W-Buffer
                if (regs.rasterizer.depthmap_enable ==
                    Pica::RasterizerRegs::DepthBuffering::WBuffering) {
                    // W-Buffer (z * scale + w * offset = (z / w * scale + offset) * w)
                    depth *= interpolated_w_inverse.ToFloat32() * wsum;
                }

                // Clamp the result
                depth = std::clamp(depth, 0.0f, 1.0f);

                /**
                 * Perspective correct attribute interpolation:
                 * Attribute values cannot be calculated by simple linear interpolation since
                 * they are not linear in screen space. For example, when interpolating a
                 * texture coordinate across two vertices, something simple like
                 *     u = (u0*w0 + u1*w1)/(w0+w1)
                 * will not work. However, the attribute value divided by the
                 * clipspace w-coordinate (u/w) and and the inverse w-coordinate (1/w) are linear
                 * in screenspace. Hence, we can linearly interpolate these two independently and
                 * calculate the interpolated attribute by dividing the results.
                 * I.e.
                 *     u_over_w   = ((u0/v0.pos.w)*w0 + (u1/v1.pos.w)*w1)/(w0+w1)
                 *     one_over_w = (( 1/v0.pos.w)*w0 + ( 1/v1.pos.w)*w1)/(w0+w1)
                 *     u = u_over_w / one_over_w
                 *
                 * The generalization to three vertices is straightforward in baricentric
                 *coordinates.
                 **/
                const auto get_interpolated_attribute = [&](f24 attr0, f24 attr1, f24 attr2) {
                    auto attr_over_w = Common::MakeVec(attr0, attr1, attr2);
                    f24 interpolated_attr_over_w =
                        Common::Dot(attr_over_w, baricentric_coordinates);
                    return interpolated_attr_over_w * interpolated_w_inverse;
                };

                // Values are non-negative, so +0.5f truncation matches round() without the
                // libm call per channel per pixel.
                const Common::Vec4<u8> primary_color{
                    static_cast<u8>(
                        get_interpolated_attribute(v0.color.r(), v1.color.r(), v2.color.r())
                                .ToFloat32() *
                            255 +
                        0.5f),
                    static_cast<u8>(
                        get_interpolated_attribute(v0.color.g(), v1.color.g(), v2.color.g())
                                .ToFloat32() *
                            255 +
                        0.5f),
                    static_cast<u8>(
                        get_interpolated_attribute(v0.color.b(), v1.color.b(), v2.color.b())
                                .ToFloat32() *
                            255 +
                        0.5f),
                    static_cast<u8>(
                        get_interpolated_attribute(v0.color.a(), v1.color.a(), v2.color.a())
                                .ToFloat32() *
                            255 +
                        0.5f),
                };

                std::array<Common::Vec2<f24>, 3> uv;
                uv[0].u() = get_interpolated_attribute(v0.tc0.u(), v1.tc0.u(), v2.tc0.u());
                uv[0].v() = get_interpolated_attribute(v0.tc0.v(), v1.tc0.v(), v2.tc0.v());
                uv[1].u() = get_interpolated_attribute(v0.tc1.u(), v1.tc1.u(), v2.tc1.u());
                uv[1].v() = get_interpolated_attribute(v0.tc1.v(), v1.tc1.v(), v2.tc1.v());
                uv[2].u() = get_interpolated_attribute(v0.tc2.u(), v1.tc2.u(), v2.tc2.u());
                uv[2].v() = get_interpolated_attribute(v0.tc2.v(), v1.tc2.v(), v2.tc2.v());

                // Sample bound texture units.
                const f24 tc0_w = get_interpolated_attribute(v0.tc0_w, v1.tc0_w, v2.tc0_w);
                const auto texture_color =
                    TextureColor(uv, textures, tc0_w, tex_planes, tex_plane_width);

                Common::Vec4<u8> primary_fragment_color = {0, 0, 0, 0};
                Common::Vec4<u8> secondary_fragment_color = {0, 0, 0, 0};

                if (!regs.lighting.disable) {
                    const auto normquat =
                        Common::Quaternion<f32>{
                            {get_interpolated_attribute(v0.quat.x, v1.quat.x, v2.quat.x)
                                 .ToFloat32(),
                             get_interpolated_attribute(v0.quat.y, v1.quat.y, v2.quat.y)
                                 .ToFloat32(),
                             get_interpolated_attribute(v0.quat.z, v1.quat.z, v2.quat.z)
                                 .ToFloat32()},
                            get_interpolated_attribute(v0.quat.w, v1.quat.w, v2.quat.w).ToFloat32(),
                        }
                            .Normalized();

                    const Common::Vec3f view{
                        get_interpolated_attribute(v0.view.x, v1.view.x, v2.view.x).ToFloat32(),
                        get_interpolated_attribute(v0.view.y, v1.view.y, v2.view.y).ToFloat32(),
                        get_interpolated_attribute(v0.view.z, v1.view.z, v2.view.z).ToFloat32(),
                    };
                    std::tie(primary_fragment_color, secondary_fragment_color) =
                        ComputeFragmentsColors(draw_lighting, pica.lighting, normquat, view,
                                               texture_color);
                }

                // Write the TEV stages.
                auto combiner_output =
                    WriteTevConfig(texture_color, tev_stages, primary_color, primary_fragment_color,
                                   secondary_fragment_color, tev_skip);

                const auto& output_merger = regs.framebuffer.output_merger;
                if (output_merger.fragment_operation_mode ==
                    FramebufferRegs::FragmentOperationMode::Shadow) {
                    const u32 depth_int = static_cast<u32>(depth * 0xFFFFFF);
                    // Use green color as the shadow intensity
                    const u8 stencil = combiner_output.y;
                    fb.DrawShadowMapPixel(x >> 4, y >> 4, depth_int, stencil);
                    // Skip the normal output merger pipeline if it is in shadow mode
                    continue;
                }

                // Does alpha testing happen before or after stencil?
                if (!DoAlphaTest(combiner_output.a())) {
                    continue;
                }
                if (draw_fog) {
                    WriteFog(depth, combiner_output);
                }
                if (!DoDepthStencilTest(x, y, depth)) {
                    continue;
                }
                const auto result = PixelColor(x, y, combiner_output);
                if (regs.framebuffer.framebuffer.allow_color_write != 0) {
                    fb.DrawPixel(x >> 4, y >> 4, result);
                }
            }
#ifdef CITRA_TRACE_PROBES
            // Empty when the span fell through to the generic loop without a kernel/NEON
            // pass (short GenericNeon spans): nothing to compare then.
            if (verify_this_span && !verify_generic.empty()) {
                for (s32 k = k_min; k <= k_max; k++) {
                    const u16 x = static_cast<u16>(x_first + (k << 4));
                    const auto cg = fb.GetPixel(x >> 4, y >> 4);
                    const u32 gen = static_cast<u32>(cg.r()) | (cg.g() << 8) | (cg.b() << 16) |
                                    (static_cast<u32>(cg.a()) << 24);
                    const u32 gen_z = fb.GetDepth(x >> 4, y >> 4);
                    const u32 gen_s = verify_stencil ? fb.GetStencil(x >> 4, y >> 4) : 0;
                    const u32 ker = verify_generic[(k - k_min) * 3];
                    const u32 ker_z = verify_generic[(k - k_min) * 3 + 1];
                    const u32 ker_s = verify_generic[(k - k_min) * 3 + 2];
                    // Tolerances (documented, per the plan): color within 2 per channel
                    // (saturating NEON narrows vs the scalar wrap in deep blend tails),
                    // depth within 4 LSBs of 24 (the scalar expression is FMA-contracted
                    // while the NEON path rounds each step; measured delta <= 2).
                    const auto ch_delta = [&](int shift) {
                        const s32 a = (gen >> shift) & 0xFF;
                        const s32 b = (ker >> shift) & 0xFF;
                        return std::abs(a - b);
                    };
                    const bool color_close = ch_delta(0) <= 2 && ch_delta(8) <= 2 &&
                                             ch_delta(16) <= 2 && ch_delta(24) <= 2;
                    const u32 z_delta = gen_z > ker_z ? gen_z - ker_z : ker_z - gen_z;
                    // Stencil is integer state with no interpolation: it must match exactly.
                    if (!color_close || z_delta > 4 || gen_s != ker_s) {
                        static std::atomic<u32> mismatch_count{0};
                        const u32 n = mismatch_count.fetch_add(1);
                        if (n < 40) {
                            const s32 vw0 = w0_row + k * w0_dx;
                            const s32 vw1 = w1_row + k * w1_dx;
                            const s32 vw2 = w2_row + k * w2_dx;
                            const auto vbary = Common::MakeVec(
                                f24::FromFloat32(static_cast<f32>(vw0)),
                                f24::FromFloat32(static_cast<f32>(vw1)),
                                f24::FromFloat32(static_cast<f32>(vw2)));
                            const f24 vinv = f24::One() / Common::Dot(w_inverse, vbary);
                            const auto vattr = [&](f24 a0, f24 a1, f24 a2) {
                                return Common::Dot(Common::MakeVec(a0, a1, a2), vbary) * vinv;
                            };
                            const f24 vu = vattr(v0.tc0.u(), v1.tc0.u(), v2.tc0.u());
                            const f24 vv = vattr(v0.tc0.v(), v1.tc0.v(), v2.tc0.v());
                            const auto& vtex = textures[0];
                            const f24 vtw =
                                f24::FromFloat32(static_cast<f32>(vtex.config.width));
                            const f24 vth =
                                f24::FromFloat32(static_cast<f32>(vtex.config.height));
                            const s32 vts = static_cast<s32>((vu * vtw).ToFloat32());
                            const s32 vtt = static_cast<s32>((vv * vth).ToFloat32());
                            const s32 vws =
                                GetWrappedTexCoord(vtex.config.wrap_s, vts, vtex.config.width);
                            const s32 vwt = static_cast<s32>(vtex.config.height) - 1 -
                                            GetWrappedTexCoord(vtex.config.wrap_t, vtt,
                                                               vtex.config.height);
                            const u32 vtexel =
                                tex_planes[0]
                                    ? tex_planes[0][static_cast<std::size_t>(vwt) *
                                                        tex_plane_width[0] +
                                                    vws]
                                    : 0xDEADBEEF;
                            const u8* vsrc = memory.GetPhysicalPointer(
                                vtex.config.GetPhysicalAddress());
                            const auto vinfo =
                                TextureInfo::FromPicaRegister(vtex.config, vtex.format);
                            const auto vgen = (vsrc && vws >= 0 && vwt >= 0)
                                                  ? LookupTexture(vsrc, vws, vwt, vinfo)
                                                  : Common::Vec4<u8>{1, 2, 3, 4};
                            const auto vprim_at = [&](int ch) {
                                const f24 a0 = v0.color[ch], a1 = v1.color[ch], a2 = v2.color[ch];
                                return static_cast<u8>(vattr(a0, a1, a2).ToFloat32() * 255 + 0.5f);
                            };
                            const Common::Vec4<u8> vprim{vprim_at(0), vprim_at(1), vprim_at(2),
                                                         vprim_at(3)};
                            std::array<Common::Vec4<u8>, 4> vtexcols{};
                            vtexcols[0] = {static_cast<u8>(vtexel), static_cast<u8>(vtexel >> 8),
                                           static_cast<u8>(vtexel >> 16),
                                           static_cast<u8>(vtexel >> 24)};
                            const auto vout = WriteTevConfig(vtexcols, tev_stages, vprim, {}, {},
                                                             tev_skip);
                            LOG_ERROR(HW_GPU,
                                      "KERNELVERIFY mismatch kernel={} x={} y={} gen={:08x} "
                                      "ker={:08x} genz={:08x} kerz={:08x} uv=({},{}) "
                                      "prim=({},{},{},{}) "
                                      "tevout=({},{},{},{}) alphapass={} planetexel={:08x} "
                                      "fmt={} gens={} kers={} wrap=({},{}) dim=({},{}) "
                                      "type={} texa=({},{},{})",
                                      static_cast<int>(span_kernel), x >> 4, y >> 4, gen, ker, gen_z, ker_z,
                                      vu.ToFloat32(), vv.ToFloat32(), vprim.r(), vprim.g(),
                                      vprim.b(), vprim.a(), vout.r(), vout.g(), vout.b(),
                                      vout.a(), DoAlphaTest(vout.a()), vtexel,
                                      static_cast<u32>(vtex.format), gen_s, ker_s,
                                      static_cast<u32>(vtex.config.wrap_s.Value()),
                                      static_cast<u32>(vtex.config.wrap_t.Value()),
                                      static_cast<u32>(vtex.config.width),
                                      static_cast<u32>(vtex.config.height),
                                      static_cast<u32>(vtex.config.type.Value()),
                                      static_cast<u32>(textures[0].enabled),
                                      static_cast<u32>(textures[1].enabled),
                                      static_cast<u32>(textures[2].enabled));
                        }
                    }
                    fb.DrawPixel(x >> 4, y >> 4,
                                 {static_cast<u8>(ker), static_cast<u8>(ker >> 8),
                                  static_cast<u8>(ker >> 16), static_cast<u8>(ker >> 24)});
                    fb.SetDepth(x >> 4, y >> 4, ker_z);
                    if (verify_stencil) {
                        fb.SetStencil(x >> 4, y >> 4, static_cast<u8>(ker_s));
                    }
                }
            }
#endif // CITRA_TRACE_PROBES
        };
        // Rows are owned by absolute stripe ((y >> 4) % stripe_count): the same row of two
        // always runs on the same executor, in submission order, so no barrier is needed
        // between triangles of a draw.
        for (u16 y = min_y + 8; y < max_y; y = static_cast<u16>(y + 0x10)) {
            if (static_cast<int>((y >> 4) % stripe_count) == stripe) {
                process_scanline(y);
            }
        }
        // One atomic per triangle-stripe, not per pixel; see Common::PipelineStats::fragments.
        if (shaded_pixels != 0) {
            Common::PipelineStats::fragments.fetch_add(shaded_pixels, std::memory_order_relaxed);
            Common::PipelineStats::fragments_total.fetch_add(shaded_pixels,
                                                             std::memory_order_relaxed);
        }
    }
}

std::array<Common::Vec4<u8>, 4> RasterizerSoftware::TextureColor(
    std::span<const Common::Vec2<f24>, 3> uv,
    std::span<const Pica::TexturingRegs::FullTextureConfig, 3> textures, f24 tc0_w,
    const std::array<const u32*, 3>& tex_planes,
    const std::array<u32, 3>& tex_plane_width) const {
    std::array<Common::Vec4<u8>, 4> texture_color{};
    for (u32 i = 0; i < 3; ++i) {
        const auto& texture = textures[i];
        if (!texture.enabled) [[unlikely]] {
            continue;
        }
        if (texture.config.address == 0) [[unlikely]] {
            texture_color[i] = {0, 0, 0, 255};
            continue;
        }

        const s32 coordinate_i = (i == 2 && regs.texturing.main_config.texture2_use_coord1) ? 1 : i;
        f24 u = uv[coordinate_i].u();
        f24 v = uv[coordinate_i].v();

        // Only unit 0 respects the texturing type (according to 3DBrew)
        PAddr texture_address = texture.config.GetPhysicalAddress();
        f24 shadow_z;
        if (i == 0) {
            switch (texture.config.type) {
            case TexturingRegs::TextureConfig::Texture2D:
                break;
            case TexturingRegs::TextureConfig::ShadowCube:
            case TexturingRegs::TextureConfig::TextureCube: {
                std::tie(u, v, shadow_z, texture_address) =
                    ConvertCubeCoord(u, v, tc0_w, regs.texturing);
                break;
            }
            case TexturingRegs::TextureConfig::Projection2D: {
                u /= tc0_w;
                v /= tc0_w;
                break;
            }
            case TexturingRegs::TextureConfig::Shadow2D: {
                if (!regs.texturing.shadow.orthographic) {
                    u /= tc0_w;
                    v /= tc0_w;
                }
                shadow_z = f24::FromFloat32(std::abs(tc0_w.ToFloat32()));
                break;
            }
            case TexturingRegs::TextureConfig::Disabled:
                continue; // skip this unit and continue to the next unit
            default:
                LOG_ERROR(HW_GPU, "Unhandled texture type {:x}", (int)texture.config.type);
                UNIMPLEMENTED();
                break;
            }
        }

        const f24 width = f24::FromFloat32(static_cast<f32>(texture.config.width));
        const f24 height = f24::FromFloat32(static_cast<f32>(texture.config.height));
        s32 s = static_cast<s32>((u * width).ToFloat32());
        s32 t = static_cast<s32>((v * height).ToFloat32());

        bool use_border_s = false;
        bool use_border_t = false;

        if (texture.config.wrap_s == TexturingRegs::TextureConfig::ClampToBorder) {
            use_border_s = s < 0 || s >= static_cast<s32>(texture.config.width);
        } else if (texture.config.wrap_s == TexturingRegs::TextureConfig::ClampToBorder2) {
            use_border_s = s >= static_cast<s32>(texture.config.width);
        }

        if (texture.config.wrap_t == TexturingRegs::TextureConfig::ClampToBorder) {
            use_border_t = t < 0 || t >= static_cast<s32>(texture.config.height);
        } else if (texture.config.wrap_t == TexturingRegs::TextureConfig::ClampToBorder2) {
            use_border_t = t >= static_cast<s32>(texture.config.height);
        }

        if (use_border_s || use_border_t) {
            const auto border_color = texture.config.border_color;
            texture_color[i] = Common::MakeVec(border_color.r.Value(), border_color.g.Value(),
                                               border_color.b.Value(), border_color.a.Value())
                                   .Cast<u8>();
        } else {
            // Textures are laid out from bottom to top, hence we invert the t coordinate.
            // NOTE: This may not be the right place for the inversion.
            // TODO: Check if this applies to ETC textures, too.
            s = GetWrappedTexCoord(texture.config.wrap_s, s, texture.config.width);
            t = texture.config.height - 1 -
                GetWrappedTexCoord(texture.config.wrap_t, t, texture.config.height);

            // TODO: Apply the min and mag filters to the texture
            // Sampling reads the pre-resolved linear plane: one Morton and format decode
            // per texture per change, instead of one per texel. The
            // address guard skips the plane when unit 0's type rewrote the source address.
            const u32* plane = tex_planes[i];
            if (plane != nullptr && texture_address == texture.config.GetPhysicalAddress()) {
                const u32 texel = plane[static_cast<std::size_t>(t) * tex_plane_width[i] + s];
                texture_color[i] = {static_cast<u8>(texel), static_cast<u8>(texel >> 8),
                                    static_cast<u8>(texel >> 16), static_cast<u8>(texel >> 24)};
            } else {
                const u8* texture_data = memory.GetPhysicalPointer(texture_address);
                const auto info = TextureInfo::FromPicaRegister(texture.config, texture.format);
                texture_color[i] = LookupTexture(texture_data, s, t, info);
            }
        }

        if (i == 0 && (texture.config.type == TexturingRegs::TextureConfig::Shadow2D ||
                       texture.config.type == TexturingRegs::TextureConfig::ShadowCube)) {

            s32 z_int = static_cast<s32>(std::min(shadow_z.ToFloat32(), 1.0f) * 0xFFFFFF);
            z_int -= regs.texturing.shadow.bias << 1;
            const auto& color = texture_color[i];
            const s32 z_ref = (color.w << 16) | (color.z << 8) | color.y;
            u8 density;
            if (z_ref >= z_int) {
                density = color.x;
            } else {
                density = 0;
            }
            texture_color[i] = {density, density, density, density};
        }
    }

    // Sample procedural texture
    if (regs.texturing.main_config.texture3_enable) {
        const auto& proctex_uv = uv[regs.texturing.main_config.texture3_coordinates];
        texture_color[3] = ProcTex(proctex_uv.u().ToFloat32(), proctex_uv.v().ToFloat32(),
                                   regs.texturing, pica.proctex);
    }

    return texture_color;
}

Common::Vec4<u8> RasterizerSoftware::PixelColor(u16 x, u16 y,
                                                Common::Vec4<u8> combiner_output) const {
    const auto dest = fb.GetPixel(x >> 4, y >> 4);
    Common::Vec4<u8> blend_output = combiner_output;

    const auto& output_merger = regs.framebuffer.output_merger;
    if (output_merger.alphablend_enable) {
        const auto params = output_merger.alpha_blending;
        // Source One, destination Zero, equation Add: the blend is a copy of the source.
        if (params.factor_source_rgb == FramebufferRegs::BlendFactor::One &&
            params.factor_dest_rgb == FramebufferRegs::BlendFactor::Zero &&
            params.factor_source_a == FramebufferRegs::BlendFactor::One &&
            params.factor_dest_a == FramebufferRegs::BlendFactor::Zero &&
            params.blend_equation_rgb == FramebufferRegs::BlendEquation::Add &&
            params.blend_equation_a == FramebufferRegs::BlendEquation::Add) {
            return {output_merger.red_enable ? blend_output.r() : dest.r(),
                    output_merger.green_enable ? blend_output.g() : dest.g(),
                    output_merger.blue_enable ? blend_output.b() : dest.b(),
                    output_merger.alpha_enable ? blend_output.a() : dest.a()};
        }
        // One switch per factor register instead of one per channel (the per-channel lambda
        // was among the hottest code of a fight frame). The vector holds the factor for all
        // four channels; the caller patches lane 3 from the alpha-factor register.
        const auto factor_vec = [&](FramebufferRegs::BlendFactor factor) -> Common::Vec4<u8> {
            switch (factor) {
            case FramebufferRegs::BlendFactor::Zero:
                return {0, 0, 0, 0};
            case FramebufferRegs::BlendFactor::One:
                return {255, 255, 255, 255};
            case FramebufferRegs::BlendFactor::SourceColor:
                return combiner_output;
            case FramebufferRegs::BlendFactor::OneMinusSourceColor:
                return {static_cast<u8>(255 - combiner_output.r()),
                        static_cast<u8>(255 - combiner_output.g()),
                        static_cast<u8>(255 - combiner_output.b()),
                        static_cast<u8>(255 - combiner_output.a())};
            case FramebufferRegs::BlendFactor::DestColor:
                return dest;
            case FramebufferRegs::BlendFactor::OneMinusDestColor:
                return {static_cast<u8>(255 - dest.r()), static_cast<u8>(255 - dest.g()),
                        static_cast<u8>(255 - dest.b()), static_cast<u8>(255 - dest.a())};
            case FramebufferRegs::BlendFactor::SourceAlpha: {
                const u8 a = combiner_output.a();
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::OneMinusSourceAlpha: {
                const u8 a = static_cast<u8>(255 - combiner_output.a());
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::DestAlpha: {
                const u8 a = dest.a();
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::OneMinusDestAlpha: {
                const u8 a = static_cast<u8>(255 - dest.a());
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::ConstantColor:
                return Common::MakeVec(output_merger.blend_const.r.Value(),
                                       output_merger.blend_const.g.Value(),
                                       output_merger.blend_const.b.Value(),
                                       output_merger.blend_const.a.Value())
                    .Cast<u8>();
            case FramebufferRegs::BlendFactor::OneMinusConstantColor:
                return {static_cast<u8>(255 - output_merger.blend_const.r.Value()),
                        static_cast<u8>(255 - output_merger.blend_const.g.Value()),
                        static_cast<u8>(255 - output_merger.blend_const.b.Value()),
                        static_cast<u8>(255 - output_merger.blend_const.a.Value())};
            case FramebufferRegs::BlendFactor::ConstantAlpha: {
                const u8 a = static_cast<u8>(output_merger.blend_const.a.Value());
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::OneMinusConstantAlpha: {
                const u8 a = static_cast<u8>(255 - output_merger.blend_const.a.Value());
                return {a, a, a, a};
            }
            case FramebufferRegs::BlendFactor::SourceAlphaSaturate: {
                // rgb: min(src.a, 255 - dst.a); the alpha channel reads 1.0.
                const u8 sat =
                    std::min(combiner_output.a(), static_cast<u8>(255 - dest.a()));
                return {sat, sat, sat, 255};
            }
            default:
                LOG_CRITICAL(HW_GPU, "Unknown blend factor {:x}", factor);
                UNIMPLEMENTED();
                return combiner_output;
            }
        };

        auto srcfactor = factor_vec(params.factor_source_rgb);
        srcfactor.a() = factor_vec(params.factor_source_a).a();
        auto dstfactor = factor_vec(params.factor_dest_rgb);
        dstfactor.a() = factor_vec(params.factor_dest_a).a();

        blend_output = EvaluateBlendEquation(combiner_output, srcfactor, dest, dstfactor,
                                             params.blend_equation_rgb);
        blend_output.a() = EvaluateBlendEquation(combiner_output, srcfactor, dest, dstfactor,
                                                 params.blend_equation_a)
                               .a();
    } else {
        blend_output =
            Common::MakeVec(LogicOp(combiner_output.r(), dest.r(), output_merger.logic_op),
                            LogicOp(combiner_output.g(), dest.g(), output_merger.logic_op),
                            LogicOp(combiner_output.b(), dest.b(), output_merger.logic_op),
                            LogicOp(combiner_output.a(), dest.a(), output_merger.logic_op));
    }

    const Common::Vec4<u8> result = {
        output_merger.red_enable ? blend_output.r() : dest.r(),
        output_merger.green_enable ? blend_output.g() : dest.g(),
        output_merger.blue_enable ? blend_output.b() : dest.b(),
        output_merger.alpha_enable ? blend_output.a() : dest.a(),
    };

    return result;
}

Common::Vec4<u8> RasterizerSoftware::WriteTevConfig(
    std::span<const Common::Vec4<u8>, 4> texture_color,
    std::span<const Pica::TexturingRegs::TevStageConfig, 6> tev_stages,
    Common::Vec4<u8> primary_color, Common::Vec4<u8> primary_fragment_color,
    Common::Vec4<u8> secondary_fragment_color, const std::array<bool, 6>& tev_skip) {
    /**
     * Texture environment - consists of 6 stages of color and alpha combining.
     * Color combiners take three input color values from some source (e.g. interpolated
     * vertex color, texture color, previous stage, etc), perform some very simple
     * operations on each of them (e.g. inversion) and then calculate the output color
     * with some basic arithmetic. Alpha combiners can be configured separately but work
     * analogously.
     **/
    Common::Vec4<u8> combiner_output = {0, 0, 0, 0};
    Common::Vec4<u8> combiner_buffer = {0, 0, 0, 0};
    Common::Vec4<u8> next_combiner_buffer = draw_combiner_buffer_color;

    // Source dispatch table: the 4-bit source field indexes live pixel values directly, so
    // fetching a source is one load instead of a switch (the switch was among the hottest
    // instructions of a fight frame). combiner_buffer/combiner_output mutate between stages
    // and are read through the table, which keeps the original semantics.
    static constexpr Common::Vec4<u8> source_zero{0, 0, 0, 0};
    const std::array<const Common::Vec4<u8>*, 16> source_map = {
        &primary_color,            // PrimaryColor
        &primary_fragment_color,   // PrimaryFragmentColor
        &secondary_fragment_color, // SecondaryFragmentColor
        &texture_color[0],         &texture_color[1],
        &texture_color[2],         &texture_color[3],
        &source_zero,              // 7: unused on hardware
        &source_zero,              &source_zero,
        &source_zero,              &source_zero,
        &source_zero,              // 8-12: unused
        &combiner_buffer,          // PreviousBuffer
        &source_zero,              // Constant: handled below, differs per stage
        &combiner_output,          // Previous
    };

    for (u32 tev_stage_index = 0; tev_stage_index < tev_stages.size(); ++tev_stage_index) {
        if (tev_skip[tev_stage_index]) {
            continue;
        }
        const auto& tev_stage = tev_stages[tev_stage_index];
        using Source = TexturingRegs::TevStageConfig::Source;

        const Common::Vec4<u8>& const_color = draw_tev_const[tev_stage_index];
        auto get_source = [&](Source source) -> const Common::Vec4<u8>& {
            if (source == Source::Constant) {
                return const_color;
            }
            return *source_map[static_cast<std::size_t>(source) & 0xF];
        };

        /**
         * Color combiner
         * NOTE: Not sure if the alpha combiner might use the color output of the previous
         *       stage as input. Hence, we currently don't directly write the result to
         *       combiner_output.rgb(), but instead store it in a temporary variable until
         *       alpha combining has been done.
         **/
        const auto source1 = tev_stage_index == 0 && tev_stage.color_source1 == Source::Previous
                                 ? tev_stage.color_source3.Value()
                                 : tev_stage.color_source1.Value();
        const auto source2 = tev_stage_index == 0 && tev_stage.color_source2 == Source::Previous
                                 ? tev_stage.color_source3.Value()
                                 : tev_stage.color_source2.Value();
        const std::array<Common::Vec3<u8>, 3> color_result = {
            GetColorModifier(tev_stage.color_modifier1, get_source(source1)),
            GetColorModifier(tev_stage.color_modifier2, get_source(source2)),
            GetColorModifier(tev_stage.color_modifier3, get_source(tev_stage.color_source3)),
        };
        const Common::Vec3<u8> color_output = ColorCombine(tev_stage.color_op, color_result);

        u8 alpha_output;
        if (tev_stage.color_op == TexturingRegs::TevStageConfig::Operation::Dot3_RGBA) {
            // result of Dot3_RGBA operation is also placed to the alpha component
            alpha_output = color_output.x;
        } else {
            // alpha combiner
            const std::array<u8, 3> alpha_result = {{
                GetAlphaModifier(tev_stage.alpha_modifier1, get_source(tev_stage.alpha_source1)),
                GetAlphaModifier(tev_stage.alpha_modifier2, get_source(tev_stage.alpha_source2)),
                GetAlphaModifier(tev_stage.alpha_modifier3, get_source(tev_stage.alpha_source3)),
            }};
            alpha_output = AlphaCombine(tev_stage.alpha_op, alpha_result);
        }

        combiner_output[0] = std::min(255U, color_output.r() * tev_stage.GetColorMultiplier());
        combiner_output[1] = std::min(255U, color_output.g() * tev_stage.GetColorMultiplier());
        combiner_output[2] = std::min(255U, color_output.b() * tev_stage.GetColorMultiplier());
        combiner_output[3] = std::min(255U, alpha_output * tev_stage.GetAlphaMultiplier());

        combiner_buffer = next_combiner_buffer;

        if (regs.texturing.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferColor(
                tev_stage_index)) {
            next_combiner_buffer.r() = combiner_output.r();
            next_combiner_buffer.g() = combiner_output.g();
            next_combiner_buffer.b() = combiner_output.b();
        }

        if (regs.texturing.tev_combiner_buffer_input.TevStageUpdatesCombinerBufferAlpha(
                tev_stage_index)) {
            next_combiner_buffer.a() = combiner_output.a();
        }
    }

    return combiner_output;
}

void RasterizerSoftware::WriteFog(float depth, Common::Vec4<u8>& combiner_output) const {
    /**
     * Apply fog combiner. Not fully accurate. We'd have to know what data type is used to
     * store the depth etc. Using float for now until we know more about Pica datatypes.
     **/
    if (draw_fog) {
        const Common::Vec3<u8>& fog_color = draw_fog_color;

        float fog_index;
        if (regs.texturing.fog_flip) {
            fog_index = (1.0f - depth) * 128.0f;
        } else {
            fog_index = depth * 128.0f;
        }

        // Generate clamped fog factor from LUT for given fog index
        const f32 fog_i = std::clamp(Common::FastFloor(fog_index), 0.0f, 127.0f);
        const f32 fog_f = fog_index - fog_i;
        const auto& fog_lut_entry = pica.fog.lut[static_cast<u32>(fog_i)];
        f32 fog_factor = fog_lut_entry.ToFloat() + fog_lut_entry.DiffToFloat() * fog_f;
        fog_factor = std::clamp(fog_factor, 0.0f, 1.0f);
        for (u32 i = 0; i < 3; i++) {
            combiner_output[i] = static_cast<u8>(fog_factor * combiner_output[i] +
                                                 (1.0f - fog_factor) * fog_color[i]);
        }
    }
}

bool RasterizerSoftware::DoAlphaTest(u8 alpha) const {
    const auto& output_merger = regs.framebuffer.output_merger;
    if (!output_merger.alpha_test.enable) {
        return true;
    }
    switch (output_merger.alpha_test.func) {
    case FramebufferRegs::CompareFunc::Never:
        return false;
    case FramebufferRegs::CompareFunc::Always:
        return true;
    case FramebufferRegs::CompareFunc::Equal:
        return alpha == output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::NotEqual:
        return alpha != output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::LessThan:
        return alpha < output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::LessThanOrEqual:
        return alpha <= output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::GreaterThan:
        return alpha > output_merger.alpha_test.ref;
    case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
        return alpha >= output_merger.alpha_test.ref;
    default:
        LOG_CRITICAL(Render_Software, "Unknown alpha test condition {}",
                     output_merger.alpha_test.func.Value());
        return false;
    }
}

u32 RasterizerSoftware::DoDepthStencilTest8(const u32* depth_offsets, const float* depth,
                                            u32 mask) const {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto& output_merger = regs.framebuffer.output_merger;
    if (!fb.DepthIsD24S8()) {
        // D16 and D24 have no stencil and are rare; they keep the per-lane path.
        u32 pass = 0;
        for (u32 lane = 0; lane < 8; ++lane) {
            if (((mask >> lane) & 1) != 0 && DoDepthStencilTestAt(depth_offsets[lane],
                                                                  depth[lane])) {
                pass |= 1u << lane;
            }
        }
        return pass;
    }

    const auto stencil_test = output_merger.stencil_test;
    const bool stencil_action_enable = stencil_test.enable != 0;
    const bool ds_write = framebuffer.allow_depth_stencil_write != 0;
    const bool depth_write = ds_write && output_merger.depth_write_enable;
    const bool depth_test = output_merger.depth_test_enable != 0;
    const u8 stencil_ref = static_cast<u8>(stencil_test.reference_value);
    const u8 stencil_input_mask = static_cast<u8>(stencil_test.input_mask);
    const u8 stencil_write_mask = static_cast<u8>(stencil_test.write_mask);
    const u8 ref_masked = static_cast<u8>(stencil_ref & stencil_input_mask);

    const auto compare = [](FramebufferRegs::CompareFunc func, u32 a, u32 b) {
        switch (func) {
        case FramebufferRegs::CompareFunc::Never:
            return false;
        case FramebufferRegs::CompareFunc::Always:
            return true;
        case FramebufferRegs::CompareFunc::Equal:
            return a == b;
        case FramebufferRegs::CompareFunc::NotEqual:
            return a != b;
        case FramebufferRegs::CompareFunc::LessThan:
            return a < b;
        case FramebufferRegs::CompareFunc::LessThanOrEqual:
            return a <= b;
        case FramebufferRegs::CompareFunc::GreaterThan:
            return a > b;
        case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
            return a >= b;
        }
        return false;
    };

    alignas(16) u32 words[8]{};
    fb.LoadDepthWords(depth_offsets, mask, words);
    u32 pass_mask = 0;
    u32 dirty = 0;
    for (u32 lane = 0; lane < 8; ++lane) {
        if (((mask >> lane) & 1) == 0) {
            continue;
        }
        const u32 word = words[lane];
        const u8 old_stencil = static_cast<u8>(word >> 24);
        u8 new_stencil = old_stencil;
        // Writes the action's result through the write mask, exactly as update_stencil does.
        const auto act = [&](Pica::FramebufferRegs::StencilAction action) {
            if (!ds_write) {
                return;
            }
            const u8 result = PerformStencilAction(action, old_stencil, stencil_ref);
            new_stencil = static_cast<u8>((result & stencil_write_mask) |
                                          (old_stencil & ~stencil_write_mask));
        };
        const auto commit = [&]() {
            if (new_stencil != old_stencil) {
                words[lane] = (words[lane] & 0x00FFFFFFu) | (static_cast<u32>(new_stencil) << 24);
                dirty |= 1u << lane;
            }
        };

        if (stencil_action_enable) {
            const u8 dest = static_cast<u8>(old_stencil & stencil_input_mask);
            if (!compare(stencil_test.func, ref_masked, dest)) {
                act(stencil_test.action_stencil_fail);
                commit();
                continue;
            }
        }

        const u32 z = static_cast<u32>(depth[lane] * 0xFFFFFF);
        if (depth_test && !compare(output_merger.depth_test_func, z, word & 0x00FFFFFFu)) {
            if (stencil_action_enable) {
                act(stencil_test.action_depth_fail);
                commit();
            }
            continue;
        }
        if (depth_write) {
            words[lane] = (words[lane] & 0xFF000000u) | (z & 0x00FFFFFFu);
            dirty |= 1u << lane;
        }
        // The depth_pass action runs even with the depth test disabled.
        if (stencil_action_enable) {
            act(stencil_test.action_depth_pass);
            commit();
        }
        pass_mask |= 1u << lane;
    }
    fb.StoreDepthWords(depth_offsets, dirty, words);
    return pass_mask;
}

bool RasterizerSoftware::DoDepthStencilTest(u16 x, u16 y, float depth) const {
    u32 depth_offset;
    fb.BlockOffsets(x >> 4, y >> 4, 1, nullptr, &depth_offset);
    return DoDepthStencilTestAt(depth_offset, depth);
}

bool RasterizerSoftware::DoDepthStencilTestAt(u32 depth_offset, float depth) const {
    const auto& framebuffer = regs.framebuffer.framebuffer;
    const auto stencil_test = regs.framebuffer.output_merger.stencil_test;
    u8 old_stencil = 0;

    const auto update_stencil = [&](Pica::FramebufferRegs::StencilAction action) {
        const u8 new_stencil =
            PerformStencilAction(action, old_stencil, stencil_test.reference_value);
        if (framebuffer.allow_depth_stencil_write != 0) {
            const u8 stencil =
                (new_stencil & stencil_test.write_mask) | (old_stencil & ~stencil_test.write_mask);
            fb.SetStencilAt(depth_offset, stencil);
        }
    };

    const bool stencil_action_enable =
        regs.framebuffer.output_merger.stencil_test.enable &&
        regs.framebuffer.framebuffer.depth_format == FramebufferRegs::DepthFormat::D24S8;

    if (stencil_action_enable) {
        old_stencil = fb.GetStencilAt(depth_offset);
        const u8 dest = old_stencil & stencil_test.input_mask;
        const u8 ref = stencil_test.reference_value & stencil_test.input_mask;
        bool pass = false;
        switch (stencil_test.func) {
        case FramebufferRegs::CompareFunc::Never:
            pass = false;
            break;
        case FramebufferRegs::CompareFunc::Always:
            pass = true;
            break;
        case FramebufferRegs::CompareFunc::Equal:
            pass = (ref == dest);
            break;
        case FramebufferRegs::CompareFunc::NotEqual:
            pass = (ref != dest);
            break;
        case FramebufferRegs::CompareFunc::LessThan:
            pass = (ref < dest);
            break;
        case FramebufferRegs::CompareFunc::LessThanOrEqual:
            pass = (ref <= dest);
            break;
        case FramebufferRegs::CompareFunc::GreaterThan:
            pass = (ref > dest);
            break;
        case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
            pass = (ref >= dest);
            break;
        }
        if (!pass) {
            update_stencil(stencil_test.action_stencil_fail);
            return false;
        }
    }

    const u32 num_bits = FramebufferRegs::DepthBitsPerPixel(framebuffer.depth_format);
    const u32 z = static_cast<u32>(depth * ((1 << num_bits) - 1));

    const auto& output_merger = regs.framebuffer.output_merger;
    if (output_merger.depth_test_enable) {
        const u32 ref_z = fb.GetDepthAt(depth_offset);
        bool pass = false;
        switch (output_merger.depth_test_func) {
        case FramebufferRegs::CompareFunc::Never:
            pass = false;
            break;
        case FramebufferRegs::CompareFunc::Always:
            pass = true;
            break;
        case FramebufferRegs::CompareFunc::Equal:
            pass = z == ref_z;
            break;
        case FramebufferRegs::CompareFunc::NotEqual:
            pass = z != ref_z;
            break;
        case FramebufferRegs::CompareFunc::LessThan:
            pass = z < ref_z;
            break;
        case FramebufferRegs::CompareFunc::LessThanOrEqual:
            pass = z <= ref_z;
            break;
        case FramebufferRegs::CompareFunc::GreaterThan:
            pass = z > ref_z;
            break;
        case FramebufferRegs::CompareFunc::GreaterThanOrEqual:
            pass = z >= ref_z;
            break;
        }
        if (!pass) {
            if (stencil_action_enable) {
                update_stencil(stencil_test.action_depth_fail);
            }
            return false;
        }
    }
    if (framebuffer.allow_depth_stencil_write != 0 && output_merger.depth_write_enable) {
        fb.SetDepthAt(depth_offset, z);
    }
    // The stencil depth_pass action is executed even if depth testing is disabled
    if (stencil_action_enable) {
        update_stencil(stencil_test.action_depth_pass);
    }

    return true;
}

} // namespace SwRenderer
