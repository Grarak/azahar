// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <bitset>
#include <deque>
#include <initializer_list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <psp2/gxm.h>
#include "video_core/rasterizer_accelerated.h"
#include "video_core/vertex_ring.h"
#include "video_core/renderer_gxm/gxm_pipeline_cache.h"
#include "video_core/renderer_gxm/gxm_shader_tier.h"
#include "video_core/shader/generator/cg_vs_shader_gen.h"
#include "video_core/renderer_gxm/gxm_texture_runtime.h"
#include "video_core/shader/generator/profile.h"

namespace VideoCore {
class CustomTexManager;
class RendererBase;
} // namespace VideoCore

namespace GxmRenderer {

/**
 * The GXM rasterizer: draws through per-framebuffer scenes with the
 * generated Cg fragment programs, and every surface - framebuffers, textures, the display
 * buffers - lives in the shared rasterizer cache instantiated on gxm_texture_runtime.
 * Fills, display transfers and texture copies are accelerated through the cache; the
 * three PICA texture units sample cached surfaces. Lighting/fog LUTs, cube maps and shadow
 * textures are P5 and sample white.
 *
 * Everything here runs on the render thread; the GXM objects that need the context are made
 * lazily on the first draw because the constructor runs on the emulation thread.
 */
class RasterizerGxm : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerGxm(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                           VideoCore::CustomTexManager& custom_tex_manager,
                           VideoCore::RendererBase& renderer);
    ~RasterizerGxm() override;

    void DrawTriangles() override;
    /// A draw shipped with its attribute bytes, shaded on the GPU by a vertex program
    /// generated from the PICA bytecode. False hands the draw to the
    /// software shader on the mirror: a geometry shader, a loader layout this backend cannot
    /// stream, or a program the decompiler refused.
    bool AccelerateShippedDraw(const Pica::DrawPayload& payload) override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void InvalidateGuestFlushedRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void NoteGuestWrite(PAddr addr, u32 size) override;

    /// The emulation thread writes triangles straight into GPU-mapped memory in guest layout;
    /// GXM reads them from there. See the ring's declaration below.
    VideoCore::VertexRing* GetVertexRing() override;
    bool WantsRingDraws() override {
        return vertex_ring.Active();
    }
    void DrawRingRange(u32 first, u32 count, u64 ring_end_pos) override;
    void RingRetireBlocking() override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;

    /// The cached surface holding the display framebuffer at `addr`, as a texture the
    /// presentation layer can sample, or null when none matches exactly (the frontend then
    /// samples guest memory). The pointer is valid until the next call for the same screen.
    /// `rect` receives the framebuffer's place inside the returned texture, in texels.
    const SceGxmTexture* AccelerateDisplay(int screen, PAddr addr, u32 width, u32 height,
                                           u32 pixel_stride, Pica::PixelFormat format,
                                           Common::Rectangle<u32>* out_rect);

    /// Once per present, from the renderer (render thread): ends the frame's scene, waits
    /// for the GPU, recycles per-frame memory, ticks the cache.
    void EndFrame();
    /// Every cached surface to ux0:data/azahar/dump (RequestSurfaceDump); GPU idle.
    void DumpSurfaces();
    /// The presentation layer's cost, charged from the renderer so the report can set it
    /// against the draws.
    void NotePresentTime(u64 us) {
        times.present += us;
    }
    [[nodiscard]] bool TimingDraws() const {
        return time_draws;
    }

    /**
     * Where one fragment program keeps every uniform this rasterizer sets, resolved once.
     *
     * sceGxmProgramFindParameterByName is a linear search over the GXP's parameter table, so
     * the names are looked up on the shader's first draw and never again. The eight light
     * sources are the reason this exists: their names are subscripted, so setting them by
     * name meant 64 fmt::format calls and 64 string hashes on every single draw, which on a
     * Cortex-A9 is more work than the draw.
     */
    /// A uniform's place in the default uniform buffer, resolved once per program: the
    /// offset in floats (the parameter's resource index, as vitaGL uses it) and how many
    /// floats are there. Writing through it is a copy; sceGxmSetUniformDataF is three
    /// libgxm calls, and a draw sets about a hundred uniforms.
    struct UniformSlot {
        const SceGxmProgramParameter* param{};
        u16 offset{};   ///< floats into the default uniform buffer
        u16 capacity{}; ///< 0 when the program has no such uniform
        /// Whether the floats sit end to end there. An array of an odd number of components
        /// does not: libgxm pads each element to eight bytes, so those go through the
        /// library call.
        bool direct{};
        explicit operator bool() const {
            return capacity != 0;
        }
    };

    struct FsUniforms {
        static constexpr u32 Lights = 8;
        UniformSlot alphatest_ref{};
        UniformSlot depth_scale{};
        UniformSlot depth_offset{};
        UniformSlot scissor_x1{};
        UniformSlot scissor_y1{};
        UniformSlot scissor_x2{};
        UniformSlot scissor_y2{};
        UniformSlot fog_lut_offset{};
        UniformSlot proctex_noise_lut_offset{};
        UniformSlot proctex_color_map_offset{};
        UniformSlot proctex_alpha_map_offset{};
        UniformSlot proctex_lut_offset{};
        UniformSlot proctex_diff_lut_offset{};
        UniformSlot proctex_bias{};
        UniformSlot lighting_lut_offset{};
        UniformSlot fog_color{};
        UniformSlot proctex_noise_f{};
        UniformSlot proctex_noise_a{};
        UniformSlot proctex_noise_p{};
        UniformSlot lighting_global_ambient{};
        UniformSlot const_color{};
        UniformSlot tev_combiner_buffer_color{};
        UniformSlot tex_lod_bias{};
        UniformSlot tex_border_color{};
        UniformSlot blend_color{};
        UniformSlot tex_dims{};
        UniformSlot fb_dims{};
        struct Light {
            UniformSlot specular_0{};
            UniformSlot specular_1{};
            UniformSlot diffuse{};
            UniformSlot ambient{};
            UniformSlot position{};
            UniformSlot spot_direction{};
            UniformSlot dist_atten_bias{};
            UniformSlot dist_atten_scale{};
        };
        std::array<Light, Lights> light_src{};
    };
    const FsUniforms& UniformsOf(Shader* fs);

    /// Where a generated vertex program keeps the PICA uniform arrays and the rest of what a
    /// hardware-shaded draw sets, resolved once per program like FsUniforms.
    struct VsUniforms {
        UniformSlot f{};
        UniformSlot i{};
        UniformSlot b{};
        UniformSlot flip_viewport{};
        UniformSlot depth_scale{};
        UniformSlot depth_offset{};
        /// The input registers no loader feeds, as uniforms; null where the program has none.
        std::array<UniformSlot, 16> default_regs{};
        bool any_default{};
    };
    const VsUniforms& VsUniformsOf(Shader* vs);

private:
    /// The vertex side of a pipeline: the program and the streams that feed it.
    struct VertexLayout {
        Shader* shader{};
        u32 stream_count{};
        std::array<u16, PipelineInfo::MaxStreams> strides{};
        u32 attribute_count{};
        std::array<VertexAttribute, PipelineInfo::MaxAttributes> attributes{};
    };
    /// What a draw holds between its framebuffer lookup and its submission: the helper's
    /// destructor invalidates the drawn region, so it lives until the draw is done.
    struct DrawSetup {
        std::optional<VideoCore::FramebufferHelper<Traits>> helper;
    };
    /// Everything a draw does before its vertices: framebuffer, textures, scene, state,
    /// programs, uniforms. False means the draw is not happening (the caller folds its ring
    /// space back and returns); the helper is left for the caller to release either way.
    bool BeginDraw(DrawSetup& setup, const VertexLayout& layout, u32 vertex_count);
    void ReleaseDraw(DrawSetup& setup);
    /// The generated vertex program for the mirror's current PICA vertex configuration, or
    /// null when the decompiler refused it (remembered, so the refusal costs once).
    Shader* VertexShaderFor(u16 default_regs, const std::array<u8, 16>& reg_components);
    void ResolveVertexUniforms();
    void EnsureRing();
    bool EnsureResources();
    /// Converts `source` into `dest` with a textured quad, letting the pixel back end do the
    /// format change on output. False when this backend cannot express the pair.
    bool GpuBlit(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);
    /// A render target of these dimensions, made once and kept: the SDK asks for render
    /// targets to be created at load time because creating one costs OS resources.
    [[nodiscard]] SceGxmRenderTarget* BlitTargetFor(u32 width, u32 height);
    void SyncToGpu(u64 serial);
    [[nodiscard]] u32 SceneFlagsFor(std::initializer_list<VideoCore::SurfaceId> touched);
    void BeginScene(Framebuffer* fb);
    /// Draws the colour clears pending on the open scene's colour surface as quads and
    /// invalidates the cached state they went through.
    void DrawPendingColourClears();

    void EndScene();
    /// Gives up the least recently used surfaces when the video memory pool is running out.
    void TrimSurfaceCache();
    void SyncState(const VideoCore::FramebufferHelper<Traits>& helper);
    /// Set by SyncState: the scissor rectangle (mode Include) holds the whole region clip,
    /// so the fragment program's scissor test could not discard anything.
    bool scissor_covers_draw = false;
    void SetStencil(SceGxmStencilFunc func, SceGxmStencilOp fail, SceGxmStencilOp zfail,
                    SceGxmStencilOp zpass, u8 compare_mask, u8 write_mask, u8 ref);
    bool BindPrograms(const VertexLayout& layout, u32 vertex_count);
    /// Writes the vertex and fragment default uniform buffers for the draw (only what
    /// changed since the buffers on the context were written).
    void UploadUniforms(Shader* fs, Shader* vs, u64 fs_hash);
    // Texture bindings persist across scenes; blits use the same cache, and EndFrame
    // invalidates it before the frontend uses the shared context.
    void BindFragmentTexture(u32 unit, const SceGxmTexture& texture);
    std::array<SceGxmTexture, 8> texture_bindings{};
    std::bitset<8> texture_binding_valid{};
    void SyncTextureUnits(const Framebuffer* framebuffer);
    /// P5a: the lighting, fog and proctex LUTs into the three 2D LUT textures the fragment
    /// programs sample on units 3-5, one row per LUT, rows handed out round-robin. Called
    /// before the scene begins: a ring wrap waits for the GPU, since the rows it reuses may
    /// still be sampled by a scene in flight.
    void SyncAndUploadLUTs();
    bool RetireRing();
    void DropDraw();

    VideoCore::RendererBase& renderer;
    std::unique_ptr<PipelineCache> pipeline_cache;
    TextureRuntime runtime;
    RasterizerCache res_cache;
    Pica::Shader::Profile profile;
    Shader* vertex_shader{};
    VertexLayout fixed_layout{}; ///< OutputVertex through the pass-through program
    /// Generated vertex programs by VertexShaderKey; null marks a configuration the
    /// decompiler refused.
    std::unordered_map<u64, Shader*> generated_vs;
    /// The second tier (gxm_shader_tier.h); null with the notier2 flag.
    std::unique_ptr<ShaderTier2> tier2;
    u32 tier2_threshold = 100; ///< draws before a program is emitted again at level 2
    /// The vertex emitter's view of the current PICA program and this draw's shape.
    Usse::VsRequest MakeVsRequest(const Pica::Shader::Generator::Cg::VSExtra& extra) const;
    /// Asks the second tier for a program past the threshold (once).
    void ConsiderTier2(Shader* shader, u64 key, const Pica::Shader::FSConfig* fs,
                       const Usse::VsRequest* vs);
    /// Swaps in the second tier's results. Render thread, at a frame's end.
    void InstallTier2();
    /// The bool uniforms each PICA program branches on, by program hash.
    std::unordered_map<u64, u16> used_bools;
    std::unordered_map<const Shader*, VsUniforms> vs_uniforms;
    /// The default attribute values the vertex uniform buffer was last written with, so a
    /// change is a compare rather than a rewrite of every draw's buffer.
    Pica::AttributeBuffer uploaded_default_attrs{};
    /// Ring space the hardware-shaded draw in flight has taken and not yet submitted.
    u64 hw_ring_end{};
    /// The conversion blit: its two programs, the quad memory it draws from (reset with the
    /// frame, after the wait that proves the GPU is done with it), and one render target per
    /// destination size.
    Shader* blit_vertex_shader{};
    Shader* blit_fragment_shader{};
    Shader* fill_fragment_shader{}; ///< a constant colour: the clear quad
    u8* blit_ring{};
    u32 blit_ring_used{};
    u32 blit_ring_limit{}; ///< the end of the half of the blit ring this frame writes
    /// The last scene serial of each of the two most recent frames: what a per-frame
    /// resource (the blit ring's halves) waits for before it is written again.
    u64 frame_serial[2]{};
    std::unordered_map<u32, SceGxmRenderTarget*> blit_targets;
    std::unordered_map<const Shader*, FsUniforms> fs_uniforms;
    /// Fragment programs emitted, and configurations the emitter refused (whose draws are
    /// skipped); counted once per configuration, never reset.
    u32 stat_usse_emitted{}, stat_usse_refused{};
    u32 stat_vs_emitted{}, stat_vs_refused{}; ///< vertex programs by the emitter / refused
    u32 stat_tier2{}, stat_tier2_refused{};  ///< programs the second tier replaced / refused
    std::unordered_set<u64> usse_refused; ///< configurations logged as refused
    /**
     * What the default uniform buffers currently on the context were written for.
     *
     * A reservation persists as context state until the next one, so a draw whose program and
     * uniform data are unchanged can use the buffer the last draw wrote and reserve nothing.
     * That matters twice over: the fragment set is around 150 floats through 40
     * sceGxmSetUniformDataF calls, and each reservation comes out of the 512 KiB fragment
     * data ring, which libgxm recycles by *splitting the scene* when it runs dry - at ~600
     * bytes a draw that is a split every few hundred draws, on top of the ones the cache
     * asks for.
     *
     * The keys are the pipeline (a blend variant is a different fragment program, and the
     * layout belongs to the program) and the scene, since the ring is recycled at scene
     * boundaries.
     */
    const Pipeline* uniform_pipeline{};
    u64 uniform_scene{};
    /// The last reserved vertex default uniform buffer (valid until the next reservation).
    const u8* last_vertex_uniforms{};
    bool tex_dims_dirty{true};
    std::array<float, 2> uniform_fb_dims{};
    const Shader* cached_fs{};      ///< the last shader UniformsOf answered for
    const FsUniforms* cached_locs{};
    // The vertex program is one program for every draw, so its three go here directly.
    /// What each scene of a frame drew into, in the order the scenes were opened - the
    /// order Razor GPU Live indexes them by, so its per-scene fragment time can be read
    /// against a target. The last frame's list is kept for the periodic report.
    struct SceneRecord {
        u32 addr;
        u16 width, height;
        u16 draws;
        bool blit;
    };
    static constexpr u32 MaxSceneRecords = 12;
    /// The last scenes handed to sceGxmEndScene, newest last, for the stuck-queue log: which
    /// scene the GPU stopped on, what it drew into and with which programs. Names point at
    /// the pipeline cache's shaders, which are never freed while it lives.
    struct SceneTrace {
        u64 serial;
        u32 addr;
        u16 width, height, draws;
        bool blit;
        u8 names;
        std::array<const char*, 4> vs;
        std::array<const char*, 4> fs;
    };
    static constexpr u32 SceneTraceLen = 24;
    std::array<SceneTrace, SceneTraceLen> scene_trace{};
    u32 scene_trace_next = 0;
    SceneTrace scene_open{};
    void OpenSceneTrace(u32 addr, u16 width, u16 height, bool blit);
    void NoteScenePipeline(const char* vs, const char* fs);
    void CloseSceneTrace();
    void ReportSceneTrace();
    std::array<SceneRecord, MaxSceneRecords> scene_records{};
    std::array<SceneRecord, MaxSceneRecords> last_frame_scenes{};
    u32 scenes_this_frame = 0;
    u32 last_frame_scene_count = 0;

    UniformSlot vs_flip_viewport{};
    UniformSlot vs_depth_scale{};
    UniformSlot vs_depth_offset{};
    bool vs_uniforms_resolved{};

    /**
     * The GXM context state this rasterizer last set, so an unchanged draw sets nothing.
     *
     * Every setter here is a write into the context's state block plus its dirty tracking,
     * and there are fifteen of them per draw against a PICA configuration that usually holds
     * still for a run of draws. The cache is dropped whenever something else can have moved
     * the state: sceGxmBeginScene resets the viewport and region clip itself (the SDK says so
     * explicitly), and between frames the presentation layer draws its own quads through the
     * same context.
     */
    struct CachedState {
        bool valid{};
        float vp[6]{};
        u32 clip[4]{};
        SceGxmCullMode cull{};
        SceGxmDepthFunc depth_func{};
        SceGxmDepthWriteMode depth_write{};
        bool stencil_set{};
        SceGxmStencilFunc stencil_func{};
        SceGxmStencilOp stencil_fail{}, stencil_zfail{}, stencil_zpass{};
        u8 stencil_compare{}, stencil_write{}, stencil_ref{};
    };
    CachedState state_cache;

    Framebuffer* scene_fb{};
    bool in_scene{};
    SceGxmNotification scene_notification{};
    u32 frame_index{};
    u32 surface_dump_index{};
    std::array<float, 6> tex_dims{1, 1, 1, 1, 1, 1};
    std::array<SceGxmTexture, 2> display_textures{};
    /// The surfaces handed to the presentation layer this frame, cleared once EndFrame has
    /// proved the frontend's scene is finished with them.
    std::array<VideoCore::SurfaceId, 2> display_surfaces{};
    /// The cached surfaces this draw's texture units sample, noted before the scene begins
    /// and given its serial once it has one.
    std::array<VideoCore::SurfaceId, 3> bound_surfaces{};

    /**
     * Vertices, in guest layout, in memory the GPU reads directly.
     *
     * The path this replaces built a HardwareVertex per vertex on the emulation thread - 88
     * bytes, twenty-two components copied out one at a time - pushed it into a std::vector,
     * and then memcpied the whole batch into GPU memory on the render thread. Pica::
     * OutputVertex is already float and already laid out; the vertex attributes just name its
     * offsets, so the emulation thread's store into the ring is the only time a vertex is
     * written, and the render thread does nothing per vertex at all.
     *
     * Ring space comes back when the GPU passes the scene that read it: `ring_retire` holds
     * one entry per scene, and CompletedEpoch() says which of them are done.
     */
    VideoCore::VertexRing vertex_ring;
    u8* ring{};
    u32 ring_size{};
    u64 ring_scene_pos{}; ///< the furthest ring position the open scene has drawn from
    struct RingRetire {
        u64 serial;
        u64 pos;
    };
    std::deque<RingRetire> ring_retire;
    u16* identity_indices{};
    void* white_mem{};
    SceGxmTexture white_texture{};
    /// The LUT textures: 256 texels wide, LUT_*_ROWS rows, sampled with nearest filtering
    /// and clamping so a row index reads exactly one LUT entry. lf and rg hold (value,
    /// delta) pairs as two floats; rgba holds proctex colours as bytes, which is what they
    /// are.
    struct LutTexture {
        void* mem{};
        SceGxmTexture texture{};
        u32 rows{};
        u32 row_cursor{};
        u32 texel_bytes{};
        u32 half{};           ///< the half of the ring rows are being handed out from
        u64 left_serial[2]{}; ///< per half: the last scene that may sample its rows
    };
    LutTexture lut_lf, lut_rg, lut_rgba;
    /// Hands out `needed` LUT rows: wraps the ring when they do not fit (the caller then
    /// re-uploads every LUT of that texture) and, on entering the other half of the ring,
    /// waits for the scenes that last sampled it, which a ring of a few frames' rows makes
    /// a wait that is over before it starts.
    bool ReserveLutRows(LutTexture& lut, u32 needed);
    u32 stat_lut_waits{};
    u32 stat_trimmed{}; ///< surfaces given up because the pool was filling
    /// The vertex program's float uniform bank as the GPU takes it: 96 PICA rows converted
    /// from f24, 32 rows of ones, rows 0-3 again. Kept across draws and converted only over
    /// the window a shipment changed, then copied whole into the reserved buffer; converting
    /// all 384 components per draw was the alternative.
    alignas(16) std::array<float, 132 * 4> vs_bank{};
    bool vs_bank_valid = false;
    /// Texture units bound to the white texture because the cache found no surface, or a
    /// surface with no memory behind it: what a blank picture after a trim is made of.
    u32 stat_white_nosurface{}, stat_white_nomem{};
    bool resources_ready{};
    bool resources_failed{};
    // Counters, reported on a wall-clock interval (a frame count is useless at 1 fps).
    static constexpr u64 ReportIntervalMs = 3000;
    u64 report_at_ms{};
    u32 stat_frames{}, stat_draws{}, stat_vertices{}, stat_scenes{}, stat_skipped{};
    u32 stat_finishes{}, stat_syncs{}, stat_scene_breaks{}, stat_ring_peak{};
    u32 stat_uniform_uploads{}, stat_uniform_skips{}, stat_dropped_draws{}, stat_scene_waits{};
    u32 stat_gpu_blit_logs{}, stat_guest_writes{}, stat_invalidates{}, stat_flushes{};
    u32 stat_hw_draws{}, stat_hw_refused{};
    u32 stat_scissor_free{}; ///< draws whose scissor test the region clip made redundant

    /**
     * Where the render thread's time actually goes, under the `timedraw` switch.
     *
     * Reasoning from call counts put every phase of a draw together at a few percent of the
     * frame, which means the estimate is wrong somewhere and there is no point guessing which
     * term. These are microseconds accumulated over a report window, from the same clock the
     * frontend's own counters use, so the phases can be read against the frame time rather
     * than against arithmetic.
     */
    struct DrawTimes {
        u64 framebuffer;  ///< GetFramebufferSurfaces: surface lookup, validation, upload
        u64 textures;     ///< SyncTextureUnits: three cache lookups and their uploads
        u64 scene;        ///< BeginScene, and the sceGxmFinish a sync inside it costs
        u64 scene_end;    ///< the sceGxmEndScene calls: what closing a scene costs this thread
        u64 state;        ///< SyncState
        u64 programs;     ///< BindPrograms without the uniforms: config, hash, patcher
        u64 uniforms;     ///< UploadUniforms
        u64 draw;         ///< sceGxmSetVertexStream and sceGxmDraw
        u64 invalidate;   ///< ~FramebufferHelper, which invalidates the drawn region
        u64 total;        ///< all of DrawTriangles, to see what the phases miss
        u64 finish;       ///< EndFrame's sceGxmFinish: the GPU, not this thread
        u64 end_frame;    ///< all of EndFrame
        // The rest of what the render thread calls this rasterizer for. The first pass timed
        // only the draws and found they were four percent of the wall, so the question is
        // entirely about what else it is doing.
        u64 guest_writes;      ///< NoteGuestWrite: the cache invalidating what the guest wrote
        u64 invalidate_region; ///< InvalidateRegion from the emulation side
        u64 flush;             ///< FlushRegion and FlushAll
        u64 transfers;         ///< accelerated display transfers, texture copies and fills
        u64 retire;            ///< RingRetireBlocking: the producer ran out of ring
        u64 sync;              ///< SyncToGpu: the cache needed the GPU idle
        u64 present;           ///< the frontend's present scene and display-queue wait
    };
    DrawTimes times{};
    bool time_draws{};
    u64 busy_total_at_report{};
};

} // namespace GxmRenderer
