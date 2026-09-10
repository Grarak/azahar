// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <deque>
#include "video_core/rasterizer_accelerated.h"
#include "video_core/shader/generator/profile.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_opengl/gl_shader_manager.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_stream_buffer.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"

namespace VideoCore {
class RendererBase;
}

namespace VideoCore {
class CustomTexManager;
}

namespace Pica {
struct DisplayTransferConfig;
struct MemoryFillConfig;
struct FramebufferConfig;
} // namespace Pica

namespace OpenGL {

struct ScreenInfo;

class Driver;
class ShaderProgramManager;

class RasterizerOpenGL : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerOpenGL(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                              VideoCore::CustomTexManager& custom_tex_manager,
                              VideoCore::RendererBase& renderer, Driver& driver);
    ~RasterizerOpenGL() override;

    void TickFrame();
    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;
    void SwitchDiskResources(u64 title_id) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void InvalidateGuestFlushedRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const Pica::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;

    bool AccelerateShippedDraw(const Pica::DrawPayload& payload) override;

    VideoCore::VertexRing* GetVertexRing() override;
    void DrawRingRange(u32 first, u32 count, u64 ring_end_pos) override;
    void RingRetireBlocking() override;
    void RingFenceTick() override;

private:
    /// Checks completed ring fences without blocking and retires their positions.
    void PollRingFences();

    /// Syncs pipeline state from PICA registers
    void SyncDrawState();

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Syncs all enabled PICA texture units
    void SyncTextureUnits(const Framebuffer* framebuffer);

    /// Binds the PICA shadow cube required for shadow mapping
    void BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture);

    /// Binds a texture cube to texture unit 0
    void BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture);

    /// Makes a temporary copy of the framebuffer if a feedback loop is detected
    bool IsFeedbackLoop(u32 texture_index, const Framebuffer* framebuffer, Surface& surface);

    /// Unbinds all special texture unit 0 texture configurations
    void UnbindSpecial();

    /// Binds the custom material referenced by surface if it exists.
    void BindMaterial(u32 texture_index, Surface& surface);

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed, const Pica::DrawPayload* shipped = nullptr);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed, const Pica::DrawPayload* shipped = nullptr);

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray(u8* array_ptr, GLintptr buffer_offset, GLuint vs_input_index_min,
                          GLuint vs_input_index_max, const Pica::DrawPayload* shipped = nullptr);

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

private:
    Driver& driver;
    OpenGLState state;
    Frontend::EmuWindow& render_window;
    std::vector<std::shared_ptr<ShaderProgramManager>> shader_managers;
    std::shared_ptr<ShaderProgramManager> curr_shader_manager{};
    TextureRuntime runtime;
    RasterizerCache res_cache;

    OGLVertexArray sw_vao; // VAO for software shader draw
    OGLVertexArray hw_vao; // VAO for hardware shader / accelerate draw
    std::array<bool, 16> hw_vao_enabled_attributes{};

    OGLStreamBuffer vertex_buffer;

    /// Zero-copy vertex path over vertex_buffer's persistent coherent mapping. Ranges drawn from
    /// the ring are fenced; PollRingFences/RingRetireBlocking move `ring.retired` forward.
    VideoCore::VertexRing ring;
    struct RingFence {
        GLsync sync;
        u64 ring_pos;
    };
    std::deque<RingFence> ring_fences;
    /// Highest ring position drawn since the last fence; fenced at RingFenceTick.
    u64 ring_unfenced_pos = 0;
    u64 ring_fenced_pos = 0;
    /// Range of the current DrawRingRange call, consumed by Draw's software-shader branch.
    u32 ring_draw_first = 0;
    u32 ring_draw_count = 0;
    bool ring_draw_active = false;

    OGLStreamBuffer uniform_buffer;
    OGLStreamBuffer index_buffer;
    OGLStreamBuffer texture_buffer;
    OGLStreamBuffer texture_lf_buffer;
    GLint uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs_pica;
    std::size_t uniform_size_aligned_vs;
    std::size_t uniform_size_aligned_fs;

    OGLTexture texture_buffer_lut_lf;
    OGLTexture texture_buffer_lut_rg;
    OGLTexture texture_buffer_lut_rgba;

    /// GLES: the LUTs live in 256-texel-wide 2D textures (no TBOs on 3.1-class hardware). Each
    /// dirty block takes the next row; the linear texel offset handed to the shader is row*256.
    bool lut_textures_2d = false;
    static constexpr u32 LUT_TEX_WIDTH = Pica::Shader::LUT_TEX_WIDTH;
    u32 lut_lf_row = 0;
    u32 lut_rg_row = 0;
    u32 lut_rgba_row = 0;
    u32 lut_lf_rows = 0;
    u32 lut_rg_rows = 0;
    u32 lut_rgba_rows = 0;

    /// Uploads one LUT block (<= 256 texels) into the next row; returns the linear texel offset.
    u32 UploadLutRow(u32 unit_enum, u32& row_cursor, u32 total_rows, GLenum format, GLenum type,
                     const void* data, u32 texel_count);
    bool emulate_minmax_blend{};
};

} // namespace OpenGL
