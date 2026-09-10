// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdlib>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/literals.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/microprofile.h"
#include "core/loader/loader.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_opengl/gl_rasterizer.h"
#include "video_core/video_core.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/renderer_opengl/pica_to_gl.h"
#include "video_core/renderer_opengl/renderer_opengl.h"
#include "video_core/shader/generator/shader_gen.h"
#include "video_core/texture/texture_decode.h"

namespace OpenGL {

namespace {

MICROPROFILE_DEFINE(OpenGL_VAO, "OpenGL", "Vertex Array Setup", MP_RGB(255, 128, 0));
MICROPROFILE_DEFINE(OpenGL_VS, "OpenGL", "Vertex Shader Setup", MP_RGB(192, 128, 128));
MICROPROFILE_DEFINE(OpenGL_GS, "OpenGL", "Geometry Shader Setup", MP_RGB(128, 192, 128));
MICROPROFILE_DEFINE(OpenGL_Drawing, "OpenGL", "Drawing", MP_RGB(128, 128, 192));
MICROPROFILE_DEFINE(OpenGL_Display, "OpenGL", "Display", MP_RGB(128, 128, 192));

using VideoCore::SurfaceType;
using namespace Common::Literals;
using namespace Pica::Shader::Generator;

// The vertex buffer doubles as the zero-copy ring the emulation thread writes shaded vertices
// into; sized for several uncapped frames of triangles so ring-full waits stay rare.
constexpr std::size_t VERTEX_BUFFER_SIZE = 64_MiB;
constexpr std::size_t INDEX_BUFFER_SIZE = 2_MiB;
constexpr std::size_t UNIFORM_BUFFER_SIZE = 8_MiB;
constexpr std::size_t TEXTURE_BUFFER_SIZE = 2_MiB;

GLenum MakePrimitiveMode(Pica::PipelineRegs::TriangleTopology topology) {
    switch (topology) {
    case Pica::PipelineRegs::TriangleTopology::Shader:
    case Pica::PipelineRegs::TriangleTopology::List:
        return GL_TRIANGLES;
    case Pica::PipelineRegs::TriangleTopology::Fan:
        return GL_TRIANGLE_FAN;
    case Pica::PipelineRegs::TriangleTopology::Strip:
        return GL_TRIANGLE_STRIP;
    default:
        UNREACHABLE();
    }
    return GL_TRIANGLES;
}

GLenum MakeAttributeType(Pica::PipelineRegs::VertexAttributeFormat format) {
    switch (format) {
    case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
        return GL_BYTE;
    case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
        return GL_UNSIGNED_BYTE;
    case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
        return GL_SHORT;
    case Pica::PipelineRegs::VertexAttributeFormat::FLOAT:
        return GL_FLOAT;
    }
    return GL_UNSIGNED_BYTE;
}

[[nodiscard]] GLsizeiptr TextureBufferSize(const Driver& driver, bool is_lf) {
    // Use the smallest texel size from the texel views
    // which corresponds to GL_RG32F
    GLint max_texel_buffer_size;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &max_texel_buffer_size);
    GLsizeiptr candidate = std::min<GLsizeiptr>(max_texel_buffer_size * 8ULL, TEXTURE_BUFFER_SIZE);

    if (driver.HasBug(DriverBug::SlowTextureBufferWithBigSize) && !is_lf) {
        constexpr GLsizeiptr FIXUP_TEXTURE_BUFFER_SIZE = static_cast<GLsizeiptr>(1 << 14); // 16384
        return FIXUP_TEXTURE_BUFFER_SIZE;
    }

    return candidate;
}

} // Anonymous namespace

RasterizerOpenGL::RasterizerOpenGL(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                   VideoCore::CustomTexManager& custom_tex_manager,
                                   VideoCore::RendererBase& renderer, Driver& driver_)
    : VideoCore::RasterizerAccelerated{memory, pica}, driver{driver_},
      render_window{renderer.GetRenderWindow()}, runtime{driver, renderer},
      res_cache{memory, custom_tex_manager, runtime, regs, renderer},
      vertex_buffer{driver, GL_ARRAY_BUFFER, VERTEX_BUFFER_SIZE, true},
      uniform_buffer{driver, GL_UNIFORM_BUFFER, UNIFORM_BUFFER_SIZE},
      index_buffer{driver, GL_ELEMENT_ARRAY_BUFFER, INDEX_BUFFER_SIZE},
      texture_buffer{driver, GL_TEXTURE_BUFFER, TextureBufferSize(driver, false)},
      texture_lf_buffer{driver, GL_TEXTURE_BUFFER, TextureBufferSize(driver, true)} {

    // Clipping plane 0 is always enabled for PICA fixed clip plane z <= 0
    state.clip_distance[0] = true;

    // Generate VAO
    sw_vao.Create();
    hw_vao.Create();

    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniform_buffer_alignment);
    uniform_size_aligned_vs_pica =
        Common::AlignUp<std::size_t>(sizeof(VSPicaUniformData), uniform_buffer_alignment);
    uniform_size_aligned_vs =
        Common::AlignUp<std::size_t>(sizeof(VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs =
        Common::AlignUp<std::size_t>(sizeof(FSUniformData), uniform_buffer_alignment);

    // Set vertex attributes for software shader path. With a persistently mapped vertex buffer
    // the ring holds Pica::OutputVertex exactly as the emulation thread produced it — the GL
    // reads the guest layout in place and the HardwareVertex relayout never happens. Without the
    // mapping the batch is HardwareVertex uploaded by memcpy, as before.
    state.draw.vertex_array = sw_vao.handle;
    state.draw.vertex_buffer = vertex_buffer.GetHandle();
    state.Apply();

    const bool ring_layout = vertex_buffer.CoherentMapping() != nullptr;
    const GLsizei stride =
        ring_layout ? sizeof(Pica::OutputVertex) : sizeof(HardwareVertex);
    const auto attrib = [&](GLuint index, GLint size, std::size_t ring_offset,
                            std::size_t batch_offset) {
        glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, stride,
                              (GLvoid*)(ring_layout ? ring_offset : batch_offset));
        glEnableVertexAttribArray(index);
    };
    attrib(ATTRIBUTE_POSITION, 4, offsetof(Pica::OutputVertex, pos),
           offsetof(HardwareVertex, position));
    attrib(ATTRIBUTE_COLOR, 4, offsetof(Pica::OutputVertex, color),
           offsetof(HardwareVertex, color));
    attrib(ATTRIBUTE_TEXCOORD0, 2, offsetof(Pica::OutputVertex, tc0),
           offsetof(HardwareVertex, tex_coord0));
    attrib(ATTRIBUTE_TEXCOORD1, 2, offsetof(Pica::OutputVertex, tc1),
           offsetof(HardwareVertex, tex_coord1));
    attrib(ATTRIBUTE_TEXCOORD2, 2, offsetof(Pica::OutputVertex, tc2),
           offsetof(HardwareVertex, tex_coord2));
    attrib(ATTRIBUTE_TEXCOORD0_W, 1, offsetof(Pica::OutputVertex, tc0_w),
           offsetof(HardwareVertex, tex_coord0_w));
    attrib(ATTRIBUTE_NORMQUAT, 4, offsetof(Pica::OutputVertex, quat),
           offsetof(HardwareVertex, normquat));
    attrib(ATTRIBUTE_VIEW, 3, offsetof(Pica::OutputVertex, view),
           offsetof(HardwareVertex, view));

    // Allocate and bind the LUT textures. Desktop GL exposes them as texture buffers over the
    // stream buffers; GLES uses 2D textures so 3.1-class hardware without TBOs works, uploaded
    // row-per-block by UploadLutRow.
    lut_textures_2d = driver.IsOpenGLES();
    texture_buffer_lut_lf.Create();
    texture_buffer_lut_rg.Create();
    texture_buffer_lut_rgba.Create();
    if (lut_textures_2d) {
        // Shape comes from the shader generator's constants: the fragment shader bakes the
        // index arithmetic in, so the allocation must match it.
        lut_lf_rows = Pica::Shader::LUT_LF_ROWS;
        lut_rg_rows = Pica::Shader::LUT_RG_ROWS;
        lut_rgba_rows = Pica::Shader::LUT_RGBA_ROWS;
        const auto setup_2d = [&](u32 unit_enum, const OGLTexture& tex, GLenum internal_format,
                                  u32 rows) {
            glActiveTexture(unit_enum);
            glBindTexture(GL_TEXTURE_2D, tex.handle);
            glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, LUT_TEX_WIDTH, rows);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            // Clamped so the normalised LUT reads used when the target has no texelFetch
            // cannot wrap at the ends of a row.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        setup_2d(TextureUnits::TextureBufferLUT_LF.Enum(), texture_buffer_lut_lf, GL_RG32F,
                 lut_lf_rows);
        setup_2d(TextureUnits::TextureBufferLUT_RG.Enum(), texture_buffer_lut_rg, GL_RG32F,
                 lut_rg_rows);
        setup_2d(TextureUnits::TextureBufferLUT_RGBA.Enum(), texture_buffer_lut_rgba, GL_RGBA32F,
                 lut_rgba_rows);
    } else {
        state.texture_buffer_lut_lf.texture_buffer = texture_buffer_lut_lf.handle;
        state.texture_buffer_lut_rg.texture_buffer = texture_buffer_lut_rg.handle;
        state.texture_buffer_lut_rgba.texture_buffer = texture_buffer_lut_rgba.handle;
        state.Apply();
        glActiveTexture(TextureUnits::TextureBufferLUT_LF.Enum());
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, texture_lf_buffer.GetHandle());
        glActiveTexture(TextureUnits::TextureBufferLUT_RG.Enum());
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, texture_buffer.GetHandle());
        glActiveTexture(TextureUnits::TextureBufferLUT_RGBA.Enum());
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, texture_buffer.GetHandle());
    }

    // Bind index buffer for hardware shader path
    state.draw.vertex_array = hw_vao.handle;
    state.Apply();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer.GetHandle());

    glEnable(GL_BLEND);

    // Zero-copy vertex path: when the vertex buffer is persistently and coherently mapped, the
    // emulation thread writes HardwareVertex data straight into it and draws arrive as ring
    // ranges. The usable size is truncated to a whole number of vertices so byte positions
    // divide evenly into first-vertex indices.
    if (u8* const mapping = vertex_buffer.CoherentMapping()) {
        ring.base = mapping;
        ring.size = (static_cast<u64>(vertex_buffer.GetSize()) / sizeof(Pica::OutputVertex)) *
                    sizeof(Pica::OutputVertex);
        LOG_INFO(Render_OpenGL, "Zero-copy vertex ring active: {} MiB", ring.size >> 20);
    } else {
        LOG_INFO(Render_OpenGL, "No persistent coherent mapping; vertex data will be copied");
    }
    // What the renderer holds, on request from the debug port.
    VideoCore::surface_dump_hook = [](void* self, const char* dir) {
        static_cast<RasterizerOpenGL*>(self)->res_cache.DumpSurfaces(dir, 0);
    };
    VideoCore::surface_dump_user = this;
}

RasterizerOpenGL::~RasterizerOpenGL() {
    VideoCore::surface_dump_hook = nullptr;
    VideoCore::surface_dump_user = nullptr;
    for (const auto& fence : ring_fences) {
        glDeleteSync(fence.sync);
    }
}

VideoCore::VertexRing* RasterizerOpenGL::GetVertexRing() {
    return ring.Active() ? &ring : nullptr;
}

void RasterizerOpenGL::PollRingFences() {
    while (!ring_fences.empty()) {
        const auto& fence = ring_fences.front();
        const GLenum status = glClientWaitSync(fence.sync, 0, 0);
        if (status != GL_ALREADY_SIGNALED && status != GL_CONDITION_SATISFIED) {
            break;
        }
        glDeleteSync(fence.sync);
        ring.Retire(fence.ring_pos);
        ring_fences.pop_front();
    }
}

void RasterizerOpenGL::RingFenceTick() {
    if (ring_unfenced_pos > ring_fenced_pos) {
        ring_fences.push_back({glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0), ring_unfenced_pos});
        ring_fenced_pos = ring_unfenced_pos;
    }
    PollRingFences();
}

void RasterizerOpenGL::RingRetireBlocking() {
    if (ring_fences.empty()) {
        // Nothing in flight is fenced yet: fence what has been drawn so far so there is
        // something to wait on. Without this a full ring between presents would spin.
        if (ring_unfenced_pos <= ring_fenced_pos) {
            return;
        }
        RingFenceTick();
        if (ring_fences.empty()) {
            return;
        }
    }
    const auto& fence = ring_fences.front();
    // A generous timeout in a loop rather than GL_TIMEOUT_IGNORED: a wedged GL should surface as
    // a stall with log context, not a silent hang.
    GLenum status;
    do {
        status = glClientWaitSync(fence.sync, GL_SYNC_FLUSH_COMMANDS_BIT, 100'000'000);
    } while (status == GL_TIMEOUT_EXPIRED);
    glDeleteSync(fence.sync);
    ring.Retire(fence.ring_pos);
    ring_fences.pop_front();
    PollRingFences();
}

void RasterizerOpenGL::DrawRingRange(u32 first, u32 count, u64 ring_end_pos) {
    ring_draw_first = first;
    ring_draw_count = count;
    ring_draw_active = true;
    Draw(false, false);
    ring_draw_active = false;
    ring_unfenced_pos = std::max(ring_unfenced_pos, ring_end_pos);
}

void RasterizerOpenGL::TickFrame() {
    // The runtime's resource tick is what the garbage collector and the trim measure age
    // against, and nothing else on this backend advanced it: sentenced surfaces were never
    // freed and every surface looked used this tick.
    runtime.Finish();
    res_cache.TickFrame();
}

void RasterizerOpenGL::LoadDefaultDiskResources(
    const std::atomic_bool& stop_loading, const VideoCore::DiskResourceLoadCallback& callback) {
    // First element in vector is the default one and cannot be removed.
    u64 program_id;
    if (Core::System::GetInstance().GetAppLoader().ReadProgramId(program_id) !=
        Loader::ResultStatus::Success) {
        program_id = 0;
    }

    shader_managers.clear();
    curr_shader_manager = shader_managers.emplace_back(std::make_shared<ShaderProgramManager>(
        render_window, driver, program_id, !driver.IsOpenGLES()));

    curr_shader_manager->LoadDiskCache(stop_loading, callback, accurate_mul);
}

void RasterizerOpenGL::SwitchDiskResources(u64 title_id) {
    // NOTE: curr_shader_manager can be null if emulation restarted without calling
    // LoadDefaultDiskResources

    // Check if the current manager is for the specified TID.
    if (curr_shader_manager && curr_shader_manager->GetProgramID() == title_id) {
        return;
    }

    // Search for an existing manager
    size_t new_pos = 0;
    for (new_pos = 0; new_pos < shader_managers.size(); new_pos++) {
        if (shader_managers[new_pos]->GetProgramID() == title_id) {
            break;
        }
    }
    // Manager does not exist, create it and append to the end
    if (new_pos >= shader_managers.size()) {
        new_pos = shader_managers.size();
        auto& new_manager = shader_managers.emplace_back(std::make_shared<ShaderProgramManager>(
            render_window, driver, title_id, !driver.IsOpenGLES()));

        if (switch_disk_resources_callback) {
            switch_disk_resources_callback(VideoCore::LoadCallbackStage::Prepare, 0, 0, "");
        }

        std::atomic_bool stop_loading;
        new_manager->LoadDiskCache(stop_loading, switch_disk_resources_callback, accurate_mul);

        if (switch_disk_resources_callback) {
            switch_disk_resources_callback(VideoCore::LoadCallbackStage::Complete, 0, 0, "");
        }
    }

    auto is_applet = [](u64 tid) {
        constexpr u32 APPLET_TID_HIGH = 0x00040030;
        return static_cast<u32>(tid >> 32) == APPLET_TID_HIGH;
    };

    bool prev_applet = curr_shader_manager ? is_applet(curr_shader_manager->GetProgramID()) : false;
    bool new_applet = is_applet(shader_managers[new_pos]->GetProgramID());
    curr_shader_manager = shader_managers[new_pos];

    if (prev_applet) {
        // If we came from an applet, clean up all other applets
        for (auto it = shader_managers.begin(); it != shader_managers.end();) {
            if (it == shader_managers.begin() || *it == curr_shader_manager ||
                !is_applet((*it)->GetProgramID())) {
                it++;
                continue;
            }
            it = shader_managers.erase(it);
        }
    }
    if (!new_applet) {
        // If we are going into a non-applet, clean up everything
        for (auto it = shader_managers.begin(); it != shader_managers.end();) {
            if (it == shader_managers.begin() || *it == curr_shader_manager) {
                it++;
                continue;
            }
            it = shader_managers.erase(it);
        }
    }
}

void RasterizerOpenGL::SyncDrawState() {
    SyncDrawUniforms();

    // SyncClipEnabled();
    state.clip_distance[1] = regs.rasterizer.clip_enable != 0;
    // SyncCullMode();
    state.cull.enabled = regs.rasterizer.cull_mode != Pica::RasterizerRegs::CullMode::KeepAll &&
                         regs.rasterizer.cull_mode != Pica::RasterizerRegs::CullMode::KeepAll2;
    if (state.cull.enabled) {
        state.cull.front_face =
            regs.rasterizer.cull_mode == Pica::RasterizerRegs::CullMode::KeepClockWise ? GL_CW
                                                                                       : GL_CCW;
    }
    // If the framebuffer is flipped, vertex shader flips vertex y, so invert culling
    const bool is_flipped = regs.framebuffer.framebuffer.IsFlipped();
    state.cull.mode = is_flipped && state.cull.enabled ? GL_FRONT : GL_BACK;
    // SyncBlendEnabled();
    state.blend.enabled = (regs.framebuffer.output_merger.alphablend_enable == 1);
    // SyncBlendFuncs();
    const bool has_minmax_factor = driver.HasBlendMinMaxFactor();
    state.blend.rgb_equation = PicaToGL::BlendEquation(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb, has_minmax_factor);
    state.blend.a_equation = PicaToGL::BlendEquation(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_a, has_minmax_factor);
    state.blend.src_rgb_func =
        PicaToGL::BlendFunc(regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    state.blend.dst_rgb_func =
        PicaToGL::BlendFunc(regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    state.blend.src_a_func =
        PicaToGL::BlendFunc(regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    state.blend.dst_a_func =
        PicaToGL::BlendFunc(regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
    if (!has_minmax_factor) {
        // Blending with min/max equations is emulated in the fragment shader so
        // configure blending to not modify the incoming fragment color.
        emulate_minmax_blend = false;
        if (state.EmulateColorBlend()) {
            emulate_minmax_blend = true;
            state.blend.rgb_equation = GL_FUNC_ADD;
            state.blend.src_rgb_func = GL_ONE;
            state.blend.dst_rgb_func = GL_ZERO;
        }
        if (state.EmulateAlphaBlend()) {
            emulate_minmax_blend = true;
            state.blend.a_equation = GL_FUNC_ADD;
            state.blend.src_a_func = GL_ONE;
            state.blend.dst_a_func = GL_ZERO;
        }
    }
    // SyncBlendColor();
    const auto blend_color = PicaToGL::ColorRGBA8(regs.framebuffer.output_merger.blend_const.raw);
    state.blend.color.red = blend_color[0];
    state.blend.color.green = blend_color[1];
    state.blend.color.blue = blend_color[2];
    state.blend.color.alpha = blend_color[3];
    // SyncLogicOp();
    // SyncColorWriteMask();
    state.logic_op = PicaToGL::LogicOp(regs.framebuffer.output_merger.logic_op);
    if (driver.IsOpenGLES() && !regs.framebuffer.output_merger.alphablend_enable &&
        regs.framebuffer.output_merger.logic_op == Pica::FramebufferRegs::LogicOp::NoOp) {
        // Color output is disabled by logic operation. We use color write mask to skip
        // color but allow depth write.
        state.color_mask = {};
    } else {
        auto is_color_write_enabled = [&](u32 value) {
            return (regs.framebuffer.framebuffer.allow_color_write != 0 && value != 0) ? GL_TRUE
                                                                                       : GL_FALSE;
        };
        state.color_mask.red_enabled =
            is_color_write_enabled(regs.framebuffer.output_merger.red_enable);
        state.color_mask.green_enabled =
            is_color_write_enabled(regs.framebuffer.output_merger.green_enable);
        state.color_mask.blue_enabled =
            is_color_write_enabled(regs.framebuffer.output_merger.blue_enable);
        state.color_mask.alpha_enabled =
            is_color_write_enabled(regs.framebuffer.output_merger.alpha_enable);
    }
    // SyncStencilTest();
    state.stencil.test_enabled =
        regs.framebuffer.output_merger.stencil_test.enable &&
        regs.framebuffer.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8;
    state.stencil.test_func =
        PicaToGL::CompareFunc(regs.framebuffer.output_merger.stencil_test.func);
    state.stencil.test_ref = regs.framebuffer.output_merger.stencil_test.reference_value;
    state.stencil.test_mask = regs.framebuffer.output_merger.stencil_test.input_mask;
    state.stencil.action_stencil_fail =
        PicaToGL::StencilOp(regs.framebuffer.output_merger.stencil_test.action_stencil_fail);
    state.stencil.action_depth_fail =
        PicaToGL::StencilOp(regs.framebuffer.output_merger.stencil_test.action_depth_fail);
    state.stencil.action_depth_pass =
        PicaToGL::StencilOp(regs.framebuffer.output_merger.stencil_test.action_depth_pass);
    // SyncDepthTest();
    state.depth.test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                               regs.framebuffer.output_merger.depth_write_enable == 1;
    state.depth.test_func =
        regs.framebuffer.output_merger.depth_test_enable == 1
            ? PicaToGL::CompareFunc(regs.framebuffer.output_merger.depth_test_func)
            : GL_ALWAYS;
    // SyncStencilWriteMask();
    state.stencil.write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<GLuint>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;
    // SyncDepthWriteMask();
    state.depth.write_mask = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                              regs.framebuffer.output_merger.depth_write_enable)
                                 ? GL_TRUE
                                 : GL_FALSE;
}

void RasterizerOpenGL::SetupVertexArray(u8* array_ptr, GLintptr buffer_offset,
                                        GLuint vs_input_index_min, GLuint vs_input_index_max,
                                        const Pica::DrawPayload* shipped) {
    MICROPROFILE_SCOPE(OpenGL_VAO);
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    PAddr base_address = vertex_attributes.GetPhysicalBaseAddress();

    state.draw.vertex_array = hw_vao.handle;
    state.draw.vertex_buffer = vertex_buffer.GetHandle();
    state.Apply();

    std::array<bool, 16> enable_attributes{};

    u32 loader_index = static_cast<u32>(-1);
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        loader_index++;
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }
        if (shipped && !(shipped->layout.used_mask & (1u << loader_index))) {
            // The emulation side recorded no attribute bytes for this loader (padding-only);
            // nothing to upload and nothing references it.
            continue;
        }

        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; ++comp) {
            u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index < 12) {
                if (vertex_attributes.GetNumElements(attribute_index) != 0) {
                    offset = Common::AlignUp(
                        offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

                    u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
                    GLint size = vertex_attributes.GetNumElements(attribute_index);
                    GLenum type = MakeAttributeType(vertex_attributes.GetFormat(attribute_index));
                    GLsizei stride = loader.byte_count;
                    glVertexAttribPointer(input_reg, size, type, GL_FALSE, stride,
                                          reinterpret_cast<GLvoid*>(buffer_offset + offset));
                    enable_attributes[input_reg] = true;

                    offset += vertex_attributes.GetStride(attribute_index);
                }
            } else {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings,
                // respectively
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            }
        }

        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        const u32 data_size = loader.byte_count * vertex_num;

        if (shipped) {
            // The draw's attribute bytes were frozen in the arena when it shipped; the arena
            // layout matches this loader walk by construction.
            std::memcpy(array_ptr, shipped->arena + shipped->layout.loader_offset[loader_index],
                        data_size);
        } else {
            const PAddr data_addr =
                base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
            res_cache.FlushRegion(data_addr, data_size);
            std::memcpy(array_ptr, memory.GetPhysicalPointer(data_addr), data_size);
        }

        array_ptr += data_size;
        buffer_offset += data_size;
    }

    for (std::size_t i = 0; i < enable_attributes.size(); ++i) {
        if (enable_attributes[i] != hw_vao_enabled_attributes[i]) {
            if (enable_attributes[i]) {
                glEnableVertexAttribArray(static_cast<GLuint>(i));
            } else {
                glDisableVertexAttribArray(static_cast<GLuint>(i));
            }
            hw_vao_enabled_attributes[i] = enable_attributes[i];
        }

        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = pica.input_default_attributes[i];
                glVertexAttrib4f(reg, attr.x.ToFloat32(), attr.y.ToFloat32(), attr.z.ToFloat32(),
                                 attr.w.ToFloat32());
            }
        }
    }
}

bool RasterizerOpenGL::SetupVertexShader() {
    MICROPROFILE_SCOPE(OpenGL_VS);
    return curr_shader_manager->UseProgrammableVertexShader(regs, pica.vs_setup, accurate_mul);
}

bool RasterizerOpenGL::SetupGeometryShader() {
    MICROPROFILE_SCOPE(OpenGL_GS);

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        LOG_ERROR(Render_OpenGL, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    // No geometry stage: the quaternion short-arc fix runs per fragment against the flat
    // varying the vertex shader emits, so this path works on hardware without geometry shaders.
    curr_shader_manager->UseTrivialGeometryShader();

    return true;
}

bool RasterizerOpenGL::AccelerateShippedDraw(const Pica::DrawPayload& payload) {
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        return false;
    }
    if (!SetupVertexShader()) {
        return false;
    }
    if (!SetupGeometryShader()) {
        return false;
    }
    return Draw(true, payload.is_indexed, &payload);
}

bool RasterizerOpenGL::AccelerateDrawBatch(bool is_indexed) {
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            return false;
        }
    }

    if (!SetupVertexShader()) {
        return false;
    }

    if (!SetupGeometryShader()) {
        return false;
    }

    return Draw(true, is_indexed);
}

bool RasterizerOpenGL::AccelerateDrawBatchInternal(bool is_indexed,
                                                   const Pica::DrawPayload* shipped) {
    const GLenum primitive_mode = MakePrimitiveMode(regs.pipeline.triangle_topology);
    const auto vertex_array_info =
        shipped ? VertexArrayInfo{shipped->vertex_min, shipped->vertex_max,
                                  shipped->layout.index_offset}
                : AnalyzeVertexArray(is_indexed);

    if (vertex_array_info.Invalid()) {
        // Do not draw anything if the vertex array is invalid.
        return true;
    }

    if (vertex_array_info.vs_input_size > VERTEX_BUFFER_SIZE) {
        LOG_WARNING(Render_OpenGL, "Too large vertex input size {}",
                    vertex_array_info.vs_input_size);
        return false;
    }

    state.draw.vertex_buffer = vertex_buffer.GetHandle();
    state.Apply();

    u8* buffer_ptr;
    GLintptr buffer_offset;
    bool ring_upload = false;
    if (shipped && ring.Active()) {
        // The vertex buffer is the persistently mapped ring: allocate from it directly. Padding
        // to the software path's vertex stride keeps the ring's first-index arithmetic exact for
        // the triangle writes that share this buffer.
        const u32 bytes = Common::AlignUp<u32>(
            std::max<u32>(vertex_array_info.vs_input_size, 1), sizeof(Pica::OutputVertex));
        u32 off = ring.TryAlloc(bytes);
        while (off == VideoCore::VertexRing::FULL) {
            RingRetireBlocking();
            off = ring.TryAlloc(bytes);
        }
        buffer_ptr = ring.base + off;
        buffer_offset = off;
        ring_upload = true;
    } else {
        std::tie(buffer_ptr, buffer_offset, std::ignore) =
            vertex_buffer.Map(vertex_array_info.vs_input_size, 4);
    }
    SetupVertexArray(buffer_ptr, buffer_offset, vertex_array_info.vs_input_index_min,
                     vertex_array_info.vs_input_index_max, shipped);
    if (!ring_upload) {
        vertex_buffer.Unmap(vertex_array_info.vs_input_size);
    }

    curr_shader_manager->ApplyTo(state, accurate_mul);
    state.Apply();

    if (is_indexed) {
        bool index_u16 = regs.pipeline.index_array.format != 0;
        std::size_t index_buffer_size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);

        if (index_buffer_size > INDEX_BUFFER_SIZE) {
            LOG_WARNING(Render_OpenGL, "Too large index input size {}", index_buffer_size);
            return false;
        }

        const u8* index_data =
            shipped ? shipped->arena + shipped->layout.index_offset
                    : memory.GetPhysicalPointer(
                          regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() +
                          regs.pipeline.index_array.offset);
        std::tie(buffer_ptr, buffer_offset, std::ignore) = index_buffer.Map(index_buffer_size, 4);
        std::memcpy(buffer_ptr, index_data, index_buffer_size);
        index_buffer.Unmap(index_buffer_size);

        glDrawRangeElementsBaseVertex(primitive_mode, vertex_array_info.vs_input_index_min,
                                      vertex_array_info.vs_input_index_max,
                                      regs.pipeline.num_vertices,
                                      index_u16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE,
                                      reinterpret_cast<const void*>(buffer_offset),
                                      -static_cast<GLint>(vertex_array_info.vs_input_index_min));
    } else {
        glDrawArrays(primitive_mode, 0, regs.pipeline.num_vertices);
    }
    if (ring_upload) {
        ring_unfenced_pos = std::max(ring_unfenced_pos, ring.cursor);
    }
    return true;
}

void RasterizerOpenGL::DrawTriangles() {
    if (!pending_ring_ranges.empty()) {
        // Triangles were written straight into the vertex ring by AddTriangle; draw the ranges.
        for (const auto& range : pending_ring_ranges) {
            DrawRingRange(range.first, range.count, range.end_pos);
        }
        pending_ring_ranges.clear();
        return;
    }
    if (vertex_batch.empty())
        return;
    Draw(false, false);
}

bool RasterizerOpenGL::Draw(bool accelerate, bool is_indexed, const Pica::DrawPayload* shipped) {
    MICROPROFILE_SCOPE(OpenGL_Drawing);
    const DebugScope scope(runtime, Common::Vec4f{}, "RasterizerOpenGL::Draw");

    SyncDrawState();

    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    if (shadow_rendering && GLES && !GLAD_GL_ES_VERSION_3_1) {
        // Shadow rendering needs image load/store, which GLES 3.0-class hardware lacks. Dropping
        // the pass loses shadow maps but keeps everything else rendering.
        LOG_WARNING(Render_OpenGL, "Skipping shadow-rendering pass: needs GLES 3.1 image "
                                   "load/store (once per session)");
        return true;
    }
    const bool has_stencil = regs.framebuffer.HasStencil();

    const bool write_color_fb = shadow_rendering || state.color_mask.red_enabled == GL_TRUE ||
                                state.color_mask.green_enabled == GL_TRUE ||
                                state.color_mask.blue_enabled == GL_TRUE ||
                                state.color_mask.alpha_enabled == GL_TRUE;

    const bool write_depth_fb =
        (state.depth.test_enabled && state.depth.write_mask == GL_TRUE) ||
        (has_stencil && state.stencil.test_enabled && state.stencil.write_mask != 0);

    const bool using_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && state.stencil.test_enabled));

    const auto fb_helper = res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb);
    const Framebuffer* framebuffer = fb_helper.Framebuffer();
    if (!framebuffer->color_id && framebuffer->shadow_rendering) {
        return true;
    }

    // Bind the framebuffer surfaces
    if (shadow_rendering) {
        state.image_shadow_buffer = framebuffer->Attachment(SurfaceType::Color);
    }
    state.draw.draw_framebuffer = framebuffer->Handle();

    // Sync the viewport
    const auto viewport = fb_helper.Viewport();
    state.viewport.x = static_cast<GLint>(viewport.x);
    state.viewport.y = static_cast<GLint>(viewport.y);
    state.viewport.width = static_cast<GLsizei>(viewport.width);
    state.viewport.height = static_cast<GLsizei>(viewport.height);

    // Viewport can have negative offsets or larger dimensions than our framebuffer sub-rect.
    // Enable scissor test to prevent drawing outside of the framebuffer region
    const auto draw_rect = fb_helper.DrawRect();
    state.scissor.enabled = true;
    state.scissor.x = draw_rect.left;
    state.scissor.y = draw_rect.bottom;
    state.scissor.width = draw_rect.GetWidth();
    state.scissor.height = draw_rect.GetHeight();

    // Update scissor uniforms
    const auto [scissor_x1, scissor_y2, scissor_x2, scissor_y1] = fb_helper.Scissor();
    if (fs_data.scissor_x1 != scissor_x1 || fs_data.scissor_x2 != scissor_x2 ||
        fs_data.scissor_y1 != scissor_y1 || fs_data.scissor_y2 != scissor_y2) {

        fs_data.scissor_x1 = scissor_x1;
        fs_data.scissor_x2 = scissor_x2;
        fs_data.scissor_y1 = scissor_y1;
        fs_data.scissor_y2 = scissor_y2;
        fs_data_dirty = true;
    }

    // Sync and bind the texture surfaces
    SyncTextureUnits(framebuffer);
    state.Apply();

    // Sync and bind the shader
    curr_shader_manager->UseFragmentShader(regs, user_config);

    // Sync the LUTs within the texture buffer
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();

    // Sync the uniform data
    UploadUniforms(accelerate);

    // Draw the vertex batch
    bool succeeded = true;
    if (accelerate) {
        succeeded = AccelerateDrawBatchInternal(is_indexed, shipped);
    } else {
        state.draw.vertex_array = sw_vao.handle;
        state.draw.vertex_buffer = vertex_buffer.GetHandle();
        curr_shader_manager->UseTrivialVertexShader();
        curr_shader_manager->UseTrivialGeometryShader();
        curr_shader_manager->ApplyTo(state, accurate_mul);
        state.Apply();

        if (ring_draw_active) {
            // The vertices are already in the vertex buffer, written there by the emulation
            // thread through the persistent mapping. Nothing to upload.
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(ring_draw_first),
                         static_cast<GLsizei>(ring_draw_count));
        } else {
            std::size_t max_vertices = 3 * (VERTEX_BUFFER_SIZE / (3 * sizeof(HardwareVertex)));
            for (std::size_t base_vertex = 0; base_vertex < vertex_batch.size();
                 base_vertex += max_vertices) {
                const std::size_t vertices =
                    std::min(max_vertices, vertex_batch.size() - base_vertex);
                const std::size_t vertex_size = vertices * sizeof(HardwareVertex);

                const auto [vbo, offset, _] =
                    vertex_buffer.Map(vertex_size, sizeof(HardwareVertex));
                std::memcpy(vbo, vertex_batch.data() + base_vertex, vertex_size);
                vertex_buffer.Unmap(vertex_size);

                glDrawArrays(GL_TRIANGLES, static_cast<GLint>(offset / sizeof(HardwareVertex)),
                             static_cast<GLsizei>(vertices));
            }
        }
    }

    vertex_batch.clear();

    if (shadow_rendering) {
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_TEXTURE_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
    }

    return succeeded;
}

void RasterizerOpenGL::SyncTextureUnits(const Framebuffer* framebuffer) {
    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

    // Reset transient draw state
    state.color_buffer.texture_2d = 0;
    user_config = {};

    const auto pica_textures = regs.texturing.GetTextures();
    for (u32 texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        // If the texture unit is disabled unbind the corresponding gl unit
        if (!texture.enabled) {
            switch (texture.config.type.Value()) {
            case TextureType::TextureCube:
            case TextureType::ShadowCube: {
                state.texture_units[texture_index].texture_2d =
                    res_cache.GetSurface(VideoCore::NULL_SURFACE_CUBE_ID).Handle();
                state.texture_units[texture_index].target = GL_TEXTURE_CUBE_MAP;
                break;
            }
            default: {
                state.texture_units[texture_index].texture_2d =
                    res_cache.GetSurface(VideoCore::NULL_SURFACE_ID).Handle();
                state.texture_units[texture_index].target = GL_TEXTURE_2D;
                break;
            }
            }
            continue;
        }

        // Handle special tex0 configurations
        if (texture_index == 0) {
            switch (texture.config.type.Value()) {
            case TextureType::Shadow2D: {
                Surface& surface = res_cache.GetTextureSurface(texture);
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                state.image_shadow_texture_px = surface.Handle();
                continue;
            }
            case TextureType::ShadowCube: {
                BindShadowCube(texture);
                continue;
            }
            case TextureType::TextureCube: {
                BindTextureCube(texture);
                continue;
            }
            default:
                UnbindSpecial();
            }
        }

        // Sync texture unit sampler
        Sampler& sampler = res_cache.GetSampler(texture.config);
        state.texture_units[texture_index].sampler = sampler.Handle();

        // Bind the texture provided by the rasterizer cache
        Surface& surface = res_cache.GetTextureSurface(texture);
        if (!IsFeedbackLoop(texture_index, framebuffer, surface)) {
            BindMaterial(texture_index, surface);
            state.texture_units[texture_index].texture_2d = surface.Handle();
            state.texture_units[texture_index].target = GL_TEXTURE_2D;
        }
    }

    if (emulate_minmax_blend && !driver.HasShaderFramebufferFetch()) {
        state.color_buffer.texture_2d = framebuffer->Attachment(SurfaceType::Color);
    }
}

void RasterizerOpenGL::BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config, texture.format);
    constexpr std::array faces = {
        CubeFace::PositiveX, CubeFace::NegativeX, CubeFace::PositiveY,
        CubeFace::NegativeY, CubeFace::PositiveZ, CubeFace::NegativeZ,
    };

    for (CubeFace face : faces) {
        const u32 binding = static_cast<u32>(face);
        info.physical_address = regs.texturing.GetCubePhysicalAddress(face);

        VideoCore::SurfaceId surface_id = res_cache.GetTextureSurface(info);
        Surface& surface = res_cache.GetSurface(surface_id);
        surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
        state.image_shadow_texture[binding] = surface.Handle();
    }
}

void RasterizerOpenGL::BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    const VideoCore::TextureCubeConfig config = {
        .px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
        .nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
        .py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
        .ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
        .pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
        .nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
        .width = texture.config.width,
        .levels = texture.config.lod.max_level + 1,
        .format = texture.format,
    };

    Surface& surface = res_cache.GetTextureCube(config);
    Sampler& sampler = res_cache.GetSampler(texture.config);
    state.texture_units[0].target = GL_TEXTURE_CUBE_MAP;
    state.texture_units[0].texture_2d = surface.Handle();
    state.texture_units[0].sampler = sampler.Handle();
}

void RasterizerOpenGL::BindMaterial(u32 texture_index, Surface& surface) {
    if (!surface.IsCustom()) {
        return;
    }

    const GLuint sampler = state.texture_units[texture_index].sampler;
    if (surface.HasNormalMap()) {
        if (regs.lighting.disable) {
            LOG_WARNING(Render_OpenGL, "Custom normal map used but scene has no light enabled");
        }
        glActiveTexture(TextureUnits::TextureNormalMap.Enum());
        glBindTexture(GL_TEXTURE_2D, surface.Handle(2));
        glBindSampler(TextureUnits::TextureNormalMap.id, sampler);
        user_config.use_custom_normal.Assign(1);
    }
}

bool RasterizerOpenGL::IsFeedbackLoop(u32 texture_index, const Framebuffer* framebuffer,
                                      Surface& surface) {
    const GLuint color_attachment = framebuffer->Attachment(SurfaceType::Color);
    const bool is_feedback_loop = color_attachment == surface.Handle();
    if (!is_feedback_loop) {
        return false;
    }

    state.texture_units[texture_index].texture_2d = surface.CopyHandle();
    return true;
}

void RasterizerOpenGL::UnbindSpecial() {
    state.texture_units[0].texture_2d = 0;
    state.texture_units[0].target = GL_TEXTURE_2D;
    state.image_shadow_texture_px = 0;
    state.image_shadow_texture_nx = 0;
    state.image_shadow_texture_py = 0;
    state.image_shadow_texture_ny = 0;
    state.image_shadow_texture_pz = 0;
    state.image_shadow_texture_nz = 0;
    state.image_shadow_buffer = 0;
}

void RasterizerOpenGL::FlushAll() {
    res_cache.FlushAll();
}

void RasterizerOpenGL::FlushRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
}

void RasterizerOpenGL::InvalidateRegion(PAddr addr, u32 size) {
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerOpenGL::InvalidateGuestFlushedRegion(PAddr addr, u32 size) {
    res_cache.InvalidateRegionUnlessDirty(addr, size);
}

void RasterizerOpenGL::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerOpenGL::ClearAll(bool flush) {
    res_cache.ClearAll(flush);
}

bool RasterizerOpenGL::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateDisplayTransfer(config);
}

bool RasterizerOpenGL::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateTextureCopy(config);
}

bool RasterizerOpenGL::AccelerateFill(const Pica::MemoryFillConfig& config) {
    return res_cache.AccelerateFill(config);
}

bool RasterizerOpenGL::AccelerateDisplay(const Pica::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    // Presenting from a cached surface is the only source the guest cannot be writing while
    // the render thread reads it - guest memory it can, and does, which is a race that shows
    // up as one screen carrying the other's picture. So this path is the correct one to be on;
    // what it needed was for the lookup below to refuse a surface that merely covers the
    // address without being this framebuffer.
    if (framebuffer_addr == 0) {
        return false;
    }
    MICROPROFILE_SCOPE(OpenGL_Display);

    VideoCore::SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    const auto [src_surface_id, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);
    if (!src_surface_id) {
        return false;
    }

    const DebugScope scope{runtime,
                           Common::Vec4f{0.f, 1.f, 1.f, 1.f},
                           "RasterizerOpenGL::AccelerateDisplay ({}x{} {} at {:#X})",
                           src_params.width,
                           src_params.height,
                           VideoCore::PixelFormatAsString(src_params.pixel_format),
                           src_params.addr};

    const Surface& src_surface = res_cache.GetSurface(src_surface_id);
    // The cache matches by coverage, so a framebuffer that a game has recycled from the other
    // screen finds the surface built while it belonged there - same address, different shape -
    // and the display ends up showing a crop of the wrong screen. Both screens carry the same
    // stride here, so width is what separates them: 400 for the top against 320 for the
    // bottom. Take the surface only when it is this framebuffer rather than something that
    // happens to contain it.
    // A framebuffer is often a sub-rectangle of a taller surface. Mario Kart 7 renders 256x416
    // and shows 240x400 from 0x2000 in, so its visible rows are the last 400 of the surface and
    // the two end at the same address; GetSurfaceSubRect hands back exactly that rectangle,
    // [0,16-240,416], and the texture coordinates below already carry it. Refusing it fell back
    // to reading the framebuffer out of guest memory, which flushes whichever surface owns
    // those bytes: 140 readbacks a second and 27 MB on the pi5 race (2026-09-09). The GXM
    // rasterizer's own AccelerateDisplay had already learned this for NSMB2 and accepts the
    // sub-rectangle by fitting it inside the surface, so this brings the desktop and pi5 tier
    // back to parity with the console rather than fixing anything on the console.
    //
    // What the guard is for stays: a framebuffer a game has recycled from the other screen
    // finds the surface built while it belonged there, same address and a different shape, and
    // the display ends up showing a crop of the wrong screen. So a sub-rectangle is taken only
    // when the two agree on stride and format, the rectangle handed back is exactly this
    // framebuffer's size, and it reaches the far edge of the surface. That last test is what
    // separates the two cases: a framebuffer sitting at the end of a taller surface passes,
    // while a shorter view of a taller surface at the same base stops short of the edge and
    // does not.
    const bool exact_surface =
        src_surface.addr == src_params.addr && src_surface.width == src_params.width;
    const bool tail_of_surface = src_surface.stride == src_params.stride &&
                                 src_rect.top == src_surface.GetScaledHeight() &&
                                 src_rect.GetWidth() == src_params.width * src_surface.res_scale &&
                                 src_rect.GetHeight() == src_params.height * src_surface.res_scale;
    if (src_surface.pixel_format != src_params.pixel_format ||
        (!exact_surface && !tail_of_surface)) {
        static const bool log_display = std::getenv("AZAHAR_BLIT_LOG") != nullptr;
        if (log_display) {
            static u32 logged = 0;
            if (logged < 8) {
                logged++;
                LOG_INFO(HW_GPU,
                         "display {}: want {:#010x} {}x{} stride {} {} -> got {:#010x} {}x{} "
                         "stride {} {} rect [{},{}-{},{}] refused on{}{}{}",
                         logged, src_params.addr, src_params.width, src_params.height,
                         src_params.stride, VideoCore::PixelFormatAsString(src_params.pixel_format),
                         src_surface.addr, src_surface.width, src_surface.height,
                         src_surface.stride,
                         VideoCore::PixelFormatAsString(src_surface.pixel_format), src_rect.left,
                         src_rect.bottom, src_rect.right, src_rect.top,
                         src_surface.stride != src_params.stride ? " stride" : "",
                         src_rect.top != src_surface.GetScaledHeight() ? " edge" : "",
                         src_surface.pixel_format != src_params.pixel_format ? " format" : "");
            }
        }
        return false;
    }
    const u32 scaled_width = src_surface.GetScaledWidth();
    const u32 scaled_height = src_surface.GetScaledHeight();

    screen_info.display_texcoords = Common::Rectangle<float>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    screen_info.display_texture = src_surface.Handle();

    return true;
}

u32 RasterizerOpenGL::UploadLutRow(u32 unit_enum, u32& row_cursor, u32 total_rows,
                                   GLenum format, GLenum type, const void* data,
                                   u32 texel_count) {
    ASSERT(texel_count <= LUT_TEX_WIDTH);
    if (row_cursor >= total_rows) {
        row_cursor = 0;
    }
    glActiveTexture(unit_enum);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, static_cast<GLint>(row_cursor),
                    static_cast<GLsizei>(texel_count), 1, format, type, data);
    return (row_cursor++) * LUT_TEX_WIDTH;
}

void RasterizerOpenGL::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 256 * Pica::LightingRegs::NumLightingSampler +
        sizeof(Common::Vec2f) * 128; // fog

    if (!pica.lighting.lut_dirty && !pica.fog.lut_dirty) {
        return;
    }

    if (lut_textures_2d) {
        // Row allocation is round-robin with no liveness tracking: a wrap hands out rows whose
        // offsets other, non-dirty LUTs still point at, and those then sample whatever was
        // written over them - measured as whole frames of terrain rendered at full fog. Mirror
        // the buffer path's invalidate handling: when this sync would wrap, restart the ring
        // and re-upload every LUT of this texture so no stale offset survives.
        {
            const u32 rows_needed = static_cast<u32>(std::popcount(pica.lighting.lut_dirty)) +
                                    (pica.fog.lut_dirty ? 1u : 0u);
            if (lut_lf_row + rows_needed > lut_lf_rows) {
                lut_lf_row = 0;
                pica.lighting.lut_dirty = pica.lighting.LutAllDirty;
                pica.fog.lut_dirty = true;
            }
        }
        std::array<Common::Vec2f, 256> scratch;
        while (pica.lighting.lut_dirty) {
            const u32 index = std::countr_zero(pica.lighting.lut_dirty);
            pica.lighting.lut_dirty &= ~(1 << index);
            const auto& source_lut = pica.lighting.luts[index];
            for (u32 i = 0; i < source_lut.size(); i++) {
                scratch[i] = {source_lut[i].ToFloat(), source_lut[i].DiffToFloat()};
            }
            fs_data.lighting_lut_offset[index / 4][index % 4] = static_cast<int>(
                UploadLutRow(TextureUnits::TextureBufferLUT_LF.Enum(), lut_lf_row, lut_lf_rows,
                             GL_RG, GL_FLOAT, scratch.data(), source_lut.size()));
            fs_data_dirty = true;
        }
        if (pica.fog.lut_dirty) {
            for (u32 i = 0; i < pica.fog.lut.size(); i++) {
                scratch[i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
            }
            fs_data.fog_lut_offset = static_cast<int>(
                UploadLutRow(TextureUnits::TextureBufferLUT_LF.Enum(), lut_lf_row, lut_lf_rows,
                             GL_RG, GL_FLOAT, scratch.data(), pica.fog.lut.size()));
            fs_data_dirty = true;
            pica.fog.lut_dirty = false;
        }
        return;
    }

    std::size_t bytes_used = 0;
    glBindBuffer(GL_TEXTURE_BUFFER, texture_lf_buffer.GetHandle());
    const auto [buffer, offset, invalidate] =
        texture_lf_buffer.Map(max_size, sizeof(Common::Vec4f));

    if (invalidate) {
        pica.lighting.lut_dirty = pica.lighting.LutAllDirty;
        pica.fog.lut_dirty = true;
    }

    // Sync the lighting luts
    while (pica.lighting.lut_dirty) {
        const u32 index = std::countr_zero(pica.lighting.lut_dirty);
        pica.lighting.lut_dirty &= ~(1 << index);

        Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
        const auto& source_lut = pica.lighting.luts[index];
        for (u32 i = 0; i < source_lut.size(); i++) {
            new_data[i] = {source_lut[i].ToFloat(), source_lut[i].DiffToFloat()};
        }
        fs_data.lighting_lut_offset[index / 4][index % 4] =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
        fs_data_dirty = true;
        bytes_used += source_lut.size() * sizeof(Common::Vec2f);
    }

    // Sync the fog lut
    if (pica.fog.lut_dirty) {
        Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
        for (u32 i = 0; i < pica.fog.lut.size(); i++) {
            new_data[i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
        }
        fs_data.fog_lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
        fs_data_dirty = true;
        bytes_used += pica.fog.lut.size() * sizeof(Common::Vec2f);
        pica.fog.lut_dirty = false;
    }

    texture_lf_buffer.Unmap(bytes_used);
}

void RasterizerOpenGL::SyncAndUploadLUTs() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 128 * 3 + // proctex: noise + color + alpha
        sizeof(Common::Vec4f) * 256 +     // proctex
        sizeof(Common::Vec4f) * 256;      // proctex diff

    if (!pica.proctex.table_dirty) {
        return;
    }

    if (lut_textures_2d) {
        // Same wrap-invalidate as SyncAndUploadLUTsLF: a ring wrap may not strand any
        // non-dirty LUT's offset on a row about to be reused.
        {
            const u32 rg_needed = (pica.proctex.noise_lut_dirty ? 1u : 0u) +
                                  (pica.proctex.color_map_dirty ? 1u : 0u) +
                                  (pica.proctex.alpha_map_dirty ? 1u : 0u);
            const u32 rgba_needed =
                (pica.proctex.lut_dirty ? 1u : 0u) + (pica.proctex.diff_lut_dirty ? 1u : 0u);
            if (lut_rg_row + rg_needed > lut_rg_rows ||
                lut_rgba_row + rgba_needed > lut_rgba_rows) {
                lut_rg_row = 0;
                lut_rgba_row = 0;
                pica.proctex.table_dirty = pica.proctex.TableAllDirty;
            }
        }
        std::array<Common::Vec2f, 256> scratch2;
        std::array<Common::Vec4f, 256> scratch4;
        const auto sync_value_lut_2d = [&](const auto& lut, GLint& lut_offset) {
            for (u32 i = 0; i < lut.size(); i++) {
                scratch2[i] = {lut[i].ToFloat(), lut[i].DiffToFloat()};
            }
            lut_offset = static_cast<int>(
                UploadLutRow(TextureUnits::TextureBufferLUT_RG.Enum(), lut_rg_row, lut_rg_rows,
                             GL_RG, GL_FLOAT, scratch2.data(), lut.size()));
            fs_data_dirty = true;
        };
        if (pica.proctex.noise_lut_dirty) {
            sync_value_lut_2d(pica.proctex.noise_table, fs_data.proctex_noise_lut_offset);
        }
        if (pica.proctex.color_map_dirty) {
            sync_value_lut_2d(pica.proctex.color_map_table, fs_data.proctex_color_map_offset);
        }
        if (pica.proctex.alpha_map_dirty) {
            sync_value_lut_2d(pica.proctex.alpha_map_table, fs_data.proctex_alpha_map_offset);
        }
        if (pica.proctex.lut_dirty) {
            for (u32 i = 0; i < pica.proctex.color_table.size(); i++) {
                scratch4[i] = pica.proctex.color_table[i].ToVector() / 255.0f;
            }
            fs_data.proctex_lut_offset = static_cast<int>(UploadLutRow(
                TextureUnits::TextureBufferLUT_RGBA.Enum(), lut_rgba_row, lut_rgba_rows, GL_RGBA,
                GL_FLOAT, scratch4.data(), pica.proctex.color_table.size()));
            fs_data_dirty = true;
        }
        if (pica.proctex.diff_lut_dirty) {
            for (u32 i = 0; i < pica.proctex.color_diff_table.size(); i++) {
                scratch4[i] = pica.proctex.color_diff_table[i].ToVector() / 255.0f;
            }
            fs_data.proctex_diff_lut_offset = static_cast<int>(UploadLutRow(
                TextureUnits::TextureBufferLUT_RGBA.Enum(), lut_rgba_row, lut_rgba_rows, GL_RGBA,
                GL_FLOAT, scratch4.data(), pica.proctex.color_diff_table.size()));
            fs_data_dirty = true;
        }
        pica.proctex.table_dirty = 0;
        return;
    }

    std::size_t bytes_used = 0;
    glBindBuffer(GL_TEXTURE_BUFFER, texture_buffer.GetHandle());
    const auto [buffer, offset, invalidate] = texture_buffer.Map(max_size, sizeof(Common::Vec4f));

    if (invalidate) {
        pica.proctex.table_dirty = pica.proctex.TableAllDirty;
    }

    // helper function for SyncProcTexNoiseLUT/ColorMap/AlphaMap
    const auto sync_proc_tex_value_lut = [&](const auto& lut, GLint& lut_offset) {
        Common::Vec2f* new_data = reinterpret_cast<Common::Vec2f*>(buffer + bytes_used);
        for (u32 i = 0; i < lut.size(); i++) {
            new_data[i] = {lut[i].ToFloat(), lut[i].DiffToFloat()};
        }
        lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
        fs_data_dirty = true;
        bytes_used += lut.size() * sizeof(Common::Vec2f);
    };

    // Sync the proctex noise lut
    if (pica.proctex.noise_lut_dirty) {
        sync_proc_tex_value_lut(pica.proctex.noise_table, fs_data.proctex_noise_lut_offset);
    }

    // Sync the proctex color map
    if (pica.proctex.color_map_dirty) {
        sync_proc_tex_value_lut(pica.proctex.color_map_table, fs_data.proctex_color_map_offset);
    }

    // Sync the proctex alpha map
    if (pica.proctex.alpha_map_dirty) {
        sync_proc_tex_value_lut(pica.proctex.alpha_map_table, fs_data.proctex_alpha_map_offset);
    }

    // Sync the proctex lut
    if (pica.proctex.lut_dirty) {
        Common::Vec4f* new_data = reinterpret_cast<Common::Vec4f*>(buffer + bytes_used);
        for (u32 i = 0; i < pica.proctex.color_table.size(); i++) {
            new_data[i] = pica.proctex.color_table[i].ToVector() / 255.0f;
        }
        fs_data.proctex_lut_offset =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
        fs_data_dirty = true;
        bytes_used += pica.proctex.color_table.size() * sizeof(Common::Vec4f);
    }

    // Sync the proctex difference lut
    if (pica.proctex.diff_lut_dirty) {
        Common::Vec4f* new_data = reinterpret_cast<Common::Vec4f*>(buffer + bytes_used);
        for (u32 i = 0; i < pica.proctex.color_diff_table.size(); i++) {
            new_data[i] = pica.proctex.color_diff_table[i].ToVector() / 255.0f;
        }
        fs_data.proctex_diff_lut_offset =
            static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
        fs_data_dirty = true;
        bytes_used += pica.proctex.color_diff_table.size() * sizeof(Common::Vec4f);
    }

    pica.proctex.table_dirty = 0;

    texture_buffer.Unmap(bytes_used);
}

void RasterizerOpenGL::UploadUniforms(bool accelerate_draw) {
    // glBindBufferRange also changes the generic buffer binding point, so we sync the state first.
    state.draw.uniform_buffer = uniform_buffer.GetHandle();
    state.Apply();

    const bool sync_vs_pica = accelerate_draw && pica.vs_setup.uniforms_dirty;
    if (!sync_vs_pica && !vs_data_dirty && !fs_data_dirty) {
        return;
    }

    std::size_t uniform_size =
        uniform_size_aligned_vs_pica + uniform_size_aligned_vs + uniform_size_aligned_fs;
    std::size_t used_bytes = 0;

    const auto [uniforms, offset, invalidate] =
        uniform_buffer.Map(uniform_size, uniform_buffer_alignment);

    if (vs_data_dirty || invalidate) {
        std::memcpy(uniforms + used_bytes, &vs_data, sizeof(vs_data));
        glBindBufferRange(GL_UNIFORM_BUFFER, UniformBindings::VSData, uniform_buffer.GetHandle(),
                          offset + used_bytes, sizeof(vs_data));
        vs_data_dirty = false;
        used_bytes += uniform_size_aligned_vs;
    }

    if (fs_data_dirty || invalidate) {
        std::memcpy(uniforms + used_bytes, &fs_data, sizeof(fs_data));
        glBindBufferRange(GL_UNIFORM_BUFFER, UniformBindings::FSData, uniform_buffer.GetHandle(),
                          offset + used_bytes, sizeof(fs_data));
        fs_data_dirty = false;
        used_bytes += uniform_size_aligned_fs;
    }

    if (sync_vs_pica || invalidate) {
        VSPicaUniformData vs_uniforms;
        vs_uniforms.SetFromRegs(pica.vs_setup);
        std::memcpy(uniforms + used_bytes, &vs_uniforms, sizeof(vs_uniforms));
        glBindBufferRange(GL_UNIFORM_BUFFER, UniformBindings::VSPicaData,
                          uniform_buffer.GetHandle(), offset + used_bytes, sizeof(vs_uniforms));
        pica.vs_setup.uniforms_dirty = false;
        used_bytes += uniform_size_aligned_vs_pica;
    }

    uniform_buffer.Unmap(used_bytes);
}

} // namespace OpenGL
