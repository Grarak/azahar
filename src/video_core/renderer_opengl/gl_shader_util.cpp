// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <array>
#include <string>
#include <vector>
#include <glad/glad.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_opengl/gl_shader_util.h"
#include "video_core/renderer_opengl/gl_vars.h"

namespace OpenGL {

GLuint LoadShader(std::string_view source, GLenum type, const std::string& debug_name) {
    std::string preamble;
    if (GLES) {
        // 3.1-class devices compile everything at 310 es; the shaders in this tree stay within
        // that plus the extensions the context ladder validated.
        preamble = GLAD_GL_ES_VERSION_3_2   ? "#version 320 es\n"
                   : GLAD_GL_ES_VERSION_3_1 ? "#version 310 es\n"
                                            : "#version 300 es\n";
        if (!GLAD_GL_ES_VERSION_3_2) {
            // 310 es gaps: interface blocks (gl_PerVertex redeclaration under separable
            // shaders) come from the io-blocks extension, and fma() does not exist yet.
            preamble += R"(
#if defined(GL_OES_shader_io_blocks)
#extension GL_OES_shader_io_blocks : enable
#elif defined(GL_EXT_shader_io_blocks)
#extension GL_EXT_shader_io_blocks : enable
#endif
#define fma(a, b, c) ((a) * (b) + (c))
)";
        }
        preamble += R"(
#if defined(GL_ANDROID_extension_pack_es31a)
#extension GL_ANDROID_extension_pack_es31a : enable
#endif // defined(GL_ANDROID_extension_pack_es31a)

#if defined(GL_EXT_clip_cull_distance)
#extension GL_EXT_clip_cull_distance : enable
#endif // defined(GL_EXT_clip_cull_distance)

#if __VERSION__ >= 310
#define UNIFORM_BINDING(b) layout(binding = b, std140)
#define SAMPLER_BINDING(b) layout(binding = b)
#define VARYING_LOCATION(l) layout(location = l)
#define EXPLICIT_UNIFORM_LOCATION(l) layout(location = l)
#else
#define UNIFORM_BINDING(b) layout(std140)
#define SAMPLER_BINDING(b)
#define VARYING_LOCATION(l)
#define EXPLICIT_UNIFORM_LOCATION(l)
#endif
)";
    } else {
        preamble = "#version 430 core\n"
                   "#define UNIFORM_BINDING(b) layout(binding = b, std140)\n"
                   "#define SAMPLER_BINDING(b) layout(binding = b)\n"
                   "#define VARYING_LOCATION(l) layout(location = l)\n"
                   "#define EXPLICIT_UNIFORM_LOCATION(l) layout(location = l)\n";
    }

    std::string_view debug_type;
    switch (type) {
    case GL_VERTEX_SHADER:
        debug_type = "vertex";
        break;
    case GL_GEOMETRY_SHADER:
        debug_type = "geometry";
        break;
    case GL_FRAGMENT_SHADER:
        debug_type = "fragment";
        break;
    default:
        UNREACHABLE();
    }

    std::array<const GLchar*, 2> src_arr{preamble.data(), source.data()};
    std::array<GLint, 2> lengths{static_cast<GLint>(preamble.size()),
                                 static_cast<GLint>(source.size())};
    GLuint shader_id = glCreateShader(type);
    if (shader_id == 0) {
        GLenum err = glGetError();
        LOG_ERROR(Render_OpenGL, "glCreateShader failed for {} shader {} (err {})", debug_type,
                  debug_name, err);
        return shader_id;
    }

    glShaderSource(shader_id, static_cast<GLsizei>(src_arr.size()), src_arr.data(), lengths.data());
    LOG_DEBUG(Render_OpenGL, "Compiling {} shader {}...", debug_type, debug_name);
    // The time here is the driver's front end only. Mesa's V3D backend defers the real
    // translation to the first draw that uses the program, so a compile logged as fast can
    // still stall the render thread later, inside glDraw*.
    const auto compile_start = std::chrono::steady_clock::now();
    glCompileShader(shader_id);
    {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - compile_start)
                            .count();
        LOG_INFO(Render_OpenGL, "GL compiled {} shader {} in {}.{:03} ms", debug_type, debug_name,
                 us / 1000, us % 1000);
    }

    GLint result = GL_FALSE;
    GLint info_log_length;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);

    if (info_log_length > 1) {
        std::vector<char> shader_error(info_log_length);
        glGetShaderInfoLog(shader_id, info_log_length, nullptr, &shader_error[0]);
        if (result == GL_TRUE) {
            LOG_DEBUG(Render_OpenGL, "Compile message for {} shader {}:\n{}", debug_type,
                      debug_name, &shader_error[0]);
        } else {
            LOG_ERROR(Render_OpenGL, "Error compiling {} shader {}:\n{}", debug_type, debug_name,
                      &shader_error[0]);
            LOG_ERROR(Render_OpenGL, "Shader source code:\n{}{}", src_arr[0], src_arr[1]);
        }
    } else if (result == GL_FALSE) {
        LOG_ERROR(Render_OpenGL, "Error compiling {} shader {}:\nNo log produced.", debug_type,
                  debug_name);
    }
    return shader_id;
}

/// GLSL 300 es has no layout(binding = N): assign the well-known sampler units and uniform
/// block bindings after linking. Harmless no-ops for names a program does not use.
static void AssignLegacyBindings(GLuint program) {
    glUseProgram(program);
    static constexpr std::pair<const char*, GLint> samplers[] = {
        {"tex0", 0},
        {"tex1", 1},
        {"tex2", 2},
        {"texture_buffer_lut_lf", 3},
        {"texture_buffer_lut_rg", 4},
        {"texture_buffer_lut_rgba", 5},
        {"tex_normal", 6},
        {"tex_color", 7},
        {"color_texture", 0},
        {"color_texture_r", 1},
        {"input_texture", 0},
        {"HOOKED", 0},
    };
    for (const auto& [name, unit] : samplers) {
        const GLint loc = glGetUniformLocation(program, name);
        if (loc >= 0) {
            glUniform1i(loc, unit);
        }
    }
    static constexpr std::pair<const char*, GLuint> blocks[] = {
        {"vs_pica_data", 0},
        {"vs_data", 1},
        {"fs_data", 2},
    };
    for (const auto& [name, binding] : blocks) {
        const GLuint index = glGetUniformBlockIndex(program, name);
        if (index != GL_INVALID_INDEX) {
            glUniformBlockBinding(program, index, binding);
        }
    }
}

GLuint LoadProgram(bool separable_program, std::span<const GLuint> shaders,
                   const std::string& debug_name) {
    // Link the program
    LOG_DEBUG(Render_OpenGL, "Linking program...");

    GLuint program_id = glCreateProgram();

    for (GLuint shader : shaders) {
        if (shader != 0) {
            glAttachShader(program_id, shader);
        }
    }

    if (separable_program) {
        glProgramParameteri(program_id, GL_PROGRAM_SEPARABLE, GL_TRUE);
    }

    glProgramParameteri(program_id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    const auto link_start = std::chrono::steady_clock::now();
    glLinkProgram(program_id);
    {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - link_start)
                            .count();
        LOG_INFO(Render_OpenGL, "GL linked program {} in {}.{:03} ms", program_id, us / 1000,
                 us % 1000);
    }

    // Check the program
    GLint result = GL_FALSE;
    GLint info_log_length;
    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);

    if (info_log_length > 1) {
        std::vector<char> program_error(info_log_length);
        glGetProgramInfoLog(program_id, info_log_length, nullptr, &program_error[0]);
        if (result == GL_TRUE) {
            LOG_DEBUG(Render_OpenGL, "Link message for shader {}:\n{}", debug_name,
                      &program_error[0]);
        } else {
            LOG_ERROR(Render_OpenGL, "Error linking shader {}:\n{}", debug_name, &program_error[0]);
        }
    } else if (result == GL_FALSE) {
        LOG_ERROR(Render_OpenGL, "Error linking shader {}: No log produced.", debug_name);
    }

    for (GLuint shader : shaders) {
        if (shader != 0) {
            glDetachShader(program_id, shader);
        }
    }

    if (GLES && !GLAD_GL_ES_VERSION_3_1) {
        AssignLegacyBindings(program_id);
    }
    return program_id;
}

} // namespace OpenGL
