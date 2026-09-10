# Turns the vendored .gxp programs into one generated header of byte arrays. The Cg sources
# next to them are the human-readable original; compile.sh regenerates the .gxp from them.
# Invoked as
#   cmake -DSHADER_DIR=... -DOUT=... -P EmbedShaders.cmake
file(GLOB programs "${SHADER_DIR}/*.gxp")
set(body "// Generated from src/citra_vita/shaders/*.gxp - do not edit.\n#pragma once\n\n")
string(APPEND body "#include <psp2/gxm.h>\n\n")
foreach(gxp IN LISTS programs)
    get_filename_component(name "${gxp}" NAME_WE)
    file(READ "${gxp}" hex HEX)
    # 32 hex digits is 16 bytes: one source line each.
    string(REGEX REPLACE "(................................)" "\\1\n" hex "${hex}")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(REPLACE ",\n" ",\n    " bytes "${bytes}")
    string(APPEND body
        "alignas(16) inline constexpr unsigned char kShaderData_${name}[] = {\n    ${bytes}\n};\n"
        "inline const SceGxmProgram* GxmShader_${name}() {\n"
        "    return reinterpret_cast<const SceGxmProgram*>(kShaderData_${name});\n}\n\n")
endforeach()
file(WRITE "${OUT}" "${body}")
