add_executable(citra_sdl
    citra_sdl_main.cpp
    emu_window_sdl2.cpp
    emu_window_sdl2.h
    input_debug_port.cpp
    input_debug_port.h
)

target_link_libraries(citra_sdl PRIVATE
    citra_common
    citra_core
    input_common
    video_core
    SDL2::SDL2
)

if (ENABLE_OPENGL)
    target_link_libraries(citra_sdl PRIVATE glad)
endif()

# Compares the shader JIT against the interpreter; run on the target after compiler changes.
add_executable(shader_jit_selftest shader_jit_selftest.cpp)
target_link_libraries(shader_jit_selftest PRIVATE citra_common citra_core video_core nihstro-headers)

# Guest code executes at its own addresses, which on the 3DS run from 0x00100000 to 0x40000000.
# On a 32-bit host that is exactly where the kernel likes to put things: a position independent
# executable lands around 0x00900000 with its heap just above, both squarely inside the guest's
# address space, and then the guest has nowhere to live. Link the emulator itself above the whole
# guest range instead, leaving everything below 0x40000000 free to be claimed at startup.
if (CITRA_HAS_NATIVE_ARM)
    target_link_options(citra_sdl PRIVATE -no-pie -Wl,-Ttext-segment=0x60000000)
endif()

if (UNIX AND NOT APPLE)
    install(TARGETS citra_sdl RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
