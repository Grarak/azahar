# vita_create_self / vita_create_vpk live here, not in the toolchain file.
include("${VITASDK}/share/vita.cmake" REQUIRED)

# The presentation shaders, embedded as byte arrays. The .gxp are vendored - compiled offline
# by the SDK's psp2cgc (shaders/compile.sh) from the .cg next to them - so the console needs no
# runtime Cg compiler to present, and a machine without docker still builds the vpk.
set(GXM_SHADER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/gxm_shaders.gen.h")
file(GLOB GXM_SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.gxp")
add_custom_command(
    OUTPUT "${GXM_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND}
        -DSHADER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders
        -DOUT=${GXM_SHADER_HEADER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/EmbedShaders.cmake
    DEPENDS ${GXM_SHADER_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/EmbedShaders.cmake"
    COMMENT "Embedding GXM shader programs")

add_executable(citra_vita
    ../vita_compat/src/vita_libc_shims.c
    citra_vita_main.cpp
    emu_window_vita.cpp
    emu_window_vita.h
    gxm_present.cpp
    gxm_present.h
    imgui_impl_gxm.cpp
    imgui_impl_gxm.h
    input_factory_vita.cpp
    input_factory_vita.h
    vita_ui.cpp
    vita_ui.h
    "${GXM_SHADER_HEADER}"
)
target_include_directories(citra_vita PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# The shims must survive LTO. clang only emits the call to __aeabi_h2f during code generation,
# which happens after LTO has decided what to keep - so with the shim in bitcode, LLVM sees a
# definition nothing calls, drops it, and the link then fails on a symbol the compiler itself
# introduced. Compiling this file to a real object keeps the definitions where the final link
# can always see them.
set_source_files_properties(../vita_compat/src/vita_libc_shims.c PROPERTIES
    COMPILE_OPTIONS "-fno-lto")

create_target_directory_groups(citra_vita)

target_link_libraries(citra_vita PRIVATE citra_common citra_core input_common video_core)

# Per-core PMU stats line, printed once a second, and the SGI window counters. Pulls those
# calls into the import table, so the vpk then only loads against a kernel module new enough to
# export them; OFF builds a vpk that runs against any module. It is a measurement, so it
# follows the diagnostic build.
option(ENABLE_VITA_PMU "Per-second per-core PMU counter line" ${ENABLE_VITA_DIAGNOSTICS})
if (ENABLE_VITA_PMU)
    target_compile_definitions(citra_vita PRIVATE CITRA_VITA_PMU)
endif()

# Render-thread phase breadcrumb: -DVITA_RENDER_TRACE=ON makes the per-second line carry the
# phase the render thread is in and how long it has been there, so a hang that prints nothing
# from the thread that hung can still be located. See common/pipeline_stats.h.
if (ENABLE_VITA_RENDER_TRACE)
    target_compile_definitions(citra_vita PRIVATE VITA_RENDER_TRACE)
endif()

option(VITA_ALLOC_TRACE "Log large heap allocations on the Vita build" OFF)
# Allocation tracer: -DVITA_ALLOC_TRACE=ON logs every allocation >= 1 MiB (with a running live
# total and the caller's return address) to the kernel printf channel, to find what eats the
# heap. Off by default: the file is not compiled and no --wrap flags are added.
if (VITA_ALLOC_TRACE)
    set(VITA_ALLOC_TRACE_THRESHOLD_KIB "64" CACHE STRING
        "Smallest heap allocation the Vita tracer records, in KiB (memblocks and threads are always recorded)")
    set(VITA_ALLOC_TRACE_RECORDS "4096" CACHE STRING
        "How many allocation records the Vita tracer keeps before wrapping")
    target_sources(citra_vita PRIVATE alloc_trace.c pte_sema.c)
    target_compile_definitions(citra_vita PRIVATE VITA_ALLOC_TRACE
        ALLOC_TRACE_THRESHOLD_KIB=${VITA_ALLOC_TRACE_THRESHOLD_KIB}
        ALLOC_TRACE_RECORDS=${VITA_ALLOC_TRACE_RECORDS})
    set_source_files_properties(pte_sema.c PROPERTIES COMPILE_OPTIONS "-fno-lto")
    # The kernel's own allocator is the one that runs out - a memblock refusal is what
    # "Cannot allocate physical page ... [ScePhyMemPartGame]" is - so those are wrapped too.
    # The kernel semaphores every pthread primitive holds are the other bounded resource, and
    # the one that fails silently: see pte_sema.c.
    target_link_options(citra_vita PRIVATE
        -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc,--wrap=memalign,--wrap=free
        -Wl,--wrap=sceKernelAllocMemBlock,--wrap=sceKernelFreeMemBlock,--wrap=sceKernelCreateThread
        -Wl,--wrap=pte_osSemaphoreCreate,--wrap=pte_osSemaphoreDelete)

    # imgui's core objects come from the VITASDK's imgui_vita2d archive (its 1.79 imgui build);
    # the vita2d backend object inside it is never referenced, so vita2d itself is not linked.
endif()

target_link_libraries(citra_vita PRIVATE
    imgui_vita2d
    taihen_stub
    SceDisplay_stub
    SceGxm_stub
    SceCommonDialog_stub
    SceCtrl_stub
    SceTouch_stub
    SceAudio_stub
    ScePower_stub
    SceShellSvc_stub
    SceAppMgr_stub
    SceAppUtil_stub
    SceSysmodule_stub
    SceKernelDmacMgr_stub
    SceRazorCapture_stub_weak
    SceRazorHud_stub_weak
    SceKernelModulemgr_stub
    SceVshBridge_stub
    m
)

target_include_directories(citra_vita PRIVATE ${VITASDK}/arm-vita-eabi/include)


# vita_create_self looks for the executable next to the target rather than in the project's
# shared output directory, so this target keeps its own.
set_target_properties(citra_vita PROPERTIES
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ldscript.ld
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ATTRIBUTE2=12 is the extended-memory mode: the app's LPDDR budget grows by 109 MB, which
# is what lets the newlib heap below hold the guest's 128 MB FCRAM and still leave the
# emulator real working room. Without it the ceiling was 210 MB with ~7 MB spare in-game,
# and one large FS read's std::bad_alloc froze the guest (see hle_ipc.h's async sections).
set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d ATTRIBUTE2=12")

set(VITA_TITLEID "AZAHARPSV")
set(VITA_APP_NAME "Azahar")
set(VITA_VERSION "00.10")

# The SceShell authid grants the extra permissions the native backend needs (taking a core, the
# kernel-module bridge), and -na disables ASLR so the guest range lands where the module expects.
# -a goes through VITA_MAKE_FSELF_FLAGS (the macro has no keyword for it); NOASLR is -na.
set(VITA_MAKE_FSELF_FLAGS "-a 0x2800000000000001")
vita_create_self(citra_vita.self citra_vita UNSAFE NOASLR)

# A second eboot for profiling. vita-make-fself keeps only the loadable segments, so the one
# the console runs carries no section headers and no symbols and a host profiler can only show
# addresses. tools/make_debug_self.py copies that finished SELF and appends the sections from
# the velf beside it, so the same build exists twice: the small one to ship, and a large one
# whose addresses have names. Built on request - "ninja -C build-vita citra_vita_tracer".
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tracer.self
  COMMAND ${CMAKE_COMMAND} -E env python3
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_debug_self.py
          ${CMAKE_CURRENT_BINARY_DIR}/citra_vita.self
          ${CMAKE_CURRENT_BINARY_DIR}/citra_vita.velf
          ${CMAKE_CURRENT_BINARY_DIR}/tracer.self
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/citra_vita.self
          ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_debug_self.py
  COMMENT "Building tracer.self (symbols kept)")
add_custom_target(citra_vita_tracer DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tracer.self)
vita_create_vpk(citra_vita.vpk ${VITA_TITLEID} citra_vita.self
    VERSION ${VITA_VERSION}
    NAME ${VITA_APP_NAME}
    # Razor HUD reads its settings from app0:; LIVE=1 is what lets GPU Live Metrics start.
    FILE ${CMAKE_CURRENT_SOURCE_DIR}/hud_settings.ini hud_settings.ini
        ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/icon0.png sce_sys/icon0.png
        ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/livearea/contents/bg.png sce_sys/livearea/contents/bg.png
        ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/livearea/contents/startup.png sce_sys/livearea/contents/startup.png
        ${CMAKE_CURRENT_SOURCE_DIR}/sce_sys/livearea/contents/template.xml sce_sys/livearea/contents/template.xml
)
