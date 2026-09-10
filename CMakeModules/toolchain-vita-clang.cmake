# Building for the PS Vita with clang and lld instead of the vitasdk's gcc.
#
# The vitasdk still supplies everything but the compiler: the sysroot, newlib, libstdc++ and its
# headers, the crt objects, libgcc, the SCE stub libraries, and the packaging tools. DSVita
# builds for this console the same way, which is worth something on a target this unusual.
#
# This deliberately does not include the vitasdk's own toolchain file. That file begins with
#
#     if( DEFINED CMAKE_CROSSCOMPILING )
#       return()
#     endif()
#
# and on a reconfigure that return propagates out through the include, so everything after it
# here would be skipped: the first configure would be correct and every one after it would
# quietly drop half the flags. What is needed from that file is short enough to state directly.

if (NOT DEFINED VITASDK)
    if (NOT DEFINED ENV{VITASDK})
        message(FATAL_ERROR "VITASDK is not set")
    endif()
    set(VITASDK "$ENV{VITASDK}")
endif()
set(VITASDK "${VITASDK}" CACHE PATH "Path to the Vita SDK")

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "armv7-a")
set(VITA True)

set(VITA_SYSROOT "${VITASDK}/arm-vita-eabi")

# The packaging tools, which src/citra_vita/CMakeLists.txt reaches through vita.cmake.
set(VITA_ELF_CREATE "${VITASDK}/bin/vita-elf-create" CACHE PATH "vita-elf-create")
set(VITA_ELF_EXPORT "${VITASDK}/bin/vita-elf-export" CACHE PATH "vita-elf-export")
set(VITA_LIBS_GEN "${VITASDK}/bin/vita-libs-gen" CACHE PATH "vita-libs-gen")
set(VITA_MAKE_FSELF "${VITASDK}/bin/vita-make-fself" CACHE PATH "vita-make-fself")
set(VITA_MKSFOEX "${VITASDK}/bin/vita-mksfoex" CACHE PATH "vita-mksfoex")
set(VITA_PACK_VPK "${VITASDK}/bin/vita-pack-vpk" CACHE PATH "vita-pack-vpk")

set(CMAKE_FIND_ROOT_PATH "${VITASDK}/bin" "${VITA_SYSROOT}")
set(CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_FIND_ROOT_PATH})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(VITA_CLANG_SUFFIX "-21" CACHE STRING "Version suffix of the clang to build with")

set(CMAKE_C_COMPILER "clang${VITA_CLANG_SUFFIX}" CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER "clang++${VITA_CLANG_SUFFIX}" CACHE PATH "C++ compiler")
set(CMAKE_ASM_COMPILER "clang${VITA_CLANG_SUFFIX}" CACHE PATH "assembler")
set(CMAKE_AR "llvm-ar${VITA_CLANG_SUFFIX}" CACHE PATH "archive")
set(CMAKE_RANLIB "llvm-ranlib${VITA_CLANG_SUFFIX}" CACHE PATH "ranlib")

# libstdc++ and the crt objects come from the vitasdk's gcc.
file(GLOB VITA_GCC_LIB_DIRS "${VITASDK}/lib/gcc/arm-vita-eabi/*")
list(GET VITA_GCC_LIB_DIRS 0 VITA_GCC_LIB_DIR)
file(GLOB VITA_CXX_INCLUDE_DIRS "${VITA_SYSROOT}/include/c++/*")
list(GET VITA_CXX_INCLUDE_DIRS 0 VITA_CXX_INCLUDE_DIR)

# The target, the sysroot and the C++ header search path go through the variables CMake has for
# them rather than into CMAKE_<LANG>_FLAGS. With gcc the target was implied by which binary was
# invoked; with clang it is a flag, and a vendored project that assigns to CMAKE_CXX_FLAGS
# instead of appending would silently drop it and compile for the host - which is exactly what
# soundtouch does. These survive that, because CMake emits them itself on every compile.
set(CMAKE_C_COMPILER_TARGET "armv7a-none-eabihf")
set(CMAKE_CXX_COMPILER_TARGET "armv7a-none-eabihf")
set(CMAKE_ASM_COMPILER_TARGET "armv7a-none-eabihf")
set(CMAKE_SYSROOT "${VITA_SYSROOT}")
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${VITA_CXX_INCLUDE_DIR}" "${VITA_CXX_INCLUDE_DIR}/arm-vita-eabi")

# -Wl,-q keeps the relocations, without which vita-elf-create cannot convert the result.
#
# -femulated-tls is what the vitasdk's own gcc does: thread_local goes through
# __emutls_get_address, which libgcc provides. Left to itself clang either reads TPIDRURO
# directly, which assumes the kernel maintains it per thread, or calls __aeabi_read_tp, which
# nothing on this platform defines. Matching the SDK's compiler is the answer known to work.
#
# `-read-tp-tpidruro` has a leading minus because it names a target feature being turned off; it
# stops the direct register read on any path emulated TLS does not cover.
#
# The architecture options are repeated in the top-level CMakeLists through
# add_compile_options, which is a directory property and therefore survives a subproject that
# assigns to CMAKE_CXX_FLAGS. That is not redundancy for its own sake: the software renderer's
# NEON came out compiled without NEON before it was added.
#
# -fno-pic is required, not preference. clang defaults to position-independent code and reaches
# globals through the GOT, emitting R_ARM_GOT_PREL (relocation type 96); vita-elf-create rejects
# it outright - "Invalid relocation type 96" - because a SELF has no GOT to resolve against. The
# vitasdk's gcc defaults to non-PIC, which is why this never came up with it.
set(VITA_CLANG_FLAGS
    "-Wl,-q -D__vita__=1 -D__VITA__=1 -mcpu=cortex-a9 -mfpu=neon -mthumb -femulated-tls -fno-pic -fno-pie -Xclang -target-feature -Xclang -read-tp-tpidruro")

set(CMAKE_C_FLAGS "${VITA_CLANG_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${VITA_CLANG_FLAGS}" CACHE STRING "c++ flags")
set(CMAKE_ASM_FLAGS "${VITA_CLANG_FLAGS}" CACHE STRING "asm flags")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "c Release flags")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "c++ Release flags")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g" CACHE STRING "c Debug flags")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "c++ Debug flags")

# The link recipe. Everything the gcc driver would have supplied has to be named, because
# -nostdlib is what keeps lld from reaching for a host libc.
#
# --no-rosegment matters: lld would otherwise put read-only data in a PT_LOAD of its own, and
# vita-elf-create expects the two-segment layout the console's loader knows about.
set(CMAKE_EXE_LINKER_FLAGS
    "-Wl,-z,nocopyreloc -no-pie -fuse-ld=lld${VITA_CLANG_SUFFIX} \
-nostdlib -nostdlib++ -Wl,-z,norelro -Wl,-z,max-page-size=4096 -Wl,--no-eh-frame-hdr -Wl,--no-rosegment \
${VITA_SYSROOT}/lib/crt0.o -L${VITA_GCC_LIB_DIR} \
${VITA_GCC_LIB_DIR}/crti.o ${VITA_GCC_LIB_DIR}/crtbegin.o ${VITA_GCC_LIB_DIR}/crtend.o ${VITA_GCC_LIB_DIR}/crtn.o \
-T ${CMAKE_CURRENT_LIST_DIR}/../src/citra_vita/ldscript.ld"
    CACHE STRING "executable linker flags")

# The libraries every executable needs, in CMAKE_<LANG>_STANDARD_LIBRARIES because CMake puts
# that at the very end of the link line. A static archive only satisfies references seen before
# it, so these have to come after the project's own libraries - naming them through
# target_link_libraries put them in the middle, where libpthread could not answer zstd.
#
# The C library's own syscall layer calls into SceNet, which is why a program that does nothing
# but write to stdout still needs the network stubs. --start-group covers the circular
# references among the C library, libgcc and the stubs.
# libpthread goes in whole: libstdc++ reaches pthread through *weak* references, and a weak
# undefined reference does not extract an archive member. `__gthread_active_p()` is literally
# `&pthread_cancel != 0` (gthr-posix.h), and pthread_cancel.o is a member nothing else pulls, so
# without this every std::thread throws "Enable multithreading to use std::thread". The same
# applies to pthread_rwlock_* (std::shared_mutex), which would otherwise be null calls once
# threading is enabled. The gcc driver gets away with it by other means; -nostdlib does not.
set(VITA_SYSTEM_LIBRARIES
    "-Wl,--start-group -lstdc++ -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -lc -lm -lgcc \
-lSceLibKernel_stub -lSceKernelThreadMgr_stub -lSceIofilemgr_stub -lSceProcessmgr_stub \
-lSceSysmem_stub -lSceRtc_stub -lSceKernelModulemgr_stub -lSceNet_stub -lSceNetCtl_stub \
-lSceSysmodule_stub -Wl,--end-group")

set(CMAKE_C_STANDARD_LIBRARIES "${VITA_SYSTEM_LIBRARIES}" CACHE STRING "c link libraries")
set(CMAKE_CXX_STANDARD_LIBRARIES "${VITA_SYSTEM_LIBRARIES}" CACHE STRING "c++ link libraries")

# A test executable would need the whole recipe above plus the SCE stubs to link; compiling to a
# static library proves the compiler works without asking that question this early.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
