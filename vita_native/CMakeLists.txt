cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(azaharnative C ASM)
include("${VITASDK}/share/vita.cmake" REQUIRED)

# -fno-tree-vectorize is a correctness flag here, not a preference. The toolchain defaults to
# -mfpu=neon, so GCC will auto-vectorise ordinary C loops into NEON - and a kernel plugin must
# not touch the VFP/NEON register file, because the kernel neither saves nor restores it around
# our code and doing so would corrupt whichever thread was interrupted. A trivial eight-element
# search loop was enough to trigger it.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-q -Wall -O2 -nostdlib -fno-optimize-sibling-calls -fno-tree-vectorize")

# The guest's translation table: 1 (default) is sealed to what the taken core needs, 0 merges
# the kernel's own tables into it. See DEVELOPMENT.md.
set(AZAHAR_SEAL "1" CACHE STRING "1 = sealed L1, 0 = merged kernel L1")
add_definitions(-DAZAHAR_SEAL=${AZAHAR_SEAL})

add_executable(azaharnative src/main.c src/entry.S)

target_link_libraries(azaharnative
  taihenForKernel_stub
  taihenModuleUtils_stub
  SceCpuForDriver_stub
  SceIofilemgrForDriver_stub
  SceProcessmgrForDriver_stub
  SceProcEventForDriver_stub
  SceSysclibForDriver_stub
  SceSysmemForDriver_stub
  SceThreadmgrForDriver_stub
  SceDebugForDriver_stub
)

# Deliberately NOT linked: SceProcessmgrForKernel_stub and SceSysmemForKernel_stub. The
# *ForKernel libraries are re-versioned every firmware - SceProcessmgrForKernel is library
# 0x7A69DE86 on 3.60 and 0xEB1F8EF7 on 3.63+ - so a link-time import of the wrong one makes the
# whole module fail to load with "Library not found". Anything needed from them is resolved at
# runtime instead.

vita_create_self(azaharnative.skprx azaharnative CONFIG exports.yml UNSAFE)
vita_create_stubs(stubs azaharnative ${CMAKE_CURRENT_SOURCE_DIR}/exports.yml KERNEL)
