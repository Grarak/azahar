# Running azahar on the PlayStation Vita

Three problems, and this document is about how each is answered.

## 1. The CPU: don't emulate it

3DS' ARMv6 ISA is backwards compatible with the Vita's ARMv7 ISA. A 3DS game's
user-mode code is, instruction for instruction, code the Vita can already
execute. What stops it running is not the instruction set but the *address
space*: a 3DS binary expects its `.text` at `0x00100000`, its heap and stack
where the 3DS kernel put them, and on the Vita that memory belongs to the
kernel, inside a single global translation table it rewrites at runtime.

So the port stops sharing that table. A kernel plugin — `vita_native/`, built
from this tree — installs a translation table and exception vectors of its own,
maps the emulator's own guest-memory pages at the guest's addresses, and enters
the guest in user mode where it expects to be. Execution comes back on the first
exception: a system call, an undefined instruction, a fault, or a preemption
timer that bounds the slice so a guest which never traps cannot hold the core.
The emulator services it exactly as it would service the same exception from an
interpreter, and asks for the next slice.

There is no recompiler and no interpreter on this path. The guest's instructions
are the host's instructions.

Two designs exist for how the core is obtained. Each `azaharRun` installs the guest's
world on whichever core called it, runs one slice, and puts Sony's world back
before returning. That costs a world switch per slice, measured at 12.9 µs and
about 3.5% of a core at three thousand slices a second. `vita_native/DEVELOPMENT.md` has the detail.

## 2. Memory: one allocation, two addresses

The guest's memory has to be at the guest's addresses while the emulator still
reaches it at its own. Guest FCRAM and VRAM are therefore `HostSharedMemory` —
a memblock on the Vita, memfd on Linux — mapped twice.

That also gives the renderer a way to know when the guest writes to something it
has cached. `GuestWriteTracker` marks pages read-only in the guest's view; the
resulting fault names the page. A page written every frame is marked hot and
stops being re-armed, because re-arming it costs more than the fault saves. With
the kernel module no longer performing cache maintenance it did not need, the
whole mechanism costs nothing measurable.

## 3. The GPU: a tiler pretending to be an immediate-mode renderer

This is the hard one, and it is not about shader translation.

The PICA200 in a 3DS is an immediate-mode rasteriser. Its render target is a
register: write an address, and the next triangle lands there. Games use that
freely, switching target every few draws to build a frame through a chain of
small intermediate buffers.

The SGX543 is a tile-based deferred renderer. Its unit of work is a *scene*: the
screen is split into tiles, geometry is binned per tile, and each tile is shaded
on chip and written out once. Changing render target ends a scene, which stores
every tile to memory, and begins another. What is free on the PICA is the single
most expensive operation on this GPU.

So `renderer_gxm` is built around keeping a scene open. Consecutive draws to the
same target share one; blits and clears fold into the open scene instead of
opening their own; a scene is broken only when a dependency genuinely forces it.
The frame report counts scenes for exactly this reason.

Two things bite hard enough to be worth writing down.

**Hidden surface removal only works inside a scene, and only for opaque
geometry.** Splitting a frame across scenes does not just cost the tile stores —
it costs the overdraw rejection as well.

**There is a background object.** When a tile begins, the on-chip pixels are
undefined, so the hardware injects a full-screen textured quad that reads the
existing colour surface back through the shader. If the first primitive to touch
a pixel is opaque, hidden surface removal deletes that quad for free. If it is
blended — which a composite pass usually is — the quad shades, and a scene which
draws two triangles can still pay for a full screen of texture reads.

### Shaders

The Vita has a shader compiler as a system module. It is not present on every
console, it wants Cg source, and it is slow enough to stall a frame. So the
PICA's fragment and vertex configuration is emitted as **USSE machine code
directly**, in microseconds, in `renderer_gxm/usse/`.

It works in two tiers: the first places registers and produces a correct program
immediately; the second runs the optimising passes on a worker thread once a
program has survived a hundred draws, and swaps it in at a frame boundary.

### The freeze

Worth recording because the symptom gave nothing away. `sceGxmNotificationWait`
returns when the notification word *equals* the value — not when it has reached
or passed it. With one shared notification word, a thread that slept while
several scenes completed would wait for a value the GPU had already gone past,
and the console froze whole, with no log output, no crash dump, and a hard power
cycle as the only way out. There is one word per in-flight scene now, a ring of
32 chosen by serial.

## Threads and cores

The emulation thread never waits for the renderer. It enqueues ops and carries
on, which is what lets the guest hold full speed while the renderer is behind.
A queue brake exists as a backstop against a renderer that has stopped
draining altogether, and frameskip handles the ordinary case: when the renderer
cannot keep up, a frame's present is dropped *and its drawing with it*, because
a skipped frame that still draws costs everything and saves nothing.

## Audio

`sceAudioOut`, 48 kHz stereo, from a worker. The interesting part was the HLE
DSP rather than the sink: the mixer's buffers are flattened, and the biquad
filters and ADPCM decoder rewritten to run a frame at a time on NEON, bit-exact
against the scalar code they replace.

## Building

The Vita build cross-compiles from Linux with clang and lld — not vitasdk's gcc,
whose front end is too old for this tree — against a vitasdk sysroot:

```sh
VITASDK=/usr/local/vitasdk cmake -S . -B build-vita -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=CMakeModules/toolchain-vita-clang.cmake \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build-vita citra_vita.vpk-vpk
```

That is the release build. Add `-DENABLE_LTO=OFF` for the diagnostic one; see
below.

That produces `citra_vita.vpk` and, as a sub-build, the kernel plugin
`azaharnative.skprx`. Both are needed; `vita_native/README.md` says where the
plugin goes.

Four flags in that toolchain file are correctness rather than preference, and
each of them fails silently rather than erroring. They are documented where they
are set — do not remove one because it looks redundant.

## Two builds

`ENABLE_LTO` picks between them, and the difference is what is *compiled in*
rather than what is switched off at runtime.

**Release**, `-O3` with LTO, is what a user runs. No logging backend is started,
no clock is read for measurement, no counter is drained, and no file or
environment variable is consulted to decide how to behave. The strings are not
even in the binary.

**Diagnostic**, `-O3` without LTO, has all of it: the per-second statistics line,
the render-thread phase breadcrumb, the GPU's own counters, the on-screen stats
overlay, and the bring-up switches. Every measurement quoted in this file and in
TODO.md was taken with it. A bug worth reporting should be reproduced on it,
because the release build cannot tell you anything about itself.

```sh
-DENABLE_LTO=ON    # release
-DENABLE_LTO=OFF   # diagnostic
```

`ENABLE_VITA_DIAGNOSTICS` is the underlying switch and can be set on its own if you
want the diagnostics with LTO, or a small binary without them.

### Bring-up switches

In the diagnostic build, whitespace-separated words are read once from
`ux0:data/azahar/gxm_flags.txt` and documented in
`src/video_core/renderer_gxm/gxm_flags.h`. They exist so a question can be
answered on the console without a rebuild-and-reinstall cycle: run with a
feature off and see whether a wrong picture follows it. The Razor GPU Live
counter groups are selected the same way.

---

## Footnote: the 32-bit ARM Linux tier

Everything above needed somewhere to be developed that was not a console with no
debugger, so the port targets 32-bit ARM Linux as well — in practice a Raspberry
Pi 5 running an armhf userspace under a 64-bit kernel. It is a real target, not
a stub: `citra_sdl` is a working frontend, and the OpenGL renderer is what runs
there.

It shares everything below the renderer. The same native CPU backend, running
the guest in the emulator's own process through the second mapping rather than
through a kernel module; the same write tracking; the same threading; the same
NEON work in the software renderer, the texture codec and the audio DSP.

What it buys is that most of the port can be written, profiled and bisected with
`perf` and a debugger attached, and only the GXM renderer and the kernel module
genuinely require the hardware. It is also the reference: when a picture is
wrong on the console, the same title on the GL tier says whether the fault is in
the shared half or the Vita's own.

The frontend's debug port matters more than it looks. Over a socket it takes
button presses, screenshots, frame dumps and savestate commands, so a script can
drive a title to a particular screen and measure there — which by hand is
minutes of button-pressing before every run.
