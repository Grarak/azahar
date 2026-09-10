# Known problems, and things worth trying

Ordered roughly by what would pay off most. Anything with a number attached was
measured; anything without is a guess and says so.

## Problems

### Frameskip decides from the wrong signal

`skip_next` fires when the render queue reaches half its ceiling. A queue that
is deep but *stable* is not overload — it means the render thread is retiring
work exactly as fast as the emulation thread produces it — and the skip latches
on anyway. On Smash's victory screen the queue sat between 233 and 399 without
trending, and the screen got one frame a second out of sixty.

`frames_behind >= PresentBacklogSkip` is the signal that belongs there and fires
on its own. The op-based one should go.

The deeper fix is further off. A frame ought to be *handed over* at one point,
so the decision to drop it can be made once, before any work exists — which is
what a snapshot-based renderer does. A PICA frame is a command stream replayed
op by op and has no such point. Reading the command list before executing it
would give us one.

### The GPU is at the frame budget on composite-heavy screens

Measured on the victory screen: 16.7 ms of GPU per guest frame against a 16.6 ms
budget at 60 fps, GPU 100% busy. That is not a comfortable margin, and the same
screen runs at a full 60 fps on the OpenGL tier with a quarter of its render
thread spare — so the work itself is not inherently expensive, the mapping onto
a tiler is.

Where to look first, in order:

- **The background object.** When a scene's first draw is blended, the hardware's
  full-screen colour-read quad shades instead of being deleted by hidden surface
  removal. Making the first draw an opaque far-plane quad, wherever the previous
  contents are not needed, would cost one trivial draw and delete a screen of
  texture reads per scene.
- **The valid region.** `sceGxmBeginScene` takes one and is passed `nullptr`
  everywhere. Tiles outside it get neither fragment processing nor a store, and
  most composite scenes touch a small rectangle of their target.
- **Depth store.** Every depth surface is created with force-load and force-store
  on. The load is genuinely needed when a later scene must see earlier depth,
  which target switching makes common; the store is not needed for the last
  scene to touch a surface before the guest clears it again.
- **`scenesPerFrame`.** Render targets are created with 8. The SDK is explicit
  that exceeding it blocks the CPU inside `sceGxmBeginScene`. The phase timer
  already wraps that call and its duration has never been read.

### Emitter gaps

These configurations are refused, and the draws that use them are skipped:
procedural texture noise, cube and shadow textures on unit 0, gas fog,
shadow-map rendering, logic operations other than copy, and minimum/maximum
blend emulation.

Logic ops are the interesting one, because the hardware can do them and we are
not using it. GXM has real framebuffer fetch: `__nativecolor` on the entry point
plus the `FRAGCOLOR` semantic gives the shader the current pixel as an integer,
which is exactly what the PICA's sixteen logic ops need. The emitter would need
to read the output register and to have bitwise operations; it has neither.

### SM3DL's textures are noise on the console only

The world renders correctly. The logo and the bottom screen's textures are
noise, and only on hardware — the same build under the host emulator is fine,
and a GPU capture says the texture *data* is right, so it is sampling rather
than upload. Comparing a surface dump from the console against the same dump
from the host emulator is the next step.

### Untested on hardware

- Savestate autoload. It works on the Linux tier; it has never been run on the
  console.
- Mirrored repeat on a linear texture, written by setting the control word
  directly past the library's refusal.
- The Vita audio sink has never actually been listened to.

### Titles that need hardware to progress

Pokémon Alpha Sapphire renders 2D only under the host emulator, because it reads
its 3D target back through the CPU every frame and that emulator returns nothing
for readbacks. Nothing says whether it works on the console.

## Optimisations worth trying

### Reduce scene count

The measured count is 25 target switches a guest frame on the screen we looked
at hardest — not the disaster earlier numbers suggested, but each one still ends
a scene. Two approaches, neither taken:

- Read the target graph out of the command list before executing it
  (`PicaCore::ProcessCmdList` already receives the whole buffer) and classify
  which targets never reach the screen. Start with a pass that only *logs* its
  classification, so it can be checked before anything acts on it.
- `SCE_GXM_RENDER_TARGET_MACROTILE_SYNC` is the SDK's own answer to
  "post-processing chains that would otherwise require a large number of
  expensive render target switches": several targets in one render target, with
  a synchronisation point between macrotiles. Blocked on geometry — macrotiles
  are at least 128x128 in a 2x2 or 4x4 grid, and 400x240 with 320x240 does not
  pack into that — and it forbids the valid region above.

### Sort opaque geometry before blended within a scene

Hidden surface removal only helps opaque primitives, and the ISP flushes its
tracked pixels whenever the shader pass type changes from discard or depth
replace to opaque or translucent — stalling until the discarded pixels have been
computed. 3DS titles alpha-test constantly and we replay in submission order, so
this churns. Opaque draws under a depth test are order-independent, so hoisting
them is legal.

### Vertex programs

Two known costs. Output attributes are held to the epilogue rather than written
to their output registers as they are produced. And the load hoisting is limited
by the secondary attribute budget, which reusing slots would relieve.

### Free downscale for 2x2 anti-aliasing

`SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE` with `SCE_GXM_MULTISAMPLE_NONE`
gives a 2x2 box downscale as a tile is stored, which is exactly what a title
using the 3DS's 2x2 mode wants, and its display transfer would stop being a
scene. Titles using 2x1 only map onto `SCE_GXM_MULTISAMPLE_2X`, at the cost of
the horizontal supersampling they asked for. The transfer engine cannot help:
its downscale is a fixed 50% in both axes.

### Textures

Swizzled rather than linear, and F16 where the emitter currently uses F32.
Neither measured.

### Overlap the CPU and GPU across the frame boundary

`sceGxmFinish` at the end of a frame means the two never overlap there. Whether
this still costs anything after the notification ring landed has not been
re-measured.

## Testing that does not exist yet

A set of small `.3dsx` scenes, one per PICA feature — texture modulate, alpha
test, scissor, multi-stage TEV with the combiner buffer, fog, lighting, lighting
with texture — that can be run on both the console and the host emulator and
diffed. Bisecting a wrong picture through a real game is far slower than it
needs to be, and every emitter bug so far has been found that way.
