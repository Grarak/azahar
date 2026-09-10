# How azahar-native works

Both the 3DS and the Vita are ARMv7-A. A 3DS game's user-mode code is already valid Vita
instructions, so the interesting problem is not translation but *address space*: the binary
wants `.text` at `0x00100000`, and in any ordinary Vita process that megabyte belongs to the
kernel, inside a single global TTBR0 table the kernel rewrites at runtime. Identity mapping
through the shared table is therefore impossible.

The way around it is to stop sharing the table. This plugin takes one core out of Sony's
scheduler, installs a translation table and exception vectors of its own on that core, and runs
the guest there in user mode at its own addresses. Nothing of Sony's is delivered to the core
while the guest owns it, so nothing else has to agree with the mapping.

## The interface

Nine syscalls, declared in `src/azahar_native.h`. The emulator drives them in this order:

| Call | What it does |
|---|---|
| `azaharTakeCore` | Claim a core. Returns 0 for the held design or `AZAHAR_TAKE_FOLDED` for the folded one. |
| `azaharMap` | Describe the guest's memory: ranges of the emulator's own user pages, and the guest addresses they appear at. Builds and installs the table. |
| `azaharRun` | Enter the guest with a register file, return it at the first exception, with the reason. |
| `azaharEdit` | Add, remove or re-permission guest pages between slices, and sync code the emulator wrote. |
| `azaharRelease` | Put Sony's registers back and hand the core over. |
| `azaharPmuRead` | The Cortex-A9 counters accumulated over guest slices. |
| `azaharPmuReadAll` | The same for all four cores, sampled by per-core threads. |
| `azaharSgiStats` | Interrupt-window counters, and which core the last slice ran on. |
| `azaharThreadDump` | Every thread of the client with its kernel-held registers, for diagnosing a hang. |

`src/main.c` is the support half — the report buffer, cp15 accessors, address translation, the
runtime export resolver, the private-peripheral discovery. `src/native.c` is the emulator half
and holds everything the calls do. `src/entry.S` is the entry and exit path. They are one
translation unit on purpose: the emulator half runs on the taken core, where a call through a
pointer into another object is one more thing that can go wrong.

## Two designs, one of them current

**Held.** A resident kernel thread occupies the core, then the core is cleared from the
scheduler's active mask. The thread never makes a kernel call afterwards — the scheduler is no
longer managing it, and a blocking call there is a way to never come back — so it talks to the
syscall side through volatiles and every wait on it is bounded. Enabled by creating
`ux0:data/azahar/resident`, kept because it is the design every measurement was made under.

**Folded (default).** No resident thread and no core taken. `azaharRun` installs the guest world
on *the calling core*, runs one slice, and restores Sony's world before returning. The emulator
pins its own thread to a core and the guest goes with it. This costs a world switch per slice —
measured at 12.9 µs, about 3.5% of the core at three thousand slices a second — and buys back
the whole core for everything else, including the emulator's own render thread.

Folded mode is what made the reattach problem go away as well. A core detached from the active
mask with a thread resident on it is never dispatched to again, and restoring the mask froze the
console twice; the held design therefore never restores it, and leaves the core detached with
the thread parked for the next title to reuse.

## The translation table

`azaharMap` takes up to eight ranges of the emulator's own user memory with the guest addresses
they should appear at. The pages behind a range need not be physically contiguous: each guest
megabyte gets a coarse L2 built over whatever physical pages the emulator's memblock actually
holds, resolved one page at a time through the MMU's own translation operations.

- TTBCR.N = 0, so the whole 4 GB goes through one 4096-entry L1.
- Short descriptors with SCTLR.AFE = 1, which this console sets: AP[0] is the Access Flag and
  must be set, and the permission lives in AP[2:1].
- Attributes TEX=001 C=1 B=1 (write-back, write-allocate) and S=1, so the other cores see what
  the guest writes.
- Domain 8 throughout. Domain 0 is No-access under this console's DACR and hangs it.
- Per-page W^X: `AZAHAR_PERM_W` clears APX, `AZAHAR_PERM_X` clears XN. The emulator uses this
  for write tracking — mark a page read-only, take the fault, learn the guest touched it.

`AZAHAR_SEAL` (a CMake option, on by default) decides whether the L1 is sealed to what the guest
needs or has the kernel's own tables merged into it. Sealed means a stray guest access faults
rather than reaching something of Sony's.

The vector page is a 1 MB section, PL1 read-write and executable, holding our own exception
vectors. Sony's handler for each slot is decoded at map time and kept, so an exception that
arrives when the guest is not running can be chained back to where it belongs.

## Entering and leaving

`entry.S` holds it. `azahar_enter` saves the host's state, loads the guest register file
(r0-r15, CPSR, TPIDRURO/TPIDRURW, FPSCR and the 32 double-precision VFP registers), and drops to
user mode at the guest's PC. It comes back through one of five paths:

| Reason | r15 on exit |
|---|---|
| `AZAHAR_EXIT_SVC` | the instruction after the `svc` |
| `AZAHAR_EXIT_UND` | the undefined instruction itself |
| `AZAHAR_EXIT_PABT` | the address whose fetch aborted |
| `AZAHAR_EXIT_DABT` | the instruction whose data access aborted |
| `AZAHAR_EXIT_TIMER` | the next instruction |

The timer is the Cortex-A9 private timer, PPI 29, loaded with `quantum_us` before each slice. A
guest that never traps would otherwise hold the core until the console is rebooted. Passing 0
runs without one.

While the guest is in, the GIC CPU interface's priority mask is set so that only the private
timer gets through. Leaving interrupts open between installing the table and the first run took
a prefetch abort within a tenth of a second — Sony's own IRQ path running under a copy of its L1
that had already gone stale.

## Cache and TLB discipline

The rules that matter, each of them learned the hard way:

- The table switch is a break-before-make block with a `dsb` on each side for erratum 754322.
  TTBCR and TTBR0 have to change inside one block: between them the CPU would read a 1024-entry
  table as though it had 4096.
- The table walker is coherent on this part, so descriptors written by `azaharEdit` are **not**
  cleaned to memory. A `dsb` for ordering is all that is needed. Cleaning them cost 25% of the
  emulator's speed, because every write-tracking edit cleaned 256 KB of second-level descriptors
  plus 16 KB of the top level.
- `TLBIALL` whenever the tables changed. `BPIALL` and `ICIALLU` only when guest *code* changed —
  the instruction cache is physically tagged, so a code sync is needed on the way in and never
  on the way out, where Sony's code has not moved.
- The VFP is saved and restored exactly as it was found, including FPEXC.

## Performance counters

`azaharPmuRead` gives the A9's counters accumulated over guest slices only: the cycle counter
and six events are reset just before the guest is entered and read at the exit, so the totals
cover guest execution and the few hundred cycles of the entry path, and nothing else that ever
ran on the core. The events are instructions renamed (the A9 has no retired-instruction event),
cycles stalled on an instruction-cache miss, cycles stalled on a data-cache miss, L1 data
refills, L1 instruction refills, and mispredicted branches.

`azaharPmuReadAll` adds whole-core samples for all four cores, taken every 10 ms by small
threads pinned to each. Those count everything the scheduler runs there, so they are a per-core
profile rather than a per-thread one.

## Build traps

Three of these fail silently, which is why they are written down.

- **No NEON, ever.** The toolchain defaults to `-mfpu=neon`, so GCC will auto-vectorise ordinary
  C loops. A kernel plugin must not touch the VFP/NEON register file — the kernel neither saves
  nor restores it around this code — so `-fno-tree-vectorize` is a correctness flag here. A
  trivial eight-element search loop was enough to trigger it.
- **`vita-elf-create` drops relocations to a symbol at a segment's exact end.** A label at a
  one-past-the-end address keeps its link-time value after load, with no warning, and faults
  only when used. `entry.S` pads after `azahar_irq_stack_top` for exactly this reason; it cost
  two system wedges and a kernel crash under three wrong theories before it was found.
- **Never link the `*ForKernel` stubs.** They are re-versioned every firmware —
  `SceProcessmgrForKernel` is library `0x7A69DE86` on 3.60 and `0xEB1F8EF7` on 3.63+ — so a
  link-time import of the wrong one makes the module fail to load with "Library not found".
  Anything needed from them is resolved at runtime through taiHEN's exporter, tried against both
  firmware generations.

Nothing runs at module load. Code on the boot path that faults bootloops the console, and the
only evidence is whatever reached the memory card first.

## Where this came from

This module was extracted from `vaprobe`, a diagnostic plugin written to answer whether any of
this was possible: whether a core could be taken, what was routed to it, what the low quarter of
the address space held, and whether the scheduler would give a core back. The `STAGE_*` names in
the comments are its measurement stages, and the dates beside them say when each fact was
established. Those stages are not part of this module — it carries only what the emulator calls.

The commit history here is vaprobe's, replayed for the files that carried over, under this
module's names.
