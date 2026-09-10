// azahar-native — the ABI between the kernel plugin and the emulator. Included by both.
//
// The PS Vita is Sony's handheld games console; this is a hobby emulator project running on a
// console the author owns. The plugin takes one CPU core away from the Vita's scheduler and
// runs guest (Nintendo 3DS) code on it natively, at the guest's own addresses. DEVELOPMENT.md
// beside this file is the design.

#ifndef AZAHAR_NATIVE_H
#define AZAHAR_NATIVE_H

#include <stdint.h>

#define AZAHAR_MAX_RANGES 8

// Permissions for a mapped range. Every guest page is reachable from user mode (the guest runs
// at PL0); these decide read/write/execute. W without R is not supported.
#define AZAHAR_PERM_R 1u
#define AZAHAR_PERM_W 2u
#define AZAHAR_PERM_X 4u

// One range of the app's memory to appear at a guest virtual address. user_va, guest_va and size
// are 4 KB aligned; the pages behind user_va need not be physically contiguous — the plugin
// builds coarse page tables over whatever pages the app's memblock has.
typedef struct AzaharRange {
    uint32_t user_va;
    uint32_t guest_va;
    uint32_t size;
    uint32_t perm;
} AzaharRange;

typedef struct AzaharMapRequest {
    uint32_t count;             // may be 0: install the table with no windows, add them with azaharEdit
    AzaharRange ranges[AZAHAR_MAX_RANGES];
} AzaharMapRequest;

// The guest register file. r[13] and r[14] are the user-mode SP and LR; r[15] is the PC to
// resume at; cpsr is the full user-mode CPSR (mode bits 0x10, T for Thumb). tpidruro is the
// guest's TLS register.
// Layout is shared with entry.S: r0-r15 at 0x00, cpsr 0x40, tpidruro 0x44, tpidrurw 0x48,
// fpscr 0x4c, the 32 double-precision VFP registers as 64 words at 0x50. 0x150 bytes.
typedef struct AzaharContext {
    uint32_t r[16];
    uint32_t cpsr;
    uint32_t tpidruro;
    uint32_t tpidrurw;
    uint32_t fpscr;
    uint32_t fpregs[64];
} AzaharContext;

// Why the guest stopped. The context is updated in place; r[15] is:
//   SVC   the instruction after the svc (resume here to continue)
//   UND   the undefined instruction itself
//   PABT  the address whose fetch aborted
//   DABT  the instruction whose data access aborted
#define AZAHAR_EXIT_SVC 1u
#define AZAHAR_EXIT_UND 2u
#define AZAHAR_EXIT_PABT 3u
#define AZAHAR_EXIT_DABT 4u
#define AZAHAR_EXIT_TIMER 5u   // the preemption quantum expired; r[15] is the next instruction

typedef struct AzaharRunRequest {
    AzaharContext ctx;       // in: the state to enter with; out: the state at exit
    uint32_t reason;      // out: AZAHAR_EXIT_*
    uint32_t svc_number;  // out: for AZAHAR_EXIT_SVC, the immediate (24-bit ARM, 8-bit Thumb)
    uint32_t fsr;         // out: DFSR (DABT) or IFSR (PABT), else 0
    uint32_t far;         // out: DFAR (DABT) or IFAR (PABT), else 0
    uint32_t spsr;        // out: the SPSR the exception saw (the guest CPSR at the exception)
    uint32_t raw_lr;      // out: the exception-mode LR before adjustment, for cross-checking
    uint32_t quantum_us;  // in: preemption quantum; 0 runs without a timer (the guest must trap)
    uint32_t timer_hz;    // out: the private timer's measured clock
    uint32_t timer_left;  // out: timer counter at exit (ticks not yet spent)
} AzaharRunRequest;

// Runtime edits to the guest's mapping, for svcControlMemory / svcMapMemoryBlock and friends.
// Only between runs: the guest is stopped at an exit while the app handles its svc. MAP adds
// pages (a new megabyte gets an L2 from the pool); UNMAP clears them; PROTECT rewrites the
// permission of pages already mapped, keeping their physical pages. The next azaharRun flushes
// core 2's TLB, branch predictor and instruction cache before entering the guest.
#define AZAHAR_EDIT_MAP 1u
#define AZAHAR_EDIT_UNMAP 2u
#define AZAHAR_EDIT_PROTECT 3u
#define AZAHAR_EDIT_SYNC_CODE 4u  // the app wrote code into mapped pages (a CRO link, a loader):
                               // clean them to memory and have core 2 drop its icache. user_va
                               // names the app's view of the range; guest_va/perm are ignored

typedef struct AzaharEditRequest {
    uint32_t op;        // AZAHAR_EDIT_*
    AzaharRange range;     // user_va is ignored for UNMAP; perm is ignored for UNMAP
} AzaharEditRequest;

// Cortex-A9 performance counters, accumulated over guest slices. The plugin reprograms core 2's
// PMU at every azaharRun: the cycle counter and the six event counters are reset just before the
// guest is entered and read back at the exit, so the totals cover guest execution (plus the few
// hundred cycles of the entry/exit path) and nothing else that ever ran on the core. azaharPmuRead
// copies the totals out and zeroes them. Call it from the thread that calls azaharRun: the totals
// are not synchronized against a run in flight.
// events[] order (Cortex-A9 event numbers):
//   0  0x68 instructions passing the rename stage (the A9 has no retired-instruction event)
//   1  0x60 cycles stalled on an instruction-cache miss
//   2  0x61 cycles stalled on a data-cache miss
//   3  0x03 L1 data cache refills
//   4  0x01 L1 instruction cache refills
//   5  0x10 mispredicted branches
typedef struct AzaharPmuStats {
    uint64_t cycles;
    uint64_t events[6];
    uint32_t runs;    // azaharRun slices the totals cover
    uint32_t pad;
} AzaharPmuStats;

// All four cores' counters. core[2] is the guest slice totals described above; the other
// cores are whole-core samples, taken every 10 ms by a small kernel thread pinned to each,
// counting everything scheduled there. The first azaharPmuReadAll starts the samplers; every
// read drains the totals.
typedef struct AzaharPmuAll {
    AzaharPmuStats core[4];
} AzaharPmuAll;

// Error codes (negative), beyond the kernel's own.
#define AZAHAR_ERR_STATE (-1000)      // call out of order (map before take, run before map...)
#define AZAHAR_ERR_ARG (-1001)        // bad request (alignment, count, perm)
#define AZAHAR_ERR_RESOLVE (-1002)    // a kernel export needed did not resolve
#define AZAHAR_ERR_ALLOC (-1003)      // a kernel allocation failed
#define AZAHAR_ERR_CORE (-1004)       // the core could not be taken, or the thread did not answer
#define AZAHAR_ERR_TABLE (-1005)      // the table could not be built (kernel L1 not found, gap)
#define AZAHAR_ERR_PAGE (-1006)       // a user page did not translate
#define AZAHAR_ERR_TIMEOUT (-1007)    // the core-2 thread did not complete a command in time

#ifdef __cplusplus
extern "C" {
#endif

// User-side prototypes (the kernel exports these as syscalls).
/// azaharTakeCore's success value when the module folded the guest into the caller: no core is
/// held, and each slice runs on whichever core calls azaharRun. Plain 0 means a core was taken
/// and the caller must stay off it.
#define AZAHAR_TAKE_FOLDED 1
int azaharTakeCore(void);
int azaharMap(const AzaharMapRequest *req);
int azaharRun(AzaharRunRequest *req);
int azaharRelease(void);
int azaharEdit(const AzaharEditRequest *req);
int azaharPmuRead(AzaharPmuStats *out);
int azaharPmuReadAll(AzaharPmuAll *out);

/// The SGI service-window counters and the resident loop's heartbeat (resident mode; the
/// folded mode reports zeros except installed). pending_now is the banked GICD_ISPENDR0.
typedef struct AzaharSgiStats {
    uint32_t windows;
    uint32_t last_pend;
    uint32_t stuck;
    uint32_t heartbeat;
    uint32_t installed;
    uint32_t pending_now;
    /// The core the last slice ran on: the resident thread's in the held design, the caller's
    /// own in the folded one, where it answers "is the guest where the partition put it".
    uint32_t core;
} AzaharSgiStats;
int azaharSgiStats(AzaharSgiStats *out);
/// Writes every thread of the client with its kernel-held registers (user and kernel pc,
/// lr, sp; status, priority, last core, wait) to ux0:data/azahar/native.txt, when logging
/// is enabled (ux0:data/azahar/log). Returns the number of threads seen, 0 without a client.
int azaharThreadDump(void);

#ifdef __cplusplus
}
#endif


#endif
