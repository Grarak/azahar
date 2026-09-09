// azaharnative — native.c: the emulator half. Takes one PS Vita core away from Sony's scheduler and runs
// guest code on it natively, at the guest's own addresses, under translation tables and
// exception vectors of ours.
//
// The PS Vita is Sony's handheld games console (2011–2019); "Vita" is product branding. This is
// a hobby emulator project (Nintendo 3DS games on Vita hardware) on a console the author owns.
// The plugin loads through taiHEN, the public plugin framework every Vita homebrew uses. It has
// no networking and touches no system but the one it runs on.
//
// Included from main.c, after the stages, whose helpers it reuses (emit, translate, the cp15
// accessors, resolve_export, the mask getter and setter, gic_locate). Exported as four syscalls
// alongside azaharnativeRun. What it implements is ~/3ds-vita/PLAN.md §8.11 steps 3–9 and 12, the
// pieces the stages measured one at a time and never joined:
//
//   azaharTakeCore   occupy core 2 with a resident kernel thread, then clear core 2 from the
//                 scheduler's mask (STAGE_OWN / STAGE_HOLD, 2026-08-26 / 08-29)
//   azaharMap        the app owns the guest memory as ordinary user memblocks; the plugin resolves
//                 their physical pages, builds an N=0 translation table — the kernel's own two
//                 tables merged, with the guest windows as coarse L2s over the app's page lists
//                 and per-page W^X — plus a vector page, and installs both on core 2 (STAGE_OWN
//                 N=0, STAGE_L2, 2026-08-29)
//   azaharRun        enter the guest in user mode with a register file, leave it on the first
//                 exception (svc, undefined, prefetch or data abort), return the register file
//                 and the reason (entry.S; new)
//   azaharRelease    put Sony's registers back on core 2, end the resident thread, and hand the
//                 core back to the scheduler. The order is what makes the hand-back safe:
//                 restoring the mask while the thread still occupied the core left the core
//                 unschedulable and froze the app at exit (measured 2026-08-29), so the thread
//                 is stopped and waited for first - the order azahar_on_proc_gone has always used
//                 for a dead client. A release that cannot end the thread leaves the core
//                 detached and refuses to be taken again.
//
// Everything the resident thread does after the detach avoids kernel calls: the scheduler is no
// longer managing it, and a blocking call there is a way to never come back. It talks to the
// syscall side through volatiles, and every wait on it is bounded.
//
// Not here yet: a preemption timer (a guest that never takes an exception holds the core until
// reboot), the HLE bridge, and the guest scheduler. One guest thread, run to its next exception.

#include <psp2kern/kernel/proc_event.h>
#include <psp2kern/kernel/sysmem/data_transfers.h>
#include "azahar_native.h"

// ---------------------------------------------------------------------------- configuration

#define AZAHAR_TARGET_CORE 2
#define AZAHAR_LOG_PATH OUT_DIR "/native.txt"

#define AZAHAR_L1_ENTRIES 4096      // TTBCR.N = 0: the whole 4 GB through our table
#define AZAHAR_L2_COUNT 256         // one 1 KB L2 per guest megabyte: 256 MB of windows, 256 KB
#define AZAHAR_DOMAIN 8u            // a Client domain under this console's DACR (15450000)
#define AZAHAR_WAIT_STEP_US 50000
#define AZAHAR_WAIT_STEPS 40        // 2 s
#define AZAHAR_RUN_WAIT_STEPS 200   // 10 s: no preemption yet, so a guest may take a while to svc
#define AZAHAR_PMR_MASK 0x10u       // PLAN.md §8.9: passes the private timer, masks everything else

#ifndef GICC_PMR
#define GICC_PMR 0x104u
#endif
// Cortex-A9 MPCore private timer, per core, at PERIPHBASE + 0x600; its interrupt is PPI 29.
#ifndef PTIMER_LOAD
#define PTIMER_LOAD 0x600u
#endif
#ifndef PTIMER_COUNTER
#define PTIMER_COUNTER 0x604u
#endif
#ifndef PTIMER_CONTROL
#define PTIMER_CONTROL 0x608u
#endif
#ifndef PTIMER_STATUS
#define PTIMER_STATUS 0x60Cu
#endif
#define PTIMER_CTRL_ENABLE 1u
#define PTIMER_CTRL_RELOAD 2u
#define PTIMER_CTRL_IRQ 4u
#define PTIMER_PPI 29u
#ifndef GICD_ISENABLER0
#define GICD_ISENABLER0 0x1100u   // banked: this core's IDs 0-31
#endif
#ifndef GICD_ICENABLER0
#define GICD_ICENABLER0 0x1180u
#endif
#ifndef GICD_ISPENDR0
#define GICD_ISPENDR0 0x1200u     // banked: this core's pending IDs 0-31
#endif
#define AZAHAR_TIMER_MEASURE_US 300000u

// 1: the table holds only what core 2 needs — guest windows, the vector page, this module's
//    megabytes, the resident thread's stack, the GIC alias — and nothing of the kernel's.
// 0: the kernel's two tables merged, as the first runs used. With interrupts masked from install
//    nothing of Sony's executes on core 2, so the merge has no purpose left; without the seal a
//    guest that ran off its window executed 5.7 MB of kernel-mapped memory (2026-08-29).
#ifndef AZAHAR_SEAL
#define AZAHAR_SEAL 1
#endif

// ---------------------------------------------------------------------------- log
//
// azaharnative's emit() writes to out_fd, which run_probe opens for a report and closes after. The
// native calls keep a file of their own open from the first call to the release — the emulator
// makes thousands of these calls a second, and an open per call would cost more than the
// guest work between them — and point out_fd at it for the duration of each call. Lines are
// still synced as written; azaharRun writes none on the success path for the same reason.
static SceUID azahar_fd = -1;

// Logging is opt-in: ux0:data/azahar/log present means native.txt and the kernel printf
// mirror as before; absent (the default) means no file, no printf, and emit() returns at
// once. Checked once per take.
static int azahar_log_checked, azahar_log_wanted;
static int azahar_log_enabled(void) {
    if (!azahar_log_checked) {
        azahar_log_checked = 1;
        const SceUID fd = ksceIoOpen(OUT_DIR "/log", SCE_O_RDONLY, 0);
        azahar_log_wanted = fd >= 0;
        if (fd >= 0) {
            ksceIoClose(fd);
        }
    }
    return azahar_log_wanted;
}

static void log_open(void) {
    if (!azahar_log_enabled()) {
        azahar_mute = 1;
        out_fd = -1;
        buf_len = 0;
        log_sync = 0;
        return;
    }
    azahar_mute = 0;
    if (azahar_fd < 0) {
        ksceIoMkdir(OUT_DIR, 0777);
        azahar_fd = ksceIoOpen(AZAHAR_LOG_PATH, SCE_O_WRONLY | SCE_O_CREAT | SCE_O_APPEND, 0777);
    }
    out_fd = azahar_fd;
    buf_len = 0;
    log_sync = 1;
}

static void log_close(void) {
    if (azahar_mute) {
        return;
    }
    flush_buf();
    out_fd = -1;
    log_sync = 0;
}

static void log_release(void) {
    log_close();
    if (azahar_fd >= 0) {
        ksceIoClose(azahar_fd);
        azahar_fd = -1;
    }
}

// ---------------------------------------------------------------------------- cp15 extras

CP15_READ(tpidruro, c13, 0, c0, 3)

static inline void wr_tpidruro(uint32_t v) {
    asm volatile("mcr p15, 0, %0, c13, c0, 3 \n isb" ::"r"(v) : "memory");
}

static inline uint32_t rd_cpsr(void) {
    uint32_t v;
    asm volatile("mrs %0, cpsr" : "=r"(v));
    return v;
}

// ---------------------------------------------------------------------------- descriptors
//
// Short-descriptor format with SCTLR.AFE = 1 (sctlr 20c5587d on this console): AP[0] is the
// Access Flag and must be set, the permission is AP[2:1]. Measured to behave exactly so by
// azaharnative's STAGE_L2 on 2026-08-29. Memory attributes: TEX=001 C=1 B=1 (write-back,
// write-allocate), S=1 so the other cores see what the guest writes. Domain 8 throughout — D0 is
// No-access under this console's DACR and hung the console on 2026-08-26.

#define AZAHAR_DOMAIN_FIELD (AZAHAR_DOMAIN << 5)

// Section, PL1 RW only, executable: for the vector page. AP[2:1] = 00, AF = 1 -> bits 11:10 = 01.
#define AZAHAR_SEC_PRIV_RWX(pa) (((pa) & 0xFFF00000u) | (1u << 16) | (1u << 12) | (1u << 10) | \
                              AZAHAR_DOMAIN_FIELD | (1u << 3) | (1u << 2) | 2u)

// Coarse page table descriptor: L2 base (1 KB aligned), domain, type 01.
#define AZAHAR_COARSE(l2_pa) (((l2_pa) & 0xFFFFFC00u) | AZAHAR_DOMAIN_FIELD | 1u)

// Small page for a guest window: reachable from PL0 (AP[1] = 1), AF set, XN and APX per perm.
static uint32_t azahar_small_page(uint32_t pa, uint32_t perm) {
    uint32_t d = (pa & 0xFFFFF000u) | (1u << 10) /* S */ | (1u << 6) /* TEX */ | (1u << 3) |
                 (1u << 2) | (1u << 4) /* AF */ | (1u << 5) /* PL0 */ | 2u;
    if (!(perm & AZAHAR_PERM_W)) {
        d |= (1u << 9); // APX: read-only
    }
    if (!(perm & AZAHAR_PERM_X)) {
        d |= 1u; // XN
    }
    return d;
}

// ---------------------------------------------------------------------------- kernel exports

static int resolve_exports(void) {
    uintptr_t f = 0;
    if (!resolve_export("ChangeActiveCpuMask", "SceKernelThreadMgr", 0xE2C40624, 0x001173F8,
                        0x859A24B1, 0x001173F8, &f)) {
        return 0;
    }
    p_ChangeActiveCpuMask = (fn_change_active_cpu_mask)f;
    sysinfo_ready();
    return 1;
}

// ---------------------------------------------------------------------------- GIC
//
// gic_locate() (main.c) finds PERIPHBASE's alias and sets gic_va / gic_has_dist. It normally runs
// as part of the probe; the native calls run it themselves if it has not.
static void azahar_gic(void) {
    if (!gic_va) {
        gic_locate();
    }
    if (gic_va && !gic_has_dist) {
        emit("  gic: CPU interface reachable but not the distributor; PMR masking still works\n");
    }
    emit("  gic va %08x%s\n", gic_va,
         gic_va ? "" : " (not found: the guest will run with interrupts unmasked at the GIC)");
}

// ---------------------------------------------------------------------------- state

enum { ST_NONE, ST_TAKEN, ST_MAPPED, ST_RELEASED };
static int azahar_state;

static SceUID azahar_l1_uid = -1, azahar_l2_uid = -1, azahar_vec_uid = -1;
static uint32_t *azahar_l1, *azahar_l2, *azahar_vec;
static uintptr_t azahar_l1_pa, azahar_l2_pa, azahar_vec_pa;
static uint32_t azahar_vec_va;      // where the vector page appears in our table
static unsigned azahar_l2_used;
static uint32_t azahar_l2_slot[AZAHAR_L2_COUNT]; // which L1 index each L2 serves

static SceUID azahar_thread = -1;
static SceUID azahar_client_pid = -1;      // the process azaharTakeCore ran for
static int azahar_detached;                // whether this session took core 2 out of the mask
static SceUID azahar_proc_handler = -1;    // registered once, on the first take
static uint32_t azahar_saved_mask;         // the active CPU mask before the detach
static uint32_t azahar_sony[8]; // Sony's handler per vector slot, decoded at map time

// Resident thread <-> syscall side. Everything volatile, no kernel calls on the core-2 side.
enum { CMD_NONE, CMD_INSTALL, CMD_RUN, CMD_UNINSTALL };
static volatile uint32_t azahar_cmd, azahar_cmd_done, azahar_cmd_status;
static volatile uint32_t azahar_heartbeat, azahar_on_core, azahar_stop, azahar_park, azahar_park_cycles;

// PMU totals over guest slices; see AzaharPmuStats in vanative.h. Written by core 2 between
// slices, read and zeroed by azaharPmuRead on the syscall core. The emulator calls azaharRun and
// azaharPmuRead from one thread, so the two never overlap and the 64-bit sums need no lock.
static volatile uint64_t azahar_pmu_cycles;
static volatile uint64_t azahar_pmu_ev[6];
static volatile uint32_t azahar_pmu_runs;
static const uint32_t azahar_pmu_events[6] = {0x68u, 0x60u, 0x61u, 0x03u, 0x01u, 0x10u};

// Host-core sampling (azaharPmuReadAll): one kernel thread pinned to each schedulable core
// programs that core's PMU with the same six events and folds 10 ms deltas into per-core
// totals. The PMU is banked per core and counts whatever the scheduler runs there, so this is
// a per-core profile, not a per-thread one. Deltas are unsigned 32-bit subtractions, so a
// counter wrapping between samples is harmless; the readout races the samplers by design
// (statistics, not bookkeeping).
static volatile uint64_t azahar_pmu_host_cycles[4];
static volatile uint64_t azahar_pmu_host_ev[4][6];
static volatile uint32_t azahar_pmu_host_samples[4];
static volatile uint32_t azahar_pmu_host_stop;
static SceUID azahar_pmu_host_thread[4] = {-1, -1, -1, -1};
static uint32_t azahar_pmu_host_core_arg[4] = {0, 1, 2, 3};

static uint32_t pmu_read_event(uint32_t i) {
    uint32_t v;
    asm volatile("mcr p15, 0, %0, c9, c12, 5" ::"r"(i));  // PMSELR
    asm volatile("isb");
    asm volatile("mrc p15, 0, %0, c9, c13, 2" : "=r"(v)); // PMXEVCNTR
    return v;
}

static int pmu_host_entry(SceSize args, void *argp) {
    (void)args;
    const uint32_t core = *(const uint32_t *)argp;
    PmuSave save;
    pmu_begin(&save); // PMCR.E on, cycle counter on, previous state saved; PMCR.C stays clear
    for (uint32_t i = 0; i < 6; i++) {
        asm volatile("mcr p15, 0, %0, c9, c12, 5" ::"r"(i));                  // PMSELR
        asm volatile("mcr p15, 0, %0, c9, c13, 1" ::"r"(azahar_pmu_events[i]));  // PMXEVTYPER
    }
    asm volatile("mcr p15, 0, %0, c9, c12, 1" ::"r"(0x3Fu));  // PMCNTENSET: events 0-5
    asm volatile("isb" ::: "memory");
    uint32_t prev[7];
    prev[6] = rd_pmccntr();
    for (uint32_t i = 0; i < 6; i++) {
        prev[i] = pmu_read_event(i);
    }
    while (!azahar_pmu_host_stop) {
        ksceKernelDelayThread(10000);
        uint32_t cur = rd_pmccntr();
        azahar_pmu_host_cycles[core] += (uint32_t)(cur - prev[6]);
        prev[6] = cur;
        for (uint32_t i = 0; i < 6; i++) {
            cur = pmu_read_event(i);
            azahar_pmu_host_ev[core][i] += (uint32_t)(cur - prev[i]);
            prev[i] = cur;
        }
        azahar_pmu_host_samples[core]++;
    }
    asm volatile("mcr p15, 0, %0, c9, c12, 2" ::"r"(0x3Fu));  // PMCNTENCLR: events 0-5
    pmu_end(&save);
    return 0;
}

static void pmu_host_start(void) {
    static const uint32_t cores[3] = {0, 1, 3};
    static const char *const names[3] = {"vanpmu_c0", "vanpmu_c1", "vanpmu_c3"};
    if (azahar_pmu_host_thread[0] >= 0) {
        return;
    }
    azahar_pmu_host_stop = 0;
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t c = cores[i];
        const SceUID t = ksceKernelCreateThread(names[i], pmu_host_entry, 0x10000100, 0x1000, 0,
                                                CPU_AFFINITY(c), NULL);
        if (t < 0) {
            continue;
        }
        azahar_pmu_host_thread[c] = t;
        ksceKernelStartThread(t, sizeof(azahar_pmu_host_core_arg[c]), &azahar_pmu_host_core_arg[c]);
    }
}

static void pmu_host_stop_all(void) {
    azahar_pmu_host_stop = 1;
    for (unsigned c = 0; c < 4; c++) {
        if (azahar_pmu_host_thread[c] >= 0) {
            SceUInt timeout = 1000000;
            ksceKernelWaitThreadEnd(azahar_pmu_host_thread[c], NULL, &timeout);
            ksceKernelDeleteThread(azahar_pmu_host_thread[c]);
            azahar_pmu_host_thread[c] = -1;
        }
    }
}
static volatile uint32_t azahar_core2_vbar, azahar_core2_cpsr, azahar_core2_ttbr1, azahar_core2_sp;

// Saved on core 2 at install, restored at uninstall.
static volatile uint32_t sv_ttbcr, sv_ttbr0, sv_ctxid, sv_vbar, sv_sp_und, sv_sp_abt, sv_tpidruro;
static volatile uint32_t sv_tpidrurw, sv_cpacr, sv_fpexc;
static uint32_t sv_fpscr;
static uint64_t sv_vfp_bank[32];

static inline uint32_t rd_tpidrurw(void) {
    uint32_t v;
    asm volatile("mrc p15, 0, %0, c13, c0, 2" : "=r"(v));
    return v;
}
static inline void wr_tpidrurw(uint32_t v) {
    asm volatile("mcr p15, 0, %0, c13, c0, 2 \n isb" ::"r"(v) : "memory");
}
static inline uint32_t rd_fpexc_here(void) {
    uint32_t v;
    asm volatile(".fpu neon \n vmrs %0, fpexc" : "=r"(v));
    return v;
}
static inline void wr_fpexc(uint32_t v) {
    asm volatile(".fpu neon \n vmsr fpexc, %0 \n isb" ::"r"(v) : "memory");
}
static inline uint32_t rd_fpscr_here(void) {
    uint32_t v;
    asm volatile(".fpu neon \n vmrs %0, fpscr" : "=r"(v));
    return v;
}
static inline void wr_fpscr(uint32_t v) {
    asm volatile(".fpu neon \n vmsr fpscr, %0" ::"r"(v));
}
static void vfp_bank_save(uint64_t *p) {
    asm volatile(".fpu neon \n vstmia %0!, {d0-d15} \n vstmia %0, {d16-d31}"
                 : "+r"(p)
                 :
                 : "memory");
}
static void vfp_bank_load(const uint64_t *p) {
    asm volatile(".fpu neon \n vldmia %0!, {d0-d15} \n vldmia %0, {d16-d31}" : "+r"(p));
}
static volatile uint32_t azahar_ttbcr_seen;

// The guest context and the host block live here, word aligned; entry.S addresses them through
// azahar_ctx_ptr / azahar_host_ptr.
static AzaharContext azahar_ctx __attribute__((aligned(8)));
static uint32_t azahar_host[4] __attribute__((aligned(8)));
static volatile uint32_t azahar_run_reason, azahar_run_svc, azahar_run_fsr, azahar_run_far, azahar_run_spsr,
    azahar_run_rawlr, azahar_run_pmr_before, azahar_run_pmr_after;

extern int azahar_enter(AzaharContext *ctx, uint32_t *host);
extern uint32_t azahar_get_banked_sp(uint32_t mode);
extern void azahar_set_banked_sp(uint32_t mode, uint32_t sp);
extern void azahar_vec_und(void);
extern void azahar_vec_svc(void);
extern void azahar_vec_pabt(void);
extern void azahar_vec_dabt(void);
extern uint32_t azahar_exit_info[7];
extern volatile uint32_t azahar_progress, azahar_prog_und, azahar_prog_svc, azahar_prog_pabt, azahar_prog_dabt;
extern volatile uint32_t azahar_guest_active, azahar_stray[8];
extern uint32_t azahar_chain_und[2], azahar_chain_svc[2], azahar_chain_pabt[2], azahar_chain_dabt[2];
extern uint32_t azahar_chain_irq[2];
extern volatile uint32_t azahar_prog_irq, azahar_gic_va, azahar_irq_last, azahar_irq_stray, azahar_seal_flag;
extern void azahar_vec_irq(void);
extern uint8_t azahar_stray_stack_top[], azahar_irq_stack_top[];
extern volatile uint32_t azahar_stray_retry;
// taiHEN's module utility, imported but not declared by any header vitasdk installs.
extern int module_get_offset(SceUID pid, SceUID modid, int segidx, size_t offset,
                             uintptr_t *addr);

#define MODE_IRQ 0x12u
static volatile uint32_t sv_sp_irq, sv_isenabler_timer;
static volatile uint32_t azahar_timer_sample;   // idle loop: the free-running counter
static uint32_t azahar_timer_hz;                // measured at map
static volatile uint32_t azahar_run_quantum_ticks, azahar_run_timer_left;
static volatile uint32_t azahar_tlb_dirty;      // an edit happened since the guest last ran
static uint32_t azahar_run_count;
// Core-2 side: our table/vectors/PMR are in (set at the end of core2_install, cleared first
// thing in core2_uninstall), so the resident loop knows when the SGI poll applies.
static volatile uint32_t azahar_installed;
// SGI service windows (core2_sgi_window): how many opened, the last pending mask that opened
// one, and how many timed out with a bit still pending. Read on the syscall core at release.
static volatile uint32_t azahar_sgi_windows, azahar_sgi_last_pend, azahar_sgi_stuck;
// 1: every azaharRun logs its entry and exit (the trigger app's test); 0: only faults, preemptions
// and errors (the emulator). The test app asks for it through azaharMap's count sign — see azaharMap.
static int azahar_verbose;
// Folded (the default): no resident thread. azaharRun switches the calling core into the guest
// world for one slice and back, so the emulation thread and the guest share a core and the
// other three are the kernel's. ux0:data/azahar/resident selects the old design, a kernel
// thread holding core 2 and the emulation thread posting commands to it from another core.
static int azahar_folded = 1;
static int azahar_resident_requested(void) {
    const SceUID fd = ksceIoOpen(OUT_DIR "/resident", SCE_O_RDONLY, 0);
    if (fd < 0) {
        return 0;
    }
    ksceIoClose(fd);
    return 1;
}

#define MODE_SVC 0x13u
static volatile uint32_t sv_sp_svc, sv_pmr;

#define MODE_UND 0x1Bu
#define MODE_ABT 0x17u

// ---------------------------------------------------------------------------- core 2 side

// The table switch, both directions. ARM ARM B3.10.4 example B3-3 with erratum 754322's dsb on
// each side; TTBCR and TTBR0 change inside one break-before-make block because between them the
// CPU would read a 1024-entry table as though it had 4096 (azaharnative STAGE_OWN, 2026-08-26).
static void core2_switch(uint32_t ttbcr, uint32_t ttbr0, uint32_t ctxid, int icache) {
    asm volatile("dsb                             \n"
                 "mcr p15, 0, %[zero], c13, c0, 1 \n"
                 "isb                             \n"
                 "mcr p15, 0, %[tcr],  c2, c0, 2  \n"
                 "mcr p15, 0, %[ttb],  c2, c0, 0  \n"
                 "isb                             \n"
                 "mcr p15, 0, %[ctx],  c13, c0, 1 \n"
                 "dsb                             \n"
                 "mcr p15, 0, %[zero], c8, c7, 0  \n" // TLBIALL
                 "mcr p15, 0, %[zero], c7, c5, 6  \n" // BPIALL
                 "dsb                             \n"
                 "isb                             \n"
                 :
                 : [tcr] "r"(ttbcr), [ttb] "r"(ttbr0), [ctx] "r"(ctxid), [zero] "r"(0u)
                 : "memory");
    if (icache) {
        // Guest code was written as data: the instruction cache is physically tagged, so this
        // is only needed on the way in, and Sony's code on the way out has not changed.
        asm volatile("mcr p15, 0, %0, c7, c5, 0 \n dsb \n isb" ::"r"(0u) : "memory");
    }
}

static void core2_install(void) {
    // Own the core from here on: nothing of Sony's is delivered to it while our table is in.
    // The 2026-08-29 run that left interrupts open between install and the first run took a
    // prefetch abort within 0.1 s — Sony's IRQ path running under a copy of its L1 that had
    // already gone stale.
    volatile uint32_t *const pmr = gic_va ? (volatile uint32_t *)(gic_va + GICC_PMR) : NULL;
    if (pmr) {
        sv_pmr = *pmr & 0xFFu;
        *pmr = AZAHAR_PMR_MASK;
        asm volatile("dsb \n isb" ::: "memory");
    }

    sv_ttbcr = rd_ttbcr();
    sv_ttbr0 = rd_ttbr0();
    sv_ctxid = rd_contextidr();
    sv_vbar = rd_vbar();
    sv_tpidruro = rd_tpidruro();
    sv_tpidrurw = rd_tpidrurw();
    // The VFP, for the guest and for entry.S's own vldm/vstm: cp10/cp11 open at both privilege
    // levels, FPEXC.EN set. Sony's threads are lazily given the VFP on first use; this core is
    // no longer Sony's, so it is simply on.
    sv_cpacr = rd_cpacr();
    wr_cpacr(sv_cpacr | (0xFu << 20));
    sv_fpexc = rd_fpexc_here();
    wr_fpexc(sv_fpexc | (1u << 30));
    // The whole register bank and FPSCR belong to whichever Sony thread lazily owned the VFP
    // on this core, and the guest is about to clobber both. Hand back at uninstall exactly
    // what was here: a guest FPSCR with exception bits left live panicked Sony's first VFP
    // operation after release ("VFP/NEON exception occurred in kernel thread", 2026-08-30).
    sv_fpscr = rd_fpscr_here();
    vfp_bank_save(sv_vfp_bank);
    sv_sp_und = azahar_get_banked_sp(MODE_UND);
    sv_sp_abt = azahar_get_banked_sp(MODE_ABT);
    sv_sp_svc = azahar_get_banked_sp(MODE_SVC);
    sv_sp_irq = azahar_get_banked_sp(MODE_IRQ);
    azahar_chain_irq[0] = sv_sp_irq;
    azahar_gic_va = gic_va;
    azahar_seal_flag = AZAHAR_SEAL;

    // The private timer: enable its PPI at this core's banked GIC enable register (measured
    // disabled on every core), and start it free-running with no interrupt so its clock can be
    // measured against the system clock from the idle loop's samples.
    if (pmr) {
        volatile uint32_t *const isen = (volatile uint32_t *)(gic_va + GICD_ISENABLER0);
        sv_isenabler_timer = (*isen >> PTIMER_PPI) & 1u;
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
        *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
        *(volatile uint32_t *)(gic_va + PTIMER_LOAD) = 0xFFFFFFFFu;
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = PTIMER_CTRL_ENABLE | PTIMER_CTRL_RELOAD;
        *isen = 1u << PTIMER_PPI;
        asm volatile("dsb \n isb" ::: "memory");
    }

    // The stray-exception chains: this mode's SP and Sony's handler, decoded from Sony's own
    // vector instruction (azahar_sony_target, at map time).
    azahar_chain_und[0] = sv_sp_und;
    azahar_chain_und[1] = azahar_sony[1];
    azahar_chain_svc[0] = sv_sp_svc;
    azahar_chain_svc[1] = azahar_sony[2];
    azahar_chain_pabt[0] = sv_sp_abt;
    azahar_chain_pabt[1] = azahar_sony[3];
    azahar_chain_dabt[0] = sv_sp_abt;
    azahar_chain_dabt[1] = azahar_sony[4];
    azahar_guest_active = 0;
    azahar_stray_retry = 0;
    for (unsigned i = 0; i < 8; i++) {
        azahar_stray[i] = 0;
    }
    asm volatile("dsb" ::: "memory");

    // N = 0: the table base is bits [31:14].
    const uint32_t ours = ((uint32_t)azahar_l1_pa & 0xFFFFC000u) | (sv_ttbr0 & 0x3FFFu);
    core2_switch(sv_ttbcr & ~7u, ours, sv_ctxid, 1);
    azahar_ttbcr_seen = rd_ttbcr();

    // Our vectors, executable only because our table says so.
    asm volatile("mcr p15, 0, %0, c12, c0, 0 \n isb" ::"r"(azahar_vec_va) : "memory");
    azahar_installed = 1;
    azahar_cmd_status = 0;
}

static void core2_uninstall(void) {
    azahar_installed = 0;
    asm volatile("mcr p15, 0, %0, c12, c0, 0 \n isb" ::"r"(sv_vbar) : "memory");
    core2_switch(sv_ttbcr, sv_ttbr0, sv_ctxid, 0);
    azahar_set_banked_sp(MODE_UND, sv_sp_und);
    azahar_set_banked_sp(MODE_ABT, sv_sp_abt);
    azahar_set_banked_sp(MODE_SVC, sv_sp_svc);
    azahar_set_banked_sp(MODE_IRQ, sv_sp_irq);
    wr_tpidruro(sv_tpidruro);
    wr_tpidrurw(sv_tpidrurw);
    // Bank and FPSCR back before FPEXC/CPACR close the door (and before the PMR reopens
    // interrupts: Sony's IRQ path must never see the guest's VFP state).
    vfp_bank_load(sv_vfp_bank);
    wr_fpscr(sv_fpscr);
    wr_fpexc(sv_fpexc);
    wr_cpacr(sv_cpacr);
    volatile uint32_t *const pmr = gic_va ? (volatile uint32_t *)(gic_va + GICC_PMR) : NULL;
    if (pmr) {
        // Timer off and its PPI back to disabled before the mask lifts.
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
        *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
        if (!sv_isenabler_timer) {
            *(volatile uint32_t *)(gic_va + GICD_ICENABLER0) = 1u << PTIMER_PPI;
        }
        asm volatile("dsb \n isb" ::: "memory");
        *pmr = sv_pmr;
        asm volatile("dsb \n isb" ::: "memory");
    }
    azahar_cmd_status = 0;
}

// The kernel's cross-core calls — Threadmgr on SGI 2/7, Processmgr on SGI 6 (FINDINGS,
// 2026-08-17) — broadcast to every core, and the sender spins until each core's handler has
// run. With ICCPMR at AZAHAR_PMR_MASK a taken core never takes them, so the sender spins forever
// holding whatever lock it holds, and the whole console freezes. Hardware showed this the
// first time a system process spawned mid-title (2026-09-01): module loading does an all-core
// icache rendezvous, and the burst of module starts ended in a full-device freeze. Steady-state
// gameplay never triggers it — the 2026-08-29 count saw no foreign SGI at all — which is why
// every earlier hold survived.
//
// The answer is to service the SGI rather than emulate its protocol: put Sony's world back for
// a moment (core2_uninstall restores the PMR last, so the pending SGI vectors straight into
// Sony's own handler, which does the flush and acks the barrier), then take the core again.
// The scheduler still dispatches nothing here — the active-mask bit stays clear, and STAGE_HOLD
// ran 180 s that way with interrupts open — so the resident thread resumes the instant the
// handler returns. The TLBIALL/ICIALLU each way is the price of the round trip; a window opens
// only when an SGI actually pends. Latency to the answer is bounded by one guest slice
// (16 ms cap, sub-millisecond in practice with the SVC exit rate).
static void core2_sgi_window(void) {
    core2_uninstall();
    // The interrupt delivers within a few cycles of the PMR write above. Stay open while
    // pending bits keep appearing — a process start loads many modules back to back — bounded
    // in case a bit sticks (a pending-but-disabled SGI would otherwise pin the loop).
    uint32_t spins = 0;
    while (spins < 100000u) {
        const uint32_t pend = *(volatile uint32_t *)(gic_va + GICD_ISPENDR0) &
                              *(volatile uint32_t *)(gic_va + GICD_ISENABLER0) & 0x7FFFu;
        if (!pend) {
            break;
        }
        spins++;
    }
    if (spins >= 100000u) {
        azahar_sgi_stuck++;
    }
    core2_install();
    azahar_sgi_windows++;
}

// One trip into the guest. Interrupts are masked at the GIC for its duration — the core is
// Program core 2's PMU for a slice: the six A9 event counters get the events AzaharPmuStats
// documents, overflow interrupts stay off (nothing may interrupt the detached core but the
// preemption timer), and PMCR.E|P|C starts everything from zero. Resetting PMCCNTR is fine
// here, unlike in the surveys: the core is ours, and Sony's kernel never reads its counters
// again (azaharRelease parks it for good).
static void pmu_slice_start(void) {
    for (uint32_t i = 0; i < 6; i++) {
        asm volatile("mcr p15, 0, %0, c9, c12, 5" ::"r"(i));                  // PMSELR
        asm volatile("mcr p15, 0, %0, c9, c13, 1" ::"r"(azahar_pmu_events[i]));  // PMXEVTYPER
    }
    asm volatile("mcr p15, 0, %0, c9, c14, 2" ::"r"(0xFFFFFFFFu));  // PMINTENCLR
    asm volatile("mcr p15, 0, %0, c9, c12, 1" ::"r"(0x8000003Fu));  // PMCNTENSET: cycle + 0-5
    asm volatile("mcr p15, 0, %0, c9, c12, 0" ::"r"(0x7u));         // PMCR E|P|C
    asm volatile("isb" ::: "memory");
}

static void pmu_slice_end(void) {
    uint32_t v;
    asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(v));  // PMCCNTR
    azahar_pmu_cycles += v;
    for (uint32_t i = 0; i < 6; i++) {
        asm volatile("mcr p15, 0, %0, c9, c12, 5" ::"r"(i));   // PMSELR
        asm volatile("mrc p15, 0, %0, c9, c13, 2" : "=r"(v));  // PMXEVCNTR
        azahar_pmu_ev[i] += v;
    }
    azahar_pmu_runs++;
}

// owned, not shared, while guest code runs — and put back afterwards.
static void core2_run(void) {
    volatile uint32_t *const pmr = gic_va ? (volatile uint32_t *)(gic_va + GICC_PMR) : NULL;
    azahar_progress = 1;
    azahar_prog_und = azahar_prog_svc = azahar_prog_pabt = azahar_prog_dabt = azahar_prog_irq = 0;
    if (pmr) {
        azahar_run_pmr_before = *pmr & 0xFFu; // masked since install
    }

    if (azahar_tlb_dirty) {
        // Descriptors changed under us since the last run: drop every stale translation,
        // predicted branch and cached instruction before the guest sees the new map.
        asm volatile("dsb                            \n"
                     "mcr p15, 0, %0, c8, c7, 0      \n" // TLBIALL
                     "mcr p15, 0, %0, c7, c5, 6      \n" // BPIALL
                     "mcr p15, 0, %0, c7, c5, 0      \n" // ICIALLU
                     "dsb                            \n"
                     "isb                            \n"
                     :
                     : "r"(0u)
                     : "memory");
        azahar_tlb_dirty = 0;
    }
    azahar_progress = 2;
    wr_tpidruro(azahar_ctx.tpidruro);
    wr_tpidrurw(azahar_ctx.tpidrurw);
    if (pmr && azahar_run_quantum_ticks) {
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
        *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
        *(volatile uint32_t *)(gic_va + PTIMER_LOAD) = azahar_run_quantum_ticks;
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) =
            PTIMER_CTRL_ENABLE | PTIMER_CTRL_IRQ; // one shot: no reload
        asm volatile("dsb \n isb" ::: "memory");
    }
    azahar_progress = 3;
    pmu_slice_start();
    const uint32_t reason = (uint32_t)azahar_enter(&azahar_ctx, azahar_host);
    azahar_progress = 42;
    pmu_slice_end();
    if (pmr) {
        azahar_run_timer_left = *(volatile uint32_t *)(gic_va + PTIMER_COUNTER);
        *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
        *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
        asm volatile("dsb \n isb" ::: "memory");
    }
    azahar_ctx.tpidruro = rd_tpidruro();
    azahar_ctx.tpidrurw = rd_tpidrurw();

    if (pmr) {
        azahar_run_pmr_after = *pmr & 0xFFu;
    }

    // entry.S left the raw exception LR in ctx.r[15] and the SPSR in ctx.cpsr. Turn the LR into
    // the address the reason means (vanative.h), and pull the svc immediate out of the
    // instruction — readable here because our table is still installed and the code window is
    // readable at PL1.
    const uint32_t thumb = azahar_ctx.cpsr & (1u << 5);
    const uint32_t raw_lr = azahar_ctx.r[15];
    uint32_t pc = raw_lr, svc = 0;
    switch (reason) {
        case AZAHAR_EXIT_SVC:
            pc = raw_lr; // the instruction after the svc
            if (thumb) {
                svc = *(const volatile uint16_t *)(uintptr_t)(raw_lr - 2) & 0xFFu;
            } else {
                svc = *(const volatile uint32_t *)(uintptr_t)(raw_lr - 4) & 0x00FFFFFFu;
            }
            break;
        case AZAHAR_EXIT_UND: pc = raw_lr - (thumb ? 2u : 4u); break;
        case AZAHAR_EXIT_PABT: pc = raw_lr - 4u; break;
        case AZAHAR_EXIT_DABT: pc = raw_lr - 8u; break;
        case AZAHAR_EXIT_TIMER: pc = raw_lr - 4u; break; // IRQ: LR is the next instruction + 4
        default: break;
    }
    azahar_ctx.r[15] = pc;
    azahar_run_reason = reason;
    azahar_run_svc = svc;
    azahar_run_rawlr = raw_lr;
    azahar_run_spsr = azahar_exit_info[2];
    azahar_run_fsr = (reason == AZAHAR_EXIT_DABT) ? azahar_exit_info[3]
                  : (reason == AZAHAR_EXIT_PABT) ? azahar_exit_info[5] : 0u;
    azahar_run_far = (reason == AZAHAR_EXIT_DABT) ? azahar_exit_info[4]
                  : (reason == AZAHAR_EXIT_PABT) ? azahar_exit_info[6] : 0u;
    azahar_progress = 50;
    azahar_cmd_status = 0;
}

// Folded mode: one slice on the calling core. The same install, run and uninstall the
// resident thread does once at map, once per command and once at release, done here per
// slice around azahar_enter. The PMR mask holds from install to uninstall, so nothing of Sony's
// runs on this core in between; the moment the mask lifts, anything that pended (the tick,
// a cross-core call) is delivered into Sony's own handler, which replaces the SGI window.
// The syscall's kernel stack is the host stack azahar_enter saves and the vectors come back to.
static void fold_run(void) {
    core2_install();
    core2_run();
    core2_uninstall();
}

// Folded mode's timer calibration: the private timer of this core, free-running against the
// system clock for AZAHAR_TIMER_MEASURE_US. The thread is pinned, so the counter read after the
// delay is the same core's.
static void fold_measure_timer(void) {
    azahar_timer_hz = 0;
    if (!gic_va) {
        return;
    }
    volatile uint32_t *const isen = (volatile uint32_t *)(gic_va + GICD_ISENABLER0);
    const uint32_t was_enabled = (*isen >> PTIMER_PPI) & 1u;
    *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
    *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
    *(volatile uint32_t *)(gic_va + PTIMER_LOAD) = 0xFFFFFFFFu;
    *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = PTIMER_CTRL_ENABLE | PTIMER_CTRL_RELOAD;
    asm volatile("dsb \n isb" ::: "memory");
    const uint32_t s0 = *(volatile uint32_t *)(gic_va + PTIMER_COUNTER);
    const uint32_t t0 = ksceKernelGetSystemTimeLow();
    ksceKernelDelayThread(AZAHAR_TIMER_MEASURE_US);
    const uint32_t s1 = *(volatile uint32_t *)(gic_va + PTIMER_COUNTER);
    const uint32_t t1 = ksceKernelGetSystemTimeLow();
    *(volatile uint32_t *)(gic_va + PTIMER_CONTROL) = 0;
    *(volatile uint32_t *)(gic_va + PTIMER_STATUS) = 1;
    if (!was_enabled) {
        *(volatile uint32_t *)(gic_va + GICD_ICENABLER0) = 1u << PTIMER_PPI;
    }
    asm volatile("dsb \n isb" ::: "memory");
    const uint32_t ticks = s0 - s1, us = t1 - t0; // counts down
    if (us) {
        azahar_timer_hz = (uint32_t)(((uint64_t)ticks * 1000000ull) / us);
    }
    emit("  private timer (folded): %u ticks in %u us -> %u Hz\n", ticks, us, azahar_timer_hz);
}

static int core2_entry(SceSize args, void *argp) {
    (void)args;
    (void)argp;
    azahar_on_core = (uint32_t)ksceKernelCpuId() & 3;
    azahar_core2_vbar = rd_vbar();
    azahar_core2_cpsr = rd_cpsr();
    azahar_core2_ttbr1 = rd_ttbr1();
    {
        uint32_t sp;
        asm volatile("mov %0, sp" : "=r"(sp));
        azahar_core2_sp = sp;
    }

    int pmu_on = 0;
    PmuSave pmu;
    while (!azahar_stop) {
        azahar_heartbeat++;
        if (azahar_park) {
            // PMCCNTR stops in WFI, so its delta across the park is awake cycles: ~250M per
            // 0.5 s means spinning, a few thousand means sleeping between interrupts. The
            // kernel's own idle accounting cannot see this — it counts Sony's idle thread,
            // which never runs on a held core — so any overlay will read 100% regardless.
            if (!pmu_on) {
                pmu_begin(&pmu);
                pmu_on = 1;
            }
            // WFE, not WFI: a taken core sees no interrupt for seconds at a time, and the
            // next azaharTakeCore has to be able to wake this. SEV from any core does.
            asm volatile("wfe" ::: "memory");
            azahar_park_cycles = rd_pmccntr();
            continue;
        }
        if (azahar_gic_va) {
            azahar_timer_sample = *(volatile uint32_t *)(azahar_gic_va + PTIMER_COUNTER);
        }
        // An SGI pending against this core while our world is installed is a cross-core call
        // the kernel is spinning on (core2_sgi_window). Checked between commands only — during
        // a slice the guest owns the core and the pend bit simply waits for the next exit.
        if (azahar_installed && gic_va) {
            const uint32_t pend = *(volatile uint32_t *)(gic_va + GICD_ISPENDR0) &
                                  *(volatile uint32_t *)(gic_va + GICD_ISENABLER0) & 0x7FFFu;
            if (pend) {
                azahar_sgi_last_pend = pend;
                core2_sgi_window();
            }
        }
        const uint32_t cmd = azahar_cmd;
        if (cmd == CMD_NONE || azahar_cmd_done) {
            continue;
        }
        switch (cmd) {
            case CMD_INSTALL: core2_install(); break;
            case CMD_RUN: core2_run(); break;
            case CMD_UNINSTALL: core2_uninstall(); break;
            default: azahar_cmd_status = 1; break;
        }
        // Publish the results before the done flag: without the barrier the other core can
        // observe done set while the status (and the exit context) are still the old values,
        // which read as a failed command once in a while.
        asm volatile("dsb" ::: "memory");
        azahar_cmd_done = 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------- syscall side

static void report_stray(void) {
    if (!azahar_stray[0]) {
        return;
    }
    static const char *const names[] = {"?", "svc", "undefined", "prefetch abort", "data abort"};
    emit("  STRAY exception(s) on core 2 with no guest active: %u, first %s lr %08x spsr %08x "
         "dfsr %08x dfar %08x ifsr %08x ifar %08x (%u retried store(s))\n",
         azahar_stray[0], azahar_stray[1] < 5 ? names[azahar_stray[1]] : "?", azahar_stray[2], azahar_stray[3],
         azahar_stray[4], azahar_stray[5], azahar_stray[6], azahar_stray[7], azahar_stray_retry);
}

static int core2_command(uint32_t cmd, unsigned steps) {
    azahar_cmd_done = 0;
    azahar_cmd_status = 0xFFFFFFFFu;
    asm volatile("dsb" ::: "memory");
    azahar_cmd = cmd;
    // A guest exit is microseconds away most of the time — an svc round trip measured ~1.5 us
    // — and the emulator waits here on every one of them, so the wait spins first and only
    // sleeps once it is clear the guest is running for real. Sleeping in 1 ms steps after that
    // keeps a preemption quantum's worth of latency bounded.
    int ret = AZAHAR_ERR_TIMEOUT;
    for (uint32_t spin = 0; spin < 400000u && !azahar_cmd_done; spin++) {
        asm volatile("" ::: "memory");
    }
    for (unsigned i = 0; i < steps * (AZAHAR_WAIT_STEP_US / 1000u); i++) {
        if (azahar_cmd_done) {
            break;
        }
        ksceKernelDelayThread(1000);
    }
    if (azahar_cmd_done) {
        // Pair with the worker's dsb: nothing written before its done flag may be read ahead
        // of the flag itself.
        asm volatile("dmb" ::: "memory");
        azahar_cmd = CMD_NONE;
        ret = (int)azahar_cmd_status == 0 ? 0 : AZAHAR_ERR_CORE;
    }
    if (ret == AZAHAR_ERR_TIMEOUT) {
        emit("  command %u did not complete (heartbeat %u)\n", cmd, azahar_heartbeat);
    }
    report_stray();
    return ret;
}

static int alloc_block(const char *name, SceSize size, SceSize align, SceUID *uid, uint32_t **va,
                       uintptr_t *pa) {
    SceKernelAllocMemBlockKernelOpt opt;
    __builtin_memset(&opt, 0, sizeof(opt));
    opt.size = sizeof(opt);
    opt.attr = 0x4;
    opt.alignment = align;
    *uid = ksceKernelAllocMemBlock(name, SCE_KERNEL_MEMBLOCK_TYPE_KERNEL_ROOT_PHYCONT_RW, size, &opt);
    if (*uid < 0) {
        emit("  %s: alloc failed 0x%08x\n", name, (uint32_t)*uid);
        return AZAHAR_ERR_ALLOC;
    }
    if (ksceKernelGetMemBlockBase(*uid, (void **)va) < 0 || ksceKernelVAtoPA(*va, pa) < 0) {
        emit("  %s: base/pa lookup failed\n", name);
        return AZAHAR_ERR_ALLOC;
    }
    return 0;
}

static void free_blocks(void) {
    if (azahar_vec_uid >= 0) { ksceKernelFreeMemBlock(azahar_vec_uid); azahar_vec_uid = -1; }
    if (azahar_l2_uid >= 0) { ksceKernelFreeMemBlock(azahar_l2_uid); azahar_l2_uid = -1; }
    if (azahar_l1_uid >= 0) { ksceKernelFreeMemBlock(azahar_l1_uid); azahar_l1_uid = -1; }
}

// Find the kernel virtual address at which physical page `pa` is currently mapped, by asking
// the MMU about each page of the low 8 MB — where the kernel keeps its translation tables, at
// a different virtual address every boot.
static const uint32_t *find_kernel_page(uint32_t pa) {
    for (uint32_t va = 0; va < (8u << 20); va += 0x1000) {
        const uint32_t par = translate(va, AT_PRIV_R);
        if (!PAR_FAULTED(par) && PAR_PA(par) == pa) {
            return (const uint32_t *)(uintptr_t)va;
        }
    }
    return NULL;
}

// One page of the app's memory: its physical address, and its contents pushed to memory so core
// 2 can fetch it as instructions through a different mapping.
//
// The app's pages live in a domain the kernel holds at No-access while a syscall runs (that is
// why kernel code copies user memory through ksceKernelCopyFromUser), so from here ATS1CUR
// answers with a domain fault — PAR 00000017, measured 2026-08-29 — and a by-VA cache clean
// would fault the same way, inside the syscall. ksceKernelVAtoPA is tried first. Failing that,
// every No-access domain is granted Client for one ATS and one page clean, with interrupts
// suspended: DACR is process context, and a reschedule mid-window would put the old value back
// under the clean. Grant-only, never revoke — the direction STAGE_DOMAINS proved safe.
static uint32_t dacr_grant_all(uint32_t d) {
    uint32_t g = d;
    for (unsigned k = 0; k < 16; k++) {
        if (((d >> (2 * k)) & 3u) == 0) {
            g |= 1u << (2 * k);
        }
    }
    return g;
}

static int azahar_user_page(uint32_t uva, uint32_t *pa, int *how, uint32_t *par_out) {
    uintptr_t p = 0;
    uint32_t par = 0;
    if (ksceKernelVAtoPA((const void *)(uintptr_t)uva, &p) >= 0 && p) {
        *pa = (uint32_t)p & 0xFFFFF000u;
        *how = 0;
    } else {
        SceKernelIntrStatus st = ksceKernelCpuSuspendIntr();
        const uint32_t d = rd_dacr();
        wr_dacr(dacr_grant_all(d));
        asm volatile("mcr p15, 0, %0, c7, c8, 2 \n isb" ::"r"(uva) : "memory"); // ATS1CUR
        par = rd_par();
        wr_dacr(d);
        ksceKernelCpuResumeIntr(st);
        *par_out = par;
        if (PAR_FAULTED(par)) {
            return AZAHAR_ERR_PAGE;
        }
        *pa = PAR_PA(par);
        *how = 1;
    }
    // Clean the page to the point of unification under the same grant.
    SceKernelIntrStatus st = ksceKernelCpuSuspendIntr();
    const uint32_t d = rd_dacr();
    wr_dacr(dacr_grant_all(d));
    ksceKernelDcacheCleanRange((void *)(uintptr_t)uva, 0x1000);
    wr_dacr(d);
    ksceKernelCpuResumeIntr(st);
    return 0;
}

// Ends the resident thread and gives core 2 back to the scheduler, in the only order measured
// to work: the thread is stopped and waited for first, so the scheduler receives an idle core
// rather than an occupied one (restoring the mask under a live occupant froze the console,
// 2026-08-29). Returns 0 when the core was handed back, AZAHAR_ERR_CORE when the thread would not
// end - in which case the core stays detached, because a wedged console is worse than a lost
// core. Callers must have uninstalled first if the state was ST_MAPPED.
// Core 2 is held without taking it out of the scheduler's active mask, and that is the
// default (proven on hardware 2026-09-04). Interrupts alone hold it: core2_install drops
// ICCPMR to AZAHAR_PMR_MASK, which masks the rescheduling SGIs, and the private timer's
// interrupt is off, so the scheduler may assign a thread to core 2 but has no way to make it
// run there, and preemption needs an interrupt. Cross-core calls, which freeze the console
// when a held core never answers, are serviced in core2_sgi_window.
//
// What it buys is the teardown. The mask is never changed, so the one-way reattach (FINDINGS,
// "The reattach is one-way") cannot happen: the resident thread ends like any other thread and
// core 2 goes straight back to the scheduler, available to the shell and to every other app.
// Detaching could never give the core back - the restore froze the whole console under a live
// occupant (2026-08-29) and again after a clean thread end (2026-09-04) - so a detached
// session has to park the thread and keep the core until the next reboot.
//
// The detach is kept behind ux0:data/azahar/detach, for the case where a title turns out to
// need core 2 truly invisible to Sony's scheduler. Delete the file to go back to the default.
static int azahar_detach_requested(void) {
    const SceUID fd = ksceIoOpen(OUT_DIR "/detach", SCE_O_RDONLY, 0);
    if (fd < 0) {
        return 0;
    }
    ksceIoClose(fd);
    return 1;
}

// End the resident thread and, if this session took the core out of the mask, put the mask
// back. Only a detached session has a second half, and only once its client is gone: see
// azaharRelease.
static int azahar_stop_resident(void) {
    azahar_park = 0;
    azahar_stop = 1;
    asm volatile("dsb \n sev" ::: "memory");
    SceUInt timeout = 1000000;
    const int ended = ksceKernelWaitThreadEnd(azahar_thread, NULL, &timeout);
    ksceKernelDeleteThread(azahar_thread);
    azahar_thread = -1;
    if (ended < 0) {
        emit("  resident thread did not end (0x%08x); core %u stays detached\n",
             (uint32_t)ended, AZAHAR_TARGET_CORE);
        return AZAHAR_ERR_CORE;
    }
    if (!azahar_detached) {
        emit("  thread ended; the mask was never changed, core %u is the scheduler's again\n",
             AZAHAR_TARGET_CORE);
        return 0;
    }
    const int r = p_ChangeActiveCpuMask((int)azahar_saved_mask);
    emit("  thread ended; ChangeActiveCpuMask(%08x) -> 0x%08x\n", azahar_saved_mask, (uint32_t)r);
    azahar_detached = 0;
    return 0;
}

// The client process is gone: same teardown as azaharRelease, which the two paths share below.
static int azahar_on_proc_gone(SceUID pid) {
    if (pid != azahar_client_pid || azahar_state == ST_NONE) {
        return 0;
    }
    log_open();
    emit("== client process gone (state %d)\n", azahar_state);
    pmu_host_stop_all();
    if (azahar_folded) {
        free_blocks();
        azahar_state = ST_NONE;
        azahar_client_pid = -1;
        log_release();
        return 0;
    }
    if (azahar_state == ST_MAPPED) {
        const int r = core2_command(CMD_UNINSTALL, AZAHAR_WAIT_STEPS);
        emit("  uninstall -> %d\n", r);
    }
    if (!azahar_detached) {
        // Nothing was taken from the scheduler, so nothing has to be given back: the resident
        // thread ends like any other thread and core 2 is a normal core again the moment it
        // does. This is the whole point of holding the core without detaching it.
        const int ended = azahar_stop_resident();
        free_blocks();
        azahar_state = ended == 0 ? ST_NONE : ST_RELEASED;
    } else {
        azahar_park = 1;
        free_blocks();
        azahar_state = ST_RELEASED;
    }
    azahar_client_pid = -1;
    log_release();
    return 0;
}

static int azahar_proc_exit(SceUID pid, SceProcEventInvokeParam1 *p, int unk) {
    (void)p; (void)unk;
    return azahar_on_proc_gone(pid);
}

static int azahar_proc_kill(SceUID pid, SceProcEventInvokeParam1 *p, int unk) {
    (void)p; (void)unk;
    return azahar_on_proc_gone(pid);
}

static const SceProcEventHandler azahar_proc_events = {
    .size = sizeof(SceProcEventHandler),
    .create = NULL,
    .exit = azahar_proc_exit,
    .kill = azahar_proc_kill,
    .stop = NULL,
    .start = NULL,
    .switch_process = NULL,
};

int azaharTakeCore(void) {
    uint32_t state;
    ENTER_SYSCALL(state);
    log_open();
    emit("== azaharTakeCore\n");
    int ret = 0;

    if (azahar_state != ST_NONE && azahar_state != ST_RELEASED) {
        ret = AZAHAR_ERR_STATE;
        goto out;
    }
    if (!resolve_exports()) {
        ret = AZAHAR_ERR_RESOLVE;
        goto out;
    }
    azahar_gic();
    azahar_folded = !azahar_resident_requested();
    if (azahar_folded) {
        if (azahar_state == ST_RELEASED) {
            // A parked resident thread from a session that ran in the other mode.
            ret = AZAHAR_ERR_STATE;
            goto out;
        }
        if ((ret = alloc_block("azahar_l1", AZAHAR_L1_ENTRIES * 4, 0x4000, &azahar_l1_uid, &azahar_l1, &azahar_l1_pa)) ||
            (ret = alloc_block("azahar_l2", AZAHAR_L2_COUNT * 0x400, 0x1000, &azahar_l2_uid, &azahar_l2, &azahar_l2_pa)) ||
            (ret = alloc_block("azahar_vec", 0x100000, 0x100000, &azahar_vec_uid, &azahar_vec, &azahar_vec_pa))) {
            free_blocks();
            goto out;
        }
        azahar_detached = 0;
        azahar_installed = 0;
        azahar_client_pid = ksceKernelGetProcessId();
        if (azahar_proc_handler < 0) {
            azahar_proc_handler = ksceKernelRegisterProcEventHandler("vanative", &azahar_proc_events, 0);
            emit("  process-event handler: 0x%08x\n", (uint32_t)azahar_proc_handler);
        }
        emit("  folded: no resident thread, the caller's core runs the guest\n");
        azahar_state = ST_TAKEN;
        goto out;
    }

    if (azahar_state == ST_RELEASED) {
        // A previous title released the core without handing it back (azaharRelease): the
        // scheduler still excludes it, and the resident thread is idling on it. Prove the
        // thread is alive and take it over as it stands.
        if (azahar_thread < 0) {
            ret = AZAHAR_ERR_STATE;
            goto out;
        }
        const uint32_t beat = azahar_heartbeat;
        azahar_park = 0;
        asm volatile("dsb \n sev" ::: "memory");
        unsigned w;
        for (w = 0; w < 20 && azahar_heartbeat == beat; w++) {
            ksceKernelDelayThread(10000);
        }
        if (azahar_heartbeat == beat) {
            emit("  resident thread is not running on the released core\n");
            ret = AZAHAR_ERR_CORE;
            goto out;
        }
        if ((ret = alloc_block("azahar_l1", AZAHAR_L1_ENTRIES * 4, 0x4000, &azahar_l1_uid, &azahar_l1, &azahar_l1_pa)) ||
            (ret = alloc_block("azahar_l2", AZAHAR_L2_COUNT * 0x400, 0x1000, &azahar_l2_uid, &azahar_l2, &azahar_l2_pa)) ||
            (ret = alloc_block("azahar_vec", 0x100000, 0x100000, &azahar_vec_uid, &azahar_vec, &azahar_vec_pa))) {
            free_blocks();
            goto out;
        }
        azahar_cmd = azahar_cmd_done = 0;
        emit("  reusing the detached core: heartbeat %u -> %u\n", beat, azahar_heartbeat);
        azahar_state = ST_TAKEN;
        goto out;
    }

    if ((ret = alloc_block("azahar_l1", AZAHAR_L1_ENTRIES * 4, 0x4000, &azahar_l1_uid, &azahar_l1, &azahar_l1_pa)) ||
        (ret = alloc_block("azahar_l2", AZAHAR_L2_COUNT * 0x400, 0x1000, &azahar_l2_uid, &azahar_l2, &azahar_l2_pa)) ||
        (ret = alloc_block("azahar_vec", 0x100000, 0x100000, &azahar_vec_uid, &azahar_vec, &azahar_vec_pa))) {
        free_blocks();
        goto out;
    }

    // Occupy first, detach second: a thread pinned to an already-detached core never runs.
    azahar_heartbeat = azahar_stop = azahar_park = azahar_cmd = azahar_cmd_done = 0;
    azahar_thread = ksceKernelCreateThread("vanative_core2", core2_entry, 0x10000100, 0x4000, 0,
                                        CPU_AFFINITY(AZAHAR_TARGET_CORE), NULL);
    if (azahar_thread < 0 || ksceKernelStartThread(azahar_thread, 0, NULL) < 0) {
        emit("  core-2 thread create/start failed 0x%08x\n", (uint32_t)azahar_thread);
        free_blocks();
        ret = AZAHAR_ERR_CORE;
        goto out;
    }
    unsigned i;
    for (i = 0; i < AZAHAR_WAIT_STEPS && !azahar_heartbeat; i++) {
        ksceKernelDelayThread(AZAHAR_WAIT_STEP_US);
    }
    if (!azahar_heartbeat) {
        emit("  core-2 thread never ran\n");
        azahar_stop = 1;
        free_blocks();
        ret = AZAHAR_ERR_CORE;
        goto out;
    }
    emit("  resident thread on cpu%u, cpsr %08x (mode %02x), vbar %08x, ttbr1 %08x (syscall's %08x)\n",
         azahar_on_core, azahar_core2_cpsr, azahar_core2_cpsr & 0x1Fu, azahar_core2_vbar, azahar_core2_ttbr1,
         rd_ttbr1());

    uint32_t mask = active_cpu_mask();
    if (!mask) {
        mask = 0xFu;
        emit("  mask unreadable; assuming %08x\n", mask);
    }
    const uint32_t beat = azahar_heartbeat;
    azahar_detached = 0;
    if (azahar_detach_requested()) {
        const uint32_t detached = mask & ~(1u << AZAHAR_TARGET_CORE);
        emit("  DETACH: ChangeActiveCpuMask(%08x) -> 0x%08x  (mask was %08x)\n", detached,
             (uint32_t)p_ChangeActiveCpuMask((int)detached), mask);
        azahar_detached = 1;
    } else {
        emit("  core %u stays in the scheduler's mask (%08x)\n", AZAHAR_TARGET_CORE, mask);
    }
    // The same gate either way: 250 ms later the resident thread must still be ticking. A
    // detach that stopped it means the core went away under us; without one it means the
    // scheduler took the core back.
    ksceKernelDelayThread(250000);
    if (azahar_heartbeat == beat) {
        emit("  resident thread STOPPED (detached=%d); restoring the mask\n", azahar_detached);
        if (azahar_detached) {
            p_ChangeActiveCpuMask((int)mask);
            azahar_detached = 0;
        }
        azahar_stop = 1;
        free_blocks();
        ret = AZAHAR_ERR_CORE;
        goto out;
    }
    emit("  heartbeat %u -> %u: core %u is ours\n", beat, azahar_heartbeat, AZAHAR_TARGET_CORE);
    azahar_saved_mask = mask;
    azahar_client_pid = ksceKernelGetProcessId();
    if (azahar_proc_handler < 0) {
        // So the core comes back when the app dies, however it dies. Registered here rather
        // than at module_start: a fault here costs one launch, a fault there bootloops.
        azahar_proc_handler = ksceKernelRegisterProcEventHandler("vanative", &azahar_proc_events, 0);
        emit("  process-event handler: 0x%08x\n", (uint32_t)azahar_proc_handler);
    }
    azahar_state = ST_TAKEN;

out:
    emit("  -> %d\n", ret);
    log_close();
    EXIT_SYSCALL(state);
    return ret;
}

// Where Sony's vector `slot` (0..7) goes. Its vectors are `ldr pc, [pc, #imm]` for most slots
// and not for all — the 2026-08-29 run read a pointer word of 0 for Data Abort under the
// assumption they all were — so the instruction is decoded: a PC-relative ldr, or a branch.
static uint32_t azahar_sony_target(uint32_t vbar, unsigned slot) {
    const uint32_t addr = vbar + slot * 4u;
    const uint32_t w = *(const volatile uint32_t *)(uintptr_t)addr;
    if ((w & 0x0F7F0000u) == 0x051F0000u) { // ldr pc, [pc, #+/-imm12]
        const uint32_t imm = w & 0xFFFu;
        const uint32_t at = (w & (1u << 23)) ? addr + 8u + imm : addr + 8u - imm;
        return *(const volatile uint32_t *)(uintptr_t)at;
    }
    if ((w & 0x0F000000u) == 0x0A000000u) { // b
        int32_t off = (int32_t)(w << 8) >> 6;
        return addr + 8u + (uint32_t)off;
    }
    return 0;
}

// The L2 serving guest megabyte `idx`: existing, or a fresh one from the pool with its L1
// entry written. Returns the table index or -1 when the pool is exhausted.
static int l2_for(uint32_t idx, int create) {
    for (unsigned t = 0; t < azahar_l2_used; t++) {
        if (azahar_l2_slot[t] == idx) {
            return (int)t;
        }
    }
    if (!create || azahar_l2_used == AZAHAR_L2_COUNT) {
        return -1;
    }
    const unsigned t = azahar_l2_used++;
    azahar_l2_slot[t] = idx;
    for (unsigned i = 0; i < 256; i++) {
        azahar_l2[t * 256u + i] = 0;
    }
    azahar_l1[idx] = AZAHAR_COARSE((uint32_t)azahar_l2_pa + t * 0x400u);
    return (int)t;
}

// One range of the app's memory into the guest's table: every page resolved and cleaned, its
// descriptor written, the megabyte's L2 created on first use. Used at map and for MAP edits.
static int map_range(const AzaharRange *rg, unsigned label) {
    unsigned pages = 0;
    uint32_t first_pa = 0;
    int first_how = -1;
    for (uint32_t off = 0; off < rg->size; off += 0x1000) {
        const uint32_t gva = rg->guest_va + off;
        const uint32_t uva = rg->user_va + off;
        const int t = l2_for(gva >> 20, 1);
        if (t < 0) {
            emit("  out of L2 tables at guest %08x\n", gva);
            return AZAHAR_ERR_TABLE;
        }
        uint32_t pa = 0, par = 0;
        int how = -1;
        if (azahar_user_page(uva, &pa, &how, &par) != 0) {
            emit("  range %u: user va %08x does not translate (par %08x, VAtoPA refused)\n",
                 label, uva, par);
            return AZAHAR_ERR_PAGE;
        }
        if (off == 0) {
            first_pa = pa;
            first_how = how;
        }
        azahar_l2[(unsigned)t * 256u + ((gva >> 12) & 0xFFu)] = azahar_small_page(pa, rg->perm);
        pages++;
    }
    emit("  range %u: user %08x -> guest %08x, %u page(s) over %u MB, perm %c%c%c, first pa "
         "%08x (%s)\n",
         label, rg->user_va, rg->guest_va, pages,
         ((rg->guest_va + rg->size - 1) >> 20) - (rg->guest_va >> 20) + 1,
         (rg->perm & AZAHAR_PERM_R) ? 'r' : '-', (rg->perm & AZAHAR_PERM_W) ? 'w' : '-',
         (rg->perm & AZAHAR_PERM_X) ? 'x' : '-', first_pa,
         first_how == 0 ? "VAtoPA" : "ATS1CUR under a domain grant");
    return 0;
}

static int check_range(const AzaharRange *rg, int need_user) {
    if ((need_user && (rg->user_va & 0xFFFu)) || (rg->guest_va & 0xFFFu) || (rg->size & 0xFFFu) ||
        !rg->size || rg->guest_va + rg->size > 0x40000000u ||
        rg->guest_va + rg->size < rg->guest_va) {
        return 0;
    }
    return 1;
}

// Build the N=0 table: the kernel's TTBR0 entries below 0x40000000 and its TTBR1 entries above,
// as this process sees them, then our vector page in a free 1 MB slot at the top of the low
// quarter, then the guest windows as coarse L2s.
static int build_table(const AzaharMapRequest *req) {
    // TTBR0 is one global table; TTBR1 is per process, and the one core 2 needs is the resident
    // kernel thread's, not this syscall's — the first run (2026-08-29) read the app's TTBR1 here,
    // whose pages are not where the kernel keeps its own, and found nothing to merge.
    const uint32_t kernel_l1_pa = rd_ttbr0() & 0xFFFFF000u;
    const uint32_t ttbr1_pa = azahar_core2_ttbr1 & 0xFFFFC000u;
    const uint32_t *kernel_l1 = find_kernel_page(kernel_l1_pa);
    if (!kernel_l1) {
        emit("  kernel L1 pa %08x not found in the low 8 MB\n", kernel_l1_pa);
        return AZAHAR_ERR_TABLE;
    }
    const uint32_t *upper[4] = {NULL, NULL, NULL, NULL};
    for (unsigned pg = 1; pg < 4; pg++) {
        upper[pg] = find_kernel_page(ttbr1_pa + pg * 0x1000u);
        if (!upper[pg]) {
            emit("  TTBR1 page %u (pa %08x) not found\n", pg, ttbr1_pa + pg * 0x1000u);
            return AZAHAR_ERR_TABLE;
        }
    }
    // The kernel's entry for L1 index `idx`, from whichever of its two tables holds it.
#define KERNEL_L1(idx) ((idx) < 1024u ? kernel_l1[(idx)] : upper[(idx) >> 10][(idx) & 1023u])

    for (unsigned i = 0; i < AZAHAR_L1_ENTRIES; i++) {
        azahar_l1[i] = 0;
    }
#if AZAHAR_SEAL
    // What core 2 touches while our table is in, by the addresses of the things it touches:
    // this module's code and data, its stack, the GIC. Each address names one megabyte, and
    // that megabyte's entry — the kernel's own, pointing at the kernel's own L2 — is kept.
    // The module's own megabytes come from its segment table — code and data/bss both,
    // wherever this boot placed them. This replaced a by-symbol spot list: the list named
    // single objects, the module's layout differs per boot, and on 2026-08-30 the stray-IRQ
    // stack (.bss, past the last named object) landed in a megabyte the list missed — the
    // first stray IRQ's push data-aborted, and the sealed abort stub spun forever with the
    // heartbeat frozen and every command timing out. The spot list survives only as the
    // fallback when the lookup fails.
    struct KeepRange { uint32_t addr, len; };
    struct KeepRange keep[40];
    unsigned nkeep = 0;
    {
        // Segment bases from taiHEN (already imported; the module manager's own
        // ksceKernelGetModuleInfo lives in ForKernel, whose NIDs vary per firmware). taiHEN
        // gives no sizes, so each segment runs from its base to the farthest symbol this file
        // knows in it, plus slack; the base closes the front, which no symbol list can.
        tai_module_info_t tinfo;
        __builtin_memset(&tinfo, 0, sizeof(tinfo));
        tinfo.size = sizeof(tinfo);
        if (taiGetModuleInfoForKernel(0x10005 /* the kernel */, "azaharnative", &tinfo) >= 0) {
            const uint32_t text_syms[] = {
                (uint32_t)(uintptr_t)&azahar_enter,         (uint32_t)(uintptr_t)&azahar_vec_irq,
                (uint32_t)(uintptr_t)&azahar_get_banked_sp, (uint32_t)(uintptr_t)&core2_entry,
                (uint32_t)(uintptr_t)&core2_run,         (uint32_t)(uintptr_t)&core2_install,
                (uint32_t)(uintptr_t)&core2_uninstall,   (uint32_t)(uintptr_t)&translate,
            };
            const uint32_t data_syms[] = {
                (uint32_t)(uintptr_t)&azahar_ctx,       (uint32_t)(uintptr_t)azahar_host,
                (uint32_t)(uintptr_t)&azahar_progress,  (uint32_t)(uintptr_t)azahar_exit_info,
                (uint32_t)(uintptr_t)azahar_stray,      (uint32_t)(uintptr_t)&azahar_cmd,
                (uint32_t)(uintptr_t)&azahar_heartbeat, (uint32_t)(uintptr_t)&azahar_timer_sample,
                (uint32_t)(uintptr_t)&gic_va,        (uint32_t)(uintptr_t)&sv_ttbcr,
                (uint32_t)(uintptr_t)&azahar_l1_pa,     (uint32_t)(uintptr_t)&azahar_run_reason,
                (uint32_t)(uintptr_t)azahar_sony,       (uint32_t)(uintptr_t)&azahar_state,
                (uint32_t)(uintptr_t)azahar_irq_stack_top,
            };
            uint32_t bases[2] = {0, 0};
            for (unsigned seg = 0; seg < 2; seg++) {
                uintptr_t base = 0;
                if (module_get_offset(0x10005, tinfo.modid, (int)seg, 0, &base) >= 0) {
                    bases[seg] = (uint32_t)base;
                } else {
                    emit("  seal: segment %u base lookup failed\n", seg);
                }
            }
            // A symbol belongs to whichever segment starts at or below it, nearest. The two
            // lists only say what this file *thinks* is text or data, and the compiler is
            // free to disagree (.rodata lives in the text segment, for one); bucketing by
            // list stretched a segment's range across the gap to the other, into megabytes
            // the kernel never mapped, and the install failed on them (2026-08-30, second
            // commercial-title run).
            const unsigned n_text = sizeof(text_syms) / sizeof(text_syms[0]);
            const unsigned n_data = sizeof(data_syms) / sizeof(data_syms[0]);
            int bucket[32];
            uint32_t ends[2] = {bases[0], bases[1]};
            for (unsigned i = 0; i < n_text + n_data; i++) {
                const uint32_t sym = i < n_text ? text_syms[i] : data_syms[i - n_text];
                int best = -1;
                for (unsigned seg = 0; seg < 2; seg++) {
                    if (bases[seg] != 0 && sym >= bases[seg] &&
                        (best < 0 || bases[seg] > bases[best])) {
                        best = (int)seg;
                    }
                }
                bucket[i] = best;
                if (best >= 0 && sym > ends[best]) {
                    ends[best] = sym;
                }
            }
            // A range longer than any plausible module segment means the bucketing went
            // wrong: when one segment's base lookup fails, its symbols fall into the other
            // segment's bucket and stretch that range across the gap between the two -
            // hundreds of unmapped megabytes' worth on 2026-08-30 (third commercial-title
            // run: text near 0x01c-, data near 0x810-, one skip line per megabyte between).
            // Such a range is discarded and its symbols are kept one megabyte at a time,
            // like the spot-list fallback.
            for (unsigned seg = 0; seg < 2; seg++) {
                if (bases[seg] == 0) {
                    continue;
                }
                const uint32_t len = (ends[seg] - bases[seg]) + 0x1000u;
                if (len > 0x01000000u) {
                    emit("  seal: segment %u range %08x..%08x implausible, keeping its "
                         "symbols one by one\n", seg, bases[seg], bases[seg] + len);
                    bases[seg] = 0; /* rejected: its symbols fall through to spots below */
                    continue;
                }
                keep[nkeep].addr = bases[seg];
                keep[nkeep].len = len;
                emit("  seal: segment %u %08x..%08x\n", seg, bases[seg], bases[seg] + len);
                nkeep++;
            }
            for (unsigned i = 0; i < n_text + n_data; i++) {
                if (bucket[i] >= 0 && bases[bucket[i]] != 0) {
                    continue; /* covered by a kept segment range */
                }
                if (nkeep < 36) {
                    keep[nkeep].addr = i < n_text ? text_syms[i] : data_syms[i - n_text];
                    keep[nkeep].len = 4;
                    nkeep++;
                }
            }
        }
    }
    if (nkeep == 0) {
        emit("  seal: module segment lookup failed; falling back to the spot list\n");
        const uint32_t spot[] = {
            (uint32_t)(uintptr_t)&azahar_enter,       (uint32_t)(uintptr_t)&azahar_vec_irq,
            (uint32_t)(uintptr_t)&azahar_get_banked_sp, (uint32_t)(uintptr_t)&core2_entry,
            (uint32_t)(uintptr_t)&core2_run,       (uint32_t)(uintptr_t)&core2_install,
            (uint32_t)(uintptr_t)&core2_uninstall, (uint32_t)(uintptr_t)&translate,
            (uint32_t)(uintptr_t)&azahar_ctx,         (uint32_t)(uintptr_t)azahar_host,
            (uint32_t)(uintptr_t)&azahar_progress,    (uint32_t)(uintptr_t)azahar_exit_info,
            (uint32_t)(uintptr_t)azahar_stray,        (uint32_t)(uintptr_t)&azahar_cmd,
            (uint32_t)(uintptr_t)&azahar_heartbeat,   (uint32_t)(uintptr_t)&azahar_timer_sample,
            (uint32_t)(uintptr_t)&gic_va,          (uint32_t)(uintptr_t)&sv_ttbcr,
            (uint32_t)(uintptr_t)&azahar_l1_pa,       (uint32_t)(uintptr_t)&azahar_run_reason,
            (uint32_t)(uintptr_t)azahar_sony,         (uint32_t)(uintptr_t)&azahar_state,
        };
        for (unsigned i = 0; i < sizeof(spot) / sizeof(spot[0]); i++) {
            keep[nkeep].addr = spot[i];
            keep[nkeep].len = 4;
            nkeep++;
        }
    }
    // Always: both asm stacks (contiguous in .bss, stray then irq), the worker's kernel
    // stack, and the GIC window.
    keep[nkeep].addr = (uint32_t)(uintptr_t)azahar_stray_stack_top - 0x40u;
    keep[nkeep].len = (uint32_t)(uintptr_t)azahar_irq_stack_top -
                      ((uint32_t)(uintptr_t)azahar_stray_stack_top - 0x40u);
    nkeep++;
    keep[nkeep].addr = azahar_core2_sp - 0x4000u;
    keep[nkeep].len = 0x4800u;
    nkeep++;
    if (gic_va != 0) {
        keep[nkeep].addr = gic_va;
        keep[nkeep].len = 0x2000u;
        nkeep++;
    }
    unsigned kept = 0;
    char kept_list[256];
    int kl = 0;
    for (unsigned i = 0; i < nkeep; i++) {
        if (keep[i].addr == 0 || keep[i].len == 0) {
            continue;
        }
        const uint32_t first = keep[i].addr >> 20;
        const uint32_t last = (keep[i].addr + keep[i].len - 1u) >> 20;
        if (last - first >= 64u) {
            // No legitimate keep range spans 64 megabytes; this also catches wrap-around
            // (last < first underflows huge). Refusing here is the backstop that keeps a
            // miscomputed range from sweeping the whole table one log line at a time.
            emit("  seal: range %08x+%x implausible, refused\n", keep[i].addr, keep[i].len);
            continue;
        }
        unsigned skipped = 0;
        for (uint32_t idx = first; idx <= last; idx++) {
            if (azahar_l1[idx] & 3u) {
                continue;
            }
            const uint32_t e = KERNEL_L1(idx);
            if ((e & 3u) == 0) {
                // A megabyte the kernel's own table does not map cannot be one anything on
                // this core needs - the kernel could not touch it either. Ranges may
                // legitimately brush such megabytes; skip them rather than fail the install.
                skipped++;
                continue;
            }
            azahar_l1[idx] = e;
            kept++;
            if (kl < (int)sizeof(kept_list) - 8) {
                kl += snprintf(kept_list + kl, sizeof(kept_list) - kl, " %03x", idx);
            }
        }
        if (skipped != 0) {
            emit("  seal: %u unmapped megabyte(s) inside %08x+%x skipped\n", skipped,
                 keep[i].addr, keep[i].len);
        }
    }
    emit("  L1 sealed: %u kernel megabyte(s) kept (%s ), the rest invalid; kernel ttbr0 table at "
         "va %p, ttbr1 pa %08x, core-2 sp %08x\n",
         kept, kept_list, (const void *)kernel_l1, ttbr1_pa, azahar_core2_sp);
#else
    for (unsigned i = 0; i < AZAHAR_L1_ENTRIES; i++) {
        azahar_l1[i] = KERNEL_L1(i);
    }
    emit("  L1 merged from the kernel's two tables (AZAHAR_SEAL=0): ttbr0 table at va %p, ttbr1 pa "
         "%08x\n", (const void *)kernel_l1, ttbr1_pa);
#endif

    // The vector page: the highest free megabyte below 0x40000000 that no guest range wants.
    // The top half of the low quarter is fully populated on this console (2026-08-29: no free
    // entry in 0x200-0x3ff), so the scan runs down to 1; 0x090 is where STAGE_OWN put it.
    // "Free" is type bits 00, not a zero word: the 2026-08-29 run found every entry in
    // 0x200-0x3ff non-zero and none of them valid.
    unsigned upper_used = 0;
    for (uint32_t idx = 0x200; idx < 0x400; idx++) {
        upper_used += (KERNEL_L1(idx) & 3u) != 0;
    }
    azahar_vec_va = 0;
    for (uint32_t idx = 0x3FF; idx >= 1; idx--) {
        if ((KERNEL_L1(idx) & 3u) != 0 || (azahar_l1[idx] & 3u) != 0) {
            continue;
        }
        int clash = 0;
        for (unsigned r = 0; r < req->count; r++) {
            const uint32_t lo = req->ranges[r].guest_va >> 20;
            const uint32_t hi = (req->ranges[r].guest_va + req->ranges[r].size - 1) >> 20;
            if (idx >= lo && idx <= hi) {
                clash = 1;
            }
        }
        if (!clash) {
            azahar_vec_va = idx << 20;
            break;
        }
    }
    if (!azahar_vec_va) {
        emit("  no free megabyte for the vector page (%u/512 entries used in 0x200-0x3ff)\n",
             upper_used);
        return AZAHAR_ERR_TABLE;
    }
    emit("  vector page slot: %08x (kernel's entry there %08x; %u/512 kernel entries valid in "
         "0x200-0x3ff)\n",
         azahar_vec_va, KERNEL_L1(azahar_vec_va >> 20), upper_used);

    // Sony's 16 words, four slots replaced with `ldr pc, [pc, #0x34]` reading our handlers'
    // addresses from 0x40..0x4c: UND at 0x04, SVC at 0x08, PABT at 0x0c, DABT at 0x10.
    const volatile uint32_t *sony = (const volatile uint32_t *)(uintptr_t)azahar_core2_vbar;
    for (unsigned i = 0; i < 16; i++) {
        azahar_vec[i] = sony[i];
    }
    for (unsigned i = 0; i < 8; i++) {
        azahar_sony[i] = azahar_sony_target(azahar_core2_vbar, i);
    }
    emit("  sony vectors: rst %08x und %08x svc %08x pabt %08x dabt %08x irq %08x fiq %08x\n",
         azahar_sony[0], azahar_sony[1], azahar_sony[2], azahar_sony[3], azahar_sony[4], azahar_sony[6],
         azahar_sony[7]);
    azahar_vec[0x40 / 4] = (uint32_t)(uintptr_t)&azahar_vec_und;
    azahar_vec[0x44 / 4] = (uint32_t)(uintptr_t)&azahar_vec_svc;
    azahar_vec[0x48 / 4] = (uint32_t)(uintptr_t)&azahar_vec_pabt;
    azahar_vec[0x4C / 4] = (uint32_t)(uintptr_t)&azahar_vec_dabt;
    azahar_vec[0x50 / 4] = (uint32_t)(uintptr_t)&azahar_vec_irq;
    azahar_vec[0x04 / 4] = 0xE59FF034u;
    azahar_vec[0x08 / 4] = 0xE59FF034u;
    azahar_vec[0x0C / 4] = 0xE59FF034u;
    azahar_vec[0x10 / 4] = 0xE59FF034u;
    azahar_vec[0x18 / 4] = 0xE59FF030u; // IRQ: at 0x18 the PC reads 0x20; pointer at 0x50
    azahar_l1[azahar_vec_va >> 20] = AZAHAR_SEC_PRIV_RWX((uint32_t)azahar_vec_pa);
    emit("  vectors: copied from %08x, ours at %08x (pa %08x); und %08x svc %08x pabt %08x dabt %08x irq %08x\n",
         azahar_core2_vbar, azahar_vec_va, (uint32_t)azahar_vec_pa, azahar_vec[0x40 / 4], azahar_vec[0x44 / 4],
         azahar_vec[0x48 / 4], azahar_vec[0x4C / 4], azahar_vec[0x50 / 4]);

    // The guest windows.
    azahar_l2_used = 0;
    for (unsigned i = 0; i < AZAHAR_L2_COUNT * 256; i++) {
        azahar_l2[i] = 0;
    }
    for (unsigned r = 0; r < req->count; r++) {
        const int e = map_range(&req->ranges[r], r);
        if (e) {
            return e;
        }
    }
    emit("  %u L2 table(s) in use of %u\n", azahar_l2_used, AZAHAR_L2_COUNT);

    // Preflight: at N=0 there is no TTBR1 to catch a gap, so what core 2 will touch must resolve.
    const uint32_t crit[] = {(uint32_t)(uintptr_t)&azahar_enter, (uint32_t)(uintptr_t)&azahar_ctx,
                             azahar_core2_sp, gic_va, (uint32_t)(uintptr_t)&core2_run};
    static const char *const crit_name[] = {"entry code", "context", "core-2 stack", "gic",
                                            "core-2 C code"};
    for (unsigned i = 0; i < sizeof(crit) / sizeof(crit[0]); i++) {
        if (crit[i] && (azahar_l1[crit[i] >> 20] & 3u) == 0) {
            emit("  preflight: %s at %08x -> l1[%03x] invalid\n", crit_name[i], crit[i],
                 crit[i] >> 20);
            return AZAHAR_ERR_TABLE;
        }
    }

    // The app's pages were cleaned one by one above; the tables and vectors go with them.
    ksceKernelDcacheCleanRange(azahar_vec, 0x100);
    ksceKernelDcacheCleanRange(azahar_l2, AZAHAR_L2_COUNT * 0x400);
    ksceKernelDcacheCleanRange(azahar_l1, AZAHAR_L1_ENTRIES * 4);
    asm volatile("dsb" ::: "memory");
    return 0;
}

static AzaharMapRequest azahar_req;

int azaharMap(const AzaharMapRequest *user_req) {
    uint32_t state;
    ENTER_SYSCALL(state);
    log_open();
    emit("== azaharMap\n");
    int ret = 0;

    if (azahar_state != ST_TAKEN) {
        ret = AZAHAR_ERR_STATE;
        goto out;
    }
    if (ksceKernelCopyFromUser(&azahar_req, user_req, sizeof(azahar_req)) < 0) {
        ret = AZAHAR_ERR_ARG;
        goto out;
    }
    if (azahar_req.count > AZAHAR_MAX_RANGES) {
        ret = AZAHAR_ERR_ARG;
        goto out;
    }
    // The trigger app maps its windows here and wants every run logged; the emulator installs an
    // empty table and adds windows with azaharEdit, and cannot afford a synced line per exit.
    azahar_verbose = azahar_req.count > 0;
    azahar_run_count = 0;
    for (unsigned r = 0; r < azahar_req.count; r++) {
        const AzaharRange *rg = &azahar_req.ranges[r];
        if (!check_range(rg, 1) || !(rg->perm & AZAHAR_PERM_R) || (rg->perm & ~7u)) {
            emit("  range %u rejected: user %08x guest %08x size %08x perm %u\n", r, rg->user_va,
                 rg->guest_va, rg->size, rg->perm);
            ret = AZAHAR_ERR_ARG;
            goto out;
        }
    }

    if ((ret = build_table(&azahar_req)) != 0) {
        goto out;
    }
    if (azahar_folded) {
        // Prove the world switch on this core once, both ways, before any guest runs.
        core2_install();
        core2_uninstall();
        emit("  installed and removed once on cpu%u: ttbcr %08x -> %08x, vbar %08x -> %08x, "
             "pmr %02x -> %02x, cpacr %08x fpexc %08x\n",
             (uint32_t)ksceKernelCpuId(), sv_ttbcr, azahar_ttbcr_seen, sv_vbar, azahar_vec_va, sv_pmr,
             AZAHAR_PMR_MASK, sv_cpacr, sv_fpexc);
        fold_measure_timer();
        azahar_state = ST_MAPPED;
        goto out;
    }
    if ((ret = core2_command(CMD_INSTALL, AZAHAR_WAIT_STEPS)) != 0) {
        goto out;
    }
    emit("  installed on core %u: ttbcr %08x -> %08x, vbar %08x -> %08x, sp_und %08x sp_abt %08x "
         "sp_svc %08x sp_irq %08x, pmr %02x -> %02x, timer ppi was %s, cpacr %08x fpexc %08x\n",
         AZAHAR_TARGET_CORE, sv_ttbcr, azahar_ttbcr_seen, sv_vbar, azahar_vec_va, sv_sp_und, sv_sp_abt,
         sv_sp_svc, sv_sp_irq, sv_pmr, AZAHAR_PMR_MASK, sv_isenabler_timer ? "enabled" : "disabled",
         sv_cpacr, sv_fpexc);

    // The private timer's clock, from the idle loop's samples of the free-running counter
    // against the system clock. PLAN.md §14.2 carries a device-tree claim of a fixed 144 MHz
    // against the A9 ratio; this is the number the quantum is converted with.
    azahar_timer_hz = 0;
    if (gic_va) {
        const uint32_t s0 = azahar_timer_sample, t0 = ksceKernelGetSystemTimeLow();
        ksceKernelDelayThread(AZAHAR_TIMER_MEASURE_US);
        const uint32_t s1 = azahar_timer_sample, t1 = ksceKernelGetSystemTimeLow();
        const uint32_t ticks = s0 - s1, us = t1 - t0; // counts down
        if (us) {
            azahar_timer_hz = (uint32_t)(((uint64_t)ticks * 1000000ull) / us);
        }
        emit("  private timer: %u ticks in %u us -> %u Hz (%u.%02u MHz)\n", ticks, us,
             azahar_timer_hz, azahar_timer_hz / 1000000u, (azahar_timer_hz / 10000u) % 100u);
    }
    azahar_state = ST_MAPPED;

out:
    emit("  -> %d\n", ret);
    log_close();
    EXIT_SYSCALL(state);
    return ret;
}

static AzaharRunRequest azahar_run_req;

int azaharRun(AzaharRunRequest *user_req) {
    uint32_t state;
    ENTER_SYSCALL(state);
    log_open();
    int ret = 0;

    if (azahar_state != ST_MAPPED) {
        ret = AZAHAR_ERR_STATE;
        goto out;
    }
    if (ksceKernelCopyFromUser(&azahar_run_req, user_req, sizeof(azahar_run_req)) < 0) {
        ret = AZAHAR_ERR_ARG;
        goto out;
    }
    // User mode only, and interrupts unmasked at the CPU: the GIC mask does the owning.
    azahar_ctx = azahar_run_req.ctx;
    azahar_ctx.cpsr = (azahar_ctx.cpsr & ~0x1DFu) | 0x10u; // keep T (bit 5); mode user; A/I/F clear
    azahar_run_quantum_ticks = 0;
    if (azahar_run_req.quantum_us) {
        if (!azahar_timer_hz) {
            emit("  quantum requested but the timer clock is unmeasured\n");
            ret = AZAHAR_ERR_STATE;
            goto out;
        }
        const uint64_t t = ((uint64_t)azahar_run_req.quantum_us * azahar_timer_hz) / 1000000ull;
        azahar_run_quantum_ticks = t > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)t;
        if (!azahar_run_quantum_ticks) {
            azahar_run_quantum_ticks = 1;
        }
    }
    azahar_run_timer_left = 0;
    if (azahar_verbose) {
        emit("== azaharRun  pc %08x cpsr %08x sp %08x r0 %08x r1 %08x  quantum %u us = %u ticks\n",
             azahar_ctx.r[15], azahar_ctx.cpsr, azahar_ctx.r[13], azahar_ctx.r[0], azahar_ctx.r[1],
             azahar_run_req.quantum_us, azahar_run_quantum_ticks);
    }

    if (azahar_folded) {
        fold_run();
        ret = 0;
    } else if ((ret = core2_command(CMD_RUN, AZAHAR_RUN_WAIT_STEPS)) != 0) {
        emit("== azaharRun  pc %08x cpsr %08x sp %08x r0 %08x r1 %08x  quantum %u us = %u ticks\n",
             azahar_run_req.ctx.r[15], azahar_ctx.cpsr, azahar_ctx.r[13], azahar_ctx.r[0], azahar_ctx.r[1],
             azahar_run_req.quantum_us, azahar_run_quantum_ticks);
        // Where core 2 got to. 1-3 C before entry; 10-11 entry asm; a stub word equal to its
        // own address means that exception was taken; 31-34 the user file was saved; 40-41
        // back in host mode; 42-50 back in C.
        emit("  progress %u  stubs und %s svc %s pabt %s dabt %s irq %s  irq id %08x stray %u\n",
             azahar_progress,
             azahar_prog_und == (uint32_t)(uintptr_t)&azahar_prog_und ? "hit" : "-",
             azahar_prog_svc == (uint32_t)(uintptr_t)&azahar_prog_svc ? "hit" : "-",
             azahar_prog_pabt == (uint32_t)(uintptr_t)&azahar_prog_pabt ? "hit" : "-",
             azahar_prog_dabt == (uint32_t)(uintptr_t)&azahar_prog_dabt ? "hit" : "-",
             azahar_prog_irq == (uint32_t)(uintptr_t)&azahar_prog_irq ? "hit" : "-", azahar_irq_last,
             azahar_irq_stray);
        emit("  ctx now: r0 %08x r1 %08x r2 %08x r4 %08x sp %08x lr %08x pc/lr %08x cpsr/spsr %08x\n",
             azahar_ctx.r[0], azahar_ctx.r[1], azahar_ctx.r[2], azahar_ctx.r[4], azahar_ctx.r[13], azahar_ctx.r[14],
             azahar_ctx.r[15], azahar_ctx.cpsr);
        emit("  exit info: reason %u rawlr %08x spsr %08x dfsr %08x dfar %08x ifsr %08x ifar %08x\n",
             azahar_exit_info[0], azahar_exit_info[1], azahar_exit_info[2], azahar_exit_info[3],
             azahar_exit_info[4], azahar_exit_info[5], azahar_exit_info[6]);
        goto out;
    }
    azahar_run_req.ctx = azahar_ctx;
    azahar_run_req.reason = azahar_run_reason;
    azahar_run_req.svc_number = azahar_run_svc;
    azahar_run_req.fsr = azahar_run_fsr;
    azahar_run_req.far = azahar_run_far;
    azahar_run_req.spsr = azahar_run_spsr;
    azahar_run_req.raw_lr = azahar_run_rawlr;
    azahar_run_req.timer_hz = azahar_timer_hz;
    azahar_run_req.timer_left = azahar_run_timer_left;
    azahar_run_count++;
    if (azahar_verbose || (azahar_run_reason > AZAHAR_EXIT_SVC && azahar_run_reason != AZAHAR_EXIT_TIMER)) {
        // Faults are rare enough to record. svc exits are the common case, and a guest that
        // busy-waits through its quantum makes preemptions just as common (hundreds a minute
        // in the first commercial title tried), so both stay silent unless verbose.
        emit("  run %u: exit reason %u  svc %06x  pc %08x  raw lr %08x  spsr %08x  fsr %08x far "
             "%08x  r0 %08x r1 %08x sp %08x  timer left %u  irq id %08x stray %u\n",
             azahar_run_count, azahar_run_reason, azahar_run_svc, azahar_ctx.r[15], azahar_run_rawlr,
             azahar_run_spsr, azahar_run_fsr, azahar_run_far, azahar_ctx.r[0], azahar_ctx.r[1], azahar_ctx.r[13],
             azahar_run_timer_left, azahar_irq_last, azahar_irq_stray);
    }
    if (ksceKernelCopyToUser(user_req, &azahar_run_req, sizeof(azahar_run_req)) < 0) {
        ret = AZAHAR_ERR_ARG;
    }

out:
    if (ret) {
        emit("  -> %d\n", ret);
    }
    log_close();
    EXIT_SYSCALL(state);
    return ret;
}

static AzaharEditRequest azahar_edit_req;

int azaharEdit(const AzaharEditRequest *user_req) {
    uint32_t state;
    ENTER_SYSCALL(state);
    log_open();
    int ret = 0;
    int quiet = 0; /* SYNC_CODE comes once a frame on some titles: logged only when it fails */

    if (azahar_state != ST_MAPPED) {
        ret = AZAHAR_ERR_STATE;
        goto out;
    }
    if (ksceKernelCopyFromUser(&azahar_edit_req, user_req, sizeof(azahar_edit_req)) < 0) {
        ret = AZAHAR_ERR_ARG;
        goto out;
    }
    const AzaharRange *rg = &azahar_edit_req.range;
    const uint32_t op = azahar_edit_req.op;
    /* PROTECT comes per armed page from the guest write tracker: logged only when it fails */
    quiet = op == AZAHAR_EDIT_SYNC_CODE || op == AZAHAR_EDIT_PROTECT;
    if (!quiet)
        emit("== azaharEdit %s guest %08x user %08x size %08x\n",
             op == AZAHAR_EDIT_MAP ? "MAP" : op == AZAHAR_EDIT_UNMAP ? "UNMAP" : op == AZAHAR_EDIT_PROTECT ? "PROTECT"
             : op == AZAHAR_EDIT_SYNC_CODE ? "SYNC_CODE" : "?",
             rg->guest_va, rg->user_va, rg->size);
    if (op == AZAHAR_EDIT_SYNC_CODE) {
        if ((rg->user_va & 0xFFFu) || (rg->size & 0xFFFu) || !rg->size) {
            ret = AZAHAR_ERR_ARG;
            goto out;
        }
        // The app wrote instructions through its own view; they are in some core's data cache.
        // Clean them to the point of unification page by page under the domain grant, and let
        // the next run drop core 2's instruction cache and branch predictor.
        for (uint32_t off = 0; off < rg->size; off += 0x1000) {
            SceKernelIntrStatus st = ksceKernelCpuSuspendIntr();
            const uint32_t d = rd_dacr();
            wr_dacr(dacr_grant_all(d));
            ksceKernelDcacheCleanRange((void *)(uintptr_t)(rg->user_va + off), 0x1000);
            wr_dacr(d);
            ksceKernelCpuResumeIntr(st);
        }
        asm volatile("dsb" ::: "memory");
        azahar_tlb_dirty = 1;
        goto out;
    }
    if (!check_range(rg, op == AZAHAR_EDIT_MAP) ||
        (op != AZAHAR_EDIT_UNMAP && (!(rg->perm & AZAHAR_PERM_R) || (rg->perm & ~7u)))) {
        ret = AZAHAR_ERR_ARG;
        goto out;
    }

    switch (op) {
        case AZAHAR_EDIT_MAP:
            ret = map_range(rg, 0xEDu);
            break;
        case AZAHAR_EDIT_UNMAP:
        case AZAHAR_EDIT_PROTECT: {
            unsigned done = 0, missing = 0;
            for (uint32_t off = 0; off < rg->size; off += 0x1000) {
                const uint32_t gva = rg->guest_va + off;
                const int t = l2_for(gva >> 20, 0);
                uint32_t *d = t < 0 ? NULL : &azahar_l2[(unsigned)t * 256u + ((gva >> 12) & 0xFFu)];
                if (!d || (*d & 3u) == 0) {
                    missing++;
                    continue;
                }
                *d = (op == AZAHAR_EDIT_UNMAP) ? 0u : azahar_small_page(*d & 0xFFFFF000u, rg->perm);
                done++;
            }
            if (!quiet || missing)
                emit("  %u page(s) %s, %u not mapped\n", done,
                     op == AZAHAR_EDIT_UNMAP ? "unmapped" : "reprotected", missing);
            if (op == AZAHAR_EDIT_PROTECT && missing) {
                ret = AZAHAR_ERR_ARG;
            }
            break;
        }
        default:
            ret = AZAHAR_ERR_ARG;
            break;
    }
    if (ret == 0) {
        ksceKernelDcacheCleanRange(azahar_l2, AZAHAR_L2_COUNT * 0x400);
        ksceKernelDcacheCleanRange(azahar_l1, AZAHAR_L1_ENTRIES * 4);
        asm volatile("dsb" ::: "memory");
        azahar_tlb_dirty = 1;
    }

out:
    if (!quiet || ret != 0)
        emit("  -> %d\n", ret);
    log_close();
    EXIT_SYSCALL(state);
    return ret;
}

// The counters the release prints, readable while a title runs: how many SGI service
// windows core 2 has opened, the last pending mask that opened one, how many timed out, and
// the resident loop's heartbeat - so a freeze log's last second says whether core 2 was still
// polling and whether a cross-core call had just been serviced or was pending.
int azaharSgiStats(AzaharSgiStats *user_out) {
    uint32_t state;
    ENTER_SYSCALL(state);
    AzaharSgiStats out;
    out.windows = azahar_sgi_windows;
    out.last_pend = azahar_sgi_last_pend;
    out.stuck = azahar_sgi_stuck;
    out.heartbeat = azahar_heartbeat;
    out.installed = azahar_installed;
    out.pending_now = 0;
    if (azahar_installed && gic_va) {
        out.pending_now = *(volatile uint32_t *)(gic_va + GICD_ISPENDR0) & 0x7FFFu;
    }
    const int ret = ksceKernelCopyToUser(user_out, &out, sizeof(out)) < 0 ? AZAHAR_ERR_ARG : 0;
    EXIT_SYSCALL(state);
    return ret;
}

// Every thread of the client with the registers the kernel holds for it: where a thread
// that stopped retiring work actually is. ksceKernelGetThreadCpuRegisters wants the thread
// suspended, so each thread but the caller's is debug-suspended around the read and resumed
// at once. This is what found the 2026-09-08 freeze (an overtaken notification wait).
int azaharThreadDump(void) {
    uint32_t state;
    ENTER_SYSCALL(state);
    int count = 0;
    if (azahar_client_pid >= 0) {
        log_open();
        emit("== azaharThreadDump\n");
        static SceUID ids[128];
        int copied = 0;
        if (ksceKernelGetThreadIdList(azahar_client_pid, ids, 128, &copied) >= 0) {
            for (int i = 0; i < copied; i++) {
                SceKernelThreadInfo info;
                info.size = sizeof(info);
                if (ksceKernelGetThreadInfo(ids[i], &info) < 0) {
                    continue;
                }
                SceThreadCpuRegisters regs;
                __builtin_memset(&regs, 0, sizeof(regs));
                int r = -1;
                if (ids[i] != ksceKernelGetThreadId()) {
                    const int sus = ksceKernelDebugSuspendThread(ids[i], 0x100);
                    r = ksceKernelGetThreadCpuRegisters(ids[i], &regs);
                    if (sus >= 0) {
                        ksceKernelDebugResumeThread(ids[i], 0x100);
                    }
                }
                emit("  %-24s thid %08x status %x prio %3d cpu %d aff %05x wait %d/%08x  user pc %08x "
                     "lr %08x sp %08x  kernel pc %08x lr %08x  (regs %d)\n",
                     info.name, (uint32_t)ids[i], (uint32_t)info.status, info.currentPriority,
                     info.lastExecutedCpuId, (uint32_t)info.currentCpuAffinityMask,
                     (int)info.waitType, (uint32_t)info.waitId, regs.entry[0].pc, regs.entry[0].lr,
                     regs.entry[0].sp, regs.entry[1].pc, regs.entry[1].lr, r);
                count++;
            }
        }
        log_close();
    }
    EXIT_SYSCALL(state);
    return count;
}

int azaharPmuReadAll(AzaharPmuAll *user_out) {
    uint32_t state;
    ENTER_SYSCALL(state);
    pmu_host_start();
    AzaharPmuAll out;
    for (unsigned c = 0; c < 4; c++) {
        if (c == AZAHAR_TARGET_CORE) {
            out.core[c].cycles = azahar_pmu_cycles;
            for (unsigned i = 0; i < 6; i++) {
                out.core[c].events[i] = azahar_pmu_ev[i];
            }
            out.core[c].runs = azahar_pmu_runs;
            azahar_pmu_cycles = 0;
            for (unsigned i = 0; i < 6; i++) {
                azahar_pmu_ev[i] = 0;
            }
            azahar_pmu_runs = 0;
        } else {
            out.core[c].cycles = azahar_pmu_host_cycles[c];
            for (unsigned i = 0; i < 6; i++) {
                out.core[c].events[i] = azahar_pmu_host_ev[c][i];
            }
            out.core[c].runs = azahar_pmu_host_samples[c];
            azahar_pmu_host_cycles[c] = 0;
            for (unsigned i = 0; i < 6; i++) {
                azahar_pmu_host_ev[c][i] = 0;
            }
            azahar_pmu_host_samples[c] = 0;
        }
        out.core[c].pad = 0;
    }
    const int ret = ksceKernelCopyToUser(user_out, &out, sizeof(out)) < 0 ? AZAHAR_ERR_ARG : 0;
    EXIT_SYSCALL(state);
    return ret;
}

int azaharPmuRead(AzaharPmuStats *user_out) {
    uint32_t state;
    ENTER_SYSCALL(state);
    AzaharPmuStats out;
    out.cycles = azahar_pmu_cycles;
    for (unsigned i = 0; i < 6; i++) {
        out.events[i] = azahar_pmu_ev[i];
    }
    out.runs = azahar_pmu_runs;
    out.pad = 0;
    azahar_pmu_cycles = 0;
    for (unsigned i = 0; i < 6; i++) {
        azahar_pmu_ev[i] = 0;
    }
    azahar_pmu_runs = 0;
    // No log_open: this is called every second and has nothing to say.
    const int ret = ksceKernelCopyToUser(user_out, &out, sizeof(out)) < 0 ? AZAHAR_ERR_ARG : 0;
    EXIT_SYSCALL(state);
    return ret;
}

int azaharRelease(void) {
    uint32_t state;
    ENTER_SYSCALL(state);
    log_open();
    emit("== azaharRelease\n");
    int ret = 0;

    pmu_host_stop_all();

    if (azahar_folded) {
        if (azahar_state != ST_MAPPED && azahar_state != ST_TAKEN) {
            ret = AZAHAR_ERR_STATE;
            goto out;
        }
        emit("  %u run(s) since the map (folded)\n", azahar_run_count);
        free_blocks();
        azahar_state = ST_NONE;
        azahar_client_pid = -1;
        goto out;
    }
    if (azahar_state == ST_MAPPED) {
        ret = core2_command(CMD_UNINSTALL, AZAHAR_WAIT_STEPS);
        emit("  uninstall -> %d (vbar back to %08x)\n", ret, sv_vbar);
    } else if (azahar_state != ST_TAKEN) {
        ret = AZAHAR_ERR_STATE;
        goto out;
    }
    emit("  %u run(s) since the map\n", azahar_run_count);
    emit("  sgi windows %u  last pend %04x  stuck %u\n", azahar_sgi_windows, azahar_sgi_last_pend,
         azahar_sgi_stuck);
    // Sony's world is back on core 2 either way (core2_uninstall put back every register
    // install touched). What happens to the core depends on whether this session took it:
    //
    // The default, no detach: nothing was taken, so nothing is given back. The thread ends and
    // core 2 is an ordinary core again, for this app and every other.
    //
    // Detached (ux0:data/azahar/detach): it stays detached, with the resident thread parked -
    // the state STAGE_HOLD ran 180 s in. The mask is never restored, because a core detached
    // with a thread resident on it is never dispatched to again and the restore froze the
    // whole console twice (FINDINGS, "The reattach is one-way"). The guest's tables go now;
    // the next azaharTakeCore wakes the parked thread and reuses the core.
    if (!azahar_detached) {
        // Nothing was taken from the scheduler, so nothing has to be given back: the resident
        // thread ends like any other thread and core 2 is a normal core again the moment it
        // does. This is the whole point of holding the core without detaching it.
        const int ended = azahar_stop_resident();
        free_blocks();
        azahar_state = ended == 0 ? ST_NONE : ST_RELEASED;
    } else {
        azahar_park = 1;
        free_blocks();
        azahar_state = ST_RELEASED;
    }
    azahar_client_pid = -1;

out:
    emit("  -> %d\n", ret);
    log_release();
    EXIT_SYSCALL(state);
    return ret;
}

