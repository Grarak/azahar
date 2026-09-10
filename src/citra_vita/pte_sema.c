// Kernel-semaphore accounting for the PS Vita build.
//
// On this target the C library is newlib with pthreads-embedded, and pte backs a pthread
// primitive with a real kernel semaphore: exactly three objects in libpthread.a reference
// pte_osSemaphoreCreate - pthread_mutex_init.o, pthread_once.o and sem_init.o - and each of
// them ends in sceKernelCreateSema. Both mutexes and condition variables initialise lazily out
// of their static initialisers, so a std::mutex costs one UID from the first time it is locked,
// and a condition variable costs none until something waits on it (pthread_cond_wait is the
// only caller of pte_cond_check_need_init; signalling one nobody waited on allocates nothing).
// A wait would cost five more: pthread_cond_init takes a mutex and two sem_init, and each
// sem_init takes a UID and a mutex of its own. Process UIDs are a bounded resource.
//
// Running out is not a clean failure. sceKernelCreateSema returns a negative result,
// pte_osSemaphoreCreate turns that into PTE_OS_NO_RESOURCES, and pthread_mutex_init hands back
// an error that std::mutex::lock has no way to report - so the lock silently does nothing, or
// waits on a handle that was never created, and a thread stops for good. Super Mario 3D Land
// printed the refusal after nine minutes; Super Smash Bros just froze after three, with the
// render thread's queue climbing to 3408 while nothing consumed it.
//
// So under -DVITA_ALLOC_TRACE=ON, alongside the heap and memblock tracing, the counts are kept
// here, the first refusal is printed loudly, and the frontend puts the live count in its
// per-second line, where it can be read against the frame, surface and shader counts to say
// what owns them. A release build compiles none of it.
//
// Everything runs inside the allocation path of the very first mutex the C runtime touches, so
// there is nothing here but adds and subtracts on plain ints. The one print happens once.

#include <stdint.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>

extern int __real_pte_osSemaphoreCreate(int initial_value, SceUID* handle);
extern int __real_pte_osSemaphoreDelete(SceUID handle);

// .bss, so a crash dump still shows them.
static volatile int g_live;
static volatile int g_peak;
static volatile unsigned g_created;
static volatile unsigned g_deleted;
static volatile unsigned g_refused;

int __wrap_pte_osSemaphoreCreate(int initial_value, SceUID* handle) {
    const int result = __real_pte_osSemaphoreCreate(initial_value, handle);
    if (result != 0) {
        if (g_refused++ == 0) {
            sceClibPrintf("[sema] sceKernelCreateSema REFUSED: %d live, %u created, %u deleted."
                          " A pthread primitive is now broken and whatever waits on it will"
                          " never wake.\n",
                          g_live, g_created, g_deleted);
        }
        return result;
    }
    g_created++;
    const int live = ++g_live;
    if (live > g_peak) {
        g_peak = live;
    }
    return result;
}

int __wrap_pte_osSemaphoreDelete(SceUID handle) {
    g_deleted++;
    g_live--;
    return __real_pte_osSemaphoreDelete(handle);
}

/// Kernel semaphores held by pthread primitives right now.
int VitaSemaLive(void) {
    return g_live;
}

/// The most ever held at once.
int VitaSemaPeak(void) {
    return g_peak;
}

/// Created, deleted and refused since boot.
void VitaSemaTotals(unsigned* created, unsigned* deleted, unsigned* refused) {
    *created = g_created;
    *deleted = g_deleted;
    *refused = g_refused;
}
