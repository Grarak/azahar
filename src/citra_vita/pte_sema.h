// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * Kernel-semaphore accounting for the pthread primitives the C library builds on. See
 * pte_sema.c: on this target every std::mutex, condition variable and call_once holds a
 * process UID, and running out breaks a lock silently rather than failing.
 *
 * Built with the rest of the Vita tracing, under -DVITA_ALLOC_TRACE=ON; a release build has
 * neither the counters nor the --wrap flags that feed them, and everything here compiles to
 * nothing.
 */
#ifdef VITA_ALLOC_TRACE
extern "C" {
/// Kernel semaphores held by pthread primitives right now.
int VitaSemaLive(void);
/// The most ever held at once.
int VitaSemaPeak(void);
/// Created, deleted and refused since boot.
void VitaSemaTotals(unsigned* created, unsigned* deleted, unsigned* refused);
}
#endif
