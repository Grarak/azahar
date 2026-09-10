/* Definitions the PS Vita's newlib does not provide, for libraries that expect a fuller POSIX.
 *
 * These are linked into the executable itself rather than into a library, so that they satisfy
 * references from every archive that follows on the link line.
 *
 * None of them emulates anything. Each is the honest answer for a console with one user, one
 * program, and no terminal - and each is only reached by a code path this build does not use.
 */

#include <pwd.h>
#include <stdio.h>
#include <unistd.h>

/* There is no user database. FileUtil::GetHomeDirectory only calls this when no explicit user
   path was set, and the Vita frontend sets one before anything asks. */
struct passwd* getpwuid(uid_t uid) {
    (void)uid;
    return 0;
}

/* The MMU page size, which the console does not expose and libressl only wants in order to
   round an allocation up. */
int getpagesize(void) {
    return 4096;
}

/* newlib's stdio is already locked internally, so the explicit lock is a no-op rather than a
   second lock that could deadlock against it. */
void flockfile(FILE* file) {
    (void)file;
}

void funlockfile(FILE* file) {
    (void)file;
}

int ftrylockfile(FILE* file) {
    (void)file;
    return 0;
}

/* Wait for a signal that cannot arrive. Asio's scheduler names it on a path taken only when it
   has no work and no timers, which cannot happen while it owns a running io_context. */
int pause(void) {
    return -1;
}

/*
 * Half-precision conversion, and the thread pointer.
 *
 * clang emits the AEABI names for float16 conversion; the vitasdk's libgcc, built by gcc,
 * defines the same functions under gcc's own names. Same IEEE semantics, so these are aliases
 * rather than a second implementation - the alternative would be either a software reimplementation
 * or -mfpu=neon-fp16, which assumes a half-precision extension this console may not have.
 */
extern float __gnu_h2f_ieee(unsigned short a);
extern unsigned short __gnu_f2h_ieee(float a);

float __aeabi_h2f(unsigned short a) {
    return __gnu_h2f_ieee(a);
}

unsigned short __aeabi_f2h(float a) {
    return __gnu_f2h_ieee(a);
}
