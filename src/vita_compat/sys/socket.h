/* sys/socket.h for the PS Vita.
 *
 * Forwards to the console's own header and adds the one call POSIX puts here that its network
 * stack does not implement.
 */

#pragma once

#include_next <sys/socket.h>

#ifndef __vita_has_sockatmark
#define __vita_has_sockatmark

#ifdef __cplusplus
extern "C" {
#endif

/* Out-of-band data is not something this build's sockets carry, so the mark is never reached. */
static inline int sockatmark(int fd) {
    (void)fd;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
