/* net/if.h for the PS Vita.
 *
 * Interface enumeration does not exist on this console. The declarations here exist so headers
 * that reach for them compile; the functions answer "no such interface", which is true - see
 * the note in ifaddrs.h about who asks and why nothing needs a real answer.
 */

#pragma once

#include <sys/socket.h>

#define IF_NAMESIZE 16

#ifdef __cplusplus
extern "C" {
#endif

static inline unsigned int if_nametoindex(const char* name) {
    (void)name;
    return 0;
}

static inline char* if_indextoname(unsigned int index, char* name) {
    (void)index;
    (void)name;
    return 0;
}

#ifdef __cplusplus
}
#endif
