/* ifaddrs.h for the PS Vita.
 *
 * The console has no getifaddrs. Only cpp-httplib asks for it, to enumerate local addresses for
 * a listening server, and nothing in this build listens - so the enumeration is declared and
 * always comes back empty rather than being emulated over SceNetCtl.
 */

#pragma once

#include <sys/socket.h>

/* The broadcast and destination addresses share storage the way they do everywhere else, so
   the usual two macros are what callers reach for. */
#define ifa_broadaddr ifa_ifu.ifu_broadaddr
#define ifa_dstaddr ifa_ifu.ifu_dstaddr

struct ifaddrs {
    struct ifaddrs* ifa_next;
    char* ifa_name;
    unsigned int ifa_flags;
    struct sockaddr* ifa_addr;
    struct sockaddr* ifa_netmask;
    union {
        struct sockaddr* ifu_broadaddr;
        struct sockaddr* ifu_dstaddr;
    } ifa_ifu;
    void* ifa_data;
};

#ifdef __cplusplus
extern "C" {
#endif

static inline int getifaddrs(struct ifaddrs** ifap) {
    if (ifap != 0) {
        *ifap = 0;
    }
    return -1;
}

static inline void freeifaddrs(struct ifaddrs* ifa) {
    (void)ifa;
}

#ifdef __cplusplus
}
#endif
