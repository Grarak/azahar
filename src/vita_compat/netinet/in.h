/* netinet/in.h for the PS Vita.
 *
 * The console's own header covers IPv4 and the parts of IPv6 its network stack implements. What
 * is added here is what callers expect to exist even when the answer is always no: this build
 * neither listens nor joins multicast groups.
 */

#pragma once

#include_next <netinet/in.h>

#ifndef IN6_IS_ADDR_LINKLOCAL
#define IN6_IS_ADDR_LINKLOCAL(a)                                                                   \
    ((((const uint8_t*)(a))[0] == 0xfe) && ((((const uint8_t*)(a))[1] & 0xc0) == 0x80))
#endif

#ifndef IPV6_JOIN_GROUP
struct ipv6_mreq {
    struct in6_addr ipv6mr_multiaddr;
    unsigned int ipv6mr_interface;
};
#endif

/* The console's stack has no IPv6 multicast, but Asio names these constants unconditionally
   when it compiles its v6 support. The values are the usual ones; nothing here uses them. */
#ifndef IPV6_JOIN_GROUP
#define IPV6_JOIN_GROUP 20
#define IPV6_LEAVE_GROUP 21
#endif
#ifndef IPV6_MULTICAST_IF
#define IPV6_MULTICAST_IF 17
#define IPV6_MULTICAST_HOPS 18
#define IPV6_MULTICAST_LOOP 19
#endif
#ifndef IPV6_UNICAST_HOPS
#define IPV6_UNICAST_HOPS 16
#endif
#ifndef IPPROTO_ICMPV6
#define IPPROTO_ICMPV6 58
#endif
