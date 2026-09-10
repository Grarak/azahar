/* endian.h for the PS Vita.
 *
 * The console's newlib has <machine/endian.h> with the byte order macros but none of the
 * glibc/BSD conversion functions that several vendored libraries reach for. The Vita is
 * little-endian, so the host conversions are identities and the big-endian ones are byte swaps.
 */

#pragma once

#include <stdint.h>
#include <machine/endian.h>

#ifndef htobe16
#define htobe16(x) __builtin_bswap16(x)
#define htole16(x) ((uint16_t)(x))
#define be16toh(x) __builtin_bswap16(x)
#define le16toh(x) ((uint16_t)(x))

#define htobe32(x) __builtin_bswap32(x)
#define htole32(x) ((uint32_t)(x))
#define be32toh(x) __builtin_bswap32(x)
#define le32toh(x) ((uint32_t)(x))

#define htobe64(x) __builtin_bswap64(x)
#define htole64(x) ((uint64_t)(x))
#define be64toh(x) __builtin_bswap64(x)
#define le64toh(x) ((uint64_t)(x))
#endif
