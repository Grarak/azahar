/* netdb.h for the PS Vita.
 *
 * The console's own netdb.h has the lookup functions but not the h_errno result codes that go
 * with them, which Boost.Asio maps onto its error category. The values are the standard ones.
 */

#pragma once

#include_next <netdb.h>

#ifndef HOST_NOT_FOUND
#define HOST_NOT_FOUND 1
#define TRY_AGAIN 2
#define NO_RECOVERY 3
#define NO_DATA 4
#endif
