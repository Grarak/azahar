/* signal.h for the PS Vita.
 *
 * Forwards to the console's own header. SA_RESTART is named by code that installs handlers
 * portably; there is no sigaction here for it to modify, so the value only has to exist.
 */

#pragma once

#include_next <signal.h>

#ifndef SA_RESTART
#define SA_RESTART 0x10000000
#endif
