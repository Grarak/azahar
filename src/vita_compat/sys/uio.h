/* sys/uio.h for the PS Vita.
 *
 * struct iovec is declared in the console's <sys/socket.h>, which ships no <sys/uio.h> to go
 * with it. The scatter/gather calls are supplied as loops over the plain ones: the callers here
 * use them for convenience rather than for the atomicity a real readv gives.
 */

#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline ssize_t readv(int fd, const struct iovec* iov, int count) {
    ssize_t total = 0;
    for (int i = 0; i < count; i++) {
        const ssize_t got = read(fd, iov[i].iov_base, iov[i].iov_len);
        if (got < 0) {
            return total > 0 ? total : got;
        }
        total += got;
        if ((size_t)got != iov[i].iov_len) {
            break;
        }
    }
    return total;
}

static inline ssize_t writev(int fd, const struct iovec* iov, int count) {
    ssize_t total = 0;
    for (int i = 0; i < count; i++) {
        const ssize_t put = write(fd, iov[i].iov_base, iov[i].iov_len);
        if (put < 0) {
            return total > 0 ? total : put;
        }
        total += put;
        if ((size_t)put != iov[i].iov_len) {
            break;
        }
    }
    return total;
}

#ifdef __cplusplus
}
#endif

#ifndef FIONREAD
#define FIONREAD 0x4004667f
#endif
