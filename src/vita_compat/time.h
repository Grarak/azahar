/* time.h for the PS Vita.
 *
 * newlib here has no timegm, which libressl's certificate validity checking needs. It is
 * gmtime's inverse: a broken-down UTC time to a count of seconds since the epoch.
 */

#pragma once

#include_next <time.h>

#ifndef __vita_has_timegm
#define __vita_has_timegm

#ifdef __cplusplus
extern "C" {
#endif

static inline time_t timegm(struct tm* tm) {
    /* Days from 1970-01-01 to the first of the given month, computed with the year shifted so
       that a leap day lands at the end of it - which removes the special cases entirely. */
    long year = tm->tm_year + 1900;
    const int month = tm->tm_mon; /* 0-11 */
    year -= month <= 1 ? 1 : 0;
    const long era = (year >= 0 ? year : year - 399) / 400;
    const unsigned long year_of_era = (unsigned long)(year - era * 400);
    const unsigned long day_of_year =
        (153UL * (unsigned long)(month + (month > 1 ? -2 : 10)) + 2UL) / 5UL +
        (unsigned long)(tm->tm_mday - 1);
    const unsigned long day_of_era =
        year_of_era * 365UL + year_of_era / 4UL - year_of_era / 100UL + day_of_year;
    const long days = era * 146097L + (long)day_of_era - 719468L;

    return (time_t)days * 86400 + tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec;
}

#ifdef __cplusplus
}
#endif

#endif
