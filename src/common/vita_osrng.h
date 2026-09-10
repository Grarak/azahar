// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

/**
 * Crypto++'s osrng.h on the PS Vita.
 *
 * Crypto++ decides at configure time whether it has an operating system to get entropy from, and
 * on this console it decides it does not - there is no /dev/urandom and no CryptoGenRandom - so
 * it compiles out osrng.h entirely, taking AutoSeededRandomPool with it. The console does have a
 * system entropy source, it is just not one Crypto++ knows about, so the class is provided here
 * over the call it does have.
 *
 * Include this instead of <cryptopp/osrng.h>; on every other platform it forwards to the real
 * header.
 */

#ifdef __vita__

#include <cstddef>
#include <psp2/kernel/rng.h>
#include <cryptopp/cryptlib.h>

namespace CryptoPP {

/// Entropy straight from the kernel, which is what the name promises everywhere else.
class AutoSeededRandomPool : public RandomNumberGenerator {
public:
    void GenerateBlock(byte* output, std::size_t size) override {
        // The call takes a size in bytes and fills the whole buffer or fails; it is documented
        // to cap at 64 bytes per call, so larger requests are chunked.
        constexpr std::size_t max_per_call = 64;
        while (size > 0) {
            const std::size_t chunk = size < max_per_call ? size : max_per_call;
            if (sceKernelGetRandomNumber(output, static_cast<SceSize>(chunk)) < 0) {
                throw Exception(Exception::OTHER_ERROR,
                                "AutoSeededRandomPool: sceKernelGetRandomNumber failed");
            }
            output += chunk;
            size -= chunk;
        }
    }
};

} // namespace CryptoPP

#else

#include <cryptopp/osrng.h>

#endif
