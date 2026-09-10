// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "common/vector_math.h"

namespace Pica::Texture {

Common::Vec3<u8> SampleETC1Subtile(u64 value, unsigned int x, unsigned int y);

/**
 * Decodes a full 8x8 ETC1 or ETC1A4 tile in one pass. The per-texel path re-parses the same
 * 64-bit subtile word - base colours, table indices, flip - for every one of its sixteen texels;
 * this extracts them once per subtile and only the per-texel modifier lookup remains inside the
 * loop. out is indexed [y][x] with bytes r, g, b, a; ETC1 alpha is 255.
 */
void DecodeETC1TileRGBA8(const u8* tile, bool has_alpha, u8 out[8][8][4]);

} // namespace Pica::Texture
