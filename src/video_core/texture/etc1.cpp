// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstring>
#include <array>
#include "common/bit_field.h"
#include "common/color.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/texture/etc1.h"

namespace Pica::Texture {

namespace {

constexpr std::array<std::array<u8, 2>, 8> etc1_modifier_table = {{
    {2, 8},
    {5, 17},
    {9, 29},
    {13, 42},
    {18, 60},
    {24, 80},
    {33, 106},
    {47, 183},
}};

union ETC1Tile {
    u64 raw;

    // Each of these two is a collection of 16 bits (one per lookup value)
    BitField<0, 16, u64> table_subindexes;
    BitField<16, 16, u64> negation_flags;

    unsigned GetTableSubIndex(unsigned index) const {
        return (table_subindexes >> index) & 1;
    }

    bool GetNegationFlag(unsigned index) const {
        return ((negation_flags >> index) & 1) == 1;
    }

    BitField<32, 1, u64> flip;
    BitField<33, 1, u64> differential_mode;

    BitField<34, 3, u64> table_index_2;
    BitField<37, 3, u64> table_index_1;

    union {
        // delta value + base value
        BitField<40, 3, s64> db;
        BitField<43, 5, u64> b;

        BitField<48, 3, s64> dg;
        BitField<51, 5, u64> g;

        BitField<56, 3, s64> dr;
        BitField<59, 5, u64> r;
    } differential;

    union {
        BitField<40, 4, u64> b2;
        BitField<44, 4, u64> b1;

        BitField<48, 4, u64> g2;
        BitField<52, 4, u64> g1;

        BitField<56, 4, u64> r2;
        BitField<60, 4, u64> r1;
    } separate;

    const Common::Vec3<u8> GetRGB(unsigned int x, unsigned int y) const {
        int texel = 4 * x + y;

        if (flip)
            std::swap(x, y);

        // Lookup base value
        Common::Vec3<int> ret;
        if (differential_mode) {
            ret.r() = static_cast<int>(differential.r);
            ret.g() = static_cast<int>(differential.g);
            ret.b() = static_cast<int>(differential.b);
            if (x >= 2) {
                ret.r() += static_cast<int>(differential.dr);
                ret.g() += static_cast<int>(differential.dg);
                ret.b() += static_cast<int>(differential.db);
            }
            ret.r() = Common::Color::Convert5To8(ret.r());
            ret.g() = Common::Color::Convert5To8(ret.g());
            ret.b() = Common::Color::Convert5To8(ret.b());
        } else {
            if (x < 2) {
                ret.r() = Common::Color::Convert4To8(static_cast<u8>(separate.r1));
                ret.g() = Common::Color::Convert4To8(static_cast<u8>(separate.g1));
                ret.b() = Common::Color::Convert4To8(static_cast<u8>(separate.b1));
            } else {
                ret.r() = Common::Color::Convert4To8(static_cast<u8>(separate.r2));
                ret.g() = Common::Color::Convert4To8(static_cast<u8>(separate.g2));
                ret.b() = Common::Color::Convert4To8(static_cast<u8>(separate.b2));
            }
        }

        // Add modifier
        unsigned table_index =
            static_cast<int>((x < 2) ? table_index_1.Value() : table_index_2.Value());

        int modifier = etc1_modifier_table[table_index][GetTableSubIndex(texel)];
        if (GetNegationFlag(texel))
            modifier *= -1;

        ret.r() = std::clamp(ret.r() + modifier, 0, 255);
        ret.g() = std::clamp(ret.g() + modifier, 0, 255);
        ret.b() = std::clamp(ret.b() + modifier, 0, 255);

        return ret.Cast<u8>();
    }
};

} // anonymous namespace

Common::Vec3<u8> SampleETC1Subtile(u64 value, unsigned int x, unsigned int y) {
    ETC1Tile tile{value};
    return tile.GetRGB(x, y);
}

void DecodeETC1TileRGBA8(const u8* tile, bool has_alpha, u8 out[8][8][4]) {
    const std::size_t subtile_size = has_alpha ? 16 : 8;
    for (u32 sy = 0; sy < 2; ++sy) {
        for (u32 sx = 0; sx < 2; ++sx) {
            const u8* subtile_ptr = tile + (sx + 2 * sy) * subtile_size;
            // A missing alpha plane decodes as nibble 0xF everywhere, which converts to 255.
            u64 packed_alpha = 0xFFFFFFFFFFFFFFFFull;
            if (has_alpha) {
                std::memcpy(&packed_alpha, subtile_ptr, sizeof(u64));
                subtile_ptr += sizeof(u64);
            }
            u64 value;
            std::memcpy(&value, subtile_ptr, sizeof(u64));
            const ETC1Tile t{value};

            // Base colours for the two 2x4 sub-blocks, exactly as GetRGB computes them per texel.
            int base[2][3];
            if (t.differential_mode) {
                base[0][0] = static_cast<int>(t.differential.r);
                base[0][1] = static_cast<int>(t.differential.g);
                base[0][2] = static_cast<int>(t.differential.b);
                base[1][0] = base[0][0] + static_cast<int>(t.differential.dr);
                base[1][1] = base[0][1] + static_cast<int>(t.differential.dg);
                base[1][2] = base[0][2] + static_cast<int>(t.differential.db);
                for (int blk = 0; blk < 2; ++blk) {
                    for (int ch = 0; ch < 3; ++ch) {
                        base[blk][ch] = Common::Color::Convert5To8(base[blk][ch]);
                    }
                }
            } else {
                base[0][0] = Common::Color::Convert4To8(static_cast<u8>(t.separate.r1));
                base[0][1] = Common::Color::Convert4To8(static_cast<u8>(t.separate.g1));
                base[0][2] = Common::Color::Convert4To8(static_cast<u8>(t.separate.b1));
                base[1][0] = Common::Color::Convert4To8(static_cast<u8>(t.separate.r2));
                base[1][1] = Common::Color::Convert4To8(static_cast<u8>(t.separate.g2));
                base[1][2] = Common::Color::Convert4To8(static_cast<u8>(t.separate.b2));
            }
            const auto& table1 = etc1_modifier_table[t.table_index_1];
            const auto& table2 = etc1_modifier_table[t.table_index_2];

            for (u32 x = 0; x < 4; ++x) {
                for (u32 y = 0; y < 4; ++y) {
                    // The modifier bit index uses the unswapped coordinates; the sub-block and
                    // table selection use the flipped ones. Same rules as GetRGB.
                    const u32 texel = 4 * x + y;
                    u32 bx = x;
                    u32 by = y;
                    if (t.flip) {
                        std::swap(bx, by);
                    }
                    (void)by;
                    const int blk = bx < 2 ? 0 : 1;
                    int modifier =
                        (blk == 0 ? table1 : table2)[t.GetTableSubIndex(texel)];
                    if (t.GetNegationFlag(texel)) {
                        modifier = -modifier;
                    }
                    u8* px = out[sy * 4 + y][sx * 4 + x];
                    px[0] = static_cast<u8>(std::clamp(base[blk][0] + modifier, 0, 255));
                    px[1] = static_cast<u8>(std::clamp(base[blk][1] + modifier, 0, 255));
                    px[2] = static_cast<u8>(std::clamp(base[blk][2] + modifier, 0, 255));
                    px[3] = Common::Color::Convert4To8((packed_alpha >> (4 * texel)) & 0xF);
                }
            }
        }
    }
}

} // namespace Pica::Texture
