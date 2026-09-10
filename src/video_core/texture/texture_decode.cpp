// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/color.h"
#include "common/logging/log.h"
#include "common/swap.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"

using TextureFormat = Pica::TexturingRegs::TextureFormat;

namespace Pica::Texture {

constexpr std::size_t TILE_SIZE = 8 * 8;
constexpr std::size_t ETC1_SUBTILES = 2 * 2;

size_t CalculateTileSize(TextureFormat format) {
    switch (format) {
    case TextureFormat::RGBA8:
        return 4 * TILE_SIZE;

    case TextureFormat::RGB8:
        return 3 * TILE_SIZE;

    case TextureFormat::RGB5A1:
    case TextureFormat::RGB565:
    case TextureFormat::RGBA4:
    case TextureFormat::IA8:
    case TextureFormat::RG8:
        return 2 * TILE_SIZE;

    case TextureFormat::I8:
    case TextureFormat::A8:
    case TextureFormat::IA4:
        return 1 * TILE_SIZE;

    case TextureFormat::I4:
    case TextureFormat::A4:
        return TILE_SIZE / 2;

    case TextureFormat::ETC1:
        return ETC1_SUBTILES * 8;

    case TextureFormat::ETC1A4:
        return ETC1_SUBTILES * 16;

    default: // placeholder for yet unknown formats
        UNIMPLEMENTED();
        return 0;
    }
}

Common::Vec4<u8> LookupTexture(const u8* source, unsigned int x, unsigned int y,
                               const TextureInfo& info, bool disable_alpha) {
    // Coordinate in tiles
    const unsigned int coarse_x = x / 8;
    const unsigned int coarse_y = y / 8;

    // Coordinate inside the tile
    const unsigned int fine_x = x % 8;
    const unsigned int fine_y = y % 8;

    const u8* line = source + coarse_y * info.stride;
    const u8* tile = line + coarse_x * CalculateTileSize(info.format);
    return LookupTexelInTile(tile, fine_x, fine_y, info, disable_alpha);
}

void DecodeTileRGBA8(u32* dst, u32 dst_stride, const u8* tile, const TextureInfo& info) {
    using VideoCore::MortonInterleaveX;
    using VideoCore::MortonInterleaveY;
    using namespace Common::Color;

    const auto pack = [](u8 r, u8 g, u8 b, u8 a) {
        return static_cast<u32>(r) | (static_cast<u32>(g) << 8) | (static_cast<u32>(b) << 16) |
               (static_cast<u32>(a) << 24);
    };

    switch (info.format) {
    case TextureFormat::RGBA8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRGBA8(tile + (MortonInterleaveX(x) + my) * 4);
                row[x] = pack(c.r(), c.g(), c.b(), c.a());
            }
        }
        return;
    case TextureFormat::RGB8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRGB8(tile + (MortonInterleaveX(x) + my) * 3);
                row[x] = pack(c.r(), c.g(), c.b(), 255);
            }
        }
        return;
    case TextureFormat::RGB5A1:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRGB5A1(tile + (MortonInterleaveX(x) + my) * 2);
                row[x] = pack(c.r(), c.g(), c.b(), c.a());
            }
        }
        return;
    case TextureFormat::RGB565:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRGB565(tile + (MortonInterleaveX(x) + my) * 2);
                row[x] = pack(c.r(), c.g(), c.b(), 255);
            }
        }
        return;
    case TextureFormat::RGBA4:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRGBA4(tile + (MortonInterleaveX(x) + my) * 2);
                row[x] = pack(c.r(), c.g(), c.b(), c.a());
            }
        }
        return;
    case TextureFormat::IA8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const u8* p = tile + (MortonInterleaveX(x) + my) * 2;
                row[x] = pack(p[1], p[1], p[1], p[0]);
            }
        }
        return;
    case TextureFormat::RG8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = DecodeRG8(tile + (MortonInterleaveX(x) + my) * 2);
                row[x] = pack(c.r(), c.g(), 0, 255);
            }
        }
        return;
    case TextureFormat::I8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const u8 v = tile[MortonInterleaveX(x) + my];
                row[x] = pack(v, v, v, 255);
            }
        }
        return;
    case TextureFormat::A8:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                row[x] = pack(0, 0, 0, tile[MortonInterleaveX(x) + my]);
            }
        }
        return;
    case TextureFormat::IA4:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const u8 v = tile[MortonInterleaveX(x) + my];
                const u8 i = Convert4To8((v & 0xF0) >> 4);
                const u8 a = Convert4To8(v & 0xF);
                row[x] = pack(i, i, i, a);
            }
        }
        return;
    case TextureFormat::I4:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const u32 m = MortonInterleaveX(x) + my;
                const u8 v = tile[m / 2];
                const u8 i = Convert4To8((m % 2) ? ((v & 0xF0) >> 4) : (v & 0xF));
                row[x] = pack(i, i, i, 255);
            }
        }
        return;
    case TextureFormat::A4:
        for (u32 y = 0; y < 8; ++y) {
            const u32 my = MortonInterleaveY(y);
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const u32 m = MortonInterleaveX(x) + my;
                const u8 v = tile[m / 2];
                row[x] = pack(0, 0, 0, Convert4To8((m % 2) ? ((v & 0xF0) >> 4) : (v & 0xF)));
            }
        }
        return;
    case TextureFormat::ETC1:
    case TextureFormat::ETC1A4: {
        u8 rgba[8][8][4];
        DecodeETC1TileRGBA8(tile, info.format == TextureFormat::ETC1A4, rgba);
        for (u32 y = 0; y < 8; ++y) {
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                row[x] = pack(rgba[y][x][0], rgba[y][x][1], rgba[y][x][2], rgba[y][x][3]);
            }
        }
        return;
    }
    default:
        // Unknown format: keep the per-texel path's behaviour.
        for (u32 y = 0; y < 8; ++y) {
            u32* row = dst + y * dst_stride;
            for (u32 x = 0; x < 8; ++x) {
                const auto c = LookupTexelInTile(tile, x, y, info, false);
                row[x] = pack(c.r(), c.g(), c.b(), c.a());
            }
        }
        return;
    }
}

Common::Vec4<u8> LookupTexelInTile(const u8* source, unsigned int x, unsigned int y,
                                   const TextureInfo& info, bool disable_alpha) {
    DEBUG_ASSERT(x < 8);
    DEBUG_ASSERT(y < 8);

    using VideoCore::MortonInterleave;

    switch (info.format) {
    case TextureFormat::RGBA8: {
        auto res = Common::Color::DecodeRGBA8(source + MortonInterleave(x, y) * 4);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB8: {
        auto res = Common::Color::DecodeRGB8(source + MortonInterleave(x, y) * 3);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGB5A1: {
        auto res = Common::Color::DecodeRGB5A1(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB565: {
        auto res = Common::Color::DecodeRGB565(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGBA4: {
        auto res = Common::Color::DecodeRGBA4(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::IA8: {
        const u8* source_ptr = source + MortonInterleave(x, y) * 2;

        if (disable_alpha) {
            // Show intensity as red, alpha as green
            return {source_ptr[1], source_ptr[0], 0, 255};
        } else {
            return {source_ptr[1], source_ptr[1], source_ptr[1], source_ptr[0]};
        }
    }

    case TextureFormat::RG8: {
        auto res = Common::Color::DecodeRG8(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), 0, 255};
    }

    case TextureFormat::I8: {
        const u8* source_ptr = source + MortonInterleave(x, y);
        return {*source_ptr, *source_ptr, *source_ptr, 255};
    }

    case TextureFormat::A8: {
        const u8* source_ptr = source + MortonInterleave(x, y);

        if (disable_alpha) {
            return {*source_ptr, *source_ptr, *source_ptr, 255};
        } else {
            return {0, 0, 0, *source_ptr};
        }
    }

    case TextureFormat::IA4: {
        const u8* source_ptr = source + MortonInterleave(x, y);

        u8 i = Common::Color::Convert4To8(((*source_ptr) & 0xF0) >> 4);
        u8 a = Common::Color::Convert4To8((*source_ptr) & 0xF);

        if (disable_alpha) {
            // Show intensity as red, alpha as green
            return {i, a, 0, 255};
        } else {
            return {i, i, i, a};
        }
    }

    case TextureFormat::I4: {
        u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 i = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        i = Common::Color::Convert4To8(i);

        return {i, i, i, 255};
    }

    case TextureFormat::A4: {
        u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 a = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        a = Common::Color::Convert4To8(a);

        if (disable_alpha) {
            return {a, a, a, 255};
        } else {
            return {0, 0, 0, a};
        }
    }

    case TextureFormat::ETC1:
    case TextureFormat::ETC1A4: {
        bool has_alpha = (info.format == TextureFormat::ETC1A4);
        std::size_t subtile_size = has_alpha ? 16 : 8;

        // ETC1 further subdivides each 8x8 tile into four 4x4 subtiles
        constexpr unsigned int subtile_width = 4;
        constexpr unsigned int subtile_height = 4;

        unsigned int subtile_index = (x / subtile_width) + 2 * (y / subtile_height);
        x %= subtile_width;
        y %= subtile_height;

        const u8* subtile_ptr = source + subtile_index * subtile_size;

        u8 alpha = 255;
        if (has_alpha) {
            u64_le packed_alpha;
            std::memcpy(&packed_alpha, subtile_ptr, sizeof(u64));
            subtile_ptr += sizeof(u64);

            alpha =
                Common::Color::Convert4To8((packed_alpha >> (4 * (x * subtile_width + y))) & 0xF);
        }

        u64_le subtile_data;
        std::memcpy(&subtile_data, subtile_ptr, sizeof(u64));

        return Common::MakeVec(SampleETC1Subtile(subtile_data, x, y),
                               disable_alpha ? (u8)255 : alpha);
    }

    default:
        LOG_ERROR(HW_GPU, "Unknown texture format: {:x}", (u32)info.format);
        DEBUG_ASSERT(false);
        return {};
    }
}

TextureInfo TextureInfo::FromPicaRegister(const TexturingRegs::TextureConfig& config,
                                          const TexturingRegs::TextureFormat& format) {
    TextureInfo info;
    info.physical_address = config.GetPhysicalAddress();
    info.width = config.width;
    info.height = config.height;
    info.format = format;
    info.SetDefaultStride();
    info.is_shadow_source = config.type == TexturingRegs::TextureConfig::TextureType::Shadow2D ||
                            config.type == TexturingRegs::TextureConfig::TextureType::ShadowCube;
    return info;
}

} // namespace Pica::Texture
