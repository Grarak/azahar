// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>

#include <optional>
#include "common/vector_math.h"
#include "video_core/pica/packed_attribute.h"
#include "video_core/pica_types.h"

namespace Pica {

constexpr u32 MAX_PROGRAM_CODE_LENGTH = 4096;
constexpr u32 MAX_SWIZZLE_DATA_LENGTH = 4096;

using ProgramCode = std::array<u32, MAX_PROGRAM_CODE_LENGTH>;
using SwizzleData = std::array<u32, MAX_SWIZZLE_DATA_LENGTH>;

struct Uniforms {
    alignas(16) std::array<Common::Vec4<f24>, 96> f;
    std::array<bool, 16> b;
    std::array<Common::Vec4<u8>, 4> i;

    static std::size_t GetFloatUniformOffset(u32 index) {
        return offsetof(Uniforms, f) + index * sizeof(Common::Vec4<f24>);
    }

    static std::size_t GetBoolUniformOffset(u32 index) {
        return offsetof(Uniforms, b) + index * sizeof(bool);
    }

    static std::size_t GetIntUniformOffset(u32 index) {
        return offsetof(Uniforms, i) + index * sizeof(Common::Vec4<u8>);
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version) {
        ar & f;
        ar & b;
        ar & i;
    }
};

struct ShaderRegs;

/**
 * This structure contains the state information common for all shader units such as uniforms.
 * The geometry shaders has a unique configuration so when enabled it has its own setup.
 */
struct ShaderSetup {
private:
    void MakeProgramCodeDirty() {
        program_code_hash_dirty = true;
        code_sync_dirty = true;
        // program_code_pending_fixup = true;
        // has_fixup = false;
    }

    void MakeSwizzleDataDirty() {
        swizzle_data_hash_dirty = true;
        code_sync_dirty = true;
    }

public:
    explicit ShaderSetup();
    ~ShaderSetup();

    void WriteUniformBoolReg(u32 value);

    void WriteUniformIntReg(u32 index, const Common::Vec4<u8> values);

    std::optional<u32> WriteUniformFloatReg(ShaderRegs& config, u32 value);

    struct UniformWriteRange {
        u32 first_index;
        u32 count;
    };
    std::optional<UniformWriteRange> WriteUniformFloatRegRange(ShaderRegs& config,
                                                               const u32* values, u32 count);

    u64 GetProgramCodeHash();

    u64 GetSwizzleDataHash();

    void DoProgramCodeFixup();

    inline void UpdateProgramCode(size_t offset, u32 value) {
        u32& inst = program_code[offset];
        u32 old = inst;

        if (old == value)
            return;

        inst = value;
        code_sync_dirty = true;
        if (!program_code_hash_dirty) {
            MakeProgramCodeDirty();
        }
        if ((offset + 1) > biggest_program_size) {
            biggest_program_size = offset + 1;
        }
    }

    void UpdateProgramCodeRange(size_t offset, const u32* __restrict values, u32 count);

    void UpdateProgramCode(const ProgramCode& other, u32 other_size = MAX_PROGRAM_CODE_LENGTH) {
        program_code = other;
        biggest_program_size = std::max(biggest_program_size, other_size);
        MakeProgramCodeDirty();
    }

    inline void UpdateSwizzleData(size_t offset, u32 value) {
        u32& data = swizzle_data[offset];
        u32 old = data;

        if (old == value)
            return;

        data = value;
        code_sync_dirty = true;
        if (!swizzle_data_hash_dirty) {
            MakeSwizzleDataDirty();
        }
        if ((offset + 1) > biggest_swizzle_size) {
            biggest_swizzle_size = offset + 1;
        }
    }

    void UpdateSwizzleDataRange(size_t offset, const u32* __restrict values, u32 count);

    void UpdateSwizzleData(const SwizzleData& other, u32 other_size = MAX_SWIZZLE_DATA_LENGTH) {
        swizzle_data = other;
        biggest_swizzle_size = std::max(biggest_swizzle_size, other_size);
        MakeSwizzleDataDirty();
    }

    const ProgramCode& GetProgramCode() const {
        // return (has_fixup) ? program_code_fixup : program_code;
        return program_code;
    }

    const SwizzleData& GetSwizzleData() const {
        return swizzle_data;
    }

    u32 GetBiggestProgramSize() const {
        return biggest_program_size;
    }

    u32 GetBiggestSwizzleSize() const {
        return biggest_swizzle_size;
    }

    void SetRequiresShaderFixup(bool _requires_fixup) {
        // requires_fixup = _requires_fixup;
    }

public:
    Uniforms uniforms;
    PackedAttribute uniform_queue;
    u32 entry_point{};
    const void* cached_shader{};
    bool uniforms_dirty = true;

    /// Cross-thread sync flags for the render-thread mirror: set on every mutation, cleared when
    /// the corresponding block ships. Start dirty so the first shipment seeds the mirror.
    bool uniforms_sync_dirty = true;
    bool code_sync_dirty = true;

    /// Float-uniform window touched since the last shipment, [sync_f_lo, sync_f_hi). Games write
    /// a handful of vectors between draws; shipping the whole 1.5 KB block for each was a
    /// visible slice of the emulation thread's memcpy time. Starts full so the first shipment
    /// seeds the mirror, and deserialization keeps the constructed full window for the same
    /// reason. Bool and int uniforms are 32 bytes together and always travel whole.
    u32 sync_f_lo = 0;
    u32 sync_f_hi = 96;

    void WidenFloatSyncWindow(u32 first, u32 end) {
        sync_f_lo = std::min(sync_f_lo, first);
        sync_f_hi = std::max(sync_f_hi, end);
    }

    // bool requires_fixup = false;
    // bool has_fixup = false;

private:
    ProgramCode program_code{};
    SwizzleData swizzle_data{};
    bool program_code_hash_dirty{true};
    bool swizzle_data_hash_dirty{true};
    u32 biggest_program_size = 0;
    u32 biggest_swizzle_size = 0;
    u64 program_code_hash{0};
    u64 swizzle_data_hash{0};

    // ProgramCode program_code_fixup{};
    // bool program_code_pending_fixup{true};

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const u32 file_version) {
        ar & uniforms;
        ar & uniform_queue;
        ar & program_code;
        ar & swizzle_data;
        ar & program_code_hash_dirty;
        ar & swizzle_data_hash_dirty;
        ar & biggest_program_size;
        ar & biggest_swizzle_size;
        ar & program_code_hash;
        ar & swizzle_data_hash;

        // ar & program_code_fixup;
        // ar & program_code_pending_fixup;
        // ar & requires_fixup;
        // ar & has_fixup;
        if (Archive::is_loading::value) {
            uniforms_dirty = true;
            // A load rewrites everything the mirror thinks it knows; reship it all, whatever
            // the sync flags and float window said before the load.
            uniforms_sync_dirty = true;
            code_sync_dirty = true;
            sync_f_lo = 0;
            sync_f_hi = 96;
        }
    }
};

} // namespace Pica
