// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <span>
#include <vector>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/host_shared_memory.h"

/// Abstract host-side memory - for example a static buffer, or local vector
class BackingMem {
public:
    virtual ~BackingMem() = default;
    virtual u8* GetPtr() = 0;
    virtual const u8* GetPtr() const = 0;
    virtual std::size_t GetSize() const = 0;

    /**
     * The aliasable block this memory lives in, if it has one.
     *
     * Backing memory that answers this can be mapped a second time at another host address, which
     * is what the native ARM backend needs: guest memory has to appear at the address the guest
     * itself uses, with writes through either view visible to the other. Backing memory that
     * returns nullptr simply cannot be reached by natively executing guest code.
     */
    virtual const Common::HostSharedMemory* AliasableBlock() const {
        return nullptr;
    }

    /// Offset of GetPtr() within AliasableBlock().
    virtual std::size_t AliasableOffset() const {
        return 0;
    }

private:
    template <class Archive>
    void serialize(Archive&, const unsigned int) {}
    friend class boost::serialization::access;
};

/// Backing memory implemented by a local buffer
class BufferMem : public BackingMem {
public:
    BufferMem() = default;
    explicit BufferMem(std::size_t size)
        : data(std::make_unique<Common::HostSharedMemory>(size, "azahar-buffer")) {}

    u8* GetPtr() override {
        return data ? data->Data() : nullptr;
    }

    const u8* GetPtr() const override {
        return data ? data->Data() : nullptr;
    }

    std::size_t GetSize() const override {
        return data ? data->Size() : 0;
    }

    const Common::HostSharedMemory* AliasableBlock() const override {
        return (data && data->SupportsAliasing()) ? data.get() : nullptr;
    }

private:
    // Aliasable rather than a plain vector: IPC mapped buffers are backed by this and are visible
    // to the guest, so natively executing guest code has to be able to reach them.
    std::unique_ptr<Common::HostSharedMemory> data;

    template <class Archive>
    void save(Archive& ar, const unsigned int) const {
        ar& boost::serialization::base_object<BackingMem>(*this);
        const u64 size = GetSize();
        ar & size;
        if (size > 0) {
            ar& boost::serialization::make_binary_object(GetPtr(), size);
        }
    }

    template <class Archive>
    void load(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<BackingMem>(*this);
        u64 size = 0;
        ar & size;
        data = size > 0 ? std::make_unique<Common::HostSharedMemory>(
                              static_cast<std::size_t>(size), "azahar-buffer")
                        : nullptr;
        if (size > 0) {
            ar& boost::serialization::make_binary_object(GetPtr(), size);
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
    friend class boost::serialization::access;
};

BOOST_CLASS_EXPORT_KEY(BufferMem);

/**
 * A managed reference to host-side memory.
 * Fast enough to be used everywhere instead of u8*
 * Supports serialization.
 */
class MemoryRef {
public:
    MemoryRef() = default;
    MemoryRef(std::nullptr_t) {}

    MemoryRef(std::shared_ptr<BackingMem> backing_mem_)
        : backing_mem(std::move(backing_mem_)), offset(0) {
        Init();
    }
    MemoryRef(std::shared_ptr<BackingMem> backing_mem_, u64 offset_)
        : backing_mem(std::move(backing_mem_)), offset(offset_) {
        ASSERT(offset <= backing_mem->GetSize());
        Init();
    }

    const std::shared_ptr<BackingMem>& GetBackingMem() const {
        return backing_mem;
    }
    u64 GetOffset() const {
        return offset;
    }

    explicit operator bool() const {
        return cptr != nullptr;
    }

    operator u8*() {
        return cptr;
    }

    u8* GetPtr() {
        return cptr;
    }

    operator const u8*() const {
        return cptr;
    }

    const u8* GetPtr() const {
        return cptr;
    }

    std::span<u8> GetWriteBytes(std::size_t size) {
        return std::span{cptr, std::min(size, csize)};
    }

    template <typename T>
    std::span<const T> GetReadBytes(std::size_t size) const {
        const auto* cptr_t = reinterpret_cast<T*>(cptr);
        return std::span{cptr_t, std::min(size, csize) / sizeof(T)};
    }

    std::size_t GetSize() const {
        return csize;
    }

    /**
     * The aliasable block this reference points into, if any.
     * @param block_offset Receives the offset of GetPtr() within that block.
     */
    const Common::HostSharedMemory* AliasableBlock(std::size_t& block_offset) const {
        if (!backing_mem) {
            return nullptr;
        }
        const Common::HostSharedMemory* block = backing_mem->AliasableBlock();
        if (block == nullptr) {
            return nullptr;
        }
        block_offset = backing_mem->AliasableOffset() + static_cast<std::size_t>(offset);
        return block;
    }

    MemoryRef& operator+=(u32 offset_by) {
        ASSERT(offset_by < csize);
        offset += offset_by;
        Init();
        return *this;
    }

    MemoryRef operator+(u32 offset_by) const {
        ASSERT(offset_by < csize);
        return MemoryRef(backing_mem, offset + offset_by);
    }

private:
    std::shared_ptr<BackingMem> backing_mem{};
    u64 offset{};
    // Cached values for speed
    u8* cptr{};
    std::size_t csize{};

    void Init() {
        if (backing_mem) {
            cptr = backing_mem->GetPtr() + offset;
            csize = static_cast<std::size_t>(backing_mem->GetSize() - offset);
        } else {
            cptr = nullptr;
            csize = 0;
        }
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar & backing_mem;
        ar & offset;
        if (Archive::is_loading::value) {
            Init();
        }
    }
    friend class boost::serialization::access;
};
