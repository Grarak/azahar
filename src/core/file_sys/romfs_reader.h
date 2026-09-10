// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <shared_mutex>
#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include "common/alignment.h"
#include "common/common_types.h"
#include "common/file_util.h"
#include <unordered_map>
#include "common/host_shared_memory.h"
#include "core/file_sys/artic_cache.h"
#include "network/artic_base/artic_base_client.h"

namespace Loader {
enum class ResultStatus;
}

namespace FileSys {

/**
 * Interface for reading RomFS data.
 */
class RomFSReader {
public:
    virtual ~RomFSReader() = default;

    virtual std::size_t GetSize() const = 0;
    virtual std::size_t ReadFile(std::size_t offset, std::size_t length, u8* buffer) = 0;
    virtual bool AllowsCachedReads() const = 0;
    virtual bool CacheReady(std::size_t file_offset, std::size_t length) = 0;

private:
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {}
    friend class boost::serialization::access;
};

/**
 * A RomFS reader that directly reads the RomFS file.
 */
class DirectRomFSReader : public RomFSReader {
public:
    DirectRomFSReader(std::unique_ptr<FileUtil::IOFileBase>&& file)
        : file(std::move(file)), cache(PickCacheSpec()) {}

    ~DirectRomFSReader() override = default;

    std::size_t GetSize() const override {
        return file->GetSize();
    }

    std::size_t ReadFile(std::size_t offset, std::size_t length, u8* buffer) override;

    bool AllowsCachedReads() const override;

    bool CacheReady(std::size_t file_offset, std::size_t length) override;

private:
    std::unique_ptr<FileUtil::IOFileBase> file;

    /**
     * The read cache: whole lines of the file, least recently used out. Sized at
     * construction, because the console decides the size: its card reads are slow enough per
     * call that the first reader made - the title's own RomFS - gets 64 KiB lines and 16 MiB
     * of them, kept in the physically contiguous pool the heap cannot reach; every other
     * reader (update RomFS, system archives) and every other platform keeps the 16 x 8 KiB
     * the cache always had - out of the heap, because the pool holds one of these and no
     * more. A read larger than a line bypasses it, as before.
     */
    struct CacheSpec {
        std::size_t line_size;
        std::size_t line_count;
        const char* name;
        /// Where the lines live. The title's own cache is the one block the Vita's physically
        /// contiguous pool exists for, and is sized to fit it; nothing else asks.
        Common::HostSharedMemory::Placement placement =
            Common::HostSharedMemory::Placement::Heap;
    };
    class LineCache {
    public:
        explicit LineCache(const CacheSpec& spec);
        [[nodiscard]] std::size_t LineSize() const {
            return line_size;
        }
        /// The line holding `page`, and whether it was already present. A missing line is
        /// the least recently used one, handed over for the caller to fill.
        std::pair<bool, u8*> Request(std::size_t page);

    private:
        std::size_t line_size;
        std::size_t line_count;
        Common::HostSharedMemory storage;
        std::vector<std::size_t> line_page; ///< page held by each line (npos: empty)
        std::vector<u64> line_used;         ///< last use, for eviction
        std::unordered_map<std::size_t, std::size_t> index; ///< page -> line
        u64 clock = 0;
    };
    LineCache cache;
    // TODO(PabloMK7): Make cache thread safe, read the comment in CacheReady function.
    // std::shared_mutex cache_mutex;

    [[nodiscard]] std::size_t cache_line_size() const {
        return cache.LineSize();
    }

    static CacheSpec PickCacheSpec();

    DirectRomFSReader() : cache(PickCacheSpec()) {}

    std::size_t OffsetToPage(std::size_t offset) {
        return Common::AlignDown<std::size_t>(offset, cache_line_size());
    }

    std::vector<std::pair<std::size_t, std::size_t>> BreakupRead(std::size_t offset,
                                                                 std::size_t length);

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<RomFSReader>(*this);
        ar & file;
    }
    friend class boost::serialization::access;
};

/**
 * A RomFS reader that reads from an artic base server.
 */
class ArticRomFSReader : public RomFSReader {
public:
    ArticRomFSReader() = default;
    ArticRomFSReader(std::shared_ptr<Network::ArticBase::Client>& cli, bool is_update_romfs);

    ~ArticRomFSReader() override;

    std::size_t GetSize() const override {
        return data_size;
    }

    std::size_t ReadFile(std::size_t offset, std::size_t length, u8* buffer) override;

    bool AllowsCachedReads() const override;

    bool CacheReady(std::size_t file_offset, std::size_t length) override;

    Loader::ResultStatus OpenStatus() {
        return load_status;
    }

    void ClearCache() {
        cache.Clear();
    }

    void CloseFile();

private:
    std::shared_ptr<Network::ArticBase::Client> client;
    size_t data_size = 0;
    s32 romfs_handle = -1;
    Loader::ResultStatus load_status;

    ArticCache cache;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<RomFSReader>(*this);
        ar & data_size;
    }
    friend class boost::serialization::access;
};

} // namespace FileSys

BOOST_CLASS_EXPORT_KEY(FileSys::DirectRomFSReader)
BOOST_CLASS_EXPORT_KEY(FileSys::ArticRomFSReader)
