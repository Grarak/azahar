// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <sstream>
#include <zstd.h>
#include <cryptopp/hex.h>
#include <fmt/ranges.h>
#include "common/archives.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "common/swap.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "core/hle/kernel/thread.h"
#include "core/loader/loader.h"
#include "core/movie.h"
#include "core/savestate.h"
#include "video_core/gpu.h"
#include "network/network.h"

namespace Core {

namespace {

/**
 * Compresses everything written through it straight into a file, so a save never materializes
 * the raw serialized image. On a 32-bit host that image (~150 MB) plus its compressed copy was
 * enough to exhaust the address space once the GL renderer and its JIT moved in.
 */
class ZstdFileOStreamBuf final : public std::streambuf {
public:
    explicit ZstdFileOStreamBuf(FileUtil::IOFile& file_) : file{file_} {
        cctx = ZSTD_createCCtx();
        in_buffer.resize(ZSTD_CStreamInSize());
        out_buffer.resize(ZSTD_CStreamOutSize());
        setp(reinterpret_cast<char*>(in_buffer.data()),
             reinterpret_cast<char*>(in_buffer.data() + in_buffer.size()));
    }

    ~ZstdFileOStreamBuf() override {
        ZSTD_freeCCtx(cctx);
    }

    /// Flushes remaining input and the zstd epilogue. True if every write reached the file.
    bool Finish() {
        Compress(ZSTD_e_end);
        return ok;
    }

private:
    int_type overflow(int_type ch) override {
        Compress(ZSTD_e_continue);
        if (ch != traits_type::eof()) {
            *pptr() = static_cast<char>(ch);
            pbump(1);
        }
        return ok ? ch : traits_type::eof();
    }

    int sync() override {
        Compress(ZSTD_e_continue);
        return ok ? 0 : -1;
    }

    void Compress(ZSTD_EndDirective mode) {
        ZSTD_inBuffer input{in_buffer.data(), static_cast<std::size_t>(pptr() - pbase()), 0};
        std::size_t remaining;
        do {
            ZSTD_outBuffer output{out_buffer.data(), out_buffer.size(), 0};
            remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
            if (ZSTD_isError(remaining)) {
                ok = false;
                return;
            }
            if (output.pos != 0 && file.WriteBytes(out_buffer.data(), output.pos) != output.pos) {
                ok = false;
                return;
            }
        } while (mode == ZSTD_e_end ? remaining != 0 : input.pos < input.size);
        setp(reinterpret_cast<char*>(in_buffer.data()),
             reinterpret_cast<char*>(in_buffer.data() + in_buffer.size()));
    }

    FileUtil::IOFile& file;
    ZSTD_CCtx* cctx{};
    std::vector<u8> in_buffer;
    std::vector<u8> out_buffer;
    bool ok = true;
};

/// Decompresses a zstd frame on demand as the archive reads, avoiding the full decompressed
/// image (the compressed input stays in memory; it is a fifth of the size).
class ZstdIStreamBuf final : public std::streambuf {
public:
    explicit ZstdIStreamBuf(std::span<const u8> compressed_) : compressed{compressed_} {
        dctx = ZSTD_createDCtx();
        out_buffer.resize(ZSTD_DStreamOutSize());
        input = {compressed.data(), compressed.size(), 0};
    }

    ~ZstdIStreamBuf() override {
        ZSTD_freeDCtx(dctx);
    }

private:
    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        ZSTD_outBuffer output{out_buffer.data(), out_buffer.size(), 0};
        while (output.pos == 0) {
            if (input.pos >= input.size) {
                return traits_type::eof();
            }
            const std::size_t ret = ZSTD_decompressStream(dctx, &output, &input);
            if (ZSTD_isError(ret)) {
                return traits_type::eof();
            }
            if (ret == 0 && output.pos == 0) {
                return traits_type::eof();
            }
        }
        char* base = reinterpret_cast<char*>(out_buffer.data());
        setg(base, base, base + output.pos);
        return traits_type::to_int_type(*gptr());
    }

    std::span<const u8> compressed;
    ZSTD_DCtx* dctx{};
    ZSTD_inBuffer input{};
    std::vector<u8> out_buffer;
};

} // Anonymous namespace

#pragma pack(push, 1)
struct CSTHeader {
    std::array<u8, 4> filetype{};    /// Unique Identifier to check the file type (always "CST"0x1B)
    u64_le program_id{};             /// ID of the ROM being executed. Also called title_id
    std::array<u8, 20> revision{};   /// Git hash of the revision this savestate was created with
    u64_le time{};                   /// The time when this save state was created
    std::array<u8, 20> build_name{}; /// The build name (Canary/Nightly) with the version number
    u32_le zero{};                   /// Should be zero, just in case.
    std::array<u8, 20> build_version{}; /// Latest build version, used as compatibility.
    u32_le zero_2{};                    /// Should be zero, just in case.

    std::array<u8, 168> reserved{}; /// Make heading 256 bytes so it has consistent size
};
static_assert(sizeof(CSTHeader) == 256, "CSTHeader should be 256 bytes");
#pragma pack(pop)

constexpr std::array<u8, 4> header_magic_bytes{{'C', 'S', 'T', 0x1B}};

static std::string GetSaveStatePath(u64 program_id, u64 movie_id, u32 slot) {
    if (movie_id) {
        return fmt::format("{}{:016X}.movie{:016X}.{:02d}.cst",
                           FileUtil::GetUserPath(FileUtil::UserPath::StatesDir), program_id,
                           movie_id, slot);
    } else {
        return fmt::format("{}{:016X}.{:02d}.cst",
                           FileUtil::GetUserPath(FileUtil::UserPath::StatesDir), program_id, slot);
    }
}

static bool ValidateSaveState(const CSTHeader& header, SaveStateInfo& info, u64 program_id,
                              u64 movie_id) {
    const auto path = GetSaveStatePath(program_id, movie_id, info.slot);
    if (header.filetype != header_magic_bytes) {
        LOG_WARNING(Core, "Invalid save state file {}", path);
        return false;
    }
    info.time = header.time;

    if (header.program_id != program_id) {
        LOG_WARNING(Core, "Save state file isn't for the current game {}", path);
        return false;
    }
    const std::string revision = fmt::format("{:02x}", fmt::join(header.revision, ""));
    const std::string build_name =
        header.zero == 0 ? reinterpret_cast<const char*>(header.build_name.data()) : "";
    const std::string build_version =
        header.zero_2 == 0 ? reinterpret_cast<const char*>(header.build_version.data()) : "";

    if (revision == Common::g_scm_rev) {
        info.status = SaveStateInfo::ValidationStatus::OK;
    } else {
        info.build_name = build_name;
        info.build_version = build_version;

        info.status = Common::g_build_version == info.build_version
                          ? SaveStateInfo::ValidationStatus::RevisionMismatch
                          : SaveStateInfo::ValidationStatus::BuildMismatch;
    }
    return true;
}

std::vector<SaveStateInfo> ListSaveStates(u64 program_id, u64 movie_id) {
    std::vector<SaveStateInfo> result;
    result.reserve(SaveStateSlotCount);
    for (u32 slot = 0; slot <= SaveStateSlotCount; ++slot) {
        const auto path = GetSaveStatePath(program_id, movie_id, slot);
        if (!FileUtil::Exists(path)) {
            continue;
        }

        SaveStateInfo info;
        info.slot = slot;

        FileUtil::IOFile file(path, "rb");
        if (!file) {
            LOG_ERROR(Core, "Could not open file {}", path);
            continue;
        }
        CSTHeader header;
        if (file.GetSize() < sizeof(header)) {
            LOG_ERROR(Core, "File too small {}", path);
            continue;
        }
        if (file.ReadBytes(&header, sizeof(header)) != sizeof(header)) {
            LOG_ERROR(Core, "Could not read from file {}", path);
            continue;
        }
        if (!ValidateSaveState(header, info, program_id, movie_id)) {
            continue;
        }

        result.emplace_back(std::move(info));
    }
    return result;
}

SaveStateInfo GetSaveStateInfo(u64 program_id, u64 movie_id, u32 slot) {
    SaveStateInfo info{};
    info.slot = std::numeric_limits<u32>::max();

    const auto path = GetSaveStatePath(program_id, movie_id, slot);
    if (!FileUtil::Exists(path)) {
        return info;
    }

    FileUtil::IOFile file(path, "rb");
    if (!file) {
        LOG_ERROR(Core, "Could not open file {}", path);
        return info;
    }
    CSTHeader header;
    if (file.GetSize() < sizeof(header)) {
        LOG_ERROR(Core, "File too small {}", path);
        return info;
    }
    if (file.ReadBytes(&header, sizeof(header)) != sizeof(header)) {
        LOG_ERROR(Core, "Could not read from file {}", path);
        return info;
    }
    if (ValidateSaveState(header, info, program_id, movie_id)) {
        info.slot = slot;
    }
    return info;
}

void System::SaveState(u32 slot) const {
    if (app_loader) {
        if (!app_loader->SupportsSaveStates()) {
            throw std::runtime_error("The current app loader doesn't support save states");
        }
    }

    const u64 movie_id = movie.GetCurrentMovieID();
    const auto path = GetSaveStatePath(title_id, movie_id, slot);
    if (!FileUtil::CreateFullPath(path)) {
        throw std::runtime_error("Could not create path " + path);
    }

    FileUtil::IOFile file(path, "wb");
    if (!file) {
        throw std::runtime_error("Could not open file " + path);
    }

    CSTHeader header{};
    header.filetype = header_magic_bytes;
    header.program_id = title_id;
    std::string rev_bytes;
    CryptoPP::StringSource ss(Common::g_scm_rev, true,
                              new CryptoPP::HexDecoder(new CryptoPP::StringSink(rev_bytes)));
    std::memcpy(header.revision.data(), rev_bytes.data(), sizeof(header.revision));
    header.time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    const std::string build_fullname = Common::g_build_fullname;
    std::memset(header.build_name.data(), 0, sizeof(header.build_name));
    std::memcpy(header.build_name.data(), build_fullname.c_str(),
                std::min(build_fullname.length(), sizeof(header.build_name) - 1));

    if (file.WriteBytes(&header, sizeof(header)) != sizeof(header)) {
        throw std::runtime_error("Could not write to file " + path);
    }

    // Serialize straight through the compressor into the file.
    ZstdFileOStreamBuf zstd_buf{file};
    std::ostream stream{&zstd_buf};
    {
        oarchive oa{stream};
        oa&* this;
    }
    stream.flush();
    if (!zstd_buf.Finish()) {
        throw std::runtime_error("Could not write to file " + path);
    }
}

/**
 * Puts back whatever deserialization cannot.
 *
 * A save state restores the page table and the backing memory wholesale, without going through the
 * calls that normally announce such changes. Anything caching a view of guest memory is therefore
 * stale, and under native execution that means the host address space still points at the memory
 * the guest had before the load.
 */
void System::AfterStateLoaded() {
    memory->RefreshMappingObserver();

    for (u32 core_id = 0; core_id < static_cast<u32>(cpu_cores.size()); core_id++) {
        auto& cpu_core = *cpu_cores[core_id];
        cpu_core.ClearInstructionCache();

        // Deserialization restores each thread's saved context but never puts the running one back
        // into the CPU, because that normally happens on a context switch and a load is not one.
        // Left alone the core keeps the register file it had before the load while the memory
        // under it has been replaced wholesale, which a backend executing guest code directly does
        // not survive.
        if (Kernel::Thread* thread = kernel->GetThreadManager(core_id).GetCurrentThread()) {
            cpu_core.LoadContext(thread->context);
        }
    }
}

void System::LoadState(u32 slot) {
    if (app_loader) {
        if (!app_loader->SupportsSaveStates()) {
            throw std::runtime_error("The current app loader doesn't support save states");
        }
    }
#ifdef ENABLE_ROOM
    // The room member only exists once Network::Init has run, which some frontends skip.
    if (auto room_member = Network::GetRoomMember().lock()) {
        if (room_member->IsConnected()) {
            throw std::runtime_error("Unable to load while connected to multiplayer");
        }
    }
#endif

    const u64 movie_id = movie.GetCurrentMovieID();
    const auto path = GetSaveStatePath(title_id, movie_id, slot);

    std::vector<u8> buffer(FileUtil::GetSize(path) - sizeof(CSTHeader));
    {
        FileUtil::IOFile file(path, "rb");

        // load header
        CSTHeader header;
        if (file.ReadBytes(&header, sizeof(header)) != sizeof(header)) {
            throw std::runtime_error("Could not read from file at " + path);
        }

        // validate header
        SaveStateInfo info;
        info.slot = slot;
        if (!ValidateSaveState(header, info, title_id, movie_id) ||
            (!load_state_any_build &&
             info.status == SaveStateInfo::ValidationStatus::BuildMismatch)) {
            throw std::runtime_error("Invalid savestate");
        }

        if (file.ReadBytes(buffer.data(), buffer.size()) != buffer.size()) {
            throw std::runtime_error("Could not read from file at " + path);
        }
    }

    LOG_INFO(Core, "savestate: {} compressed bytes read", buffer.size());
    // Deserialize, decompressing as the archive reads; only the compressed image is resident.
    {
        ZstdIStreamBuf zstd_buf{buffer};
        std::istream stream{&zstd_buf};
        iarchive ia{stream};
        LOG_INFO(Core, "savestate: archive open");
        ia&* this;
    }
    LOG_INFO(Core, "savestate: deserialized");

    AfterStateLoaded();
    LOG_INFO(Core, "savestate: after-load fixups done");
}

std::vector<u8> System::SaveStateBuffer() const {
    std::ostringstream sstream{std::ios_base::binary};
    // Serialize
    oarchive oa{sstream};
    oa&* this;

    const std::string& str{sstream.str()};
    const auto data = std::span<const u8>{reinterpret_cast<const u8*>(str.data()), str.size()};
    auto buffer = Common::Compression::CompressDataZSTDDefault(data);

    CSTHeader header{};
    header.filetype = header_magic_bytes;
    header.program_id = title_id;
    std::string rev_bytes;
    CryptoPP::StringSource ss(Common::g_scm_rev, true,
                              new CryptoPP::HexDecoder(new CryptoPP::StringSink(rev_bytes)));
    std::memcpy(header.revision.data(), rev_bytes.data(), sizeof(header.revision));
    header.time = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
    const std::string build_fullname = Common::g_build_fullname;
    std::memset(header.build_name.data(), 0, sizeof(header.build_name));
    std::memcpy(header.build_name.data(), build_fullname.c_str(),
                std::min(build_fullname.length(), sizeof(header.build_name) - 1));

    const std::string build_version = Common::g_build_version;
    std::memset(header.build_version.data(), 0, sizeof(header.build_version));
    std::memcpy(header.build_version.data(), build_version.c_str(),
                std::min(build_version.length(), sizeof(header.build_version) - 1));

    std::vector<u8> result((u8*)&header, (u8*)&header + sizeof(header));
    std::copy(buffer.begin(), buffer.end(), std::back_inserter(result));

    return result;
}

bool System::LoadStateBuffer(std::vector<u8> buffer) {
    CSTHeader header;

    if (buffer.size() < sizeof(header)) {
        LOG_ERROR(Core, "Save state too small");
        return false;
    }

    header = *((CSTHeader*)buffer.data());

    if (header.filetype != header_magic_bytes) {
        LOG_ERROR(Core, "Invalid save state");
        return false;
    }

    if (header.program_id != title_id) {
        LOG_ERROR(Core, "Save state isn't for the current game");
        return false;
    }
    std::string revision = fmt::format("{:02x}", fmt::join(header.revision, ""));
    if (revision != Common::g_scm_rev) {
        LOG_ERROR(Core,
                  "Save state file created from a different revision (core: {}, savestate: {})",
                  Common::g_scm_rev, revision);
        return false;
    }

    std::vector<u8> state(buffer.begin() + sizeof(CSTHeader), buffer.end());
    auto decompressed = Common::Compression::DecompressDataZSTD(state);

    std::istringstream sstream{
        std::string{reinterpret_cast<char*>(decompressed.data()), decompressed.size()},
        std::ios_base::binary};
    decompressed.clear();

    // Deserialize
    {
        iarchive ia{sstream};
        ia&* this;
    }

    AfterStateLoaded();
    return true;
}

} // namespace Core
