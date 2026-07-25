// rag/store/container.cpp — .ragdb container serialization + CRC32.

#include "rag/store/container.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

namespace rag::store {

// ─── CRC32 (IEEE 802.3, reflected) ────────────────────────────────────────────
std::uint32_t crc32(std::string_view data) noexcept {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    std::uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char b : data) crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ─── Serialize ────────────────────────────────────────────────────────────────
std::string Container::serialize() const {
    Writer w;
    // Header (32 bytes).
    w.bytes(std::string_view(kMagic, 8));
    w.u<std::uint16_t>(major_);
    w.u<std::uint16_t>(minor_);
    w.u<std::uint32_t>(flags_);
    w.u<std::uint32_t>(static_cast<std::uint32_t>(sections_.size()));
    w.u<std::uint64_t>(0);  // reserved

    // Compute payload offsets: they start after header + table.
    const std::uint64_t header_size = 8 + 2 + 2 + 4 + 4 + 8;                 // 28
    const std::uint64_t entry_size  = 4 + 8 + 8;                            // 20
    std::uint64_t table_size = entry_size * sections_.size();
    std::uint64_t cursor = header_size + table_size;

    // Section table.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> layout; // (tag, offset)
    for (const auto& [tag, payload] : sections_) {
        w.u<std::uint32_t>(tag);
        w.u<std::uint64_t>(cursor);
        w.u<std::uint64_t>(static_cast<std::uint64_t>(payload.size()));
        layout.emplace_back(tag, cursor);
        cursor += payload.size();
    }
    // Payloads (in the same deterministic map order).
    for (const auto& [tag, payload] : sections_) w.bytes(payload);

    // Trailer: CRC over everything so far.
    std::uint32_t crc = crc32(w.data());
    w.u<std::uint32_t>(crc);
    return std::move(w.data());
}

// ─── Parse ────────────────────────────────────────────────────────────────────
Result<Container> Container::parse(std::string_view blob) {
    if (blob.size() < 28 + 4) return fail<Container>(Errc::corrupt_index, "too short");

    // Verify CRC first (last 4 bytes).
    std::string_view body = blob.substr(0, blob.size() - 4);
    std::uint32_t stored_crc;
    std::memcpy(&stored_crc, blob.data() + blob.size() - 4, 4);
    if (crc32(body) != stored_crc) return fail<Container>(Errc::corrupt_index, "crc mismatch");

    Reader r(blob);
    std::string_view magic;
    if (!r.bytes(8, magic) || std::memcmp(magic.data(), kMagic, 8) != 0)
        return fail<Container>(Errc::corrupt_index, "bad magic");

    Container c;
    if (!r.u(c.major_) || !r.u(c.minor_)) return fail<Container>(Errc::corrupt_index, "version");
    if (c.major_ != kFormatMajor)
        return fail<Container>(Errc::corrupt_index,
            "unsupported format major " + std::to_string(c.major_) +
            " (reader supports " + std::to_string(kFormatMajor) + ")");
    std::uint32_t nsections;
    std::uint64_t reserved;
    if (!r.u(c.flags_) || !r.u(nsections) || !r.u(reserved))
        return fail<Container>(Errc::corrupt_index, "header");

    struct Entry { std::uint32_t tag; std::uint64_t off, len; };
    std::vector<Entry> entries(nsections);
    for (auto& e : entries)
        if (!r.u(e.tag) || !r.u(e.off) || !r.u(e.len))
            return fail<Container>(Errc::corrupt_index, "section table");

    for (const auto& e : entries) {
        if (e.off + e.len > blob.size() - 4)
            return fail<Container>(Errc::corrupt_index, "section out of bounds");
        c.sections_[e.tag] = std::string(blob.substr(e.off, e.len));
    }
    return c;
}

// ─── File I/O ─────────────────────────────────────────────────────────────────
//
// Writing an index is a REPLACEMENT, and the naive form of it destroys data:
// opening the destination truncates it, so a crash (or a full disk, or a kill
// -9) part-way through leaves a half-written file where a working index used to
// be. The old index is gone and the new one is unreadable.
//
// So: write to a temporary file in the SAME directory, flush it all the way to
// the storage device, then rename() it over the destination. POSIX guarantees
// rename() within a filesystem is atomic, so at every instant an observer sees
// either the complete old index or the complete new one, never a torn mixture.
// Same directory matters twice over — rename() across filesystems fails, and a
// temp file elsewhere would be a copy rather than an atomic swap.
//
// The fsync ORDER is the part that is easy to get wrong and impossible to
// notice in testing: fsync the FILE before the rename (so its bytes are durable
// before anything points at them), and fsync the DIRECTORY after (so the
// rename itself is durable). Without the second one, a power loss can leave the
// directory entry pointing at the old inode even though the data was written.
Result<void> Container::write_file(const std::string& path) const {
    // Temp name in the destination's directory, salted so two concurrent saves
    // to the same path cannot collide on it.
    const auto slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".")
                                                         : path.substr(0, slash);
    static std::atomic<std::uint64_t> counter{0};
    std::random_device rd;
    const std::string tmp = path + ".tmp." + std::to_string(::getpid()) + "."
                          + std::to_string(counter.fetch_add(1, std::memory_order_relaxed))
                          + "." + std::to_string(rd());

    const std::string blob = serialize();
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return fail<void>(Errc::io_error, "open " + tmp);
        out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        out.flush();
        if (!out) { std::remove(tmp.c_str()); return fail<void>(Errc::io_error, "write " + tmp); }
    }

    // Flush the file's contents out of the OS page cache onto the device.
    // ofstream::flush only pushes to the kernel; it is not durability.
    if (int fd = ::open(tmp.c_str(), O_RDONLY); fd >= 0) {
        const int rc = ::fsync(fd);
        ::close(fd);
        if (rc != 0) { std::remove(tmp.c_str()); return fail<void>(Errc::io_error, "fsync " + tmp); }
    } else {
        std::remove(tmp.c_str());
        return fail<void>(Errc::io_error, "reopen " + tmp);
    }

    // The atomic swap. After this returns, `path` is the new index.
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return fail<void>(Errc::io_error, "rename " + tmp + " -> " + path);
    }

    // Make the rename itself durable. Best-effort: some filesystems refuse to
    // open a directory for fsync, and failing the whole save over that would be
    // worse than the (already atomic) swap we just completed.
    if (int dfd = ::open(dir.c_str(), O_RDONLY); dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }

    // Sweep temps orphaned by a PREVIOUS writer that died between creating its
    // temp file and renaming it. That process could not clean up after itself
    // by definition, so without this they accumulate in the data directory
    // forever — each one a full-size copy of the index.
    //
    // Only files belonging to this exact destination are considered, and only
    // those whose owning process is gone, so a concurrent save in progress is
    // never disturbed.
    sweep_orphan_temps(path);
    return {};
}

// Remove `<path>.tmp.<pid>.*` files whose <pid> is no longer running.
void Container::sweep_orphan_temps(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path target(path);
    const fs::path dir = target.has_parent_path() ? target.parent_path() : fs::path(".");
    const std::string prefix = target.filename().string() + ".tmp.";

    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;

        // Extract the pid field: <prefix><pid>.<counter>.<salt>
        const std::string rest = name.substr(prefix.size());
        const std::size_t dot  = rest.find('.');
        if (dot == std::string::npos) continue;
        long pid = 0;
        try { pid = std::stol(rest.substr(0, dot)); } catch (...) { continue; }
        if (pid <= 0) continue;

        // kill(pid, 0) probes for existence without signalling. ESRCH means the
        // process is gone and the temp is definitively garbage; EPERM means it
        // exists but belongs to someone else, so leave it alone.
        if (::kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM) continue;
        std::error_code rm;
        fs::remove(it->path(), rm);
    }
}

Result<Container> Container::read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail<Container>(Errc::io_error, "open " + path);
    std::stringstream ss; ss << in.rdbuf();
    return parse(ss.str());
}

} // namespace rag::store
