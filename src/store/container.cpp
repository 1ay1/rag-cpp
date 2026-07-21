// rag/store/container.cpp — .ragdb container serialization + CRC32.

#include "rag/store/container.hpp"

#include <array>
#include <fstream>
#include <sstream>

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
Result<void> Container::write_file(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return fail<void>(Errc::io_error, "open " + path);
    std::string blob = serialize();
    out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    if (!out) return fail<void>(Errc::io_error, "write " + path);
    return {};
}

Result<Container> Container::read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail<Container>(Errc::io_error, "open " + path);
    std::stringstream ss; ss << in.rdbuf();
    return parse(ss.str());
}

} // namespace rag::store
