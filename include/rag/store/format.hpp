#pragma once
// rag/store/format.hpp — the STABLE, versioned, on-disk container format.
//
// This is a PUBLIC CONTRACT, not an internal cache. The layout is documented
// in FORMAT.md and versioned by `kFormatVersion`; readers reject unknown major
// versions and CRC-check the payload. A `.ragdb` file is self-describing and
// portable across builds and platforms (all integers little-endian).
//
//   ┌──────────────────────────────────────────────────────────────────┐
//   │ Header (32 bytes)                                                  │
//   │   magic[8]   = "RAGDB\0\0\0"                                       │
//   │   u16 format_major, u16 format_minor                              │
//   │   u32 flags            (bit0: has_hnsw, bit1: has_embeddings)      │
//   │   u32 section_count                                               │
//   │   u64 reserved                                                    │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Section table: section_count × { u32 tag, u64 offset, u64 len }   │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Section payloads (referenced by the table)                        │
//   ├──────────────────────────────────────────────────────────────────┤
//   │ Trailer: u32 crc32 (over everything above)                        │
//   └──────────────────────────────────────────────────────────────────┘
//
// Sections are addressed by a 4-byte tag so new section types can be added
// without breaking old readers (unknown tags are skipped). This is the same
// principle as PNG chunks / RIFF.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::store {

constexpr char        kMagic[8]       = {'R','A','G','D','B',0,0,0};
constexpr std::uint16_t kFormatMajor  = 1;   // incompatible change ⇒ bump; readers reject mismatched major
constexpr std::uint16_t kFormatMinor  = 0;   // additive change ⇒ bump; older readers still work

enum Flags : std::uint32_t {
    kHasHnsw       = 1u << 0,
    kHasEmbeddings = 1u << 1,
};

// Four-char section tags (little-endian u32 of the ASCII).
enum class Tag : std::uint32_t {
    meta   = 0x4154454D,  // "META" — corpus config JSON
    docs   = 0x53434F44,  // "DOCS" — documents
    chunks = 0x4B4E4843,  // "CHNK" — chunk records (no embeddings)
    embed  = 0x42444D45,  // "EMBD" — chunk embeddings (parallel to chunks)
    bm25   = 0x35324D42,  // "BM25" — serialized inverted index
    hnsw   = 0x57534E48,  // "HNSW" — serialized ANN graph
};

// CRC32 (IEEE 802.3) over a byte range.
[[nodiscard]] std::uint32_t crc32(std::string_view data) noexcept;

// A minimal, safe little-endian byte writer/reader used by all serializers.
class Writer {
public:
    template <class T> void u(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const char* p = reinterpret_cast<const char*>(&v);
        buf_.append(p, p + sizeof(T));
    }
    void bytes(std::string_view s) { buf_.append(s.data(), s.size()); }
    void str(std::string_view s) { u<std::uint32_t>(static_cast<std::uint32_t>(s.size())); bytes(s); }
    [[nodiscard]] std::string& data() noexcept { return buf_; }
    [[nodiscard]] std::size_t size() const noexcept { return buf_.size(); }
private:
    std::string buf_;
};

class Reader {
public:
    explicit Reader(std::string_view in) : in_(in) {}
    template <class T> [[nodiscard]] bool u(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (in_.size() < sizeof(T)) { ok_ = false; return false; }
        std::memcpy(&v, in_.data(), sizeof(T)); in_.remove_prefix(sizeof(T)); return true;
    }
    [[nodiscard]] bool bytes(std::size_t n, std::string_view& out) {
        if (in_.size() < n) { ok_ = false; return false; }
        out = in_.substr(0, n); in_.remove_prefix(n); return true;
    }
    [[nodiscard]] bool str(std::string& out) {
        std::uint32_t n; if (!u(n)) return false;
        std::string_view sv; if (!bytes(n, sv)) return false;
        out.assign(sv); return true;
    }
    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return in_.size(); }
    [[nodiscard]] std::string_view rest() const noexcept { return in_; }
    void seek_to(std::string_view whole, std::uint64_t off) { in_ = whole.substr(off); }
private:
    std::string_view in_;
    bool ok_ = true;
};

} // namespace rag::store
