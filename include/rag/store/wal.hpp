#pragma once
// rag/store/wal.hpp — write-ahead log: make a mutation durable in O(record),
// not O(corpus).
//
// THE PROBLEM IT SOLVES
//
// A server that accepts writes has to choose between two bad options if a full
// snapshot is its only durability mechanism:
//
//   * save() after every mutation. Correct, but it rewrites the WHOLE index for
//     one new document. Measured: an acknowledged index/add costs 25 ms on a
//     20k-document corpus and 69.7 ms on 50k, of which the actual insert is
//     0.007 ms. The cost is entirely the rewrite, and it grows without bound.
//   * save() occasionally. Fast, but every write since the last snapshot is
//     lost on a crash — and the client was told those writes succeeded.
//
// A log removes the choice. The mutation is appended as a small record and
// synced; the snapshot happens later, on its own schedule. Measured floor for
// one durable append on this machine:
//
//   record     write only     fsync    F_FULLFSYNC
//   256 B         0.0018     0.0211         3.7686
//   4 KB          0.0077     0.0418         4.2954
//   64 KB         0.0238     0.0718         4.7803
//
// So an fsync'd append is ~600x cheaper than the 25 ms snapshot it replaces,
// and — the part that actually matters — it is CONSTANT in corpus size.
//
// DURABILITY LEVELS, and why macOS forces the question
//
// `fsync()` on macOS does NOT flush the drive's own write cache; it only pushes
// data to the device and returns. Apple documents F_FULLFSYNC as the operation
// that actually makes data survive power loss. That is a 100x difference here
// (0.04 ms vs 4.3 ms), far too large to hide behind a single "durable" flag, so
// the level is explicit and the docs say what each one actually survives.
//
// WHAT A RECORD IS
//
// Logical, not physical: "document added with this uri/text/metadata", not
// "these bytes changed at this offset". A logical log replays through the same
// Corpus API that produced it, so it cannot desynchronize from the in-memory
// structures the way a byte-diff of a half-built HNSW graph would. It also
// stays small — a document, not a re-linked graph.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/core/document.hpp"

namespace rag::store {

// How hard to try before an append is reported durable.
enum class SyncMode {
    // No flush. Survives a PROCESS crash (the kernel still owns the page cache)
    // but not power loss. ~0.002 ms.
    none,
    // fsync(2). Survives process crash and OS crash. On Linux this is normally
    // power-safe too; on macOS it is NOT, because the drive may still hold the
    // data in its own cache. ~0.04 ms.
    flush,
    // F_FULLFSYNC on macOS, fsync elsewhere. Survives power loss. ~4.3 ms \u2014
    // roughly 100x `flush`, because it waits for the physical device.
    full,
};

// One logical mutation.
enum class WalOp : std::uint8_t {
    add_document    = 1,   // uri, title, text, meta
    remove_document = 2,   // doc id
};

struct WalRecord {
    WalOp       op = WalOp::add_document;
    std::string uri;
    std::string title;
    std::string text;
    Metadata    meta;
    std::uint32_t doc_id = 0;      // remove_document only
};

// An append-only log of logical mutations.
//
// Not thread-safe by itself: Corpus owns one and appends under its write lock,
// which is also what makes the log order match the order the mutations were
// applied in memory.
class Wal {
public:
    Wal() = default;
    ~Wal();
    Wal(const Wal&) = delete;
    Wal& operator=(const Wal&) = delete;
    Wal(Wal&& o) noexcept;
    Wal& operator=(Wal&& o) noexcept;

    // Open (creating if absent) the log at `path`, positioned to append.
    [[nodiscard]] Result<void> open(std::string path, SyncMode mode = SyncMode::flush);
    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    // Append one record and make it durable to the configured level. Returns
    // only after the data is as safe as `mode` promises \u2014 callers rely on that
    // to acknowledge a client write.
    [[nodiscard]] Result<void> append(const WalRecord& rec);

    // Read every intact record. A torn trailing record \u2014 the normal outcome of
    // a crash mid-append \u2014 is DROPPED, not an error: it was never acknowledged,
    // so no one was told it succeeded. Anything torn EARLIER in the file is a
    // real corruption and does fail, because it means a record that WAS
    // acknowledged is unreadable.
    [[nodiscard]] static Result<std::vector<WalRecord>> replay(const std::string& path);

    // Discard the log. Called after a snapshot has captured everything in it.
    // The snapshot must be durable BEFORE this runs, or a crash in between
    // loses the window.
    [[nodiscard]] Result<void> truncate();

    // Bytes currently in the log \u2014 the signal for "time to checkpoint".
    [[nodiscard]] std::uint64_t size_bytes() const noexcept { return bytes_; }

    void close() noexcept;

private:
    int           fd_    = -1;
    std::string   path_;
    SyncMode      mode_  = SyncMode::flush;
    std::uint64_t bytes_ = 0;

    [[nodiscard]] Result<void> sync_now() noexcept;
};

} // namespace rag::store
