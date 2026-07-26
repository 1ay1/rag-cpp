// rag/store/wal.cpp — append-only log of logical mutations.
//
// RECORD FRAMING
//
//   u32 magic  ("WALR")   — resynchronization anchor and a cheap sanity check
//   u32 length            — payload bytes that follow the header
//   u32 crc32             — over the payload only
//   u8  payload[length]
//
// The length precedes the payload so a reader knows how much to expect before
// reading it, and the CRC lets it tell a TORN tail (crash mid-append) from a
// CORRUPT body (a bit flip in a record that was already acknowledged). Those
// two need opposite handling and conflating them is the classic WAL bug: treat
// a torn tail as corruption and every crash becomes an unrecoverable database;
// treat corruption as a torn tail and you silently discard acknowledged writes.

#include "rag/store/wal.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include "rag/store/format.hpp"

namespace rag::store {

namespace {

constexpr std::uint32_t kRecordMagic = 0x524C4157;   // "WALR" little-endian
constexpr std::size_t   kHeaderBytes = 12;           // magic + length + crc

std::string encode(const WalRecord& rec) {
    Writer w;
    w.u<std::uint8_t>(static_cast<std::uint8_t>(rec.op));
    w.str(rec.uri);
    w.str(rec.title);
    w.str(rec.text);
    w.u<std::uint32_t>(static_cast<std::uint32_t>(rec.meta.size()));
    for (const auto& [k, v] : rec.meta) { w.str(k); w.str(v); }
    w.u<std::uint32_t>(rec.doc_id);
    return std::move(w.data());
}

bool decode(std::string_view payload, WalRecord& out) {
    Reader r(payload);
    std::uint8_t op = 0;
    if (!r.u(op)) return false;
    if (op != static_cast<std::uint8_t>(WalOp::add_document) &&
        op != static_cast<std::uint8_t>(WalOp::remove_document)) return false;
    out.op = static_cast<WalOp>(op);
    if (!r.str(out.uri) || !r.str(out.title) || !r.str(out.text)) return false;
    std::uint32_t n = 0;
    if (!r.u(n)) return false;
    out.meta.clear();
    for (std::uint32_t i = 0; i < n; ++i) {
        std::string k, v;
        if (!r.str(k) || !r.str(v)) return false;
        out.meta.emplace(std::move(k), std::move(v));
    }
    return r.u(out.doc_id);
}

} // namespace

Wal::~Wal() { close(); }

Wal::Wal(Wal&& o) noexcept
    : fd_(std::exchange(o.fd_, -1)), path_(std::move(o.path_)),
      mode_(o.mode_), bytes_(std::exchange(o.bytes_, 0)) {}

Wal& Wal::operator=(Wal&& o) noexcept {
    if (this != &o) {
        close();
        fd_    = std::exchange(o.fd_, -1);
        path_  = std::move(o.path_);
        mode_  = o.mode_;
        bytes_ = std::exchange(o.bytes_, 0);
    }
    return *this;
}

void Wal::close() noexcept {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    bytes_ = 0;
}

Result<void> Wal::open(std::string path, SyncMode mode) {
    close();
    // O_APPEND makes every write atomic with respect to the file offset, so a
    // concurrent appender can never interleave into the middle of our record.
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return fail<void>(Errc::io_error, "wal open " + path + ": " + std::strerror(errno));
    fd_    = fd;
    path_  = std::move(path);
    mode_  = mode;
    const off_t end = ::lseek(fd_, 0, SEEK_END);
    bytes_ = end > 0 ? static_cast<std::uint64_t>(end) : 0;
    return {};
}

Result<void> Wal::sync_now() noexcept {
    switch (mode_) {
        case SyncMode::none:
            return {};
        case SyncMode::flush:
            if (::fsync(fd_) != 0) return fail<void>(Errc::io_error, "wal fsync");
            return {};
        case SyncMode::full:
#if defined(F_FULLFSYNC)
            // macOS: fsync() returns once the data reaches the DEVICE, which may
            // still hold it in a volatile write cache. F_FULLFSYNC is the only
            // call that makes it survive power loss. Fall back to fsync if the
            // filesystem does not implement it (it returns ENOTSUP), rather
            // than failing a write that is very likely fine.
            if (::fcntl(fd_, F_FULLFSYNC) == 0) return {};
#endif
            if (::fsync(fd_) != 0) return fail<void>(Errc::io_error, "wal fsync");
            return {};
    }
    return {};
}

Result<void> Wal::append(const WalRecord& rec) {
    if (fd_ < 0) return fail<void>(Errc::io_error, "wal not open");

    const std::string payload = encode(rec);
    Writer w;
    w.u<std::uint32_t>(kRecordMagic);
    w.u<std::uint32_t>(static_cast<std::uint32_t>(payload.size()));
    w.u<std::uint32_t>(crc32(payload));
    w.bytes(payload);
    const std::string& frame = w.data();

    // ONE write() for header+payload. Splitting them would let a crash land
    // between the two and leave a header promising bytes that do not exist —
    // still detectable, but this keeps the torn window as small as the OS
    // allows. Loop because write() may legally be partial.
    std::size_t off = 0;
    while (off < frame.size()) {
        const ssize_t n = ::write(fd_, frame.data() + off, frame.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return fail<void>(Errc::io_error, std::string("wal write: ") + std::strerror(errno));
        }
        off += static_cast<std::size_t>(n);
    }
    if (auto s = sync_now(); !s) return s;
    bytes_ += frame.size();
    return {};
}

Result<std::vector<WalRecord>> Wal::replay(const std::string& path) {
    std::vector<WalRecord> out;
    std::ifstream in(path, std::ios::binary);
    if (!in) return out;                      // absent log = nothing to replay
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string blob = ss.str();

    std::size_t pos = 0;
    while (pos + kHeaderBytes <= blob.size()) {
        Reader r(std::string_view(blob).substr(pos, kHeaderBytes));
        std::uint32_t magic = 0, len = 0, crc = 0;
        if (!r.u(magic) || !r.u(len) || !r.u(crc)) break;
        if (magic != kRecordMagic)
            return fail<std::vector<WalRecord>>(Errc::corrupt_index, "wal: bad record magic");

        const std::size_t body = pos + kHeaderBytes;
        if (body + len > blob.size()) break;   // TORN TAIL: never acknowledged

        const std::string_view payload(blob.data() + body, len);
        if (crc32(payload) != crc) {
            // A CRC failure on the LAST record is a torn tail whose header
            // happened to survive; anywhere else it is real corruption of a
            // record we already told a client had succeeded.
            if (body + len == blob.size()) break;
            return fail<std::vector<WalRecord>>(Errc::corrupt_index, "wal: record crc mismatch");
        }

        WalRecord rec;
        if (!decode(payload, rec)) {
            if (body + len == blob.size()) break;
            return fail<std::vector<WalRecord>>(Errc::corrupt_index, "wal: undecodable record");
        }
        out.push_back(std::move(rec));
        pos = body + len;
    }
    return out;
}

Result<void> Wal::truncate() {
    if (fd_ < 0) return fail<void>(Errc::io_error, "wal not open");
    if (::ftruncate(fd_, 0) != 0)
        return fail<void>(Errc::io_error, std::string("wal truncate: ") + std::strerror(errno));
    ::lseek(fd_, 0, SEEK_SET);
    // Make the truncation itself durable. Without this a crash can resurrect a
    // log whose records are already in the snapshot — replaying them would
    // duplicate every document in the checkpoint window.
    if (auto s = sync_now(); !s) return s;
    bytes_ = 0;
    return {};
}

} // namespace rag::store
