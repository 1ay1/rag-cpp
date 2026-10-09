#pragma once
// rag/bridge/process.hpp — a Channel over a line stream, for a backend that
// lives in another process.
//
// The peer reads newline-delimited JSON on its stdin and answers on its
// stdout: one compact JSON object per line in, one per line out. This is the
// universal, dependency-free way to plug an engine/retriever/graph written in
// ANY language into rag-cpp — the peer only has to read a line, parse JSON,
// and print a JSON line.
//
// rag-cpp starts no processes. The host spawns the program described by a
// ProcessSpec (HostIo::spawn, plugin/builder.hpp) and hands back a LineChannel
// over the child's pipes; this file is the protocol on top of them.
//
// Robustness: a crashed or mis-speaking peer surfaces as
// Errc::transport_error, never a throw or a hang on EOF (the host's read
// returns 0 at EOF).

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rag/bridge/channel.hpp"
#include "rag/core/types.hpp"

namespace rag::bridge {

// What to run. The host decides how (and under which sandbox).
struct ProcessSpec {
    // argv[0] is the program; the rest are arguments.
    std::vector<std::string> argv;
    // Extra environment entries "KEY=VALUE" layered on the inherited env.
    std::vector<std::string> env;
    // Working directory for the child (empty = inherit).
    std::string              cwd;
    // A label for identity()/logs; defaults to argv[0].
    std::string              name;
};

// Byte IO to the peer, supplied by the host. `read` blocks for at least one
// byte and returns 0 at EOF (or error); `write` returns false when the peer
// is gone. `keep` owns whatever backs them (the child process) and is
// released when the channel is.
struct LineIo {
    std::function<std::size_t(char* buf, std::size_t cap)> read;
    std::function<bool(std::string_view bytes)>            write;
    std::shared_ptr<void>                                  keep;
};

// The newline-JSON protocol over a LineIo. Not thread-safe: one call at a
// time (the owner serializes).
class LineChannel final : public Channel {
public:
    LineChannel(LineIo io, std::string name) : io_(std::move(io)), name_(std::move(name)) {}

    [[nodiscard]] Result<Json> call(std::string_view method, const Json& params) override;
    [[nodiscard]] std::string identity() const override { return "process:" + name_; }

    // True until the peer has closed its output or a write failed.
    [[nodiscard]] bool alive() const noexcept { return alive_; }

private:
    Result<std::string> read_line();

    LineIo      io_;
    std::string name_;
    std::string inbuf_;   // carry-over bytes past a newline
    bool        alive_ = true;
};

} // namespace rag::bridge
