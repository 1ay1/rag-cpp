#pragma once
// rag/bridge/process.hpp — a Channel backed by a long-lived subprocess.
//
// Spawns an external program (e.g. `python3 ragcpp_server.py`) and speaks
// newline-delimited JSON over its stdin/stdout: one compact JSON object per line
// in, one per line out. This is the universal, dependency-free way to plug an
// engine/retriever/graph written in ANY language into rag-cpp — the peer only
// has to read a line, parse JSON, and print a JSON line.
//
// Lifetime: the child is spawned on construction and kept alive for reuse across
// many calls (no per-call fork cost). Closing stdin / destroying the channel
// lets the child exit; the destructor reaps it.
//
// Robustness: a crashed or mis-speaking child surfaces as Errc::transport_error,
// never a throw or a hang on EOF. stderr of the child is inherited (goes to the
// host's stderr) so the peer can log freely.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rag/bridge/channel.hpp"
#include "rag/core/types.hpp"

namespace rag::bridge {

struct ProcessConfig {
    // argv[0] is the program; the rest are arguments. Resolved via PATH by execvp.
    std::vector<std::string> argv;
    // Extra environment entries "KEY=VALUE" appended to the inherited env.
    std::vector<std::string> env;
    // Working directory for the child (empty = inherit).
    std::string              cwd;
    // A label for identity()/logs; defaults to argv[0].
    std::string              name;
};

// Spawns and owns a child process. Move-only. Not thread-safe: serialize calls
// (a Remote* wrapper holding a shared_ptr<ProcessChannel> should guard if shared
// across threads).
class ProcessChannel final : public Channel {
public:
    [[nodiscard]] static Result<std::shared_ptr<ProcessChannel>> spawn(ProcessConfig cfg);

    ~ProcessChannel() override;
    ProcessChannel(const ProcessChannel&) = delete;
    ProcessChannel& operator=(const ProcessChannel&) = delete;

    [[nodiscard]] Result<Json> call(std::string_view method, const Json& params) override;
    [[nodiscard]] std::string identity() const override { return "process:" + name_; }

    // True until the child has exited or a fatal I/O error occurred.
    [[nodiscard]] bool alive() const noexcept { return alive_; }

private:
    ProcessChannel() = default;

    Result<std::string> write_line(std::string_view line);
    Result<std::string> read_line();

    std::string name_;
    long        pid_    = -1;     // pid_t stored wide to keep the header POSIX-agnostic
    int         to_child_ = -1;   // write end (child's stdin)
    int         from_child_ = -1; // read end  (child's stdout)
    std::string inbuf_;           // carry-over bytes past a newline
    bool        alive_ = false;
};

} // namespace rag::bridge
