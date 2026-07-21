// src/bridge/process.cpp — POSIX subprocess Channel implementation.

#include "rag/bridge/process.hpp"

#include <csignal>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace rag::bridge {
namespace {

// Write all bytes, retrying on EINTR/partial writes. Returns false on error.
bool write_all(int fd, const char* p, std::size_t n) {
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

} // namespace

Result<std::shared_ptr<ProcessChannel>> ProcessChannel::spawn(ProcessConfig cfg) {
    if (cfg.argv.empty())
        return fail<std::shared_ptr<ProcessChannel>>(Errc::invalid_argument,
                                                     "ProcessChannel: empty argv");

    int in_pipe[2];   // host writes -> child stdin
    int out_pipe[2];  // child stdout -> host reads
    if (::pipe(in_pipe) != 0)
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error, "pipe() failed");
    if (::pipe(out_pipe) != 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error, "pipe() failed");
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error, "fork() failed");
    }

    if (pid == 0) {
        // ── child ──
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        // stderr is inherited so the peer can log to the host's stderr.
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);

        if (!cfg.cwd.empty()) {
            if (::chdir(cfg.cwd.c_str()) != 0) ::_exit(127);
        }
        for (const auto& kv : cfg.env) ::putenv(const_cast<char*>(kv.c_str()));

        std::vector<char*> args;
        args.reserve(cfg.argv.size() + 1);
        for (auto& a : cfg.argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        ::execvp(args[0], args.data());
        ::_exit(127); // exec failed
    }

    // ── parent ──
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    auto ch = std::shared_ptr<ProcessChannel>(new ProcessChannel());
    ch->name_       = cfg.name.empty() ? cfg.argv[0] : cfg.name;
    ch->pid_        = pid;
    ch->to_child_   = in_pipe[1];
    ch->from_child_ = out_pipe[0];
    ch->alive_      = true;
    return ch;
}

ProcessChannel::~ProcessChannel() {
    if (to_child_ >= 0)   ::close(to_child_);
    if (from_child_ >= 0) ::close(from_child_);
    if (pid_ > 0) {
        // Give the child a chance to exit on stdin EOF, then reap. If it lingers,
        // signal it. We don't block forever.
        int status = 0;
        for (int i = 0; i < 50; ++i) {
            pid_t r = ::waitpid(static_cast<pid_t>(pid_), &status, WNOHANG);
            if (r == static_cast<pid_t>(pid_) || r < 0) { pid_ = -1; break; }
            ::usleep(2000);
        }
        if (pid_ > 0) {
            ::kill(static_cast<pid_t>(pid_), SIGTERM);
            ::waitpid(static_cast<pid_t>(pid_), &status, 0);
        }
    }
}

Result<std::string> ProcessChannel::write_line(std::string_view line) {
    if (!alive_ || to_child_ < 0)
        return fail<std::string>(Errc::unavailable, "process channel not alive");
    std::string buf(line);
    buf.push_back('\n');
    if (!write_all(to_child_, buf.data(), buf.size())) {
        alive_ = false;
        return fail<std::string>(Errc::transport_error, "write to child failed (broken pipe?)");
    }
    return std::string{};
}

Result<std::string> ProcessChannel::read_line() {
    // Return a full line from inbuf_ if we already have one.
    for (;;) {
        if (auto nl = inbuf_.find('\n'); nl != std::string::npos) {
            std::string line = inbuf_.substr(0, nl);
            inbuf_.erase(0, nl + 1);
            return line;
        }
        char tmp[4096];
        ssize_t r = ::read(from_child_, tmp, sizeof(tmp));
        if (r < 0) {
            if (errno == EINTR) continue;
            alive_ = false;
            return fail<std::string>(Errc::transport_error, "read from child failed");
        }
        if (r == 0) { // EOF: child closed stdout / exited
            alive_ = false;
            if (!inbuf_.empty()) { std::string line = std::move(inbuf_); inbuf_.clear(); return line; }
            return fail<std::string>(Errc::transport_error, "child closed stdout (EOF)");
        }
        inbuf_.append(tmp, static_cast<std::size_t>(r));
    }
}

Result<Json> ProcessChannel::call(std::string_view method, const Json& params) {
    Json req = Json::object();
    req["method"] = std::string(method);
    req["params"] = params;

    std::string wire = req.dump(); // compact, single line
    if (auto w = write_line(wire); !w) return std::unexpected(w.error());

    auto line = read_line();
    if (!line) return std::unexpected(line.error());

    Json reply;
    try {
        reply = Json::parse(*line);
    } catch (const std::exception& e) {
        return fail<Json>(Errc::transport_error,
                          std::string("child sent invalid JSON: ") + e.what());
    }
    return unwrap_envelope(reply);
}

} // namespace rag::bridge
