// src/bridge/process.cpp — subprocess Channel implementation (POSIX + Windows).
//
// Two spawn/reap backends, one protocol. Everything above the pipe — framing,
// buffering, the JSON envelope — is platform-independent and written once:
//
//   POSIX    pipe() + fork() + execvp(), reaped with waitpid().
//   Windows  CreatePipe() + CreateProcess(), reaped with WaitForSingleObject().
//            The parent ends of the pipes are wrapped into CRT file descriptors
//            with _open_osfhandle(), so read_line()/write_line() below operate
//            on plain `int` fds on both platforms and need no #ifdef at all.
//            mingw's ucrt64 CRT exposes ::read/::write/::close as global
//            aliases for _read/_write/_close, so even those call sites are
//            shared verbatim.
//
// Windows used to be handled by simply not compiling this file, on the stated
// grounds that "nothing in the core library references it". That was wrong:
// src/bridge/register.cpp implements open_channel()'s `"transport":"process"`
// branch by calling ProcessChannel::spawn(), and register.cpp is unconditionally
// part of the library — so dropping this TU produced a library that compiled
// cleanly and then failed at LINK time in every downstream executable with
// `undefined reference to rag::bridge::ProcessChannel::spawn(...)`. A real
// backend is cheaper than a load-bearing hole, and it makes the polyglot bridge
// (the whole point of which is "any language, any platform") actually portable.

#include "rag/bridge/process.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)

#include <fcntl.h>
#include <io.h>          // _open_osfhandle, _read/_write aliases
#include <sys/types.h>   // ssize_t

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#else

#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#endif

namespace rag::bridge {
namespace {

#if defined(_WIN32)
using io_result_t = int;

io_result_t fd_write(int fd, const char* p, std::size_t n) {
    return ::_write(fd, p, static_cast<unsigned>(
        std::min(n, static_cast<std::size_t>(INT_MAX))));
}

io_result_t fd_read(int fd, char* p, std::size_t n) {
    return ::_read(fd, p, static_cast<unsigned>(
        std::min(n, static_cast<std::size_t>(INT_MAX))));
}
#else
using io_result_t = ssize_t;

io_result_t fd_write(int fd, const char* p, std::size_t n) {
    return ::write(fd, p, n);
}

io_result_t fd_read(int fd, char* p, std::size_t n) {
    return ::read(fd, p, n);
}
#endif

// Write all bytes, retrying on EINTR/partial writes. Returns false on error.
bool write_all(int fd, const char* p, std::size_t n) {
    while (n > 0) {
        io_result_t w = fd_write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

#if defined(_WIN32)

// Windows has no argv — a process receives one flat command line and each CRT
// re-splits it. Quote per the rules the MSVC/mingw CRT parser implements, so an
// argument containing spaces, quotes or trailing backslashes arrives at the
// child exactly as it left here. Getting this wrong is how "it works until
// someone has a space in their path" bugs are born.
std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) return a;
    std::string out = "\"";
    for (auto it = a.begin();; ++it) {
        std::size_t backslashes = 0;
        while (it != a.end() && *it == '\\') { ++it; ++backslashes; }
        if (it == a.end()) {
            // Trailing backslashes must be doubled so they don't escape our
            // closing quote.
            out.append(backslashes * 2, '\\');
            break;
        }
        if (*it == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
        } else {
            out.append(backslashes, '\\');
            out.push_back(*it);
        }
    }
    out.push_back('"');
    return out;
}

// CreateProcess takes a whole environment block or nothing; there is no
// "inherit plus these extras". So materialise the parent block, apply
// ProcessConfig::env with putenv() semantics (a later KEY= replaces an earlier
// one), and hand back the double-NUL-terminated result.
std::vector<char> build_env_block(const std::vector<std::string>& extra) {
    std::vector<std::string> entries;
    if (char* parent = ::GetEnvironmentStringsA()) {
        for (const char* p = parent; *p != '\0'; p += std::strlen(p) + 1)
            entries.emplace_back(p);
        ::FreeEnvironmentStringsA(parent);
    }
    for (const auto& kv : extra) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        // Windows env names are case-insensitive; match them that way.
        const std::string key = kv.substr(0, eq + 1);
        std::erase_if(entries, [&](const std::string& e) {
            if (e.size() < key.size()) return false;
            return ::_strnicmp(e.c_str(), key.c_str(), key.size()) == 0;
        });
        entries.push_back(kv);
    }
    std::vector<char> block;
    for (const auto& e : entries) block.insert(block.end(), e.begin(), e.end()), block.push_back('\0');
    block.push_back('\0'); // terminating empty string
    return block;
}

#endif // _WIN32

} // namespace

#if defined(_WIN32)

Result<std::shared_ptr<ProcessChannel>> ProcessChannel::spawn(ProcessConfig cfg) {
    if (cfg.argv.empty())
        return fail<std::shared_ptr<ProcessChannel>>(Errc::invalid_argument,
                                                     "ProcessChannel: empty argv");

    // Only the ends we hand to the child may be inheritable, or the child would
    // hold our copy of the other end open and we'd never see EOF on its exit.
    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE in_r = nullptr, in_w = nullptr;   // host writes in_w -> child stdin
    HANDLE out_r = nullptr, out_w = nullptr; // child stdout -> host reads out_r
    if (!::CreatePipe(&in_r, &in_w, &sa, 0))
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error, "CreatePipe() failed");
    if (!::CreatePipe(&out_r, &out_w, &sa, 0)) {
        ::CloseHandle(in_r); ::CloseHandle(in_w);
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error, "CreatePipe() failed");
    }
    // Our ends must NOT leak into the child.
    ::SetHandleInformation(in_w,  HANDLE_FLAG_INHERIT, 0);
    ::SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    std::string cmdline;
    for (std::size_t i = 0; i < cfg.argv.size(); ++i) {
        if (i) cmdline.push_back(' ');
        cmdline += quote_arg(cfg.argv[i]);
    }
    std::vector<char> mutable_cmdline(cmdline.begin(), cmdline.end());
    mutable_cmdline.push_back('\0'); // CreateProcessA may write into this buffer

    std::vector<char> env_block;
    if (!cfg.env.empty()) env_block = build_env_block(cfg.env);

    STARTUPINFOA si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = in_r;
    si.hStdOutput = out_w;
    // stderr is inherited so the peer can log to the host's stderr, matching POSIX.
    si.hStdError  = ::GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi{};
    const BOOL ok = ::CreateProcessA(
        nullptr,                        // look argv[0] up on PATH, like execvp
        mutable_cmdline.data(),
        nullptr, nullptr,
        TRUE,                           // inherit the std handles set above
        0,
        env_block.empty() ? nullptr : env_block.data(),
        cfg.cwd.empty() ? nullptr : cfg.cwd.c_str(),
        &si, &pi);

    // The child's ends belong to the child now, whether or not it started.
    ::CloseHandle(in_r);
    ::CloseHandle(out_w);

    if (!ok) {
        const DWORD e = ::GetLastError();
        ::CloseHandle(in_w);
        ::CloseHandle(out_r);
        return fail<std::shared_ptr<ProcessChannel>>(
            Errc::transport_error,
            "CreateProcess('" + cfg.argv[0] + "') failed, GetLastError=" + std::to_string(e));
    }
    ::CloseHandle(pi.hThread); // we never resume/inspect the initial thread

    // Adopt the parent ends as CRT fds. O_BINARY is essential: the protocol is
    // newline-delimited and a text-mode fd would silently turn every '\n' we
    // write into "\r\n" and strip CRs on read, corrupting frames.
    const int fd_w = ::_open_osfhandle(reinterpret_cast<intptr_t>(in_w),  _O_BINARY | _O_WRONLY);
    const int fd_r = ::_open_osfhandle(reinterpret_cast<intptr_t>(out_r), _O_BINARY | _O_RDONLY);
    if (fd_w < 0 || fd_r < 0) {
        if (fd_w >= 0) ::close(fd_w); else ::CloseHandle(in_w);
        if (fd_r >= 0) ::close(fd_r); else ::CloseHandle(out_r);
        ::TerminateProcess(pi.hProcess, 1);
        ::CloseHandle(pi.hProcess);
        return fail<std::shared_ptr<ProcessChannel>>(Errc::transport_error,
                                                     "_open_osfhandle() failed");
    }
    // Ownership note: the fd now owns the HANDLE — closing the fd closes it, and
    // double-closing the HANDLE would be a bug. Only the fds are closed below.

    auto ch = std::shared_ptr<ProcessChannel>(new ProcessChannel());
    ch->name_       = cfg.name.empty() ? cfg.argv[0] : cfg.name;
    ch->pid_        = static_cast<long>(pi.dwProcessId);
    ch->handle_     = pi.hProcess;
    ch->to_child_   = fd_w;
    ch->from_child_ = fd_r;
    ch->alive_      = true;
    return ch;
}

ProcessChannel::~ProcessChannel() {
    // Closing the child's stdin is the polite exit request: a peer blocked in
    // read() sees EOF and returns.
    if (to_child_ >= 0)   ::close(to_child_);
    if (from_child_ >= 0) ::close(from_child_);
    if (handle_ != nullptr) {
        HANDLE h = static_cast<HANDLE>(handle_);
        // Same budget as the POSIX path: ~100 ms of grace, then kill. Windows
        // has no SIGTERM, so TerminateProcess is the only escalation available.
        if (::WaitForSingleObject(h, 100) != WAIT_OBJECT_0) {
            ::TerminateProcess(h, 1);
            ::WaitForSingleObject(h, 1000);
        }
        ::CloseHandle(h);
        handle_ = nullptr;
        pid_    = -1;
    }
}

#else // ── POSIX ──────────────────────────────────────────────────────────────

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

#endif // platform spawn/reap backends

// ── Shared below this line: identical on every platform, because both backends
// ── hand us ordinary CRT file descriptors.

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
        io_result_t r = fd_read(from_child_, tmp, sizeof(tmp));
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
