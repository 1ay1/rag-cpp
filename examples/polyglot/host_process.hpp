#pragma once
// examples/polyglot/host_process.hpp — a minimal HostIo::spawn for the examples.
//
// rag-cpp opens no sockets and starts no processes: a {"transport":"process"}
// bridge asks the HOST to run the child (HostIo::spawn) and speaks the
// newline-JSON protocol over the pipes it gets back (rag::bridge::LineChannel).
// A real application plugs in its own process layer here (sandboxing, job
// objects, an event loop...). This is the smallest portable one:
//
//   POSIX    pipe() + fork() + execvp(), reaped with waitpid().
//   Windows  CreatePipe() + CreateProcessA(), reaped with WaitForSingleObject().
//
// The child's stderr is inherited so the peer can log to the host's terminal.
// Teardown closes the child's stdin (a well-behaved peer exits on EOF), waits
// briefly, then kills it.

#include <rag/rag.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <cstring>
#else
#  include <cerrno>
#  include <csignal>
#  include <cstdlib>
#  include <fcntl.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace example {

using ChannelResult = rag::Result<std::shared_ptr<rag::bridge::Channel>>;

#if defined(_WIN32)

namespace detail {

// CommandLineToArgvW-compatible quoting of one argument.
inline std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) return a;
    std::string out = "\"";
    for (auto it = a.begin();; ++it) {
        std::size_t slashes = 0;
        while (it != a.end() && *it == '\\') { ++it; ++slashes; }
        if (it == a.end()) { out.append(slashes * 2, '\\'); break; }
        if (*it == '"') { out.append(slashes * 2 + 1, '\\'); out.push_back('"'); }
        else { out.append(slashes, '\\'); out.push_back(*it); }
    }
    out.push_back('"');
    return out;
}

// CreateProcess takes a whole environment block or nothing, so materialise the
// parent's block with the extra KEY=VALUE entries layered on top.
inline std::vector<char> env_block(const std::vector<std::string>& extra) {
    std::vector<std::string> entries;
    if (char* parent = ::GetEnvironmentStringsA()) {
        for (const char* p = parent; *p != '\0'; p += std::strlen(p) + 1) entries.emplace_back(p);
        ::FreeEnvironmentStringsA(parent);
    }
    for (const auto& kv : extra) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = kv.substr(0, eq + 1);
        std::erase_if(entries, [&](const std::string& e) {
            return e.size() >= key.size() && ::_strnicmp(e.c_str(), key.c_str(), key.size()) == 0;
        });
        entries.push_back(kv);
    }
    std::vector<char> block;
    for (const auto& e : entries) { block.insert(block.end(), e.begin(), e.end()); block.push_back('\0'); }
    block.push_back('\0');
    return block;
}

struct Child {
    HANDLE process = nullptr;
    HANDLE to_child = nullptr;    // child's stdin
    HANDLE from_child = nullptr;  // child's stdout
    ~Child() {
        if (to_child) ::CloseHandle(to_child);
        if (from_child) ::CloseHandle(from_child);
        if (process) {
            if (::WaitForSingleObject(process, 2000) != WAIT_OBJECT_0) {
                ::TerminateProcess(process, 1);
                ::WaitForSingleObject(process, INFINITE);
            }
            ::CloseHandle(process);
        }
    }
};

} // namespace detail

inline ChannelResult spawn_process(const rag::bridge::ProcessSpec& spec) {
    if (spec.argv.empty())
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::invalid_argument, "empty argv");

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
    if (!::CreatePipe(&in_r, &in_w, &sa, 0))
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::transport_error, "CreatePipe failed");
    if (!::CreatePipe(&out_r, &out_w, &sa, 0)) {
        ::CloseHandle(in_r); ::CloseHandle(in_w);
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::transport_error, "CreatePipe failed");
    }
    // Only the child's ends may be inherited, or it would hold ours open.
    ::SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    ::SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    std::string cmdline;
    for (std::size_t i = 0; i < spec.argv.size(); ++i) {
        if (i) cmdline.push_back(' ');
        cmdline += detail::quote_arg(spec.argv[i]);
    }
    std::vector<char> cmd(cmdline.begin(), cmdline.end());
    cmd.push_back('\0');
    std::vector<char> env;
    if (!spec.env.empty()) env = detail::env_block(spec.env);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = ::GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    const BOOL ok = ::CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0,
                                     env.empty() ? nullptr : env.data(),
                                     spec.cwd.empty() ? nullptr : spec.cwd.c_str(), &si, &pi);
    ::CloseHandle(in_r);
    ::CloseHandle(out_w);
    if (!ok) {
        ::CloseHandle(in_w); ::CloseHandle(out_r);
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(
            rag::Errc::transport_error, "CreateProcess failed for " + spec.argv[0]);
    }
    ::CloseHandle(pi.hThread);

    auto child = std::make_shared<detail::Child>();
    child->process = pi.hProcess;
    child->to_child = in_w;
    child->from_child = out_r;

    rag::bridge::LineIo io;
    io.read = [c = child.get()](char* buf, std::size_t cap) -> std::size_t {
        DWORD got = 0;
        const DWORD want = cap > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<DWORD>(cap);
        if (!::ReadFile(c->from_child, buf, want, &got, nullptr)) return 0;
        return got;
    };
    io.write = [c = child.get()](std::string_view bytes) -> bool {
        while (!bytes.empty()) {
            DWORD put = 0;
            const DWORD want = bytes.size() > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<DWORD>(bytes.size());
            if (!::WriteFile(c->to_child, bytes.data(), want, &put, nullptr) || put == 0) return false;
            bytes.remove_prefix(put);
        }
        return true;
    };
    io.keep = child;
    const std::string name = spec.name.empty() ? spec.argv[0] : spec.name;
    return std::shared_ptr<rag::bridge::Channel>(
        std::make_shared<rag::bridge::LineChannel>(std::move(io), name));
}

#else // POSIX

namespace detail {

struct Child {
    pid_t pid = -1;
    int to_child = -1;    // child's stdin
    int from_child = -1;  // child's stdout
    ~Child() {
        if (to_child >= 0) ::close(to_child);     // peer sees EOF and should exit
        if (from_child >= 0) ::close(from_child);
        if (pid <= 0) return;
        int status = 0;
        for (int i = 0; i < 100; ++i) {
            const pid_t r = ::waitpid(pid, &status, WNOHANG);
            if (r == pid || (r < 0 && errno != EINTR)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ::kill(pid, SIGKILL);
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    }
};

inline bool cloexec_pipe(int fds[2]) {
    if (::pipe(fds) != 0) return false;
    (void)::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    (void)::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    return true;
}

} // namespace detail

inline ChannelResult spawn_process(const rag::bridge::ProcessSpec& spec) {
    if (spec.argv.empty())
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::invalid_argument, "empty argv");

    // A write to a peer that already exited must fail, not kill the host.
    std::signal(SIGPIPE, SIG_IGN);

    int in_pipe[2] = {-1, -1};   // host writes [1] -> child stdin [0]
    int out_pipe[2] = {-1, -1};  // child stdout [1] -> host reads [0]
    if (!detail::cloexec_pipe(in_pipe))
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::transport_error, "pipe() failed");
    if (!detail::cloexec_pipe(out_pipe)) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::transport_error, "pipe() failed");
    }

    // Everything the child needs is prepared before fork().
    std::vector<char*> args;
    for (const auto& a : spec.argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]); ::close(out_pipe[0]); ::close(out_pipe[1]);
        return rag::fail<std::shared_ptr<rag::bridge::Channel>>(rag::Errc::transport_error, "fork() failed");
    }
    if (pid == 0) {
        // child: dup2 clears CLOEXEC on the duplicates; the originals close on exec.
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        if (!spec.cwd.empty() && ::chdir(spec.cwd.c_str()) != 0) ::_exit(127);
        for (const auto& kv : spec.env) ::putenv(const_cast<char*>(kv.c_str()));
        ::execvp(args[0], args.data());
        ::_exit(127);   // exec failed: the host sees EOF -> transport_error
    }
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    auto child = std::make_shared<detail::Child>();
    child->pid = pid;
    child->to_child = in_pipe[1];
    child->from_child = out_pipe[0];

    rag::bridge::LineIo io;
    io.read = [c = child.get()](char* buf, std::size_t cap) -> std::size_t {
        for (;;) {
            const ssize_t n = ::read(c->from_child, buf, cap);
            if (n >= 0) return static_cast<std::size_t>(n);
            if (errno != EINTR) return 0;
        }
    };
    io.write = [c = child.get()](std::string_view bytes) -> bool {
        while (!bytes.empty()) {
            const ssize_t n = ::write(c->to_child, bytes.data(), bytes.size());
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            bytes.remove_prefix(static_cast<std::size_t>(n));
        }
        return true;
    };
    io.keep = child;
    const std::string name = spec.name.empty() ? spec.argv[0] : spec.name;
    return std::shared_ptr<rag::bridge::Channel>(
        std::make_shared<rag::bridge::LineChannel>(std::move(io), name));
}

#endif

// The HostIo an example hands to rag: process spawning only (no HTTP).
inline rag::plugin::HostIo host_io() {
    rag::plugin::HostIo io;
    io.spawn = spawn_process;
    return io;
}

} // namespace example
