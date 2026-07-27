// rag/store/posix_compat.hpp — POSIX shims for the Windows (MSYS2/mingw) build.
//
// The store code is written against a small POSIX file-durability API. mingw's
// ucrt64 CRT already provides ::open/::close/::write/::lseek as global aliases,
// but has NO ::fsync, ::ftruncate, or ::kill. This header supplies those three
// in the GLOBAL namespace on Windows so the store's existing `::fsync(fd)`,
// `::ftruncate(fd, n)`, and `::kill(pid, 0)` calls compile and behave correctly
// with no source changes elsewhere.
//
// On any non-Windows platform this header just pulls in the real POSIX headers.
#pragma once

#if defined(_WIN32)

#include <io.h>          // _commit / _chsize_s / _lseeki64
#include <fcntl.h>       // _O_* flags
#include <sys/types.h>
#include <cerrno>
#include <cstddef>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>     // OpenProcess / GetExitCodeProcess for kill()

// _O_BINARY has no POSIX analogue; the O_* names used by the store already
// exist via <fcntl.h> on mingw, but define O_BINARY so callers can OR it in.
#ifndef O_BINARY
#define O_BINARY  _O_BINARY
#endif

// pid_t: mingw-w64 already provides it in <sys/types.h> (guarded internally).
// Only define our own if no toolchain guard is present, to avoid a redefinition.
#if !defined(_PID_T_) && !defined(__pid_t_defined) && !defined(_PID_T_DEFINED) \
    && !defined(__MINGW32__) && !defined(__MINGW64__)
using pid_t = long;
#define _PID_T_
#endif

// ── Global-namespace POSIX shims (Windows only) ──────────────────────────────
// Placed in the global namespace so existing `::fsync(fd)` etc. resolve here.

// fsync → _commit flushes the CRT buffer and the OS cache to the device.
inline int fsync(int fd) noexcept { return ::_commit(fd); }

inline int ftruncate(int fd, long long len) noexcept {
    return ::_chsize_s(fd, len) == 0 ? 0 : -1;
}

// kill(pid, 0): existence probe only — never signals. Returns 0 if the process
// is alive; -1 with errno=ESRCH when gone, errno=EPERM when present but
// inaccessible. Matches the single use in Container::sweep_orphan_temps.
inline int kill(pid_t pid, int /*sig*/) noexcept {
    HANDLE h = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                             static_cast<DWORD>(pid));
    if (h == nullptr) {
        errno = (::GetLastError() == ERROR_ACCESS_DENIED) ? EPERM : ESRCH;
        return -1;
    }
    DWORD code = 0;
    const BOOL ok = ::GetExitCodeProcess(h, &code);
    ::CloseHandle(h);
    if (ok && code == STILL_ACTIVE) return 0;   // alive
    errno = ESRCH;                               // exited
    return -1;
}

#else  // ── POSIX ─────────────────────────────────────────────────────────────

#include <fcntl.h>
#include <unistd.h>
#include <csignal>

#ifndef O_BINARY
#define O_BINARY 0       // no-op on POSIX
#endif

#endif
