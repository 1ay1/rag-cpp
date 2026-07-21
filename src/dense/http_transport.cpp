// rag/dense/http_transport.cpp — a minimal blocking HTTP/1.1 POST client.
//
// Plaintext HTTP over BSD sockets (POSIX) / Winsock. Intended for localhost
// model servers (Ollama, llama.cpp, LM Studio). For TLS endpoints, inject a
// custom HttpTransport backed by your TLS stack — that is exactly why the seam
// exists.

#include "rag/dense/embedder.hpp"

#include <array>
#include <cstring>
#include <string>

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
#endif

namespace rag::dense {

namespace {

#if defined(_WIN32)
struct WsaInit { WsaInit() { WSADATA d; WSAStartup(MAKEWORD(2,2), &d); } } g_wsa;
using socket_t = SOCKET;
constexpr socket_t kInvalid = INVALID_SOCKET;
void close_sock(socket_t s) { closesocket(s); }
#else
using socket_t = int;
constexpr socket_t kInvalid = -1;
void close_sock(socket_t s) { ::close(s); }
#endif

class DefaultTransport final : public HttpTransport {
public:
    Result<HttpResponse> post(const HttpRequest& req) const override {
        if (req.tls)
            return fail<HttpResponse>(Errc::unavailable,
                "default transport is plaintext; inject a TLS transport for https");

        // Resolve.
        addrinfo hints{};
        hints.ai_family   = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        std::string host_s(req.host);
        std::string port_s = std::to_string(req.port);
        addrinfo* res = nullptr;
        if (::getaddrinfo(host_s.c_str(), port_s.c_str(), &hints, &res) != 0 || !res)
            return fail<HttpResponse>(Errc::unavailable, "getaddrinfo failed");

        socket_t fd = kInvalid;
        for (addrinfo* ai = res; ai; ai = ai->ai_next) {
            fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd == kInvalid) continue;
            set_timeout(fd, req.timeout);
            if (::connect(fd, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) break;
            close_sock(fd); fd = kInvalid;
        }
        ::freeaddrinfo(res);
        if (fd == kInvalid) return fail<HttpResponse>(Errc::unavailable, "connect failed");

        // Build request.
        std::string r;
        r.reserve(req.body.size() + 256);
        r += "POST "; r += req.path; r += " HTTP/1.1\r\n";
        r += "Host: "; r += req.host; r += "\r\n";
        r += "Content-Type: application/json\r\n";
        for (const auto& [k, v] : req.headers) { r += k; r += ": "; r += v; r += "\r\n"; }
        r += "Content-Length: " + std::to_string(req.body.size()) + "\r\n";
        r += "Connection: close\r\n\r\n";
        r += req.body;

        if (!send_all(fd, r)) { close_sock(fd); return fail<HttpResponse>(Errc::transport_error, "send"); }

        // Read full response.
        std::string raw;
        std::array<char, 8192> buf;
        while (true) {
            auto n = ::recv(fd, buf.data(), static_cast<int>(buf.size()), 0);
            if (n <= 0) break;
            raw.append(buf.data(), static_cast<std::size_t>(n));
        }
        close_sock(fd);
        return parse_response(raw);
    }

private:
    static void set_timeout(socket_t fd, std::chrono::milliseconds t) {
#if defined(_WIN32)
        DWORD ms = static_cast<DWORD>(t.count());
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
        timeval tv;
        tv.tv_sec  = static_cast<long>(t.count() / 1000);
        tv.tv_usec = static_cast<long>((t.count() % 1000) * 1000);
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    }

    static bool send_all(socket_t fd, const std::string& data) {
        std::size_t off = 0;
        while (off < data.size()) {
            auto n = ::send(fd, data.data() + off, static_cast<int>(data.size() - off), 0);
            if (n <= 0) return false;
            off += static_cast<std::size_t>(n);
        }
        return true;
    }

    static Result<HttpResponse> parse_response(const std::string& raw) {
        auto hdr_end = raw.find("\r\n\r\n");
        if (hdr_end == std::string::npos) return fail<HttpResponse>(Errc::transport_error, "no header end");
        // Status line: HTTP/1.1 200 OK
        HttpResponse resp;
        {
            auto sp = raw.find(' ');
            if (sp != std::string::npos)
                resp.status = std::atoi(raw.c_str() + sp + 1);
        }
        std::string headers = raw.substr(0, hdr_end);
        std::string body = raw.substr(hdr_end + 4);

        // Handle chunked transfer-encoding (Ollama uses Content-Length, but be safe).
        std::string lower = headers;
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find("transfer-encoding: chunked") != std::string::npos) {
            resp.body = dechunk(body);
        } else {
            resp.body = std::move(body);
        }
        return resp;
    }

    static std::string dechunk(const std::string& in) {
        std::string out;
        std::size_t i = 0;
        while (i < in.size()) {
            auto eol = in.find("\r\n", i);
            if (eol == std::string::npos) break;
            std::size_t len = std::strtoul(in.substr(i, eol - i).c_str(), nullptr, 16);
            if (len == 0) break;
            i = eol + 2;
            if (i + len > in.size()) break;
            out.append(in, i, len);
            i += len + 2;
        }
        return out;
    }
};

} // namespace

std::shared_ptr<HttpTransport> default_http_transport() {
    static std::shared_ptr<HttpTransport> tp = std::make_shared<DefaultTransport>();
    return tp;
}

} // namespace rag::dense
