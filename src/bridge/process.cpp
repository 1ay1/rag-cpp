// src/bridge/process.cpp — the newline-JSON protocol over a host LineIo.
// See rag/bridge/process.hpp.

#include "rag/bridge/process.hpp"

namespace rag::bridge {

Result<std::string> LineChannel::read_line() {
    for (;;) {
        if (auto nl = inbuf_.find('\n'); nl != std::string::npos) {
            std::string line = inbuf_.substr(0, nl);
            inbuf_.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return line;
        }
        char tmp[4096];
        const std::size_t n = io_.read ? io_.read(tmp, sizeof tmp) : 0;
        if (n == 0) {   // EOF: the peer closed stdout / exited
            alive_ = false;
            if (!inbuf_.empty()) { std::string line = std::move(inbuf_); inbuf_.clear(); return line; }
            return fail<std::string>(Errc::transport_error, "peer closed its output (EOF)");
        }
        inbuf_.append(tmp, n);
    }
}

Result<Json> LineChannel::call(std::string_view method, const Json& params) {
    if (!alive_) return fail<Json>(Errc::transport_error, "peer is gone");
    Json req = Json::object();
    req["method"] = std::string(method);
    req["params"] = params;

    std::string wire = req.dump();   // compact, single line
    wire.push_back('\n');
    if (!io_.write || !io_.write(wire)) {
        alive_ = false;
        return fail<Json>(Errc::transport_error, "write to peer failed");
    }

    auto line = read_line();
    if (!line) return std::unexpected(line.error());

    Json reply;
    try {
        reply = Json::parse(*line);
    } catch (const std::exception& e) {
        return fail<Json>(Errc::transport_error,
                          std::string("peer sent invalid JSON: ") + e.what());
    }
    return unwrap_envelope(reply);
}

} // namespace rag::bridge
