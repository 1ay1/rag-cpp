#pragma once
// rag/bridge/http_channel.hpp — a Channel backed by an HTTP/REST endpoint.
//
// Reuses the library's injectable HttpTransport seam so a remote engine exposed
// over HTTP (a FastAPI/Flask service, a hosted API, a Rust axum server) becomes
// a rag-cpp backend. Each call() POSTs {"method","params"} to a base path and
// parses the {ok/result/error} envelope. The method name is also appended to the
// path (`<base>/<method>`) so servers can route per-method if they prefer.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rag/bridge/channel.hpp"
#include "rag/dense/embedder.hpp"   // HttpTransport, default_http_transport
#include "rag/core/types.hpp"

namespace rag::bridge {

struct HttpChannelConfig {
    std::string   host = "127.0.0.1";
    std::uint16_t port = 8000;
    bool          tls  = false;
    std::string   base_path = "/rag";              // method appended: /rag/embed
    std::vector<std::pair<std::string, std::string>> headers; // e.g. Bearer auth
    std::chrono::milliseconds timeout{30'000};
    std::string   name;                            // identity label
};

class HttpChannel final : public Channel {
public:
    explicit HttpChannel(HttpChannelConfig cfg,
                         std::shared_ptr<dense::HttpTransport> tp = dense::default_http_transport())
        : cfg_(std::move(cfg)), tp_(std::move(tp)) {}

    [[nodiscard]] Result<Json> call(std::string_view method, const Json& params) override {
        Json req = Json::object();
        req["method"] = std::string(method);
        req["params"] = params;
        std::string body = req.dump();

        std::string path = cfg_.base_path;
        if (!path.empty() && path.back() == '/') path.pop_back();
        path += "/";
        path += method;

        dense::HttpRequest r{cfg_.host, cfg_.port, path, body, cfg_.headers, cfg_.tls, cfg_.timeout};
        auto resp = tp_->post(r);
        if (!resp) return std::unexpected(resp.error());
        if (resp->status < 200 || resp->status >= 300)
            return fail<Json>(Errc::transport_error,
                              "HTTP " + std::to_string(resp->status) + " from remote");
        Json reply;
        try {
            reply = Json::parse(resp->body);
        } catch (const std::exception& e) {
            return fail<Json>(Errc::transport_error,
                              std::string("remote sent invalid JSON: ") + e.what());
        }
        return unwrap_envelope(reply);
    }

    [[nodiscard]] std::string identity() const override {
        return "http:" + (cfg_.name.empty() ? cfg_.host : cfg_.name);
    }

private:
    HttpChannelConfig cfg_;
    std::shared_ptr<dense::HttpTransport> tp_;
};

} // namespace rag::bridge
