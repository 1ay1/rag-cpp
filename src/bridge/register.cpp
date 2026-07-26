// src/bridge/register.cpp — open_channel() + registry factories for polyglot
// backends ("process" and "http" transports → RemoteEmbedder / RemoteReranker).

#include "rag/bridge/bridge.hpp"

#include "rag/plugin/registry.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/rerank/reranker.hpp"

#include <string>
#include <vector>

namespace rag::bridge {

Result<std::shared_ptr<Channel>> open_channel(const Json& spec) {
    if (!spec.is_object())
        return fail<std::shared_ptr<Channel>>(Errc::invalid_argument,
                                              "channel spec must be an object");
    std::string transport;
    for (const char* key : {"transport", "type", "channel"}) {
        if (auto it = spec.find(key); it != spec.end() && it->is_string()) {
            transport = it->get<std::string>();
            break;
        }
    }

    if (transport == "process" || transport == "subprocess") {
        ProcessConfig cfg;
        if (auto it = spec.find("argv"); it != spec.end() && it->is_array())
            for (const auto& a : *it) if (a.is_string()) cfg.argv.push_back(a.get<std::string>());
        if (auto it = spec.find("env"); it != spec.end() && it->is_array())
            for (const auto& e : *it) if (e.is_string()) cfg.env.push_back(e.get<std::string>());
        cfg.cwd  = spec.value("cwd", std::string{});
        cfg.name = spec.value("name", std::string{});
        auto ch = ProcessChannel::spawn(std::move(cfg));
        if (!ch) return std::unexpected(ch.error());
        return std::shared_ptr<Channel>(*ch);
    }

    if (transport == "http" || transport == "rest") {
        HttpChannelConfig cfg;
        cfg.host      = spec.value("host", cfg.host);
        cfg.port      = static_cast<std::uint16_t>(spec.value("port", int(cfg.port)));
        cfg.tls       = spec.value("tls", cfg.tls);
        cfg.base_path = spec.value("base_path", cfg.base_path);
        cfg.name      = spec.value("name", std::string{});
        if (auto it = spec.find("headers"); it != spec.end() && it->is_object())
            for (const auto& [k, v] : it->items())
                if (v.is_string()) cfg.headers.emplace_back(k, v.get<std::string>());
        return std::shared_ptr<Channel>(std::make_shared<HttpChannel>(std::move(cfg)));
    }

    return fail<std::shared_ptr<Channel>>(Errc::not_found,
                                          "unknown transport '" + transport + "' "
                                          "(want 'process' or 'http')");
}

namespace {

using ::rag::dense::AnyEmbedder;
using ::rag::rerank::AnyReranker;

// One factory body shared by "process" and "http": open the channel from the
// same spec, then wrap it. `dim` is required for an embedder (the peer's output
// width) — default 768 if omitted.
Result<AnyEmbedder> make_remote_embedder(const Json& c) {
    auto ch = open_channel(c);
    if (!ch) return std::unexpected(ch.error());
    std::size_t dim = c.value("dim", std::size_t{768});
    std::string id  = c.value("name", std::string{"remote-embed"});
    return AnyEmbedder{RemoteEmbedder{std::move(*ch), dim, std::move(id)}};
}

Result<AnyReranker> make_remote_reranker(const Json& c) {
    auto ch = open_channel(c);
    if (!ch) return std::unexpected(ch.error());
    std::string id = c.value("name", std::string{"remote-rerank"});
    return AnyReranker{RemoteReranker{std::move(*ch), std::move(id)}};
}

} // namespace

void ensure_bridge_registered() noexcept {
    static const bool once = [] {
        auto& er = plugin::Registry<AnyEmbedder>::instance();
        const std::string emb_desc =
            "out-of-process embedder over a bridge (keys: cmd/args for process, url for http/rest)";
        er.register_described("process", make_remote_embedder, emb_desc);
        er.register_described("http",    make_remote_embedder, emb_desc);
        er.register_described("rest",    make_remote_embedder, emb_desc);
        auto& rr = plugin::Registry<AnyReranker>::instance();
        const std::string rr_desc =
            "out-of-process reranker over a bridge (keys: cmd/args for process, url for http/rest)";
        rr.register_described("process", make_remote_reranker, rr_desc);
        rr.register_described("http",    make_remote_reranker, rr_desc);
        rr.register_described("rest",    make_remote_reranker, rr_desc);
        return true;
    }();
    (void)once;
}

} // namespace rag::bridge
