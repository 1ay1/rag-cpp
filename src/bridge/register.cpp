// src/bridge/register.cpp — open_channel() + registry factories for polyglot
// backends ("process" and "http" transports → RemoteEmbedder / RemoteReranker).

#include "rag/bridge/bridge.hpp"

#include "rag/plugin/builder.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/rerank/reranker.hpp"

#include <string>
#include <vector>

namespace rag::bridge {

Result<std::shared_ptr<Channel>> open_channel(const Json& spec, const plugin::HostIo& io) {
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
        if (!io.spawn)
            return fail<std::shared_ptr<Channel>>(Errc::unavailable,
                "process transport: the host runs no processes for rag");
        ProcessSpec cfg;
        if (auto it = spec.find("argv"); it != spec.end() && it->is_array())
            for (const auto& a : *it) if (a.is_string()) cfg.argv.push_back(a.get<std::string>());
        if (auto it = spec.find("env"); it != spec.end() && it->is_array())
            for (const auto& e : *it) if (e.is_string()) cfg.env.push_back(e.get<std::string>());
        cfg.cwd  = spec.value("cwd", std::string{});
        cfg.name = spec.value("name", std::string{});
        if (cfg.argv.empty())
            return fail<std::shared_ptr<Channel>>(Errc::invalid_argument, "process transport: empty argv");
        if (cfg.name.empty()) cfg.name = cfg.argv.front();
        return io.spawn(cfg);
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
        if (!io.http)
            return fail<std::shared_ptr<Channel>>(Errc::unavailable,
                "http transport: the host provides no HTTP for rag");
        return std::shared_ptr<Channel>(std::make_shared<HttpChannel>(std::move(cfg), io.http));
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
Result<AnyEmbedder> make_remote_embedder(const Json& c, const plugin::Registries& all) {
    auto ch = open_channel(c, all.io);
    if (!ch) return std::unexpected(ch.error());
    std::size_t dim = c.value("dim", std::size_t{768});
    std::string id  = c.value("name", std::string{"remote-embed"});
    return AnyEmbedder{RemoteEmbedder{std::move(*ch), dim, std::move(id)}};
}

Result<AnyReranker> make_remote_reranker(const Json& c, const plugin::Registries& all) {
    auto ch = open_channel(c, all.io);
    if (!ch) return std::unexpected(ch.error());
    std::string id = c.value("name", std::string{"remote-rerank"});
    return AnyReranker{RemoteReranker{std::move(*ch), std::move(id)}};
}

} // namespace

void register_bridge(plugin::Registries& r) {
    const std::string emb_desc =
        "out-of-process embedder over a bridge (keys: cmd/args for process, url for http/rest)";
    for (const char* name : {"process", "http", "rest"})
        r.embedders.register_described(name,
            make_remote_embedder, emb_desc);
    const std::string rr_desc =
        "out-of-process reranker over a bridge (keys: cmd/args for process, url for http/rest)";
    for (const char* name : {"process", "http", "rest"})
        r.rerankers.register_described(name,
            make_remote_reranker, rr_desc);
}

} // namespace rag::bridge
