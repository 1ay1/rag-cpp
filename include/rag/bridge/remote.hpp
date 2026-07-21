#pragma once
// rag/bridge/remote.hpp — backends whose logic lives on the other side of a
// Channel (a subprocess, an HTTP service, another language).
//
// These are thin concept models: they marshal arguments to JSON, call() the
// Channel, and unmarshal the reply. Each one satisfies the same concept as a
// native C++ backend (Embedder / Reranker / Retriever), so a remote Python
// engine drops into the Corpus, the Pipeline, and the Engine with no special
// casing. This is what makes "any system, any language" first-class.
//
// Wire contracts (method → params → result):
//
//   embed:    {"texts": [str, ...]}                  -> {"vectors": [[float,...], ...]}
//   rerank:   {"query": str, "passages": [str, ...]} -> {"scores": [float, ...]}
//   retrieve: {"query": str, "k": int}               -> {"hits": [{"id": int|str,
//                                                                   "score": float,
//                                                                   "text": str?}, ...]}
//   graph:    {"op": str, "query": str?, "k": int?}  -> arbitrary JSON (op-specific)

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rag/bridge/channel.hpp"
#include "rag/core/document.hpp"
#include "rag/core/types.hpp"

namespace rag::bridge {

// ── RemoteEmbedder — models rag::Embedder ────────────────────────────────────
class RemoteEmbedder {
public:
    RemoteEmbedder(std::shared_ptr<Channel> ch, std::size_t dim, std::string identity)
        : ch_(std::move(ch)), dim_(dim), id_(std::move(identity)) {}

    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return id_; }

    [[nodiscard]] Result<std::vector<Vector>> embed(std::span<const std::string> texts) const {
        Json params;
        params["texts"] = Json::array();
        for (const auto& t : texts) params["texts"].push_back(t);

        auto r = ch_->call("embed", params);
        if (!r) return std::unexpected(r.error());
        const Json& res = *r;
        const Json* arr = res.contains("vectors") ? &res["vectors"] : &res;
        if (!arr->is_array())
            return fail<std::vector<Vector>>(Errc::transport_error, "embed: expected 'vectors' array");

        std::vector<Vector> out;
        out.reserve(arr->size());
        for (const auto& row : *arr) {
            if (!row.is_array())
                return fail<std::vector<Vector>>(Errc::transport_error, "embed: row not an array");
            Vector v;
            v.reserve(row.size());
            for (const auto& x : row) v.push_back(x.get<float>());
            out.push_back(std::move(v));
        }
        return out;
    }

private:
    std::shared_ptr<Channel> ch_;
    std::size_t              dim_;
    std::string              id_;
};

// ── RemoteReranker — models rag::rerank::Reranker (score list per passage) ────
class RemoteReranker {
public:
    RemoteReranker(std::shared_ptr<Channel> ch, std::string identity)
        : ch_(std::move(ch)), id_(std::move(identity)) {}

    [[nodiscard]] std::string_view identity() const noexcept { return id_; }

    [[nodiscard]] Result<std::vector<float>>
    rerank(std::string_view query, std::span<const std::string> passages) const {
        Json params;
        params["query"] = std::string(query);
        params["passages"] = Json::array();
        for (const auto& p : passages) params["passages"].push_back(p);

        auto r = ch_->call("rerank", params);
        if (!r) return std::unexpected(r.error());
        const Json& res = *r;
        const Json* arr = res.contains("scores") ? &res["scores"] : &res;
        if (!arr->is_array())
            return fail<std::vector<float>>(Errc::transport_error, "rerank: expected 'scores' array");
        std::vector<float> out;
        out.reserve(arr->size());
        for (const auto& s : *arr) out.push_back(s.get<float>());
        return out;
    }

private:
    std::shared_ptr<Channel> ch_;
    std::string              id_;
};

// A retrieval result from a remote engine: the peer speaks in its OWN id space
// (an int or an opaque string uri) plus an optional text payload, because it may
// index documents the C++ Corpus has never seen. Callers that want to fuse these
// with local hits key on `uri`.
struct RemoteHit {
    std::string uri;        // opaque id from the remote (stringified)
    float       score = 0;
    std::string text;       // optional passage text (empty if not returned)
};

// ── RemoteRetriever — a self-contained external search engine ─────────────────
// Unlike Embedder/Reranker (which feed the local index), a RemoteRetriever IS
// the index: it owns its own corpus on the far side (a Python FAISS store, an
// Elasticsearch cluster, a graph engine) and answers full queries.
class RemoteRetriever {
public:
    RemoteRetriever(std::shared_ptr<Channel> ch, std::string identity)
        : ch_(std::move(ch)), id_(std::move(identity)) {}

    [[nodiscard]] std::string_view identity() const noexcept { return id_; }

    [[nodiscard]] Result<std::vector<RemoteHit>>
    retrieve(std::string_view query, std::size_t k) const {
        Json params;
        params["query"] = std::string(query);
        params["k"] = k;
        auto r = ch_->call("retrieve", params);
        if (!r) return std::unexpected(r.error());
        const Json& res = *r;
        const Json* arr = res.contains("hits") ? &res["hits"] : &res;
        if (!arr->is_array())
            return fail<std::vector<RemoteHit>>(Errc::transport_error,
                                                "retrieve: expected 'hits' array");
        std::vector<RemoteHit> out;
        out.reserve(arr->size());
        for (const auto& h : *arr) {
            RemoteHit rh;
            if (h.contains("id")) {
                const auto& id = h["id"];
                rh.uri = id.is_string() ? id.get<std::string>() : id.dump();
            } else if (h.contains("uri") && h["uri"].is_string()) {
                rh.uri = h["uri"].get<std::string>();
            }
            if (h.contains("score")) rh.score = h["score"].get<float>();
            if (h.contains("text") && h["text"].is_string()) rh.text = h["text"].get<std::string>();
            out.push_back(std::move(rh));
        }
        return out;
    }

    // Escape hatch for arbitrary remote graph/engine operations (GraphRAG local
    // /global search, community summaries, custom pipelines). Returns raw JSON.
    [[nodiscard]] Result<Json> op(std::string_view name, const Json& params = Json::object()) const {
        Json p = params;
        p["op"] = std::string(name);
        return ch_->call("graph", p);
    }

private:
    std::shared_ptr<Channel> ch_;
    std::string              id_;
};

} // namespace rag::bridge
