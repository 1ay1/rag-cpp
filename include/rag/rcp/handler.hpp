#pragma once
// rag/rcp/handler.hpp — rag::Engine exposed as an RCP/1 handler.
//
// EngineHandler is the concrete bridge: it wraps a reference to a host-owned
// `rag::Engine` and satisfies the RCP SDK `Handler` concept, translating every
// advertised RCP method onto the engine's retrieval surface. The host builds
// and fills the Engine; this class makes it speak the wire.
//
// Design (the rag-cpp / acp-cpp house style):
//
//   * The Engine is NOT owned — a reference is held. The host keeps ingesting,
//     rebuilding, persisting through its own handle; the server reads through
//     the same live corpus. (A server that copied the engine would answer from
//     a stale snapshot.)
//
//   * Capabilities are DATA, not inheritance. `Options` is a fluent record of
//     which RCP capabilities to advertise and their metadata; a method whose
//     capability is not advertised is rejected with -32003 by the SDK Server
//     *before* the hook is called, so a disabled feature is unreachable, not
//     merely unimplemented.
//
//   * `Hooks` lets a host OVERRIDE or EXTEND any method with its own
//     std::function (custom rerank model, auth-scoped retrieve, a graph engine)
//     without subclassing — the acp-cpp ClientHandlers pattern. An unset hook
//     falls through to the built-in Engine mapping.
//
//   * Every method returns Result<Json> = expected<Json, rcp::Error>. No
//     exceptions cross the wire; a domain failure is mapped by error.hpp.

#include "rag/engine.hpp"
#include "rag/rcp/convert.hpp"
#include "rag/rcp/error.hpp"

#include <rcp/protocol.hpp>
#include <rcp/server.hpp>
#include <rcp/types.hpp>
#include <rcp/vectors.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace rag::rcp {

using Json   = ::rcp::Json;
using Result = ::rcp::Result<Json>;

// ─────────────────────────────────────────────────────────────────────────────
// Options — what this server advertises and how it behaves. Fluent, all
// defaulted; a bare `Options{}` is a correct read-only hybrid retrieval server.
// ─────────────────────────────────────────────────────────────────────────────
struct Options {
    std::string name    = "rag-cpp";
    std::string version = "0.1.0";

    std::size_t max_k        = 100;    // advertised retrieve.maxK (§6.1)
    std::size_t default_k    = 10;

    bool enable_retrieve = true;       // the workhorse (§7.7)
    bool enable_embed    = true;       // gated on the engine actually having one
    bool enable_rerank   = false;      // needs a reranker hook or Engine reranker
    bool enable_graph    = false;      // GraphRAG local/global (§7.9)
    bool enable_index    = false;      // index/add + index/delete (§7.10/7.11)
    bool index_writable  = false;      // if enable_index: accept writes vs. read
    bool enable_feedback = false;      // relevance feedback sink (§7.16)

    // Advertised filter fields → type (§8). Empty ⇒ filter capability not
    // advertised; a non-empty map both advertises `filter` and constrains which
    // metadata keys a client may filter on.
    Json filter_fields = Json::object();

    // Fluent setters (return *this) — reads like a spec at the call site.
    Options& named(std::string n, std::string v) { name = std::move(n); version = std::move(v); return *this; }
    Options& with_rerank(bool on = true)   { enable_rerank = on; return *this; }
    Options& with_graph(bool on = true)    { enable_graph  = on; return *this; }
    Options& with_index(bool writable)     { enable_index = true; index_writable = writable; return *this; }
    Options& with_feedback(bool on = true) { enable_feedback = on; return *this; }
    Options& filter_on(std::string field, std::string type = "keyword") {
        filter_fields[std::move(field)] = std::move(type); return *this;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Hooks — optional host overrides. Any set function REPLACES the built-in
// mapping for that method; an unset function leaves the Engine default in place.
// This is the framework seam: a host adds capabilities the base Engine lacks
// (an external reranker, an LLM query rewriter, a policy-scoped retrieve) with
// zero subclassing.
// ─────────────────────────────────────────────────────────────────────────────
struct Hooks {
    std::function<Result(const Json&)> retrieve;
    std::function<Result(const Json&)> rerank;
    std::function<Result(const Json&)> embed;
    std::function<Result(const Json&)> graph;
    std::function<Result(const Json&)> index_add;
    std::function<Result(const Json&)> index_delete;
    std::function<Result(const Json&)> feedback;
    std::function<Result(const Json&)> transform;
};

// ─────────────────────────────────────────────────────────────────────────────
// EngineHandler — satisfies rcp::Handler. Holds Engine& + Options + Hooks.
// ─────────────────────────────────────────────────────────────────────────────
class EngineHandler {
public:
    EngineHandler(Engine& engine, Options opts = {}, Hooks hooks = {})
        : engine_(engine), opts_(std::move(opts)), hooks_(std::move(hooks)) {}

    // ── Required surface (Handler concept) ──────────────────────────────────
    [[nodiscard]] ::rcp::PeerInfo info() const {
        return ::rcp::PeerInfo{opts_.name, opts_.version};
    }

    [[nodiscard]] ::rcp::Capabilities capabilities() const {
        ::rcp::Capabilities c;
        c.citations = true;               // every Hit carries a citation (§14)
        c.log_      = true;

        if (opts_.enable_retrieve)
            c.with_retrieve(opts_.max_k, {"dense", "sparse", "hybrid"}, {"text"});

        // embed is advertised only when the engine actually has an embedder OR a
        // host hook supplies one — never claim a capability we can't honour.
        if (opts_.enable_embed && (engine_.corpus().embedder() || hooks_.embed)) {
            std::size_t dim = engine_.corpus().embedder() ? engine_.corpus().embedder()->dimension() : 0;
            std::string id  = engine_.corpus().embedder()
                                  ? std::string(engine_.corpus().embedder()->identity())
                                  : std::string("host");
            c.with_embed(::rcp::Dimension{dim}, std::move(id));
        }
        if (opts_.enable_rerank || hooks_.rerank) c.with_rerank({"cross-encoder"});
        if (opts_.enable_graph  || hooks_.graph)  c.with_graph({"local", "global"});
        if (opts_.enable_index  || hooks_.index_add)
            c.with_index(opts_.index_writable || (hooks_.index_add != nullptr));
        if (opts_.enable_feedback || hooks_.feedback) c.with_feedback();
        if (hooks_.transform) c.with_transform({"rewrite"});
        if (opts_.filter_fields.is_object() && !opts_.filter_fields.empty())
            c.with_filter(opts_.filter_fields,
                          {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "exists"});
        return c;
    }

    // ── retrieve (§7.7) — the workhorse ─────────────────────────────────────
    [[nodiscard]] Result retrieve(const Json& p) {
        if (hooks_.retrieve) return hooks_.retrieve(p);

        auto parsed = parse_retrieve(p, opts_.max_k);
        if (!parsed) return std::unexpected(parsed.error());
        const RetrieveParams& rp = *parsed;

        // Compile the filter (§8) into an engine predicate, if any advertised.
        index::MetaFilter filter;
        if (!rp.filter.is_null()) {
            auto f = compile_filter(rp.filter, opts_.filter_fields);
            if (!f) return std::unexpected(f.error());
            filter = std::move(*f);
        }

        // Recall width honours the funnel (§3.3): retrieve candidateK, rerank
        // (if any) narrows to topN, then we emit k. The Engine's pipeline fuses
        // dense+sparse internally; we drive it at the candidate width and trim.
        std::size_t want = rp.candidate_k.value_or(std::max<std::size_t>(4 * rp.k, rp.k));

        auto hits = engine_.search(rp.query, want, filter);
        if (!hits) return std::unexpected(to_wire(hits.error()));

        std::vector<rag::SearchResult> results = std::move(*hits);

        // minScore floor (§7.7).
        if (rp.min_score)
            std::erase_if(results, [&](const auto& r){ return r.score.get() < *rp.min_score; });

        // Trim to k (the funnel's narrow end).
        if (results.size() > rp.k) results.resize(rp.k);

        Json hit_arr = Json::array();
        std::size_t tokens = 0;
        for (const auto& r : results) {
            Json h = to_hit(r, engine_.corpus(), rp.include_text, rp.include_vectors);
            // tokenBudget (§7.7.2): stop packing once the budget is exhausted;
            // an approximate 4-chars-per-token heuristic, honest and cheap.
            if (rp.token_budget) {
                std::size_t t = h.contains("text") ? h["text"].get<std::string>().size() / 4 : 0;
                if (!hit_arr.empty() && tokens + t > *rp.token_budget) break;
                tokens += t;
            }
            hit_arr.push_back(std::move(h));
        }

        Json result{{"hits", std::move(hit_arr)}};
        // usage (§7.7): report the funnel widths actually used, so a client can
        // reason about recall/precision without guessing.
        result["usage"] = Json{{"candidateK", want},
                               {"returned", result["hits"].size()},
                               {"mode", rp.mode}};
        return result;
    }

    // ── embed (§7.3) ────────────────────────────────────────────────────────
    [[nodiscard]] Result embed(const Json& p) {
        if (hooks_.embed) return hooks_.embed(p);
        const index::Corpus& corpus = engine_.corpus();
        if (!corpus.embedder())
            return wire_fail(::rcp::errc::BackendUnavailable, "no embedder attached");

        if (!p.contains("input"))
            return wire_fail(::rcp::errc::InvalidParams, "input is required", Json{{"field", "input"}});

        std::vector<std::string> texts;
        const Json& in = p["input"];
        if (in.is_string()) texts.push_back(in.get<std::string>());
        else if (in.is_array())
            for (const auto& t : in) texts.push_back(t.is_string() ? t.get<std::string>() : t.dump());
        else return wire_fail(::rcp::errc::InvalidParams, "input must be string or array", Json{{"field", "input"}});

        // Compact base64 f32 encoding by default (§7.3.1); "float" array on request.
        bool compact = p.value("encoding", std::string("base64")) != "float";
        Json vectors = Json::array();
        std::size_t dim = corpus.embedder()->dimension();
        for (const auto& t : texts) {
            auto v = corpus.embed_text(t);
            if (!v) return std::unexpected(to_wire(v.error()));
            if (compact) vectors.push_back(::rcp::vectors::encode_f32_base64(*v));
            else { Json a = Json::array(); for (float x : *v) a.push_back(x); vectors.push_back(std::move(a)); }
        }
        return Json{{"vectors", std::move(vectors)},
                    {"dimension", dim},
                    {"encoding", compact ? "base64" : "float"},
                    {"model", std::string(corpus.embedder()->identity())}};
    }

    // ── rerank (§7.6) ───────────────────────────────────────────────────────
    [[nodiscard]] Result rerank(const Json& p) {
        if (hooks_.rerank) return hooks_.rerank(p);
        return wire_fail(::rcp::errc::CapabilityMissing,
                         "rerank requires a reranker hook; none configured");
    }

    // ── query/transform (§7.8) ──────────────────────────────────────────────
    [[nodiscard]] Result transform(const Json& p) {
        if (hooks_.transform) return hooks_.transform(p);
        return wire_fail(::rcp::errc::CapabilityMissing, "transform not configured");
    }

    // ── graph (§7.9) — GraphRAG local/global ────────────────────────────────
    [[nodiscard]] Result graph(const Json& p) {
        if (hooks_.graph) return hooks_.graph(p);
        std::string op = p.value("op", std::string("local"));
        std::string query = p.value("query", std::string{});
        std::size_t k = p.value("k", opts_.default_k);
        if (query.empty())
            return wire_fail(::rcp::errc::InvalidParams, "query is required", Json{{"field", "query"}});

        auto hits = (op == "global") ? engine_.graph_global(query, k)
                                     : engine_.graph_local(query, k);
        if (!hits) return std::unexpected(to_wire(hits.error()));

        Json arr = Json::array();
        for (const auto& r : *hits) {
            Json h = to_hit(r, engine_.corpus(), /*text*/true, /*vec*/false);
            h["unit"] = (op == "global") ? "community" : "node";
            arr.push_back(std::move(h));
        }
        return Json{{"hits", std::move(arr)}, {"usage", Json{{"op", op}}}};
    }

    // ── index/add (§7.10) — upsert, idempotent by id (uri) ──────────────────
    [[nodiscard]] Result index_add(const Json& p) {
        if (hooks_.index_add) return hooks_.index_add(p);
        if (!opts_.index_writable)
            return wire_fail(::rcp::errc::CapabilityMissing, "index is read-only");
        if (!p.contains("documents") || !p["documents"].is_array())
            return wire_fail(::rcp::errc::InvalidParams, "documents[] required", Json{{"field", "documents"}});

        // §7.10: ids MUST be returned positionally, one per input document; an
        // explicit id is an UPSERT (replace, never duplicate). We honour that by
        // tombstoning any live document with the same uri before re-adding.
        Json ids = Json::array();
        for (const auto& d : p["documents"]) {
            std::string uri  = d.value("id", d.value("uri", std::string{}));
            std::string text = d.value("text", std::string{});
            std::string title = d.value("title", std::string{});
            Metadata meta;
            if (d.contains("meta") && d["meta"].is_object())
                for (auto it = d["meta"].begin(); it != d["meta"].end(); ++it)
                    meta[it.key()] = it->is_string() ? it->get<std::string>() : it->dump();

            if (!uri.empty())
                if (auto existing = engine_.corpus().find_by_uri(uri))
                    engine_.corpus().remove_document(*existing);   // upsert

            auto id = engine_.corpus().add_document(uri, text, meta, title);
            if (!id) return std::unexpected(to_wire(id.error()));
            ids.push_back(uri.empty() ? std::to_string(id->get()) : uri);
        }
        auto b = engine_.build();
        if (!b) return std::unexpected(to_wire(b.error()));
        return Json{{"ids", std::move(ids)}, {"chunks", engine_.corpus().chunk_count()}};
    }

    // ── index/delete (§7.11) — idempotent, by id (uri) or numeric DocId ─────
    [[nodiscard]] Result index_delete(const Json& p) {
        if (hooks_.index_delete) return hooks_.index_delete(p);
        if (!opts_.index_writable)
            return wire_fail(::rcp::errc::CapabilityMissing, "index is read-only");
        // §7.11: idempotent — deleting an absent/already-deleted id is not an
        // error, and `deleted` counts only documents actually removed here.
        std::size_t deleted = 0;
        if (p.contains("ids") && p["ids"].is_array())
            for (const auto& idj : p["ids"]) {
                std::optional<DocId> target;
                if (idj.is_string()) target = engine_.corpus().find_by_uri(idj.get<std::string>());
                else if (idj.is_number_integer()) target = DocId{idj.get<std::uint32_t>()};
                if (target && engine_.corpus().remove_document(*target)) ++deleted;
            }
        return Json{{"deleted", deleted}};
    }

    // ── feedback (§7.16) ────────────────────────────────────────────────────
    [[nodiscard]] Result feedback(const Json& p) {
        if (hooks_.feedback) return hooks_.feedback(p);
        // Base engine has no online learner; accept & acknowledge (a host hook
        // can persist signals). Idempotent, never fails on well-formed input.
        return Json{{"accepted", true}, {"received", p.value("signals", Json::array()).size()}};
    }

private:
    Engine& engine_;
    Options opts_;
    Hooks   hooks_;
};

static_assert(::rcp::Handler<EngineHandler>, "EngineHandler must satisfy the RCP Handler concept");

} // namespace rag::rcp
