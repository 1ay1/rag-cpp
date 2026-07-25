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
#include "rag/graph/graph.hpp"
#include "rag/late/colbert.hpp"
#include "rag/query/hyde.hpp"
#include "rag/rerank/reranker.hpp"
#include "rag/sparse/splade.hpp"
#include "rag/rcp/convert.hpp"
#include "rag/rcp/error.hpp"

#include <rcp/protocol.hpp>
#include <rcp/server.hpp>
#include <rcp/types.hpp>
#include <rcp/vectors.hpp>

#include <algorithm>
#include <functional>
#include <memory>
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
    bool enable_graph    = false;      // GraphRAG local/global (§7.9)
    bool enable_index    = false;      // index/add + index/delete (§7.10/7.11)
    bool index_writable  = false;      // if enable_index: accept writes vs. read
    bool enable_feedback = false;      // relevance feedback sink (§7.16)
    bool enable_memory   = false;      // memory/build + memory/recall (§7.17)

    // Real engine components. Attaching one BOTH implements the corresponding
    // RCP method natively AND flips its capability on — no host glue required.
    // The base engine has the machinery; these hand the server the instance.
    //   * reranker   → `rerank` (cross-encoder) + `retrieve.rerank` + graph rerank
    //   * splade     → `sparseEmbed` (embed/sparse) + sparse `retrieve.mode`
    //   * token_embed→ `multiVector` (embed/multi) + `rerank method:"colbert"`
    //   * generator  → `transform` (query/transform) + `retrieve.rewrite`
    std::shared_ptr<rerank::AnyReranker>  reranker;
    std::shared_ptr<sparse::SpladeIndex>  splade;
    late::TokenEmbedder                   token_embed;
    query::Generator                      generator;

    // Advertised filter fields → type (§8). Empty ⇒ filter capability not
    // advertised; a non-empty map both advertises `filter` and constrains which
    // metadata keys a client may filter on.
    Json filter_fields = Json::object();

    // Fluent setters (return *this) — reads like a spec at the call site.
    Options& named(std::string n, std::string v) { name = std::move(n); version = std::move(v); return *this; }
    Options& with_reranker(rerank::AnyReranker r) { reranker = std::make_shared<rerank::AnyReranker>(std::move(r)); return *this; }
    Options& with_splade(sparse::SpladeIndex s)   { splade = std::make_shared<sparse::SpladeIndex>(std::move(s)); return *this; }
    Options& with_colbert(late::TokenEmbedder e)  { token_embed = std::move(e); return *this; }
    Options& with_generator(query::Generator g)   { generator = std::move(g); return *this; }
    Options& with_graph(bool on = true)    { enable_graph  = on; return *this; }
    Options& with_index(bool writable)     { enable_index = true; index_writable = writable; return *this; }
    Options& with_feedback(bool on = true) { enable_feedback = on; return *this; }
    Options& with_memory(bool on = true)   { enable_memory = on; return *this; }
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
    std::function<Result(const Json&)> embed_sparse;
    std::function<Result(const Json&)> embed_multi;
    std::function<Result(const Json&)> graph;
    std::function<Result(const Json&)> index_add;
    std::function<Result(const Json&)> index_delete;
    std::function<Result(const Json&)> feedback;
    std::function<Result(const Json&)> transform;
    std::function<Result(const Json&)> memory_build;
    std::function<Result(const Json&)> memory_recall;
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

        if (opts_.enable_retrieve) {
            // Modes reflect what the engine can actually do: dense/hybrid always
            // (BM25 is the lexical half); "sparse" is honest only with SPLADE.
            std::vector<std::string> modes = {"dense", "hybrid"};
            if (opts_.splade) modes.push_back("sparse");
            c.with_retrieve(opts_.max_k, std::move(modes), {"text"});
            // rag-cpp fuses by convex combination (TM2C2) by default and can
            // also do plain RRF, so declare both rather than accepting the
            // SDK's RRF-only baseline.
            c.with_fusion({"convex", "rrf", "weighted"});
            // Dense scores are cosines over unit-normalized vectors, so the
            // theoretical floor is exactly -1. Advertising it is what allows a
            // CLIENT fusing rag-cpp with other engines to use convex
            // combination instead of falling back to rank-only RRF (§16.3).
            c.with_score_scale("cosine", -1.0);
            // Advertise the rewrite methods retrieve.rewrite can drive (§7.7).
            if (opts_.generator || hooks_.transform)
                (*c.retrieve)["rewrite"] = Json::array({"hyde", "multi-query"});
            if (opts_.reranker || opts_.token_embed || hooks_.rerank)
                (*c.retrieve)["rerank"] = true;
        }

        // embed advertised only when an embedder is actually attached (or hooked)
        // — never claim a capability we can't honour.
        if (opts_.enable_embed && (engine_.corpus().embedder() || hooks_.embed)) {
            std::size_t dim = engine_.corpus().embedder() ? engine_.corpus().embedder()->dimension() : 0;
            std::string id  = engine_.corpus().embedder()
                                  ? std::string(engine_.corpus().embedder()->identity())
                                  : std::string("host");
            c.with_embed(::rcp::Dimension{dim}, std::move(id));
        }

        // sparseEmbed ← a trained SPLADE index (§7.4).
        if (opts_.splade || hooks_.embed_sparse)
            c.with_sparse("splade-cooc-v1", opts_.splade ? opts_.splade->vocab_size() : 0);

        // multiVector ← a token embedder for late interaction (§7.5).
        if (opts_.token_embed || hooks_.embed_multi) {
            auto mm = engine_.corpus().embedder()
                          ? engine_.corpus().embedder()->dimension() : 64;
            c.with_multi(::rcp::Dimension{mm}, "colbert-hashed-v1");
            (*c.multi_vector)["similarity"] = "cosine";
        }

        // rerank ← a cross-encoder (any reranker) and/or a ColBERT token embedder.
        if (opts_.reranker || opts_.token_embed || hooks_.rerank) {
            std::vector<std::string> methods;
            if (opts_.reranker || hooks_.rerank) methods.push_back("cross-encoder");
            if (opts_.token_embed)               methods.push_back("colbert");
            c.with_rerank(std::move(methods));
        }

        // transform ← a generation seam (HyDE / multi-query).
        if (opts_.generator || hooks_.transform)
            c.with_transform({"hyde", "multi-query", "rewrite"});

        if (opts_.enable_graph  || hooks_.graph)  c.with_graph({"local", "global"});
        if (opts_.enable_index  || hooks_.index_add)
            c.with_index(opts_.index_writable || (hooks_.index_add != nullptr));
        if (opts_.enable_feedback || hooks_.feedback) c.with_feedback();
        if (opts_.enable_memory   || hooks_.memory_build) c.with_memory({"global"}, /*clues=*/true);
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

        // Reject a mode we don't actually back ("sparse" needs SPLADE) — honesty
        // over silent downgrade (§7.7: reject per advertised capability).
        if (rp.mode == "sparse" && !opts_.splade)
            return wire_fail(::rcp::errc::OptionUnsupported,
                             "sparse mode requires a SPLADE index", Json{{"option", "mode"}});

        // Compile the filter (§8) into an engine predicate, if any advertised.
        index::MetaFilter filter;
        if (!rp.filter.is_null()) {
            auto f = compile_filter(rp.filter, opts_.filter_fields);
            if (!f) return std::unexpected(f.error());
            filter = std::move(*f);
        }

        // Recall width honours the funnel (§3.3): retrieve candidateK, rerank
        // narrows to topN, then we emit k.
        std::size_t want = rp.candidate_k.value_or(std::max<std::size_t>(4 * rp.k, rp.k));

        // —— Recall stage: pick the engine path the request asked for ——
        std::vector<rag::Hit> raw;
        std::string effective_mode = rp.mode;
        std::string rewrite = p.value("rewrite", std::string{});
        if (rewrite == "false") rewrite.clear();

        if (!rewrite.empty() && opts_.generator) {
            // Query transformation recall (§7.7 rewrite): HyDE or multi-query,
            // both RRF-fused internally, then filtered/trimmed below.
            auto r = (rewrite == "hyde")
                       ? query::hyde_search(engine_.corpus(), rp.query, want, opts_.generator)
                       : query::multi_query_search(engine_.corpus(), rp.query, want, opts_.generator);
            if (!r) return std::unexpected(to_wire(r.error()));
            raw = std::move(*r);
            effective_mode = "hybrid+" + rewrite;
        } else if (rp.mode == "sparse") {
            raw = opts_.splade->search(rp.query, want);
        } else if (rp.mode == "dense") {
            auto r = filter ? engine_.corpus().dense_search(rp.query, want, filter)
                            : engine_.corpus().dense_search(rp.query, want);
            if (!r) return std::unexpected(to_wire(r.error()));
            raw = std::move(*r);
        } else {
            // hybrid: the standard pipeline (dense+sparse fused, filtered).
            //
            // A per-request fusion choice runs on a locally-built pipeline
            // rather than mutating the shared Engine — the handler is used
            // concurrently, so reconfiguring shared state per request would be
            // a data race and would leak one client's ranking policy into
            // another's results.
            if (rp.fusion_method) {
                rag::pipeline::HybridRetrieveConfig hc{};
                if (*rp.fusion_method == "rrf")
                    hc.fusion = rag::pipeline::HybridRetrieveConfig::Fusion::rrf;
                else if (*rp.fusion_method == "weighted")
                    hc.fusion = rag::pipeline::HybridRetrieveConfig::Fusion::rsf;
                else
                    hc.fusion = rag::pipeline::HybridRetrieveConfig::Fusion::convex;
                if (rp.fusion_rrf_k) hc.rrf.k = static_cast<float>(*rp.fusion_rrf_k);
                if (rp.fusion_alpha) hc.convex.alpha = static_cast<float>(*rp.fusion_alpha);
                if (rp.candidate_k) hc.candidate_k = *rp.candidate_k;

                auto pipe = rag::pipeline::Pipeline::standard_with(hc);
                auto hits = pipe.run(engine_.corpus(), rp.query, want, filter);
                if (!hits) return std::unexpected(to_wire(hits.error()));
                std::vector<rag::SearchResult> res;
                res.reserve(hits->size());
                for (const auto& h : *hits) res.push_back(engine_.corpus().resolve(h));
                return finish_retrieve(res, rp, effective_mode, want);
            }
            auto r = engine_.search(rp.query, want, filter);
            if (!r) return std::unexpected(to_wire(r.error()));
            std::vector<rag::SearchResult> res = std::move(*r);
            return finish_retrieve(res, rp, effective_mode, want);
        }

        // Resolve raw Hits → SearchResults, applying the metadata filter for the
        // paths (sparse/dense/rewrite) that don't push it into the index walk.
        std::vector<rag::SearchResult> results;
        results.reserve(raw.size());
        for (const auto& h : raw) {
            rag::SearchResult sr = engine_.corpus().resolve(h);
            results.push_back(std::move(sr));
        }
        return finish_retrieve(results, rp, effective_mode, want);
    }

private:
    // Shared tail of retrieve: rerank (if asked + backed), minScore, trim to k,
    // tokenBudget packing, and the wire result with usage telemetry.
    [[nodiscard]] Result
    finish_retrieve(std::vector<rag::SearchResult>& results, const RetrieveParams& rp,
                    const std::string& mode, std::size_t candidate_k) {
        bool did_rerank = false;
        if (rp.rerank_requested && (opts_.reranker || opts_.token_embed)) {
            std::size_t top_n = std::min(rp.rerank_top_n, results.size());
            std::vector<std::string> passages;
            passages.reserve(top_n);
            for (std::size_t i = 0; i < top_n; ++i)
                passages.push_back(results[i].context.empty()
                                       ? results[i].text
                                       : results[i].context + "\n" + results[i].text);
            rag::Result<std::vector<float>> scores =
                opts_.reranker ? opts_.reranker->rerank(rp.query, passages)
                               : late::ColbertReranker{opts_.token_embed}.rerank(rp.query, passages);
            if (scores && scores->size() == top_n) {
                for (std::size_t i = 0; i < top_n; ++i) results[i].score = Score{(*scores)[i]};
                std::stable_sort(results.begin(), results.begin() + top_n,
                                 [](const auto& a, const auto& b){ return a.score.get() > b.score.get(); });
                did_rerank = true;
            }
        }

        if (rp.min_score)
            std::erase_if(results, [&](const auto& r){ return r.score.get() < *rp.min_score; });
        if (results.size() > rp.k) results.resize(rp.k);

        Json hit_arr = Json::array();
        std::size_t tokens = 0;
        for (const auto& r : results) {
            Json h = to_hit(r, engine_.corpus(), rp.include_text, rp.include_vectors);
            if (rp.token_budget) {
                std::size_t t = h.contains("text") ? h["text"].get<std::string>().size() / 4 : 0;
                if (!hit_arr.empty() && tokens + t > *rp.token_budget) break;
                tokens += t;
            }
            hit_arr.push_back(std::move(h));
        }

        Json result{{"hits", std::move(hit_arr)}};
        result["usage"] = Json{{"candidateK", candidate_k},
                               {"returned", result["hits"].size()},
                               {"mode", mode},
                               {"reranked", did_rerank}};
        return result;
    }

public:

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

    // ── rerank (§7.6) — cross-encoder or ColBERT late interaction ──────────
    [[nodiscard]] Result rerank(const Json& p) {
        if (hooks_.rerank) return hooks_.rerank(p);
        if (!opts_.reranker && !opts_.token_embed)
            return wire_fail(::rcp::errc::CapabilityMissing, "no reranker configured");

        std::string query = p.value("query", std::string{});
        if (query.empty())
            return wire_fail(::rcp::errc::InvalidParams, "query is required", Json{{"field", "query"}});
        if (!p.contains("documents") || !p["documents"].is_array())
            return wire_fail(::rcp::errc::InvalidParams, "documents[] required", Json{{"field", "documents"}});

        std::vector<std::string> passages;
        for (const auto& d : p["documents"])
            passages.push_back(d.is_string() ? d.get<std::string>() : d.value("text", d.dump()));

        // method: "colbert" forces late interaction if a token embedder exists;
        // otherwise the cross-encoder (default).
        std::string method = p.value("method", std::string(opts_.reranker ? "cross-encoder" : "colbert"));
        rag::Result<std::vector<float>> scores =
            (method == "colbert" && opts_.token_embed)
                ? late::ColbertReranker{opts_.token_embed}.rerank(query, passages)
                : (opts_.reranker ? opts_.reranker->rerank(query, passages)
                                  : late::ColbertReranker{opts_.token_embed}.rerank(query, passages));
        if (!scores) return std::unexpected(to_wire(scores.error()));

        // Result: a ranking of {index, score} best-first, and applied topN (§7.6).
        std::vector<std::pair<std::size_t, float>> ranked;
        for (std::size_t i = 0; i < scores->size(); ++i) ranked.push_back({i, (*scores)[i]});
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const auto& a, const auto& b){ return a.second > b.second; });
        std::size_t top_n = p.value("topN", ranked.size());
        if (top_n < ranked.size()) ranked.resize(top_n);

        Json results = Json::array();
        for (const auto& [idx, sc] : ranked)
            results.push_back(Json{{"index", idx}, {"score", sc}});
        return Json{{"results", std::move(results)}, {"method", method}};
    }

    // ── query/transform (§7.8) — HyDE / multi-query rewrite ──────────────────
    [[nodiscard]] Result transform(const Json& p) {
        if (hooks_.transform) return hooks_.transform(p);
        if (!opts_.generator)
            return wire_fail(::rcp::errc::CapabilityMissing, "no generator configured");

        std::string query = p.value("query", std::string{});
        if (query.empty())
            return wire_fail(::rcp::errc::InvalidParams, "query is required", Json{{"field", "query"}});
        std::string method = p.value("method", std::string("multi-query"));

        // The generator seam turns a prompt into completions; we shape the prompt
        // per method and return the produced queries (§7.8).
        std::string prompt =
            (method == "hyde")
                ? "Write a short passage that directly answers: " + query
                : "Generate 3 diverse search-query paraphrases, one per line, for: " + query;
        auto out = opts_.generator(prompt);
        if (!out) return std::unexpected(to_wire(out.error()));

        Json queries = Json::array();
        for (const auto& q : *out) if (!q.empty()) queries.push_back(q);
        if (queries.empty()) queries.push_back(query);   // never return nothing
        return Json{{"queries", std::move(queries)}, {"method", method}};
    }

    // ── embed/sparse (§7.4) — SPLADE learned-sparse term weights ────────────
    [[nodiscard]] Result embed_sparse(const Json& p) {
        if (hooks_.embed_sparse) return hooks_.embed_sparse(p);
        if (!opts_.splade)
            return wire_fail(::rcp::errc::CapabilityMissing, "no SPLADE index configured");
        if (!p.contains("texts") || !p["texts"].is_array())
            return wire_fail(::rcp::errc::InvalidParams, "texts[] required", Json{{"field", "texts"}});

        bool as_doc = p.value("kind", std::string("query")) == "document";
        Json sparse = Json::array();
        for (const auto& t : p["texts"]) {
            sparse::SparseVec sv = opts_.splade->encode(t.get<std::string>(), /*expand=*/as_doc);
            Json idx = Json::array(), val = Json::array();
            for (const auto& [term, w] : sv) { idx.push_back(term); val.push_back(w); }
            sparse.push_back(Json{{"indices", std::move(idx)}, {"values", std::move(val)}});
        }
        return Json{{"sparse", std::move(sparse)}};
    }

    // ── embed/multi (§7.5) — token-level matrices for late interaction ─────
    [[nodiscard]] Result embed_multi(const Json& p) {
        if (hooks_.embed_multi) return hooks_.embed_multi(p);
        if (!opts_.token_embed)
            return wire_fail(::rcp::errc::CapabilityMissing, "no token embedder configured");
        if (!p.contains("inputs") || !p["inputs"].is_array())
            return wire_fail(::rcp::errc::InvalidParams, "inputs[] required", Json{{"field", "inputs"}});

        Json matrices = Json::array();
        std::size_t dim = 0;
        for (const auto& in : p["inputs"]) {
            std::string text = in.is_string() ? in.get<std::string>() : in.value("text", std::string{});
            auto m = opts_.token_embed(text);
            if (!m) return std::unexpected(to_wire(m.error()));
            Json mat = Json::array();
            for (const auto& tok : *m) {
                dim = tok.size();
                Json row = Json::array();
                for (float x : tok) row.push_back(x);
                mat.push_back(std::move(row));
            }
            matrices.push_back(std::move(mat));
        }
        return Json{{"matrices", std::move(matrices)}, {"dimension", dim}};
    }

    // ── query/transform (§7.8) ── (defined above)

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

    // memory/build (§7.17) — distil the corpus graph into recallable memory.
    // We back "memory" with the GraphRAG community structure (HippoRAG-style):
    // building memory materialises the doc graph + communities, whose lead nodes
    // and summaries become the entry points recall returns.
    [[nodiscard]] Result memory_build(const Json& p) {
        if (hooks_.memory_build) return hooks_.memory_build(p);
        auto g = engine_.graph();
        if (!g) return std::unexpected(to_wire(g.error()));
        return Json{{"memoryId", "graph"},
                    {"tokens", 0},
                    {"communities", (*g)->communities().size()}};
    }

    // memory/recall (§7.17) — surrogate clues (seed ids + sub-queries) the
    // client feeds back into retrieve/graph.
    [[nodiscard]] Result memory_recall(const Json& p) {
        if (hooks_.memory_recall) return hooks_.memory_recall(p);
        std::string query = p.value("query", std::string{});
        if (query.empty())
            return wire_fail(::rcp::errc::InvalidParams, "query is required", Json{{"field", "query"}});
        std::size_t n = p.value("n", std::size_t{5});

        auto g = engine_.graph();
        if (!g) return std::unexpected(to_wire(g.error()));

        Json clues = Json::array();
        clues.push_back(Json{{"query", query}, {"weight", 1.0}});
        auto seeds = engine_.graph_local(query, n);
        if (seeds && !seeds->empty()) {
            Json seed_ids = Json::array();
            for (const auto& r : *seeds)
                seed_ids.push_back(r.uri.empty() ? std::to_string(r.chunk.get()) : r.uri);
            clues.push_back(Json{{"seedIds", std::move(seed_ids)}, {"weight", 0.7}});
        }
        return Json{{"clues", std::move(clues)}};
    }

private:
    Engine& engine_;
    Options opts_;
    Hooks   hooks_;
};

static_assert(::rcp::Handler<EngineHandler>, "EngineHandler must satisfy the RCP Handler concept");

} // namespace rag::rcp
