#pragma once
// rag/rcp/convert.hpp — the wire↔engine data mapping (spec §7.7, §8, §4.8).
//
// Pure, side-effect-free translation between RCP JSON and rag-cpp values:
//
//   * RetrieveParams   — the parsed, validated `retrieve` request (§7.7).
//   * to_hit()         — one rag::SearchResult → one wire Hit (§7.7 shape).
//   * compile_filter() — an RCP filter tree (§8) → a rag::index::MetaFilter,
//                        validated against advertised fields with a precise
//                        -32602 on anything malformed or unauthorized.
//
// Keeping this layer free of the Engine means it is trivially unit-testable and
// the handler (handler.hpp) reads as pure orchestration.

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/rcp/error.hpp"

#include <rcp/filter.hpp>
#include <rcp/protocol.hpp>
#include <rcp/types.hpp>
#include <rcp/vectors.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rag::rcp {

using Json = ::rcp::Json;

// ─────────────────────────────────────────────────────────────────────────────
// RetrieveParams — the subset of §7.7 `retrieve` params this engine honours,
// parsed into typed fields. Unknown options are ignored (§7.7: "MUST be
// ignored-or-rejected only per the advertised capability"); options we DO
// advertise but receive malformed are rejected upstream in parse().
// ─────────────────────────────────────────────────────────────────────────────
struct RetrieveParams {
    std::string              query;
    std::size_t              k          = 10;
    std::string              mode       = "hybrid";  // dense|sparse|hybrid
    std::optional<std::size_t> candidate_k;          // recall width before rerank
    std::optional<double>    min_score;
    std::optional<std::size_t> token_budget;
    bool                     include_text    = true;
    bool                     include_vectors = false;
    bool                     rerank_requested = false;
    std::size_t              rerank_top_n     = 0;
    Json                     filter = Json(nullptr);  // raw tree; compiled later
};

// Parse + shallow-validate a `retrieve` params object. Enforces the funnel
// invariant candidateK ≥ rerank.topN ≥ k (spec §3.3) after defaulting.
[[nodiscard]] inline ::rcp::Result<RetrieveParams>
parse_retrieve(const Json& p, std::size_t max_k) {
    RetrieveParams r;

    // query: string | Content | [Content]. We index text; extract the text.
    const Json& q = p.contains("query") ? p["query"] : Json(nullptr);
    if (q.is_string()) {
        r.query = q.get<std::string>();
    } else if (q.is_object()) {
        r.query = q.value("text", std::string{});
    } else if (q.is_array()) {
        for (const auto& c : q)
            if (c.is_string()) r.query += c.get<std::string>() + " ";
            else if (c.is_object()) r.query += c.value("text", std::string{}) + " ";
    }
    if (r.query.empty())
        return wire_fail(::rcp::errc::InvalidParams, "query is required and must carry text",
                         Json{{"field", "query"}});

    if (p.contains("k") && p["k"].is_number_integer()) r.k = p["k"].get<std::size_t>();
    if (r.k == 0) return wire_fail(::rcp::errc::InvalidParams, "k must be >= 1",
                                   Json{{"field", "k"}});
    if (r.k > max_k) r.k = max_k;   // clamp to advertised maxK (§6.1)

    if (p.contains("mode") && p["mode"].is_string()) {
        r.mode = p["mode"].get<std::string>();
        if (r.mode != "dense" && r.mode != "sparse" && r.mode != "hybrid")
            return wire_fail(::rcp::errc::InvalidParams, "mode must be dense|sparse|hybrid",
                             Json{{"field", "mode"}, {"option", r.mode}});
    }
    if (p.contains("candidateK") && p["candidateK"].is_number_integer())
        r.candidate_k = p["candidateK"].get<std::size_t>();
    if (p.contains("minScore") && p["minScore"].is_number())
        r.min_score = p["minScore"].get<double>();
    if (p.contains("tokenBudget") && p["tokenBudget"].is_number_integer())
        r.token_budget = p["tokenBudget"].get<std::size_t>();
    if (p.contains("includeText") && p["includeText"].is_boolean())
        r.include_text = p["includeText"].get<bool>();
    if (p.contains("includeVectors") && p["includeVectors"].is_boolean())
        r.include_vectors = p["includeVectors"].get<bool>();

    // rerank: false | { method?, topN? }. `false` disables (§7.7).
    if (p.contains("rerank")) {
        const Json& rr = p["rerank"];
        if (rr.is_object()) {
            r.rerank_requested = true;
            r.rerank_top_n = rr.value("topN", r.k > 10 ? r.k : 10);
        } else if (rr.is_boolean() && rr.get<bool>()) {
            r.rerank_requested = true;
            r.rerank_top_n = r.k > 10 ? r.k : 10;
        }
    }

    if (p.contains("filter")) r.filter = p["filter"];

    // Funnel invariant (§3.3): candidateK ≥ rerank.topN ≥ k. Widen rather than
    // reject — a client under-specifying recall shouldn't fail, but a client
    // that INVERTS the funnel explicitly is a bug we surface.
    std::size_t top_n = r.rerank_requested ? std::max(r.rerank_top_n, r.k) : r.k;
    r.rerank_top_n = top_n;
    std::size_t want_cand = std::max<std::size_t>(top_n, r.k);
    if (!r.candidate_k) r.candidate_k = std::max<std::size_t>(want_cand, 4 * r.k);
    if (*r.candidate_k < top_n)
        return wire_fail(::rcp::errc::InvalidParams,
                         "funnel invariant violated: candidateK must be >= rerank.topN >= k",
                         Json{{"field", "candidateK"}});
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// to_hit — one resolved rag SearchResult → the wire Hit shape (§7.7).
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline Json
to_hit(const rag::SearchResult& r, const index::Corpus& corpus,
       bool include_text, bool include_vectors) {
    // Stable id: the document uri disambiguated by chunk (a doc has many chunks).
    std::string id = r.uri.empty() ? ("chunk-" + std::to_string(r.chunk.get()))
                                    : (r.uri + "#" + std::to_string(r.chunk.get()));
    Json h{{"id", id},
           {"score", static_cast<double>(r.score.get())},
           {"unit", "chunk"},
           {"modality", "text"}};

    if (include_text)
        h["text"] = r.context.empty() ? r.text : (r.context + "\n" + r.text);

    // citation (§7.7/§14): exact provenance for grounded generation + audit.
    Json citation{{"source", r.uri}};
    if (r.start_line || r.end_line) {
        citation["startLine"] = r.start_line;
        citation["endLine"]   = r.end_line;
    }
    h["citation"] = std::move(citation);

    // trust (§15.2): a curated corpus is provenance-known and, for a text
    // engine, not a prompt-injection surface. Honest defaults, not decoration.
    h["trust"] = Json{{"level", "corpus"}, {"injectionSuspected", false}};

    if (include_vectors) {
        if (const Chunk* c = corpus.chunk(r.chunk); c && !c->embedding.empty())
            h["vector"] = ::rcp::vectors::encode_f32_base64(c->embedding);
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// compile_filter — RCP filter tree (§8) → rag::index::MetaFilter predicate.
//
// rag-cpp metadata is string-valued (Metadata = map<string,string>), so numeric
// comparisons operate lexically-or-numerically as the value parses. The tree is
// validated against `fields` first: a leaf naming an unadvertised field, or a
// disallowed operator, yields -32602 with error.data.field (§8, §15.4).
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline ::rcp::Result<index::MetaFilter>
compile_filter(const Json& raw, const Json& advertised_fields) {
    std::vector<::rcp::filter::FieldSpec> specs;
    if (advertised_fields.is_object())
        for (auto it = advertised_fields.begin(); it != advertised_fields.end(); ++it)
            specs.push_back({it.key(), it->is_string() ? it->get<std::string>() : "keyword"});

    auto norm = ::rcp::filter::validate(
        raw, specs, {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "exists"});
    if (!norm) return std::unexpected(norm.error());
    Json tree = *norm;
    if (tree.is_null()) return index::MetaFilter{};   // absent filter ⇒ pass-all

    // The evaluator recurses over the normalized tree. Shared-ptr so the
    // returned std::function can hold it by value and stay self-contained.
    auto eval = std::make_shared<std::function<bool(const Json&, const Metadata&)>>();
    *eval = [eval](const Json& n, const Metadata& meta) -> bool {
        if (n.contains("and")) {
            for (const auto& c : n["and"]) if (!(*eval)(c, meta)) return false;
            return true;
        }
        if (n.contains("or")) {
            for (const auto& c : n["or"]) if ((*eval)(c, meta)) return true;
            return false;
        }
        if (n.contains("not")) return !(*eval)(n["not"], meta);

        const std::string field = n.value("field", std::string{});
        const std::string op    = n.value("op", std::string{});
        const Json&       val   = n.contains("value") ? n["value"] : Json(nullptr);
        auto it = meta.find(field);
        const bool present = it != meta.end();

        if (op == "exists") return present == (val.is_boolean() ? val.get<bool>() : true);
        if (!present) return op == "ne" || op == "nin";   // absent satisfies only negations

        const std::string& mv = it->second;
        auto as_str = [](const Json& j) {
            return j.is_string() ? j.get<std::string>() : j.dump();
        };
        auto num = [](const std::string& s, double& out) {
            try { std::size_t pos; out = std::stod(s, &pos); return pos == s.size(); }
            catch (...) { return false; }
        };
        auto cmp = [&](auto pred) -> bool {
            double a, b;
            if (num(mv, a) && val.is_number()) return pred(a, val.get<double>());
            return pred(std::string_view(mv), std::string_view(as_str(val)));
        };

        if (op == "eq")  return mv == as_str(val);
        if (op == "ne")  return mv != as_str(val);
        if (op == "gt")  return cmp([](auto a, auto b){ return a > b;  });
        if (op == "gte") return cmp([](auto a, auto b){ return a >= b; });
        if (op == "lt")  return cmp([](auto a, auto b){ return a < b;  });
        if (op == "lte") return cmp([](auto a, auto b){ return a <= b; });
        if (op == "contains") return mv.find(as_str(val)) != std::string::npos;
        if (op == "in" || op == "nin") {
            bool found = false;
            if (val.is_array())
                for (const auto& e : val)
                    if (mv == (e.is_string() ? e.get<std::string>() : e.dump())) { found = true; break; }
            return op == "in" ? found : !found;
        }
        return false;
    };

    return index::MetaFilter{
        [eval, tree](const Metadata& meta) { return (*eval)(tree, meta); }};
}

} // namespace rag::rcp
