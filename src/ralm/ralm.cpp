// rag/ralm/ralm.cpp — Retrieval-Augmented Language Modeling assembly.

#include "rag/ralm/ralm.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include "rag/text/tokenizer.hpp"

namespace rag::ralm {
namespace {

std::string resolve_text(const index::Corpus& corpus, ChunkId id) {
    const Chunk* ch = corpus.chunk(id);
    if (!ch) return {};
    return ch->indexed_text();
}

// Approximate whitespace-token split for stride windows (RETRO/RALM operate on
// token strides; we don't need the model's exact tokenizer for the retrieval
// frontend — whitespace strides are a faithful, deterministic proxy).
std::vector<std::string_view> ws_tokens(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        std::size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

std::string join(std::span<const std::string_view> toks, std::size_t from, std::size_t to) {
    std::string out;
    for (std::size_t i = from; i < to && i < toks.size(); ++i) {
        if (!out.empty()) out += ' ';
        out.append(toks[i].data(), toks[i].size());
    }
    return out;
}

} // namespace

// ── Ensemble weighting (RAG / REPLUG) ───────────────────────────────────────
std::vector<WeightedDoc>
ensemble_weights(const index::Corpus& corpus, std::span<const Hit> hits,
                 float temperature) {
    std::vector<WeightedDoc> out;
    out.reserve(hits.size());
    if (hits.empty()) return out;
    float t = std::max(temperature, 1e-4f);

    // Numerically-stable softmax over scores / t.
    float mx = -std::numeric_limits<float>::infinity();
    for (const auto& h : hits) mx = std::max(mx, h.score.get());
    float denom = 0.0f;
    std::vector<float> exps;
    exps.reserve(hits.size());
    for (const auto& h : hits) {
        float e = std::exp((h.score.get() - mx) / t);
        exps.push_back(e);
        denom += e;
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (std::size_t i = 0; i < hits.size(); ++i) {
        WeightedDoc w;
        w.hit = hits[i];
        w.weight = exps[i] / denom;
        w.text = resolve_text(corpus, hits[i].chunk);
        out.push_back(std::move(w));
    }
    return out;
}

std::vector<float>
replug_combine(std::span<const WeightedDoc> docs,
               std::span<const std::vector<float>> per_doc_logits) {
    std::vector<float> out;
    if (docs.empty() || per_doc_logits.empty()) return out;
    std::size_t vocab = 0;
    for (const auto& row : per_doc_logits) vocab = std::max(vocab, row.size());
    out.assign(vocab, 0.0f);
    std::size_t n = std::min(docs.size(), per_doc_logits.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& row = per_doc_logits[i];
        float w = docs[i].weight;
        for (std::size_t v = 0; v < row.size(); ++v) out[v] += w * row[v];
    }
    return out;
}

Result<std::vector<float>>
replug_step(const index::Corpus& corpus, std::string_view query, std::size_t k,
            const LmScorer& score, float temperature) {
    std::vector<Hit> hits;
    if (corpus.has_embedder()) {
        auto d = corpus.dense_search(query, k);
        if (d) hits = std::move(*d);
    }
    if (hits.empty()) hits = corpus.lexical_search(query, k);
    auto weighted = ensemble_weights(corpus, hits, temperature);
    if (weighted.empty())
        return std::unexpected(Error{Errc::empty_corpus, "replug: no documents retrieved"});

    std::vector<std::vector<float>> rows;
    rows.reserve(weighted.size());
    for (const auto& wd : weighted) {
        auto r = score(query, wd.text);
        if (!r) return std::unexpected(r.error());
        rows.push_back(std::move(*r));
    }
    return replug_combine(weighted, rows);
}

// ── RETRO chunked-neighbour retrieval ───────────────────────────────────────
Result<std::vector<RetroRow>>
retro_retrieve(const index::Corpus& corpus, std::string_view input,
               RetroConfig cfg) {
    std::vector<RetroRow> rows;
    auto toks = ws_tokens(input);
    if (toks.empty()) return rows;
    std::size_t stride = std::max<std::size_t>(cfg.stride, 1);

    for (std::size_t s = 0; s < toks.size(); s += stride) {
        RetroRow row;
        row.stride_text = join(toks, s, s + stride);
        std::vector<Hit> hits;
        if (corpus.has_embedder()) {
            auto d = corpus.dense_search(row.stride_text, cfg.neighbours);
            if (d) hits = std::move(*d);
        }
        if (hits.empty()) hits = corpus.lexical_search(row.stride_text, cfg.neighbours);

        for (const auto& h : hits) {
            RetroNeighbour nb;
            nb.neighbour = h.chunk;
            nb.score = h.score;
            const Chunk* ch = corpus.chunk(h.chunk);
            if (ch) nb.neighbour_text = ch->indexed_text();
            // continuation = next chunk id in the SAME document, if any.
            if (cfg.with_continuation && ch) {
                ChunkId cand{h.chunk.get() + 1};
                const Chunk* nxt = corpus.chunk(cand);
                if (nxt && nxt->doc.get() == ch->doc.get()) {
                    nb.continuation = cand;
                    nb.continuation_text = nxt->indexed_text();
                }
            }
            row.neighbours.push_back(std::move(nb));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

// ── In-Context RALM stride plan ─────────────────────────────────────────────
Result<std::vector<RalmDecision>>
incontext_plan(const index::Corpus& corpus, std::size_t n_tokens,
               const std::function<std::string(std::size_t)>& context_at,
               RalmConfig cfg, RerankPick pick) {
    std::vector<RalmDecision> plan;
    if (n_tokens == 0) return plan;
    std::size_t stride = std::max<std::size_t>(cfg.stride, 1);

    for (std::size_t pos = 0; pos < n_tokens; pos += stride) {
        std::string ctx = context_at ? context_at(pos) : std::string{};
        // query = trailing `query_len` whitespace tokens of the context.
        auto toks = ws_tokens(ctx);
        std::size_t from = toks.size() > cfg.query_len ? toks.size() - cfg.query_len : 0;
        std::string query = join(toks, from, toks.size());
        if (query.empty()) continue;

        std::vector<Hit> hits;
        if (corpus.has_embedder()) {
            auto d = corpus.dense_search(query, cfg.retrieve_k);
            if (d) hits = std::move(*d);
        }
        if (hits.empty()) hits = corpus.lexical_search(query, cfg.retrieve_k);
        if (hits.empty()) continue;

        std::size_t choice = 0;
        if (pick) {
            std::vector<std::string> passages;
            passages.reserve(hits.size());
            for (const auto& h : hits) passages.push_back(resolve_text(corpus, h.chunk));
            choice = pick(query, passages);
            if (choice >= hits.size()) choice = 0;
        }
        RalmDecision dec;
        dec.position = pos;
        dec.chosen = hits[choice];
        dec.text = resolve_text(corpus, hits[choice].chunk);
        plan.push_back(std::move(dec));
    }
    return plan;
}

// ── Prompt assembly ─────────────────────────────────────────────────────────
std::string
assemble_prompt(std::string_view query, std::span<const WeightedDoc> docs,
                std::string_view instruction) {
    std::ostringstream os;
    if (!instruction.empty()) os << instruction << "\n\n";
    if (!docs.empty()) {
        os << "Context:\n";
        std::size_t i = 1;
        for (const auto& d : docs) {
            os << "[" << i++ << "] " << d.text << "\n";
        }
        os << "\n";
    }
    os << "Question: " << query << "\n";
    os << "Answer:";
    return os.str();
}

} // namespace rag::ralm
