#pragma once
// rag/eval/beir.hpp — a BEIR-format evaluation harness (nDCG, Recall, MRR, MAP).
//
// Claiming "SOTA retrieval" is worthless without measurement. BEIR (Thakur et
// al. 2021) is the standard zero-shot IR benchmark; each dataset (NFCorpus,
// SciFact, FiQA, …) ships three files:
//
//   corpus.jsonl   {"_id","title","text"}          — the documents
//   queries.jsonl  {"_id","text"}                   — the queries
//   qrels/*.tsv    query-id  corpus-id  relevance   — the graded judgments
//
// This module loads that format, runs any Retriever over it, and computes the
// standard trec_eval metrics so you can put a NUMBER on the engine and diff it
// across changes. It is the harness the README's SOTA claims are backed by.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"

namespace rag::eval {

// A BEIR dataset in memory.
struct BeirDataset {
    struct Doc   { std::string id; std::string title; std::string text; };
    struct Query { std::string id; std::string text; };
    std::vector<Doc>   corpus;
    std::vector<Query> queries;
    // qrels[query_id][doc_id] = graded relevance (0 = irrelevant).
    std::unordered_map<std::string, std::unordered_map<std::string, int>> qrels;

    // Load from a BEIR dataset directory (corpus.jsonl, queries.jsonl,
    // qrels/test.tsv). `qrels_split` selects the tsv under qrels/.
    [[nodiscard]] static Result<BeirDataset>
    load(const std::string& dir, const std::string& qrels_split = "test");
};

// Per-metric results, averaged over evaluated queries.
struct Metrics {
    std::size_t queries = 0;
    std::map<std::size_t, double> ndcg;      // k → nDCG@k
    std::map<std::size_t, double> recall;    // k → Recall@k
    std::map<std::size_t, double> precision; // k → Precision@k
    double mrr = 0.0;                         // mean reciprocal rank
    double map = 0.0;                         // mean average precision

    [[nodiscard]] std::string report() const;   // pretty multi-line summary
};

// A ranked run for one query: doc ids best-first.
using Ranking = std::vector<std::string>;

// The retrieval function under test: (query text, top-k) → ranked doc ids.
using RetrieveFn = std::function<Ranking(const std::string& query, std::size_t k)>;

struct EvalConfig {
    std::vector<std::size_t> cutoffs = {1, 3, 5, 10, 100};  // k values to report
    std::size_t depth = 100;   // retrieval depth per query
};

// Evaluate `retrieve` over the dataset's queries against its qrels.
[[nodiscard]] Metrics
evaluate(const BeirDataset& ds, const RetrieveFn& retrieve, EvalConfig cfg = {});

// Convenience: index the dataset into a fresh Corpus (BM25 or hybrid if you set
// an embedder on the returned corpus BEFORE calling — here we build lexical),
// then evaluate the corpus's own hybrid/lexical search. Returns (corpus, metrics).
[[nodiscard]] Result<Metrics>
evaluate_corpus(const BeirDataset& ds, index::Corpus& corpus, EvalConfig cfg = {});

// ── Metric primitives (exposed for unit tests / custom harnesses) ────────────
[[nodiscard]] double ndcg_at_k(const Ranking& run,
                               const std::unordered_map<std::string, int>& rel, std::size_t k);
[[nodiscard]] double recall_at_k(const Ranking& run,
                                 const std::unordered_map<std::string, int>& rel, std::size_t k);
[[nodiscard]] double average_precision(const Ranking& run,
                                       const std::unordered_map<std::string, int>& rel);
[[nodiscard]] double reciprocal_rank(const Ranking& run,
                                     const std::unordered_map<std::string, int>& rel);

} // namespace rag::eval
