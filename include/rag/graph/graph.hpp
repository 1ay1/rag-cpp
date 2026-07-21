#pragma once
// rag/graph/graph.hpp — GraphRAG: retrieval over the document graph.
//
// A flat chunk index treats every passage as an island. Real corpora are not
// flat: documents LINK to one another and documents about the same thing SHARE
// vocabulary. This module builds that structure into an explicit graph and
// retrieves over it — the full GraphRAG recipe (Microsoft 2024):
//
//     entity/document graph → communities → community summaries →
//     graph-aware local search + community-level global search
//
// …but with the expensive LLM ingredients made DETERMINISTIC and FREE by
// default. Nodes are documents. Two edge kinds:
//
//   • link edges  — explicit references mined from the text (markdown links,
//     bare URLs, [[wikilinks]]) resolved against document URIs/titles.
//   • similarity edges — dense (cosine of mean chunk embedding) when an
//     embedder is present, else lexical (shared-term Jaccard). k-NN sparsified.
//
// Community detection is synchronous LABEL PROPAGATION (Raghavan 2007): O(E)
// per round, deterministic with a fixed tie-break, no model. Community
// SUMMARIES are extractive by default (the highest-centrality sentences of the
// community's documents) — a real, free, deterministic "report". An optional
// LLM summarizer seam upgrades them to abstractive prose.
//
// Two retrieval entry points:
//   • local  — seed with a normal hybrid retrieval, then EXPAND along graph
//     edges via Personalized PageRank, re-ranking chunks by their document's
//     restart-biased centrality. Recovers linked context a flat index misses.
//   • global — answer a corpus-level question ("how does X fit together?") by
//     ranking COMMUNITY SUMMARIES against the query and returning the lead
//     chunks (or the summary text) of the best communities.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"

namespace rag::graph {

// An undirected weighted edge between two documents (by dense index into the
// corpus's document vector, i.e. DocId::value()).
struct Edge {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    float         weight = 1.0f;
    enum class Kind : std::uint8_t { link, similarity } kind = Kind::similarity;
};

// A detected community: a set of documents plus its extractive/abstractive
// summary and a representative "lead" chunk.
struct Community {
    std::uint32_t              id = 0;
    std::vector<DocId>         docs;
    std::string                summary;      // report text (extractive default)
    ChunkId                    lead = ChunkId::invalid();  // most central chunk
    float                      density = 0.0f;             // intra/inter edge ratio
};

struct GraphConfig {
    // Similarity-edge sparsification: keep each node's top-`knn` neighbours…
    std::size_t knn            = 8;
    // …above this similarity floor (cosine or Jaccard).
    float       sim_threshold  = 0.15f;
    bool        use_links      = true;    // mine markdown/url/[[wiki]] links
    bool        use_similarity = true;    // add lexical/dense similarity edges
    // Label-propagation rounds cap (converges far earlier in practice).
    std::size_t lp_rounds      = 20;
    // Community summary: sentences to extract per community report.
    std::size_t summary_sentences = 3;
    // Personalized PageRank (local search graph expansion).
    float       ppr_restart    = 0.15f;   // teleport probability α
    std::size_t ppr_iters      = 30;
};

// Optional abstractive summarizer seam. Given the community's concatenated lead
// text, return a short natural-language report. Absent → extractive default.
using Summarizer = std::function<Result<std::string>(std::string_view)>;

// The built graph. Construct via DocGraph::build(corpus, cfg).
class DocGraph {
public:
    [[nodiscard]] static Result<DocGraph>
    build(const index::Corpus& corpus, GraphConfig cfg = {}, Summarizer summarize = {});

    // ── Introspection ────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t node_count() const noexcept { return adj_.size(); }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges_.size(); }
    [[nodiscard]] const std::vector<Edge>&      edges()       const noexcept { return edges_; }
    [[nodiscard]] const std::vector<Community>& communities() const noexcept { return communities_; }
    // Neighbours of a document (by DocId value) with edge weights.
    [[nodiscard]] std::span<const std::pair<std::uint32_t, float>>
    neighbours(std::uint32_t doc) const;

    // ── GraphRAG retrieval ────────────────────────────────────────────────────

    // LOCAL search: hybrid-seed → Personalized PageRank expansion over the doc
    // graph → re-score chunks by their document's PPR mass. `seed_k` controls
    // the initial hybrid pool; returns up to `k` chunk Hits, best-first.
    [[nodiscard]] Result<std::vector<Hit>>
    local_search(const index::Corpus& corpus, std::string_view query,
                 std::size_t k, std::size_t seed_k = 30) const;

    // GLOBAL search: rank COMMUNITIES by their summary's lexical/dense match to
    // the query; return the lead chunk of each of the top communities. This is
    // GraphRAG's "map-reduce over community reports" reduced to its retrieval
    // core. Returns up to `k` community-lead Hits.
    [[nodiscard]] Result<std::vector<Hit>>
    global_search(const index::Corpus& corpus, std::string_view query,
                  std::size_t k) const;

    // Community summary text for a community id (empty if out of range).
    [[nodiscard]] std::string_view community_summary(std::uint32_t id) const;

    // Personalized PageRank vector given a restart distribution over documents
    // (doc-value → mass). Exposed for callers wanting raw centrality.
    [[nodiscard]] std::vector<float>
    personalized_pagerank(const std::vector<std::pair<std::uint32_t, float>>& restart) const;

private:
    GraphConfig                                                   cfg_{};
    std::vector<Edge>                                             edges_;
    std::vector<std::vector<std::pair<std::uint32_t, float>>>     adj_;   // node → (nbr,w)
    std::vector<Community>                                        communities_;
    std::vector<std::uint32_t>                                    doc_community_;  // node → community id
};

} // namespace rag::graph
