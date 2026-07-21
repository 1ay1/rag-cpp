#pragma once
// rag/raptor/raptor.hpp — RAPTOR: Recursive Abstractive Processing for
// Tree-Organized Retrieval (Sarthi et al., ICLR 2024).
//
// Flat chunk retrieval returns short contiguous windows and misses the "big
// picture" a question may need. RAPTOR builds, bottom-up, a TREE of increasingly
// abstract summaries:
//
//   level 0 : the leaf chunks (the corpus as-is)
//   level 1 : cluster the leaves, SUMMARIZE each cluster → new nodes
//   level 2 : cluster level-1 summaries, summarize again …            (recurse)
//
// At query time RAPTOR uses the "collapsed tree": ALL nodes across ALL levels
// are pooled into one flat set and retrieved together, so a query can match a
// fine leaf OR a high-level synopsis, whichever answers it. Controlled
// experiments show large gains on multi-hop / holistic questions.
//
// Clustering here is embedding-based agglomerative (cosine) when an embedder is
// present, else lexical-Jaccard agglomerative — same graceful-degradation
// contract as the rest of the library. Summaries are EXTRACTIVE by default
// (centroid-nearest + highest-coverage sentences, free, no model) with an
// optional abstractive `Summarizer` seam that upgrades them to real prose.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"

namespace rag::raptor {

// A node in the RAPTOR tree. Leaves reference a real corpus ChunkId; internal
// nodes carry a synthesized summary + their own embedding + child links.
struct Node {
    std::uint32_t          id      = 0;
    std::uint32_t          level   = 0;                 // 0 = leaf
    ChunkId                chunk   = ChunkId::invalid(); // valid only at leaves
    std::string            text;                         // leaf body or summary
    Vector                 embedding;                    // for retrieval
    std::vector<std::uint32_t> children;                 // node ids (internal)
};

// Abstractive summarizer seam: given the concatenated child text, return a short
// summary. Absent → extractive default.
using Summarizer = std::function<Result<std::string>(std::string_view)>;

struct RaptorConfig {
    std::size_t max_levels        = 4;     // stop after this many levels
    std::size_t cluster_size      = 5;     // target leaves per cluster
    std::size_t min_cluster       = 2;     // don't summarize singletons
    std::size_t summary_sentences = 3;     // extractive report length
    float       sim_threshold     = 0.10f; // agglomeration floor
};

// The built tree. Construct via RaptorTree::build(corpus, cfg).
class RaptorTree {
public:
    [[nodiscard]] static Result<RaptorTree>
    build(const index::Corpus& corpus, RaptorConfig cfg = {}, Summarizer summarize = {});

    [[nodiscard]] std::size_t node_count()  const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t level_count() const noexcept { return levels_; }
    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }

    // Collapsed-tree retrieval: score EVERY node (all levels) against the query
    // and return the top-k. Leaf hits resolve to their chunk; summary hits carry
    // an invalid ChunkId and their synopsis text (see `resolve`).
    struct Retrieved {
        std::uint32_t node;
        ChunkId       chunk;    // invalid ⇒ a summary node
        Score         score;
        std::uint32_t level;
    };
    [[nodiscard]] Result<std::vector<Retrieved>>
    retrieve(const index::Corpus& corpus, std::string_view query, std::size_t k) const;

    // Text of a node (leaf body or summary).
    [[nodiscard]] std::string_view node_text(std::uint32_t id) const;

private:
    std::vector<Node> nodes_;
    std::size_t       levels_ = 1;
    bool              dense_  = false;
};

} // namespace rag::raptor
