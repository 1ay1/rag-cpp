#pragma once
// rag/sparse/splade.hpp — learned-sparse retrieval (SPLADE-style).
//
// Dense retrieval loses exact-term signal; BM25 loses semantics. SPLADE
// (Formal et al. 2021) bridges them: a model projects text onto a SPARSE vector
// over the vocabulary with LEARNED per-term weights AND term EXPANSION (a query
// activates related vocabulary terms it never literally contained). Retrieval is
// then a sparse dot product over an inverted index — as fast as BM25, but with
// dense-like recall.
//
// Shipping a trained SPLADE model would need a transformer at index time. We do
// the next best dependency-free thing: a MODEL-FREE approximation that captures
// SPLADE's two ideas —
//
//   1. Saturated term weighting: w(t) = log(1 + tf) · idf   (ReLU-log-like, the
//      shape SPLADE's log-saturation activation produces), giving impactful,
//      non-linear weights instead of raw counts.
//   2. Term EXPANSION via a co-occurrence graph mined from the corpus: a term's
//      top co-occurring neighbours are added to the sparse vector at a damped
//      weight, so a query term activates semantically-adjacent vocabulary — the
//      essence of SPLADE expansion, learned from corpus statistics.
//
// A real SPLADE model can be dropped in by supplying its (term→weight) maps
// through `SparseEncoder` — the index and scorer are model-agnostic.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/index/corpus.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::sparse {

// A sparse vector over the term vocabulary: term-id → weight (impact).
using SparseVec = std::unordered_map<std::uint32_t, float>;

struct SpladeConfig {
    std::size_t expansion_terms = 3;      // neighbours added per source term
    float       expansion_decay = 0.35f;  // damping on expanded-term weight
    float       min_weight      = 1e-3f;  // prune tiny impacts
    std::size_t max_cooc_vocab  = 50000;  // cap co-occurrence graph size
};

// The learned-sparse index: builds a term vocabulary, a co-occurrence graph for
// expansion, per-document sparse vectors, and an inverted index for fast dot
// scoring. Serializable-ready (same style as Bm25Index).
class SpladeIndex {
public:
    [[nodiscard]] static Result<SpladeIndex>
    build(const index::Corpus& corpus, SpladeConfig cfg = {});

    // Encode arbitrary text into a sparse impact vector (with expansion).
    [[nodiscard]] SparseVec encode(std::string_view text, bool expand) const;

    // Retrieve top-k chunks by sparse dot product against the encoded query.
    [[nodiscard]] std::vector<Hit> search(std::string_view query, std::size_t k) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return term_id_.size(); }
    [[nodiscard]] std::size_t doc_count()  const noexcept { return doc_vecs_.size(); }

    // Persist the trained index (vocab, idf, expansion graph, doc vectors) to a
    // versioned blob (magic "SPL1"); reopening never rebuilds. Mirrors the
    // Bm25Index / HnswIndex serialization contract.
    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static Result<SpladeIndex> deserialize(std::string_view blob);

private:
    SpladeConfig                                          cfg_{};
    text::Tokenizer                                       tok_;
    std::unordered_map<std::string, std::uint32_t>        term_id_;
    std::vector<float>                                    idf_;          // per term
    // Expansion graph: term → [(neighbour term, weight)] top-N co-occurrences.
    std::vector<std::vector<std::pair<std::uint32_t, float>>> expand_;
    // Per-chunk sparse vectors + the ChunkId they resolve to.
    std::vector<SparseVec>                                doc_vecs_;
    std::vector<ChunkId>                                  doc_ids_;
    // Inverted index: term → [(doc index, weight)].
    std::vector<std::vector<std::pair<std::uint32_t, float>>> inverted_;

    std::uint32_t intern(std::string_view term);
    [[nodiscard]] const std::uint32_t* lookup(std::string_view term) const;
};

} // namespace rag::sparse
