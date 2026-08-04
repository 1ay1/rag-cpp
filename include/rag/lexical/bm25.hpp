#pragma once
// rag/lexical/bm25.hpp — Okapi BM25 over an inverted index.
//
// The lexical half of hybrid retrieval: exact-term/proper-noun matching that
// dense embeddings miss. Pure C++/STL, deterministic, no dependencies.
//
//   score(q, d) = Σ_{t∈q} idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1-b + b·|d|/avgdl))
//
// with idf(t) = ln(1 + (N - n_t + 0.5)/(n_t + 0.5))  (BM25+ smoothed idf,
// always positive so common terms never contribute negatively).

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/text/tokenizer.hpp"

namespace rag::lexical {

struct Bm25Params {
    float k1 = 1.2f;
    float b  = 0.75f;
};

// One posting: which document, and the term frequency there.
struct Posting {
    std::uint32_t doc;   // dense internal doc ordinal (== ChunkId in the corpus)
    std::uint32_t tf;
};

class Bm25Index {
public:
    Bm25Index() = default;
    explicit Bm25Index(Bm25Params p, text::TokenizeOptions topts = {})
        : params_(p), tok_(topts) {}

    // Add a document identified by ordinal `id` (must be unique, monotonically
    // assigned by the caller). Returns the number of indexed terms.
    std::size_t add(std::uint32_t id, std::string_view text);

    // Finalize idf/avgdl after all adds. Call once before querying; cheap to
    // recall after incremental adds.
    void finalize();

    // True once finalize() has run and no add() has invalidated it since.
    // Lets a caller (Corpus) skip the work when it is already up to date, and
    // do it lazily on the read path when it is not.
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }

    // Top-k by BM25. Returns hits sorted by descending score.
    [[nodiscard]] std::vector<Hit> search(std::string_view query, std::size_t k) const;

    // Score a single already-tokenized query against one doc (used by fusion
    // to rescore a candidate set consistently).
    [[nodiscard]] float score_doc(const std::vector<std::string>& q_terms,
                                  std::uint32_t doc_id) const;

    // Per-term breakdown of score_doc(): which query terms actually matched
    // this document, and what each contributed to its BM25 score.
    //
    // This is the evidence behind a lexical hit. A retrieval system that can
    // only say "score 7.3" is unauditable; one that can say "7.3, of which 5.1
    // came from 'pneumonia' (idf 4.2, tf 3)" can be debugged, explained to a
    // user, and tested. The terms are returned in DESCENDING contribution
    // order, so the head of the vector is the reason the document ranked.
    struct TermScore {
        std::string   term;
        float         contribution;  // this term's addend in the BM25 sum
        float         idf;           // its inverse document frequency
        std::uint32_t tf;            // its frequency in this document
    };
    [[nodiscard]] std::vector<TermScore>
    explain_doc(const std::vector<std::string>& q_terms, std::uint32_t doc_id) const;

    // For each doc in `docs`, how many of the DISTINCT terms in `q_terms`
    // occur in it, written to `out` (same order, same size).
    //
    // This is the lexical-coverage feature the reranker needs, and computing
    // it from the inverted index is asymptotically better than the obvious
    // per-candidate approach: re-tokenizing every candidate's full text costs
    // O(candidates × doc_length) with a string allocation per token, whereas
    // walking the postings of the (few) query terms costs O(query_terms ×
    // log postings) with no tokenization and no allocation at all. The index
    // already knows which documents contain which terms — asking it is both
    // faster and, since it is the same analysis chain that produced the
    // postings, strictly more consistent than re-deriving the answer.
    void term_coverage(const std::vector<std::string>& q_terms,
                       std::span<const std::uint32_t> docs,
                       std::vector<std::uint32_t>& out) const;

    [[nodiscard]] std::size_t size()      const noexcept { return doc_len_.size(); }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return postings_.size(); }
    [[nodiscard]] const text::Tokenizer& tokenizer() const noexcept { return tok_; }

    // Serialization to/from a binary blob (little-endian, versioned).
    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static Result<Bm25Index> deserialize(std::string_view blob);

private:
    Bm25Params params_{};
    text::Tokenizer tok_{};

    // term -> postings list (sorted by doc id).
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    // doc ordinal -> token count.
    std::unordered_map<std::uint32_t, std::uint32_t> doc_len_;

    // Dense mirror of doc_len_, built by finalize(). Doc ordinals are dense and
    // contiguous in a corpus, so scoring can index an array instead of hashing
    // once per POSTING — the inner loop of every lexical query. dense_len_ is
    // empty until finalize(); scoring falls back to the map if so.
    std::vector<std::uint32_t> dense_len_;
    std::uint32_t              max_doc_ = 0;

    // ── Precomputed posting weights ─────────────────────────────────
    // A posting's BM25 contribution is
    //     idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
    // and every factor except idf(t) depends only on (tf, dl), both fixed at
    // index time. Recomputing it per query meant a float DIVISION for each of
    // the ~80k postings a query touches on a large corpus — work whose answer
    // never changes between queries.
    //
    // finalize() therefore evaluates the tf/dl half once into a flat array laid
    // out to mirror `postings_` term-by-term, leaving the query's inner loop a
    // single multiply-add over a sequential float stream. Kept separate from
    // the Posting struct (rather than widening it) so the SERIALIZED format is
    // untouched: this is derived data, rebuilt by finalize() on load.
    std::vector<float> pw_;
    // term -> [begin,end) range into pw_, aligned with that term's postings.
    std::unordered_map<std::string, std::pair<std::uint32_t, std::uint32_t>> pw_span_;

    // ── Block-max metadata for BlockMax-WAND dynamic pruning ────────────
    // The TAAT fast path scores EVERY posting of EVERY query term, then keeps
    // top-k. That is O(total postings touched) regardless of k, so a common
    // word in a large corpus is paid in full even when only 10 results are
    // wanted. BlockMax-WAND (Ding & Suel, SIGIR 2011) instead walks postings
    // document-at-a-time and, using a per-block upper bound on each term's
    // contribution, skips whole blocks that provably cannot enter the current
    // top-k. Same results, a fraction of the work once k << corpus.
    //
    // We chunk each term's postings into fixed-size blocks and store, per block,
    // the last doc id in the block (for skip-to) and the maximum PRECOMPUTED
    // weight in it. A term's block-max CONTRIBUTION is idf(term)*block_max_pw;
    // the global per-term max (used for the WAND pivot bound) is the max over
    // its blocks. All derived from pw_, rebuilt by finalize(), never serialized.
    static constexpr std::uint32_t kBlockSize = 128;
    struct BlockMeta {
        std::uint32_t last_doc;   // largest doc id in this block
        float         max_pw;     // largest precomputed weight in this block
    };
    // term -> its blocks (aligned with the term's postings slice).
    std::unordered_map<std::string, std::vector<BlockMeta>> block_meta_;

    double  total_len_ = 0.0;
    float   avgdl_     = 0.0f;
    bool    finalized_ = false;

    [[nodiscard]] float idf(std::size_t n_t) const;

    // BlockMax-WAND top-k over dense ordinals + precomputed weights. Returns
    // std::nullopt when its preconditions do not hold (sparse ids, no weights,
    // degenerate k) so search() can fall back to the exhaustive path.
    [[nodiscard]] std::optional<std::vector<Hit>>
    search_wand(const std::vector<std::string>& q_terms, std::size_t k) const;
};

} // namespace rag::lexical
