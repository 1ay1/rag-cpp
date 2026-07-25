#pragma once
// rag/index/corpus.hpp — the concrete store: documents, chunks, BM25 + HNSW,
// hybrid search, incremental update, and persistence.
//
// The Corpus owns the ground truth (chunks + their embeddings) and both
// indexes built over it. It is the substrate the high-level Engine and the
// pipeline stages operate on. Embedding is optional and lazy: without an
// Embedder the Corpus is a first-class BM25 lexical store; with one it becomes
// a hybrid store and (past a size threshold) builds an HNSW graph.

#include <memory>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/index/hnsw.hpp"
#include "rag/lexical/bm25.hpp"
#include "rag/text/chunker.hpp"
#include "rag/text/semantic_chunker.hpp"

namespace rag::index {

struct CorpusConfig {
    text::ChunkOptions     chunk;
    lexical::Bm25Params    bm25;
    text::TokenizeOptions  tokenize;
    HnswConfig             hnsw;
    // Build HNSW once the chunk count crosses this; below it, brute-force
    // cosine is faster and exact.
    std::size_t            hnsw_threshold = 2000;
    std::size_t            embed_batch    = 32;

    // How documents are split. `fixed` is the structural chunker (headings +
    // token windows): fast, deterministic, and the right default. `semantic`
    // places boundaries where the topic actually drifts, which produces more
    // self-contained chunks on prose at the cost of a similarity pass over
    // adjacent sentences.
    //
    // `semantic` uses the embedder when one is attached and falls back to
    // lexical (Jaccard) drift otherwise, so it never becomes a hard dependency
    // on an embedding backend.
    enum class Chunking { fixed, semantic };
    Chunking               chunking = Chunking::fixed;
    text::SemanticChunkOptions semantic{};
};

// Predicate over a chunk's document metadata for filtered retrieval.
using MetaFilter = std::function<bool(const Metadata&)>;

class Corpus {
public:
    Corpus() = default;
    explicit Corpus(CorpusConfig cfg) : cfg_(std::move(cfg)) {}

    // Chunks hold a borrowed pointer into `docs_` metadata; any move must
    // re-link those pointers to the NEW storage, and copying is disabled to
    // avoid silently sharing dangling pointers. (Move is cheap: vector steals.)
    Corpus(const Corpus&) = delete;
    Corpus& operator=(const Corpus&) = delete;
    Corpus(Corpus&& o) noexcept { move_from(std::move(o)); }
    Corpus& operator=(Corpus&& o) noexcept { if (this != &o) move_from(std::move(o)); return *this; }

    // Attach a dense embedder (enables the dense half). Optional.
    void set_embedder(dense::AnyEmbedder e) { embedder_ = std::move(e); }
    [[nodiscard]] bool has_embedder() const noexcept { return embedder_.has_value(); }
    // Read-only view of the attached embedder (its dimension/identity/embed),
    // or nullptr when none is set. Lets front-ends (e.g. the RCP server) both
    // advertise the dense capability and answer raw `embed` requests without
    // reaching into corpus internals.
    [[nodiscard]] const dense::AnyEmbedder* embedder() const noexcept {
        return embedder_ ? &*embedder_ : nullptr;
    }

    // Embed arbitrary text with the attached embedder (unit-normalized, same as
    // indexed chunks). Fails with Errc::unavailable if no embedder is set. Used
    // by RAPTOR / HyDE / late-interaction which embed synthetic text.
    [[nodiscard]] Result<Vector> embed_text(const std::string& text) const;

    // Ingest a document: chunk it, assign ids, index lexically, and (if an
    // embedder is set) embed its chunks. Returns the assigned DocId.
    Result<DocId> add_document(std::string uri, std::string text, Metadata meta = {}, std::string title = {});

    // Rebuild dense structures (embed any un-embedded chunks, (re)build HNSW if
    // over threshold). Idempotent; safe to call after a batch of add_document.
    Result<void> build();

    // Soft-delete a document: tombstone its chunks so they never appear in
    // lexical/dense results. Chunk/doc ids stay stable (no renumbering, so the
    // meta-pointer invariant holds). Returns not_found if the id is unknown or
    // already deleted. Call compact() to physically reclaim space.
    Result<void> remove_document(DocId id);
    [[nodiscard]] bool is_deleted(DocId id) const noexcept;
    [[nodiscard]] std::size_t live_document_count() const noexcept;

    // ── Retrieval primitives ────────────────────────────────────────────────
    [[nodiscard]] std::vector<Hit> lexical_search(std::string_view query, std::size_t k) const;
    [[nodiscard]] Result<std::vector<Hit>> dense_search(std::string_view query, std::size_t k) const;

    // Dense search with a metadata pre-filter pushed into the ANN walk (or the
    // brute-force scan when HNSW isn't built). Beats post-filtering under
    // selective predicates — it won't return fewer than k just because the
    // top-k were filtered out.
    [[nodiscard]] Result<std::vector<Hit>>
    dense_search(std::string_view query, std::size_t k, const MetaFilter& filter) const;

    // ── Resolution / access ─────────────────────────────────────────────────
    [[nodiscard]] const Chunk*    chunk(ChunkId id) const;
    [[nodiscard]] const Document* document(DocId id) const;
    // Look up a live (non-tombstoned) document by its stable external uri, or
    // nullopt if none. Enables upsert semantics (RCP index/add §7.10 mandates an
    // explicit id is an upsert, not a duplicate).
    [[nodiscard]] std::optional<DocId> find_by_uri(std::string_view uri) const;
    [[nodiscard]] SearchResult    resolve(const Hit& h) const;
    [[nodiscard]] std::size_t     chunk_count()    const noexcept { return chunks_.size(); }
    [[nodiscard]] std::size_t     document_count() const noexcept { return docs_.size(); }
    [[nodiscard]] const std::vector<Chunk>& chunks() const { ensure_linked(); return chunks_; }
    [[nodiscard]] const text::Tokenizer& tokenizer() const noexcept { return bm25_.tokenizer(); }

    // Distinct-query-term coverage for a candidate set, answered from the
    // inverted index rather than by re-tokenizing each candidate's text.
    // See Bm25Index::term_coverage.
    void term_coverage(const std::vector<std::string>& q_terms,
                       std::span<const std::uint32_t> chunk_ids,
                       std::vector<std::uint32_t>& out) const {
        bm25_.term_coverage(q_terms, chunk_ids, out);
    }

    // Apply a metadata filter, returning the ids that pass (for pre/post-filter).
    [[nodiscard]] bool passes(ChunkId id, const MetaFilter& f) const;

    // ── Persistence ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<void> save(const std::string& path) const;
    [[nodiscard]] static Result<Corpus> load(const std::string& path);

private:
    CorpusConfig                      cfg_{};
    std::optional<dense::AnyEmbedder> embedder_;
    std::vector<Document>             docs_;
    std::vector<Chunk>                chunks_;
    lexical::Bm25Index                bm25_{cfg_.bm25, cfg_.tokenize};
    std::optional<HnswIndex>          hnsw_;
    bool                              dirty_ = false;
    // A chunk borrows a pointer to its document's metadata, and add_document()
    // may reallocate `docs_`. Rather than relink every chunk on every add (O(n)
    // per document — quadratic ingest), we mark the pointers stale and repair
    // them lazily, before any accessor can hand one out. Mutable + const
    // ensure_linked() because this is memoization, not observable state.
    mutable bool                      meta_stale_ = false;
    std::unordered_set<std::uint32_t> deleted_docs_;   // tombstoned DocId values

    [[nodiscard]] Result<void> embed_pending();

    // Repair borrowed meta pointers if a preceding add_document() invalidated
    // them. Called by every read path that can expose a Chunk.
    void ensure_linked() const;

    // Re-point every chunk's borrowed `meta` at the current `docs_` storage.
    void relink_meta();
    // Steal all state from `o` and relink meta pointers to our storage.
    void move_from(Corpus&& o);
};

} // namespace rag::index
