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
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/index/hnsw.hpp"
#include "rag/lexical/bm25.hpp"
#include "rag/text/chunker.hpp"

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
};

// Predicate over a chunk's document metadata for filtered retrieval.
using MetaFilter = std::function<bool(const Metadata&)>;

class Corpus {
public:
    Corpus() = default;
    explicit Corpus(CorpusConfig cfg) : cfg_(std::move(cfg)) {}

    // Attach a dense embedder (enables the dense half). Optional.
    void set_embedder(dense::AnyEmbedder e) { embedder_ = std::move(e); }
    [[nodiscard]] bool has_embedder() const noexcept { return embedder_.has_value(); }

    // Ingest a document: chunk it, assign ids, index lexically, and (if an
    // embedder is set) embed its chunks. Returns the assigned DocId.
    Result<DocId> add_document(std::string uri, std::string text, Metadata meta = {}, std::string title = {});

    // Rebuild dense structures (embed any un-embedded chunks, (re)build HNSW if
    // over threshold). Idempotent; safe to call after a batch of add_document.
    Result<void> build();

    // ── Retrieval primitives ────────────────────────────────────────────────
    [[nodiscard]] std::vector<Hit> lexical_search(std::string_view query, std::size_t k) const;
    [[nodiscard]] Result<std::vector<Hit>> dense_search(std::string_view query, std::size_t k) const;

    // ── Resolution / access ─────────────────────────────────────────────────
    [[nodiscard]] const Chunk*    chunk(ChunkId id) const;
    [[nodiscard]] const Document* document(DocId id) const;
    [[nodiscard]] SearchResult    resolve(const Hit& h) const;
    [[nodiscard]] std::size_t     chunk_count()    const noexcept { return chunks_.size(); }
    [[nodiscard]] std::size_t     document_count() const noexcept { return docs_.size(); }
    [[nodiscard]] const std::vector<Chunk>& chunks() const noexcept { return chunks_; }
    [[nodiscard]] const text::Tokenizer& tokenizer() const noexcept { return bm25_.tokenizer(); }

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

    [[nodiscard]] Result<void> embed_pending();
};

} // namespace rag::index
