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
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/index/hnsw.hpp"
#include "rag/lexical/bm25.hpp"
#include "rag/store/container.hpp"
#include "rag/store/wal.hpp"
#include "rag/text/chunker.hpp"
#include "rag/text/contextual.hpp"
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
    //
    // `proposition` goes further and indexes ATOMIC STATEMENTS: one chunk per
    // self-contained fact. It maximises precision — a retrieved unit carries
    // exactly one claim, so nothing irrelevant rides along with it — which is
    // what fact-checking and claim-verification workloads want.
    //
    // IT IS NOT A FREE WIN, AND ON GENERAL IR IT LOSES. MEASURED with
    // Pipeline::standard() over BEIR (nDCG@10 / chunk count):
    //
    //             SciFact                    NFCorpus
    //   fixed        0.6809  (  5183)          0.3261  ( 3633)
    //   semantic     0.6703  (  8866)          0.3233  ( 5195)
    //   proposition  0.5887  ( 62624)          0.2895  (56341)
    //
    // That is -0.092 and -0.037 nDCG@10 for a 12x larger index. The mechanism
    // is straightforward: BM25 scores a one-sentence chunk on almost no term
    // evidence, and splitting a document into 12 pieces splits its term
    // statistics with it. Enable this only if your queries are claim-shaped and
    // you have measured a win on YOUR data — the numbers above are the reason
    // it is not the default and never will be.
    //
    // The built-in splitter is deterministic and dependency-free (sentence
    // segmentation with pronoun-free heuristics). Set a propositionizer via
    // set_propositionizer() to use an LLM instead; as with the contextualizer,
    // a failing seam degrades to the deterministic splitter rather than
    // failing ingest.
    // `source` chunks CODE on definition boundaries rather than prose windows.
    // For a recognized extension it uses the hand-written per-language chunker;
    // for anything else — an in-house DSL, a vendor format, a language invented
    // last quarter — it INFERS the file's own conventions and chunks on those
    // (loaders/structure.hpp). Non-code documents are unaffected: inference
    // declines on prose and the fixed chunker runs instead, so this is safe to
    // enable on a mixed corpus of code and documentation.
    //
    // Which language a document is written in comes from its `ext` metadata
    // key, or from the extension of its URI when that key is absent, so the
    // directory loader gets this for free.
    enum class Chunking { fixed, semantic, proposition, source };
    Chunking               chunking = Chunking::fixed;
    text::SemanticChunkOptions semantic{};

    // Contextual Retrieval (Anthropic 2024). Before indexing, prepend to each
    // chunk a short blurb SITUATING it in its source document, so that a
    // fragment which says "revenue grew 3%" still carries the name of the
    // company it is about. Both the BM25 postings and the dense vector are
    // built from indexed_text(), so turning this on changes what BOTH halves
    // of the hybrid see — which is the whole point.
    //
    // Off by default because it is not free. Measured by
    // bench/contextual_bench.cpp on its own synthetic corpus (2000 documents
    // that name their subject ONCE, in the title, 18k chunks at max_lines=4,
    // this machine): ingest+build 22.7 ms -> 67.7 ms (2.98x) and indexed text
    // 1.95x, against chunk-level recall@5 0.0026 -> 0.9998.
    //
    // That recall figure is a CEILING, not a promise. It is that large only
    // because on that corpus the subject is unreachable from any chunk but the
    // title chunk, which is the paper's premise taken to its limit. On a corpus
    // whose chunks already repeat their subject there is nothing to situate,
    // and the 2.98x is pure cost. Measure before enabling this.
    //
    // (Document-level recall on the same corpus is 1.0 in BOTH arms — the
    // subject token is unique, so the document is always found via its title
    // chunk. A doc-level metric cannot see this feature at all.)
    //
    // This lives in the INGEST config rather than in a pipeline stage on
    // purpose: the paper's method rewrites what is indexed, so it has to run
    // before the indexes are built. A query-time stage cannot reproduce it.
    bool                   contextual = false;
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
    //
    // Moving is NOT thread-safe and must not race with any other access: the
    // mutexes below are not themselves moved (a locked mutex cannot be), so a
    // move while another thread holds a lock is undefined. Move a Corpus only
    // while you have exclusive ownership of it — typically right after load().
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

    // Attach an LLM to write the situating context (Anthropic's method as
    // published). Only consulted when cfg.contextual is true; with contextual
    // on and no contextualizer set, ingest uses the deterministic extractive
    // fallback, which needs no model and never fails. A contextualizer that
    // returns an error for one chunk falls back for THAT chunk rather than
    // failing the ingest — a flaky model must not be able to reject a document.
    void set_contextualizer(text::Contextualizer c) { contextualizer_ = std::move(c); }
    [[nodiscard]] bool has_contextualizer() const noexcept {
        return static_cast<bool>(contextualizer_);
    }

    // Attach an LLM to split documents into atomic propositions. Only consulted
    // when cfg.chunking is Chunking::proposition; without one, ingest uses the
    // deterministic sentence splitter, which needs no model and never fails.
    // A propositionizer that errors on a document falls back to that splitter
    // for THAT document rather than failing the ingest — same contract as the
    // contextualizer, for the same reason: a flaky model must not be able to
    // reject a document.
    void set_propositionizer(text::PropositionFn p) { propositionizer_ = std::move(p); }
    [[nodiscard]] bool has_propositionizer() const noexcept {
        return static_cast<bool>(propositionizer_);
    }

    // Ingest a document: chunk it, assign ids, index lexically, and (if an
    // embedder is set) embed its chunks. Returns the assigned DocId.
    Result<DocId> add_document(std::string uri, std::string text, Metadata meta = {}, std::string title = {});

    // Add, replacing any live document that already has this uri — as ONE
    // atomic step.
    //
    // The obvious spelling of an upsert at the call site is
    //   if (auto old = find_by_uri(u)) remove_document(*old);
    //   add_document(u, ...);
    // which takes the lock three separate times. Two threads upserting the same
    // uri can then both observe "not present" and both insert, producing the
    // duplicate that RCP §7.10 explicitly forbids. Holding the write lock across
    // the whole read-modify-write is the only way to make it a real upsert.
    Result<DocId> upsert_document(std::string uri, std::string text, Metadata meta = {},
                                  std::string title = {});

    // A single document to ingest, for the batch path below.
    struct DocInput {
        std::string uri;
        std::string text;
        Metadata    meta{};
        std::string title{};
    };

    // Bulk ingest, acquiring the write lock ONCE for the whole batch.
    //
    // The per-document add_document() takes mu_ per call and (when a WAL is
    // attached) issues one durable append per document. For bulk loads that is
    // N lock round-trips and N fsyncs where one of each suffices: this holds the
    // lock across the whole batch and, if a WAL is open, wraps the appends in a
    // single group-commit so the batch costs ONE sync, not N. Chunking/indexing
    // per document is unchanged (and already avoids the O(corpus) finalize —
    // that stays lazy until build()/read). Returns one DocId per input, in
    // order; on the first failure it stops and returns the error (documents
    // already appended in this batch remain, exactly as a sequence of
    // add_document() calls would leave them).
    Result<std::vector<DocId>> add_documents(std::vector<DocInput> docs);

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

    // Dense search for MANY queries at once, one result list per query.
    //
    // This is the entry point the GPU backend exists for, and the reason it is
    // a distinct method rather than a loop over dense_search(): scoring q
    // queries against n candidates is a [q x dim] * [dim x n] matrix product,
    // which has q times the arithmetic intensity of q separate scans. A single
    // scan is bandwidth-bound (measured ~47 GB/s on an M1, f32 and f16 alike),
    // so no amount of kernel tuning helps it; the batch is where a GPU can
    // actually win. Callers that have several queries in hand — RAG-Fusion /
    // multi-query, HyDE with several hypotheticals, offline evaluation — should
    // use this rather than looping.
    //
    // Routing is automatic and conservative: this uses the GPU only when there
    // is no HNSW graph to walk (a graph walk is pointer-chasing and must never
    // be offloaded), no metadata filter, the batch clears
    // gpu::min_batch_work(), and a GPU is present. Otherwise it runs the same
    // threaded NEON path a loop would, so it is never SLOWER than looping.
    [[nodiscard]] Result<std::vector<std::vector<Hit>>>
    dense_search_batch(std::span<const std::string> queries, std::size_t k,
                       const MetaFilter& filter = {}) const;

    // ── Resolution / access ─────────────────────────────────────────────────
    [[nodiscard]] const Chunk*    chunk(ChunkId id) const;
    [[nodiscard]] const Document* document(DocId id) const;
    // Look up a live (non-tombstoned) document by its stable external uri, or
    // nullopt if none. Enables upsert semantics (RCP index/add §7.10 mandates an
    // explicit id is an upsert, not a duplicate).
    [[nodiscard]] std::optional<DocId> find_by_uri(std::string_view uri) const;
    [[nodiscard]] SearchResult    resolve(const Hit& h) const;
    [[nodiscard]] std::size_t     chunk_count()    const noexcept {
        std::shared_lock lk(mu_); return chunks_.size();
    }
    [[nodiscard]] std::size_t     document_count() const noexcept {
        std::shared_lock lk(mu_); return docs_.size();
    }

    // How many chunk rows dense_search_batch() would actually score.
    //
    // This is the packed GPU mirror's row count, which is NOT chunk_count():
    // rows are omitted for chunks with no embedding, for ragged
    // (wrong-dimension) rows, and for chunks belonging to soft-deleted
    // documents. That last exclusion is load-bearing — the batch path builds
    // hits straight from the mirror with no later tombstone check — and it is
    // otherwise unobservable, since the batch path only reaches the mirror when
    // a GPU is present. Exposed so the invariant is testable on any machine.
    [[nodiscard]] std::size_t batch_row_count() const noexcept {
        std::shared_lock lk(mu_);
        ensure_packed();
        return packed_ids_.size();
    }
    // A borrowed, READ-LOCKED view of the chunk vector.
    //
    // This exists because the obvious accessor — `const std::vector<Chunk>&
    // chunks() const` — was a data race by construction: it took a shared lock,
    // released it at the return statement, and handed the caller a reference
    // into storage a concurrent add_document() may reallocate. Every bulk
    // consumer (graph, raptor, splade) iterates that vector for as long as it
    // takes to build an index over the whole corpus, which is exactly the
    // window in which a served corpus accepts a write.
    //
    // The lease keeps the shared lock alive for as long as the view is, so the
    // structure cannot change underneath the loop. It is move-only and must
    // not outlive the Corpus. Holding one BLOCKS WRITERS — that is the trade:
    // a consistent snapshot without copying the chunks. Take it, iterate, drop
    // it; do not stash one in a long-lived object.
    //
    // Deadlock note: mu_ is not recursive, so do not call another Corpus
    // method that locks (chunk(), document(), chunk_count(), ...) while a lease
    // is held on the same thread. tokenizer() is safe — it takes no lock.
    class ChunkLease {
    public:
        ChunkLease(std::shared_lock<std::shared_mutex> lk, const std::vector<Chunk>& v)
            : lk_(std::move(lk)), v_(&v) {}

        [[nodiscard]] const std::vector<Chunk>& get() const noexcept { return *v_; }
        operator const std::vector<Chunk>&() const noexcept { return *v_; }
        [[nodiscard]] auto begin() const noexcept { return v_->begin(); }
        [[nodiscard]] auto end()   const noexcept { return v_->end(); }
        [[nodiscard]] std::size_t size()  const noexcept { return v_->size(); }
        [[nodiscard]] bool        empty() const noexcept { return v_->empty(); }
        [[nodiscard]] const Chunk& operator[](std::size_t i) const noexcept { return (*v_)[i]; }

    private:
        std::shared_lock<std::shared_mutex> lk_;
        const std::vector<Chunk>*           v_;
    };

    [[nodiscard]] ChunkLease chunks() const {
        std::shared_lock lk(mu_);
        ensure_linked();
        return ChunkLease(std::move(lk), chunks_);
    }
    [[nodiscard]] const text::Tokenizer& tokenizer() const noexcept { return bm25_.tokenizer(); }

    // The ingest policy in force — chunking mode and geometry, contextual flag,
    // BM25 params. Needed to answer "what settings was this corpus built with?"
    // after a load(), which is otherwise unanswerable: the config round-trips
    // through META precisely so a reopened corpus keeps chunking new documents
    // the way the stored ones were chunked.
    [[nodiscard]] const CorpusConfig& config() const noexcept { return cfg_; }

    // ── Explainability ──────────────────────────────────────────────────────
    // Why did this chunk come back for this query?
    //
    // A ranked list with no justification is unauditable: you cannot tell a
    // relevance bug from a tokenizer bug from a scoring bug, and you cannot show
    // a user why they are reading what they are reading. This answers it with
    // the actual evidence — which query terms matched, what each contributed to
    // BM25, and the dense cosine when an embedder is attached.
    //
    // Costs one index lookup per query term plus (optionally) one embedding, so
    // it is a debugging/UI call, not something to run inside a hot loop.
    struct Explanation {
        ChunkId                             chunk = ChunkId::invalid();
        std::vector<lexical::Bm25Index::TermScore> terms;   // desc contribution
        float  lexical_score = 0.0f;   // sum of terms[].contribution
        float  dense_score   = 0.0f;   // cosine, 0 when no embedder
        bool   has_dense     = false;
        std::size_t matched_terms = 0; // distinct query terms present
        std::size_t query_terms   = 0; // distinct query terms asked for

        // One-line human summary, e.g.
        //   "chunk 12: lexical 7.31 (3/4 terms: pneumonia 5.10, chest 1.42, ...)"
        [[nodiscard]] std::string summary() const;
    };

    [[nodiscard]] Explanation explain(std::string_view query, ChunkId id) const;

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

    // ── Write-ahead log ─────────────────────────────────────────────────────
    //
    // Attach a log so that each mutation becomes durable in O(record) instead
    // of O(corpus). With no log attached, nothing below changes and durability
    // remains "whatever the caller's last save() captured".
    //
    // The log is the SERVER's tool, not the library's default: a batch indexer
    // that ends in one save() wants nothing to do with it, while a process
    // accepting writes over RCP cannot honestly acknowledge one without it.
    // Measured on this machine: acknowledging an index/add by rewriting the
    // snapshot costs 25 ms at 20k documents and 69.7 ms at 50k and grows
    // forever; appending + fsync costs 0.04 ms and is flat.
    //
    // `open_wal` REPLAYS any existing log into this corpus first — that is how
    // a crash recovers — then positions the log for appends. Call it right
    // after load().
    [[nodiscard]] Result<void> open_wal(const std::string& path,
                                        store::SyncMode mode = store::SyncMode::flush);
    [[nodiscard]] bool has_wal() const noexcept;
    [[nodiscard]] std::uint64_t wal_bytes() const noexcept;

    // Snapshot to `path`, then discard the log — the checkpoint.
    //
    // Ordering is the whole point and is not negotiable: the snapshot must be
    // durable BEFORE the log is truncated. Truncating first would leave a
    // window where a crash loses every mutation the snapshot had not yet
    // written, and those mutations were already acknowledged to a client.
    [[nodiscard]] Result<void> checkpoint(const std::string& path);

private:
    CorpusConfig                      cfg_{};
    std::optional<dense::AnyEmbedder> embedder_;
    // Optional LLM seam for Contextual Retrieval; empty means "use the
    // deterministic extractive context". Not serialized — it is a backend
    // binding like embedder_, and the CONTEXT it produced is what persists.
    text::Contextualizer              contextualizer_;
    text::PropositionFn               propositionizer_;
    std::vector<Document>             docs_;
    std::vector<Chunk>                chunks_;
    lexical::Bm25Index                bm25_{cfg_.bm25, cfg_.tokenize};
    std::optional<HnswIndex>          hnsw_;
    bool                              dirty_ = false;

    // ── Concurrency ────────────────────────────────────────────────
    // A served Corpus is read by many request threads and written by index/add
    // at the same time. Without this, add_document() reallocating docs_ frees
    // the storage that every Chunk::meta pointer borrows into, and a concurrent
    // reader dereferences it: measured as an immediate segfault on every run of
    // a 4-reader/1-writer harness.
    //
    // TWO locks, because there are two distinct hazards:
    //
    //   mu_ (shared_mutex) guards the STRUCTURE — docs_, chunks_, bm25_, hnsw_,
    //     deleted_docs_. Readers share it, mutators take it exclusively. It is
    //     a shared_mutex rather than a plain mutex because the workload is
    //     overwhelmingly read-heavy and queries must not serialize behind each
    //     other.
    //
    //   lazy_mu_ guards the LAZY REPAIRS that the read path performs —
    //     relink_meta(), bm25_.finalize(), hnsw_->seal(). These are the reason
    //     a shared_mutex alone is not enough: the const read path MUTATES
    //     (that laziness is what made bulk ingest non-quadratic), so two
    //     concurrent readers holding only a shared lock would race each other
    //     with no writer present at all. Each repair is idempotent, so a short
    //     exclusive lock around it is both correct and cheap — it is taken only
    //     when the corresponding dirty flag is set, which is never on a steady
    //     -state read.
    //
    // Lock ORDER is always mu_ then lazy_mu_; nothing ever acquires them the
    // other way round, so the pair cannot deadlock.
    //
    // `mutable` so const read methods can take a shared lock.
    mutable std::shared_mutex mu_;
    mutable std::mutex        lazy_mu_;
    // A chunk borrows a pointer to its document's metadata, and add_document()
    // may reallocate `docs_`. Rather than relink every chunk on every add (O(n)
    // per document — quadratic ingest), we mark the pointers stale and repair
    // them lazily, before any accessor can hand one out. Mutable + const
    // ensure_linked() because this is memoization, not observable state.
    mutable bool                      meta_stale_ = false;

    // Monotonic count of structural mutations (add/upsert/remove/build). Every
    // snapshot records the epoch it was taken at, so save() can refuse to
    // publish a snapshot older than one already on disk. Guarded by mu_ like
    // the rest of the structure.
    std::uint64_t                     epoch_ = 0;

    // Serializes the PUBLISH step of save() (the temp-file write + rename) and
    // remembers the newest epoch that has reached disk, PER DESTINATION PATH.
    //
    // Static because two different Corpus objects can legitimately target the
    // same path, so the ordering hazard is a property of the path, not of the
    // object. Per-path rather than one global lock because the publish step is
    // the expensive half of a save — a single mutex would make a host serving
    // several corpora serialize their file I/O against each other for no
    // reason, trading one stall for another.
    struct PathState {
        std::mutex    mu;
        std::uint64_t published = 0;
        bool          any       = false;
    };
    static PathState& path_state(const std::string& path);
    std::unordered_set<std::uint32_t> deleted_docs_;   // tombstoned DocId values

    // Optional write-ahead log. Appended under the write lock, so log order
    // matches the order mutations were applied in memory — which is what makes
    // replay reproduce the same corpus.
    //
    // `replaying_` suppresses logging while replay() is feeding records back
    // through add_document/remove_document. Without it, recovery would append
    // everything it just read, doubling the log on every restart.
    store::Wal                        wal_;
    bool                              replaying_ = false;

    [[nodiscard]] Result<void> embed_pending();

    // add/remove with the write lock ALREADY held, so compound operations
    // (upsert) can be performed atomically.
    Result<DocId> add_document_locked(std::string uri, std::string text, Metadata meta,
                                      std::string title);
    Result<void>  remove_document_locked(DocId id);
    [[nodiscard]] std::optional<DocId> find_by_uri_locked(std::string_view uri) const;

    // Unlocked internals. Every public method acquires the appropriate lock and
    // then delegates here, so that methods which call ONE ANOTHER (e.g.
    // lexical_search -> chunk, dense_search -> passes -> chunk) do not attempt
    // to re-acquire a lock they already hold. std::shared_mutex is NOT
    // recursive: a second shared_lock on the same thread deadlocks if a writer
    // is queued between them.
    [[nodiscard]] const Chunk*    chunk_locked(ChunkId id) const;
    [[nodiscard]] const Document* document_locked(DocId id) const;
    [[nodiscard]] SearchResult    resolve_locked(const Hit& h) const;
    [[nodiscard]] bool            passes_locked(ChunkId id, const MetaFilter& f) const;
    [[nodiscard]] std::vector<Hit> lexical_search_locked(std::string_view query, std::size_t k) const;
    [[nodiscard]] Result<std::vector<Hit>>
    dense_search_locked(std::string_view query, std::size_t k, const MetaFilter& filter) const;
    [[nodiscard]] Result<void>    build_locked();

    // A contiguous [rows x dim] copy of every embedded chunk's vector, plus the
    // chunk id each row came from.
    //
    // This exists because gpu::score_batch() needs one contiguous matrix while
    // chunks_ stores each embedding in its own vector. Packing PER CALL was
    // measured and is a losing trade: at n=200k, dim=384 the pack alone costs
    // 37 ms against a 46.7 ms CPU scan, so a 32-query batch went 1.70x -> 0.73x
    // once the copy was counted. It has to be amortized or not done at all.
    //
    // Built lazily on first batch use and invalidated by any structural
    // mutation (via epoch_), so a corpus that never runs a batch query never
    // pays the memory. Guarded by lazy_mu_, like the other read-path repairs.
    mutable std::vector<float>         packed_;
    mutable std::vector<std::uint32_t> packed_ids_;
    mutable std::size_t                packed_dim_   = 0;
    mutable std::uint64_t              packed_epoch_ = 0;
    mutable bool                       packed_valid_ = false;

    // Ensure packed_ reflects the current chunks_. Caller must hold mu_.
    void ensure_packed() const;
    // Pack the whole corpus into container sections. Caller must hold at least a
    // shared lock; the returned Container owns copies, so the caller can (and
    // should) release the lock before paying for CRC + file I/O.
    [[nodiscard]] Result<store::Container> snapshot_locked() const;

    // Repair borrowed meta pointers if a preceding add_document() invalidated
    // them. Called by every read path that can expose a Chunk.
    void ensure_linked() const;

    // Re-point every chunk's borrowed `meta` at the current `docs_` storage.
    void relink_meta();
    // Steal all state from `o` and relink meta pointers to our storage.
    void move_from(Corpus&& o);
};

} // namespace rag::index
