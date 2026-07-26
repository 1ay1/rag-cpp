// rag/index/corpus.cpp — document ingest, hybrid indexing, persistence.

#include "rag/index/corpus.hpp"
#include "rag/dense/simd.hpp"
#include "rag/store/container.hpp"
#include "rag/util/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>

#include <nlohmann/json.hpp>

namespace rag::index {

using json = nlohmann::json;

void Corpus::relink_meta() {
    for (auto& ch : chunks_)
        ch.meta = (ch.doc.get() < docs_.size()) ? &docs_[ch.doc.get()].meta : nullptr;
}

void Corpus::move_from(Corpus&& o) {
    cfg_       = std::move(o.cfg_);
    embedder_  = std::move(o.embedder_);
    contextualizer_ = std::move(o.contextualizer_);
    docs_      = std::move(o.docs_);
    chunks_    = std::move(o.chunks_);
    bm25_      = std::move(o.bm25_);
    hnsw_      = std::move(o.hnsw_);
    dirty_     = o.dirty_;
    epoch_     = o.epoch_;
    deleted_docs_ = std::move(o.deleted_docs_);
    wal_       = std::move(o.wal_);
    replaying_ = o.replaying_;
    relink_meta();   // borrowed pointers now point at OUR docs_ storage
    meta_stale_ = false;
}

Result<DocId> Corpus::add_document(std::string uri, std::string text, Metadata meta, std::string title) {
    std::unique_lock lk(mu_);
    return add_document_locked(std::move(uri), std::move(text), std::move(meta), std::move(title));
}

Result<DocId> Corpus::upsert_document(std::string uri, std::string text, Metadata meta,
                                      std::string title) {
    // ONE lock across find + remove + add, so two threads upserting the same
    // uri cannot both miss and both insert.
    std::unique_lock lk(mu_);
    if (!uri.empty())
        if (auto existing = find_by_uri_locked(uri))
            (void)remove_document_locked(*existing);
    return add_document_locked(std::move(uri), std::move(text), std::move(meta), std::move(title));
}

Result<DocId> Corpus::add_document_locked(std::string uri, std::string text, Metadata meta,
                                          std::string title) {
    // Log BEFORE mutating. If the append fails, nothing has changed and the
    // caller gets an honest error; logging after the mutation would leave a
    // corpus that has a document its log does not, so a crash would silently
    // roll it back after the client was told it succeeded.
    if (wal_.is_open() && !replaying_) {
        store::WalRecord rec;
        rec.op = store::WalOp::add_document;
        rec.uri = uri; rec.title = title; rec.text = text; rec.meta = meta;
        if (auto w = wal_.append(rec); !w) return std::unexpected(w.error());
    }

    DocId did{static_cast<std::uint32_t>(docs_.size())};
    Document doc;
    doc.id = did; doc.uri = std::move(uri); doc.title = std::move(title);
    doc.text = std::move(text); doc.meta = std::move(meta);
    docs_.push_back(std::move(doc));
    const Document& stored = docs_.back();

    auto new_chunks = [&] {
        if (cfg_.chunking != CorpusConfig::Chunking::semantic)
            return text::chunk_document(did, stored.text, cfg_.chunk);
        // Semantic chunking prefers the embedder (true topical drift) but must
        // degrade rather than fail: an unavailable backend should change chunk
        // BOUNDARIES, never make ingest impossible.
        if (embedder_) {
            if (auto sc = text::semantic_chunk(did, stored.text, *embedder_, cfg_.semantic))
                return std::move(*sc);
        }
        return text::semantic_chunk_lexical(did, stored.text, cfg_.semantic);
    }();

    // Contextual Retrieval, before anything is indexed.
    //
    // Order matters and is not arbitrary: indexed_text() is context ⊕ body, and
    // it is what feeds BOTH bm25_.add() below and the embedder in
    // embed_pending(). Situating the chunk after either of those would leave
    // the indexes describing text no longer in the store. So this runs here,
    // between chunking and id assignment, and nowhere else.
    //
    // contextualize() APPENDS to the existing `context` rather than replacing
    // it, so the structural chunker's heading breadcrumb survives — the two
    // signals are complementary (where in the document vs. what the document
    // is about).
    if (cfg_.contextual)
        text::contextualize(new_chunks, stored.text, contextualizer_);

    for (auto& ch : new_chunks) {
        ChunkId cid{static_cast<std::uint32_t>(chunks_.size())};
        ch.id   = cid;
        bm25_.add(cid.get(), ch.indexed_text());
        chunks_.push_back(std::move(ch));
    }
    // NOTE: this used to relink_meta() and bm25_.finalize() here — both O(total
    // corpus), per document, which made bulk ingest quadratic (it dominated a
    // 20k-document build at ~1.8s). Neither is needed until something READS:
    //   • chunk meta pointers are borrowed into docs_, which push_back above may
    //     have reallocated — so they are relinked lazily in ensure_linked(),
    //     driven by the `meta_stale_` flag, before any accessor hands one out;
    //   • bm25 idf/avgdl are pure functions of the accumulated counts, so
    //     finalize() is idempotent and only has to run before a query.
    // build() does both; the read paths do them on demand for callers who
    // query without an explicit build().
    meta_stale_ = true;
    dirty_      = true;
    ++epoch_;
    return did;
}

void Corpus::ensure_linked() const {
    // Double-checked: the common case is a clean corpus, where this is a single
    // relaxed read and no lock at all. Only the rare stale case pays for the
    // mutex, and the re-check inside it stops two readers from both relinking.
    if (!meta_stale_) return;
    std::lock_guard lk(lazy_mu_);
    if (!meta_stale_) return;
    const_cast<Corpus*>(this)->relink_meta();
    meta_stale_ = false;
}

Result<void> Corpus::embed_pending() {
    if (!embedder_) return {};
    // Collect chunks with empty embeddings.
    std::vector<std::size_t> pending;
    for (std::size_t i = 0; i < chunks_.size(); ++i)
        if (chunks_[i].embedding.empty()) pending.push_back(i);
    if (pending.empty()) return {};

    const std::size_t bs = cfg_.embed_batch ? cfg_.embed_batch : 1;
    const std::size_t nb = (pending.size() + bs - 1) / bs;

    // Batches are independent: each reads a disjoint slice of `pending` and
    // writes only the chunks named by that slice, so they can be in flight
    // concurrently. HOW concurrently is the backend's call — an in-process
    // model already owns every core (hint 1 ⇒ serial), a hosted endpoint is
    // latency-bound (hint 8) — see dense::ConcurrencyAware.
    const std::size_t workers = std::min(embedder_->max_concurrency(), nb);

    // First failure wins; later batches short-circuit rather than pile up
    // retries against a backend that is already known to be down.
    std::atomic<bool> failed{false};
    std::mutex        err_mu;
    Error             first_err{};

    auto run_batch = [&](std::size_t b) {
        if (failed.load(std::memory_order_relaxed)) return;
        const std::size_t off = b * bs;
        const std::size_t end = std::min(off + bs, pending.size());
        std::vector<std::string> batch;
        batch.reserve(end - off);
        for (std::size_t j = off; j < end; ++j) batch.push_back(chunks_[pending[j]].indexed_text());

        auto res = embedder_->embed(batch);
        if (!res) {
            std::lock_guard lk(err_mu);
            if (!failed.exchange(true, std::memory_order_acq_rel)) first_err = res.error();
            return;
        }
        auto& vecs = *res;
        for (std::size_t j = 0; j < vecs.size() && off + j < end; ++j)
            chunks_[pending[off + j]].embedding = std::move(vecs[j]);
    };

    if (workers <= 1)
        for (std::size_t b = 0; b < nb; ++b) run_batch(b);
    else
        util::parallel_for_dynamic(nb, workers, run_batch);

    if (failed.load(std::memory_order_acquire)) return std::unexpected(first_err);
    return {};
}

Result<void> Corpus::build() {
    std::unique_lock lk(mu_);
    return build_locked();
}

Result<void> Corpus::build_locked() {
    ensure_linked();
    bm25_.finalize();
    if (embedder_) {
        if (auto r = embed_pending(); !r) return r;  // degrade: propagate, caller may ignore
        // Build HNSW past threshold.
        if (chunks_.size() >= cfg_.hnsw_threshold) {
            HnswIndex idx(cfg_.hnsw);
            // Collect the embedded chunks, then construct the graph in parallel.
            // build_batch stages all nodes first and links them across every
            // core — the dominant cost of indexing a large corpus.
            std::vector<std::size_t> rows;
            rows.reserve(chunks_.size());
            for (std::size_t i = 0; i < chunks_.size(); ++i)
                if (!chunks_[i].embedding.empty()) rows.push_back(i);

            idx.build_batch(rows.size(),
                [&](std::size_t i) -> std::span<const float> { return chunks_[rows[i]].embedding; },
                [&](std::size_t i) { return chunks_[rows[i]].id.get(); });
            hnsw_ = std::move(idx);
        }
    }
    dirty_ = false;
    ++epoch_;
    return {};
}

std::vector<Hit> Corpus::lexical_search(std::string_view query, std::size_t k) const {
    std::shared_lock lk(mu_);
    return lexical_search_locked(query, k);
}

std::vector<Hit> Corpus::lexical_search_locked(std::string_view query, std::size_t k) const {
    // add_document() no longer finalizes on every insert (that was quadratic);
    // a caller may therefore query without an intervening build(). finalize()
    // is idempotent and pure in the accumulated counts, so bringing it up to
    // date here is both cheap and correct.
    //
    // It is also a WRITE performed under a shared lock, so it needs lazy_mu_:
    // without it two concurrent readers would finalize the same index at the
    // same time. Double-checked, so a finalized index costs one bool read.
    if (!bm25_.finalized()) {
        std::lock_guard lz(lazy_mu_);
        if (!bm25_.finalized()) const_cast<Corpus*>(this)->bm25_.finalize();
    }
    if (deleted_docs_.empty()) return bm25_.search(query, k);
    // Over-fetch, then drop tombstoned chunks and truncate to k.
    auto hits = bm25_.search(query, k + deleted_docs_.size() * 2 + k);
    std::vector<Hit> out;
    out.reserve(std::min(hits.size(), k));
    for (const auto& h : hits) {
        const Chunk* ch = chunk_locked(h.chunk);
        if (ch && deleted_docs_.count(ch->doc.get())) continue;
        out.push_back(h);
        if (out.size() >= k) break;
    }
    return out;
}

Result<void> Corpus::remove_document(DocId id) {
    std::unique_lock lk(mu_);
    return remove_document_locked(id);
}

Result<void> Corpus::remove_document_locked(DocId id) {
    ensure_linked();
    if (id.get() >= docs_.size() || deleted_docs_.count(id.get()))
        return fail<void>(Errc::not_found, "remove_document: unknown or already-deleted id");
    // Logged only after the validity check, so a no-op delete writes nothing.
    if (wal_.is_open() && !replaying_) {
        store::WalRecord rec;
        rec.op = store::WalOp::remove_document;
        rec.doc_id = id.get();
        if (auto w = wal_.append(rec); !w) return w;
    }
    deleted_docs_.insert(id.get());
    ++epoch_;
    // Tombstone the doc's chunks in the HNSW graph so dense search skips them.
    if (hnsw_)
        for (const auto& ch : chunks_)
            if (ch.doc.get() == id.get()) hnsw_->remove(ch.id.get());
    return {};
}

bool Corpus::is_deleted(DocId id) const noexcept {
    std::shared_lock lk(mu_);
    return deleted_docs_.count(id.get()) != 0;
}
std::size_t Corpus::live_document_count() const noexcept {
    std::shared_lock lk(mu_);
    return docs_.size() - deleted_docs_.size();
}

Result<std::vector<Hit>> Corpus::dense_search(std::string_view query, std::size_t k) const {
    return dense_search(query, k, MetaFilter{});
}

Result<Vector> Corpus::embed_text(const std::string& text) const {
    if (!embedder_) return fail<Vector>(Errc::unavailable, "no embedder");
    auto v = embedder_->embed_one(text);
    if (!v) return std::unexpected(v.error());
    dense::normalize(*v);
    return v;
}

Result<std::vector<Hit>> Corpus::dense_search(std::string_view query, std::size_t k,
                                              const MetaFilter& filter) const {
    std::shared_lock lk(mu_);
    return dense_search_locked(query, k, filter);
}

Result<std::vector<Hit>> Corpus::dense_search_locked(std::string_view query, std::size_t k,
                                                     const MetaFilter& filter) const {
    if (!embedder_) return fail<std::vector<Hit>>(Errc::unavailable, "no embedder");
    ensure_linked();     // the allow-predicate below dereferences chunk->meta
    auto qv = embedder_->embed_one(std::string(query));
    if (!qv) return std::unexpected(qv.error());
    dense::normalize(*qv);

    // Build an allow-predicate over chunk id from the metadata filter.
    HnswIndex::AllowFn allow;
    if (filter) {
        allow = [this, &filter](std::uint32_t id) -> bool {
            const Chunk* ch = (id < chunks_.size()) ? &chunks_[id] : nullptr;
            if (!ch || !ch->meta) return false;
            return filter(*ch->meta);
        };
    }

    // Use HNSW if built, else brute-force cosine (with the same pre-filter).
    if (hnsw_) {
        return allow ? hnsw_->search_filtered(*qv, k, allow) : hnsw_->search(*qv, k);
    }

    // Brute-force scan. Parallel over contiguous blocks: each worker scores its
    // own range into a private buffer, then we concatenate and select. Scoring
    // is pure (reads immutable embeddings), so no synchronization is needed
    // beyond the join.
    const std::size_t n = chunks_.size();
    std::vector<std::vector<Hit>> parts(util::block_count(n));

    util::parallel_blocks(n, [&](std::size_t lo, std::size_t hi, std::size_t b) {
        auto& local = parts[b];
        local.reserve((hi - lo) / 4 + 8);
        for (std::size_t i = lo; i < hi; ++i) {
            const auto& ch = chunks_[i];
            if (ch.embedding.empty()) continue;
            if (!deleted_docs_.empty() && deleted_docs_.count(ch.doc.get())) continue;
            if (allow && !allow(ch.id.get())) continue;
            local.push_back(Hit{ch.id, Score{dense::dot(ch.embedding, *qv)}});
        }
    });

    std::vector<Hit> hits;
    std::size_t total = 0;
    for (const auto& p : parts) total += p.size();
    hits.reserve(total);
    for (auto& p : parts) hits.insert(hits.end(), p.begin(), p.end());

    const std::size_t kk = std::min(k, hits.size());
    std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk), hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    hits.resize(kk);
    return hits;
}

const Chunk* Corpus::chunk(ChunkId id) const {
    std::shared_lock lk(mu_);
    return chunk_locked(id);
}
const Chunk* Corpus::chunk_locked(ChunkId id) const {
    ensure_linked();
    return id.get() < chunks_.size() ? &chunks_[id.get()] : nullptr;
}
const Document* Corpus::document(DocId id) const {
    std::shared_lock lk(mu_);
    return document_locked(id);
}
const Document* Corpus::document_locked(DocId id) const {
    return id.get() < docs_.size() ? &docs_[id.get()] : nullptr;
}

std::optional<DocId> Corpus::find_by_uri(std::string_view uri) const {
    std::shared_lock lk(mu_);
    return find_by_uri_locked(uri);
}

std::optional<DocId> Corpus::find_by_uri_locked(std::string_view uri) const {
    if (uri.empty()) return std::nullopt;
    for (const auto& d : docs_) {
        if (deleted_docs_.count(d.id.get())) continue;   // skip tombstones
        if (d.uri == uri) return d.id;
    }
    return std::nullopt;
}

SearchResult Corpus::resolve(const Hit& h) const {
    std::shared_lock lk(mu_);
    return resolve_locked(h);
}

SearchResult Corpus::resolve_locked(const Hit& h) const {
    SearchResult r;
    const Chunk* ch = chunk_locked(h.chunk);
    if (!ch) return r;
    r.chunk = ch->id; r.doc = ch->doc; r.score = h.score;
    r.text = ch->text; r.context = ch->context;
    r.start_line = ch->start_line; r.end_line = ch->end_line;
    if (const Document* d = document_locked(ch->doc)) r.uri = d->uri;
    return r;
}

bool Corpus::passes(ChunkId id, const MetaFilter& f) const {
    std::shared_lock lk(mu_);
    return passes_locked(id, f);
}

bool Corpus::passes_locked(ChunkId id, const MetaFilter& f) const {
    const Chunk* ch = chunk_locked(id);
    if (!ch || !ch->meta) return !f;   // no metadata: pass only if no filter
    return f ? f(*ch->meta) : true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence — the stable, versioned .ragdb container (see rag/store + FORMAT.md).
// Sections: META (config json), DOCS, CHNK (chunk records), EMBD (embeddings),
// BM25 (inverted index), HNSW (ANN graph). CRC-verified on load.
// ─────────────────────────────────────────────────────────────────────────────
Result<void> Corpus::save(const std::string& path) const {
    // save() splits into two phases that want very different lock treatment.
    //
    //   1. SNAPSHOT — walk docs_/chunks_/bm25_/hnsw_ and pack them into section
    //      blobs. This touches corpus state, so it must hold a lock, and the
    //      lock is what makes the snapshot consistent: no document can be
    //      appended halfway through serializing the arrays.
    //
    //   2. WRITE — concatenate those blobs, CRC the result, write it to a temp
    //      file, fsync, rename, fsync the directory. This touches NO corpus
    //      state whatsoever; the Container owns its own copies.
    //
    // Phase 2 is the expensive one — measured on a 20k-doc corpus it is ~24 of
    // the ~28 ms, almost all of it CRC and the big concatenating copy — and it
    // used to run with the shared lock still held. Readers did not care (they
    // share it too), but every WRITER blocked for the whole thing: an
    // add_document() concurrent with a save loop measured 6.0 ms mean against
    // 0.001 ms uncontended. Dropping the lock between the phases takes that
    // stall down to just the snapshot.
    store::Container snap;
    std::uint64_t    at_epoch = 0;
    {
        std::shared_lock lk(mu_);
        auto s = snapshot_locked();
        if (!s) return std::unexpected(s.error());
        snap     = std::move(*s);
        at_epoch = epoch_;
    }

    // PUBLISH, in epoch order.
    //
    // Splitting the phases means two concurrent save()s can interleave as:
    // A snapshots at epoch 10, B snapshots at epoch 20, B renames, A renames —
    // and the file on disk goes BACKWARDS to epoch 10.
    //
    // This is rare but real, and it took an honest harness to see it. A loop
    // that loads the file while savers and a writer run showed nothing at all
    // with 3-8 savers on 8 cores: the savers stay roughly in lockstep, so their
    // renames happen to come out in snapshot order. Only when the savers are
    // OVERSUBSCRIBED (16 and 32 threads on 8 cores, so the scheduler preempts
    // one between its snapshot and its rename) does the on-disk doc count go
    // backwards — 2 and 4 times per 300 loads. With this gate: 0, at 16, 32 and
    // 64 savers.
    //
    // The rename is therefore serialized and gated on the epoch that last
    // reached this path. A snapshot that lost the race is DROPPED, not written:
    // its content is a strict prefix of what is already there, so discarding it
    // loses nothing, and reporting success is honest — the caller asked for the
    // state at their save() call to be durable, and a newer state that contains
    // it is on disk.
    auto& st = path_state(path);
    std::lock_guard pk(st.mu);
    if (st.any && st.published > at_epoch) return {};
    if (auto r = snap.write_file(path); !r) return r;
    st.published = at_epoch;
    st.any       = true;
    return {};
}

// ─── Write-ahead log ────────────────────────────────────────────────────
Result<void> Corpus::open_wal(const std::string& path, store::SyncMode mode) {
    std::unique_lock lk(mu_);

    // REPLAY FIRST, then open for appending.
    //
    // Replay runs through the ordinary mutation paths rather than poking at
    // docs_/chunks_ directly, so recovery produces exactly the corpus the
    // original run had — same chunking, same ids, same index state. `replaying_`
    // stops those calls from logging what they are reading; without it every
    // restart would double the log.
    auto records = store::Wal::replay(path);
    if (!records) return std::unexpected(records.error());

    if (!records->empty()) {
        replaying_ = true;
        for (const auto& rec : *records) {
            if (rec.op == store::WalOp::add_document) {
                // Replayed as an UPSERT for the same reason index/add is one:
                // a log may contain several writes to the same uri, and the
                // last must win rather than accumulate duplicates.
                std::optional<DocId> existing;
                if (!rec.uri.empty()) existing = find_by_uri_locked(rec.uri);
                if (existing) (void)remove_document_locked(*existing);
                if (auto r = add_document_locked(rec.uri, rec.text, rec.meta, rec.title); !r) {
                    replaying_ = false;
                    return std::unexpected(r.error());
                }
            } else {
                // A delete of a document the snapshot never had is not an
                // error: the log can outlive the doc it refers to.
                (void)remove_document_locked(DocId{rec.doc_id});
            }
        }
        replaying_ = false;
        if (auto b = build_locked(); !b) return b;
    }

    return wal_.open(path, mode);
}

bool Corpus::has_wal() const noexcept {
    std::shared_lock lk(mu_);
    return wal_.is_open();
}

std::uint64_t Corpus::wal_bytes() const noexcept {
    std::shared_lock lk(mu_);
    return wal_.size_bytes();
}

Result<void> Corpus::checkpoint(const std::string& path) {
    // save() takes its own lock and does the snapshot/publish dance, so it runs
    // OUTSIDE the write lock here — taking mu_ around it would deadlock (mu_ is
    // not recursive) and would also block writers for the whole serialization,
    // which is exactly what the snapshot/publish split was built to avoid.
    if (auto s = save(path); !s) return s;

    // Only now is it safe to drop the log. save() has fsync'd the snapshot and
    // renamed it into place, so every record in the log is represented on disk.
    // Truncating first would leave a window in which a crash loses mutations
    // that were already acknowledged.
    std::unique_lock lk(mu_);
    if (!wal_.is_open()) return {};
    return wal_.truncate();
}

Corpus::PathState& Corpus::path_state(const std::string& path) {
    // The registry lock is held only long enough to find/insert the entry; the
    // per-path lock — which is what the slow file I/O runs under — is taken by
    // the caller. node-based map so references stay valid as it grows.
    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::unique_ptr<PathState>> registry;
    std::lock_guard lk(registry_mu);
    auto& slot = registry[path];
    if (!slot) slot = std::make_unique<PathState>();
    return *slot;
}

Result<store::Container> Corpus::snapshot_locked() const {
    ensure_linked();
    store::Container c;
    std::uint32_t flags = 0;

    // META — corpus config (round-trips the knobs that affect query behaviour).
    {
        json m;
        m["hnsw_threshold"] = cfg_.hnsw_threshold;
        m["embed_batch"]    = cfg_.embed_batch;
        // Contextual Retrieval is an INGEST policy, not a query knob, but it
        // still has to round-trip: a corpus reopened for writing (serve
        // --write) must keep situating the documents it accepts, or the ones
        // added after the restart are indexed differently from the ones added
        // before it, and only half the store carries the disambiguating text.
        //
        // The same argument applies to the chunker's geometry, which was NOT
        // persisted before: reopening a corpus built with max_lines=3 and
        // adding a document chunked it at the default 40, so one store ended up
        // holding two incompatible chunk granularities. Both are ingest policy
        // and both round-trip.
        m["contextual"]     = cfg_.contextual;
        m["chunk"] = { {"max_lines", cfg_.chunk.max_lines},
                       {"max_chars", cfg_.chunk.max_chars},
                       {"overlap_lines", cfg_.chunk.overlap_lines},
                       {"heading_context", cfg_.chunk.heading_context} };
        m["bm25"] = { {"k1", cfg_.bm25.k1}, {"b", cfg_.bm25.b} };
        c.put(store::Tag::meta, m.dump());
    }

    // DOCS.
    {
        store::Writer w;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(docs_.size()));
        for (const auto& d : docs_) {
            w.u<std::uint32_t>(d.id.get());
            w.str(d.uri); w.str(d.title); w.str(d.text);
            w.u<std::uint32_t>(static_cast<std::uint32_t>(d.meta.size()));
            for (const auto& [k, v] : d.meta) { w.str(k); w.str(v); }
        }
        c.put(store::Tag::docs, std::move(w.data()));
    }

    // CHNK + EMBD (parallel arrays).
    {
        store::Writer w, e;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(chunks_.size()));
        bool any_emb = false;
        for (const auto& ch : chunks_) {
            w.u<std::uint32_t>(ch.id.get());
            w.u<std::uint32_t>(ch.doc.get());
            w.str(ch.text); w.str(ch.context);
            w.u<std::uint32_t>(ch.start_line); w.u<std::uint32_t>(ch.end_line);
            e.u<std::uint32_t>(static_cast<std::uint32_t>(ch.embedding.size()));
            if (!ch.embedding.empty()) {
                any_emb = true;
                e.bytes(std::string_view(reinterpret_cast<const char*>(ch.embedding.data()),
                                         ch.embedding.size() * sizeof(float)));
            }
        }
        c.put(store::Tag::chunks, std::move(w.data()));
        if (any_emb) { c.put(store::Tag::embed, std::move(e.data())); flags |= store::kHasEmbeddings; }
    }

    // BM25 + HNSW blobs (already self-describing).
    c.put(store::Tag::bm25, bm25_.serialize());
    if (hnsw_) { c.put(store::Tag::hnsw, hnsw_->serialize()); flags |= store::kHasHnsw; }

    // TOMB — soft-delete tombstones.
    //
    // These are NOT derivable from anything else in the file. A tombstoned
    // document keeps its row in DOCS and its rows in CHNK and its postings in
    // BM25 (that is what "soft" means — ids stay stable so no other structure
    // has to be rewritten); the ONLY thing that hides it from results is
    // membership in deleted_docs_. Omitting this section therefore resurrects
    // every deleted document on the next load, fully searchable.
    //
    // Written sorted so the file is byte-identical for identical corpora —
    // deleted_docs_ is an unordered_set, whose iteration order is not stable.
    // Skipped entirely when empty so the common case costs zero bytes.
    if (!deleted_docs_.empty()) {
        std::vector<std::uint32_t> ids(deleted_docs_.begin(), deleted_docs_.end());
        std::sort(ids.begin(), ids.end());
        store::Writer w;
        w.u<std::uint32_t>(static_cast<std::uint32_t>(ids.size()));
        for (std::uint32_t id : ids) w.u<std::uint32_t>(id);
        c.put(store::Tag::tomb, std::move(w.data()));
    }

    c.set_flags(flags);
    return c;
}

Result<Corpus> Corpus::load(const std::string& path) {
    auto cont = store::Container::read_file(path);
    if (!cont) return std::unexpected(cont.error());

    Corpus c;

    if (const std::string* meta = cont->get(store::Tag::meta)) {
        auto m = json::parse(*meta, nullptr, false);
        if (!m.is_discarded()) {
            c.cfg_.hnsw_threshold = m.value("hnsw_threshold", c.cfg_.hnsw_threshold);
            c.cfg_.embed_batch    = m.value("embed_batch", c.cfg_.embed_batch);
            c.cfg_.contextual     = m.value("contextual", c.cfg_.contextual);
            if (m.contains("chunk")) {
                const auto& ck = m["chunk"];
                c.cfg_.chunk.max_lines       = ck.value("max_lines", c.cfg_.chunk.max_lines);
                c.cfg_.chunk.max_chars       = ck.value("max_chars", c.cfg_.chunk.max_chars);
                c.cfg_.chunk.overlap_lines   = ck.value("overlap_lines", c.cfg_.chunk.overlap_lines);
                c.cfg_.chunk.heading_context = ck.value("heading_context", c.cfg_.chunk.heading_context);
            }
            if (m.contains("bm25")) {
                c.cfg_.bm25.k1 = m["bm25"].value("k1", c.cfg_.bm25.k1);
                c.cfg_.bm25.b  = m["bm25"].value("b", c.cfg_.bm25.b);
            }
        }
    }

    // DOCS.
    if (const std::string* docs = cont->get(store::Tag::docs)) {
        store::Reader r(*docs);
        std::uint32_t n; if (!r.u(n)) return fail<Corpus>(Errc::corrupt_index, "docs count");
        c.docs_.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            Document d; std::uint32_t id;
            if (!r.u(id) || !r.str(d.uri) || !r.str(d.title) || !r.str(d.text))
                return fail<Corpus>(Errc::corrupt_index, "doc");
            d.id = DocId{id};
            std::uint32_t mn; if (!r.u(mn)) return fail<Corpus>(Errc::corrupt_index, "doc meta");
            for (std::uint32_t j = 0; j < mn; ++j) {
                std::string k, v;
                if (!r.str(k) || !r.str(v)) return fail<Corpus>(Errc::corrupt_index, "doc meta kv");
                d.meta[k] = v;
            }
            c.docs_.push_back(std::move(d));
        }
    }

    // CHNK.
    if (const std::string* chunks = cont->get(store::Tag::chunks)) {
        store::Reader r(*chunks);
        std::uint32_t n; if (!r.u(n)) return fail<Corpus>(Errc::corrupt_index, "chunk count");
        c.chunks_.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            Chunk ch; std::uint32_t id, doc;
            if (!r.u(id) || !r.u(doc) || !r.str(ch.text) || !r.str(ch.context) ||
                !r.u(ch.start_line) || !r.u(ch.end_line))
                return fail<Corpus>(Errc::corrupt_index, "chunk");
            ch.id = ChunkId{id}; ch.doc = DocId{doc};
            c.chunks_.push_back(std::move(ch));
        }
    }
    // Link all chunk meta pointers now that docs_ and chunks_ are populated.
    c.relink_meta();
    c.meta_stale_ = false;

    // EMBD (parallel to CHNK).
    if (const std::string* emb = cont->get(store::Tag::embed)) {
        store::Reader r(*emb);
        for (auto& ch : c.chunks_) {
            std::uint32_t dim; if (!r.u(dim)) break;
            if (dim == 0) continue;
            std::string_view raw;
            if (!r.bytes(dim * sizeof(float), raw)) return fail<Corpus>(Errc::corrupt_index, "embedding");
            ch.embedding.resize(dim);
            std::memcpy(ch.embedding.data(), raw.data(), dim * sizeof(float));
        }
    }

    // BM25 (required).
    if (const std::string* b = cont->get(store::Tag::bm25)) {
        auto idx = lexical::Bm25Index::deserialize(*b);
        if (!idx) return std::unexpected(idx.error());
        c.bm25_ = std::move(*idx);
    }
    // HNSW (optional).
    if (const std::string* h = cont->get(store::Tag::hnsw)) {
        auto idx = HnswIndex::deserialize(*h);
        if (!idx) return std::unexpected(idx.error());
        c.hnsw_ = std::move(*idx);
    }
    // TOMB (optional; absent in format minor 0 and when nothing is deleted).
    if (const std::string* t = cont->get(store::Tag::tomb)) {
        store::Reader r(*t);
        std::uint32_t n; if (!r.u(n)) return fail<Corpus>(Errc::corrupt_index, "tomb count");
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t id; if (!r.u(id)) return fail<Corpus>(Errc::corrupt_index, "tomb id");
            c.deleted_docs_.insert(id);
        }
    }
    return c;
}

} // namespace rag::index
