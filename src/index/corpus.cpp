// rag/index/corpus.cpp — document ingest, hybrid indexing, persistence.

#include "rag/index/corpus.hpp"
#include "rag/dense/simd.hpp"
#include "rag/gpu/device.hpp"
#include "rag/loaders/structure.hpp"
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
    propositionizer_ = std::move(o.propositionizer_);
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
    // The packed mirror is a cache keyed on epoch_, and epoch_ came from `o`.
    // Not invalidating it here would let a stale (or empty) matrix be accepted
    // as current, because the epoch it was stamped with is now ours too.
    packed_valid_ = false;
    packed_.clear();
    packed_ids_.clear();
    packed_dim_ = 0;
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
        if (cfg_.chunking == CorpusConfig::Chunking::proposition)
            // One chunk per atomic statement. The seam is optional: absent (or
            // failing) it uses the deterministic sentence splitter, so ingest
            // never depends on a model being reachable.
            return text::proposition_chunk(did, stored.text, propositionizer_);
        if (cfg_.chunking == CorpusConfig::Chunking::source) {
            // Code, chunked on definitions. The extension comes from metadata
            // when the loader knew it, else from the URI; when it names no
            // language chunk_source infers the file's own structure, and when
            // there is no structure to infer it degrades to windows. So this
            // never has to guess wrong in a way that costs the caller anything.
            std::string ext;
            if (auto it = stored.meta.find("ext"); it != stored.meta.end()) ext = it->second;
            else if (auto dot = stored.uri.rfind('.'); dot != std::string::npos &&
                     stored.uri.size() - dot <= 8)
                ext = stored.uri.substr(dot);
            for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return loaders::chunk_source(did, ext, stored.text);
        }
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

Result<std::vector<DocId>> Corpus::add_documents(std::vector<DocInput> docs) {
    std::unique_lock lk(mu_);   // taken ONCE for the whole batch

    // Group-commit the WAL up front: one durable append for all N documents
    // instead of N. We log first (before any mutation) for the same reason
    // add_document_locked does — a crash after the sync replays cleanly, a crash
    // before it simply loses an unacknowledged batch. After this succeeds we
    // insert with WAL logging SUPPRESSED (the records are already on disk), which
    // is exactly what the replay path relies on via `replaying_`.
    const bool log = wal_.is_open() && !replaying_;
    if (log) {
        std::vector<store::WalRecord> recs;
        recs.reserve(docs.size());
        for (const auto& d : docs) {
            store::WalRecord rec;
            rec.op = store::WalOp::add_document;
            rec.uri = d.uri; rec.title = d.title; rec.text = d.text; rec.meta = d.meta;
            recs.push_back(std::move(rec));
        }
        if (auto w = wal_.append_batch(recs); !w) return std::unexpected(w.error());
    }

    // Suppress per-document WAL logging while inserting (already logged above).
    const bool prev_replaying = replaying_;
    replaying_ = true;
    struct RestoreFlag {
        bool& f; bool v; ~RestoreFlag() { f = v; }
    } restore{replaying_, prev_replaying};

    std::vector<DocId> ids;
    ids.reserve(docs.size());
    for (auto& d : docs) {
        auto r = add_document_locked(std::move(d.uri), std::move(d.text),
                                     std::move(d.meta), std::move(d.title));
        if (!r) return std::unexpected(r.error());
        ids.push_back(*r);
    }
    return ids;
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

namespace {

// Ranking order for the dense scan: score descending, chunk id ascending.
//
// The id tiebreak is not cosmetic. std::partial_sort is NOT stable, so a
// comparator that only looks at score leaves tied hits in an order decided by
// the algorithm's internals — which means the ranking depends on how the
// candidate list happened to be laid out. That made the parallel scan and the
// batched/GPU scan return the SAME SET in DIFFERENT ORDERS, caught by driving
// both paths on a 200k-chunk corpus where a hash embedder produces many exact
// score ties. Identical scores are common in practice (duplicate passages,
// quantized vectors), so ties must resolve to something stable that does not
// depend on the execution path or the thread count.
constexpr auto hit_order = [](const Hit& a, const Hit& b) {
    if (a.score.get() != b.score.get()) return a.score.get() > b.score.get();
    return a.chunk.get() < b.chunk.get();
};

} // namespace

std::string Corpus::Explanation::summary() const {
    std::string s = "chunk " + std::to_string(chunk.get()) + ": lexical ";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f", lexical_score);
    s += buf;
    if (has_dense) {
        std::snprintf(buf, sizeof buf, "%.3f", dense_score);
        s += std::string(", dense ") + buf;
    }
    s += " (" + std::to_string(matched_terms) + "/" + std::to_string(query_terms) + " terms";
    if (!terms.empty()) {
        s += ": ";
        const std::size_t show = std::min<std::size_t>(terms.size(), 4);
        for (std::size_t i = 0; i < show; ++i) {
            if (i) s += ", ";
            std::snprintf(buf, sizeof buf, "%.2f", terms[i].contribution);
            s += terms[i].term + " " + buf;
        }
        if (terms.size() > show) s += ", …";
    }
    s += ")";
    return s;
}

Corpus::Explanation Corpus::explain(std::string_view query, ChunkId id) const {
    std::shared_lock lk(mu_);
    ensure_linked();
    Explanation e;
    e.chunk = id;

    // finalize() is a lazy read-path repair elsewhere; explain() must see the
    // same idf/avgdl the scorer would, or its numbers would not match a search.
    // Same double-checked lazy_mu_ dance as lexical_search_locked, and for the
    // same reason: this is a write under a shared lock.
    if (!bm25_.finalized()) {
        std::lock_guard lz(lazy_mu_);
        if (!bm25_.finalized()) const_cast<Corpus*>(this)->bm25_.finalize();
    }

    auto terms = bm25_.tokenizer().tokenize(query);
    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
    e.query_terms = terms.size();
    if (terms.empty()) return e;

    const std::uint32_t cid = id.get();
    if (cid >= chunks_.size()) return e;   // unknown chunk: an empty explanation

    e.terms = bm25_.explain_doc(terms, cid);
    e.matched_terms = e.terms.size();
    for (const auto& t : e.terms) e.lexical_score += t.contribution;

    // Dense half, when there is one. Reuses the stored chunk embedding rather
    // than re-embedding its text, so the number is exactly what dense_search
    // would have scored.
    if (embedder_ && !chunks_[cid].embedding.empty()) {
        if (auto qv = embedder_->embed_one(std::string(query))) {
            dense::normalize(*qv);
            e.dense_score = dense::dot(*qv, chunks_[cid].embedding);
            e.has_dense = true;
        }
    }
    return e;
}

void Corpus::ensure_packed() const {
    std::lock_guard lz(lazy_mu_);
    if (packed_valid_ && packed_epoch_ == epoch_) return;

    packed_.clear();
    packed_ids_.clear();
    packed_dim_ = 0;
    for (const auto& ch : chunks_)
        if (!ch.embedding.empty()) { packed_dim_ = ch.embedding.size(); break; }

    if (packed_dim_ != 0) {
        packed_.reserve(chunks_.size() * packed_dim_);
        packed_ids_.reserve(chunks_.size());
        for (const auto& ch : chunks_) {
            // Skip ragged rows rather than packing a matrix whose stride lies.
            // A mixed-dimension corpus is already broken, but it must not turn
            // into an out-of-bounds GPU read.
            if (ch.embedding.size() != packed_dim_) continue;
            // Skip SOFT-DELETED documents. The batch path scores this matrix
            // directly and builds hits straight from packed_ids_, so anything
            // packed here is returnable — unlike the per-query path, there is no
            // later deleted_docs_ check to catch it. Excluding at pack time is
            // also strictly cheaper than filtering afterwards: the GPU never
            // scores a row that could not have been returned. The mirror is
            // keyed on epoch_, which remove_document bumps, so a delete
            // invalidates and rebuilds it.
            if (!deleted_docs_.empty() && deleted_docs_.count(ch.doc.get())) continue;
            packed_.insert(packed_.end(), ch.embedding.begin(), ch.embedding.end());
            packed_ids_.push_back(ch.id.get());
        }
    }
    packed_epoch_ = epoch_;
    packed_valid_ = true;
}

Result<std::vector<std::vector<Hit>>>
Corpus::dense_search_batch(std::span<const std::string> queries, std::size_t k,
                           const MetaFilter& filter) const {
    std::shared_lock lk(mu_);
    // Empty in, empty out — checked BEFORE the embedder, because asking zero
    // questions is not a question a missing embedder can fail to answer.
    if (queries.empty()) return std::vector<std::vector<Hit>>{};
    if (!embedder_) return fail<std::vector<std::vector<Hit>>>(Errc::unavailable, "no embedder");
    ensure_linked();

    // The GPU path is only reachable for a plain, unfiltered, graph-less scan.
    // Everything else falls back to running the existing per-query path, which
    // is exactly what a caller would have written by hand — so this method is
    // never worse than the loop it replaces.
    const bool scan_path = !hnsw_ && !filter;
    if (scan_path) {
        ensure_packed();
        const std::size_t n   = packed_ids_.size();
        const std::size_t dim = packed_dim_;
        const std::size_t nq  = queries.size();

        if (n > 0 && dim > 0 && gpu::available() && nq * n * dim >= gpu::min_batch_work()) {
            // Embed every query first; a partial batch is not worth salvaging.
            std::vector<float> qmat;
            qmat.reserve(nq * dim);
            bool ok = true;
            for (const auto& q : queries) {
                auto qv = embedder_->embed_one(q);
                if (!qv || qv->size() != dim) { ok = false; break; }
                dense::normalize(*qv);
                qmat.insert(qmat.end(), qv->begin(), qv->end());
            }

            if (ok) {
                std::vector<float> scores(nq * n);
                if (gpu::score_batch(packed_, qmat, dim, scores)) {
                    std::vector<std::vector<Hit>> out(nq);
                    for (std::size_t q = 0; q < nq; ++q) {
                        // Partial top-k per query over that query's score row.
                        const float* row = scores.data() + q * n;
                        std::vector<Hit> hits;
                        hits.reserve(n);
                        for (std::size_t i = 0; i < n; ++i)
                            hits.push_back(Hit{ChunkId{packed_ids_[i]}, Score{row[i]}});
                        const std::size_t want = std::min(k, hits.size());
                        std::partial_sort(hits.begin(), hits.begin() + (long)want, hits.end(),
                                          hit_order);
                        hits.resize(want);
                        out[q] = std::move(hits);
                    }
                    return out;
                }
                // score_batch declining is a ROUTING answer, not an error: fall
                // through to the CPU path below rather than failing the query.
            }
        }
    }

    std::vector<std::vector<Hit>> out;
    out.reserve(queries.size());
    for (const auto& q : queries) {
        auto r = dense_search_locked(q, k, filter);
        if (!r) return std::unexpected(r.error());
        out.push_back(std::move(*r));
    }
    return out;
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

    // Build an allow-predicate over chunk id from the metadata filter AND the
    // soft-delete tombstones.
    //
    // deleted_docs_ must be consulted HERE, not left to the graph's own
    // tombstones, because HnswIndex::serialize() does NOT persist its deleted_
    // set. In memory the graph tombstone (set by remove_document) hides the
    // chunk; after save()/load() the graph comes back with no tombstones at all,
    // and this path would happily return a deleted document — a
    // delete-then-restart silently resurrected it, fully searchable, while
    // is_deleted() still reported true. The lexical and brute-force dense paths
    // already filtered on deleted_docs_; the HNSW branch was the one hole.
    //
    // deleted_docs_ IS persisted (the TOMB section), so it is the authoritative
    // source of truth on both sides of a reload. Keeping the graph tombstones
    // too is still worthwhile: they prune the walk instead of merely filtering
    // its output, so a live process does less work.
    const bool have_tombs = !deleted_docs_.empty();
    HnswIndex::AllowFn allow;
    if (filter || have_tombs) {
        allow = [this, &filter, have_tombs](std::uint32_t id) -> bool {
            const Chunk* ch = (id < chunks_.size()) ? &chunks_[id] : nullptr;
            if (!ch) return false;
            if (have_tombs && deleted_docs_.count(ch->doc.get())) return false;
            if (!filter) return true;
            if (!ch->meta) return false;
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
                      hit_order);
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
        // The chunking MODE is ingest policy too, and was missing for exactly the
        // same reason the geometry was: a corpus built with `proposition` (or
        // `semantic`) reopened as `fixed` and chunked new documents on a
        // different granularity from the ones already stored.
        m["chunking"] = cfg_.chunking == CorpusConfig::Chunking::proposition ? "proposition"
                      : cfg_.chunking == CorpusConfig::Chunking::semantic    ? "semantic"
                      : cfg_.chunking == CorpusConfig::Chunking::source      ? "source"
                                                                             : "fixed";
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
            if (m.contains("chunking")) {
                const std::string mode = m.value("chunking", std::string("fixed"));
                c.cfg_.chunking = mode == "proposition" ? CorpusConfig::Chunking::proposition
                                : mode == "semantic"    ? CorpusConfig::Chunking::semantic
                                : mode == "source"      ? CorpusConfig::Chunking::source
                                                        : CorpusConfig::Chunking::fixed;
            }
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
