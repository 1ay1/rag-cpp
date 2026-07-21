// rag/index/corpus.cpp — document ingest, hybrid indexing, persistence.

#include "rag/index/corpus.hpp"
#include "rag/dense/simd.hpp"
#include "rag/store/container.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

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
    docs_      = std::move(o.docs_);
    chunks_    = std::move(o.chunks_);
    bm25_      = std::move(o.bm25_);
    hnsw_      = std::move(o.hnsw_);
    dirty_     = o.dirty_;
    relink_meta();   // borrowed pointers now point at OUR docs_ storage
}

Result<DocId> Corpus::add_document(std::string uri, std::string text, Metadata meta, std::string title) {
    DocId did{static_cast<std::uint32_t>(docs_.size())};
    Document doc;
    doc.id = did; doc.uri = std::move(uri); doc.title = std::move(title);
    doc.text = std::move(text); doc.meta = std::move(meta);
    docs_.push_back(std::move(doc));
    const Document& stored = docs_.back();

    auto new_chunks = text::chunk_document(did, stored.text, cfg_.chunk);
    for (auto& ch : new_chunks) {
        ChunkId cid{static_cast<std::uint32_t>(chunks_.size())};
        ch.id   = cid;
        bm25_.add(cid.get(), ch.indexed_text());
        chunks_.push_back(std::move(ch));
    }
    // docs_.push_back above may have reallocated; re-link ALL chunk meta ptrs.
    relink_meta();
    bm25_.finalize();
    dirty_ = true;
    return did;
}

Result<void> Corpus::embed_pending() {
    if (!embedder_) return {};
    // Collect chunks with empty embeddings.
    std::vector<std::size_t> pending;
    for (std::size_t i = 0; i < chunks_.size(); ++i)
        if (chunks_[i].embedding.empty()) pending.push_back(i);
    if (pending.empty()) return {};

    for (std::size_t off = 0; off < pending.size(); off += cfg_.embed_batch) {
        std::size_t end = std::min(off + cfg_.embed_batch, pending.size());
        std::vector<std::string> batch;
        for (std::size_t j = off; j < end; ++j) batch.push_back(chunks_[pending[j]].indexed_text());
        auto res = embedder_->embed(batch);
        if (!res) return std::unexpected(res.error());
        auto& vecs = *res;
        for (std::size_t j = 0; j < vecs.size() && off + j < pending.size(); ++j)
            chunks_[pending[off + j]].embedding = std::move(vecs[j]);
    }
    return {};
}

Result<void> Corpus::build() {
    bm25_.finalize();
    if (embedder_) {
        if (auto r = embed_pending(); !r) return r;  // degrade: propagate, caller may ignore
        // Build HNSW past threshold.
        if (chunks_.size() >= cfg_.hnsw_threshold) {
            HnswIndex idx(cfg_.hnsw);
            for (const auto& ch : chunks_)
                if (!ch.embedding.empty()) idx.add(ch.id.get(), ch.embedding);
            hnsw_ = std::move(idx);
        }
    }
    dirty_ = false;
    return {};
}

std::vector<Hit> Corpus::lexical_search(std::string_view query, std::size_t k) const {
    return bm25_.search(query, k);
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
    if (!embedder_) return fail<std::vector<Hit>>(Errc::unavailable, "no embedder");
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

    std::vector<Hit> hits;
    hits.reserve(chunks_.size());
    for (const auto& ch : chunks_) {
        if (ch.embedding.empty()) continue;
        if (allow && !allow(ch.id.get())) continue;
        float s = dense::dot(ch.embedding, *qv);
        hits.push_back(Hit{ch.id, Score{s}});
    }
    const std::size_t kk = std::min(k, hits.size());
    std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(kk), hits.end(),
        [](const Hit& a, const Hit& b) { return a.score.get() > b.score.get(); });
    hits.resize(kk);
    return hits;
}

const Chunk* Corpus::chunk(ChunkId id) const {
    return id.get() < chunks_.size() ? &chunks_[id.get()] : nullptr;
}
const Document* Corpus::document(DocId id) const {
    return id.get() < docs_.size() ? &docs_[id.get()] : nullptr;
}

SearchResult Corpus::resolve(const Hit& h) const {
    SearchResult r;
    const Chunk* ch = chunk(h.chunk);
    if (!ch) return r;
    r.chunk = ch->id; r.doc = ch->doc; r.score = h.score;
    r.text = ch->text; r.context = ch->context;
    r.start_line = ch->start_line; r.end_line = ch->end_line;
    if (const Document* d = document(ch->doc)) r.uri = d->uri;
    return r;
}

bool Corpus::passes(ChunkId id, const MetaFilter& f) const {
    const Chunk* ch = chunk(id);
    if (!ch || !ch->meta) return !f;   // no metadata: pass only if no filter
    return f ? f(*ch->meta) : true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence — the stable, versioned .ragdb container (see rag/store + FORMAT.md).
// Sections: META (config json), DOCS, CHNK (chunk records), EMBD (embeddings),
// BM25 (inverted index), HNSW (ANN graph). CRC-verified on load.
// ─────────────────────────────────────────────────────────────────────────────
Result<void> Corpus::save(const std::string& path) const {
    store::Container c;
    std::uint32_t flags = 0;

    // META — corpus config (round-trips the knobs that affect query behaviour).
    {
        json m;
        m["hnsw_threshold"] = cfg_.hnsw_threshold;
        m["embed_batch"]    = cfg_.embed_batch;
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

    c.set_flags(flags);
    return c.write_file(path);
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
    return c;
}

} // namespace rag::index
