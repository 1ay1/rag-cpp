// rag/index/corpus.cpp — document ingest, hybrid indexing, persistence.

#include "rag/index/corpus.hpp"
#include "rag/dense/simd.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

namespace rag::index {

using json = nlohmann::json;

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
        ch.meta = &docs_.back().meta;
        bm25_.add(cid.get(), ch.indexed_text());
        chunks_.push_back(std::move(ch));
    }
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
    if (!embedder_) return fail<std::vector<Hit>>(Errc::unavailable, "no embedder");
    auto qv = embedder_->embed_one(std::string(query));
    if (!qv) return std::unexpected(qv.error());
    dense::normalize(*qv);

    // Use HNSW if built, else brute-force cosine.
    if (hnsw_) return hnsw_->search(*qv, k);

    std::vector<Hit> hits;
    hits.reserve(chunks_.size());
    for (const auto& ch : chunks_) {
        if (ch.embedding.empty()) continue;
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
// Persistence — a JSON sidecar for documents/chunks + binary BM25/HNSW blobs.
// ─────────────────────────────────────────────────────────────────────────────
Result<void> Corpus::save(const std::string& path) const {
    json j;
    j["version"] = 1;
    j["docs"] = json::array();
    for (const auto& d : docs_) {
        json jd;
        jd["id"] = d.id.get(); jd["uri"] = d.uri; jd["title"] = d.title; jd["text"] = d.text;
        jd["meta"] = d.meta;
        j["docs"].push_back(std::move(jd));
    }
    j["chunks"] = json::array();
    for (const auto& c : chunks_) {
        json jc;
        jc["id"] = c.id.get(); jc["doc"] = c.doc.get();
        jc["text"] = c.text; jc["context"] = c.context;
        jc["start"] = c.start_line; jc["end"] = c.end_line;
        jc["embedding"] = c.embedding;
        j["chunks"].push_back(std::move(jc));
    }
    j["bm25"]  = bm25_.serialize();
    j["has_hnsw"] = hnsw_.has_value();
    if (hnsw_) j["hnsw"] = hnsw_->serialize();

    std::ofstream out(path, std::ios::binary);
    if (!out) return fail<void>(Errc::io_error, "open " + path);
    out << j.dump();
    return {};
}

Result<Corpus> Corpus::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail<Corpus>(Errc::io_error, "open " + path);
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto j = json::parse(body, nullptr, false);
    if (j.is_discarded()) return fail<Corpus>(Errc::parse_error, "corpus json");

    Corpus c;
    for (const auto& jd : j["docs"]) {
        Document d;
        d.id = DocId{jd["id"].get<std::uint32_t>()};
        d.uri = jd.value("uri", ""); d.title = jd.value("title", ""); d.text = jd.value("text", "");
        if (jd.contains("meta")) d.meta = jd["meta"].get<Metadata>();
        c.docs_.push_back(std::move(d));
    }
    for (const auto& jc : j["chunks"]) {
        Chunk ch;
        ch.id = ChunkId{jc["id"].get<std::uint32_t>()};
        ch.doc = DocId{jc["doc"].get<std::uint32_t>()};
        ch.text = jc.value("text", ""); ch.context = jc.value("context", "");
        ch.start_line = jc.value("start", 0u); ch.end_line = jc.value("end", 0u);
        if (jc.contains("embedding")) ch.embedding = jc["embedding"].get<Vector>();
        if (ch.doc.get() < c.docs_.size()) ch.meta = &c.docs_[ch.doc.get()].meta;
        c.chunks_.push_back(std::move(ch));
    }
    if (auto b = lexical::Bm25Index::deserialize(j["bm25"].get<std::string>()); b) c.bm25_ = std::move(*b);
    else return std::unexpected(b.error());
    if (j.value("has_hnsw", false)) {
        if (auto h = HnswIndex::deserialize(j["hnsw"].get<std::string>()); h) c.hnsw_ = std::move(*h);
        else return std::unexpected(h.error());
    }
    return c;
}

} // namespace rag::index
