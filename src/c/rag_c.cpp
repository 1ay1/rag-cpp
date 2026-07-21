// src/c/rag_c.cpp — the C ABI implementation over the C++ Engine.

#include "rag/c/rag.h"

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "rag/rag.hpp"
#include "rag/loaders/loaders.hpp"
#include "rag/loaders/code_chunker.hpp"
#include "rag/rerank/reranker.hpp"

namespace {

// Per-thread last-error message.
thread_local std::string g_last_error;

rag_status map_errc(rag::Errc e) {
    using E = rag::Errc;
    switch (e) {
        case E::ok:                 return RAG_OK;
        case E::not_found:          return RAG_ERR_NOT_FOUND;
        case E::invalid_argument:   return RAG_ERR_INVALID_ARGUMENT;
        case E::dimension_mismatch: return RAG_ERR_DIMENSION_MISMATCH;
        case E::io_error:           return RAG_ERR_IO;
        case E::parse_error:        return RAG_ERR_PARSE;
        case E::transport_error:    return RAG_ERR_TRANSPORT;
        case E::unavailable:        return RAG_ERR_UNAVAILABLE;
        case E::empty_corpus:       return RAG_ERR_EMPTY_CORPUS;
        case E::already_exists:     return RAG_ERR_ALREADY_EXISTS;
        case E::corrupt_index:      return RAG_ERR_CORRUPT;
    }
    return RAG_ERR_UNKNOWN;
}

template <class T>
rag_status set_err(const rag::Result<T>& r) {
    if (r) { g_last_error.clear(); return RAG_OK; }
    g_last_error = r.error().message;
    return map_errc(r.error().code);
}
rag_status set_err_void(const rag::Result<void>& r) {
    if (r) { g_last_error.clear(); return RAG_OK; }
    g_last_error = r.error().message;
    return map_errc(r.error().code);
}

rag::Metadata make_meta(const char** keys, const char** vals, size_t n) {
    rag::Metadata m;
    for (size_t i = 0; i < n; ++i)
        if (keys[i] && vals[i]) m[keys[i]] = vals[i];
    return m;
}

} // namespace

// Opaque wrappers.
struct rag_engine   { rag::Engine eng; };
struct rag_embedder { std::optional<rag::dense::AnyEmbedder> e; };
struct rag_reranker { std::optional<rag::rerank::AnyReranker> r; };
struct rag_results  { std::vector<rag::SearchResult> items; };

extern "C" {

const char* rag_version(void) { return "0.1.0"; }
const char* rag_last_error(void) { return g_last_error.c_str(); }

// ── Embedders ────────────────────────────────────────────────────────────────
rag_embedder* rag_embedder_hash(size_t dim) {
    auto* h = new (std::nothrow) rag_embedder;
    if (h) h->e.emplace(rag::dense::HashEmbedder{dim ? dim : 256});
    return h;
}
rag_embedder* rag_embedder_ollama(const char* host, uint16_t port, const char* model, size_t dim) {
    auto* h = new (std::nothrow) rag_embedder;
    if (!h) return nullptr;
    rag::dense::OllamaConfig c;
    if (host) c.host = host;
    if (port) c.port = port;
    if (model) c.model = model;
    if (dim) c.dim = dim;
    h->e.emplace(rag::dense::OllamaEmbedder{std::move(c)});
    return h;
}
rag_embedder* rag_embedder_openai(const char* host, uint16_t port, int tls,
                                  const char* path, const char* model,
                                  const char* api_key, size_t dim) {
    auto* h = new (std::nothrow) rag_embedder;
    if (!h) return nullptr;
    rag::dense::OpenAIConfig c;
    if (host) c.host = host;
    if (port) c.port = port;
    c.tls = tls != 0;
    if (path) c.path = path;
    if (model) c.model = model;
    if (api_key) c.api_key = api_key;
    if (dim) c.dim = dim;
    h->e.emplace(rag::dense::OpenAIEmbedder{std::move(c)});
    return h;
}
rag_embedder* rag_embedder_llamacpp(const char* host, uint16_t port, size_t dim) {
    auto* h = new (std::nothrow) rag_embedder;
    if (!h) return nullptr;
    rag::dense::LlamaCppConfig c;
    if (host) c.host = host;
    if (port) c.port = port;
    if (dim) c.dim = dim;
    h->e.emplace(rag::dense::LlamaCppEmbedder{std::move(c)});
    return h;
}
void rag_embedder_free(rag_embedder* h) { delete h; }

// ── Rerankers ────────────────────────────────────────────────────────────────
rag_reranker* rag_reranker_tei(const char* host, uint16_t port) {
    auto* h = new (std::nothrow) rag_reranker;
    if (!h) return nullptr;
    h->r.emplace(rag::rerank::CrossEncoderReranker{
        rag::rerank::CrossEncoderConfig::tei(host ? host : "127.0.0.1", port ? port : 8080)});
    return h;
}
rag_reranker* rag_reranker_cohere(const char* host, uint16_t port, int tls,
                                  const char* model, const char* api_key) {
    auto* h = new (std::nothrow) rag_reranker;
    if (!h) return nullptr;
    rag::rerank::CrossEncoderConfig c;
    if (host) c.host = host;
    if (port) c.port = port;
    c.tls = tls != 0;
    c.path = "/v1/rerank";
    c.wire = rag::rerank::CrossEncoderConfig::Wire::cohere;
    if (model) c.model = model;
    if (api_key) c.api_key = api_key;
    h->r.emplace(rag::rerank::CrossEncoderReranker{std::move(c)});
    return h;
}
void rag_reranker_free(rag_reranker* h) { delete h; }

// ── Engine ───────────────────────────────────────────────────────────────────
rag_engine* rag_engine_new(void) { return new (std::nothrow) rag_engine; }
void rag_engine_free(rag_engine* h) { delete h; }

rag_status rag_engine_set_embedder(rag_engine* h, rag_embedder* e) {
    if (!h || !e || !e->e) return RAG_ERR_INVALID_ARGUMENT;
    h->eng.with_embedder(*e->e);
    rag_embedder_free(e);
    return RAG_OK;
}

rag_status rag_engine_set_reranker(rag_engine* h, rag_reranker* r, size_t top_n, float blend) {
    if (!h || !r || !r->r) return RAG_ERR_INVALID_ARGUMENT;
    // Rebuild the standard pipeline with the rerank stage inserted before top-k.
    rag::pipeline::Pipeline p;
    p.add(std::make_shared<rag::pipeline::HybridRetrieveStage>())
     .add(std::make_shared<rag::pipeline::FilterStage>())
     .add(rag::rerank::make_rerank_stage(*r->r, top_n ? top_n : 50, blend))
     .add(std::make_shared<rag::pipeline::TopKStage>());
    h->eng.with_pipeline(std::move(p));
    rag_reranker_free(r);
    return RAG_OK;
}

rag_status rag_engine_add(rag_engine* h, const char* uri, const char* text, const char* title,
                          const char** mk, const char** mv, size_t mn, uint32_t* out_id) {
    if (!h || !uri || !text) return RAG_ERR_INVALID_ARGUMENT;
    auto r = h->eng.add(uri, text, make_meta(mk, mv, mn), title ? title : "");
    if (!r) return set_err(r);
    if (out_id) *out_id = r->get();
    return RAG_OK;
}

rag_status rag_engine_add_directory(rag_engine* h, const char* root, size_t* out_count) {
    if (!h || !root) return RAG_ERR_INVALID_ARGUMENT;
    auto docs = rag::loaders::load_directory(root);
    if (!docs) return set_err(docs);
    size_t n = 0;
    for (auto& d : *docs) {
        auto r = h->eng.add(d.uri, d.text, d.meta, d.title);
        if (r) ++n;
    }
    if (out_count) *out_count = n;
    return RAG_OK;
}

rag_status rag_engine_build(rag_engine* h) {
    if (!h) return RAG_ERR_INVALID_ARGUMENT;
    return set_err_void(h->eng.build());
}

rag_status rag_engine_search(rag_engine* h, const char* query, size_t k,
                             const char** mk, const char** mv, size_t mn, rag_results** out) {
    if (!h || !query || !out) return RAG_ERR_INVALID_ARGUMENT;
    rag::index::MetaFilter filter;
    if (mn > 0) {
        auto want = make_meta(mk, mv, mn);
        filter = [want](const rag::Metadata& m) {
            for (const auto& [k, v] : want) {
                auto it = m.find(k);
                if (it == m.end() || it->second != v) return false;
            }
            return true;
        };
    }
    auto res = h->eng.search(query, k ? k : 10, std::move(filter));
    if (!res) return set_err(res);
    auto* r = new (std::nothrow) rag_results;
    if (!r) return RAG_ERR_UNKNOWN;
    r->items = std::move(*res);
    *out = r;
    return RAG_OK;
}

size_t   rag_results_count(const rag_results* r) { return r ? r->items.size() : 0; }
float    rag_results_score(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].score.get() : 0.0f; }
uint32_t rag_results_chunk_id(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].chunk.get() : 0; }
uint32_t rag_results_doc_id(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].doc.get() : 0; }
uint32_t rag_results_start_line(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].start_line : 0; }
uint32_t rag_results_end_line(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].end_line : 0; }
const char* rag_results_uri(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].uri.c_str() : ""; }
const char* rag_results_text(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].text.c_str() : ""; }
const char* rag_results_context(const rag_results* r, size_t i) { return (r && i < r->items.size()) ? r->items[i].context.c_str() : ""; }
void rag_results_free(rag_results* r) { delete r; }

rag_status rag_engine_save(rag_engine* h, const char* path) {
    if (!h || !path) return RAG_ERR_INVALID_ARGUMENT;
    return set_err_void(h->eng.save(path));
}

rag_engine* rag_engine_load(const char* path, rag_status* out_status) {
    if (!path) { if (out_status) *out_status = RAG_ERR_INVALID_ARGUMENT; return nullptr; }
    auto corpus = rag::index::Corpus::load(path);
    if (!corpus) {
        g_last_error = corpus.error().message;
        if (out_status) *out_status = map_errc(corpus.error().code);
        return nullptr;
    }
    auto* h = new (std::nothrow) rag_engine;
    if (!h) { if (out_status) *out_status = RAG_ERR_UNKNOWN; return nullptr; }
    h->eng.corpus() = std::move(*corpus);
    if (out_status) *out_status = RAG_OK;
    return h;
}

void rag_string_free(char* s) { std::free(s); }

} // extern "C"
