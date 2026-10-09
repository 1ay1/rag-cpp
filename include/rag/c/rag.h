/* rag/c/rag.h — the STABLE C ABI for rag-cpp.
 *
 * A flat, opaque-handle C surface so Python (ctypes/cffi), Rust (bindgen/FFI),
 * Go (cgo), and any other language can drive the engine without touching C++.
 * Everything the C++ Engine exposes is reachable here; errors are returned as
 * rag_status codes and never thrown across the boundary.
 *
 * ABI stability contract:
 *   • Handles are opaque pointers; their layout is private and may change.
 *   • Enum values and struct layouts in THIS header are append-only within a
 *     major version. New fields go at the end; existing ones never move.
 *   • Every string returned by the library is heap-owned by the library and
 *     must be released with rag_string_free() (or the whole result via the
 *     matching *_free()). Input strings are borrowed for the call only.
 *
 * Threading: an Engine handle is NOT internally synchronized; use one per
 * thread or guard it externally. Search is const and may run concurrently on
 * an engine that is not being mutated.
 */
#ifndef RAG_CPP_C_API_H
#define RAG_CPP_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Versioning ─────────────────────────────────────────────────────────── */
#define RAG_C_API_VERSION_MAJOR 1
#define RAG_C_API_VERSION_MINOR 0

/* Semantic version of the linked library (not the header). */
const char* rag_version(void);

/* ── Status codes (mirror rag::Errc) ────────────────────────────────────── */
typedef enum rag_status {
    RAG_OK = 0,
    RAG_ERR_NOT_FOUND = 1,
    RAG_ERR_INVALID_ARGUMENT = 2,
    RAG_ERR_DIMENSION_MISMATCH = 3,
    RAG_ERR_IO = 4,
    RAG_ERR_PARSE = 5,
    RAG_ERR_TRANSPORT = 6,
    RAG_ERR_UNAVAILABLE = 7,
    RAG_ERR_EMPTY_CORPUS = 8,
    RAG_ERR_ALREADY_EXISTS = 9,
    RAG_ERR_CORRUPT = 10,
    RAG_ERR_UNKNOWN = 255
} rag_status;

/* Human-readable message for the last error on `eng` (borrowed, valid until
 * the next call on `eng`). The library keeps no per-thread state. */
typedef struct rag_engine rag_engine;
const char* rag_engine_last_error(const rag_engine* eng);

/* ── Opaque handles ─────────────────────────────────────────────────────── */
typedef struct rag_embedder rag_embedder;   /* an AnyEmbedder wrapper        */
typedef struct rag_reranker rag_reranker;   /* an AnyReranker wrapper        */
typedef struct rag_results  rag_results;    /* a search result set           */

/* ── Embedders ──────────────────────────────────────────────────────────── */
/* Deterministic local embedder (no network) — great default / fallback.     */
rag_embedder* rag_embedder_hash(size_t dim);
/* (Network embedders and rerankers are C++ only: rag-cpp opens no sockets,
 * and they take the host's HttpTransport.) */
void          rag_embedder_free(rag_embedder*);
void          rag_reranker_free(rag_reranker*);

/* ── Engine lifecycle ───────────────────────────────────────────────────── */
rag_engine* rag_engine_new(void);
void        rag_engine_free(rag_engine*);

/* Attach a dense embedder (consumes the embedder handle). */
rag_status  rag_engine_set_embedder(rag_engine*, rag_embedder*);
/* Attach a cross-encoder rerank stage (consumes the reranker handle).
 * top_n = how many candidates to rerank; blend in [0,1]. */
rag_status  rag_engine_set_reranker(rag_engine*, rag_reranker*,
                                    size_t top_n, float blend);

/* ── Ingest ─────────────────────────────────────────────────────────────── */
/* Add a document. meta_keys/meta_vals are parallel arrays of length meta_n
 * (may be NULL/0). Returns the assigned document id via out_doc_id. */
rag_status rag_engine_add(rag_engine*, const char* uri, const char* text,
                          const char* title,
                          const char** meta_keys, const char** meta_vals,
                          size_t meta_n, uint32_t* out_doc_id);

/* Recursively ingest a directory of files (md/txt/html/pdf/code). Returns the
 * number of documents loaded via out_count. */
rag_status rag_engine_add_directory(rag_engine*, const char* root,
                                    size_t* out_count);

/* Build indexes (embed pending chunks, build ANN over threshold). */
rag_status rag_engine_build(rag_engine*);

/* ── Search ─────────────────────────────────────────────────────────────── */
/* Run the pipeline. Returns an owned rag_results (free with rag_results_free).
 * meta_* provide an optional exact-match metadata filter (all keys must match).
 */
rag_status rag_engine_search(rag_engine*, const char* query, size_t k,
                             const char** meta_keys, const char** meta_vals,
                             size_t meta_n, rag_results** out);

size_t      rag_results_count(const rag_results*);
float       rag_results_score(const rag_results*, size_t i);
uint32_t    rag_results_chunk_id(const rag_results*, size_t i);
uint32_t    rag_results_doc_id(const rag_results*, size_t i);
uint32_t    rag_results_start_line(const rag_results*, size_t i);
uint32_t    rag_results_end_line(const rag_results*, size_t i);
/* Borrowed strings, valid until rag_results_free. */
const char* rag_results_uri(const rag_results*, size_t i);
const char* rag_results_text(const rag_results*, size_t i);
const char* rag_results_context(const rag_results*, size_t i);
void        rag_results_free(rag_results*);

/* ── Persistence (.ragdb container) ─────────────────────────────────────── */
rag_status  rag_engine_save(rag_engine*, const char* path);
/* Load a saved corpus into a fresh engine. Reattach an embedder afterward if
 * you want live query embedding. */
/* On failure returns NULL and sets *out_status; `out_message` (optional,
 * may be NULL) receives an owned message to free with rag_string_free(). */
rag_engine* rag_engine_load(const char* path, rag_status* out_status, char** out_message);

/* ── Memory ─────────────────────────────────────────────────────────────── */
void rag_string_free(char*);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RAG_CPP_C_API_H */
