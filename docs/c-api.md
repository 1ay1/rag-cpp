# The C API & bindings

rag-cpp exposes a flat, opaque-handle **C ABI** (`include/rag/c/rag.h`) so any
language with a C FFI can drive the engine. Errors cross as status codes, never
exceptions; strings you receive must be freed with the matching `*_free`.

## The shape

Everything is an opaque handle (`rag_engine`, `rag_embedder`, `rag_reranker`,
`rag_results`) plus functions that take one. Fallible calls return a
`rag_status`; on failure, `rag_last_error()` returns a human-readable message.

```c
#include <rag/c/rag.h>

rag_engine* e = rag_engine_new();

rag_engine_add(e, "intro.md", "rag-cpp fuses BM25 with dense vectors.", NULL, NULL);
rag_engine_build(e);

rag_results* r = NULL;
if (rag_engine_search(e, "how does hybrid search work?", 5, &r) == RAG_OK) {
    for (size_t i = 0; i < rag_results_count(r); ++i)
        printf("[%.3f] chunk=%u lines %u-%u\n",
               rag_results_score(r, i),
               rag_results_chunk_id(r, i),
               rag_results_start_line(r, i),
               rag_results_end_line(r, i));
    rag_results_free(r);
}
rag_engine_free(e);
```

## The surface

| Group | Functions |
|-------|-----------|
| Version / errors | `rag_version`, `rag_last_error` |
| Embedders | `rag_embedder_hash`, `rag_embedder_ollama`, `rag_embedder_openai`, `rag_embedder_llamacpp`, `rag_embedder_free` |
| Rerankers | `rag_reranker_tei`, `rag_reranker_cohere`, `rag_reranker_free` |
| Engine lifecycle | `rag_engine_new`, `rag_engine_free`, `rag_engine_set_embedder`, `rag_engine_set_reranker` |
| Ingest | `rag_engine_add`, `rag_engine_add_directory`, `rag_engine_build` |
| Search | `rag_engine_search` → `rag_results_*` accessors |
| Results | `rag_results_count`, `rag_results_score`, `rag_results_chunk_id`, `rag_results_doc_id`, `rag_results_start_line`, `rag_results_end_line`, `rag_results_free` |

The full signatures and ownership rules are in the header — it is short and
authoritative.

## Memory & ownership rules

- Every handle you create with a constructor must be released with its `*_free`.
- Strings returned by the library must be released with `rag_string_free()` (or
  the whole result via `rag_results_free()`).
- The ABI is exception-free: a C++ exception never crosses the boundary; failures
  surface as `rag_status` + `rag_last_error()`.

## Bindings

Because the ABI is flat and stable, high-level bindings are thin:

- **Python** — `examples/ragcpp.py` wraps the C ABI with `ctypes`. Load the shared
  library, declare the signatures once, and you have `Engine.add/build/search`.
- **Rust / Go / anything with a C FFI** — declare the handful of `extern`
  functions above against `libragcpp` and go. `examples/polyglot/` shows the
  cross-language pattern end to end.

## Building the shared library

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build          # produces build/libragcpp.{so,dylib,a}
```

Point your FFI loader at the built `libragcpp` and include `rag/c/rag.h` for the
declarations (or transcribe them into your language's FFI, as `ragcpp.py` does).

## When to use the C API vs RCP

- **C API** — you're embedding rag-cpp *in-process* in a non-C++ application and
  want direct, low-latency calls.
- **[RCP server](rcp-server.md)** — you want rag-cpp as a *separate service* other
  processes (or machines) talk to over a standard protocol, with capability
  discovery, filtering, and citations. Use the RCP client SDK in your language
  rather than the C ABI.
