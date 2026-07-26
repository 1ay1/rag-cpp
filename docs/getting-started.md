# Getting Started

This guide takes you from an empty checkout to a working search — twice: once
from the command line, once from C++.

## Prerequisites

- A **C++23** compiler. GCC 14+ or a recent Clang. (Homebrew `g++-15` and Apple
  Clang are both used in CI.)
- **CMake ≥ 3.24** (for `FetchContent` of the one required dependency).
- That single dependency, **nlohmann/json**, is fetched automatically — you do
  not install anything by hand.

Everything else — embedding backends, ONNX, llama.cpp, the Metal GPU path — is
**optional and off unless you ask for it**. With nothing enabled the engine
still does full hybrid retrieval using its built-in hash embedder.

## Build

```sh
git clone https://github.com/1ay1/rag-cpp
cd rag-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces:

- `build/libragcpp.*` — the library
- `build/cli/ragcpp` — the CLI
- `build/tests/ragcpp_tests` — the test suite (run it: `ctest --test-dir build`)

### Build options

All default to a sensible value; flip them at configure time with `-D`.

| Option | Default | Effect |
|--------|---------|--------|
| `RAGCPP_BUILD_TESTS` | `ON` | Build the test suite. |
| `RAGCPP_BUILD_BENCH` | `ON` | Build the benchmark harness (`build/bench/*`). |
| `RAGCPP_BUILD_EXAMPLES` | `ON` | Build the programs in `examples/`. |
| `RAGCPP_BUILD_CLI` | `ON` | Build the `ragcpp` CLI. |
| `RAGCPP_USE_SIMD` | `ON` | Enable the SIMD distance kernels (AVX2 / NEON). |
| `RAGCPP_WITH_RCP` | `ON` | Build the RCP/1 server front-end. See [Serving over RCP](rcp-server.md). |
| `RAGCPP_WITH_METAL` | `ON` on Apple | The Metal GPU batch-scoring backend. See [GPU](gpu.md). |
| `RAGCPP_WITH_ONNX` | `OFF` | In-process ONNX Runtime embedder (needs ONNX Runtime installed). |
| `RAGCPP_WITH_LLAMA` | `OFF` | In-process GGUF embedder via llama.cpp (needs llama.cpp). |
| `RAGCPP_WERROR` | `OFF` | Treat warnings as errors. |

## Path 1 — the CLI

Index a directory of Markdown, then query it:

```sh
./build/cli/ragcpp index ./docs corpus.ragdb --ext=.md
./build/cli/ragcpp query corpus.ragdb "how does hybrid retrieval work?" -k 5
```

`index` walks the directory, chunks and indexes every matching file, and writes
a single self-contained `corpus.ragdb`. `query` reopens it — reopening never
rebuilds — and prints the top-`k` hits. The full command reference is in
[The CLI](cli.md).

## Path 2 — the C++ library

```cpp
#include <rag/rag.hpp>

int main() {
    rag::Engine engine;

    engine.add("intro.md", "rag-cpp fuses BM25 with dense vector search.");
    engine.add("hnsw.md",  "HNSW gives sub-linear approximate nearest neighbours.");
    engine.build();

    auto hits = engine.search("how does vector search work?", 5);
    if (!hits) { /* hits.error() is a closed Errc sum type */ return 1; }

    for (const auto& r : *hits)
        std::printf("[%.3f] %s\n", r.score.get(), r.uri.c_str());
}
```

Link against the `ragcpp` target (see [Consuming from CMake](#consuming-from-cmake)).

### The Engine facade

`rag::Engine` is the one-call surface over the whole library. The methods you'll
use most:

| Method | Purpose |
|--------|---------|
| `add(uri, text, meta = {}, title = {})` | Ingest a document. Returns `Result<DocId>`. |
| `build()` | Build BM25 + (optionally) HNSW + dense vectors. |
| `search(query, k = 10, filter = {}, trace = nullptr)` | Ranked `Result<vector<SearchResult>>`. |
| `with_embedder(AnyEmbedder)` | Attach a dense backend → enables hybrid. Fluent. |
| `with_embedder_spec(json)` | Attach a backend *by name* from a JSON spec (config-driven). |
| `with_pipeline(Pipeline)` | Swap the retrieval pipeline (see [The Pipeline](pipeline.md)). |
| `save(path)` / `Engine::open(path)` | Persist / reopen a `.ragdb`. |
| `graph_local` / `graph_global` | GraphRAG search (see [Advanced Retrieval](advanced-retrieval.md)). |

Every fallible call returns `Result<T> = std::expected<T, rag::Error>`. Check it
before dereferencing — there are no exceptions on the expected-failure path.

### Adding a real embedder

The default engine uses a deterministic **hash embedder** so hybrid retrieval
works with zero setup. For real semantic search, attach a backend:

```cpp
#include <rag/rag.hpp>

rag::Engine engine;
engine.with_embedder(rag::dense::AnyEmbedder{
    rag::dense::OllamaEmbedder{{.model = "nomic-embed-text"}}});
```

Or wire it from config with no backend knowledge at the call site:

```cpp
engine.with_embedder_spec({{"type", "ollama"}, {"model", "nomic-embed-text"}});
```

The full list of backends, the retry/fallback wrappers, and the in-process ONNX
/ GGUF paths are in [Embedders](embedders.md).

## Consuming from CMake

Installed:

```cmake
find_package(ragcpp CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE ragcpp::ragcpp)
```

```sh
cmake --install build --prefix /usr/local
```

Or vendor it directly with `FetchContent` / `add_subdirectory` — the nlohmann/json
dependency is resolved either way.

## Where to go next

- Understand what happens inside `search()` → [Retrieval](retrieval.md)
- Change *how* results are assembled → [The Pipeline](pipeline.md)
- Turn the engine into a server → [Serving over RCP](rcp-server.md)
- The big picture → [`ARCHITECTURE.md`](../ARCHITECTURE.md)
