# Serving over RCP

rag-cpp turns into a conformant **[RCP/1](https://github.com/1ay1/rcp)** server
with no glue code. RCP (Retrieval Context Protocol) is a JSON-RPC protocol for
retrieval services; rag-cpp **consumes the RCP C++ SDK** directly — it does not
reimplement the wire.

## What "consumes the SDK" means

The server front-end (`include/rag/rcp/`) is a thin framework layer over the SDK:

- The handler is bound to the SDK's `Handler` **concept** —
  `static_assert(::rcp::Handler<EngineHandler>)`. If rag-cpp's handler ever drifts
  from the shape the SDK requires, it fails to compile.
- Transports are the SDK's `rcp::serve_stdio` / `rcp::serve_http`.
- Capabilities are `::rcp::Capabilities`; errors map to `rcp::errc`.
- Metadata filters are **validated by `::rcp::filter::validate`** against the
  fields the server advertises — only filter *evaluation* is rag-cpp's own code,
  since the SDK can't know rag-cpp's metadata.

Where the SDK comes from at build time (CMake resolution order):
`-DRAGCPP_RCP_SDK_DIR` → a sibling `../rcp/sdk/cpp` if present → `FetchContent` of
the pinned tag (`RAGCPP_RCP_SDK_TAG`). CI's `pinned-sdk` job builds against the
pinned FetchContent path with no sibling present — that's the build a stranger
gets.

## Starting a server

Over stdio (one line):

```cpp
#include <rag/rag.hpp>
#include <rag/rcp/rcp.hpp>

rag::Engine engine = /* built or opened */;
rag::rcp::serve_stdio(engine);              // now an RCP/1 server on stdin/stdout
```

Over HTTP, with a fluent builder:

```cpp
rag::rcp::ServerBuilder{engine}
    .named("docs", "1.0")
    .with_index(true)          // expose index/add + index/delete
    .with_graph()              // expose graph/* (GraphRAG)
    .with_feedback()           // accept relevance feedback
    .with_memory()             // community-summary memory (HippoRAG-style)
    .serve_http(8000);
```

Or from the CLI (no code):

```sh
ragcpp serve corpus.ragdb --all --write        # everything on, writable
ragcpp serve corpus.ragdb --http 8000          # HTTP instead of stdio
```

See [The CLI](cli.md#serve) for the full flag set.

## What gets advertised

Capabilities are derived from what you turned on, so a client discovers exactly
what this server can do:

- **`retrieve`** — always. Advertises the supported `modes` (hybrid / dense /
  sparse), `max_k`, and, when enabled, the `rewrite` methods (`hyde`,
  `multi-query`) and `rerank` support.
- **`rerank`** — lists `cross-encoder` (if a reranker or rerank hook is set) and
  `colbert` (if a token embedder is set).
- **`index`** — `index/add` (upsert, idempotent by id) and `index/delete`
  (idempotent), when `with_index`.
- **`graph`** — `graph/local` and `graph/global`, when `with_graph`.
- **`memory`** — community-level memory backed by the GraphRAG community
  structure, when `with_memory`.

## Per-request options

A client can steer a single `retrieve` without touching shared server state
(which would be a data race, since the handler is used concurrently):

- **`mode`**: `"hybrid"` (default), `"dense"`, or `"sparse"`.
- **`fusion_method`**: `"rrf"`, `"weighted"` (→ RSF), or default (→ convex). See
  [Retrieval § Fusion](retrieval.md#fusion--combining-the-two-rankings).
- **`filter`**: a JSON filter tree (§8), validated against advertised fields.
- **`rewrite`**: `"hyde"` / `"multi-query"` when advertised.

Each such request runs on a **locally-built pipeline** rather than mutating the
shared `Engine`, so one client's ranking policy never leaks into another's
results.

## Conformance

The server is verified against the RCP spec's conformance suite:

```sh
python3 path/to/rcp/conformance/check.py -- ./build/cli/ragcpp serve corpus.ragdb --all --write
```

rag-cpp is certified **Level L2** (the suite reports the level and exits non-zero
if any MUST check fails). CI runs conformance against the spec cloned at the same
pinned tag on every push and weekly.

## Provenance for grounded generation

Every hit carries a **citation** — `source` (uri) plus `startLine` / `endLine` —
so a downstream generator can ground its answer and an auditor can trace every
claim back to an exact span. This is §7.7 / §14 of the spec; it's not optional
decoration, it's the point of a retrieval *protocol*.
