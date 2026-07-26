# Advanced Retrieval

Beyond hybrid BM25 + dense, rag-cpp implements a family of research-grade
retrieval techniques. Each is a self-contained module reachable from the library
(and several from the [RCP server](rcp-server.md)). All share the totality
contract: they return `Result<T>` and degrade gracefully when an optional
dependency (an LLM, a token embedder) is absent.

Every technique below cites its paper in the [README references](../README.md#references).

## SPLADE — learned sparse retrieval

`rag::sparse::SpladeIndex` — an inverted index over **learned** term weights with
term expansion, so a query for "car" can match a passage that only says
"automobile". Model-agnostic: you inject a `SparseEncoder`; without one it uses a
model-free default suitable for tests and demos.

```cpp
auto splade = rag::sparse::SpladeIndex::build(corpus /*, cfg, encoder */);
auto hits   = splade->search("electric vehicle range", 10);
```

Persists to its own versioned blob (magic `"SPL1"`); reopening never rebuilds.

## ColBERT — late interaction (MaxSim)

Bi-encoders compress a whole passage to one vector and lose token detail;
cross-encoders score every query-passage pair jointly and are slow. **ColBERT**
sits between: it embeds each token independently, then scores a pair by summing,
over each query token, its maximum similarity to any document token (**MaxSim**).

The MaxSim scorer itself (`rag::late::`) is exact and model-agnostic — you inject
a `TokenEmbedder`. A built-in fallback embedder scores exact-match query tokens
at 1.0, which is enough for tests and as a drop-in when no token model is
available.

## RAPTOR — hierarchical summary tree

Some questions need the "big picture" a single chunk can't hold.
`rag::raptor::RaptorTree` builds, bottom-up, a **tree of increasingly abstract
summaries** by clustering chunks and summarizing each cluster, recursively.
Retrieval can then match against the abstraction level the question needs.

```cpp
auto tree = rag::raptor::RaptorTree::build(corpus, cfg, my_summarizer);
```

The `Summarizer` is an injected `std::function` (typically an LLM call); the tree
structure and traversal are the library's.

## HyDE and multi-query

Both live in `rag::query::` and take an injected `Generator` (an LLM). Both fall
back to a **normal dense search** when no generator is supplied — the
graceful-degradation contract.

- **`hyde_search`** — generates a *hypothetical* answer document, embeds *that*,
  and searches with it. A hypothetical answer is closer in vector space to real
  answers than the question is, so zero-shot dense recall improves.
- **`multi_query_search`** — generates several paraphrases of the query, searches
  each, and fuses the results, covering more of the ways an answer might be
  phrased.

```cpp
auto hits = rag::query::hyde_search(corpus, "why did latency regress?", 10, generator);
```

Both **batch** their generated hypotheticals/paraphrases into a single embedding
call, and on Apple hardware that batch is scored on the [GPU](gpu.md).

## Corrective RAG (CRAG)

`rag::crag::` evaluates whether retrieved passages are actually relevant, and
**acts** on the verdict — `correct`, `ambiguous`, or `incorrect`. A model-free
default judges relevance from the top retrieval score's separation from the pack
combined with lexical query-overlap; you can inject a real relevance model and an
`ExternalSource` (CRAG's web-search fallback) to replace irrelevant passages with
freshly fetched ones.

## The cascade

`rag::cascade::Cascade` is how production search actually scales: a cheap, wide
first stage followed by progressively more expensive, more accurate re-scorers
applied to progressively smaller candidate sets. Each stage has a **budget** (how
many candidates it may score), so total cost is bounded no matter how large the
corpus. A `StageTrace` records what each stage saw and emitted.

## GraphRAG

The engine can build a document graph and answer two kinds of graph query
directly:

```cpp
auto local  = engine.graph_local("what depends on the auth service?", 10);
auto global = engine.graph_global("summarize the system's failure modes", 10);
```

- **`graph_local`** — walks the neighbourhood of the entities in the query
  (node-level answers).
- **`graph_global`** — answers over community summaries (community-level answers),
  for questions that span the whole corpus.

The graph (`rag::graph::DocGraph`) is built lazily on first use and cached.

## Choosing

- Exact terms / rare words matter → BM25 (always on) or **SPLADE** for expansion.
- Token-level precision on reranking → **ColBERT**.
- Big-picture / multi-hop questions → **RAPTOR** or **GraphRAG**.
- The query is under-specified → **HyDE** / **multi-query**.
- You need to detect and fix bad retrievals → **CRAG**.
- You need bounded cost at scale → the **cascade**.

Most of these are also selectable over the wire — see
[Serving over RCP](rcp-server.md) — and demonstrated in `examples/advanced.cpp`
and `examples/graphrag.cpp`.
