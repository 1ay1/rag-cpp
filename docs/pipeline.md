# The Pipeline

Retrieval in rag-cpp is a **pipeline of composable stages**. A query flows
through an ordered list of `RetrievalStage`s, each transforming a `Context`
(query + running candidate set + corpus + filter). Stages are runtime-
polymorphic so a pipeline can be assembled at runtime from config, but the hot
scoring loops they call stay non-virtual inside `Corpus`.

## The Context

Every stage receives and returns a `Context`:

```cpp
struct Context {
    std::string              query;           // may be rewritten by a stage
    std::string              original_query;  // never mutated
    std::vector<Hit>         candidates;      // the running result set
    std::size_t              k = 10;          // desired final count
    const index::Corpus*     corpus = nullptr;
    index::MetaFilter        filter;          // optional metadata predicate
    std::vector<std::string> trace;           // per-stage diagnostics
};
```

A stage returns `Result<Context>` — a failed stage short-circuits the pipeline
with an `Error`, never an exception.

## Built-in stages

| Stage | What it does | Typical slot |
|-------|--------------|--------------|
| `PrfExpandStage` | RM3-lite pseudo-relevance-feedback query expansion — mines terms from top pseudo-relevant chunks and appends them to the query. | **before** retrieve |
| `HybridRetrieveStage` | Runs BM25 + dense concurrently and fuses them (see [Retrieval](retrieval.md)). | first |
| `FilterStage` | Applies the metadata predicate. | after retrieve |
| `RerankStage` | Applies a `RerankFn` (e.g. the built-in `feature_rerank`, or a cross-encoder). | after filter |
| MMR stage (`rerank::make_mmr_stage`) | Maximal Marginal Relevance — diversifies the candidate set by refusing to spend slots on near-duplicates. | after rerank, before top-k |
| `ParentStitchStage` | Small-to-big / parent-document: folds adjacent same-document fragments into their higher-ranked sibling. | after rerank, before top-k |
| `TopKStage` | Trims to the final `k`. | last |

## The three pre-assembled factories

Rather than making you wire stages by hand, three factories ship as static
methods on `Pipeline`:

### `Pipeline::standard()` — the default

```
HybridRetrieve → Filter → feature_rerank → TopK
```

This is what an `Engine` uses unless you swap it. It is deliberately narrow: it
improves *ranking accuracy* and nothing else. Anything that trades accuracy for a
different good (coverage, diversity) is **opt-in**, below.

Configure the retrieval stage per request without mutating shared state:

```cpp
auto pipe = rag::pipeline::Pipeline::standard_with(hybrid_cfg);
```

### `Pipeline::quality(λ = 0.5)` — + MMR diversity

```
HybridRetrieve → Filter → feature_rerank → MMR → TopK
```

MMR does **not** make ranking more accurate — it makes the top-k *cover more* of
the answer by refusing to spend several slots on near-duplicates of one passage.
That's a different good, measurable only with a coverage metric, which is why
it's not the default. `λ` trades relevance (1.0) against diversity (0.0); the
paper's balanced 0.5 is kept rather than a value tuned to a synthetic benchmark.
The measured coverage numbers are in the `quality()` declaration comment.

### `Pipeline::context(max_gap = 1)` — + small-to-big

```
HybridRetrieve → Filter → feature_rerank → ParentStitch → TopK
```

`ParentStitch` folds a matched chunk into a higher-ranked **adjacent** sibling of
the same document (line ranges within `max_gap`), on the theory that the sibling
already represents the content, and hands the freed top-k slot to the next
distinct location. Like MMR, the good is **coverage, not accuracy** — and it only
does anything when a query pulls a *run* of adjacent fragments of one document.
On a corpus whose documents each produce a single chunk, it folds nothing.

Measured (`bench/stitch_bench.cpp`, few long documents chunked small so one
document supplies >k matching fragments):

| pipeline | distinct documents in top-10 (of 10) |
|----------|--------------------------------------|
| `standard()` | 2 |
| `context()` | 4 |

The stage runs **after rerank, before top-k** because folding must see the
relevance order (it keeps the higher-ranked sibling) and there must still be a
pool below the cut to promote distinct locations from — top-k would have
destroyed both. The same slot-ordering argument applies to MMR.

### Why coverage stages are opt-in

`standard()` stays honest: it only claims to improve accuracy. `quality()` and
`context()` each change *what* you retrieve, not *how accurately*, so each is a
named factory carrying a measured justification at its declaration and an
explicit "why this is not the default." Turn them on when your corpus has the
redundancy (`quality`) or the fragmentation (`context`) they address — and
measure, because on a corpus without it the cost is real and the gain is zero.

## Selecting a pipeline

```cpp
rag::Engine engine;
engine.with_pipeline(rag::pipeline::Pipeline::quality(0.3f));   // diversity-leaning
// ...or...
engine.with_pipeline(rag::pipeline::Pipeline::context());       // small-to-big
```

`with_pipeline` swaps the Engine's pipeline; subsequent `search()` calls use it.

> The RCP handler assembles pipelines per request (for the fusion override) but
> does not expose `quality`/`context` as wire-level knobs — they are library
> factories. Select them in-process via `with_pipeline`.

## Writing your own stage

A stage is any type implementing `RetrievalStage`:

```cpp
class MyStage final : public rag::pipeline::RetrievalStage {
public:
    std::string_view name() const noexcept override { return "my_stage"; }

    rag::Result<rag::pipeline::Context> process(rag::pipeline::Context ctx) const override {
        // read ctx.candidates / ctx.query / *ctx.corpus, mutate ctx.candidates,
        // push a diagnostic to ctx.trace, and return it.
        ctx.trace.push_back("my_stage: touched " + std::to_string(ctx.candidates.size()));
        return ctx;
    }
};

rag::pipeline::Pipeline p;
p.add(std::make_shared<rag::pipeline::HybridRetrieveStage>())
 .add(std::make_shared<MyStage>())
 .add(std::make_shared<rag::pipeline::TopKStage>());
engine.with_pipeline(std::move(p));
```

Every stage shares the same interface, so they compose in any order. `add()` is
fluent. `examples/full_pipeline.cpp` shows a hand-assembled pipeline including a
cross-encoder rerank stage and `ParentStitch`.

## A note on tracing

Pass a `std::vector<std::string>*` as the last argument to `Pipeline::run` (or
`Engine::search`) to collect the per-stage `trace`. It's the quickest way to see
which stage changed the candidate set and by how much.
