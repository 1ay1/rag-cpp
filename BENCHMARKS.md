# Benchmarks

Every number here is reproducible from this repository. Claims without a
reproduction command do not belong in this file.

## Retrieval quality — BEIR

[BEIR](https://github.com/beir-cellar/beir) (Thakur et al., NeurIPS 2021) is the
standard zero-shot IR benchmark. `nDCG@10` is the headline metric the literature
reports.

### Reproduce

```sh
# Get a dataset (SciFact: 5183 docs, 300 judged test queries)
curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip

cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bench/ragcpp_beir_bench ./scifact --split=test
```

That run is **lexical-only**. To measure the engine as it actually ships — with
the dense half live and hybrid fusion doing real work — attach an embedding
model (see [Hybrid retrieval](#hybrid-retrieval-with-a-neural-embedder) below):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRAGCPP_WITH_ONNX=ON && cmake --build build -j
./build/bench/ragcpp_beir_bench ./scifact --split=test \
    --model=minilm/model.onnx --tokenizer=minilm/tokenizer.json
```

`ragcpp_beir_bench` evaluates the **real production path** (`Pipeline::run`, i.e.
what `Engine::search` executes) and A/Bs the shipped ranking policies over one
corpus. `ragcpp eval` is a separate, simpler harness that drives
`corpus.lexical_search` directly — useful as a pure-BM25 reference point.

### Results

All figures in this table are **lexical-only** — no embedding model, no network,
no GPU. This is rag-cpp with zero external dependencies. The hybrid numbers, with
a real model attached, are in the [next section](#hybrid-retrieval-with-a-neural-embedder).

| Dataset | policy | nDCG@10 | R@10 | R@100 | MAP | MRR |
|---------|--------|---------|------|-------|-----|-----|
| SciFact  | `lexical`  | 0.6800 | 0.8212 | 0.9160 | 0.6346 | 0.6434 |
| SciFact  | `standard` | **0.6809** | 0.8212 | 0.9160 | 0.6354 | 0.6447 |
| SciFact  | `context`  | 0.6809 | 0.8212 | 0.9160 | 0.6354 | 0.6447 |
| SciFact  | `quality`  | 0.6744 | 0.8069 | 0.9109 | 0.6324 | 0.6424 |
| NFCorpus | `lexical`  | 0.3263 | 0.1510 | 0.2482 | 0.1449 | 0.5280 |
| NFCorpus | `standard` | 0.3261 | 0.1510 | 0.2485 | 0.1448 | 0.5281 |
| ArguAna  | `lexical`  | **0.3720** | 0.7724 | 0.9666 | 0.2558 | 0.2558 |
| ArguAna  | `standard` | 0.3628 | 0.7582 | 0.9651 | 0.2492 | 0.2492 |

`quality` (MMR) and `context` (ParentStitch) optimise **coverage, not accuracy** —
they are expected to cost a little nDCG and are opt-in for exactly that reason.
See [docs/pipeline.md](docs/pipeline.md).

### Hybrid retrieval with a neural embedder

The table above is the floor, not the engine. With an embedding model attached,
the dense retriever goes live and `standard` performs the hybrid fusion it was
built for. Measured with **`all-MiniLM-L6-v2`** (a 22 M-parameter model from
2021 — deliberately the *small* choice, running in-process on CPU via
`-DRAGCPP_WITH_ONNX=ON`, no GPU, no network):

| Dataset | retrieval | nDCG@10 | R@10 | R@100 | MAP | MRR |
|---------|-----------|---------|------|-------|-----|-----|
| SciFact  | `lexical` (BM25 only)  | 0.6800 | 0.8212 | 0.9160 | 0.6346 | 0.6434 |
| SciFact  | `dense` (MiniLM only)  | 0.6518 | 0.7872 | 0.9239 | 0.6103 | 0.6171 |
| SciFact  | **`standard` (hybrid)** | **0.7347** | **0.8489** | **0.9572** | **0.6973** | **0.7091** |
| NFCorpus | `lexical` (BM25 only)  | 0.3263 | 0.1510 | 0.2482 | 0.1449 | 0.5280 |
| NFCorpus | `dense` (MiniLM only)  | 0.3179 | 0.1585 | 0.3127 | 0.1434 | 0.5130 |
| NFCorpus | **`standard` (hybrid)** | **0.3602** | **0.1719** | **0.3160** | **0.1700** | **0.5728** |
| ArguAna  | `lexical` (BM25 only)  | 0.3720 | 0.7724 | 0.9666 | 0.2558 | 0.2558 |
| ArguAna  | `dense` (MiniLM only)  | 0.3614 | 0.7489 | 0.9723 | 0.2516 | 0.2516 |
| ArguAna  | **`standard` (hybrid)** | **0.3848** | **0.7902** | **0.9829** | **0.2678** | **0.2678** |

The result that matters is not that hybrid wins — it is that **hybrid beats both
of its own halves**, on all three datasets, by more than either half beats the
other:

| | SciFact | NFCorpus | ArguAna |
|---|---------|----------|---------|
| hybrid − best single retriever | **+0.055** | **+0.034** | **+0.013** |
| lexical − dense (the gap being fused) | +0.028 | +0.008 | +0.011 |

That is fusion contributing signal, not a better retriever being picked. A
weighted-sum or take-the-max fusion would land *between* its inputs; landing
above both is what reciprocal-rank fusion over decorrelated rankers is supposed
to do, and it is why the engine ships hybrid as `standard`.

Reproduce (models are ~86 MB, one `curl`; see
[docs/embedders.md](docs/embedders.md#running-a-real-model-in-process)):

```sh
./build/bench/ragcpp_beir_bench ./scifact --split=test \
    --model=minilm/model.onnx --tokenizer=minilm/tokenizer.json
```

### Context: published numbers on SciFact

| System | nDCG@10 | Source |
|--------|---------|--------|
| BM25 | 0.665 | BEIR paper (Thakur et al. 2021) |
| **rag-cpp (lexical, no model)** | **0.6809** | this repo, reproducible above |
| ColBERTv2 | 0.693 | published |
| SPLADE++ | 0.710 | published |
| BGE-base / E5-base (dense) | ~0.72–0.74 | published |
| **rag-cpp (hybrid, MiniLM-L6)** | **0.7347** | this repo, reproducible above |

Two separate claims, both measured:

1. rag-cpp's **model-free** path beats the published BM25 baseline and lands
   between it and the neural late-interaction systems — while requiring no
   model, no GPU, and no network.
2. rag-cpp's **hybrid** path, using a 2021 22 M-parameter embedder that scores
   only 0.6518 on its own, reaches 0.7347 — above ColBERTv2, above SPLADE++, and
   at the top of the BGE/E5 band. The fusion is doing the work, not the model.

The honest caveat on (2): these are single-dataset comparisons against numbers
from papers, not a re-run of those systems in this harness, and MiniLM-L6 is a
small model — a `bge-base` or `e5-base` embedder in the same slot should go
higher still. It is not measured here, so it is not claimed here.

**ArguAna, previously a known open gap, is closed by this.** With no model the
`standard` pipeline was *behind* pure lexical there (0.3628 vs 0.3720) — ArguAna
asks for counter-argument retrieval, where the query is itself a long argument
and term-overlap features misfire. Fusing a semantic retriever in is exactly the
missing signal: 0.3848, ahead of lexical for the first time. The remaining
weirdness on that dataset is `nDCG@1 ≈ 0`, which is inherent to the task — the
gold document is a *rebuttal*, so it is never the nearest neighbour of the query.

**One thing got worse, and it is worth stating:** `quality` (MMR) collapses on
ArguAna once a real embedder is attached — nDCG@10 **0.0380** against `standard`'s
0.3848, with R@100 still 0.8378. The candidates are there; MMR throws them away.
The mechanism is that ArguAna documents are near-duplicate arguments on the same
motion, so a diversity penalty computed in a real semantic space suppresses the
entire relevant cluster — the one dataset where "show me something different" is
precisely the wrong instruction. MMR stays opt-in and this is why. Use
`standard`, or Dartboard (below), when relevance is the objective.

## Retrieval-quality fixes, measured

Three defects found by building the harness above, each with the measurement that
found it:

| Fix | Before | After |
|-----|--------|-------|
| Candidate pool was a flat 60 regardless of requested `k` | R@100 **0.9002** | **0.9160** |
| `feature_rerank` coverage weight 0.4 (a guess) | ArguAna nDCG@10 **0.3275** | **0.3628** |
| MMR selection `O(k²n)` | k=100: **2727 ms/query** | **66 ms/query** |

## Chunking code nobody has a parser for

The usual way to "support every language" is a grammar per language. That has a
hard ceiling: it cannot help with an in-house DSL, a vendor format with no
public spec, or a dialect invented last quarter — which is precisely the code
most worth searching, because none of it is on the public internet.

rag-cpp infers the structure from the file itself (see
[docs/formats.md](docs/formats.md) for the scoring model). The metric is
**definition integrity**: the fraction of definitions that land intact —
signature *and* body — inside a single chunk. It is the right target because a
split definition is the specific failure that makes code retrieval useless: the
chunk carries the name without the logic, or the logic with no name to match a
query against. It also cannot be gamed by chunking more or less aggressively.

The baseline is fixed-size line windows, which is what every language-blind
system falls back to. Definition sizes are deliberately varied (3–17 lines) —
with uniform sizes a 40-line window lands on a boundary by luck and the
baseline column is a gift rather than a comparison.

| Corpus | strategy | integrity | windows |
|--------|----------|-----------|---------|
| python | known_language | **1.000** | 0.850 |
| go | known_language | **1.000** | 0.750 |
| rust | known_language | **1.000** | 0.800 |
| cpp | known_language | **1.000** | 0.850 |
| ruby | known_language | **1.000** | 0.850 |
| dsl:config (`@service`) | inferred | **1.000** | 0.875 |
| dsl:4gl (`PROCEDURE`) | inferred | **1.000** | 0.850 |
| dsl:rules (`rule`) | inferred | **1.000** | 0.800 |
| dsl:hdl (`MODULE`) | inferred | **1.000** | 0.800 |
| dsl:build (`target`) | inferred | **1.000** | 0.875 |

The first five rows are a control: where a hand-written per-language chunker
exists it is used, so inference is not being credited for them. **The last five
are the claim** — languages that exist nowhere on the internet, with no parser,
no grammar, and no file extension, chunked correctly on their own conventions
the first time they are seen.

The negative controls matter more than the positives, because the dangerous
failure is not missing structure but **hallucinating** it — prose chunked as if
it were code is strictly worse than the prose chunker:

| Corpus | confidence | outcome |
|--------|-----------|---------|
| prose (an annual report) | 0.00 | declined |
| flat key-value config | 0.00 | declined |

The prose row was found the hard way and is why the model has a nesting gate.
Blank-line-separated paragraphs beat every other term: each paragraph is one
long line at column 0, so a word that opens several of them ("Revenue",
"However") is repeated, perfectly aligned, spread across the whole file, and
announced by the blank line above it. It scored **0.75 confidence** and would
have chunked an annual report on the word "Revenue". The one thing it cannot do
is open a body — prose has no bodies — which is the property that actually
distinguishes a definition from a sentence.

Building the bench also surfaced a real bug in the existing per-language
chunker: it cut at the definition line, which files each doc comment with the
**previous** definition. Go, Rust and C++ scored **0.000** integrity before the
fix, because every definition was missing its header — and a doc comment is
usually the most query-matchable text a definition has.

```sh
./build/bench/ragcpp_structure_bench
```

## Chunking strategy, measured

`CorpusConfig::Chunking` offers `fixed` (structural), `semantic` (topic drift),
and `proposition` (one atomic statement per chunk). Popular RAG guides recommend
proposition chunking; on general IR benchmarks it **loses**, and by a lot.
Measured with `Pipeline::standard()`:

| Chunking | SciFact nDCG@10 | chunks | NFCorpus nDCG@10 | chunks |
|----------|-----------------|--------|------------------|--------|
| **`fixed`** (default) | **0.6809** | 5183 | **0.3261** | 3633 |
| `semantic` | 0.6703 | 8866 | 0.3233 | 5195 |
| `proposition` | 0.5887 | 62624 | 0.2895 | 56341 |

That is **−0.092 / −0.037 nDCG@10 for a 12× larger index**. The mechanism is
straightforward: BM25 scores a one-sentence chunk on almost no term evidence, and
splitting a document into twelve pieces splits its term statistics with it.

`proposition` is still shipped and reachable (`--proposition`, or
`CorpusConfig::chunking`) because precision-per-unit is genuinely what
claim-verification workloads want — but it is opt-in, it is not the default, and
these numbers are why.

## Diversity policies: Dartboard vs MMR

Both diversify a result set; they optimise different things. MMR subtracts a
pairwise penalty ("is this unlike what I already picked?"); Dartboard
([Pickett et al. 2024](https://arxiv.org/abs/2407.12101)) maximises relevant
information gain ("does this cover something nothing else covers?").

Measured on SciFact, same corpus and candidate pool, only the selection policy
differs:

| Policy | nDCG@10 | R@10 | MAP |
|--------|---------|------|-----|
| `standard` (no diversification) | **0.6809** | **0.8212** | 0.6354 |
| `quality` (MMR, λ=0.5) | 0.6744 | 0.8069 | 0.6324 |
| Dartboard (`relevance_weight`=0.7) | **0.6798** | **0.8212** | 0.6344 |

Diversification always costs something on a benchmark that rewards pure
relevance — but **Dartboard costs about 6× less than MMR** (−0.0011 vs −0.0065
nDCG@10) and, unlike MMR, does not lose any Recall@10. If you need diversity,
prefer Dartboard.

### Dartboard complexity

The gain of adding a document is a sum over every intent, so the naive greedy
form recomputes all `n²` similarities on each of `k` steps. Precomputing the
symmetric similarity matrix once reduces selection to array lookups:

| pool (k) | before | after | speedup |
|----------|--------|-------|---------|
| 40 (k=10) | 24.68 ms | 2.39 ms | 10× |
| 100 (k=25) | 369.09 ms | 11.32 ms | 33× |
| 200 (k=50) | **2803.39 ms** | **38.68 ms** | **72×** |

The same trap MMR fell into, in a different shape — caching a running max is not
enough when the objective sums over intents rather than taking a max over the
chosen set.

### The coverage-weight sweep

`feature_rerank` blends `(1-w)·fusion + w·term_coverage`. The shipped default was
`w = 0.4`, never measured. Sweeping across three datasets (nDCG@10):

| w | SciFact | NFCorpus | ArguAna | mean |
|---|---------|----------|---------|------|
| 0.00 | 0.6800 | **0.3266** | **0.3720** | **0.4595** |
| **0.10** | 0.6809 | 0.3261 | 0.3628 | 0.4566 |
| 0.20 | **0.6836** | 0.3254 | 0.3508 | 0.4533 |
| 0.30 | 0.6798 | 0.3236 | 0.3398 | 0.4477 |
| 0.40 | 0.6751 | 0.3221 | 0.3275 | 0.4416 |
| 0.50 | 0.6681 | 0.3190 | 0.3093 | 0.4321 |

Coverage helps only on SciFact and monotonically hurts elsewhere, because
counting distinct query terms discards the IDF weighting BM25 computed. Tuning to
SciFact's 0.20 peak would be overfitting to one dataset; the default is now
`0.10`, which keeps a tie-breaking signal for degenerate score distributions at
~0.003 mean nDCG cost.

### MMR complexity

Greedy MMR recomputed each candidate's similarity to every already-selected
document: `O(k²n)` similarity calls, and with no embedder each call is a Jaccard
intersection over token bags. Caching the running max-similarity (only the
just-chosen document can raise it) makes it `O(kn)`.

Measured on SciFact (5183 docs, lexical):

| pool (k) | before | after | speedup |
|----------|--------|-------|---------|
| 40 (k=10) | 5.38 ms | 2.42 ms | 2.2× |
| 100 (k=25) | 42.42 ms | 6.08 ms | 7.0× |
| 200 (k=50) | 332.45 ms | 19.53 ms | 17× |
| 400 (k=100) | **2727.03 ms** | **66.05 ms** | **41×** |

The output is bit-identical — greedy selection is unchanged, so this is a pure
optimisation (gated by `mmr_cached_selection_matches_the_naive_greedy`).

## Engine throughput

Apple M-series, Release, single thread unless noted.

| Operation | Corpus | Measurement |
|-----------|--------|-------------|
| Index + build | 5183 docs (SciFact) | 202 ms |
| `lexical_search` (depth 100) | 5183 docs | 0.01 ms/query |
| `lexical_search` (depth 400) | 5183 docs | 0.02 ms/query |
| `Pipeline::standard().run` k=10 | 5183 docs | 0.02 ms/query |
| `Pipeline::standard().run` k=100 | 5183 docs | 0.06 ms/query |

```sh
./build/bench/ragcpp_bench 20000        # ablation + latency
```

## ANN quality — HNSW

Measured on standard ANN datasets, recall@10 vs exact cosine, `M=16`,
`ef_construction=200`, single thread:

| Dataset | ef | recall@10 | µs/query | QPS |
|---------|-----|-----------|----------|-----|
| GloVe-25-angular (1.18M) | 16 | 0.856 | 31.6 | 31.6k |
| GloVe-25-angular | 64 (default) | 0.973 | 91.8 | 10.9k |
| GloVe-25-angular | 256 | 0.999 | 266.7 | 3.7k |
| SIFT1M | 64 (default) | 0.965 | 124.6 | 8.0k |
| SIFT1M | 512 | 0.999 | 679.0 | 1.5k |

337 bytes/vector on GloVe; build 73 s for 1.18M vectors (8 threads).

```sh
./build/bench/ragcpp_ann_bench <base.fvecs> [query.fvecs] [limit]
```

## GPU batch scoring

Two interchangeable backends: **Metal** (Apple) and **OpenCL** (NVIDIA, AMD,
Intel, and Apple as a fallback). Selection is automatic; `RAGCPP_GPU_BACKEND`
overrides it.

dim 384, k=10, Apple M-series:

| batch | corpus | CPU | GPU | speedup |
|-------|--------|-----|-----|---------|
| 32 queries | 200k chunks | 282 ms | 45 ms | 6.2× |
| 128 queries | 200k chunks | 1154 ms | 123 ms | 9.4× |

```sh
./build/bench/ragcpp_gpu_bench
```

### Backend comparison (M1, 200k × 384, 64 queries)

| Backend | Time | vs CPU | Max error vs CPU |
|---------|------|--------|------------------|
| CPU (8 threads) | 482 ms | 1.0× | — |
| **Metal** | **41 ms** | **11.6×** | 8.9e-08 |
| OpenCL | 692 ms | 0.7× | 2.4e-07 |

Both backends are numerically equivalent to the CPU: **zero** of 12.8M scores
differ by more than `1e-4`.

The OpenCL wall time on Apple is **driver overhead, not the kernel**. With
`CL_QUEUE_PROFILING_ENABLE`:

| Component | Time |
|-----------|------|
| Kernel execution (device timestamps) | **16.1 ms** — faster than Metal |
| Readback of the 48 MB result | 4.3 ms |
| Wall time of the same dispatch | **672 ms** — Apple's deprecated ICD |

Reproducible on every dispatch and unchanged across work-group sizes 16–256. This
is why Metal is preferred on Apple; on NVIDIA/AMD/Intel ICDs the overhead is
absent.

## Durability

| Operation | Before WAL | With WAL |
|-----------|-----------|----------|
| Acknowledged `index/add` (20k corpus) | 25.1 ms | **1.35 ms** |
| Acknowledged `index/add` (50k corpus) | 69.7 ms | 1.35 ms (flat) |

## Methodology

- Every number is produced by a checked-in bench target, runnable by anyone.
- Quality claims are measured on **multiple** datasets; a win on one dataset is
  treated as overfitting until it replicates.
- Performance claims compare against the code that **actually ships**, not a
  strawman.
- Every fix is gated by a test validated by reverting the fix.
