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

`ragcpp_beir_bench` evaluates the **real production path** (`Pipeline::run`, i.e.
what `Engine::search` executes) and A/Bs the shipped ranking policies over one
corpus. `ragcpp eval` is a separate, simpler harness that drives
`corpus.lexical_search` directly — useful as a pure-BM25 reference point.

### Results

All figures below are **lexical-only** — no embedding model, no network, no GPU.
This is rag-cpp with zero external dependencies. Attaching a dense embedder is
expected to move these numbers further; that measurement is not yet recorded, so
it is not claimed.

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

### Context: published numbers on SciFact

| System | nDCG@10 | Source |
|--------|---------|--------|
| BM25 | 0.665 | BEIR paper (Thakur et al. 2021) |
| **rag-cpp (lexical, no model)** | **0.6809** | this repo, reproducible above |
| ColBERTv2 | 0.693 | published |
| SPLADE++ | 0.710 | published |
| BGE-base / E5-base (dense) | ~0.72–0.74 | published |

rag-cpp's model-free path beats the published BM25 baseline and lands between it
and the neural late-interaction systems — while requiring no model, no GPU, and
no network. It does **not** beat a well-tuned dense or learned-sparse retriever,
and this file will not claim otherwise until that is measured.

**Honesty note on ArguAna:** the `standard` pipeline is currently *behind* pure
lexical there (0.3628 vs 0.3720). ArguAna's task is counter-argument retrieval,
where the query is itself a long argument; term-overlap features misfire. This is
a known open gap, not a rounding error.

## Retrieval-quality fixes, measured

Three defects found by building the harness above, each with the measurement that
found it:

| Fix | Before | After |
|-----|--------|-------|
| Candidate pool was a flat 60 regardless of requested `k` | R@100 **0.9002** | **0.9160** |
| `feature_rerank` coverage weight 0.4 (a guess) | ArguAna nDCG@10 **0.3275** | **0.3628** |
| MMR selection `O(k²n)` | k=100: **2727 ms/query** | **66 ms/query** |

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

## GPU batch scoring (Metal)

dim 384, k=10, Apple M-series:

| batch | corpus | CPU | GPU | speedup |
|-------|--------|-----|-----|---------|
| 32 queries | 200k chunks | 282 ms | 45 ms | 6.2× |
| 128 queries | 200k chunks | 1154 ms | 123 ms | 9.4× |

```sh
./build/bench/ragcpp_gpu_bench
```

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
