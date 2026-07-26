# The CLI

The `ragcpp` binary (built to `build/cli/ragcpp`) is a thin wrapper over the
library. It has five subcommands.

```
ragcpp index <dir> <out.ragdb> [--ext=.md] [--semantic] [--proposition] [--source] [--contextual]
ragcpp query <db.ragdb> "<query>" [-k N] [--mmr] [--explain]
ragcpp serve <db.ragdb> [--http PORT] [--write] [--graph] [--memory] [--feedback] [--all]
ragcpp eval  <beir-dir> [--split=test]
ragcpp info  <db.ragdb>
ragcpp list  [embedders|rerankers] [--plugins=DIR]
```

---

## `index` — build a corpus from a directory

```sh
ragcpp index ./docs corpus.ragdb --ext=.md
```

Walks `<dir>`, chunks and indexes every matching file, and writes a single
self-contained `<out.ragdb>`. Reopening it later never rebuilds.

| Flag | Effect |
|------|--------|
| `--ext=.md` | Restrict to a file **extension**. Accepts `.md`, `md`, and `*.md` alike. **Repeats and comma-separated lists accumulate** — `--ext=.docx,.pptx` or `--ext=.docx --ext=.pptx` both work. (The old `--glob` spelling still works but only ever matched by extension.) |
| `--semantic` | Use the semantic chunker — place boundaries where the topic drifts, rather than at fixed token windows. More self-contained chunks on prose; costs a similarity pass. See [Configuration](configuration.md#chunking). |
| `--source` | Chunk **code on definition boundaries** instead of prose windows. Uses the hand-written chunker for known languages and **infers the conventions** of any language it does not know — an in-house DSL, a vendor format, anything. Measured 1.000 definition integrity vs 0.75–0.875 for windows ([BENCHMARKS.md](../BENCHMARKS.md#chunking-code-nobody-has-a-parser-for)). Safe on a mixed corpus: inference declines on prose. See [Formats and languages](formats.md). |
| `--proposition` | Index one **atomic statement** per chunk. Maximises precision-per-unit for claim-shaped queries — but it **loses on general IR**: −0.092 nDCG@10 on SciFact for a 12× larger index ([BENCHMARKS.md](../BENCHMARKS.md#chunking-strategy-measured)). Opt in only if you have measured a win on your data. |
| `--contextual` | [Contextual Retrieval](configuration.md#contextual-retrieval): situate each chunk in its document before indexing. The CLI has no LLM binding, so this uses the deterministic extractive context. Costs ~3× ingest — measure before enabling. |

### Indexing a CSV / TSV

Point `index` at a `.csv`, `.tsv`, or `.tab` file and it is ingested **row-wise**
instead of walked as a directory:

```sh
ragcpp index tickets.csv tickets.ragdb --csv-title=subject --csv-id=ticket_id
```

Each **row becomes one document**, and every column is attached as filterable
metadata. That matters: a table chunked as prose produces chunks spanning
unrelated rows, and the column structure — the most useful thing about a table —
is lost. Here it survives into the query layer, so you can filter on
`meta["status"] == "open"` rather than hoping the word "open" was retrieved.

| Flag | Effect |
|------|--------|
| `--csv-title=COL` | Column whose value becomes the document title. |
| `--csv-id=COL` | Column whose value becomes the uri suffix (default: the row number). |
| `--csv-text=A,B` | Only these columns are indexed as searchable text. All columns remain filterable metadata. Use this when the table has one prose column and several id columns — indexing opaque ids dilutes BM25's term statistics. |

The parser handles the parts of RFC 4180 that occur in practice: quoted fields,
delimiters and newlines inside quotes, and `""` as an escaped quote. A row whose
field count disagrees with the header is a **typed error**, not a silent column
shift.

## `query` — search a corpus

```sh
ragcpp query corpus.ragdb "how does hybrid retrieval work?" -k 5
```

Reopens `<db.ragdb>` and prints the top-`k` hits with scores and provenance.

| Flag | Default | Effect |
|------|---------|--------|
| `-k N` | 10 | Number of results. |
| `--mmr` | off | Use the `quality()` pipeline (MMR diversity) instead of `standard()`. See [The Pipeline](pipeline.md). |
| `--explain` | off | Print **why** each hit ranked: the query terms it matched and what each contributed to its BM25 score. The contributions sum exactly to the score shown. |

### Explaining a result

```sh
ragcpp query corpus.ragdb "hybrid retrieval fusion" -k 3 --explain
```

```
  [6.441] ./docs/retrieval.md      pipeline runs them **concurrently** — hybrid latency is
          chunk 9: lexical 6.44 (3/3 terms: fusion 4.07, hybrid 1.25, retriev 1.12)
```

Read it as: this chunk matched all three query terms, and `fusion` — the rarest,
most discriminative of them — supplied most of the score. Note `retriev`: terms
are shown **post-stemming**, which is exactly what makes a tokenizer mismatch
visible rather than mysterious.

The same data is available programmatically via `Corpus::explain(query, chunk)`,
which returns per-term contributions, IDF, term frequency, and the dense cosine
when an embedder is attached.

## `serve` — run an RCP server

```sh
ragcpp serve corpus.ragdb --all --write        # everything on, writable
ragcpp serve corpus.ragdb --http 8000          # HTTP transport
```

Serves the corpus over [RCP/1](rcp-server.md). Default transport is stdio.

| Flag | Effect |
|------|--------|
| `--http PORT` | Serve over HTTP on `PORT` instead of stdio. |
| `--write` | Open the write-ahead log so `index/add` and `index/delete` are durable (see [Persistence](persistence.md)). |
| `--graph` | Advertise and serve `graph/local` + `graph/global` (GraphRAG). |
| `--memory` | Advertise community-summary memory. |
| `--feedback` | Accept relevance feedback. |
| `--all` | Shorthand for `--memory --feedback --graph`. |

## `eval` — BEIR evaluation

```sh
ragcpp eval ./beir/scifact --split=test
```

Runs the [BEIR](https://github.com/beir-cellar/beir) zero-shot IR benchmark
against a corpus directory and reports the standard metrics. `--split` selects
the qrels split (default `test`).

## `info` — inspect a corpus

```sh
ragcpp info corpus.ragdb
```

Prints document/chunk counts, index geometry, embedder identity, and other
metadata from the `.ragdb` header — without rebuilding anything.

## `list` — show registered backends

```sh
ragcpp list                                  # all embedders + rerankers
ragcpp list embedders
ragcpp list embedders --plugins=./plugins    # include third-party .so backends
```

Lists every backend the plugin registry knows about, each with the config keys it
takes — the user-facing face of the registry's `describe()`. Answers "what can I
put in a config?" without reading source. With `--plugins=DIR` it loads every
shared library in `DIR` first, so third-party backends show up too. See
[`PLUGINS.md`](../PLUGINS.md).

---

Anything the CLI does is a few library calls; if you need a knob the CLI doesn't
expose, reach for the [C++ API](getting-started.md#path-2--the-c-library) or the
[configuration surface](configuration.md).
