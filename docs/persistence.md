# Persistence

A corpus lives in a single **`.ragdb`** file: a stable, versioned, CRC-checked
container. Everything round-trips — BM25 postings, HNSW graph, dense vectors,
metadata, config geometry — so **reopening never rebuilds**.

The byte-level layout is specified in [`FORMAT.md`](../FORMAT.md). This guide
covers how to use it and the durability model.

## Saving and opening

```cpp
rag::Engine engine;
engine.add("intro.md", "...");
engine.build();
engine.save("corpus.ragdb");                 // Result<void>

auto reopened = rag::Engine::open("corpus.ragdb");   // Result<Engine>
if (reopened) for (auto& r : *reopened->search("query", 5)) { /* ... */ }
```

At the corpus level the same is `index::Corpus::save(path)` /
`index::Corpus::load(path)`.

## The container format

A `.ragdb` is self-describing:

- **magic** `"RAGDB\0\0\0"` — identifies the file.
- **version** (`kFormatVersion`) — readers reject an unknown *major* version
  rather than misparsing.
- **CRC32** (IEEE 802.3) trailer over the payload — load verifies magic, major
  version, and CRC before handing any section back, so a truncated or corrupted
  file fails with a typed `Error`, never a silent garbage read.
- **sections by tag** — each index is its own tagged section.

Config geometry is part of what's persisted: a corpus built at
`chunk.max_lines = 3` reopens chunking new documents the same way, rather than
silently reverting to the default. (This was a real bug once — the chunk geometry
wasn't round-tripped — now it is.)

## The write-ahead log (WAL)

Rebuilding and rewriting the whole snapshot on every acknowledged write is
`O(corpus)` and gets slower as the corpus grows. The WAL fixes that: a durable
append log alongside the `.ragdb`.

```cpp
auto corpus = rag::index::Corpus::load("corpus.ragdb");
corpus->open_wal("corpus.ragdb.wal");        // opening IS recovery — replays the log
// ... adds/deletes are appended durably ...
corpus->checkpoint("corpus.ragdb");          // fold the log into a fresh snapshot
```

- **`open_wal(path)`** — opens (creating if absent) the log. If the log has
  entries from a previous run, opening **replays** them: opening is recovery.
- **`wal_bytes()`** — the current log size; the signal for "time to checkpoint".
- **`checkpoint(path)`** — writes a fully durable new snapshot, then (and only
  then) truncates the log. The ordering matters: the snapshot must contain
  everything the log did before the log is discarded, or truncation would destroy
  acknowledged writes.

An acknowledged `index/add` goes from a full snapshot rewrite (tens of ms,
growing with corpus size) to a flat ~1.3 ms append with the WAL enabled.

Over the CLI / RCP server, `--write` opens the WAL automatically, and opening it
performs recovery — so a server restart after a crash comes back with every
acknowledged write intact.

## When you need durability

| Scenario | What to do |
|----------|-----------|
| Read-only serving of a fixed corpus | Just `open()` — no WAL needed. |
| A server that accepts `index/add` / `index/delete` | Run with `--write` (CLI) or `open_wal()` (library). Checkpoint periodically when `wal_bytes()` grows. |
| Batch rebuild from source | `add()` everything, `build()`, `save()` once. No WAL. |

See [`FORMAT.md`](../FORMAT.md) for the exact on-disk contract and version rules.
