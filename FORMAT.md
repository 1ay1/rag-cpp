# The `.ragdb` on-disk format (v1.0)

`rag-cpp` persists a corpus to a single self-describing binary file with the
extension `.ragdb`. This document is the **stable public contract** for that
format. It is versioned; a reader rejects a file whose *major* version it does
not understand, and CRC-verifies every payload before use.

All multi-byte integers are **little-endian**. All offsets and lengths are
byte counts. Floats are IEEE-754 32-bit.

## Top-level layout

```
┌────────────────────────────────────────────────────────────┐
│ Header            (28 bytes)                                │
├────────────────────────────────────────────────────────────┤
│ Section table     (section_count × 20 bytes)               │
├────────────────────────────────────────────────────────────┤
│ Section payloads  (concatenated, addressed by the table)   │
├────────────────────────────────────────────────────────────┤
│ Trailer           (4 bytes: CRC32 over everything above)   │
└────────────────────────────────────────────────────────────┘
```

### Header (28 bytes)

| Offset | Type   | Field           | Notes |
|-------:|--------|-----------------|-------|
| 0      | u8[8]  | `magic`         | `52 41 47 44 42 00 00 00` = `"RAGDB\0\0\0"` |
| 8      | u16    | `format_major`  | `1`. Reader rejects a mismatched major. |
| 10     | u16    | `format_minor`  | `0`. Additive; older readers ignore unknown sections. |
| 12     | u32    | `flags`         | bit0 `HAS_HNSW`, bit1 `HAS_EMBEDDINGS` |
| 16     | u32    | `section_count` | number of entries in the section table |
| 20     | u64    | `reserved`      | must be `0` in v1 |

### Section table (`section_count` entries, 20 bytes each)

| Type | Field    | Notes |
|------|----------|-------|
| u32  | `tag`    | four-char code identifying the section (see below) |
| u64  | `offset` | absolute byte offset of the payload from file start |
| u64  | `length` | payload length in bytes |

Sections are addressed by tag, PNG/RIFF-style: **an unknown tag is skipped**,
so new section types can be added in a minor bump without breaking old readers.

### Trailer

The final 4 bytes are a `CRC32` (IEEE 802.3, reflected, init `0xFFFFFFFF`, final
XOR `0xFFFFFFFF`) computed over **all preceding bytes** (header + table +
payloads). Load fails with `corrupt_index` if it does not match.

## Section tags

| Tag ASCII | u32 (LE)    | Contents |
|-----------|-------------|----------|
| `META`    | `0x4154454D`| UTF-8 JSON of corpus config (`hnsw_threshold`, `embed_batch`, `bm25.{k1,b}`) |
| `DOCS`    | `0x53434F44`| document records (see below) |
| `CHNK`    | `0x4B4E4843`| chunk records, embeddings excluded |
| `EMBD`    | `0x42444D45`| chunk embeddings, parallel to `CHNK` |
| `BM25`    | `0x35324D42`| serialized inverted index (its own magic `2BM1`, versioned) |
| `HNSW`    | `0x57534E48`| serialized ANN graph (its own magic `HNW1`, versioned) |
| `TOMB`    | `0x424D4F54`| soft-deleted document ids (added in minor 1; omitted when empty) |

### `DOCS` payload

```
u32 doc_count
repeat doc_count:
    u32 id
    str uri            # str = u32 length prefix + raw bytes
    str title
    str text
    u32 meta_count
    repeat meta_count: str key, str value
```

### `CHNK` payload

```
u32 chunk_count
repeat chunk_count:
    u32 id
    u32 doc_id
    str text
    str context        # heading breadcrumb / synthesized context
    u32 start_line
    u32 end_line
```

### `EMBD` payload (present iff `HAS_EMBEDDINGS`)

```
repeat chunk_count:    # same order as CHNK
    u32 dim            # 0 ⇒ this chunk has no embedding
    f32[dim] vector    # unit-normalized
```

### `BM25` / `HNSW` payloads

Opaque, self-describing blobs produced by `Bm25Index::serialize()` /
`HnswIndex::serialize()`. Each carries its own 4-byte magic and version and is
independently CRC-independent (the container CRC covers them). Their internal
layouts are versioned separately and documented in their headers.

### `TOMB` payload (present iff at least one document is soft-deleted)

```
u32 tomb_count
repeat tomb_count:
    u32 doc_id         # ascending
```

`remove_document` is a *soft* delete: the document keeps its `DOCS` row, its
`CHNK` rows and its `BM25` postings, so every id in the file stays stable and no
other section has to be rewritten. Membership in this set is the only thing that
hides it from results, so a file without `TOMB` serves every deleted document
again. Ids are written ascending so the same corpus always produces the same
bytes.

## Compatibility policy

- **Major bump** (e.g. 1 → 2): incompatible; old readers must refuse.
- **Minor bump** (e.g. 1.0 → 1.1): additive only — new sections, or new
  trailing fields in a section. Old readers keep working by skipping unknown
  tags and stopping at the known field boundary.
- A file written by a newer minor version is safe to read with an older reader
  of the same major.

## Forward/backward guarantees

A `.ragdb` written on any platform reads on any other (endianness is fixed).
Reopening a corpus never rebuilds indexes — the BM25 postings and HNSW graph
are restored verbatim.
