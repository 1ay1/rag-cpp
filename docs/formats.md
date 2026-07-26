# Formats and languages

Two questions this page answers:

1. **My code is in a language nobody has a parser for.** An in-house DSL, a
   vendor format, a fourth-generation language, something a colleague invented
   last quarter. Can rag-cpp index it usefully?
2. **My documents are Word, Excel, PowerPoint, and scanned PDFs.** Can it read
   those without dragging in a stack of libraries?

Yes to both, with one honest exception (scanned pages need OCR, and rag-cpp
tells you so rather than pretending).

## Languages nobody has a parser for

The usual approach to "support every language" is a grammar per language:
tree-sitter and friends. That approach has a hard ceiling. It cannot help with
the code that is most valuable to search — the code that is internal to one
company, that no model has memorized and no parser has ever been written for.

rag-cpp takes the other route. A source file is not arbitrary text: it is
written by people following a convention, and a convention repeated a few
hundred times leaves statistical fingerprints that are visible **without
knowing what the language means**. `infer_structure()` reads a file once and
scores every candidate leading token on four signals:

- **repeated** — a token seen once is a coincidence; seen twenty times it is a
  convention.
- **aligned** — its occurrences sit at *one* indentation column. This is the
  strongest signal and the reason the whole thing works: `if` and `return`
  scatter across every depth of a file, while `def` / `func` / `SECTION` are
  pinned to one.
- **anchoring** — the line after it is more indented (it opens a body), or the
  line before it is blank or a comment (it is announced).
- **spread** — its occurrences run the length of the file. This is what demotes
  `import` blocks, which are repeated and perfectly aligned but confined to the
  top.

Multiplying these gives a score that is near zero unless a token is doing all
four things, which in practice only definition keywords do.

```cpp
auto profile = rag::loaders::infer_structure(file_contents);
if (profile.usable()) {
    // profile.definition_tokens — e.g. {"@service", "rule", "MODULE"}
    // profile.comment_prefix    — e.g. ";" or "--"
    // profile.confidence        — [0,1]
}
```

You rarely call that directly. `chunk_source()` is the front door and picks the
best available strategy:

| Order | Strategy | When |
|-------|----------|------|
| 1 | hand-written per-language chunker | the extension names a known language |
| 2 | inferred structure | the file has structure worth trusting |
| 3 | size-bounded windows | it does not |

```cpp
auto chunks = rag::loaders::chunk_source(doc_id, ".svc", body);
```

### Turning it on

```sh
ragcpp index ./repo out.ragdb --source
```

```cpp
rag::index::CorpusConfig cfg;
cfg.chunking = rag::index::CorpusConfig::Chunking::source;
```

The language comes from each document's `ext` metadata (the directory loader
sets it) or from its URI. Safe on a mixed corpus of code and documentation:
inference **declines** on prose, and the fixed chunker runs instead.

### Measured

`bench/structure_bench.cpp` measures **definition integrity** — the fraction of
definitions that land intact, signature *and* body, inside a single chunk. A
split definition is the specific failure that makes code RAG useless: the chunk
has the name but not the logic, or the logic with no name to match against.

Five real languages as a control, five **invented** languages that exist nowhere
on the internet, against the fixed windows every language-blind system falls
back to:

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

And the negative controls, which matter more than the positives — the dangerous
failure is not missing structure, it is **hallucinating** it:

| Corpus | outcome |
|--------|---------|
| prose (an annual report) | declined, confidence 0.00 |
| flat key-value config | declined, confidence 0.00 |

```sh
./build/bench/ragcpp_structure_bench
```

Integrity is necessary but not sufficient, so it is also measured **end to end**:
`bench/code_retrieval_bench.cpp` retrieves over 517 real files and 4005 queries
from CodeSearchNet, and `--source` chunking beats the window fallback on every
metric — **MRR@10 0.8193 vs 0.7507**, **Recall@1 0.7328 vs 0.6517** — in a 34%
smaller index. Details in
[BENCHMARKS.md](../BENCHMARKS.md#does-integrity-translate-to-retrieval).

## Office documents

`.docx`, `.xlsx` and `.pptx` are ZIP archives of XML. So are OpenDocument
(`.odt`, `.ods`, `.odp` — LibreOffice / OpenOffice) and `.epub`. rag-cpp reads
every one of them **in-process with no dependency at all** — it owns a ZIP
central-directory reader and a DEFLATE decompressor (RFC 1951, fixed and dynamic
Huffman), and one XML-to-text pass that recognises both the OOXML (`w:`/`a:`) and
the OpenDocument (`text:`/`table:`) element names. The argument for owning rather
than linking: a loader that pulls in three transitive dependencies is one nobody
enables in a hardened build, and it becomes dead code.

```cpp
auto text = rag::loaders::zip_document_to_text(bytes); // sniffs OOXML/ODF/EPUB
auto doc  = rag::loaders::docx_to_text(bytes);         // or be explicit
auto odf  = rag::loaders::odf_to_text(bytes);
auto book = rag::loaders::epub_to_text(bytes);
```

What is preserved, because retrieval actually uses it:

- **Word / OpenDocument text** — paragraphs, headings, line breaks, footnotes.
  Table **rows stay on one line** (cells tab-separated), because a row is one
  record and exploding a 2×2 table into four lines destroys the association
  between label and value. Tracked deletions and field instructions are dropped.
- **Excel / OpenDocument spreadsheet** — the **shared-string table is resolved**.
  Excel interns repeated strings and references them by index, so a naive
  tag-stripper extracts a grid of integers, confidently and wrongly. Each row
  becomes a tab-separated line under a sheet heading.
- **PowerPoint / OpenDocument presentation** — slides in **presentation order**
  (not archive order), each under a `## Slide N` heading so "what was on slide
  12" is answerable, plus speaker notes, which are often where the substance is.
- **EPUB** — chapters in **spine reading order** (from the OPF package, not
  archive order), each XHTML content document stripped to text.

Entities are decoded, including numeric references, emitted as UTF-8 so
documents in any script survive. `.rtf` is also read in-process — control words
stripped, `\'hh` codepage bytes and `\uN` escapes decoded to UTF-8, font and
style tables skipped — with `textutil`/`unrtf` kept only as a fallback.

**Encodings.** A file that begins with a Unicode byte-order mark (UTF-8, or
UTF-16 LE/BE — what Windows Notepad and Excel CSV export write) is decoded to
UTF-8 rather than rejected. Without this, a UTF-16 file's first NUL byte gets it
thrown out as "binary", which is how half of Windows-authored corpora silently
fail to index.

## Everything else

```cpp
auto r = rag::loaders::extract_file(path);
// r->text, r->kind (native | external | ocr), r->tool
```

| Group | Formats | How |
|-------|---------|-----|
| We own it | txt, md, html, csv, source code, docx, xlsx, pptx, **odt, ods, odp, epub, rtf**, UTF-16/BOM text | in-process, no dependency |
| A tool owns it | pdf, doc, xls, ppt | external converter if installed |
| Needs a model | scanned PDFs | detected and reported; **no OCR bundled** |

The legacy binary formats (`.doc` is an OLE compound file — a filesystem inside
a file, `.xls`/`.ppt` likewise) are delegated to `antiword` / `catdoc` /
`textutil` / `xls2csv` / `catppt`, whichever is present. A half-working in-house
parser for them would be worse than none, because it produces plausible garbage
that silently poisons an index.

`capabilities()` reports what this machine can actually do, before you ingest
ten thousand files and find out afterwards:

```cpp
for (const auto& f : rag::loaders::capabilities())
    std::printf("%-16s %s  %s\n", f.extension.c_str(),
                f.available ? "yes" : "no ", f.note.c_str());
```

### The design rule

**Never return plausible-looking garbage.** A caller can handle
`unavailable: install poppler-utils` — that is a legible, fixable condition. A
caller cannot handle an index quietly full of mojibake, because nothing surfaces
the problem until retrieval has been bad for a month. So binary content with no
handler is an error, a ZIP that is not an office document is an error, and a
scanned PDF says so by name:

```
PDF appears to be scanned images with no text layer (312 chars from 4820 KB):
OCR required, e.g. `ocrmypdf in.pdf out.pdf` then re-ingest
```

## Formats internal to one company

The escape hatch. A proprietary report container, an instrument's binary log, a
message archive with an in-house schema — register an extractor once and every
ingest path picks it up: `load_file`, `load_directory`, and the CLI.

```cpp
rag::loaders::register_extractor(".acmerpt", [](std::string_view bytes) {
    return rag::Result<std::string>{parse_acme_report(bytes)};
}, "ACME internal report");
```

Registering an extension that already has a handler **replaces** it, which is
deliberate: it lets you override the built-in PDF path with a better one.

Combined with inferred chunking, that is the complete story for internal code
and internal formats: your extractor turns bytes into text, and inference finds
the structure in it — neither step needing anyone outside your company to have
ever seen the format.
