#pragma once
// rag/loaders/extract.hpp — one place that turns any file into text.
//
// Formats fall into three groups, and the difference is not cosmetic:
//
//   1. WE OWN IT. Plain text, Markdown, HTML, CSV, source code, and the OOXML
//      family (.docx/.xlsx/.pptx) are extracted in-process with no dependency
//      at all -- see ooxml.hpp for why that is worth doing rather than linking
//      a stack of libraries.
//
//   2. AN EXTERNAL TOOL OWNS IT. Legacy binary Office (.doc/.xls/.ppt) and
//      PDF are extracted by a converter program (pdftotext, antiword, …) if
//      one is installed. rag-cpp starts no processes: the host runs the
//      converter through Extractors::run. These formats are genuinely large -- the .doc OLE compound
//      binary format is a filesystem inside a file, and a half-working parser
//      for it is worse than none, because it produces plausible garbage that
//      silently poisons an index.
//
//   3. IT NEEDS A MODEL. A scanned page is an image; there is no text to
//      extract, and the only way through is OCR. rag-cpp does not bundle an
//      OCR engine and will not pretend to: it detects the case and reports it.
//
// The design rule across all three: NEVER return plausible-looking garbage.
// A caller can handle "unavailable, install X" -- that is a fixable, legible
// condition. A caller cannot handle an index quietly full of mojibake, because
// nothing surfaces the problem until retrieval has been bad for a month.
//
// Extractors (below) is the extension point: a value the caller builds and
// passes to load_file / load_directory. Add a handler for any extension
// (including one nobody has heard of) and every loader path that gets the
// value picks it up.

#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::loaders {

// How a format's text was (or would be) obtained.
enum class ExtractKind {
    native,       // in-process, no dependency
    external,     // needs a converter binary on PATH
    ocr,          // needs an OCR engine (not bundled)
    unknown,      // no handler; treated as plain text
};

struct ExtractResult {
    std::string text;
    ExtractKind kind = ExtractKind::native;
    std::string tool;      // which external tool produced it, when applicable
};

// Runs a converter program and returns its stdout, or nullopt only if it
// could not start (not installed). A non-zero exit still returns what it
// printed. argv[0] is the program name. The host implements it (agentty: its
// sandboxed exec); rag-cpp never spawns.
using RunFn = std::function<std::optional<std::string>(const std::vector<std::string>& argv)>;

using ExtractFn = std::function<Result<std::string>(std::string_view bytes)>;

// How files become text: the custom extractors and the way to run converter
// programs. Default-constructed, it handles every in-process format and
// reports the external ones as unavailable.
struct Extractors {
    RunFn run;   // empty: no converters

    // A handler for `extension` (".acmerpt"). Registering an extension that
    // already has a handler REPLACES it, which lets a caller override the
    // built-in path for a format with a better one.
    void add(std::string extension, ExtractFn fn, std::string description = {});

    // Names of every custom extractor, for diagnostics.
    [[nodiscard]] std::vector<std::string> custom() const;

    struct Custom { ExtractFn fn; std::string desc; };
    std::map<std::string, Custom, std::less<>> customs;
};

// Extract text from bytes already in memory. `hint` is a filename or extension
// used only to pick a handler; content sniffing wins when the two disagree,
// because a renamed file is common and a lying extension should not corrupt an
// index. Pass "" when you have no hint at all.
[[nodiscard]] Result<ExtractResult> extract_bytes(std::string_view bytes, std::string_view hint = {},
                                                  const Extractors& ex = {});

// Extract text from a path. Uses extract_bytes for formats we own, and runs
// the external converter (through ex.run) for those we do not.
[[nodiscard]] Result<ExtractResult> extract_file(const std::filesystem::path& path,
                                                 const Extractors& ex = {});

// ─── Capability reporting ─────────────────────────────────────────────────────
//
// "Does this build actually support .doc?" is a question worth being able to
// answer before ingesting ten thousand files and finding out. `capabilities()`
// asks ex.run for each converter and reports honestly.
struct FormatSupport {
    std::string extension;
    ExtractKind kind = ExtractKind::unknown;
    bool available = false;        // true if it would work right now
    std::string requires_tool;     // "" for native formats
    std::string note;
};
[[nodiscard]] std::vector<FormatSupport> capabilities(const Extractors& ex = {});

} // namespace rag::loaders
