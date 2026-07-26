// bench/structure_bench.cpp — does inferred chunking actually find definitions?
//
// The claim under test is uncomfortable: that rag-cpp can chunk a language it
// has never seen, as well as a hand-written per-language chunker chunks one it
// has. A claim like that is worth nothing unless it is measured against a
// falsifiable target, so this bench uses a metric that cannot be gamed by
// chunking more or less aggressively:
//
//   DEFINITION INTEGRITY — of all the definitions in a file (known, because
//   this bench SYNTHESIZES the files and therefore knows where every one
//   starts), what fraction land intact inside a single chunk, with their body,
//   rather than being split across a chunk boundary?
//
// Splitting a function in half is the specific failure that makes code RAG
// useless: the retrieved chunk has the signature but not the logic, or the
// logic but no name to match the query against. Integrity is the thing to
// maximize, and windows-based chunking is bad at it in a way that shows up
// immediately at this metric.
//
// The corpora:
//   * five real languages, as a control — inference should roughly match the
//     hand-written chunkers, since those encode real knowledge
//   * five INVENTED languages that exist nowhere on the internet and that no
//     parser exists for, which is the actual claim: a config DSL, a mainframe
//     4GL, a rules engine, a hardware description dialect, a build language
//
// Run: ./build/bench/ragcpp_structure_bench

#include <cstdio>
#include <string>
#include <vector>

#include "rag/loaders/structure.hpp"

using rag::loaders::ChunkStrategy;
using rag::loaders::CodeChunkOptions;

namespace {

struct Corpus {
    std::string name;
    std::string ext;                    // "" for the invented languages
    std::string body;
    std::vector<std::size_t> def_lines; // ground truth: 0-based definition starts
};

// Build a file by repeating a template, recording where each definition begins.
Corpus synth(std::string name, std::string ext, std::size_t n,
             const std::string& header,
             const std::string& (*unit)(std::size_t)) {
    Corpus c;
    c.name = std::move(name);
    c.ext = std::move(ext);
    c.body = header;
    std::size_t line = 0;
    for (char ch : header) if (ch == '\n') ++line;
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& u = unit(i);
        c.def_lines.push_back(line);
        for (char ch : u) if (ch == '\n') ++line;
        c.body += u;
    }
    return c;
}

// Real definitions are not all the same length, and a bench where they are is
// a bench that flatters fixed windows: uniform 8-line units mean a 40-line
// window lands on a boundary every fifth definition by luck. `filler(i, n)`
// gives each unit a different body size (3..17 lines), which is what makes the
// windows column an honest baseline rather than a gift.
std::string filler(std::size_t i, const char* fmt_line) {
    std::string out;
    const std::size_t n = 3 + (i * 7 + i * i) % 15;
    for (std::size_t k = 0; k < n; ++k) {
        out += fmt_line;
        out += std::to_string(k);
        out += "\n";
    }
    return out;
}

// Ground-truth body size we require to be intact: the signature plus a few
// lines. Kept small so the metric measures "did the definition survive", not
// "is the chunker generous".

// ─── Real languages (control) ────────────────────────────────────────

const std::string& py_unit(std::size_t i) {
    static std::string s;
    s = "def handler_" + std::to_string(i) + "(request, context):\n"
        "    \"\"\"Handle request variant " + std::to_string(i) + ".\"\"\"\n"
        "    payload = request.get('payload')\n"
        "    if payload is None:\n"
        "        raise ValueError('missing payload')\n" +
        filler(i, "    step = transform(payload, ") +
        "    return {'status': 'ok', 'result': step}\n\n";
    return s;
}

const std::string& go_unit(std::size_t i) {
    static std::string s;
    s = "// Process" + std::to_string(i) + " handles variant " + std::to_string(i) + ".\n"
        "func Process" + std::to_string(i) + "(ctx context.Context, in *Input) (*Output, error) {\n"
        "\tif in == nil {\n"
        "\t\treturn nil, errors.New(\"nil input\")\n"
        "\t}\n" +
        filler(i, "\tout.Add(in.Field") +
        "\treturn out, nil\n"
        "}\n\n";
    return s;
}

const std::string& rs_unit(std::size_t i) {
    static std::string s;
    s = "/// Compute stage " + std::to_string(i) + ".\n"
        "pub fn stage_" + std::to_string(i) + "(input: &Frame) -> Result<Frame, Error> {\n"
        "    let mut out = Frame::new(input.width, input.height);\n"
        "    for px in input.pixels() {\n"
        "        out.push(px.saturating_add(1));\n"
        "    }\n" +
        filler(i, "    out.tag(") +
        "    Ok(out)\n"
        "}\n\n";
    return s;
}

const std::string& cpp_unit(std::size_t i) {
    static std::string s;
    s = "// Apply filter " + std::to_string(i) + ".\n"
        "void apply_filter_" + std::to_string(i) + "(Buffer& buf, const Params& p) {\n"
        "    const std::size_t n = buf.size();\n"
        "    for (std::size_t k = 0; k < n; ++k) {\n"
        "        buf[k] = clamp(buf[k] * p.gain, p.lo, p.hi);\n"
        "    }\n" +
        filler(i, "    buf.mark(") +
        "}\n\n";
    return s;
}

const std::string& rb_unit(std::size_t i) {
    static std::string s;
    s = "def compute_" + std::to_string(i) + "(records)\n"
        "  total = 0\n"
        "  records.each do |r|\n"
        "    total += r.value if r.valid?\n"
        "  end\n" +
        filler(i, "  audit total, ") +
        "  total\n"
        "end\n\n";
    return s;
}

// ─── Invented languages (the actual claim) ────────────────────────────────────
// None of these exist. No parser, no grammar, no training data, no extension.

// 1. An in-house service-config DSL with sigil-marked definitions.
const std::string& dsl_config_unit(std::size_t i) {
    static std::string s;
    s = "@service payments_" + std::to_string(i) + "\n"
        "  ; routes traffic for shard " + std::to_string(i) + "\n"
        "  bind    tcp://0.0.0.0:" + std::to_string(9000 + i) + "\n"
        "  timeout 30s\n"
        "  retries 3 backoff=exponential\n"
        "  health  /healthz interval=5s\n" +
        filler(i, "  depends upstream_service_") +
        "\n";
    return s;
}

// 2. A mainframe-flavoured 4GL: uppercase keywords, terminator lines.
const std::string& dsl_4gl_unit(std::size_t i) {
    static std::string s;
    s = "PROCEDURE DIVISION-" + std::to_string(i) + " USING WS-RECORD.\n"
        "    MOVE ZEROS TO WS-COUNTER-" + std::to_string(i) + ".\n"
        "    PERFORM VARYING WS-IDX FROM 1 BY 1 UNTIL WS-IDX > 100\n"
        "        ADD WS-AMOUNT TO WS-TOTAL\n"
        "    END-PERFORM.\n" +
        filler(i, "    ADD 1 TO WS-TALLY-") +
        "    DISPLAY 'BATCH " + std::to_string(i) + " COMPLETE'.\n"
        "    EXIT PROCEDURE.\n\n";
    return s;
}

// 3. A rules engine with a bespoke when/then shape.
const std::string& dsl_rules_unit(std::size_t i) {
    static std::string s;
    s = "rule \"escalate_tier_" + std::to_string(i) + "\"\n"
        "  -- fires when the account crosses tier " + std::to_string(i) + "\n"
        "  salience " + std::to_string(100 - static_cast<int>(i)) + "\n"
        "  when\n"
        "    $a : Account( balance > " + std::to_string(1000 * (i + 1)) + " )\n"
        "    $h : History( account == $a, breaches > 2 )\n"
        "  then\n"
        "    escalate($a, TIER_" + std::to_string(i) + ");\n" +
        filler(i, "    audit.log($a.id, ") +
        "\n";
    return s;
}

// 4. A hardware-description dialect with block terminators.
const std::string& dsl_hdl_unit(std::size_t i) {
    static std::string s;
    s = "MODULE alu_slice_" + std::to_string(i) + " (clk, rst, a, b, y)\n"
        "  INPUT  clk, rst\n"
        "  INPUT  [31:0] a, b\n"
        "  OUTPUT [31:0] y\n"
        "  ALWAYS @posedge clk\n"
        "    IF rst THEN y <= 0\n"
        "    ELSE y <= a + b + " + std::to_string(i) + "\n" +
        filler(i, "    ASSIGN tmp <= stage_") +
        "  ENDALWAYS\n"
        "ENDMODULE\n\n";
    return s;
}

// 5. An internal build/pipeline language with indentation nesting.
const std::string& dsl_build_unit(std::size_t i) {
    static std::string s;
    s = "target lib_component_" + std::to_string(i) + ":\n"
        "  # builds component " + std::to_string(i) + " for all platforms\n"
        "  sources  src/component_" + std::to_string(i) + "/**.cx\n"
        "  requires runtime>=4.2, codec_core\n"
        "  compile  --opt=3 --lto --arch=native\n" +
        filler(i, "  artifact out/component_") +
        "  publish  registry://internal/components\n\n";
    return s;
}

// ─── Metric ───────────────────────────────────────────────────────────────────

struct Score {
    double integrity = 0.0;      // fraction of definitions intact in one chunk
    std::size_t chunks = 0;
    double avg_lines = 0.0;
    ChunkStrategy strategy = ChunkStrategy::windows;
};

// A definition is INTACT if some chunk contains its first line and at least the
// next `body_lines` lines that follow it — i.e. the signature and its body
// arrived together.
Score evaluate(const Corpus& c, const std::vector<rag::Chunk>& chunks,
               std::size_t body_lines) {
    Score s;
    s.chunks = chunks.size();
    if (chunks.empty()) return s;

    std::size_t total_lines = 0;
    for (const auto& ch : chunks) total_lines += (ch.end_line - ch.start_line + 1);
    s.avg_lines = static_cast<double>(total_lines) / static_cast<double>(chunks.size());

    std::size_t intact = 0;
    for (std::size_t d : c.def_lines) {
        for (const auto& ch : chunks) {
            if (ch.start_line <= d && ch.end_line >= d + body_lines) { ++intact; break; }
        }
    }
    s.integrity = static_cast<double>(intact) / static_cast<double>(c.def_lines.size());
    return s;
}

// The baseline every RAG system that lacks language awareness actually uses:
// fixed-size line windows.
std::vector<rag::Chunk> window_chunks(const std::string& body, std::size_t win) {
    std::vector<rag::Chunk> out;
    std::vector<std::string> lines;
    std::string cur;
    for (char ch : body) {
        if (ch == '\n') { lines.push_back(std::move(cur)); cur.clear(); }
        else cur.push_back(ch);
    }
    lines.push_back(std::move(cur));
    for (std::size_t i = 0; i < lines.size(); i += win) {
        rag::Chunk c;
        c.doc = rag::DocId{0};
        c.start_line = static_cast<std::uint32_t>(i);
        c.end_line = static_cast<std::uint32_t>(std::min(i + win, lines.size()) - 1);
        for (std::size_t j = i; j < std::min(i + win, lines.size()); ++j) {
            c.text += lines[j];
            c.text.push_back('\n');
        }
        out.push_back(std::move(c));
    }
    return out;
}

// ─── Negative controls ────────────────────────────────────────────────────────
//
// The dangerous failure mode is not missing structure, it is HALLUCINATING it.
// Prose chunked as if it were code produces chunks cut at "The" and "However",
// which is strictly worse than the prose chunker. So inference must DECLINE on
// text that has no definitions, and these corpora exist to prove it does.

std::string prose_corpus() {
    std::string s = "Annual Report on Operations\n\n";
    const char* paras[] = {
        "The company continued to expand its regional footprint during the period "
        "under review, opening additional distribution centers in three markets.",
        "Revenue grew modestly against a difficult comparison, with margin pressure "
        "from input costs partially offset by pricing actions taken in the spring.",
        "However, the board notes that the competitive environment remains "
        "challenging and that further investment will be required to sustain share.",
        "Management believes the balance sheet is well positioned to support the "
        "planned capital program without recourse to additional external financing.",
        "This report should be read alongside the audited financial statements and "
        "the accompanying notes, which form an integral part of the disclosure.",
    };
    for (std::size_t i = 0; i < 12; ++i) {
        s += paras[i % 5];
        s += "\n\n";
    }
    return s;
}

// A flat key-value file. It is highly repetitive and beautifully aligned, which
// is exactly the shape that fools a naive frequency-based detector -- but it has
// no definitions and no nesting, so there is nothing to chunk on and inference
// should say so rather than invent boundaries.
std::string flat_config_corpus() {
    std::string s;
    const char* keys[] = {"host", "port", "user", "timeout", "retries", "verbose"};
    for (std::size_t i = 0; i < 30; ++i) {
        s += keys[i % 6];
        s += "_" + std::to_string(i) + " = value_" + std::to_string(i * 3) + "\n";
    }
    return s;
}

void report_negative(const char* name, const std::string& body) {
    auto prof = rag::loaders::infer_structure(body);
    auto strat = rag::loaders::strategy_for("", body);
    std::printf("  %-18s %-15s %-8s %-8s %5s  %5s   %.2f  %s\n",
                name,
                std::string(rag::loaders::strategy_name(strat)).c_str(),
                "-", "-", "-", "-", prof.confidence,
                prof.usable() ? "USED (bad!)" : "declined (correct)");
}

void report(const char* label, const Corpus& c, std::size_t body_lines) {
    CodeChunkOptions opts;
    auto inferred = rag::loaders::chunk_source(rag::DocId{0}, c.ext, c.body, opts);
    Score si = evaluate(c, inferred, body_lines);
    si.strategy = rag::loaders::strategy_for(c.ext, c.body);

    Score sw = evaluate(c, window_chunks(c.body, 40), body_lines);

    auto prof = rag::loaders::infer_structure(c.body);
    std::string toks;
    for (std::size_t i = 0; i < prof.definition_tokens.size() && i < 3; ++i) {
        if (i) toks += ",";
        toks += prof.definition_tokens[i];
    }
    if (toks.empty()) toks = "-";

    std::printf("  %-18s %-15s %6.3f   %6.3f   %5zu  %5.1f   %.2f  %s\n",
                c.name.c_str(),
                std::string(rag::loaders::strategy_name(si.strategy)).c_str(),
                si.integrity, sw.integrity, si.chunks, si.avg_lines,
                prof.confidence, toks.c_str());
    (void)label;
}

} // namespace

int main() {
    std::printf("Structure inference — definition integrity vs fixed windows\n\n");
    std::printf("  Integrity = fraction of definitions that land INTACT (signature +\n"
                "  body together) inside a single chunk. Higher is better; a split\n"
                "  definition is the failure that makes code retrieval useless.\n\n");

    std::printf("  %-18s %-15s %-8s %-8s %-6s %-6s %-5s %s\n",
                "corpus", "strategy", "chunked", "windows", "n", "lines", "conf", "tokens");
    std::printf("  %s\n", std::string(92, '-').c_str());

    std::printf("\n  Real languages (control: hand-written chunkers should win or tie)\n");
    report("py",  synth("python",     ".py", 40, "import os\nimport sys\n\n", py_unit),  6);
    report("go",  synth("go",         ".go", 40, "package main\n\nimport \"context\"\n\n", go_unit), 6);
    report("rs",  synth("rust",       ".rs", 40, "use std::fmt;\n\n", rs_unit), 6);
    report("cpp", synth("cpp",        ".cpp", 40, "#include <vector>\n\n", cpp_unit), 6);
    report("rb",  synth("ruby",       ".rb", 40, "require 'set'\n\n", rb_unit), 6);

    std::printf("\n  Invented languages — no parser, no grammar, no extension, not on the internet\n");
    report("cfg",   synth("dsl:config",  "", 40, "; internal service topology\nversion 3\n\n", dsl_config_unit), 5);
    report("4gl",   synth("dsl:4gl",     "", 40, "IDENTIFICATION DIVISION.\n\n", dsl_4gl_unit), 5);
    report("rules", synth("dsl:rules",   "", 40, "package fraud.rules\n\n", dsl_rules_unit), 6);
    report("hdl",   synth("dsl:hdl",     "", 40, "LIBRARY ieee_internal\n\n", dsl_hdl_unit), 6);
    report("build", synth("dsl:build",   "", 40, "# internal build graph\n\n", dsl_build_unit), 5);

    std::printf("\n  Negative controls — inference must DECLINE here, not invent boundaries\n");
    report_negative("prose:report", prose_corpus());
    report_negative("config:flat_kv", flat_config_corpus());

    std::printf("\n  The control rows show inference is not cheating: where a hand-written\n"
                "  chunker exists it is used. The invented rows are the claim — a file in\n"
                "  a language that has never been parsed by anything, chunked on its own\n"
                "  conventions, measured against the windows every other system falls back to.\n");
    return 0;
}
