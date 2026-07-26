// cli/main.cpp — the `ragcpp` turnkey command-line tool.
//
//   ragcpp index  <dir> <out.ragdb> [--ext=.md] [--semantic]
//   ragcpp query  <db.ragdb> "<query>" [-k N] [--mmr]
//   ragcpp serve  <db.ragdb> [--http PORT] [--write] [--graph] [--memory] [--feedback] [--all]
//   ragcpp eval   <beir-dir> [--split=test]
//   ragcpp info   <db.ragdb>
//
// A thin driver over the library so you can build, search, and SERVE a corpus
// without writing a line of C++. Lexical/BM25 by default (no model, no
// network); attach an embedder in code for hybrid. `serve` brings up a
// conformant RCP/1 endpoint (github.com/1ay1/rcp) over stdio or HTTP.

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <rag/rag.hpp>
#if RAGCPP_WITH_RCP
#include <rag/rcp/rcp.hpp>
#endif

namespace {

int usage() {
    std::printf(
        "ragcpp — a type-theoretic RAG engine\n\n"
        "usage:\n"
        "  ragcpp index <dir> <out.ragdb> [--ext=.md] [--semantic] [--contextual]\n"
        "  ragcpp query <db.ragdb> \"<query>\" [-k N] [--mmr]\n"
        "  ragcpp serve <db.ragdb> [--http PORT] [--write] [--graph] [--memory] [--feedback] [--all]\n"
        "  ragcpp eval  <beir-dir> [--split=test]\n"
        "  ragcpp info  <db.ragdb>\n"
        "  ragcpp list  [embedders|rerankers]\n");
    return 2;
}

std::string opt(const std::vector<std::string>& a, std::string_view key, std::string def) {
    for (const auto& s : a)
        if (s.rfind(key, 0) == 0 && s.size() > key.size() && s[key.size()] == '=')
            return s.substr(key.size() + 1);
    return def;
}
bool flag(const std::vector<std::string>& a, std::string_view f) {
    for (const auto& s : a) if (s == f) return true;
    return false;
}

int cmd_index(const std::vector<std::string>& args) {
    if (args.size() < 2) return usage();
    const std::string& dir = args[0];
    const std::string& out = args[1];

    rag::index::CorpusConfig ccfg;
    if (flag(args, "--semantic"))
        ccfg.chunking = rag::index::CorpusConfig::Chunking::semantic;
    // Contextual Retrieval (Anthropic 2024): situate each chunk in its document
    // before indexing. No model required — the CLI has no LLM binding, so this
    // uses the deterministic extractive context.
    if (flag(args, "--contextual")) ccfg.contextual = true;
    rag::index::Corpus corpus{ccfg};
    rag::loaders::DirOptions lo;
    // --ext restricts by FILE EXTENSION. This was previously spelled --glob,
    // which was actively misleading: it never accepted a pattern, so
    // `--glob='*.md'` silently matched nothing and indexed zero documents.
    // The old spelling still works so existing invocations do not break.
    std::string ext = opt(args, "--ext", "");
    if (ext.empty()) ext = opt(args, "--glob", "");
    if (!ext.empty()) {
        // Accept .md, md, and *.md alike rather than failing silently on the
        // form a user would most naturally reach for.
        if (ext.rfind("*.", 0) == 0) ext.erase(0, 1);
        if (ext[0] != '.') ext = "." + ext;
        lo.include_ext = {ext};
    }
    auto docs = rag::loaders::load_directory(dir, lo);
    if (!docs) { std::printf("load error: %s\n", docs.error().message.c_str()); return 1; }

    std::size_t n = 0;
    for (auto& d : *docs) {
        if (corpus.add_document(d.uri, d.text, d.meta, d.title)) ++n;
    }
    if (auto b = corpus.build(); !b) { std::printf("build error: %s\n", b.error().message.c_str()); return 1; }
    if (auto s = corpus.save(out); !s) { std::printf("save error: %s\n", s.error().message.c_str()); return 1; }
    std::printf("indexed %zu documents, %zu chunks → %s\n", n, corpus.chunk_count(), out.c_str());
    return 0;
}

int cmd_query(const std::vector<std::string>& args) {
    if (args.size() < 2) return usage();
    auto corpus = rag::index::Corpus::load(args[0]);
    if (!corpus) { std::printf("open error: %s\n", corpus.error().message.c_str()); return 1; }
    const std::string& q = args[1];
    std::size_t k = 5;
    for (std::size_t i = 0; i < args.size(); ++i)
        if (args[i] == "-k" && i + 1 < args.size()) k = std::stoul(args[i + 1]);

    auto hits = corpus->lexical_search(q, flag(args, "--mmr") ? k * 4 : k);
    if (flag(args, "--mmr")) {
        rag::rerank::MmrConfig mc; mc.k = k;
        hits = rag::rerank::mmr(*corpus, hits, mc);
    }
    if (hits.size() > k) hits.resize(k);
    std::printf("query: %s  (%zu results)\n", q.c_str(), hits.size());
    for (const auto& h : hits) {
        auto r = corpus->resolve(h);
        std::printf("  [%.3f] %-24s %.70s\n", h.score.get(), r.uri.c_str(), r.text.c_str());
    }
    return 0;
}

int cmd_eval(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    std::string split = opt(args, "--split", "test");
    auto ds = rag::eval::BeirDataset::load(args[0], split);
    if (!ds) { std::printf("load error: %s\n", ds.error().message.c_str()); return 1; }
    rag::index::Corpus corpus;
    auto m = rag::eval::evaluate_corpus(*ds, corpus);
    if (!m) { std::printf("eval error: %s\n", m.error().message.c_str()); return 1; }
    std::printf("%s\n", m->report().c_str());
    return 0;
}

#if RAGCPP_WITH_RCP
// Bring up an RCP/1 server backed by a saved corpus. stdio by default (the
// convention for editor / agent integration); `--http PORT` for loopback HTTP.
// `--write` advertises a writable index; `--graph` advertises GraphRAG.
int cmd_serve(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto engine = rag::Engine::open(args[0]);
    if (!engine) { std::printf("open error: %s\n", engine.error().message.c_str()); return 1; }

    // Resolve --http PORT (space-separated) and boolean feature flags.
    int http_port = -1;
    for (std::size_t i = 1; i < args.size(); ++i)
        if (args[i] == "--http" && i + 1 < args.size()) http_port = std::stoi(args[i + 1]);

    rag::rcp::Options opts;
    opts.named("ragcpp", "1.0");
    if (flag(args, "--write")) {
        opts.with_index(true);
        // A writable index that never writes anything back is a lie to the
        // client: index/add and index/delete were accepted, acknowledged, and
        // then lost when the process exited. Persist to the same file we were
        // opened from, so a mutation that was acknowledged is a mutation that
        // survives a restart.
        opts.persisting_to(args[0]);

        // ...and do it through a write-ahead log, so "acknowledged" costs an
        // O(record) append instead of an O(corpus) snapshot rewrite. Measured:
        // index/add over RCP on a 20k-document corpus went from 25.1 ms to
        // 0.25 ms, and the old cost grew with the corpus while the new one does
        // not. The log sits beside the index as <db>.wal.
        //
        // Opened AFTER the engine loads, because open_wal replays any log left
        // by a previous crash into the corpus — that replay IS the recovery.
        const std::string wal_path = args[0] + ".wal";
        if (auto w = engine->corpus().open_wal(wal_path); !w) {
            std::printf("wal error: %s\n", w.error().message.c_str());
            return 1;
        }
        opts.with_wal(wal_path);
    }
    if (flag(args, "--graph")) opts.with_graph(true);
    // memory/* and feedback/* are backed entirely by the engine itself — they
    // need no external model, no API key and no extra process. Leaving them off
    // by default meant the turnkey server advertised strictly less than it could
    // actually do; --all turns on everything that costs nothing to provide.
    if (flag(args, "--all")) {
        opts.with_memory(true);
        opts.with_feedback(true);
        opts.with_graph(true);
    } else {
        if (flag(args, "--memory"))   opts.with_memory(true);
        if (flag(args, "--feedback")) opts.with_feedback(true);
    }

    if (http_port > 0) {
        std::fprintf(stderr, "ragcpp: RCP/1 server on http://127.0.0.1:%d  (corpus: %s)\n",
                     http_port, args[0].c_str());
        auto r = rag::rcp::serve_http(*engine, static_cast<std::uint16_t>(http_port), std::move(opts));
        if (!r) { std::printf("serve error: %s\n", r.error().message.c_str()); return 1; }
        return 0;
    }
    std::fprintf(stderr, "ragcpp: RCP/1 server on stdio  (corpus: %s)\n", args[0].c_str());
    rag::rcp::serve_stdio(*engine, std::move(opts));
    return 0;
}
#else
int cmd_serve(const std::vector<std::string>&) {
    std::printf("serve: rebuild with -DRAGCPP_WITH_RCP=ON to enable the RCP/1 server\n");
    return 1;
}
#endif

int cmd_info(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto corpus = rag::index::Corpus::load(args[0]);
    if (!corpus) { std::printf("open error: %s\n", corpus.error().message.c_str()); return 1; }
    std::printf("%s\n  documents: %zu (live %zu)\n  chunks:    %zu\n  embedder:  %s\n",
                args[0].c_str(), corpus->document_count(), corpus->live_document_count(),
                corpus->chunk_count(), corpus->has_embedder() ? "yes" : "no (lexical only)");
    return 0;
}

// List every registered backend and the config keys it takes — the user-facing
// face of the plugin registry's describe(). Answers "what can I put in a config?"
// without reading source. A plugin dir can be loaded first so third-party names
// show up too: `ragcpp list embedders --plugins=./plugins`.
int cmd_list(const std::vector<std::string>& args) {
    rag::plugin::ensure_builtins_registered();
    if (std::string dir = opt(args, "--plugins", ""); !dir.empty())
        (void)rag::plugin::load_plugin_dir(dir);

    std::string which = args.empty() || args[0].rfind("--", 0) == 0 ? "" : args[0];
    auto dump = [](const char* title, auto&& rows) {
        std::printf("%s:\n", title);
        for (const auto& [name, desc] : rows)
            std::printf("  %-14s %s\n", name.c_str(), desc.empty() ? "(no description)" : desc.c_str());
    };
    if (which.empty() || which == "embedders")
        dump("embedders", rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().describe());
    if (which.empty() || which == "rerankers")
        dump("rerankers", rag::plugin::Registry<rag::plugin::AnyReranker>::instance().describe());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);
    if (cmd == "index") return cmd_index(args);
    if (cmd == "query") return cmd_query(args);
    if (cmd == "serve") return cmd_serve(args);
    if (cmd == "eval")  return cmd_eval(args);
    if (cmd == "info")  return cmd_info(args);
    if (cmd == "list")  return cmd_list(args);
    return usage();
}
