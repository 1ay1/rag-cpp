// cli/main.cpp — the `ragcpp` turnkey command-line tool.
//
//   ragcpp index  <dir> <out.ragdb> [--glob=*.md] [--semantic]
//   ragcpp query  <db.ragdb> "<query>" [-k N] [--mmr]
//   ragcpp eval   <beir-dir> [--split=test]
//   ragcpp info   <db.ragdb>
//
// A thin driver over the library so you can build and search a corpus without
// writing a line of C++. Lexical/BM25 by default (no model, no network); attach
// an embedder in code for hybrid.

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <rag/rag.hpp>

namespace {

int usage() {
    std::printf(
        "ragcpp — a type-theoretic RAG engine\n\n"
        "usage:\n"
        "  ragcpp index <dir> <out.ragdb> [--glob=PATTERN] [--semantic]\n"
        "  ragcpp query <db.ragdb> \"<query>\" [-k N] [--mmr]\n"
        "  ragcpp eval  <beir-dir> [--split=test]\n"
        "  ragcpp info  <db.ragdb>\n");
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

    rag::index::Corpus corpus;
    rag::loaders::DirOptions lo;
    std::string ext = opt(args, "--glob", "");   // e.g. --glob=.md restricts to one ext
    if (!ext.empty()) { if (ext[0] != '.') ext = "." + ext; lo.include_ext = {ext}; }
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

int cmd_info(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto corpus = rag::index::Corpus::load(args[0]);
    if (!corpus) { std::printf("open error: %s\n", corpus.error().message.c_str()); return 1; }
    std::printf("%s\n  documents: %zu (live %zu)\n  chunks:    %zu\n  embedder:  %s\n",
                args[0].c_str(), corpus->document_count(), corpus->live_document_count(),
                corpus->chunk_count(), corpus->has_embedder() ? "yes" : "no (lexical only)");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);
    if (cmd == "index") return cmd_index(args);
    if (cmd == "query") return cmd_query(args);
    if (cmd == "eval")  return cmd_eval(args);
    if (cmd == "info")  return cmd_info(args);
    return usage();
}
