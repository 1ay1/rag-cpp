#pragma once
// rag/engine.hpp — the one-stop facade.
//
// Engine bundles a Corpus + a Pipeline behind a tiny API: add documents,
// build, search. Most users never touch anything below this; power users reach
// into corpus() / with_pipeline() to customize stages, fusion, and rerankers.

#include <memory>
#include <string>
#include <vector>

#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/graph/graph.hpp"
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"
#include "rag/plugin/plugin.hpp"
#include "rag/ralm/ralm.hpp"

namespace rag {

class Engine {
public:
    Engine() : corpus_(index::CorpusConfig{}), pipeline_(pipeline::Pipeline::standard()) {}
    explicit Engine(index::CorpusConfig cfg)
        : corpus_(std::move(cfg)), pipeline_(pipeline::Pipeline::standard()) {}

    // Attach a dense embedder (enables hybrid). Fluent.
    Engine& with_embedder(dense::AnyEmbedder e) { corpus_.set_embedder(std::move(e)); return *this; }
    Engine& with_pipeline(pipeline::Pipeline p) { pipeline_ = std::move(p); return *this; }

    // Config-driven embedder attach: build a registered embedder by name from a
    // JSON spec (bare name string, or object with a "type" field). Enables the
    // whole framework to be wired from a config file with zero backend knowledge
    // at the call site, and picks up any plugin loaded via load_plugin_dir().
    //   engine.with_embedder_spec({{"type","ollama"},{"model","nomic-embed-text"}});
    Result<void> with_embedder_spec(const plugin::Json& spec) {
        auto e = plugin::make_embedder(spec);
        if (!e) return std::unexpected(e.error());
        corpus_.set_embedder(std::move(*e));
        return {};
    }

    Result<DocId> add(std::string uri, std::string text, Metadata meta = {}, std::string title = {}) {
        return corpus_.add_document(std::move(uri), std::move(text), std::move(meta), std::move(title));
    }
    Result<void> build() { return corpus_.build(); }

    // Search: returns resolved, ranked results.
    [[nodiscard]] Result<std::vector<SearchResult>>
    search(std::string_view query, std::size_t k = 10, index::MetaFilter filter = {},
           std::vector<std::string>* trace = nullptr) const {
        auto hits = pipeline_.run(corpus_, query, k, std::move(filter), trace);
        if (!hits) return std::unexpected(hits.error());
        std::vector<SearchResult> out;
        out.reserve(hits->size());
        for (const auto& h : *hits) out.push_back(corpus_.resolve(h));
        return out;
    }

    [[nodiscard]] index::Corpus&       corpus()       noexcept { return corpus_; }
    [[nodiscard]] const index::Corpus& corpus() const noexcept { return corpus_; }

    // ── GraphRAG ──────────────────────────────────────────────────────────────
    // Build (and cache) the document graph. Rebuilds if the corpus changed.
    Result<const graph::DocGraph*> graph(graph::GraphConfig cfg = {}, graph::Summarizer s = {}) {
        if (!graph_ || graph_docs_ != corpus_.document_count()) {
            auto g = graph::DocGraph::build(corpus_, cfg, std::move(s));
            if (!g) return std::unexpected(g.error());
            graph_ = std::make_unique<graph::DocGraph>(std::move(*g));
            graph_docs_ = corpus_.document_count();
        }
        return graph_.get();
    }

    // GraphRAG local search: hybrid seed → PPR graph expansion → resolved hits.
    Result<std::vector<SearchResult>> graph_local(std::string_view query, std::size_t k) {
        auto g = graph(); if (!g) return std::unexpected(g.error());
        auto hits = (*g)->local_search(corpus_, query, k);
        if (!hits) return std::unexpected(hits.error());
        return resolve_all(*hits);
    }

    // GraphRAG global search: rank community summaries → community lead chunks.
    Result<std::vector<SearchResult>> graph_global(std::string_view query, std::size_t k) {
        auto g = graph(); if (!g) return std::unexpected(g.error());
        auto hits = (*g)->global_search(corpus_, query, k);
        if (!hits) return std::unexpected(hits.error());
        return resolve_all(*hits);
    }

    Result<void> save(const std::string& path) const { return corpus_.save(path); }

private:
    std::vector<SearchResult> resolve_all(const std::vector<Hit>& hits) const {
        std::vector<SearchResult> out;
        out.reserve(hits.size());
        for (const auto& h : hits) out.push_back(corpus_.resolve(h));
        return out;
    }

    index::Corpus                     corpus_;
    pipeline::Pipeline                pipeline_;
    std::unique_ptr<graph::DocGraph>  graph_;
    std::size_t                       graph_docs_ = static_cast<std::size_t>(-1);
};

} // namespace rag
