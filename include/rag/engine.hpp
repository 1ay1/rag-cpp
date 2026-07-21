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
#include "rag/index/corpus.hpp"
#include "rag/pipeline/pipeline.hpp"

namespace rag {

class Engine {
public:
    Engine() : corpus_(index::CorpusConfig{}), pipeline_(pipeline::Pipeline::standard()) {}
    explicit Engine(index::CorpusConfig cfg)
        : corpus_(std::move(cfg)), pipeline_(pipeline::Pipeline::standard()) {}

    // Attach a dense embedder (enables hybrid). Fluent.
    Engine& with_embedder(dense::AnyEmbedder e) { corpus_.set_embedder(std::move(e)); return *this; }
    Engine& with_pipeline(pipeline::Pipeline p) { pipeline_ = std::move(p); return *this; }

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

    Result<void> save(const std::string& path) const { return corpus_.save(path); }

private:
    index::Corpus     corpus_;
    pipeline::Pipeline pipeline_;
};

} // namespace rag
