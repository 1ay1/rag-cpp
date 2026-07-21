// rag/query/hyde.cpp — HyDE + multi-query (RAG-Fusion) query transformation.

#include "rag/query/hyde.hpp"

#include <string>

namespace rag::query {
namespace {

// Retrieve for a piece of text: dense if available, else lexical. Since the
// corpus embeds text internally, HyDE just feeds the HYPOTHETICAL document text
// through the same dense path — its embedding lands near real relevant docs.
std::vector<Hit> retrieve_text(const index::Corpus& corpus, const std::string& text, std::size_t k) {
    if (corpus.has_embedder()) {
        auto d = corpus.dense_search(text, k);
        if (d) return std::move(*d);
    }
    return corpus.lexical_search(text, k);
}

} // namespace

Result<std::vector<Hit>>
hyde_search(const index::Corpus& corpus, std::string_view query, std::size_t k,
            const Generator& generate, HydeConfig cfg) {
    std::vector<fusion::RankedList> lists;

    // Generate hypothetical answer documents and retrieve for each.
    if (generate && cfg.hypotheticals > 0) {
        std::string prompt = cfg.instruction;
        prompt += "\n\nQuestion: ";
        prompt += std::string(query);
        prompt += "\nPassage:";
        auto docs = generate(prompt);
        if (docs) {
            std::size_t take = std::min(docs->size(), cfg.hypotheticals);
            for (std::size_t i = 0; i < take; ++i) {
                auto hits = retrieve_text(corpus, (*docs)[i], k);
                if (!hits.empty()) lists.push_back({std::move(hits), 1.0f});
            }
        }
        // On generator failure we fall through to the raw-query path below.
    }

    // Optionally (or as fallback) fuse in the raw-query retrieval.
    if (cfg.include_query || lists.empty()) {
        auto hits = retrieve_text(corpus, std::string(query), k);
        if (!hits.empty()) lists.push_back({std::move(hits), 1.0f});
    }
    if (lists.empty())
        return std::unexpected(Error{Errc::empty_corpus, "hyde: no results"});
    if (lists.size() == 1) {
        auto out = std::move(lists[0].hits);
        if (out.size() > k) out.resize(k);
        return out;
    }
    return fusion::rrf(lists, cfg.rrf, k);
}

Result<std::vector<Hit>>
multi_query_search(const index::Corpus& corpus, std::string_view query, std::size_t k,
                   const Generator& generate, std::size_t n, fusion::RrfParams rrf) {
    std::vector<fusion::RankedList> lists;
    // Always include the original query.
    if (auto h = retrieve_text(corpus, std::string(query), k); !h.empty())
        lists.push_back({std::move(h), 1.0f});

    if (generate && n > 0) {
        std::string prompt =
            "Generate " + std::to_string(n) +
            " alternative phrasings of the following question, one per line.\n\nQuestion: " +
            std::string(query);
        if (auto paras = generate(prompt)) {
            for (const auto& p : *paras) {
                if (p.empty()) continue;
                if (auto h = retrieve_text(corpus, p, k); !h.empty())
                    lists.push_back({std::move(h), 1.0f});
            }
        }
    }
    if (lists.empty())
        return std::unexpected(Error{Errc::empty_corpus, "multi_query: no results"});
    return fusion::rrf(lists, rrf, k);
}

} // namespace rag::query
