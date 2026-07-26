// rag/query/hyde.cpp — HyDE + multi-query (RAG-Fusion) query transformation.

#include "rag/query/hyde.hpp"

#include <algorithm>
#include <string>
#include <vector>

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

// Retrieve for SEVERAL texts at once.
//
// HyDE and multi-query are the two places in the library that genuinely hold a
// batch of queries at the same time, which is the only shape a GPU can help
// with: q queries against n candidates is a matrix product with q times the
// arithmetic intensity of q separate scans, whereas a single scan is
// bandwidth-bound and cannot be accelerated at all. Handing the whole batch to
// the corpus lets it make that routing decision once; dense_search_batch falls
// back to exactly this loop when the GPU is absent, declines, or the corpus has
// an HNSW graph to walk instead.
std::vector<std::vector<Hit>>
retrieve_texts(const index::Corpus& corpus, const std::vector<std::string>& texts, std::size_t k) {
    std::vector<std::vector<Hit>> out;
    if (texts.empty()) return out;

    if (corpus.has_embedder()) {
        if (auto d = corpus.dense_search_batch(texts, k)) out = std::move(*d);
    }
    if (out.size() != texts.size()) out.assign(texts.size(), {});
    // Lexical fallback per text, matching retrieve_text()'s behaviour exactly:
    // a text the dense path could not serve still gets a lexical ranking.
    for (std::size_t i = 0; i < texts.size(); ++i)
        if (out[i].empty()) out[i] = corpus.lexical_search(texts[i], k);
    return out;
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
            std::vector<std::string> batch(docs->begin(), docs->begin() + (long)take);
            // One batched retrieval for all hypotheticals rather than `take`
            // separate scans over the same corpus.
            for (auto& hits : retrieve_texts(corpus, batch, k))
                if (!hits.empty()) lists.push_back({std::move(hits), 1.0f});
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
    // The original query plus every paraphrase, retrieved as ONE batch. This is
    // the canonical multi-query shape and the reason dense_search_batch exists.
    std::vector<std::string> batch{std::string(query)};

    if (generate && n > 0) {
        std::string prompt =
            "Generate " + std::to_string(n) +
            " alternative phrasings of the following question, one per line.\n\nQuestion: " +
            std::string(query);
        if (auto paras = generate(prompt))
            for (const auto& p : *paras)
                if (!p.empty()) batch.push_back(p);
    }

    for (auto& hits : retrieve_texts(corpus, batch, k))
        if (!hits.empty()) lists.push_back({std::move(hits), 1.0f});

    if (lists.empty())
        return std::unexpected(Error{Errc::empty_corpus, "multi_query: no results"});
    return fusion::rrf(lists, rrf, k);
}

} // namespace rag::query
