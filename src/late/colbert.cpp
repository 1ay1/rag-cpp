// rag/late/colbert.cpp — ColBERT late-interaction (MaxSim) reranking.

#include "rag/late/colbert.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>

#include "rag/dense/simd.hpp"

namespace rag::late {
namespace {

// FNV-1a over a byte range.
std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

std::vector<std::string_view> ws_split(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        std::size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

} // namespace

float maxsim(std::span<const Vector> query_tokens,
             std::span<const Vector> doc_tokens) noexcept {
    if (query_tokens.empty() || doc_tokens.empty()) return 0.0f;
    float total = 0.0f;
    for (const auto& q : query_tokens) {
        float best = -1.0f;
        for (const auto& d : doc_tokens) {
            if (q.size() != d.size()) continue;
            best = std::max(best, dense::dot(q, d));
        }
        if (best > 0.0f) total += best;   // ReLU: negative matches contribute 0
    }
    return total;
}

TokenEmbedder hashed_token_embedder(std::size_t dim) {
    return [dim](std::string_view text) -> Result<TokenMatrix> {
        TokenMatrix out;
        for (auto tok : ws_split(text)) {
            std::string lower;
            lower.reserve(tok.size());
            for (char c : tok) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            std::vector<float> v(dim, 0.0f);
            // hash character 3-grams (with boundary padding) into the vector,
            // signed by a second hash bit — a hashing-trick token encoder.
            std::string padded = "^" + lower + "$";
            if (padded.size() < 3) padded += "$$";
            for (std::size_t i = 0; i + 3 <= padded.size(); ++i) {
                std::string_view g(padded.data() + i, 3);
                std::uint64_t h = fnv1a(g);
                std::size_t idx = h % dim;
                float sign = (h & 0x8000000000000000ull) ? -1.0f : 1.0f;
                v[idx] += sign;
            }
            dense::normalize(v);
            out.push_back(std::move(v));
        }
        if (out.empty()) return std::unexpected(Error{Errc::invalid_argument, "empty text"});
        return out;
    };
}

Result<std::vector<float>>
ColbertReranker::rerank(std::string_view query,
                        std::span<const std::string> passages) const {
    auto qm = embed_(query);
    if (!qm) return std::unexpected(qm.error());
    std::vector<float> scores;
    scores.reserve(passages.size());
    for (const auto& p : passages) {
        auto dm = embed_(p);
        if (!dm) { scores.push_back(0.0f); continue; }
        scores.push_back(maxsim(*qm, *dm));
    }
    return scores;
}

Result<void>
ColbertReranker::rerank_hits(std::string_view query, std::vector<Hit>& hits,
                             const std::function<std::string(const Hit&)>& text_of) const {
    auto qm = embed_(query);
    if (!qm) return std::unexpected(qm.error());
    std::vector<std::pair<float, Hit>> scored;
    scored.reserve(hits.size());
    for (const auto& h : hits) {
        auto dm = embed_(text_of(h));
        float s = dm ? maxsim(*qm, *dm) : 0.0f;
        scored.emplace_back(s, h);
    }
    std::stable_sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t i = 0; i < hits.size(); ++i) {
        hits[i] = scored[i].second;
        hits[i].score = Score{scored[i].first};
    }
    return {};
}

} // namespace rag::late
