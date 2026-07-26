// examples/plugin_backend/local_embedder_plugin.cpp
//
// A COMPLETE, powerful out-of-tree LOCAL embedder plugin. Unlike the toy
// reverse_hash plugin next door, this one is a genuine in-process embedding
// model: a character n-gram feature-hashing embedder with sub-word structure,
// L2-normalized so it is cosine-ready. It needs NO external library and NO
// network — everything runs in this process — which is exactly what "local
// embedder" means. Swap the body for ONNX Runtime or llama.cpp and the wiring
// below is unchanged.
//
// The point of the example: a third party ships this .so, the host loads it at
// runtime, and "local_ngram" then works everywhere a config accepts an embedder
// type — CLI, C ABI, RCP server, or engine.with_embedder_spec — with NO recompile
// of the host and no host knowledge that this type exists.
//
// Build (standalone, MUST match the host toolchain and rag headers/ABI):
//   g++-15 -std=c++23 -fPIC -shared -I<rag>/include \
//       local_embedder_plugin.cpp -o liblocal_ngram.so
// Or via this repo's CMake (target: local_ngram_plugin).

#include <rag/plugin/plugin.hpp>
#include <rag/plugin/builder.hpp>
#include <rag/core/concepts.hpp>
#include <rag/dense/embedder.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ── The model ────────────────────────────────────────────────────────────────
// A local, deterministic embedder that models rag::dense::Embedder. It hashes
// overlapping character n-grams (default 3) into a fixed-width vector, then
// L2-normalizes. Character n-grams give it real sub-word signal: "retrieval" and
// "retrieved" share most of their grams and so land close in vector space, which
// a whole-token hash embedder cannot capture. This is a legitimate baseline
// local model, not a placeholder — it is what a fastText-style hashing embedder
// does, minus the trained weights.
class LocalNgramEmbedder {
public:
    LocalNgramEmbedder(std::size_t dim, std::size_t n) : dim_(dim), n_(std::max<std::size_t>(2, n)) {}

    [[nodiscard]] std::size_t      dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity()  const noexcept { return "local-ngram-v1"; }

    [[nodiscard]] rag::Result<std::vector<rag::Vector>>
    embed(std::span<const std::string> texts) const {
        std::vector<rag::Vector> out;
        out.reserve(texts.size());
        for (const auto& text : texts) out.push_back(embed_one(text));
        return out;
    }

private:
    // FNV-1a over a byte range — cheap, well-distributed, deterministic.
    static std::uint64_t fnv1a(std::string_view s) noexcept {
        std::uint64_t h = 1469598103934665603ull;
        for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ull; }
        return h;
    }

    rag::Vector embed_one(std::string_view text) const {
        rag::Vector v(dim_, 0.0f);
        if (text.size() >= n_) {
            for (std::size_t i = 0; i + n_ <= text.size(); ++i) {
                std::string_view gram = text.substr(i, n_);
                std::uint64_t h = fnv1a(gram);
                // Signed hashing (the fastText trick): the top bit chooses the
                // sign so collisions cancel in expectation instead of piling up.
                float sign = (h & 1ull) ? 1.0f : -1.0f;
                v[h % dim_] += sign;
            }
        }
        // L2-normalize so dot product == cosine similarity, which is what the
        // dense retriever assumes.
        double norm = 0.0;
        for (float x : v) norm += double(x) * x;
        if (norm > 0.0) {
            float inv = static_cast<float>(1.0 / std::sqrt(norm));
            for (float& x : v) x *= inv;
        }
        return v;
    }

    std::size_t dim_;
    std::size_t n_;
};

static_assert(rag::Embedder<LocalNgramEmbedder>,
              "LocalNgramEmbedder must model the Embedder concept");

} // namespace

// ── Registration ─────────────────────────────────────────────────────────────
// The ergonomic way (rag/plugin/builder.hpp): return your concept type from a
// Config -> Result<T> lambda. The AnyEmbedder wrap, the not-throwing config
// parsing, and the describe() text are all handled — no macro, no comma trap.
// Registered from rag_plugin_register(), which load_plugin() calls after dlopen.
extern "C" void rag_plugin_register() {
    rag::plugin::register_embedder(
        "local_ngram",
        "in-process character n-gram feature-hash embedder, no deps (keys: dim, ngram)",
        [](rag::plugin::Config c) -> rag::Result<LocalNgramEmbedder> {
            int dim   = c.get("dim", 384);
            int ngram = c.get("ngram", 3);
            if (dim <= 0)
                return rag::fail<LocalNgramEmbedder>(
                    rag::Errc::invalid_argument, "local_ngram: dim must be positive");
            return LocalNgramEmbedder(static_cast<std::size_t>(dim), static_cast<std::size_t>(ngram));
        });
}
