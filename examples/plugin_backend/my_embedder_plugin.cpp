// examples/plugin_backend/my_embedder_plugin.cpp
//
// A COMPLETE out-of-tree embedder plugin. Compile this as a shared library and
// the host loads it at runtime via rag::plugin::load_plugin(...) — the name
// "reverse_hash" then works everywhere a config accepts an embedder type, with
// no recompile of the host application.
//
// Build (standalone, matching the host toolchain):
//   g++-15 -std=c++23 -fPIC -shared -I<rag>/include \
//       my_embedder_plugin.cpp -o libreverse_hash.so
// Or via this repo's CMake (see CMakeLists.txt here).

#include <rag/plugin/plugin.hpp>
#include <rag/dense/backends.hpp>

#include <cstddef>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace {

// A toy embedder that models rag::Embedder: it hashes the REVERSED text. The
// point is only to demonstrate a custom concept model flowing through the
// registry; swap the body for a real model (ONNX, a REST call, etc.).
class ReverseHashEmbedder {
public:
    explicit ReverseHashEmbedder(std::size_t dim) : dim_(dim) {}
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "reverse-hash-v1"; }

    [[nodiscard]] rag::Result<std::vector<rag::Vector>>
    embed(std::span<const std::string> texts) const {
        std::vector<rag::Vector> out;
        out.reserve(texts.size());
        for (const auto& t : texts) {
            std::string r(t.rbegin(), t.rend());
            rag::Vector v(dim_, 0.0f);
            std::size_t h = 1469598103934665603ull % dim_;
            for (unsigned char c : r) {
                h = (h * 131 + c) % dim_;
                v[h] += 1.0f;
            }
            // L2 normalize so cosine is well-defined.
            float n = 0.0f;
            for (float x : v) n += x * x;
            if (n > 0) { n = 1.0f / std::sqrt(n); for (float& x : v) x *= n; }
            out.push_back(std::move(v));
        }
        return out;
    }

private:
    std::size_t dim_;
};

} // namespace

// Convention 1: a static registrar. Runs on dlopen. This alone is enough.
RAG_REGISTER(rag::plugin::AnyEmbedder, "reverse_hash",
    [](const nlohmann::json& cfg) -> rag::Result<rag::plugin::AnyEmbedder> {
        auto dim = cfg.value("dim", 256);
        return rag::plugin::AnyEmbedder{ReverseHashEmbedder{static_cast<std::size_t>(dim)}};
    });

// Convention 2 (optional): an explicit hook, called by load_plugin after dlopen.
// Useful if a toolchain would strip the "unused" static registrar above. Here it
// is a no-op because the static registrar already did the work.
extern "C" void rag_plugin_register() { /* registrars already ran */ }
