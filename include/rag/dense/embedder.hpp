#pragma once
// rag/dense/embedder.hpp — the dense-embedding abstraction + transport seam.
//
// This is the library's one true boundary to the outside world. Embedding text
// means calling SOME model server; rather than hard-wire an HTTP client, we
// inject an `HttpTransport`. The library ships a default socket-based transport
// (rag/dense/http_transport.hpp) but a host application can supply its own
// (its existing HTTP stack, a mock for tests, a gRPC bridge — anything).
//
// Embedders model the `Embedder` concept (core/concepts.hpp) and are exposed
// both as concrete types and via the type-erased `AnyEmbedder` so a pipeline
// can hold one chosen at runtime.

#include <chrono>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/concepts.hpp"
#include "rag/core/types.hpp"

namespace rag::dense {

// ─────────────────────────────────────────────────────────────────────────────
// HttpTransport — the injectable network seam. A single POST primitive.
// ─────────────────────────────────────────────────────────────────────────────
struct HttpResponse {
    int         status = 0;
    std::string body;
};

struct HttpTransport {
    virtual ~HttpTransport() = default;
    // POST `body` (with Content-Type application/json) to http://host:port/path.
    // Returns the response, or an Error (transport_error / unavailable) on
    // failure. Must be thread-safe for concurrent embed() calls.
    [[nodiscard]] virtual Result<HttpResponse>
    post_json(std::string_view host, std::uint16_t port, std::string_view path,
              std::string_view body, std::chrono::milliseconds timeout) const = 0;
};

// The library's default transport (blocking POSIX/Winsock sockets, plaintext
// HTTP/1.1 — intended for localhost model servers like Ollama / llama.cpp).
[[nodiscard]] std::shared_ptr<HttpTransport> default_http_transport();

// ─────────────────────────────────────────────────────────────────────────────
// AnyEmbedder — type-erased embedder for the runtime-polymorphic path.
// ─────────────────────────────────────────────────────────────────────────────
class AnyEmbedder {
public:
    template <Embedder E>
    explicit AnyEmbedder(E e)
        : self_(std::make_shared<Model<E>>(std::move(e))) {}

    [[nodiscard]] std::size_t dimension() const { return self_->dimension(); }
    [[nodiscard]] std::string_view identity() const { return self_->identity(); }
    [[nodiscard]] Result<std::vector<Vector>>
    embed(std::span<const std::string> texts) const { return self_->embed(texts); }

    [[nodiscard]] Result<Vector> embed_one(const std::string& text) const {
        std::array<std::string, 1> one{text};
        auto r = embed(one);
        if (!r) return std::unexpected(r.error());
        if (r->empty()) return fail<Vector>(Errc::transport_error, "empty embed result");
        return std::move((*r)[0]);
    }

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::size_t dimension() const = 0;
        virtual std::string_view identity() const = 0;
        virtual Result<std::vector<Vector>> embed(std::span<const std::string>) const = 0;
    };
    template <Embedder E>
    struct Model final : Concept {
        E e;
        explicit Model(E x) : e(std::move(x)) {}
        std::size_t dimension() const override { return e.dimension(); }
        std::string_view identity() const override { return e.identity(); }
        Result<std::vector<Vector>> embed(std::span<const std::string> t) const override {
            return e.embed(t);
        }
    };
    std::shared_ptr<const Concept> self_;
};

} // namespace rag::dense
