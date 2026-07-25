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
#include <utility>
#include <vector>

#include "rag/core/concepts.hpp"
#include "rag/core/types.hpp"

namespace rag::dense {

// ─────────────────────────────────────────────────────────────────────────────
// HttpTransport — the injectable network seam.
//
// A single POST primitive, but rich enough for every hosted backend: custom
// headers carry Bearer auth (OpenAI, TEI, Cohere) and `tls` requests an
// encrypted connection. The default transport speaks plaintext HTTP/1.1 for
// localhost model servers; for TLS endpoints inject a transport backed by your
// own TLS stack — that is exactly why the seam exists.
// ─────────────────────────────────────────────────────────────────────────────
struct HttpResponse {
    int         status = 0;
    std::string body;
};

struct HttpRequest {
    std::string_view host;
    std::uint16_t    port = 80;
    std::string_view path;
    std::string_view body;                                   // JSON payload
    std::vector<std::pair<std::string, std::string>> headers; // extra headers
    bool             tls     = false;                        // https
    std::chrono::milliseconds timeout{30'000};
};

struct HttpTransport {
    virtual ~HttpTransport() = default;

    // The rich primitive every backend uses. Must be thread-safe for concurrent
    // embed()/rerank() calls.
    [[nodiscard]] virtual Result<HttpResponse> post(const HttpRequest& req) const = 0;

    // Back-compat convenience: plaintext JSON POST with no extra headers.
    [[nodiscard]] Result<HttpResponse>
    post_json(std::string_view host, std::uint16_t port, std::string_view path,
              std::string_view body, std::chrono::milliseconds timeout) const {
        return post(HttpRequest{host, port, path, body, {}, false, timeout});
    }
};

// The library's default transport (blocking POSIX/Winsock sockets, plaintext
// HTTP/1.1 — intended for localhost model servers like Ollama / llama.cpp).
// A `tls=true` request against this transport fails with Errc::unavailable.
[[nodiscard]] std::shared_ptr<HttpTransport> default_http_transport();

// ─────────────────────────────────────────────────────────────────────────────
// Embedding concurrency hint.
//
// Batches are the unit of parallelism when indexing a corpus, but the right
// number of in-flight batches is a property of the BACKEND, not of the corpus:
//
//   • a hosted/remote embedder is latency-bound — a dozen concurrent POSTs
//     costs the client nothing and multiplies throughput;
//   • an in-process embedder (ONNX Runtime, llama.cpp) already saturates every
//     core inside a single `embed` call, so issuing batches concurrently only
//     oversubscribes the machine and makes things slower;
//   • a pure-CPU toy embedder scales with the core count.
//
// So an embedder MAY expose `max_concurrency()`; if it doesn't, we assume 1
// (serial), which is always correct if not always fastest. Opt-in rather than
// required, so third-party embedders satisfying `Embedder` keep compiling.
template <class E>
concept ConcurrencyAware = requires(const E& e) {
    { e.max_concurrency() } -> std::convertible_to<std::size_t>;
};

template <Embedder E>
[[nodiscard]] constexpr std::size_t embedder_concurrency(const E& e) noexcept {
    if constexpr (ConcurrencyAware<E>) {
        const std::size_t c = e.max_concurrency();
        return c ? c : 1;
    } else {
        return 1;
    }
}

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

    // How many `embed` calls the wrapped backend wants in flight at once.
    // 1 unless the concrete embedder opted in (see ConcurrencyAware above).
    [[nodiscard]] std::size_t max_concurrency() const { return self_->max_concurrency(); }

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::size_t dimension() const = 0;
        virtual std::string_view identity() const = 0;
        virtual Result<std::vector<Vector>> embed(std::span<const std::string>) const = 0;
        virtual std::size_t max_concurrency() const = 0;
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
        std::size_t max_concurrency() const override { return embedder_concurrency(e); }
    };
    std::shared_ptr<const Concept> self_;
};

} // namespace rag::dense
