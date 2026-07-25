#pragma once
// rag/rcp/server.hpp — the one-liner front door.
//
// The framework layer: turn a host-owned `rag::Engine` into a running RCP/1
// server over stdio or HTTP with a single call, or assemble a customized server
// with a fluent builder. This is the acp-cpp ergonomics goal — a host plugs its
// engine in and gets a conformant protocol endpoint, defaults sane, every knob
// reachable.
//
//   // Minimal: read-only hybrid retrieval over stdio.
//   rag::Engine engine = build_my_engine();
//   rag::rcp::serve_stdio(engine);
//
//   // Customized: writable index + graph + an external reranker, over HTTP.
//   rag::rcp::ServerBuilder(engine)
//       .options(rag::rcp::Options{}.named("docs", "1.0").with_index(true).with_graph())
//       .on_rerank(my_reranker_fn)
//       .filter_on("lang", "keyword")
//       .serve_http(8000);

#include "rag/engine.hpp"
#include "rag/rcp/handler.hpp"

#include <rcp/server.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace rag::rcp {

// ─────────────────────────────────────────────────────────────────────────────
// ServerBuilder — fluent assembly of an RCP server around a live Engine.
// Holds the Engine by reference (host owns its lifetime), accumulates Options
// and Hooks, then materialises an ::rcp::Server<EngineHandler> on serve/build.
// ─────────────────────────────────────────────────────────────────────────────
class ServerBuilder {
public:
    explicit ServerBuilder(Engine& engine) : engine_(engine) {}

    ServerBuilder& options(Options o) { opts_ = std::move(o); return *this; }

    // Capability toggles that read at the call site (delegate to Options).
    ServerBuilder& named(std::string n, std::string v) { opts_.named(std::move(n), std::move(v)); return *this; }
    ServerBuilder& with_index(bool writable) { opts_.with_index(writable); return *this; }
    ServerBuilder& with_graph(bool on = true) { opts_.with_graph(on); return *this; }
    ServerBuilder& with_feedback(bool on = true) { opts_.with_feedback(on); return *this; }
    ServerBuilder& with_memory(bool on = true) { opts_.with_memory(on); return *this; }
    ServerBuilder& with_reranker(rag::rerank::AnyReranker r) { opts_.with_reranker(std::move(r)); return *this; }
    ServerBuilder& with_splade(rag::sparse::SpladeIndex s)   { opts_.with_splade(std::move(s)); return *this; }
    ServerBuilder& with_colbert(rag::late::TokenEmbedder e)  { opts_.with_colbert(std::move(e)); return *this; }
    ServerBuilder& with_generator(rag::query::Generator g)   { opts_.with_generator(std::move(g)); return *this; }
    ServerBuilder& filter_on(std::string field, std::string type = "keyword") {
        opts_.filter_on(std::move(field), std::move(type)); return *this;
    }
    ServerBuilder& max_k(std::size_t k) { opts_.max_k = k; return *this; }

    // Method-override hooks — each REPLACES the built-in mapping for that method
    // and, where relevant, advertises the corresponding capability.
    ServerBuilder& on_retrieve(std::function<Result(const Json&)> f) { hooks_.retrieve = std::move(f); return *this; }
    ServerBuilder& on_rerank(std::function<Result(const Json&)> f)   { hooks_.rerank = std::move(f); return *this; }
    ServerBuilder& on_embed(std::function<Result(const Json&)> f)    { hooks_.embed = std::move(f); return *this; }
    ServerBuilder& on_graph(std::function<Result(const Json&)> f)    { hooks_.graph = std::move(f); opts_.enable_graph = true; return *this; }
    ServerBuilder& on_transform(std::function<Result(const Json&)> f){ hooks_.transform = std::move(f); return *this; }
    ServerBuilder& on_feedback(std::function<Result(const Json&)> f) { hooks_.feedback = std::move(f); return *this; }
    ServerBuilder& on_index_add(std::function<Result(const Json&)> f){ hooks_.index_add = std::move(f); return *this; }
    ServerBuilder& on_index_delete(std::function<Result(const Json&)> f){ hooks_.index_delete = std::move(f); return *this; }

    // Materialise the SDK server. The handler holds Engine& + the accumulated
    // config; the returned Server owns framing/dispatch/gating.
    [[nodiscard]] ::rcp::Server<EngineHandler> build() {
        return ::rcp::Server<EngineHandler>{EngineHandler{engine_, opts_, hooks_}};
    }

    void serve_stdio() { build().serve_stdio(); }
    [[nodiscard]] ::rcp::Result<void> serve_http(std::uint16_t port) { return build().serve_http(port); }

private:
    Engine& engine_;
    Options opts_{};
    Hooks   hooks_{};
};

// ── Free-function shortcuts (the 90% case) ──────────────────────────────────

// Serve a read-only hybrid-retrieval RCP server over stdio. Blocks until EOF /
// shutdown. `opts` lets a caller flip capabilities without the builder.
inline void serve_stdio(Engine& engine, Options opts = {}) {
    ::rcp::Server<EngineHandler>{EngineHandler{engine, std::move(opts)}}.serve_stdio();
}

// Serve over loopback HTTP on `port`. Blocks; returns on socket error.
[[nodiscard]] inline ::rcp::Result<void>
serve_http(Engine& engine, std::uint16_t port, Options opts = {}) {
    return ::rcp::Server<EngineHandler>{EngineHandler{engine, std::move(opts)}}.serve_http(port);
}

} // namespace rag::rcp
