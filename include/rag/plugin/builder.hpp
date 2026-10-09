#pragma once
// rag/plugin/builder.hpp — the ergonomic layer for AUTHORING plugins.
//
// registry.hpp is the mechanism; this is the authoring UX. Adding a backend
// should be one small function with no boilerplate, no macro comma traps, and no
// raw nlohmann::json. This header gives you:
//
//   * Config      — a non-throwing typed view over the JSON blob:
//                      cfg.get("model", "bge")     // with default
//                      cfg.require("api_key")      // error if absent
//                      cfg.sub("primary")          // a nested component spec
//   * cfg.resolve<I>(spec) — build a nested sub-component through the SAME
//                   registries (what fallback/retry need), one call.
//   * Registries::embedder / ::reranker — register a factory that returns
//     your CONCEPT TYPE directly; the AnyX wrap and the description are
//     handled for you.
//
// A complete new embedder becomes:
//
//   struct MeanPool { std::size_t dimension() const; std::string_view identity() const;
//                     Result<std::vector<Vector>> embed(std::span<const std::string>) const; };
//
//   auto regs = Registries::builtin();
//   regs.embedder("meanpool", "average of char-hash vectors (keys: dim)",
//       [](Config c) -> Result<MeanPool> { return MeanPool{c.get("dim", 384)}; });
//
// That's it — reachable by name from config/CLI/C-ABI, self-described in
// describe(), and it degrades cleanly if it returns an error.

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "rag/bridge/channel.hpp"
#include "rag/bridge/process.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/plugin/registry.hpp"
#include "rag/rerank/reranker.hpp"

namespace rag::plugin {

using AnyEmbedder = ::rag::dense::AnyEmbedder;
using AnyReranker = ::rag::rerank::AnyReranker;

// ── Config: a total, typed view over a component spec ────────────────────────
// Never throws. A missing or wrong-typed key falls back to the default (get) or
// yields a typed Error you can propagate (require). Wraps the raw JSON so a
// factory author never touches nlohmann::json directly.
struct HostIo;

class Config {
public:
    Config(const Json& j, const Registries& all) : j_(j), all_(&all) {}
    // Config is a non-owning VIEW over a spec that must outlive it (in practice
    // the const Json& a factory receives). Binding a temporary would dangle, so
    // that is a compile error rather than UB.
    Config(Json&&) = delete;

    // Value for `key`, or `dflt` if absent / null / wrong type. Total.
    template <class T>
    [[nodiscard]] T get(std::string_view key, T dflt) const {
        if (auto it = j_.find(std::string(key)); it != j_.end() && !it->is_null()) {
            try { return it->template get<T>(); } catch (...) { return dflt; }
        }
        return dflt;
    }

    // Convenience overload so string literals deduce std::string, not const char*.
    [[nodiscard]] std::string get(std::string_view key, const char* dflt) const {
        return get<std::string>(key, std::string(dflt));
    }

    [[nodiscard]] bool has(std::string_view key) const {
        auto it = j_.find(std::string(key));
        return it != j_.end() && !it->is_null();
    }

    // Required value. Returns a typed error if the key is absent — the factory
    // just propagates it: `auto key = c.require<std::string>("api_key"); if (!key) ...`
    template <class T>
    [[nodiscard]] Result<T> require(std::string_view key) const {
        auto it = j_.find(std::string(key));
        if (it == j_.end() || it->is_null())
            return fail<T>(Errc::invalid_argument,
                           "missing required config key '" + std::string(key) + "'");
        try { return it->template get<T>(); }
        catch (...) {
            return fail<T>(Errc::invalid_argument,
                           "config key '" + std::string(key) + "' has the wrong type");
        }
    }

    // A nested sub-spec (an embedder inside a decorator, say). Returns the raw
    // Json so it can be handed to resolve<>(). Errors if the key is absent.
    [[nodiscard]] Result<Json> sub(std::string_view key) const {
        auto it = j_.find(std::string(key));
        if (it == j_.end())
            return fail<Json>(Errc::invalid_argument,
                              "missing nested spec '" + std::string(key) + "'");
        return *it;
    }

    [[nodiscard]] const Json& raw() const noexcept { return j_; }

    // The host's IO for this build (registries' `io`).
    [[nodiscard]] const HostIo& io() const noexcept;

    // Build a nested component through the registries this one came from.
    template <class Interface>
    [[nodiscard]] Result<Interface> resolve(const Json& spec) const;

private:
    const Json&       j_;
    const Registries* all_;
};

// ── resolve: build a nested sub-component through the registry ────────────────
namespace detail {
// Wrap a factory that returns `Concrete` (a concept model, or the AnyX itself)
// into one that returns `AnyX`. Accepts both so you can return AnyEmbedder
// directly from a decorator, or a bare concept model from a leaf backend.
template <class AnyX, class Concrete>
AnyX wrap_any(Concrete&& c) {
    if constexpr (std::is_same_v<std::decay_t<Concrete>, AnyX>)
        return std::forward<Concrete>(c);
    else
        return AnyX{std::forward<Concrete>(c)};
}
} // namespace detail

// What the network and subprocess backends need from the host. rag-cpp
// opens no socket and starts no process itself: an HTTP backend posts through
// `http`, a {"transport":"process"} bridge talks to the channel `spawn` opens,
// and a retry waits with `wait`. Leave one empty and the backends that need
// it report Errc::unavailable when built.
struct HostIo {
    std::shared_ptr<dense::HttpTransport> http;
    std::function<Result<std::shared_ptr<bridge::Channel>>(const bridge::ProcessSpec&)> spawn;
    std::function<void(std::chrono::milliseconds)> wait;
};

// Every component registry, as one value. builtin() is the set this library
// ships; a host that adds its own backends copies it and adds to the copy.
struct Registries {
    Registry<AnyEmbedder> embedders;
    Registry<AnyReranker> rerankers;
    HostIo                io;   // see HostIo; empty in builtin()

    // The built-in backends (hash, ollama, openai, …, retry, fallback, and the
    // process/http bridges). Built once; immutable. Its `io` is empty, so the
    // network and process backends resolve by name but report unavailable;
    // copy it and set `io` to use them.
    [[nodiscard]] static const Registries& builtin();

    // builtin() with the host's IO.
    [[nodiscard]] static Registries with_io(HostIo io) {
        Registries r = builtin();
        r.io = std::move(io);
        return r;
    }

    // Register an embedder factory. `fn` takes a Config and returns Result<T>
    // where T is any Embedder concept model OR AnyEmbedder.
    template <class Fn>
    void embedder(std::string name, std::string description, Fn fn) {
        embedders.register_described(std::move(name),
            [fn = std::move(fn)](const Json& j, const Registries& all) -> Result<AnyEmbedder> {
                auto r = fn(Config{j, all});
                if (!r) return std::unexpected(r.error());
                return detail::wrap_any<AnyEmbedder>(std::move(*r));
            },
            std::move(description));
    }

    // Register a reranker factory, same contract.
    template <class Fn>
    void reranker(std::string name, std::string description, Fn fn) {
        rerankers.register_described(std::move(name),
            [fn = std::move(fn)](const Json& j, const Registries& all) -> Result<AnyReranker> {
                auto r = fn(Config{j, all});
                if (!r) return std::unexpected(r.error());
                return detail::wrap_any<AnyReranker>(std::move(*r));
            },
            std::move(description));
    }

    [[nodiscard]] Result<AnyEmbedder> make_embedder(const Json& spec) const {
        return embedders.create_from(spec, *this);
    }
    [[nodiscard]] Result<AnyReranker> make_reranker(const Json& spec) const {
        return rerankers.create_from(spec, *this);
    }
};

inline const HostIo& Config::io() const noexcept { return all_->io; }

template <class Interface>
Result<Interface> Config::resolve(const Json& spec) const {
    if constexpr (std::is_same_v<Interface, AnyEmbedder>) return all_->make_embedder(spec);
    else return all_->make_reranker(spec);
}

} // namespace rag::plugin
