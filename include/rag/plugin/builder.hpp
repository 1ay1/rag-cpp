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
//   * resolve<I>  — build a nested sub-component through the SAME registry
//                   (what fallback/retry need), one call.
//   * register_embedder / register_reranker (+ _fn variants) — register a
//     factory that returns your CONCEPT TYPE directly; the AnyX wrap and the
//     description are handled for you. No macro, so no comma trap.
//
// A complete new embedder becomes:
//
//   struct MeanPool { std::size_t dimension() const; std::string_view identity() const;
//                     Result<std::vector<Vector>> embed(std::span<const std::string>) const; };
//
//   register_embedder("meanpool", "average of char-hash vectors (keys: dim)",
//       [](Config c) -> Result<MeanPool> { return MeanPool{c.get("dim", 384)}; });
//
// That's it — reachable by name from config/CLI/C-ABI, self-described in
// describe(), and it degrades cleanly if it returns an error.

#include <optional>
#include <string>
#include <string_view>

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
class Config {
public:
    explicit Config(const Json& j) : j_(j) {}
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

private:
    const Json& j_;
};

// ── resolve: build a nested sub-component through the registry ────────────────
// The one call decorators need. `resolve<AnyEmbedder>(spec)` builds whatever the
// spec names, recursively. Works because Registry invokes factories unlocked.
template <class Interface>
[[nodiscard]] Result<Interface> resolve(const Json& spec) {
    return Registry<Interface>::instance().create_from(spec);
}

// ── Typed registration helpers ───────────────────────────────────────────────
// You return your CONCEPT TYPE (or its AnyX); the wrap + description are handled.
// No macro → no top-level-comma-in-braces trap. Call these at load time, e.g.
// from a `register_all()` you invoke once, or from a plugin's rag_plugin_register.

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

// Register an embedder factory. `fn` takes a Config and returns Result<T> where
// T is any Embedder concept model OR AnyEmbedder. Description powers describe().
template <class Fn>
bool register_embedder(std::string name, std::string description, Fn fn) {
    Registry<AnyEmbedder>::instance().register_described(
        std::move(name),
        [fn = std::move(fn)](const Json& j) -> Result<AnyEmbedder> {
            auto r = fn(Config{j});
            if (!r) return std::unexpected(r.error());
            return detail::wrap_any<AnyEmbedder>(std::move(*r));
        },
        std::move(description));
    return true;
}

// Register a reranker factory, same contract.
template <class Fn>
bool register_reranker(std::string name, std::string description, Fn fn) {
    Registry<AnyReranker>::instance().register_described(
        std::move(name),
        [fn = std::move(fn)](const Json& j) -> Result<AnyReranker> {
            auto r = fn(Config{j});
            if (!r) return std::unexpected(r.error());
            return detail::wrap_any<AnyReranker>(std::move(*r));
        },
        std::move(description));
    return true;
}

} // namespace rag::plugin
