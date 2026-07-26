#pragma once
// rag/plugin/registry.hpp — the framework's universal extension point.
//
// Concepts (core/concepts.hpp) make the framework extensible at COMPILE time:
// any struct that models `Embedder` is an embedder, no inheritance needed. The
// `AnyX` type-erasers make it extensible at LINK time: a runtime pipeline can
// hold heterogeneous stages. This file adds the third and final axis —
// extensibility at LOAD / CONFIG time:
//
//   * A backend is registered under a string NAME with a factory that builds it
//     from a JSON config blob.
//   * Anything downstream — the CLI, the C ABI, a config file, a REST server —
//     can then construct "the thing called \"ollama\"" from `{"model": ...}`
//     WITHOUT the core library knowing that backend exists at compile time.
//   * A third party ships a shared library that self-registers on load
//     (see plugin/loader.hpp) and their backend becomes usable by name.
//
// The registry is the single choke point through which every kind of component
// (embedders, rerankers, retrievers, tokenizers, generators, chunkers, ...)
// becomes swappable by name. One template serves them all.
//
// Totality: a factory returns Result<T> — a bad config is a value, not a throw.
// Thread-safety: registration and lookup are mutex-guarded; registration is
// meant to happen once at startup (static initializers) but is safe any time.

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "rag/core/types.hpp"

namespace rag::plugin {

using Json = nlohmann::json;

// A Registry<Interface> maps a name → a factory that builds an `Interface`
// (typically one of the `AnyX` type-erased handles) from a JSON config.
//
// `Interface` must be movable. It is usually AnyEmbedder / AnyReranker /
// AnyRetriever / ... — copyable shared handles — but any movable type works.
template <class Interface>
class Registry {
public:
    using Factory = std::function<Result<Interface>(const Json& config)>;

    // A one-line human description of what a registered factory builds and the
    // config keys it reads. Optional; enables `describe()` for --help / discovery.
    struct Entry {
        Factory     factory;
        std::string description;   // e.g. "OpenAI embeddings API (keys: model, api_key, dim)"
    };

    // The process-wide registry for this Interface. Meyers singleton: the
    // static local is initialized on first use, so self-registering plugins in
    // other translation units / shared objects all target the same instance.
    static Registry& instance() {
        static Registry reg;
        return reg;
    }

    // Register (or replace) a factory under `name`. Returns false if it
    // replaced an existing entry (useful for detecting accidental clobber).
    bool register_factory(std::string name, Factory factory) {
        return register_described(std::move(name), std::move(factory), {});
    }

    // Register with a human description for introspection. Same semantics as
    // register_factory; the description shows up in describe().
    bool register_described(std::string name, Factory factory, std::string description) {
        std::lock_guard lk(mu_);
        auto [it, inserted] = factories_.insert_or_assign(
            std::move(name), Entry{std::move(factory), std::move(description)});
        (void)it;
        return inserted;
    }

    [[nodiscard]] bool contains(std::string_view name) const {
        std::lock_guard lk(mu_);
        return factories_.find(std::string(name)) != factories_.end();
    }

    // Build `name` from `config`. Errc::not_found if no such factory.
    [[nodiscard]] Result<Interface> create(std::string_view name, const Json& config) const {
        Factory f;
        {
            std::lock_guard lk(mu_);
            auto it = factories_.find(std::string(name));
            if (it == factories_.end())
                return fail<Interface>(Errc::not_found,
                                       "no registered factory named '" + std::string(name) + "'");
            f = it->second.factory; // copy under lock, invoke unlocked (factory may recurse)
        }
        return f(config);
    }

    // Config convenience: a blob shaped `{"type": "ollama", ...}` (or the alias
    // key "backend"/"name") is dispatched to the matching factory, passing the
    // whole object through as config. This is the one call a config-file loader
    // needs.
    [[nodiscard]] Result<Interface> create_from(const Json& spec) const {
        std::string name;
        if (spec.is_string()) {
            name = spec.get<std::string>();
            return create(name, Json::object());
        }
        if (!spec.is_object())
            return fail<Interface>(Errc::invalid_argument,
                                   "component spec must be a string or object");
        for (const char* key : {"type", "backend", "name", "kind"}) {
            if (auto it = spec.find(key); it != spec.end() && it->is_string()) {
                name = it->get<std::string>();
                break;
            }
        }
        if (name.empty())
            return fail<Interface>(Errc::invalid_argument,
                                   "component spec missing a 'type' field");
        return create(name, spec);
    }

    // All registered names, sorted. For `rag list` / introspection / help text.
    [[nodiscard]] std::vector<std::string> names() const {
        std::lock_guard lk(mu_);
        std::vector<std::string> out;
        out.reserve(factories_.size());
        for (const auto& [k, _] : factories_) out.push_back(k);
        return out;
    }

    // (name, description) pairs, sorted by name. Powers `--help` / `rag list`:
    // a user can see every registered backend AND what config it takes without
    // reading source. Descriptions are empty for factories registered via the
    // bare register_factory / RAG_REGISTER path.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> describe() const {
        std::lock_guard lk(mu_);
        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(factories_.size());
        for (const auto& [k, e] : factories_) out.emplace_back(k, e.description);
        return out;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard lk(mu_);
        return factories_.size();
    }

private:
    Registry() = default;
    mutable std::mutex mu_;
    std::map<std::string, Entry, std::less<>> factories_;
};

// Static self-registration helper. Construct one at namespace scope to register
// a factory before main() runs:
//
//   static const rag::plugin::AutoRegister<rag::AnyEmbedder>
//       _reg_hash{"hash", [](const Json&) -> rag::Result<rag::AnyEmbedder> {
//           return rag::AnyEmbedder{rag::dense::HashEmbedder{}};
//       }};
//
// or via the RAG_REGISTER macro below.
template <class Interface>
struct AutoRegister {
    AutoRegister(std::string name, typename Registry<Interface>::Factory factory) {
        Registry<Interface>::instance().register_factory(std::move(name), std::move(factory));
    }
};

} // namespace rag::plugin

// RAG_REGISTER(Interface, "name", <lambda returning Result<Interface>>)
// Declares a uniquely-named static registrar. Use at file scope in a .cpp.
#define RAG_REGISTER_IMPL2(iface, name, factory, ctr)                              \
    static const ::rag::plugin::AutoRegister<iface> _rag_autoreg_##ctr{name, factory}
#define RAG_REGISTER_IMPL(iface, name, factory, ctr) RAG_REGISTER_IMPL2(iface, name, factory, ctr)
#define RAG_REGISTER(iface, name, factory) RAG_REGISTER_IMPL(iface, name, factory, __COUNTER__)
