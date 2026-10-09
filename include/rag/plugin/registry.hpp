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
//   * A third party ships a shared library with a registration hook (see
//     plugin/loader.hpp) and their backend becomes usable by name.
//
// A Registry is a VALUE: build it, then read it. There is no global one to
// mutate; `Registries::builtin()` (builder.hpp) is the immutable set of
// built-in backends, and a host that adds plugins copies it and adds to its
// copy. Sharing one across threads while adding to it is the host's to guard.
//
// Totality: a factory returns Result<T> — a bad config is a value, not a throw.

#include "rag/core/types.hpp"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rag::plugin {

using Json = nlohmann::json;

struct Registries;

template <class Interface>
class Registry {
public:
    // A factory may build nested components (retry, fallback) through the
    // same registries it was called with.
    using Factory = std::function<Result<Interface>(const Json& config, const Registries& all)>;

    struct Entry {
        Factory     factory;
        std::string description;   // e.g. "OpenAI embeddings API (keys: model, api_key, dim)"
    };

    bool register_factory(std::string name, Factory factory) {
        return register_described(std::move(name), std::move(factory), {});
    }

    bool register_described(std::string name, Factory factory, std::string description) {
        auto [it, inserted] = factories_.insert_or_assign(
            std::move(name), Entry{std::move(factory), std::move(description)});
        (void)it;
        return inserted;
    }

    [[nodiscard]] bool contains(std::string_view name) const {
        return factories_.find(name) != factories_.end();
    }

    [[nodiscard]] Result<Interface> create(std::string_view name, const Json& config,
                                           const Registries& all) const {
        auto it = factories_.find(name);
        if (it == factories_.end())
            return fail<Interface>(Errc::not_found,
                                   "no registered factory named '" + std::string(name) + "'");
        return it->second.factory(config, all);
    }

    [[nodiscard]] Result<Interface> create_from(const Json& spec, const Registries& all) const {
        std::string name;
        if (spec.is_string()) {
            name = spec.get<std::string>();
            return create(name, Json::object(), all);
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
        return create(name, spec, all);
    }

    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(factories_.size());
        for (const auto& [k, _] : factories_) out.push_back(k);
        return out;
    }

    [[nodiscard]] std::vector<std::pair<std::string, std::string>> describe() const {
        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(factories_.size());
        for (const auto& [k, e] : factories_) out.emplace_back(k, e.description);
        return out;
    }

    [[nodiscard]] std::size_t size() const { return factories_.size(); }

private:
    std::map<std::string, Entry, std::less<>> factories_;
};

} // namespace rag::plugin
