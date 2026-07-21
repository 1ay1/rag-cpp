// src/plugin/loader.cpp — directory scan for the plugin loader.

#include "rag/plugin/loader.hpp"

#include <algorithm>
#include <filesystem>

namespace rag::plugin {

std::vector<LoadedPlugin> load_plugin_dir(std::string_view dir, std::vector<Error>* errors) {
    namespace fs = std::filesystem;
    std::vector<LoadedPlugin> loaded;

    std::error_code ec;
    fs::path root{dir};
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return loaded;

#if defined(_WIN32)
    constexpr std::string_view suffix = ".dll";
#elif defined(__APPLE__)
    constexpr std::string_view suffix = ".dylib";
#else
    constexpr std::string_view suffix = ".so";
#endif

    std::vector<fs::path> candidates;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            candidates.push_back(entry.path());
        }
    }
    // Deterministic load order.
    std::sort(candidates.begin(), candidates.end());

    for (const auto& p : candidates) {
        auto r = load_plugin(p.string());
        if (r) {
            loaded.push_back(std::move(*r));
        } else if (errors) {
            errors->push_back(r.error());
        }
    }
    return loaded;
}

} // namespace rag::plugin
