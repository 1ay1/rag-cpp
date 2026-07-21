#pragma once
// rag/plugin/loader.hpp — load out-of-tree backends from shared libraries.
//
// The registry (registry.hpp) lets a backend register itself by name; the
// built-ins do so from src/plugin/builtins.cpp. This header closes the loop for
// THIRD parties: they compile a shared object that contains RAG_REGISTER blocks
// (or a `rag_plugin_register` entry point) and drop it somewhere; at runtime you
// `load_plugin("/path/to/libmyembedder.so")` and their names appear in every
// registry — no recompile of the app, no core change.
//
// Two supported conventions, tried in order:
//   1. Static registrars: the .so's global initializers run on dlopen and call
//      Registry::instance().register_factory() directly. Nothing else needed.
//   2. An explicit hook: the .so exports `extern "C" void rag_plugin_register()`
//      which we look up and call after load (useful when the toolchain would
//      otherwise strip "unused" static objects).
//
// The registries are Meyers singletons keyed on the Interface type. As long as
// the plugin and host share the same rag headers/ABI, both see the same
// instance. (Build plugins with the same compiler + flags as the host.)

#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/plugin/registry.hpp"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace rag::plugin {

// An opaque handle keeping a loaded plugin alive. Destroying it unloads the
// library (dlclose). Keep it alive for as long as any factory it registered may
// be invoked. `keep()` leaks it deliberately for process-lifetime plugins.
class LoadedPlugin {
public:
    LoadedPlugin() = default;
    LoadedPlugin(void* handle, std::string path) : handle_(handle), path_(std::move(path)) {}
    LoadedPlugin(LoadedPlugin&& o) noexcept
        : handle_(o.handle_), path_(std::move(o.path_)) { o.handle_ = nullptr; }
    LoadedPlugin& operator=(LoadedPlugin&& o) noexcept {
        if (this != &o) { close(); handle_ = o.handle_; path_ = std::move(o.path_); o.handle_ = nullptr; }
        return *this;
    }
    LoadedPlugin(const LoadedPlugin&) = delete;
    LoadedPlugin& operator=(const LoadedPlugin&) = delete;
    ~LoadedPlugin() { close(); }

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] void* native_handle() const noexcept { return handle_; }

    // Detach: never unload. Use for plugins that must live for the whole process.
    void keep() noexcept { handle_ = nullptr; }

private:
    void close() noexcept {
        if (!handle_) return;
#if defined(_WIN32)
        ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
        ::dlclose(handle_);
#endif
        handle_ = nullptr;
    }
    void* handle_ = nullptr;
    std::string path_;
};

// The optional explicit hook a plugin may export.
using PluginRegisterFn = void (*)();
inline constexpr const char* kPluginRegisterSymbol = "rag_plugin_register";

// Load one shared library. Its static initializers run immediately (convention
// 1); if it exports `rag_plugin_register`, that is called too (convention 2).
[[nodiscard]] inline Result<LoadedPlugin> load_plugin(std::string_view path) {
#if defined(_WIN32)
    HMODULE h = ::LoadLibraryA(std::string(path).c_str());
    if (!h) return fail<LoadedPlugin>(Errc::not_found,
                                      "LoadLibrary failed for '" + std::string(path) + "'");
    if (auto sym = ::GetProcAddress(h, kPluginRegisterSymbol))
        reinterpret_cast<PluginRegisterFn>(sym)();
    return LoadedPlugin{reinterpret_cast<void*>(h), std::string(path)};
#else
    ::dlerror(); // clear
    void* h = ::dlopen(std::string(path).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = ::dlerror();
        return fail<LoadedPlugin>(Errc::not_found,
                                  "dlopen failed for '" + std::string(path) + "': " +
                                  (e ? e : "unknown"));
    }
    ::dlerror();
    if (void* sym = ::dlsym(h, kPluginRegisterSymbol))
        reinterpret_cast<PluginRegisterFn>(sym)();
    return LoadedPlugin{h, std::string(path)};
#endif
}

// Load every plugin whose filename matches the platform's shared-lib suffix in
// `dir` (non-recursive). Returns the handles; caller keeps them alive. Missing
// dir is not an error (returns empty). Individual load failures are collected in
// `errors` if provided, and skipped.
[[nodiscard]] std::vector<LoadedPlugin>
load_plugin_dir(std::string_view dir, std::vector<Error>* errors = nullptr);

} // namespace rag::plugin
