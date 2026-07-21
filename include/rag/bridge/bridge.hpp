#pragma once
// rag/bridge/bridge.hpp — polyglot bridge umbrella + registry helpers.
//
// Include this to plug in engines / retrievers / rerankers / graphs written in
// ANY language, reached over a subprocess pipe or an HTTP endpoint:
//
//   auto ch = rag::bridge::open_channel({{"transport","process"},
//                                        {"argv", {"python3","ragcpp_server.py"}}});
//   auto emb = rag::bridge::RemoteEmbedder{*ch, 384, "python:my_model"};
//   engine.with_embedder(rag::dense::AnyEmbedder{std::move(emb)});
//
// or straight from config through the plugin registry (the "process" / "http"
// embedder + reranker factories are registered by src/bridge/register.cpp):
//
//   engine.with_embedder_spec({{"type","process"},
//                              {"argv",{"python3","ragcpp_server.py"}},
//                              {"dim",384}});
//
// See PLUGINS.md § "Polyglot backends".

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "rag/bridge/channel.hpp"
#include "rag/bridge/http_channel.hpp"
#include "rag/bridge/process.hpp"
#include "rag/bridge/remote.hpp"
#include "rag/core/types.hpp"

namespace rag::bridge {

// Build a Channel from a JSON spec. `transport` (or `type`) selects:
//   "process": { "argv": [str,...], "env": [str,...], "cwd": str, "name": str }
//   "http"   : { "host","port","tls","base_path","headers":{...},"name" }
[[nodiscard]] Result<std::shared_ptr<Channel>> open_channel(const Json& spec);

// Registers "process" and "http" embedder + reranker factories with the plugin
// registry. Referenced by rag::plugin::ensure_builtins_registered() so config
// like {"type":"process", "argv":[...], "dim":384} builds a RemoteEmbedder.
void ensure_bridge_registered() noexcept;

} // namespace rag::bridge
