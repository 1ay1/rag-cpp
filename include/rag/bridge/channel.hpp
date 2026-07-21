#pragma once
// rag/bridge/channel.hpp — the polyglot RPC seam.
//
// The plugin registry (rag/plugin) lets any C++ type register as a backend by
// name. This file lets that backend live in ANOTHER PROCESS or ANOTHER LANGUAGE
// (Python, Node, a Rust service, a hosted API): a `Channel` is a request/reply
// pipe that carries JSON, and every "remote" backend (RemoteEmbedder,
// RemoteReranker, RemoteRetriever, RemoteGraph) is written once against this
// seam. Concrete channels:
//
//   * ProcessChannel — spawn a subprocess and speak newline-delimited JSON over
//     its stdin/stdout (bridge/process.hpp). Zero deps; the universal bridge for
//     a Python engine/graph.
//   * HttpChannel    — POST JSON to a REST endpoint using the library's existing
//     injectable HttpTransport (bridge/http_channel.hpp).
//
// Protocol (deliberately minimal, JSON-RPC-ish): a request is
//   {"method": "<name>", "params": { ... }}
// and a reply is either
//   {"ok": true, "result": { ... }}   or   {"error": {"code": "...", "message": "..."}}
// The remote decides how to implement each method; the C++ side only knows the
// method names its Remote* wrappers call (embed / rerank / retrieve / graph_*).

#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "rag/core/types.hpp"

namespace rag::bridge {

using Json = nlohmann::json;

// A Channel is a synchronous JSON request → JSON response transport. It hides
// whether the peer is a subprocess, an HTTP server, or an in-memory fake (tests).
// Totality: transport/decode failures come back as Result, never exceptions.
struct Channel {
    virtual ~Channel() = default;

    // Send `params` under `method`, get the peer's `result` object back.
    // Implementations map the wire envelope (ok/result/error) onto Result.
    [[nodiscard]] virtual Result<Json> call(std::string_view method, const Json& params) = 0;

    // A human-readable identity for cache keys / logs (e.g. "python:my_engine").
    [[nodiscard]] virtual std::string identity() const = 0;
};

// Helper: unwrap a `{"ok":...,"result":...,"error":...}` envelope into Result.
[[nodiscard]] inline Result<Json> unwrap_envelope(const Json& reply) {
    if (reply.is_object()) {
        if (auto e = reply.find("error"); e != reply.end() && !e->is_null()) {
            std::string msg = "remote error";
            if (e->is_object()) {
                if (auto m = e->find("message"); m != e->end() && m->is_string())
                    msg = m->get<std::string>();
            } else if (e->is_string()) {
                msg = e->get<std::string>();
            }
            return fail<Json>(Errc::transport_error, std::move(msg));
        }
        if (auto r = reply.find("result"); r != reply.end())
            return *r;
        // No explicit envelope: treat the whole object as the result.
        return reply;
    }
    return reply; // arrays / scalars pass through
}

} // namespace rag::bridge
