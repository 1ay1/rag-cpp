#pragma once
// rag/rcp/rcp.hpp — single include for the RCP/1 server front-end.
//
// rag-cpp speaks the Retrieval Context Protocol (github.com/1ay1/rcp): any RCP
// client can drive this engine's hybrid retrieval, embedding, GraphRAG, and
// index over stdio or HTTP. The subsystem is layered like the rest of the
// library; include a lower header alone for a smaller compile.
//
//   Layer 1 (algebra):  error.hpp    — rag::Error ↔ rcp::Error mapping
//   Layer 2 (mapping):   convert.hpp  — wire JSON ↔ engine values (§7.7, §8)
//   Layer 3 (handler):   handler.hpp  — rag::Engine as an rcp::Handler
//   Layer 4 (facade):    server.hpp   — ServerBuilder + serve_stdio/serve_http
//
// Quick start:
//
//   #include <rag/rcp/rcp.hpp>
//   int main() {
//       rag::Engine engine;
//       engine.add("doc://1", "…text…");
//       engine.build();
//       rag::rcp::serve_stdio(engine);   // now an RCP/1 server
//   }

#include "rag/rcp/error.hpp"
#include "rag/rcp/convert.hpp"
#include "rag/rcp/handler.hpp"
#include "rag/rcp/server.hpp"
