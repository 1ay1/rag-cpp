// examples/polyglot/python_backend.cpp
//
// Drive a PYTHON backend from rag-cpp. The host spawns `python3 ragcpp_server.py`
// as a subprocess and speaks JSON over the pipe — the Python side provides the
// embedder AND a full retriever, with no C++ knowledge of Python and no Python
// knowledge of C++ beyond the wire protocol.

#include <rag/rag.hpp>

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string server = argc > 1 ? argv[1] : "examples/polyglot/ragcpp_server.py";

    rag::plugin::ensure_builtins_registered(); // pulls in process/http transports

    // 1) Use the Python model as the engine's embedder, purely from config.
    rag::Engine engine;
    auto attach = engine.with_embedder_spec(nlohmann::json{
        {"type", "process"},
        {"argv", {"python3", server}},
        {"dim", 384},
        {"name", "python:reference"},
    });
    if (!attach) {
        std::printf("attach failed: %s\n", attach.error().message.c_str());
        std::printf("(is python3 on PATH? pass the server path as argv[1])\n");
        return 1;
    }

    engine.add("a", "The Eiffel Tower is in Paris.");
    engine.add("b", "Plants perform photosynthesis in their leaves.");
    engine.build();
    if (auto hits = engine.search("landmark in the French capital", 2)) {
        std::printf("[embedder-via-python] top = %s (%zu hits)\n",
                    hits->empty() ? "(none)" : (*hits)[0].uri.c_str(), hits->size());
    }

    // 2) Use the Python side as a SELF-CONTAINED retriever (it owns its own index).
    auto ch = rag::bridge::open_channel(nlohmann::json{
        {"transport", "process"},
        {"argv", {"python3", server}},
        {"name", "python:retriever"},
    });
    if (ch) {
        rag::bridge::RemoteRetriever retr{*ch, "python:retriever"};
        if (auto r = retr.retrieve("great wall", 3)) {
            std::printf("[remote-retriever] %zu hits from python engine:\n", r->size());
            for (const auto& h : *r)
                std::printf("   %-4s score=%.3f  %s\n", h.uri.c_str(), h.score, h.text.c_str());
        }
        // GraphRAG-style op over the same channel.
        if (auto g = retr.op("global")) std::printf("[remote-graph] %s\n", g->dump().c_str());
    }
    return 0;
}
