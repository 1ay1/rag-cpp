// examples/rcp_server.cpp — rag-cpp as a Retrieval Context Protocol server.
//
// A complete, conformant RCP/1 endpoint over a live rag::Engine. The engine is
// seeded with a tiny demo corpus (a deterministic HashEmbedder keeps it
// dependency-free and reproducible for conformance runs), then handed to the
// RCP front-end. Any RCP client — the reference clients in ~/projects/rcp, or
// the conformance checker — can now drive retrieval / embed / index over the
// wire.
//
//   Build:  cmake --build build --target ragcpp_rcp_server
//   stdio:  ./ragcpp_rcp_server
//   http :  ./ragcpp_rcp_server --http 8000
//   check:  python3 ~/projects/rcp/conformance/check.py -- ./ragcpp_rcp_server
//
// Everything below the corpus seeding is the ENTIRE integration surface: build
// an Engine, describe what to advertise, serve. That is the framework promise.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <rag/rag.hpp>
#include <rag/rcp/rcp.hpp>

namespace {

// A small, self-contained demo corpus so the server answers real queries out of
// the box (and the conformance checker has something to retrieve).
rag::Engine build_demo_engine() {
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{256}});

    engine.add("doc://eiffel", "The Eiffel Tower is a wrought-iron lattice tower on "
               "the Champ de Mars in Paris, France, completed in 1889.",
               {{"lang", "en"}, {"topic", "landmarks"}});
    engine.add("doc://photosynthesis", "Photosynthesis is the process by which plants "
               "convert light energy into chemical energy stored in glucose.",
               {{"lang", "en"}, {"topic", "biology"}});
    engine.add("doc://hnsw", "HNSW is a graph index for approximate nearest-neighbour "
               "search, giving logarithmic query time over high-dimensional vectors.",
               {{"lang", "en"}, {"topic", "retrieval"}});
    engine.add("doc://rrf", "Reciprocal rank fusion merges ranked lists from lexical "
               "and dense retrievers into one robust hybrid ranking.",
               {{"lang", "en"}, {"topic", "retrieval"}});
    engine.build();
    return engine;
}

} // namespace

int main(int argc, char** argv) {
    rag::Engine engine = build_demo_engine();

    // Describe the server: identity + which RCP capabilities to advertise. A
    // writable index and metadata filtering on `lang`/`topic` are turned on so
    // the endpoint exercises the full method surface for conformance.
    auto opts = rag::rcp::Options{}
                    .named("rag-cpp", "0.1.0")
                    .with_index(/*writable=*/true);
    opts.filter_on("lang", "keyword").filter_on("topic", "keyword");

    rag::rcp::ServerBuilder server{engine};
    server.options(std::move(opts));

    if (argc >= 3 && std::strcmp(argv[1], "--http") == 0) {
        auto r = server.serve_http(static_cast<std::uint16_t>(std::atoi(argv[2])));
        if (!r) { std::fprintf(stderr, "http serve failed: %s\n", r.error().message.c_str()); return 1; }
        return 0;
    }
    server.serve_stdio();
    return 0;
}
