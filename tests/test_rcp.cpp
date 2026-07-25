// tests/test_rcp.cpp — the rag-cpp ⇄ RCP/1 integration, tested in-process.
//
// Drives rag::rcp::EngineHandler directly (no subprocess, no sockets) so the
// wire-mapping contract — capabilities, retrieve shape, filter compilation,
// funnel invariant, embed, index — is regression-guarded at build time. Uses
// the same tiny harness idiom as test_main.cpp.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <rag/rag.hpp>
#include <rag/rcp/rcp.hpp>

namespace {
struct Case { std::string name; std::function<void()> fn; };
std::vector<Case>& registry() { static std::vector<Case> r; return r; }
int g_failures = 0, g_checks = 0;
std::string g_current;
struct Reg { Reg(std::string n, std::function<void()> f) { registry().push_back({std::move(n), std::move(f)}); } };
#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; \
    std::printf("  FAIL [%s]: %s (line %d)\n", g_current.c_str(), #cond, __LINE__); } } while(0)
#define REQUIRE(cond) do { ++g_checks; if (!(cond)) { ++g_failures; \
    std::printf("  REQUIRE FAIL [%s]: %s (line %d)\n", g_current.c_str(), #cond, __LINE__); return; } } while(0)

rag::Engine make_engine() {
    rag::Engine e;
    e.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{128}});
    e.add("doc://a", "HNSW is a graph index for approximate nearest neighbour search.",
          {{"lang", "en"}, {"topic", "retrieval"}});
    e.add("doc://b", "The Eiffel Tower is a lattice tower in Paris, France.",
          {{"lang", "en"}, {"topic", "landmarks"}});
    e.add("doc://c", "Photosynthesis converts light into chemical energy in plants.",
          {{"lang", "fr"}, {"topic", "biology"}});
    e.build();
    return e;
}
} // namespace

TEST(rcp_capabilities_reflect_engine) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}.with_index(true).filter_on("topic")};
    auto caps = h.capabilities();
    CHECK(caps.retrieve.has_value());   // always advertised
    CHECK(caps.embed.has_value());      // engine has a HashEmbedder
    CHECK(caps.index.has_value());      // opted in
    CHECK(caps.filter.has_value());     // filter_on advertised a field
    CHECK(caps.citations);              // every hit carries a citation
    CHECK(h.info().name == "rag-cpp");
}

TEST(rcp_embed_not_advertised_without_embedder) {
    using namespace rag::rcp;
    rag::Engine bare;                   // no embedder attached
    bare.add("doc://x", "lexical only text"); bare.build();
    EngineHandler h{bare, Options{}};
    CHECK(!h.capabilities().embed.has_value());   // must not claim embed
    CHECK(h.capabilities().retrieve.has_value());  // BM25 retrieve still works
}

TEST(rcp_retrieve_shape_and_ranking) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}};
    auto r = h.retrieve(Json{{"query", "nearest neighbour search index"}, {"k", 2}});
    REQUIRE(r.has_value());
    const Json& res = *r;
    REQUIRE(res.contains("hits") && res["hits"].is_array());
    CHECK(res["hits"].size() <= 2);
    REQUIRE(!res["hits"].empty());
    const Json& top = res["hits"][0];
    CHECK(top.contains("id") && top["id"].is_string());
    CHECK(top.contains("score") && top["score"].is_number());
    CHECK(top.contains("citation"));
    CHECK(top.contains("trust"));
    CHECK(top["unit"] == "chunk");
    // ordering: descending score
    for (std::size_t i = 1; i < res["hits"].size(); ++i)
        CHECK(res["hits"][i-1]["score"].get<double>() >= res["hits"][i]["score"].get<double>());
    // usage block present (§7.7)
    CHECK(res.contains("usage"));
}

TEST(rcp_retrieve_rejects_bad_params) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}};
    // missing query
    CHECK(!h.retrieve(Json{{"k", 3}}).has_value());
    // k = 0
    CHECK(!h.retrieve(Json{{"query", "x"}, {"k", 0}}).has_value());
    // funnel inversion: candidateK < k
    auto bad = h.retrieve(Json{{"query", "x"}, {"k", 10}, {"candidateK", 2}});
    REQUIRE(!bad.has_value());
    CHECK(bad.error().code == ::rcp::errc::InvalidParams);
}

TEST(rcp_filter_compiles_and_selects) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}.filter_on("topic").filter_on("lang")};
    auto r = h.retrieve(Json{{"query", "tower paris landmark"}, {"k", 5},
                             {"filter", Json{{"field", "topic"}, {"op", "eq"}, {"value", "landmarks"}}}});
    REQUIRE(r.has_value());
    for (const auto& hit : (*r)["hits"])
        CHECK(hit["id"].get<std::string>().find("doc://b") != std::string::npos);
    // filtering an UNadvertised field must be rejected -32602
    auto bad = h.retrieve(Json{{"query", "x"}, {"k", 3},
                               {"filter", Json{{"field", "secret"}, {"op", "eq"}, {"value", "1"}}}});
    REQUIRE(!bad.has_value());
    CHECK(bad.error().code == ::rcp::errc::InvalidParams);
}

TEST(rcp_embed_roundtrip) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}};
    auto r = h.embed(Json{{"input", Json::array({"hello", "world"})}, {"encoding", "float"}});
    REQUIRE(r.has_value());
    CHECK((*r)["dimension"].get<int>() == 128);
    REQUIRE((*r)["vectors"].is_array());
    CHECK((*r)["vectors"].size() == 2);
    CHECK((*r)["vectors"][0].size() == 128);
}

TEST(rcp_index_add_then_retrieve) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}.with_index(true)};
    auto add = h.index_add(Json{{"documents", Json::array({
        Json{{"id", "doc://new"}, {"text", "Quantum entanglement links particle states."},
             {"meta", Json{{"topic", "physics"}}}}})}});
    REQUIRE(add.has_value());
    CHECK((*add)["ids"].size() == 1);
    CHECK((*add)["ids"][0] == "doc://new");
    auto r = h.retrieve(Json{{"query", "quantum entanglement particle"}, {"k", 1}});
    REQUIRE(r.has_value());
    REQUIRE(!(*r)["hits"].empty());
    CHECK((*r)["hits"][0]["id"].get<std::string>().find("doc://new") != std::string::npos);
}

TEST(rcp_index_write_gated_when_readonly) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}};   // index not enabled
    auto add = h.index_add(Json{{"documents", Json::array()}});
    REQUIRE(!add.has_value());
    CHECK(add.error().code == ::rcp::errc::CapabilityMissing);
}

TEST(rcp_index_add_is_upsert_not_duplicate) {
    using namespace rag::rcp;
    auto engine = make_engine();
    EngineHandler h{engine, Options{}.with_index(true)};
    // §7.10: re-adding the same explicit id MUST replace, not duplicate.
    std::size_t before = engine.corpus().live_document_count();
    Json doc{{"documents", Json::array({
        Json{{"id", "doc://dup"}, {"text", "first version of the text"}}})}};
    REQUIRE(h.index_add(doc).has_value());
    std::size_t after_first = engine.corpus().live_document_count();
    CHECK(after_first == before + 1);
    // upsert with new text under the SAME id
    Json doc2{{"documents", Json::array({
        Json{{"id", "doc://dup"}, {"text", "second and updated version of the text"}}})}};
    REQUIRE(h.index_add(doc2).has_value());
    CHECK(engine.corpus().live_document_count() == after_first);   // no duplicate
    // and delete by uri is idempotent
    auto del = h.index_delete(Json{{"ids", Json::array({"doc://dup"})}});
    REQUIRE(del.has_value());
    CHECK((*del)["deleted"].get<int>() == 1);
    auto del2 = h.index_delete(Json{{"ids", Json::array({"doc://dup"})}});
    REQUIRE(del2.has_value());
    CHECK((*del2)["deleted"].get<int>() == 0);   // idempotent
}

int main() {
    std::printf("running %zu rcp integration tests\n", registry().size());
    for (auto& c : registry()) { g_current = c.name; c.fn(); }
    std::printf("%s: %d checks, %d failures\n", g_failures ? "FAIL" : "OK", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
