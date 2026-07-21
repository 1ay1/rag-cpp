// tests/test_main.cpp — a minimal, dependency-free test harness + all tests.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <rag/rag.hpp>

namespace {
struct Case { std::string name; std::function<void()> fn; };
std::vector<Case>& registry() { static std::vector<Case> r; return r; }
int g_failures = 0;
int g_checks   = 0;
std::string g_current;

struct Reg { Reg(std::string n, std::function<void()> f) { registry().push_back({std::move(n), std::move(f)}); } };
#define TEST(name) \
    static void name(); \
    static Reg reg_##name(#name, name); \
    static void name()

#define CHECK(cond) do { ++g_checks; if (!(cond)) { \
    ++g_failures; std::printf("  FAIL [%s]: %s (line %d)\n", g_current.c_str(), #cond, __LINE__); } } while(0)

#define CHECK_EQ(a,b) do { ++g_checks; if (!((a)==(b))) { \
    ++g_failures; std::printf("  FAIL [%s]: %s == %s (line %d)\n", g_current.c_str(), #a, #b, __LINE__); } } while(0)

#define REQUIRE(cond) do { ++g_checks; if (!(cond)) { \
    ++g_failures; std::printf("  REQUIRE FAIL [%s]: %s (line %d)\n", g_current.c_str(), #cond, __LINE__); return; } } while(0)
}

TEST(strong_ids_are_distinct) {
    using namespace rag;
    DocId d{5};
    CHECK_EQ(d.get(), 5u);
    CHECK(DocId::invalid().valid() == false);
    CHECK(ChunkId{3}.valid());
}

TEST(result_monad) {
    using namespace rag;
    Result<int> ok = 42;
    Result<int> err = fail<int>(Errc::not_found, "nope");
    CHECK(ok.has_value());
    CHECK_EQ(*ok, 42);
    CHECK(!err.has_value());
    CHECK_EQ(err.error().code, Errc::not_found);
}

TEST(porter_stemmer) {
    using rag::text::porter_stem;
    CHECK_EQ(porter_stem("running"), "run");
    CHECK_EQ(porter_stem("happiness"), "happi");
    CHECK_EQ(porter_stem("relational"), "relat");
    CHECK_EQ(porter_stem("caresses"), "caress");
}

TEST(tokenizer_drops_stopwords_and_stems) {
    rag::text::Tokenizer tok;
    auto t = tok.tokenize("The cats are running quickly");
    bool has_cat = false, has_run = false, has_the = false;
    for (auto& s : t) { if (s=="cat") has_cat=true; if (s=="run") has_run=true; if (s=="the") has_the=true; }
    CHECK(has_cat); CHECK(has_run); CHECK(!has_the);
}

TEST(chunker_produces_chunks) {
    std::string body = "# Title\n\nFirst paragraph here.\n\n## Section\n\nSecond paragraph body.\n";
    auto chunks = rag::text::chunk_document(rag::DocId{0}, body, {});
    CHECK(!chunks.empty());
    bool ctx_ok = false;
    for (auto& c : chunks) if (c.context.find("Title") != std::string::npos) ctx_ok = true;
    CHECK(ctx_ok);
}

TEST(bm25_ranks_exact_term) {
    rag::lexical::Bm25Index idx;
    idx.add(0, "the quick brown fox jumps");
    idx.add(1, "lazy dogs sleep all day");
    idx.add(2, "quick foxes are clever animals");
    idx.finalize();
    auto hits = idx.search("quick fox", 3);
    REQUIRE(!hits.empty());
    CHECK(hits[0].chunk.get() != 1u);
}

TEST(bm25_serialize_roundtrip) {
    rag::lexical::Bm25Index idx;
    idx.add(0, "alpha beta gamma");
    idx.add(1, "beta delta epsilon");
    idx.finalize();
    auto blob = idx.serialize();
    auto back = rag::lexical::Bm25Index::deserialize(blob);
    REQUIRE(back.has_value());
    auto h1 = idx.search("beta", 2);
    auto h2 = back->search("beta", 2);
    CHECK_EQ(h1.size(), h2.size());
}

TEST(simd_dot_and_normalize) {
    std::vector<float> a{3, 4};
    rag::dense::normalize(a);
    CHECK(std::abs(a[0]-0.6f) < 1e-5f);
    CHECK(std::abs(a[1]-0.8f) < 1e-5f);
    std::vector<float> b{1,0,0}, c{1,0,0};
    CHECK(std::abs(rag::dense::cosine(b,c) - 1.0f) < 1e-5f);
}

TEST(hash_embedder_deterministic) {
    rag::dense::HashEmbedder e(64);
    std::vector<std::string> in{"hello world"};
    auto r1 = e.embed(in); auto r2 = e.embed(in);
    REQUIRE(r1.has_value()); REQUIRE(r2.has_value());
    REQUIRE(r1->size()==1);
    CHECK(std::abs(rag::dense::cosine((*r1)[0], (*r2)[0]) - 1.0f) < 1e-5f);
}

TEST(hnsw_finds_nearest) {
    rag::index::HnswIndex idx(rag::index::HnswConfig{});
    idx.add(0, std::vector<float>{1,0,0});
    idx.add(1, std::vector<float>{0,1,0});
    idx.add(2, std::vector<float>{0,0,1});
    idx.add(3, std::vector<float>{0.9f,0.1f,0});
    auto hits = idx.search(std::vector<float>{1,0,0}, 2);
    REQUIRE(!hits.empty());
    CHECK_EQ(hits[0].chunk.get(), 0u);
}

TEST(hnsw_serialize_roundtrip) {
    rag::index::HnswIndex idx(rag::index::HnswConfig{});
    for (int i = 0; i < 20; ++i) {
        std::vector<float> v(8, 0);
        v[i % 8] = 1.0f; v[(i+1)%8] = 0.5f;
        idx.add(static_cast<std::uint32_t>(i), v);
    }
    auto blob = idx.serialize();
    auto back = rag::index::HnswIndex::deserialize(blob);
    REQUIRE(back.has_value());
    CHECK_EQ(back->size(), idx.size());
}

TEST(rrf_fuses_lists) {
    using namespace rag;
    std::vector<fusion::RankedList> lists;
    lists.push_back({{Hit{ChunkId{1},Score{9}}, Hit{ChunkId{2},Score{8}}}, 1.0f});
    lists.push_back({{Hit{ChunkId{2},Score{5}}, Hit{ChunkId{3},Score{4}}}, 1.0f});
    auto fused = fusion::rrf(lists);
    REQUIRE(!fused.empty());
    CHECK_EQ(fused[0].chunk.get(), 2u);
}

TEST(engine_hybrid_search) {
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{128}});
    engine.add("d1", "The mitochondria is the powerhouse of the cell. Cells produce energy.");
    engine.add("d2", "Rust is a systems programming language with memory safety.");
    engine.add("d3", "Photosynthesis converts sunlight into chemical energy in plants.");
    auto b = engine.build();
    REQUIRE(b.has_value());
    auto res = engine.search("cell energy production", 2);
    REQUIRE(res.has_value());
    REQUIRE(!res->empty());
    CHECK(res->front().uri != "d2");
}

TEST(engine_metadata_filter) {
    rag::Engine engine;
    engine.add("d1", "quantum entanglement physics", {{"topic","physics"}});
    engine.add("d2", "quantum computing algorithms", {{"topic","cs"}});
    engine.build();
    auto res = engine.search("quantum", 5, [](const rag::Metadata& m){
        auto it = m.find("topic"); return it != m.end() && it->second == "cs";
    });
    REQUIRE(res.has_value());
    for (auto& r : *res) CHECK_EQ(r.uri, std::string("d2"));
}

// ─── Filtered-HNSW pre-filter ─────────────────────────────────────────
TEST(hnsw_filtered_search) {
    rag::index::HnswIndex idx(rag::index::HnswConfig{});
    for (int i = 0; i < 50; ++i) {
        std::vector<float> v(8, 0);
        v[i % 8] = 1.0f; v[(i+3) % 8] = 0.5f;
        idx.add(static_cast<std::uint32_t>(i), v);
    }
    // Only allow even ids.
    auto allow = [](std::uint32_t id) { return id % 2 == 0; };
    std::vector<float> q(8, 0); q[0] = 1.0f;
    auto hits = idx.search_filtered(q, 5, allow);
    REQUIRE(!hits.empty());
    for (auto& h : hits) CHECK(h.chunk.get() % 2 == 0);
}

// ─── Persistence container round-trip ──────────────────────────────────
TEST(container_roundtrip_and_crc) {
    rag::store::Container c;
    c.put(rag::store::Tag::docs, "hello-docs-payload");
    c.put(rag::store::Tag::bm25, std::string(1000, 'x'));
    c.set_flags(rag::store::kHasEmbeddings);
    auto blob = c.serialize();
    auto back = rag::store::Container::parse(blob);
    REQUIRE(back.has_value());
    REQUIRE(back->get(rag::store::Tag::docs) != nullptr);
    CHECK_EQ(*back->get(rag::store::Tag::docs), std::string("hello-docs-payload"));
    CHECK(back->flags() == rag::store::kHasEmbeddings);
    // Corrupt one byte in the payload region: CRC must reject.
    blob[40] ^= 0xFF;
    auto bad = rag::store::Container::parse(blob);
    CHECK(!bad.has_value());
}

TEST(corpus_save_load_ragdb) {
    std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/ragcpp_test.ragdb";
    {
        rag::Engine engine;
        engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{64}});
        engine.add("a.md", "# Vectors\n\nEmbeddings map text to a dense space.", {{"k","v"}});
        engine.add("b.md", "# Lexical\n\nBM25 scores exact term overlap.");
        engine.build();
        auto s = engine.save(path);
        REQUIRE(s.has_value());
    }
    auto loaded = rag::index::Corpus::load(path);
    REQUIRE(loaded.has_value());
    CHECK_EQ(loaded->document_count(), 2u);
    CHECK(loaded->chunk_count() >= 2u);
    // Lexical search survives the round-trip.
    auto hits = loaded->lexical_search("bm25 term", 3);
    CHECK(!hits.empty());
    std::remove(path.c_str());
}

// ─── Code-aware chunker ───────────────────────────────────────────
TEST(code_chunker_splits_on_functions) {
    std::string py =
        "import os\n\n"
        "def alpha():\n    return 1\n\n"
        "def beta(x):\n    return x + 1\n\n"
        "class Gamma:\n    def method(self):\n        return 2\n";
    auto chunks = rag::loaders::chunk_code(rag::DocId{0}, ".py", py);
    CHECK(chunks.size() >= 2);
    CHECK_EQ(rag::loaders::detect_language(".py"), rag::loaders::Language::python);
}

// ─── HTML → text ────────────────────────────────────────────────
TEST(html_to_text_strips_tags) {
    std::string html = "<html><head><style>x{}</style></head><body>"
                       "<h1>Title</h1><p>Hello &amp; welcome</p><script>bad()</script></body></html>";
    auto text = rag::loaders::html_to_text(html);
    CHECK(text.find("Title") != std::string::npos);
    CHECK(text.find("Hello & welcome") != std::string::npos);
    CHECK(text.find("bad()") == std::string::npos);   // script dropped
    CHECK(text.find("x{}") == std::string::npos);      // style dropped
}

// ─── Reranker (local scoring fn) via pipeline stage ────────────────────────
TEST(scorefn_reranker_reorders) {
    // A reranker that prefers passages containing the exact word "target".
    rag::rerank::ScoreFnReranker rr([](std::string_view q, std::string_view p) {
        (void)q; return p.find("target") != std::string_view::npos ? 1.0f : 0.0f;
    });
    std::vector<std::string> passages = {"nothing here", "the target is here", "also nothing"};
    auto scores = rr.rerank("q", passages);
    REQUIRE(scores.has_value());
    REQUIRE(scores->size() == 3);
    CHECK((*scores)[1] > (*scores)[0]);
}

TEST(prf_expand_stage_grows_query) {
    rag::Engine engine;
    engine.add("d1", "neural networks deep learning gradient descent backpropagation");
    engine.add("d2", "neural networks activation functions training epochs");
    engine.build();
    rag::pipeline::Pipeline p;
    p.add(std::make_shared<rag::pipeline::PrfExpandStage>())
     .add(std::make_shared<rag::pipeline::HybridRetrieveStage>())
     .add(std::make_shared<rag::pipeline::TopKStage>());
    std::vector<std::string> trace;
    auto hits = p.run(engine.corpus(), "neural", 5, {}, &trace);
    REQUIRE(hits.has_value());
    bool expanded = false;
    for (auto& t : trace) if (t.find("prf:") != std::string::npos) expanded = true;
    CHECK(expanded);
}

int main() {
    std::printf("Running %zu test cases...\n", registry().size());
    for (auto& c : registry()) {
        g_current = c.name;
        int before = g_failures;
        c.fn();
        std::printf("  %s %s\n", (g_failures==before ? "ok  " : "FAIL"), c.name.c_str());
    }
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
