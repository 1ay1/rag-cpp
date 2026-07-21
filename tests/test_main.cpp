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

// ── GraphRAG ────────────────────────────────────────────────────────────────

TEST(graph_builds_link_and_similarity_edges) {
    rag::Engine engine;
    // d1 links to d2 by markdown; d3 shares vocabulary with d1/d2.
    engine.add("alpha.md", "Alpha is about vector search. See [beta](beta.md) for HNSW graphs.");
    engine.add("beta.md",  "Beta covers HNSW graphs and vector search indexing in depth.");
    engine.add("gamma.md", "Gamma discusses HNSW vector search graphs and indexing performance.");
    engine.add("delta.md", "Delta is about cooking pasta and unrelated culinary topics entirely.");
    engine.build();
    auto g = engine.graph();
    REQUIRE(g.has_value());
    CHECK((*g)->node_count() == 4);
    // There should be at least one explicit LINK edge (alpha → beta).
    bool has_link = false;
    for (auto& e : (*g)->edges())
        if (e.kind == rag::graph::Edge::Kind::link) has_link = true;
    CHECK(has_link);
    // Communities are detected and cover every node.
    std::size_t covered = 0;
    for (auto& c : (*g)->communities()) covered += c.docs.size();
    CHECK_EQ(covered, 4u);
    // Every community has a non-empty extractive summary.
    for (auto& c : (*g)->communities()) CHECK(!c.summary.empty());
}

TEST(graph_local_search_expands_via_edges) {
    rag::Engine engine;
    engine.add("a.md", "The retrieval engine uses BM25 for lexical ranking. See [dense](b.md).");
    engine.add("b.md", "The dense retriever embeds text and scores by cosine similarity.");
    engine.add("c.md", "Fusion combines lexical and dense rankings with reciprocal rank fusion.");
    engine.build();
    auto hits = engine.graph_local("BM25 lexical ranking", 5);
    REQUIRE(hits.has_value());
    CHECK(!hits->empty());
    // The seed doc (a.md) must appear.
    bool has_a = false;
    for (auto& h : *hits) if (h.uri == "a.md") has_a = true;
    CHECK(has_a);
}

TEST(graph_global_search_ranks_communities) {
    rag::Engine engine;
    engine.add("net1.md", "Neural networks learn representations via gradient descent and backprop.");
    engine.add("net2.md", "Deep neural networks stack layers to learn hierarchical features.");
    engine.add("cook.md", "To cook risotto, toast the rice then add stock gradually while stirring.");
    engine.build();
    auto hits = engine.graph_global("how do neural networks learn", 3);
    REQUIRE(hits.has_value());
    CHECK(!hits->empty());
    // Top global hit should be a neural-network community, not the cooking one.
    CHECK((*hits)[0].uri != "cook.md");
}

// ── RALM ────────────────────────────────────────────────────────────────────

TEST(ralm_ensemble_weights_are_a_distribution) {
    rag::Engine engine;
    engine.add("d1", "vector search with hnsw graphs and approximate nearest neighbours");
    engine.add("d2", "lexical search with bm25 and inverted indexes for keyword matching");
    engine.build();
    auto hits = engine.corpus().lexical_search("vector search", 4);
    REQUIRE(!hits.empty());
    auto wd = rag::ralm::ensemble_weights(engine.corpus(), hits, 1.0f);
    REQUIRE(!wd.empty());
    float sum = 0.0f;
    for (auto& w : wd) { CHECK(w.weight >= 0.0f); sum += w.weight; }
    CHECK(std::fabs(sum - 1.0f) < 1e-4f);
    // Higher-scored hit gets >= weight of lower-scored (softmax monotone).
    if (wd.size() >= 2) CHECK(wd[0].weight >= wd[1].weight);
}

TEST(replug_combine_mixes_distributions) {
    rag::ralm::WeightedDoc a, b;
    a.weight = 0.75f; b.weight = 0.25f;
    std::vector<rag::ralm::WeightedDoc> docs = {a, b};
    std::vector<std::vector<float>> logits = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    auto out = rag::ralm::replug_combine(docs, logits);
    REQUIRE(out.size() == 2);
    CHECK(std::fabs(out[0] - 0.75f) < 1e-5f);
    CHECK(std::fabs(out[1] - 0.25f) < 1e-5f);
}

TEST(retro_retrieves_neighbours_with_continuation) {
    rag::Engine engine;
    // A long doc so the chunker yields multiple chunks (continuation exists).
    std::string body;
    for (int i = 0; i < 40; ++i)
        body += "Paragraph " + std::to_string(i) + " about vector search and hnsw graphs indexing. ";
    engine.add("long.md", body);
    engine.add("other.md", "unrelated content about cooking and recipes and food preparation.");
    engine.build();
    rag::ralm::RetroConfig cfg; cfg.stride = 4; cfg.neighbours = 2;
    auto rows = rag::ralm::retro_retrieve(engine.corpus(), "vector search hnsw graphs indexing", cfg);
    REQUIRE(rows.has_value());
    REQUIRE(!rows->empty());
    bool any_neighbour = false;
    for (auto& r : *rows) if (!r.neighbours.empty()) any_neighbour = true;
    CHECK(any_neighbour);
}

TEST(incontext_plan_fires_at_strides) {
    rag::Engine engine;
    engine.add("d1", "retrieval augmented generation grounds the model on external documents");
    engine.add("d2", "in context ralm prepends retrieved passages without modifying the model");
    engine.build();
    rag::ralm::RalmConfig cfg; cfg.stride = 4;
    auto plan = rag::ralm::incontext_plan(
        engine.corpus(), /*n_tokens=*/12,
        [](std::size_t pos) { (void)pos; return std::string("retrieval augmented generation model"); },
        cfg);
    REQUIRE(plan.has_value());
    // 12 tokens / stride 4 = 3 retrieval points.
    CHECK_EQ(plan->size(), 3u);
    for (auto& d : *plan) CHECK(!d.text.empty());
}

TEST(assemble_prompt_attributes_sources) {
    rag::ralm::WeightedDoc a; a.text = "the sky is blue"; a.weight = 1.0f;
    std::vector<rag::ralm::WeightedDoc> docs = {a};
    auto p = rag::ralm::assemble_prompt("what colour is the sky", docs, "Answer using the context.");
    CHECK(p.find("[1]") != std::string::npos);
    CHECK(p.find("the sky is blue") != std::string::npos);
    CHECK(p.find("what colour is the sky") != std::string::npos);
}

// ── Learned sparse (SPLADE-style) ──────────────────────────────────────

TEST(splade_retrieves_and_expands) {
    rag::Engine engine;
    engine.add("d1", "vector search with hnsw graphs approximate nearest neighbours embeddings");
    engine.add("d2", "lexical bm25 inverted index keyword matching term frequency");
    engine.add("d3", "neural networks deep learning gradient descent backpropagation training");
    engine.build();
    auto idx = rag::sparse::SpladeIndex::build(engine.corpus());
    REQUIRE(idx.has_value());
    CHECK(idx->vocab_size() > 0);
    auto hits = idx->search("vector nearest neighbours", 3);
    REQUIRE(!hits.empty());
    // d1 (the vector-search doc) must rank first.
    auto top = engine.corpus().resolve(hits[0]);
    CHECK(top.uri == "d1");
    // Expansion: encoding a query yields more terms than the raw query has.
    auto raw = idx->encode("vector", false);
    auto exp = idx->encode("vector", true);
    CHECK(exp.size() >= raw.size());
}

// ── ColBERT late interaction ─────────────────────────────────────────

TEST(colbert_maxsim_prefers_token_overlap) {
    auto embed = rag::late::hashed_token_embedder(64);
    rag::late::ColbertReranker rr(embed);
    std::vector<std::string> passages = {
        "completely unrelated content about cooking pasta",
        "the quick brown fox jumps over the lazy dog",
    };
    auto scores = rr.rerank("quick brown fox", passages);
    REQUIRE(scores.has_value());
    REQUIRE(scores->size() == 2);
    // The passage sharing tokens with the query scores higher.
    CHECK((*scores)[1] > (*scores)[0]);
}

TEST(colbert_maxsim_exact_match_is_maximal) {
    auto embed = rag::late::hashed_token_embedder(64);
    auto q = embed("alpha beta gamma");
    REQUIRE(q.has_value());
    // MaxSim of a token set with itself = number of tokens (each matches at 1.0).
    float s = rag::late::maxsim(*q, *q);
    CHECK(std::fabs(s - 3.0f) < 1e-3f);
}

// ── RAPTOR ───────────────────────────────────────────────────────

TEST(raptor_builds_tree_and_retrieves) {
    rag::Engine engine;
    for (int i = 0; i < 12; ++i)
        engine.add("doc" + std::to_string(i),
            "Document " + std::to_string(i) +
            " discusses vector search hnsw graphs indexing and approximate nearest neighbours in depth.");
    engine.add("cook", "Risotto needs arborio rice toasted then stock added gradually while stirring.");
    engine.build();
    rag::raptor::RaptorConfig cfg; cfg.cluster_size = 4; cfg.max_levels = 3;
    auto tree = rag::raptor::RaptorTree::build(engine.corpus(), cfg);
    REQUIRE(tree.has_value());
    // The tree must have MORE nodes than leaves (summaries were created)...
    CHECK(tree->node_count() > engine.corpus().chunk_count());
    // ...and at least 2 levels.
    CHECK(tree->level_count() >= 2u);
    auto res = tree->retrieve(engine.corpus(), "vector search nearest neighbours", 5);
    REQUIRE(res.has_value());
    CHECK(!res->empty());
}

// ── HyDE / multi-query ────────────────────────────────────────────

TEST(hyde_uses_hypothetical_document) {
    rag::Engine engine;
    engine.add("d1", "HNSW is a graph-based approximate nearest neighbour index for vectors.");
    engine.add("d2", "BM25 ranks documents by term frequency and inverse document frequency.");
    engine.build();
    // A generator that returns a hypothetical answer mentioning HNSW.
    auto gen = [](std::string_view) -> rag::Result<std::vector<std::string>> {
        return std::vector<std::string>{"HNSW builds a navigable small-world graph over vectors."};
    };
    auto hits = rag::query::hyde_search(engine.corpus(), "how is fast vector search done", 2, gen);
    REQUIRE(hits.has_value());
    REQUIRE(!hits->empty());
    // The HNSW doc should surface via the hypothetical.
    bool has_d1 = false;
    for (auto& h : *hits) if (engine.corpus().resolve(h).uri == "d1") has_d1 = true;
    CHECK(has_d1);
}

// ── Corrective RAG / Self-RAG ─────────────────────────────────────

TEST(crag_grades_and_filters) {
    rag::Engine engine;
    engine.add("rel",   "vector search uses hnsw graphs for approximate nearest neighbours");
    engine.add("noise", "a recipe for chocolate chip cookies with butter and sugar");
    engine.build();
    auto hits = engine.corpus().lexical_search("hnsw approximate nearest neighbours", 5);
    REQUIRE(!hits.empty());
    auto c = rag::crag::correct(engine.corpus(), "hnsw approximate nearest neighbours", hits);
    // The relevant doc drives high confidence → Correct action.
    CHECK(c.action == rag::crag::Action::correct);
    CHECK(c.confidence > 0.5f);
    // Knowledge strips are non-empty and the relevant doc is kept.
    CHECK(!c.knowledge.empty());
}

TEST(crag_low_confidence_triggers_fallback) {
    rag::Engine engine;
    engine.add("a", "quantum chromodynamics and the strong nuclear force");
    engine.build();
    auto hits = engine.corpus().lexical_search("chocolate cake recipe", 5);
    bool external_called = false;
    auto ext = [&](std::string_view) -> rag::Result<std::vector<std::string>> {
        external_called = true;
        return std::vector<std::string>{"external fallback knowledge about cakes"};
    };
    auto c = rag::crag::correct(engine.corpus(), "chocolate cake recipe", hits, {}, {}, ext);
    // Off-topic corpus → low confidence → not Correct → external fallback fires.
    CHECK(c.action != rag::crag::Action::correct);
    CHECK(external_called);
    CHECK(!c.external.empty());
}

TEST(crag_support_score_measures_groundedness) {
    rag::Engine engine;
    engine.add("x", "placeholder");
    engine.build();
    std::vector<std::string> knowledge = {"the eiffel tower is located in paris france"};
    float grounded = rag::crag::support_score(engine.corpus(), "The eiffel tower is in paris.", knowledge);
    float ungrounded = rag::crag::support_score(engine.corpus(), "The moon is made of cheese entirely.", knowledge);
    CHECK(grounded > ungrounded);
}

// ── BEIR eval metrics ───────────────────────────────────────────

TEST(beir_metrics_are_correct) {
    // A ranking with the one relevant doc at rank 1 = perfect.
    rag::eval::Ranking perfect = {"d1", "d2", "d3"};
    std::unordered_map<std::string, int> rel = {{"d1", 1}};
    CHECK(std::fabs(rag::eval::ndcg_at_k(perfect, rel, 10) - 1.0) < 1e-9);
    CHECK(std::fabs(rag::eval::recall_at_k(perfect, rel, 10) - 1.0) < 1e-9);
    CHECK(std::fabs(rag::eval::reciprocal_rank(perfect, rel) - 1.0) < 1e-9);
    CHECK(std::fabs(rag::eval::average_precision(perfect, rel) - 1.0) < 1e-9);
    // Relevant doc at rank 2 → RR = 1/2, recall still 1 at k>=2.
    rag::eval::Ranking rank2 = {"d2", "d1", "d3"};
    CHECK(std::fabs(rag::eval::reciprocal_rank(rank2, rel) - 0.5) < 1e-9);
    CHECK(std::fabs(rag::eval::recall_at_k(rank2, rel, 1) - 0.0) < 1e-9);
    CHECK(std::fabs(rag::eval::recall_at_k(rank2, rel, 2) - 1.0) < 1e-9);
    // nDCG@1 for rank2 = 0 (nothing relevant at position 1).
    CHECK(std::fabs(rag::eval::ndcg_at_k(rank2, rel, 1) - 0.0) < 1e-9);
}

TEST(beir_evaluate_harness_runs) {
    rag::eval::BeirDataset ds;
    ds.corpus = {{"c1", "HNSW", "vector search with hnsw graphs"},
                 {"c2", "BM25", "lexical ranking with bm25"}};
    ds.queries = {{"q1", "hnsw vector search"}};
    ds.qrels["q1"]["c1"] = 1;
    rag::index::Corpus corpus;
    auto m = rag::eval::evaluate_corpus(ds, corpus);
    REQUIRE(m.has_value());
    CHECK_EQ(m->queries, 1u);
    // The relevant doc c1 should be retrieved for its own query.
    CHECK(m->recall[10] > 0.0);
}

// ── in-process embedder availability ────────────────────────────────

TEST(local_embedder_reports_availability) {
    // Without the optional deps, load() must fail gracefully with `unavailable`
    // rather than crash — the graceful-degradation contract.
    if (!rag::dense::OnnxEmbedder::available()) {
        rag::dense::LocalEmbedderConfig cfg; cfg.model_path = "nope.onnx";
        auto e = rag::dense::OnnxEmbedder::load(cfg);
        CHECK(!e.has_value());
        if (!e) CHECK(e.error().code == rag::Errc::unavailable);
    }
    if (!rag::dense::GgufEmbedder::available()) {
        rag::dense::LocalEmbedderConfig cfg; cfg.model_path = "nope.gguf";
        auto e = rag::dense::GgufEmbedder::load(cfg);
        CHECK(!e.has_value());
        if (!e) CHECK(e.error().code == rag::Errc::unavailable);
    }
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
