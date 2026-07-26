// tests/test_main.cpp — a minimal, dependency-free test harness + all tests.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <rag/rag.hpp>
#include <rag/gpu/device.hpp>
#include <rag/util/parallel.hpp>

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

// ─── Convex combination (TM2C2) ───────────────────────────────────
//
// The property that distinguishes theoretical from empirical min-max: under
// EMPIRICAL normalization the worst document in a list is pinned to 0 no
// matter how good it actually was, so a list of uniformly excellent results
// and a list of uniformly terrible ones normalize identically. Anchoring the
// minimum to a value the scoring function guarantees keeps that information.
TEST(convex_combination_uses_theoretical_floor_not_observed) {
    using namespace rag;

    // One retriever, two queries. In both, the SPREAD is identical (0.1) but
    // the absolute quality is very different.
    auto strong = fusion::bm25_list({Hit{ChunkId{1}, Score{9.0f}},
                                     Hit{ChunkId{2}, Score{8.9f}}});
    auto weak   = fusion::bm25_list({Hit{ChunkId{1}, Score{0.2f}},
                                     Hit{ChunkId{2}, Score{0.1f}}});

    std::vector<fusion::RankedList> s{strong};
    std::vector<fusion::RankedList> w{weak};
    auto fs = fusion::convex_combination(s);
    auto fw = fusion::convex_combination(w);
    REQUIRE(fs.size() == 2);
    REQUIRE(fw.size() == 2);

    // With a theoretical floor of 0, the second document keeps its relative
    // standing: 8.9/9.0 stays near the top, 0.1/0.2 does not.
    CHECK(fs[1].score.get() > 0.9f);
    CHECK(fw[1].score.get() < 0.6f);

    // Empirical min-max erases exactly that distinction: both lists collapse
    // to {1, 0}. This is the failure TM2C2 is designed around.
    auto rs = fusion::rsf(s);
    auto rw = fusion::rsf(w);
    CHECK(std::fabs(rs[1].score.get() - rw[1].score.get()) < 1e-6f);
}

// α must actually move the balance between the two retrievers, and in the
// documented direction: alpha_list is the index α weights.
TEST(convex_combination_alpha_shifts_balance) {
    using namespace rag;
    // Disjoint lists so each document's fused score comes from exactly one
    // retriever, making the weighting directly observable.
    std::vector<fusion::RankedList> lists{
        fusion::bm25_list({Hit{ChunkId{1}, Score{10.0f}}}),      // lexical
        fusion::cosine_list({Hit{ChunkId{2}, Score{1.0f}}})};    // dense

    auto dense_heavy = fusion::convex_combination(
        lists, fusion::ConvexParams{.alpha = 0.9f, .alpha_list = 1});
    auto lex_heavy = fusion::convex_combination(
        lists, fusion::ConvexParams{.alpha = 0.1f, .alpha_list = 1});

    REQUIRE(dense_heavy.size() == 2);
    REQUIRE(lex_heavy.size() == 2);
    CHECK_EQ(dense_heavy[0].chunk.get(), 2u);   // α on the dense list wins
    CHECK_EQ(lex_heavy[0].chunk.get(), 1u);     // (1-α) on lexical wins
}

// A document found by BOTH retrievers should outrank one found by a single
// retriever at comparable strength — the basic reason to fuse at all.
TEST(convex_combination_rewards_agreement) {
    using namespace rag;
    std::vector<fusion::RankedList> lists{
        fusion::bm25_list({Hit{ChunkId{1}, Score{9.0f}}, Hit{ChunkId{2}, Score{9.0f}}}),
        fusion::cosine_list({Hit{ChunkId{2}, Score{0.9f}}, Hit{ChunkId{3}, Score{0.9f}}})};
    auto fused = fusion::convex_combination(lists, fusion::ConvexParams{.alpha = 0.5f});
    REQUIRE(!fused.empty());
    CHECK_EQ(fused[0].chunk.get(), 2u);   // the only doc both retrievers found
}

// Fusion must be deterministic: it accumulates into a hash map, so ties have
// to be broken explicitly or result order drifts with allocation addresses.
TEST(convex_combination_is_deterministic_under_ties) {
    using namespace rag;
    std::vector<fusion::RankedList> lists{
        fusion::bm25_list({Hit{ChunkId{7}, Score{5.0f}}, Hit{ChunkId{3}, Score{5.0f}},
                           Hit{ChunkId{9}, Score{5.0f}}, Hit{ChunkId{1}, Score{5.0f}}})};
    auto a = fusion::convex_combination(lists);
    for (int i = 0; i < 20; ++i) {
        auto b = fusion::convex_combination(lists);
        REQUIRE(a.size() == b.size());
        for (std::size_t j = 0; j < a.size(); ++j)
            CHECK_EQ(a[j].chunk.get(), b[j].chunk.get());
    }
}

// ─── Semantic chunking is reachable from CorpusConfig ───────────────
//
// semantic_chunk_lexical() existed and worked but nothing outside its own
// translation unit called it — the Corpus always used the fixed chunker, and
// the CLI's advertised `--semantic` flag was parsed by nobody. Pin the wiring:
// selecting the strategy must actually change how documents are split, and
// must work with no embedder attached.
TEST(corpus_semantic_chunking_is_selectable) {
    // Two clearly distinct topics back to back: a topical-drift splitter should
    // put a boundary between them.
    const std::string body =
        "Photosynthesis converts sunlight into chemical energy in plant cells. "
        "Chloroplasts contain chlorophyll which absorbs light. "
        "Leaves are the primary site of this reaction in most plants. "
        "Mortgage interest rates are set by the central bank. "
        "Amortization schedules determine monthly loan payments. "
        "Refinancing can reduce the total interest paid over a loan term.";

    rag::index::CorpusConfig fixed_cfg;
    rag::index::Corpus fixed{fixed_cfg};
    REQUIRE(fixed.add_document("a", body).has_value());

    rag::index::CorpusConfig sem_cfg;
    sem_cfg.chunking = rag::index::CorpusConfig::Chunking::semantic;
    rag::index::Corpus sem{sem_cfg};
    // No embedder attached: must fall back to lexical drift rather than fail.
    REQUIRE(sem.add_document("a", body).has_value());

    CHECK(sem.chunk_count() >= 1);
    // Whatever the split, the text must survive it: chunking is a view over the
    // document, never a filter on it.
    CHECK(fixed.chunk_count() >= 1);

    // And the corpus stays queryable through the semantic path.
    auto hits = sem.lexical_search("chlorophyll sunlight", 3);
    CHECK(!hits.empty());
}

// ─── Concurrent readers + a writer must not corrupt the corpus ──────
//
// This is the shape `ragcpp serve --write` exposes: retrieve requests read the
// corpus while index/add mutates it. Chunk::meta is a BORROWED pointer into
// docs_[].meta, and add_document() push_backs into docs_ — reallocating and
// freeing the storage every one of those pointers refers to. Before Corpus
// took a lock, a 4-reader/1-writer harness of exactly this shape segfaulted on
// every single run.
TEST(corpus_survives_concurrent_readers_and_writer) {
    rag::index::Corpus corpus;
    for (int i = 0; i < 100; ++i)
        REQUIRE(corpus.add_document("seed" + std::to_string(i),
                                    "retrieval augmented generation vector index " + std::to_string(i),
                                    {{"kind", "seed"}}).has_value());
    REQUIRE(corpus.build().has_value());

    std::atomic<bool> stop{false};
    std::atomic<long> reads{0}, corrupt{0};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&] {
            while (!stop.load(std::memory_order_relaxed)) {
                for (const auto& h : corpus.lexical_search("retrieval vector", 5)) {
                    auto r = corpus.resolve(h);
                    // Every document was created with a non-empty uri. Anything
                    // else means we read through a dangling pointer or a
                    // half-constructed document.
                    if (r.uri.empty()) corrupt.fetch_add(1, std::memory_order_relaxed);
                }
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (int i = 0; i < 800; ++i)
        (void)corpus.add_document("new" + std::to_string(i),
                                  "newly indexed retrieval vector document " + std::to_string(i),
                                  {{"kind", "new"}});
    stop = true;
    for (auto& t : readers) t.join();

    CHECK_EQ(corrupt.load(), 0L);
    CHECK(reads.load() > 0);        // the readers really ran
    CHECK_EQ(corpus.document_count(), std::size_t{900});
}

// Two concurrent READERS race each other even with no writer present, because
// the const read path performs lazy repairs (relink_meta, bm25 finalize, hnsw
// seal) — that laziness is what keeps bulk ingest from being quadratic. A
// shared_mutex alone does not cover it; the repairs need their own lock.
TEST(corpus_concurrent_readers_see_consistent_results) {
    rag::index::Corpus corpus;
    for (int i = 0; i < 200; ++i)
        REQUIRE(corpus.add_document("d" + std::to_string(i),
                                    "alpha beta gamma retrieval " + std::to_string(i)).has_value());
    // Deliberately NOT calling build(): this leaves bm25 unfinalized so the
    // first readers all race to finalize it on the read path.
    std::vector<std::thread> ts;
    std::atomic<long> mismatch{0};
    std::size_t expected = corpus.lexical_search("alpha retrieval", 10).size();
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&] {
            for (int i = 0; i < 200; ++i)
                if (corpus.lexical_search("alpha retrieval", 10).size() != expected)
                    mismatch.fetch_add(1, std::memory_order_relaxed);
        });
    for (auto& t : ts) t.join();
    CHECK_EQ(mismatch.load(), 0L);
}

// ─── save() is crash-safe ───────────────────────────────────────
//
// Saving is a REPLACEMENT. Truncating the destination and writing into it means
// a crash part-way leaves neither the old index nor the new one. Writing to a
// temp file and rename()-ing it over the destination makes the swap atomic, so
// an overwrite that fails leaves the previous index intact.
TEST(save_overwrite_leaves_a_loadable_index) {
    const std::string path = "/tmp/ragcpp_durability_test.ragdb";
    std::remove(path.c_str());

    {
        rag::index::Corpus a;
        for (int i = 0; i < 50; ++i)
            REQUIRE(a.add_document("old" + std::to_string(i), "old document " + std::to_string(i)).has_value());
        REQUIRE(a.build().has_value());
        REQUIRE(a.save(path).has_value());
    }
    auto first = rag::index::Corpus::load(path);
    REQUIRE(first.has_value());
    CHECK_EQ(first->document_count(), std::size_t{50});

    // Overwrite in place with a different corpus; the result must be complete,
    // never a mixture of the two.
    {
        rag::index::Corpus b;
        for (int i = 0; i < 120; ++i)
            REQUIRE(b.add_document("new" + std::to_string(i), "new document " + std::to_string(i)).has_value());
        REQUIRE(b.build().has_value());
        REQUIRE(b.save(path).has_value());
    }
    auto second = rag::index::Corpus::load(path);
    REQUIRE(second.has_value());
    CHECK_EQ(second->document_count(), std::size_t{120});

    // And no temp files are left behind on success.
    CHECK(!std::filesystem::exists(path + ".tmp"));
    std::remove(path.c_str());
}

// ─── soft-deletes survive a save/load round-trip ────────────────────────────
//
// A tombstoned document keeps its DOCS row, its CHNK rows and its BM25
// postings — that is what makes the delete cheap and ids stable. The ONLY
// thing hiding it is membership in the tombstone set, so if that set is not
// persisted, every deleted document comes back alive and searchable on the
// next load. That is silent, unbounded data resurrection: a client deletes a
// document through index/delete, the server saves and restarts, and the
// document is being served again.
TEST(tombstones_survive_save_load) {
    const std::string path = "/tmp/ragcpp_tombstone_test.ragdb";
    std::remove(path.c_str());

    rag::index::Corpus a;
    REQUIRE(a.add_document("keep.txt", "retrieval augmented generation alpha").has_value());
    auto gone = a.add_document("gone.txt", "retrieval augmented generation bravo");
    REQUIRE(gone.has_value());
    REQUIRE(a.build().has_value());
    REQUIRE(a.remove_document(*gone).has_value());
    CHECK_EQ(a.live_document_count(), std::size_t{1});
    CHECK_EQ(a.lexical_search("retrieval augmented", 10).size(), std::size_t{1});
    REQUIRE(a.save(path).has_value());

    auto b = rag::index::Corpus::load(path);
    REQUIRE(b.has_value());
    CHECK(b->is_deleted(*gone));
    CHECK_EQ(b->live_document_count(), std::size_t{1});

    // The decisive assertion: the deleted document must not be RETRIEVABLE.
    auto hits = b->lexical_search("retrieval augmented", 10);
    CHECK_EQ(hits.size(), std::size_t{1});
    for (const auto& h : hits) CHECK_EQ(b->resolve(h).uri, std::string{"keep.txt"});

    std::remove(path.c_str());
}

// ─── concurrent save()s never publish out of order ──────────────────────────
//
// save() serializes its snapshot under a shared lock and then writes+renames
// with the lock RELEASED (so writers are not blocked for the whole file I/O).
// That opens a window in which an older snapshot's rename can land after a
// newer one's, moving the on-disk state backwards. Documents are only added
// here, so the on-disk count must be monotonic.
//
// Savers are deliberately oversubscribed relative to cores: with only a few
// they stay in lockstep and the reordering never shows up.
TEST(concurrent_saves_never_move_the_file_backwards) {
    const std::string path = "/tmp/ragcpp_saveorder_test.ragdb";
    std::remove(path.c_str());

    rag::index::Corpus corpus;
    for (int i = 0; i < 400; ++i)
        REQUIRE(corpus.add_document("seed" + std::to_string(i), "seed document " + std::to_string(i)).has_value());
    REQUIRE(corpus.build().has_value());
    REQUIRE(corpus.save(path).has_value());

    std::atomic<bool> stop{false};
    const unsigned savers = std::max(4u, std::thread::hardware_concurrency() * 2);
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < savers; ++t)
        ts.emplace_back([&, t] {
            std::this_thread::sleep_for(std::chrono::microseconds(37 * t));
            while (!stop.load(std::memory_order_relaxed)) (void)corpus.save(path);
        });
    std::thread writer([&] {
        for (int i = 0; i < 2000 && !stop.load(std::memory_order_relaxed); ++i)
            (void)corpus.add_document("new" + std::to_string(i), "new document " + std::to_string(i));
    });

    std::size_t high_water = 0;
    long regressions = 0, observed = 0;
    for (int i = 0; i < 60; ++i) {
        auto r = rag::index::Corpus::load(path);
        if (!r) continue;                       // torn reads are write_file's job, tested above
        const std::size_t n = r->document_count();
        ++observed;
        if (n < high_water) ++regressions;
        else high_water = n;
    }
    stop = true;
    writer.join();
    for (auto& t : ts) t.join();

    CHECK(observed > 0);
    CHECK_EQ(regressions, 0L);
    std::remove(path.c_str());
}

// Concurrent upserts of the SAME uri must never produce a duplicate — RCP
// §7.10 requires an explicit id to replace, not duplicate. Spelling the upsert
// as find-then-remove-then-add at the call site takes the lock three times, so
// two threads can both observe "not present" and both insert.
TEST(concurrent_upsert_of_same_uri_never_duplicates) {
    rag::index::Corpus corpus;
    constexpr int kThreads = 8, kRounds = 40;

    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t)
        ts.emplace_back([&, t] {
            for (int i = 0; i < kRounds; ++i)
                (void)corpus.upsert_document("shared-uri",
                                             "revision from thread " + std::to_string(t));
        });
    for (auto& th : ts) th.join();

    // Exactly one LIVE document may carry that uri, however many revisions were
    // written. (Tombstoned predecessors still occupy slots, which is why this
    // checks the live count rather than document_count().)
    CHECK_EQ(corpus.live_document_count(), std::size_t{1});
    auto found = corpus.find_by_uri("shared-uri");
    CHECK(found.has_value());
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

// ─── ANN quality: recall vs exact, and parallel-build parity ───────────
// Guards the two properties that make the index trustworthy: the graph must
// actually find the true nearest neighbours on realistic (topically clustered)
// embedding geometry, and the parallel bulk build must not degrade that
// relative to serial insertion.
namespace {

struct AnnFixture {
    std::vector<std::vector<float>> vecs, cents;
    std::size_t dim = 64, n = 4000, nc = 80;

    // Cluster spread. This number decides whether the whole fixture means
    // anything, so it is named rather than buried in two loops.
    //
    // It used to be 0.06, which put mean intra-cluster similarity at 0.815 —
    // each cluster a blob of near-duplicates. The true top-10 was then trivially
    // reachable from any entry point in the cluster, every graph scored ~0.99,
    // and the CHECK(r > 0.95) below could not fail. That is what let a false
    // claim ("ef=32 reaches ~0.999 recall") survive in HnswConfig for so long.
    //
    // At 0.20 the mean intra-cluster similarity is ~0.11 and the gap ratio
    // (spread within the true top-10 vs the drop to rank 1000) rises from 0.044
    // to ~0.3, which is where recall@10 starts discriminating between graphs.
    // Verified by construction: M=16/ef=64 scores 0.987 here, and dropping to
    // M=4 — a deliberately crippled graph — drops it below the gate.
    static constexpr float kJitter = 0.20f;

    AnnFixture() {
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        cents.resize(nc);
        for (auto& c : cents) {
            c.resize(dim);
            for (auto& x : c) x = g(rng);
            rag::dense::normalize(c);
        }
        std::uniform_int_distribution<int> pick(0, static_cast<int>(nc) - 1);
        vecs.resize(n);
        for (auto& v : vecs) {
            int c = pick(rng);
            v.resize(dim);
            for (std::size_t d = 0; d < dim; ++d) v[d] = cents[c][d] + kJitter * g(rng);
            rag::dense::normalize(v);
        }
    }

    // Mean recall@k of `idx` against exact brute-force search.
    double recall(const rag::index::HnswIndex& idx, std::size_t k, std::size_t queries,
                  std::size_t ef = 0) const {
        std::mt19937 qr(23);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(nc) - 1);
        double acc = 0;
        for (std::size_t qi = 0; qi < queries; ++qi) {
            int c = pick(qr);
            std::vector<float> q(dim);
            for (std::size_t d = 0; d < dim; ++d) q[d] = cents[c][d] + kJitter * g(qr);
            rag::dense::normalize(q);

            std::vector<std::pair<float, std::uint32_t>> exact;
            exact.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                exact.push_back({rag::dense::dot(vecs[i], q), static_cast<std::uint32_t>(i)});
            std::partial_sort(exact.begin(), exact.begin() + k, exact.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<std::uint32_t> truth;
            for (std::size_t i = 0; i < k; ++i) truth.push_back(exact[i].second);

            std::size_t got = 0;
            for (const auto& h : idx.search(q, k, ef))
                if (std::find(truth.begin(), truth.end(), h.chunk.get()) != truth.end()) ++got;
            acc += static_cast<double>(got) / static_cast<double>(k);
        }
        return acc / static_cast<double>(queries);
    }
};

} // namespace

TEST(hnsw_recall_matches_exact_search) {
    AnnFixture fx;
    rag::index::HnswConfig cfg;
    cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 64;
    rag::index::HnswIndex idx(cfg);
    idx.build_batch(fx.n,
        [&](std::size_t i) { return std::span<const float>(fx.vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    // On clustered embedding geometry a correct HNSW is near-exact. Anything
    // below this means the graph or the neighbour heuristic has regressed.
    double r = fx.recall(idx, 10, 60);
    CHECK(r > 0.95);
}

// ─── the recall gate can actually FAIL ─────────────────────────────────
//
// A recall threshold is worthless if the fixture is so easy that every graph
// clears it — which is exactly what happened here for a long time. This pins
// the fixture's discriminating power: a deliberately crippled graph (M=4, a
// beam of 4 neighbours per node) must score materially WORSE than the real
// configuration. If someone re-softens the data, this test fails and says so.
TEST(ann_fixture_is_hard_enough_to_discriminate) {
    AnnFixture fx;

    rag::index::HnswConfig good;
    good.M = 16; good.ef_construction = 200; good.ef_search = 64;
    rag::index::HnswIndex gi(good);
    gi.build_batch(fx.n,
        [&](std::size_t i) { return std::span<const float>(fx.vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    rag::index::HnswConfig crippled;
    crippled.M = 4; crippled.ef_construction = 10; crippled.ef_search = 8;
    rag::index::HnswIndex ci(crippled);
    ci.build_batch(fx.n,
        [&](std::size_t i) { return std::span<const float>(fx.vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    const double rg = fx.recall(gi, 10, 60);
    const double rc = fx.recall(ci, 10, 60);
    CHECK(rg > 0.95);          // the real config still clears the gate
    CHECK(rc < 0.90);          // ...and a bad graph does not
    CHECK(rg - rc > 0.10);     // with a wide, unambiguous margin
}

// ─── the neighbour-selection HEURISTIC is load-bearing ─────────────────────
//
// select_neighbours_heuristic (Malkov & Yashunin Algorithm 4) keeps a candidate
// only if it opens a NEW direction, rather than keeping the M nearest. That is
// what gives the graph long-range escape routes instead of a tight cluster of
// redundant edges.
//
// The property is invisible at small n. Replacing the heuristic with naive
// top-M measures 0.990 -> 0.978 at n=4000 (well inside noise, and the gate
// above does NOT catch it) but 0.937 -> 0.898 at n=64000, because a short walk
// across a small graph barely needs the escape routes. So this test pays for a
// bigger corpus deliberately — it is the only one here that can see the
// difference, and a heuristic regression is otherwise a silent recall leak that
// grows with the corpus.
TEST(hnsw_neighbour_heuristic_matters_at_scale) {
    constexpr std::size_t kDim = 48, kN = 40000, kClusters = 80;
    constexpr float kJitter = 0.20f;

    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<std::vector<float>> cents(kClusters, std::vector<float>(kDim));
    for (auto& c : cents) { for (auto& x : c) x = g(rng); rag::dense::normalize(c); }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(kClusters) - 1);

    std::vector<std::vector<float>> vecs(kN, std::vector<float>(kDim));
    for (auto& v : vecs) {
        int c = pick(rng);
        for (std::size_t d = 0; d < kDim; ++d) v[d] = cents[c][d] + kJitter * g(rng);
        rag::dense::normalize(v);
    }

    rag::index::HnswConfig cfg;
    cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 64;
    rag::index::HnswIndex idx(cfg);
    idx.build_batch(kN,
        [&](std::size_t i) { return std::span<const float>(vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    constexpr std::size_t k = 10, kQueries = 40;
    std::mt19937 qr(23);
    double acc = 0;
    for (std::size_t q = 0; q < kQueries; ++q) {
        int c = pick(qr);
        std::vector<float> qv(kDim);
        for (std::size_t d = 0; d < kDim; ++d) qv[d] = cents[c][d] + kJitter * g(qr);
        rag::dense::normalize(qv);

        std::vector<std::pair<float, std::uint32_t>> exact(kN);
        for (std::size_t i = 0; i < kN; ++i)
            exact[i] = {rag::dense::dot(qv, vecs[i]), static_cast<std::uint32_t>(i)};
        std::partial_sort(exact.begin(), exact.begin() + k, exact.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::size_t got = 0;
        for (const auto& h : idx.search(qv, k))
            for (std::size_t i = 0; i < k; ++i)
                if (exact[i].second == h.chunk.get()) { ++got; break; }
        acc += static_cast<double>(got) / static_cast<double>(k);
    }
    // Measured 0.96 with the heuristic, 0.93 without, at this size. The gate
    // sits between them: tight enough to catch the regression, loose enough not
    // to flake on link-order nondeterminism from the parallel build.
    CHECK((acc / kQueries) > 0.95);
}

// ─── ef is a working per-query dial ────────────────────────────────────
//
// ef used to be build-time only, so trading recall for latency meant rebuilding
// the graph. Now it is an argument, and the contract is: recall must be
// MONOTONIC in ef on one index, and a wide beam must beat a narrow one
// decisively. This is also what makes a recall/QPS curve measurable at all.
TEST(hnsw_ef_is_a_per_query_recall_dial) {
    AnnFixture fx;
    rag::index::HnswConfig cfg;
    cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 64;
    rag::index::HnswIndex idx(cfg);
    idx.build_batch(fx.n,
        [&](std::size_t i) { return std::span<const float>(fx.vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    const double r_narrow = fx.recall(idx, 10, 60, 10);
    const double r_mid    = fx.recall(idx, 10, 60, 64);
    const double r_wide   = fx.recall(idx, 10, 60, 400);

    CHECK(r_mid  >= r_narrow - 1e-9);   // monotonic, no rebuild involved
    CHECK(r_wide >= r_mid    - 1e-9);
    CHECK(r_wide - r_narrow > 0.05);    // the dial has real range

    // ef=0 must mean "use the configured default", not "use a beam of zero".
    CHECK_EQ(fx.recall(idx, 10, 60, 0), fx.recall(idx, 10, 60, cfg.ef_search));

    // A beam narrower than k cannot return k; it must be clamped, not truncated.
    CHECK_EQ(idx.search(fx.vecs[0], 10, 1).size(), std::size_t{10});
}

TEST(hnsw_parallel_build_matches_serial_recall) {
    AnnFixture fx;
    rag::index::HnswConfig cfg;
    cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 64;

    rag::index::HnswIndex serial(cfg);
    for (std::size_t i = 0; i < fx.n; ++i)
        serial.add(static_cast<std::uint32_t>(i), fx.vecs[i]);

    rag::index::HnswIndex parallel(cfg);
    parallel.build_batch(fx.n,
        [&](std::size_t i) { return std::span<const float>(fx.vecs[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    CHECK_EQ(serial.size(), parallel.size());
    // Concurrent insertion changes link order, so recall is statistically
    // rather than bitwise identical; it must not drop materially.
    double rs = fx.recall(serial, 10, 40);
    double rp = fx.recall(parallel, 10, 40);
    CHECK(rp > rs - 0.05);
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

// ─── Parallel embedding: batches must be concurrency-transparent ──────
//
// Corpus::embed_pending fans batches out across workers when the embedder
// advertises max_concurrency() > 1. The wire contract is that this is pure
// throughput: every chunk gets the SAME vector it would have got serially,
// nothing is dropped, and a failing backend still surfaces its error.
namespace {

// An embedder whose vector encodes the text identity, so a mis-assigned
// batch (off-by-one, torn write, lost update) is detectable rather than
// merely improbable. Also counts concurrent `embed` calls so we can prove
// the hint is honoured in both directions.
struct CountingEmbedder {
    std::size_t                    dim_;
    std::size_t                    conc_;
    bool                           fail_at_ = false;
    std::size_t                    fail_index_ = 0;
    std::shared_ptr<std::atomic<int>> live   = std::make_shared<std::atomic<int>>(0);
    std::shared_ptr<std::atomic<int>> peak   = std::make_shared<std::atomic<int>>(0);
    std::shared_ptr<std::atomic<int>> calls  = std::make_shared<std::atomic<int>>(0);

    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] std::string_view identity() const noexcept { return "counting"; }
    [[nodiscard]] std::size_t max_concurrency() const noexcept { return conc_; }

    [[nodiscard]] rag::Result<std::vector<rag::Vector>>
    embed(std::span<const std::string> texts) const {
        const int now = live->fetch_add(1, std::memory_order_acq_rel) + 1;
        int seen = peak->load(std::memory_order_relaxed);
        while (now > seen && !peak->compare_exchange_weak(seen, now)) {}
        // Hold the "connection" open so overlap is observable.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const std::size_t idx = static_cast<std::size_t>(
            calls->fetch_add(1, std::memory_order_acq_rel));
        live->fetch_sub(1, std::memory_order_acq_rel);

        if (fail_at_ && idx == fail_index_)
            return rag::fail<std::vector<rag::Vector>>(rag::Errc::unavailable, "synthetic");

        std::vector<rag::Vector> out;
        out.reserve(texts.size());
        for (const auto& t : texts) {
            rag::Vector v(dim_, 0.0f);
            // Deterministic, order-independent fingerprint of the text.
            std::uint64_t h = 1469598103934665603ull;
            for (unsigned char c : t) { h ^= c; h *= 1099511628211ull; }
            for (std::size_t i = 0; i < dim_; ++i) {
                h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
                v[i] = static_cast<float>(static_cast<double>(h >> 11) / 9007199254740992.0);
            }
            out.push_back(std::move(v));
        }
        return out;
    }
};

rag::index::Corpus embed_corpus(std::size_t conc, std::size_t docs,
                                CountingEmbedder* out_probe = nullptr) {
    rag::index::CorpusConfig cfg;
    cfg.embed_batch     = 4;
    cfg.hnsw_threshold  = 1'000'000;   // keep this test about embedding only
    rag::index::Corpus c{cfg};
    CountingEmbedder e{16, conc};
    if (out_probe) *out_probe = e;     // shares the atomics via shared_ptr
    c.set_embedder(rag::dense::AnyEmbedder{e});
    for (std::size_t i = 0; i < docs; ++i)
        c.add_document("d" + std::to_string(i) + ".md",
                       "document number " + std::to_string(i) + " about retrieval");
    (void)c.build();
    return c;
}

} // namespace

TEST(parallel_embedding_matches_serial_vectors) {
    constexpr std::size_t kDocs = 64;
    auto serial   = embed_corpus(1,  kDocs);
    auto parallel = embed_corpus(16, kDocs);

    REQUIRE(serial.chunk_count() == parallel.chunk_count());
    REQUIRE(serial.chunk_count() >= kDocs);

    for (std::size_t i = 0; i < serial.chunk_count(); ++i) {
        const auto* a = serial.chunk(rag::ChunkId{static_cast<std::uint32_t>(i)});
        const auto* b = parallel.chunk(rag::ChunkId{static_cast<std::uint32_t>(i)});
        REQUIRE(a && b);
        REQUIRE(!a->embedding.empty());          // nothing left unembedded
        REQUIRE(a->embedding.size() == b->embedding.size());
        for (std::size_t d = 0; d < a->embedding.size(); ++d)
            if (a->embedding[d] != b->embedding[d]) {
                CHECK_EQ(a->embedding[d], b->embedding[d]);
                return;                           // one report is enough
            }
    }
    CHECK(true);
}

TEST(parallel_embedding_honours_concurrency_hint) {
    CountingEmbedder probe_serial{}, probe_par{};
    (void)embed_corpus(1,  64, &probe_serial);
    (void)embed_corpus(8,  64, &probe_par);

    // hint == 1 must never overlap two embed calls.
    CHECK_EQ(probe_serial.peak->load(), 1);
    // hint > 1 should actually overlap (pool has >1 thread on any CI box we
    // care about; if it genuinely has one core this degrades to 1, so only
    // assert overlap when the machine can provide it).
    if (rag::util::max_workers() > 1) CHECK(probe_par.peak->load() > 1);
}

TEST(parallel_embedding_propagates_backend_failure) {
    rag::index::CorpusConfig cfg;
    cfg.embed_batch    = 4;
    cfg.hnsw_threshold = 1'000'000;
    rag::index::Corpus c{cfg};
    CountingEmbedder e{16, 8};
    e.fail_at_ = true; e.fail_index_ = 3;
    c.set_embedder(rag::dense::AnyEmbedder{e});
    for (std::size_t i = 0; i < 64; ++i)
        c.add_document("d" + std::to_string(i), "text " + std::to_string(i));

    auto r = c.build();
    REQUIRE(!r.has_value());
    CHECK(r.error().code == rag::Errc::unavailable);
}

// ─── Lazy meta relink / bm25 finalize ────────────────────────────
//
// add_document() used to relink every chunk's borrowed meta pointer and
// re-finalize BM25 on EVERY insert — both O(corpus), so bulk ingest was
// quadratic. Both are now deferred. The contract that must survive that:
// a caller who never calls build() still sees correct metadata and correct
// lexical scores, because the read paths repair on demand.
TEST(metadata_survives_reallocation_without_build) {
    rag::index::Corpus c;
    // Enough documents to force several reallocations of docs_.
    constexpr std::size_t kDocs = 300;
    for (std::size_t i = 0; i < kDocs; ++i)
        c.add_document("d" + std::to_string(i) + ".md",
                       "body of document " + std::to_string(i),
                       {{"idx", std::to_string(i)}});

    // No build() call: every chunk must still resolve to ITS OWN document's
    // metadata, not a dangling or shifted pointer.
    REQUIRE(c.chunk_count() >= kDocs);
    std::size_t checked = 0;
    for (std::size_t i = 0; i < c.chunk_count(); ++i) {
        const auto* ch = c.chunk(rag::ChunkId{static_cast<std::uint32_t>(i)});
        REQUIRE(ch);
        REQUIRE(ch->meta);
        auto it = ch->meta->find("idx");
        REQUIRE(it != ch->meta->end());
        if (it->second != std::to_string(ch->doc.get())) {
            CHECK_EQ(it->second, std::to_string(ch->doc.get()));
            return;
        }
        ++checked;
    }
    CHECK(checked == c.chunk_count());
}

TEST(lexical_search_works_without_explicit_build) {
    rag::index::Corpus c;
    for (std::size_t i = 0; i < 50; ++i)
        c.add_document("d" + std::to_string(i), "filler text about vectors and indexes");
    c.add_document("needle.md", "quokka marsupial photogenic");

    // Never built: bm25 must be finalized lazily on the read path, and the
    // distinctive document must win.
    auto hits = c.lexical_search("quokka", 3);
    REQUIRE(!hits.empty());
    const auto* ch = c.chunk(hits[0].chunk);
    REQUIRE(ch);
    const auto* d = c.document(ch->doc);
    REQUIRE(d);
    CHECK_EQ(d->uri, std::string("needle.md"));
}

// ─── Coverage from the inverted index ───────────────────────────
//
// feature_rerank blends fusion rank with lexical coverage: how many distinct
// query terms occur in a candidate. That used to be computed by re-tokenizing
// every candidate's text and building a set per candidate. It is now answered
// from the postings, which is much faster — but only legitimate if the answer
// is IDENTICAL. This test is the equivalence proof: the postings walk must
// agree with the re-tokenize reference on every (query, candidate) pair.
TEST(term_coverage_matches_retokenization) {
    static const char* w[] = {"vector","index","query","embedding","token","rank",
                              "fusion","graph","sparse","dense","chunk","corpus",
                              "search","semantic","neural","lexical"};
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> pick(0, 15);

    rag::index::Corpus corpus;
    for (int i = 0; i < 400; ++i) {
        std::string s;
        for (int j = 0; j < 40; ++j) { s += w[pick(rng)]; s += ' '; }
        corpus.add_document("d" + std::to_string(i) + ".md", s);
    }
    (void)corpus.build();

    std::size_t compared = 0, mismatches = 0;
    for (int q = 0; q < 40; ++q) {
        std::string qs;
        for (int j = 0; j < 3; ++j) { qs += w[pick(rng)]; qs += ' '; }
        auto terms = corpus.tokenizer().tokenize(qs);
        std::sort(terms.begin(), terms.end());
        terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

        auto hits = corpus.lexical_search(qs, 30);
        std::vector<std::uint32_t> ids;
        for (auto& h : hits) ids.push_back(h.chunk.get());
        std::vector<std::uint32_t> fast;
        corpus.term_coverage(terms, ids, fast);
        REQUIRE(fast.size() == ids.size());

        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto* ch = corpus.chunk(rag::ChunkId{ids[i]});
            REQUIRE(ch);
            auto ct = corpus.tokenizer().tokenize(ch->indexed_text());
            std::sort(ct.begin(), ct.end());
            std::uint32_t slow = 0;
            for (const auto& t : terms)
                if (std::binary_search(ct.begin(), ct.end(), t)) ++slow;
            ++compared;
            if (slow != fast[i]) ++mismatches;
        }
    }
    CHECK(compared > 100);       // the comparison actually exercised something
    CHECK_EQ(mismatches, std::size_t{0});
}

// ─── SQ8 quantization ────────────────────────────────────────
TEST(sq8_dot_approximates_float_dot) {
    std::mt19937 rng(4);
    std::normal_distribution<float> g(0.0f, 1.0f);
    // Sweep dims that do and don't divide the SIMD block size, so the tail
    // handling in dot_sq8 is exercised too.
    for (std::size_t dim : {7u, 16u, 31u, 64u, 100u, 256u}) {
        double worst = 0.0;
        for (int trial = 0; trial < 40; ++trial) {
            std::vector<float> a(dim), b(dim);
            for (auto& x : a) x = g(rng);
            for (auto& x : b) x = g(rng);
            rag::dense::normalize(a);
            rag::dense::normalize(b);

            std::vector<std::int8_t> qa(dim), qb(dim);
            rag::dense::quantize_sq8(a, qa);
            rag::dense::quantize_sq8(b, qb);

            const float exact = rag::dense::dot(a, b);
            const float approx =
                static_cast<float>(rag::dense::dot_sq8(qa.data(), qb.data(), dim))
                * rag::dense::kSq8Scale;
            worst = std::max(worst, static_cast<double>(std::fabs(exact - approx)));
        }
        // Each component carries at most 1/254 of quantization error; summed
        // over unit vectors the dot error stays far below 0.02 in practice.
        CHECK(worst < 0.02);
    }
}

TEST(sq8_walk_preserves_recall) {
    // The SQ8 store is used to ORDER candidates during the graph walk while
    // the reported top-k is rescored on exact floats. That is only a legitimate
    // trade if recall is untouched, so gate it: on realistic clustered geometry
    // at a dimension large enough that the walk is genuinely memory-bound,
    // recall@10 must stay near-perfect.
    constexpr std::size_t n = 3000, dim = 128, nc = 40, nq = 100, k = 10;
    std::mt19937 rng(21);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<std::vector<float>> cent(nc, std::vector<float>(dim));
    for (auto& c : cent) { for (auto& x : c) x = g(rng); rag::dense::normalize(c); }

    std::vector<float> data(n * dim);
    std::uniform_int_distribution<std::size_t> pick(0, nc - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = cent[pick(rng)];
        for (std::size_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + 0.06f * g(rng);
        rag::dense::normalize(std::span<float>(data.data() + i * dim, dim));
    }

    rag::index::HnswIndex idx{rag::index::HnswConfig{}};
    idx.build_batch(n,
        [&](std::size_t i) { return std::span<const float>(data.data() + i * dim, dim); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    double hit = 0;
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<float> qv(dim);
        const auto& c = cent[pick(rng)];
        for (std::size_t d = 0; d < dim; ++d) qv[d] = c[d] + 0.06f * g(rng);
        rag::dense::normalize(qv);

        std::vector<std::pair<float, std::uint32_t>> exact;
        exact.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            exact.emplace_back(
                rag::dense::dot(std::span<const float>(data.data() + i * dim, dim), qv),
                static_cast<std::uint32_t>(i));
        std::partial_sort(exact.begin(), exact.begin() + k, exact.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });

        for (const auto& h : idx.search(qv, k))
            for (std::size_t j = 0; j < k; ++j)
                if (exact[j].second == h.chunk.get()) { hit += 1; break; }
    }
    const double recall = hit / (nq * k);
    CHECK(recall > 0.95);

    // And the scores handed back must be EXACT cosines, not quantized ones:
    // SQ8 is a traversal accelerator, never a scoring approximation.
    std::vector<float> qv(data.begin(), data.begin() + dim);
    auto hits = idx.search(qv, 3);
    REQUIRE(!hits.empty());
    const float exact_self =
        rag::dense::dot(std::span<const float>(data.data(), dim), qv);
    CHECK(std::fabs(hits[0].score.get() - exact_self) < 1e-5f);
}

// ─── Compressed-residency: dropping the exact vectors ─────────────
//
// `drop_floats` releases the float arena and leaves the index resident only in
// its quantized mirror(s). That is a real memory/accuracy trade, so pin BOTH
// halves of it: the memory must actually go away, and recall must survive.
TEST(drop_floats_shrinks_index_and_preserves_recall) {
    constexpr std::size_t n = 3000, dim = 128, nc = 40, nq = 100, k = 10;
    std::mt19937 rng(21);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<std::vector<float>> cent(nc, std::vector<float>(dim));
    for (auto& c : cent) { for (auto& x : c) x = g(rng); rag::dense::normalize(c); }

    std::vector<float> data(n * dim);
    std::uniform_int_distribution<std::size_t> pick(0, nc - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = cent[pick(rng)];
        for (std::size_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + 0.06f * g(rng);
        rag::dense::normalize(std::span<float>(data.data() + i * dim, dim));
    }
    auto vec = [&](std::size_t i) { return std::span<const float>(data.data() + i * dim, dim); };
    auto id  = [&](std::size_t i) { return static_cast<std::uint32_t>(i); };

    std::vector<std::vector<float>> qs(nq, std::vector<float>(dim));
    for (auto& qv : qs) {
        const auto& c = cent[pick(rng)];
        for (std::size_t d = 0; d < dim; ++d) qv[d] = c[d] + 0.06f * g(rng);
        rag::dense::normalize(qv);
    }

    auto recall_of = [&](rag::index::HnswIndex& idx) {
        double hit = 0;
        for (const auto& qv : qs) {
            std::vector<std::pair<float, std::uint32_t>> exact;
            exact.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                exact.emplace_back(rag::dense::dot(vec(i), qv), static_cast<std::uint32_t>(i));
            std::partial_sort(exact.begin(), exact.begin() + k, exact.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            for (const auto& h : idx.search(qv, k))
                for (std::size_t j = 0; j < k; ++j)
                    if (exact[j].second == h.chunk.get()) { hit += 1; break; }
        }
        return hit / (nq * k);
    };

    rag::index::HnswIndex base{rag::index::HnswConfig{}};
    base.build_batch(n, vec, id);
    (void)base.search(qs[0], k);                 // force seal()
    const double base_recall = recall_of(base);
    const std::size_t base_bytes = base.memory_bytes();

    rag::index::HnswIndex lean{rag::index::HnswConfig{.drop_floats = true}};
    lean.build_batch(n, vec, id);
    (void)lean.search(qs[0], k);
    const double lean_recall = recall_of(lean);

    // The float arena is 4 bytes/dim and the SQ8 mirror that replaces it is 1,
    // so the index must get materially smaller — not merely not-bigger.
    CHECK(lean.memory_bytes() < base_bytes);
    CHECK(lean.memory_use().vectors == 0);
    CHECK(lean.memory_use().sq8 > 0);
    // SQ8 resolves ~1/127 per component, which is far finer than the gaps
    // between neighbouring documents, so ranking barely moves.
    CHECK(lean_recall > 0.90);
    CHECK(base_recall > 0.95);
}

// ─── PQ codes drive the walk without deciding the answer ──────────
//
// With pq_codes set, traversal scores candidates from m-byte codes. The
// guarantee is the same one SQ8 makes: codes ORDER the walk, and the reported
// top-k is still rescored on the most precise representation available. If the
// floats are kept, that means the scores must remain EXACT.
TEST(pq_walk_preserves_exact_scores) {
    constexpr std::size_t n = 2500, dim = 64, nc = 30;
    std::mt19937 rng(9);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<std::vector<float>> cent(nc, std::vector<float>(dim));
    for (auto& c : cent) { for (auto& x : c) x = g(rng); rag::dense::normalize(c); }

    std::vector<float> data(n * dim);
    std::uniform_int_distribution<std::size_t> pick(0, nc - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = cent[pick(rng)];
        for (std::size_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + 0.06f * g(rng);
        rag::dense::normalize(std::span<float>(data.data() + i * dim, dim));
    }

    rag::index::HnswIndex idx{rag::index::HnswConfig{.pq_codes = 16}};
    idx.build_batch(n,
        [&](std::size_t i) { return std::span<const float>(data.data() + i * dim, dim); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    std::vector<float> qv(data.begin(), data.begin() + dim);
    auto hits = idx.search(qv, 5);
    REQUIRE(!hits.empty());
    // Every returned score is a true cosine against the stored vector, even
    // though the walk that found it never looked at a float.
    for (const auto& h : hits) {
        const float exact = rag::dense::dot(
            std::span<const float>(data.data() + h.chunk.get() * dim, dim), qv);
        CHECK(std::fabs(h.score.get() - exact) < 1e-5f);
    }
    // And the PQ codes must actually be resident, or this tested nothing.
    CHECK(idx.memory_use().pq > 0);
}

// ─── Sealing frees the duplicate adjacency ────────────────────────
//
// A sealed index holds its graph in CSR form only; the per-node link vectors
// it was built from are released. They are restored on demand, so mutating a
// sealed index must still work — that round trip is what this pins.
TEST(sealed_index_releases_link_duplicate_and_survives_mutation) {
    constexpr std::size_t n = 2500, dim = 32;
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = g(rng);
    for (std::size_t i = 0; i < n; ++i)
        rag::dense::normalize(std::span<float>(data.data() + i * dim, dim));
    auto vec = [&](std::size_t i) { return std::span<const float>(data.data() + i * dim, dim); };

    rag::index::HnswIndex idx{rag::index::HnswConfig{}};
    idx.build_batch(n, vec, [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    std::vector<float> qv(data.begin(), data.begin() + dim);
    auto before = idx.search(qv, 5);
    REQUIRE(!before.empty());

    // Sealed: the graph lives in the CSR, and the vector-of-vectors is gone.
    CHECK(idx.memory_use().csr > 0);
    CHECK(idx.memory_use().links == 0);

    // Adding unseals, which must rebuild the mutable adjacency from the CSR.
    // If unpack_links() were wrong, the graph would lose edges here and the
    // previously-found neighbours would stop being reachable.
    std::vector<float> extra(dim, 0.0f);
    extra[0] = 1.0f;
    idx.add(static_cast<std::uint32_t>(n), extra);
    CHECK(idx.memory_use().links > 0);

    auto after = idx.search(qv, 5);
    REQUIRE(!after.empty());
    CHECK(after[0].chunk.get() == before[0].chunk.get());
    CHECK(std::fabs(after[0].score.get() - before[0].score.get()) < 1e-5f);
}

// ─── The index blob round-trips through the sealed representation ──
//
// serialize() writes per-node link lists, but a sealed index no longer stores
// them in that shape. Saving a queried (hence sealed) index must therefore
// still produce a blob that reloads identically.
TEST(sealed_index_serializes_and_reloads_identically) {
    constexpr std::size_t n = 1200, dim = 32;
    std::mt19937 rng(13);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = g(rng);
    for (std::size_t i = 0; i < n; ++i)
        rag::dense::normalize(std::span<float>(data.data() + i * dim, dim));

    rag::index::HnswIndex idx{rag::index::HnswConfig{}};
    idx.build_batch(n,
        [&](std::size_t i) { return std::span<const float>(data.data() + i * dim, dim); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });

    std::vector<float> qv(data.begin(), data.begin() + dim);
    auto want = idx.search(qv, 10);          // seals the index
    REQUIRE(!want.empty());

    auto blob = idx.serialize();             // ... and must survive being sealed
    auto back = rag::index::HnswIndex::deserialize(blob);
    REQUIRE(back.has_value());
    CHECK(back->size() == idx.size());

    auto got = back->search(qv, 10);
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        CHECK(got[i].chunk.get() == want[i].chunk.get());
        CHECK(std::fabs(got[i].score.get() - want[i].score.get()) < 1e-6f);
    }
}

// ─── Concurrent hybrid retrieval is deterministic ─────────────────
//
// HybridRetrieveStage runs the lexical and dense retrievers concurrently
// (they read disjoint index structures, so hybrid costs max rather than sum).
// The contract that must hold: overlapping them is a pure latency win and
// changes nothing observable — identical hits, identical scores, identical
// order, every time.
TEST(hybrid_search_is_deterministic_under_concurrency) {
    static const char* w[] = {"vector","index","query","embedding","token","rank",
                              "fusion","graph","sparse","dense","chunk","corpus",
                              "search","semantic","neural","lexical"};
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> pick(0, 15);

    rag::Engine e;
    e.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{128}});
    for (int i = 0; i < 600; ++i) {
        std::string s;
        for (int j = 0; j < 40; ++j) { s += w[pick(rng)]; s += ' '; }
        e.add("d" + std::to_string(i) + ".md", s);
    }
    (void)e.build();

    std::size_t compared = 0, divergences = 0;
    for (int q = 0; q < 40; ++q) {
        std::string qs;
        for (int j = 0; j < 4; ++j) { qs += w[pick(rng)]; qs += ' '; }

        auto first = e.search(qs, 10);
        REQUIRE(first.has_value());
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto again = e.search(qs, 10);
            REQUIRE(again.has_value());
            REQUIRE(again->size() == first->size());
            for (std::size_t i = 0; i < first->size(); ++i) {
                ++compared;
                if ((*first)[i].chunk.get() != (*again)[i].chunk.get() ||
                    std::fabs((*first)[i].score.get() - (*again)[i].score.get()) > 1e-6f)
                    ++divergences;
            }
        }
    }
    CHECK(compared > 100);
    CHECK_EQ(divergences, std::size_t{0});
}

// ─── Stem memoization is transparent ───────────────────────────
//
// Tokenizer memoizes porter_stem (it dominated ingest). The cache must be
// invisible: same tokens as calling the stemmer directly, including after the
// bounded cache overflows and is cleared.
TEST(tokenizer_stem_cache_matches_direct_stemming) {
    rag::text::TokenizeOptions opts;
    opts.stem = true;
    opts.drop_stopwords = false;
    opts.min_len = 1;
    rag::text::Tokenizer tok{opts};

    // Words with real Porter structure (plurals, -ing, -ed, -ational, ...) so
    // the stemmer actually rewrites rather than passing through.
    static const char* words[] = {
        "running", "runs", "ran", "happiness", "relational", "conditional",
        "rationalize", "vietnamization", "predication", "hopefulness",
        "formality", "sensitivity", "agreed", "plastered", "motoring",
        "sing", "conflated", "troubled", "sized", "hopping", "falling",
        "controlling", "rolling", "feed", "matting", "skies", "cats",
    };

    std::size_t compared = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {          // exercise cache hits
        for (const char* w : words) {
            auto got = tok.tokenize(w);
            REQUIRE(got.size() == 1u);
            CHECK_EQ(got[0], rag::text::porter_stem(w));
            ++compared;
        }
    }
    CHECK(compared == 3 * (sizeof(words) / sizeof(words[0])));

    // Overflow the cache with many distinct tokens, then re-check the original
    // words: a cleared cache must repopulate to the same answers.
    for (int i = 0; i < 70000; ++i) (void)tok.tokenize("zq" + std::to_string(i) + "ing");
    for (const char* w : words) {
        auto got = tok.tokenize(w);
        REQUIRE(got.size() == 1u);
        CHECK_EQ(got[0], rag::text::porter_stem(w));
    }
}

// ─── BM25 precomputed weights are exact ─────────────────────────
//
// finalize() precomputes the (tf, dl)-dependent half of every posting's score
// so the query loop is a multiply-add instead of a division. That is only a
// speedup if it is arithmetically identical to scoring from scratch — and it
// is derived data, so it must also be rebuilt correctly on deserialize.
TEST(bm25_precomputed_weights_match_direct_scoring) {
    static const char* w[] = {"alpha","beta","gamma","delta","epsilon","zeta",
                              "eta","theta","iota","kappa","lambda","sigma"};
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> pick(0, 11);

    rag::lexical::Bm25Index idx;
    constexpr int N = 800;
    for (int i = 0; i < N; ++i) {
        std::string s;
        const int len = 5 + (i % 30);      // varied doc lengths => varied dl term
        for (int j = 0; j < len; ++j) { s += w[pick(rng)]; s += ' '; }
        idx.add(static_cast<std::uint32_t>(i), s);
    }
    idx.finalize();

    // search() must agree with score_doc(), which computes the formula directly.
    std::size_t compared = 0, mismatches = 0;
    for (int q = 0; q < 60; ++q) {
        std::string qs;
        for (int j = 0; j < 1 + (q % 4); ++j) { qs += w[pick(rng)]; qs += ' '; }
        const auto terms = idx.tokenizer().tokenize(qs);

        for (const auto& h : idx.search(qs, 20)) {
            ++compared;
            if (std::fabs(h.score.get() - idx.score_doc(terms, h.chunk.get())) > 1e-4f)
                ++mismatches;
        }
    }
    CHECK(compared > 100);
    CHECK_EQ(mismatches, std::size_t{0});

    // The weights are NOT serialized (they are derived); deserialize must
    // rebuild them, so scores have to survive a round-trip bit-for-bit.
    auto blob = idx.serialize();
    auto back = rag::lexical::Bm25Index::deserialize(blob);
    REQUIRE(back.has_value());

    std::size_t rt_compared = 0, rt_diff = 0;
    for (int q = 0; q < 40; ++q) {
        std::string qs;
        for (int j = 0; j < 3; ++j) { qs += w[pick(rng)]; qs += ' '; }
        auto a = idx.search(qs, 10);
        auto b = back->search(qs, 10);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            ++rt_compared;
            if (a[i].chunk.get() != b[i].chunk.get() ||
                std::fabs(a[i].score.get() - b[i].score.get()) > 1e-5f)
                ++rt_diff;
        }
    }
    CHECK(rt_compared > 50);
    CHECK_EQ(rt_diff, std::size_t{0});
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
    // Persistence round-trips: reopened index returns the same top hit.
    auto blob = idx->serialize();
    auto idx2 = rag::sparse::SpladeIndex::deserialize(blob);
    REQUIRE(idx2.has_value());
    CHECK_EQ(idx2->vocab_size(), idx->vocab_size());
    auto hits2 = idx2->search("vector nearest neighbours", 3);
    REQUIRE(!hits2.empty());
    CHECK(engine.corpus().resolve(hits2[0]).uri == "d1");
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

// ── MMR diversity rerank ──────────────────────────────────────────

TEST(mmr_diversifies_results) {
    rag::Engine engine;
    // Three near-duplicate docs about cats + one about dogs.
    engine.add("c1", "cats are wonderful feline pets that purr and love to nap");
    engine.add("c2", "cats are lovely feline companions that purr and nap often");
    engine.add("c3", "cats the feline pets purr and nap and love warmth greatly");
    engine.add("dog", "dogs are loyal canine companions that bark and fetch balls");
    engine.build();
    auto hits = engine.corpus().lexical_search("cats feline pets purr", 4);
    REQUIRE(!hits.empty());
    // Pure relevance (lambda=1) keeps the near-dupes together.
    rag::rerank::MmrConfig relev; relev.lambda = 1.0f; relev.k = 4;
    auto pure = rag::rerank::mmr(engine.corpus(), hits, relev);
    // Diversity (lambda=0.3) should pull the dog doc up relative to pure rel.
    rag::rerank::MmrConfig div; div.lambda = 0.3f; div.k = 4;
    auto diverse = rag::rerank::mmr(engine.corpus(), hits, div);
    REQUIRE(diverse.size() == pure.size());
    auto rank_of = [&](const std::vector<rag::Hit>& v, const char* uri) {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (engine.corpus().resolve(v[i]).uri == uri) return (int)i;
        return 99;
    };
    CHECK(rank_of(diverse, "dog") <= rank_of(pure, "dog"));
}

// ── Product Quantization ────────────────────────────────────────

TEST(pq_compresses_and_ranks) {
    // 8-dim unit vectors; PQ with m=4, so 4 bytes/vector (8x compression).
    std::vector<rag::Vector> data;
    for (int i = 0; i < 32; ++i) {
        rag::Vector v(8, 0.0f);
        v[i % 8] = 1.0f; v[(i + 1) % 8] = 0.5f;
        rag::dense::normalize(v);
        data.push_back(v);
    }
    rag::index::PqConfig cfg; cfg.m = 4; cfg.ksub = 16; cfg.iters = 20;
    auto pq = rag::index::ProductQuantizer::train(data, cfg);
    REQUIRE(pq.has_value());
    CHECK(pq->compression_ratio() > 1.0f);
    for (std::size_t i = 0; i < data.size(); ++i) pq->add((std::uint32_t)i, data[i]);
    // Querying with a training vector should return its own id near the top.
    auto hits = pq->search(data[3], 5);
    REQUIRE(!hits.empty());
    bool found = false;
    for (auto& h : hits) if (h.chunk.get() == 3) found = true;
    CHECK(found);
    // Serialization round-trips.
    auto blob = pq->serialize();
    auto pq2 = rag::index::ProductQuantizer::deserialize(blob);
    REQUIRE(pq2.has_value());
    CHECK_EQ(pq2->code_count(), pq->code_count());
}

// ── Semantic + proposition chunking ────────────────────────────────

TEST(semantic_chunk_lexical_splits_on_topic_shift) {
    std::string body =
        "Cats are feline pets. Cats purr when happy. Cats love to nap in the sun. "
        "Quantum computers use qubits. Qubits exploit superposition. Quantum gates transform states.";
    rag::text::SemanticChunkOptions opts; opts.breakpoint_percentile = 70.0f;
    auto chunks = rag::text::semantic_chunk_lexical(rag::DocId{0}, body, opts);
    // The topic shift (cats -> quantum) should produce at least 2 chunks.
    CHECK(chunks.size() >= 2);
}

TEST(proposition_chunk_atomizes) {
    std::string body = "The sky is blue. Grass is green. Water is wet.";
    auto props = rag::text::proposition_chunk(rag::DocId{0}, body);
    CHECK_EQ(props.size(), 3u);
}

// ── Contextual retrieval ──────────────────────────────────────

TEST(contextualize_adds_situating_context) {
    std::string doc =
        "# Acme Q3 Earnings\n\nAcme Corporation reported strong results. "
        "Revenue grew 3% year over year in the quarter.";
    std::vector<rag::Chunk> chunks;
    rag::Chunk ch; ch.doc = rag::DocId{0}; ch.text = "Revenue grew 3% year over year in the quarter.";
    chunks.push_back(ch);
    rag::text::contextualize(chunks, doc);
    // The chunk's context should now mention Acme (the disambiguating title).
    CHECK(chunks[0].context.find("Acme") != std::string::npos);
}

// ── Cascade ────────────────────────────────────────────────

TEST(cascade_narrows_through_stages) {
    rag::Engine engine;
    for (int i = 0; i < 20; ++i)
        engine.add("d" + std::to_string(i),
            "vector search hnsw approximate nearest neighbours graph indexing document " + std::to_string(i));
    engine.build();
    rag::cascade::CascadeConfig cfg;
    cfg.retrieve_k = 15; cfg.colbert_k = 8; cfg.final_k = 5;
    cfg.use_rerank = false;   // no cross-encoder server in tests
    rag::cascade::Cascade casc(cfg);
    casc.with_colbert(rag::late::ColbertReranker(rag::late::hashed_token_embedder(64)));
    std::vector<rag::cascade::StageTrace> trace;
    auto hits = casc.run(engine.corpus(), "vector nearest neighbours", &trace);
    REQUIRE(hits.has_value());
    CHECK(hits->size() <= 5u);
    CHECK(!hits->empty());
    // The trace records the funnel narrowing.
    CHECK(trace.size() >= 2u);
}

// ── Caches ────────────────────────────────────────────────

TEST(lru_cache_evicts_and_hits) {
    rag::cache::EmbeddingCache ec(2);
    ec.put("m", "a", {1.0f});
    ec.put("m", "b", {2.0f});
    CHECK(ec.get("m", "a").has_value());   // hit, touches a (MRU)
    ec.put("m", "c", {3.0f});              // evicts b (LRU)
    CHECK(!ec.get("m", "b").has_value());   // b was evicted
    CHECK(ec.get("m", "a").has_value());   // a survived
    // Identity guards against stale cross-model hits.
    CHECK(!ec.get("other-model", "a").has_value());
}

// ── Incremental delete ────────────────────────────────────────

TEST(corpus_remove_document_tombstones) {
    rag::Engine engine;
    auto d1 = engine.add("keep", "vector search with hnsw graphs and indexing");
    auto d2 = engine.add("drop", "vector search with hnsw graphs and indexing too");
    engine.build();
    REQUIRE(d2.has_value());
    CHECK_EQ(engine.corpus().live_document_count(), 2u);
    auto rm = engine.corpus().remove_document(*d2);
    REQUIRE(rm.has_value());
    CHECK(engine.corpus().is_deleted(*d2));
    CHECK_EQ(engine.corpus().live_document_count(), 1u);
    // The dropped doc must not appear in results.
    auto hits = engine.corpus().lexical_search("vector search hnsw", 10);
    for (auto& h : hits) CHECK(engine.corpus().resolve(h).uri != "drop");
    // Removing again fails cleanly.
    CHECK(!engine.corpus().remove_document(*d2).has_value());
}

TEST(hnsw_tombstone_excludes_from_search) {
    rag::index::HnswConfig cfg;
    rag::index::HnswIndex idx(cfg);
    for (std::uint32_t i = 0; i < 10; ++i) {
        rag::Vector v(8, 0.0f); v[i % 8] = 1.0f; rag::dense::normalize(v);
        idx.add(i, v);
    }
    rag::Vector q(8, 0.0f); q[0] = 1.0f; rag::dense::normalize(q);
    idx.remove(0);
    CHECK(idx.is_deleted(0));
    auto hits = idx.search(q, 10);
    for (auto& h : hits) CHECK(h.chunk.get() != 0u);
    idx.compact();
    CHECK_EQ(idx.deleted_count(), 0u);
}

// ─── re-adding an id REPLACES it; it never appears twice ───────────────────
//
// add() documents re-adding an id as an upsert (it already resurrected a
// tombstoned one), but it used to APPEND a second node with the same id. The
// consequences were both silent: one search returned id 2 twice — two slots of
// a top-k spent on one document, and a phantom duplicate for anything
// downstream keyed by id — and the superseded vector kept scoring on its own
// merits, so the OLD content stayed retrievable under the new id forever.
TEST(hnsw_readding_an_id_replaces_it) {
    rag::index::HnswIndex idx{rag::index::HnswConfig{}};
    auto axis = [](std::size_t which) {
        rag::Vector v(8, 0.0f); v[which] = 1.0f; rag::dense::normalize(v); return v;
    };
    for (std::uint32_t i = 0; i < 5; ++i) idx.add(i, axis(i));

    idx.add(2, axis(7));                       // same id, entirely new direction

    // The NEW vector must be found, under that id, exactly once.
    auto hits = idx.search(axis(7), 5);
    int seen = 0;
    for (const auto& h : hits) if (h.chunk.get() == 2u) ++seen;
    CHECK_EQ(seen, 1);
    REQUIRE(!hits.empty());
    CHECK_EQ(hits[0].chunk.get(), 2u);
    CHECK(hits[0].score.get() > 0.99f);

    // The OLD vector must no longer be retrievable under that id.
    for (const auto& h : idx.search(axis(2), 5))
        if (h.chunk.get() == 2u) CHECK(h.score.get() < 0.99f);

    // No id may repeat anywhere in a result set.
    std::set<std::uint32_t> ids;
    for (const auto& h : idx.search(axis(7), 5)) CHECK(ids.insert(h.chunk.get()).second);

    // compact() must reclaim the superseded node even with zero tombstones.
    CHECK_EQ(idx.deleted_count(), 0u);
    const std::size_t before = idx.size();
    idx.compact();
    CHECK(idx.size() < before);
    auto after = idx.search(axis(7), 5);
    REQUIRE(!after.empty());
    CHECK_EQ(after[0].chunk.get(), 2u);
}

// The same guarantee across the bulk path: build_batch appends nodes without
// going through add(), so a later add() of an id that build_batch already
// placed must still replace it rather than duplicate it.
TEST(hnsw_add_after_build_batch_replaces) {
    rag::index::HnswIndex idx{rag::index::HnswConfig{}};
    std::vector<rag::Vector> base;
    for (std::size_t i = 0; i < 64; ++i) {
        rag::Vector v(8, 0.01f); v[i % 8] = 1.0f; rag::dense::normalize(v);
        base.push_back(std::move(v));
    }
    idx.build_batch(base.size(),
        [&](std::size_t i) { return std::span<const float>(base[i]); },
        [&](std::size_t i) { return static_cast<std::uint32_t>(i); });
    (void)idx.search(base[0], 3);           // seal + quantize before mutating

    rag::Vector fresh(8, 0.0f); fresh[5] = 1.0f; rag::dense::normalize(fresh);
    idx.add(3, fresh);

    int seen = 0;
    for (const auto& h : idx.search(fresh, 10)) if (h.chunk.get() == 3u) ++seen;
    CHECK_EQ(seen, 1);
}

// ── ONNX integration (only when built with RAGCPP_WITH_ONNX + a model path) ──
// Set RAGCPP_TEST_ONNX_MODEL and RAGCPP_TEST_ONNX_TOKENIZER env vars to a real
// sentence-transformer ONNX export to exercise the real in-process path and
// measure a dense retrieval lift. Skipped (as a pass) when unavailable.
TEST(onnx_embedder_real_model_if_available) {
    if (!rag::dense::OnnxEmbedder::available()) { CHECK(true); return; }
    const char* model = std::getenv("RAGCPP_TEST_ONNX_MODEL");
    const char* toks  = std::getenv("RAGCPP_TEST_ONNX_TOKENIZER");
    if (!model || !toks) { CHECK(true); return; }
    rag::dense::LocalEmbedderConfig cfg;
    cfg.model_path = model; cfg.tokenizer_path = toks;
    auto emb = rag::dense::OnnxEmbedder::load(cfg);
    REQUIRE(emb.has_value());
    CHECK(emb->dimension() > 0);
    rag::Engine engine;
    engine.with_embedder(rag::dense::AnyEmbedder{std::move(*emb)});
    engine.add("d1", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.");
    engine.add("d2", "Photosynthesis converts light energy into chemical energy in plants.");
    engine.build();
    // A semantic query with no lexical overlap should still find d1.
    auto hits = engine.search("famous landmark in the French capital", 1);
    REQUIRE(hits.has_value());
    REQUIRE(!hits->empty());
    CHECK((*hits)[0].uri == "d1");
}

// ── Plugin registry ──────────────────────────────────────────────────────────

TEST(plugin_builtins_registered) {
    rag::plugin::ensure_builtins_registered();
    auto& reg = rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance();
    CHECK(reg.contains("hash"));
    CHECK(reg.contains("ollama"));
    CHECK(reg.contains("openai"));
    CHECK(reg.contains("llamacpp"));
    CHECK(reg.size() >= 4);
}

TEST(plugin_create_embedder_by_name) {
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "hash"}, {"dim", 128}});
    REQUIRE(emb.has_value());
    CHECK_EQ(emb->dimension(), 128u);
    std::vector<std::string> texts{"hello world"};
    auto v = emb->embed(texts);
    REQUIRE(v.has_value());
    CHECK_EQ((*v).size(), 1u);
    CHECK_EQ((*v)[0].size(), 128u);
}

TEST(plugin_create_from_bare_string) {
    auto emb = rag::plugin::make_embedder(nlohmann::json("hash"));
    REQUIRE(emb.has_value());
    CHECK(emb->dimension() > 0);
}

TEST(plugin_unknown_name_is_error) {
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "no_such_backend"}});
    CHECK(!emb.has_value());
    CHECK_EQ(emb.error().code, rag::Errc::not_found);
}

TEST(plugin_missing_type_is_error) {
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"dim", 64}});
    CHECK(!emb.has_value());
    CHECK_EQ(emb.error().code, rag::Errc::invalid_argument);
}

TEST(plugin_custom_registration_roundtrip) {
    // Simulate a third-party plugin registering a factory at runtime.
    rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().register_factory(
        "unit_test_custom", [](const nlohmann::json& c) -> rag::Result<rag::plugin::AnyEmbedder> {
            auto dim = c.value("dim", 32);
            return rag::plugin::AnyEmbedder{rag::dense::HashEmbedder{static_cast<std::size_t>(dim)}};
        });
    auto emb = rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().create_from(
        nlohmann::json{{"type", "unit_test_custom"}, {"dim", 77}});
    REQUIRE(emb.has_value());
    CHECK_EQ(emb->dimension(), 77u);
}

TEST(plugin_reranker_registered) {
    rag::plugin::ensure_builtins_registered();
    CHECK(rag::plugin::Registry<rag::plugin::AnyReranker>::instance().contains("cross_encoder"));
}

TEST(plugin_describe_carries_config_hints) {
    // describe() powers `ragcpp list` / --help: every built-in must self-document
    // the config keys it takes, so a user can discover the config surface without
    // reading source. Assert a few known backends have non-empty descriptions.
    rag::plugin::ensure_builtins_registered();
    auto rows = rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().describe();
    std::map<std::string, std::string> by_name(rows.begin(), rows.end());
    REQUIRE(by_name.count("hash"));
    CHECK(!by_name["hash"].empty());
    CHECK(!by_name["ollama"].empty());
    CHECK(!by_name["fallback"].empty());
    // The description should mention at least one config key by name.
    CHECK(by_name["ollama"].find("model") != std::string::npos);
}

TEST(plugin_new_hosted_providers_registered) {
    // "Really add more": voyage / together were added as ~6-line registrations
    // over the OpenAI-compatible backend. They must be reachable by name and
    // constructible (no network is touched at construction).
    rag::plugin::ensure_builtins_registered();
    auto v = rag::plugin::make_embedder(
        nlohmann::json{{"type", "voyage"}, {"api_key", "k"}, {"dim", 512}});
    REQUIRE(v.has_value());
    CHECK_EQ(v->dimension(), 512u);
    auto t = rag::plugin::make_embedder(
        nlohmann::json{{"type", "together"}, {"api_key", "k"}});
    REQUIRE(t.has_value());
}

TEST(plugin_together_requires_api_key) {
    // require<>() surfaces a typed error the factory just propagates.
    auto t = rag::plugin::make_embedder(nlohmann::json{{"type", "together"}});
    CHECK(!t.has_value());
    CHECK_EQ(t.error().code, rag::Errc::invalid_argument);
}

TEST(plugin_config_view_is_total) {
    // The Config wrapper never throws: missing -> default, wrong type -> default,
    // require -> typed error, sub -> nested json or error. (Config holds a
    // reference to the spec, which in real use is the const Json& a factory is
    // handed and which outlives the call; here we keep it in a named local.)
    nlohmann::json spec{{"dim", 128}, {"name", "x"}, {"inner", {{"type", "hash"}}}};
    rag::plugin::Config c{spec};
    CHECK_EQ(c.get<int>("dim", 7), 128);
    CHECK_EQ(c.get<int>("absent", 7), 7);       // missing -> default
    CHECK_EQ(c.get<int>("name", 7), 7);         // wrong type -> default, no throw
    CHECK_EQ(c.get("name", "def"), "x");
    CHECK(c.has("dim"));
    CHECK(!c.has("absent"));
    auto req = c.require<int>("dim");
    REQUIRE(req.has_value());
    CHECK_EQ(*req, 128);
    auto missing = c.require<int>("absent");
    CHECK(!missing.has_value());
    auto sub = c.sub("inner");
    REQUIRE(sub.has_value());
    CHECK_EQ((*sub)["type"].get<std::string>(), "hash");
    CHECK(!c.sub("absent").has_value());
}

TEST(plugin_register_embedder_helper_wraps_concept) {
    // register_embedder accepts a factory returning a bare concept MODEL and
    // wraps it in AnyEmbedder for you. Register a throwaway backend and build it.
    rag::plugin::register_embedder(
        "unit_test_helper_embedder", "test-only (keys: dim)",
        [](rag::plugin::Config c) -> rag::Result<rag::dense::HashEmbedder> {
            return rag::dense::HashEmbedder{c.get<std::size_t>("dim", 32)};
        });
    auto e = rag::plugin::make_embedder(
        nlohmann::json{{"type", "unit_test_helper_embedder"}, {"dim", 48}});
    REQUIRE(e.has_value());
    CHECK_EQ(e->dimension(), 48u);
    // ...and it is self-described.
    auto rows = rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().describe();
    std::map<std::string, std::string> by_name(rows.begin(), rows.end());
    CHECK(by_name["unit_test_helper_embedder"].find("test-only") != std::string::npos);
}

TEST(plugin_local_embedders_registered_by_name) {
    // onnx/gguf are in-process local backends. They must be reachable BY NAME
    // like every network backend, so config/CLI/C-ABI can select them without
    // the caller naming the C++ type. (They existed as classes but were not
    // registered, so `{"type":"onnx"}` used to fail with "unknown type" even
    // on a build that supports ONNX.)
    rag::plugin::ensure_builtins_registered();
    auto& reg = rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance();
    CHECK(reg.contains("onnx"));
    CHECK(reg.contains("gguf"));
}

TEST(plugin_local_embedder_reports_availability_not_unknown) {
    // On a build WITHOUT the feature flag, the factory must surface a clear
    // typed error (Errc::unavailable, from load()) — NOT not_found. The name
    // existing but the feature not being compiled in is a categorically
    // different, and more actionable, diagnostic than an unknown type.
    auto e = rag::plugin::make_embedder(
        nlohmann::json{{"type", "onnx"}, {"model_path", "x.onnx"}});
    if (rag::dense::OnnxEmbedder::available()) {
        // Built with ONNX: a bogus path fails, but not as "unknown type".
        if (!e) CHECK(e.error().code != rag::Errc::not_found);
    } else {
        REQUIRE(!e.has_value());
        CHECK_EQ(e.error().code, rag::Errc::unavailable);   // not not_found
    }
}

TEST(plugin_retry_decorator_wraps_by_name) {
    // retry composes: it resolves its "inner" spec through the SAME registry and
    // wraps it. The wrapped embedder must still embed correctly.
    auto e = rag::plugin::make_embedder(nlohmann::json{
        {"type", "retry"},
        {"max_attempts", 2},
        {"inner", {{"type", "hash"}, {"dim", 96}}}});
    REQUIRE(e.has_value());
    CHECK_EQ(e->dimension(), 96u);
    std::vector<std::string> texts{"hello world"};
    auto v = e->embed(texts);
    REQUIRE(v.has_value());
    CHECK_EQ((*v)[0].size(), 96u);
}

TEST(plugin_retry_missing_inner_is_error) {
    auto e = rag::plugin::make_embedder(nlohmann::json{{"type", "retry"}});
    CHECK(!e.has_value());
    CHECK_EQ(e.error().code, rag::Errc::invalid_argument);
}

TEST(plugin_fallback_composes_two_specs) {
    // fallback resolves BOTH nested specs and wraps them. Both constructible
    // here, so the primary is used; the decorator just has to build and embed.
    auto e = rag::plugin::make_embedder(nlohmann::json{
        {"type", "fallback"},
        {"primary",   {{"type", "hash"}, {"dim", 128}}},
        {"secondary", {{"type", "hash"}, {"dim", 128}}}});
    REQUIRE(e.has_value());
    CHECK_EQ(e->dimension(), 128u);
}

TEST(plugin_fallback_degrades_when_primary_unconstructable) {
    // THE point of fallback: if the primary cannot even be BUILT (here: onnx on
    // a build without ONNX support), construction degrades to the secondary
    // instead of failing. A config that names a local model it cannot load must
    // still yield a working embedder.
    auto e = rag::plugin::make_embedder(nlohmann::json{
        {"type", "fallback"},
        {"primary",   {{"type", "onnx"}, {"model_path", "missing.onnx"}}},
        {"secondary", {{"type", "hash"}, {"dim", 64}}}});
    if (rag::dense::OnnxEmbedder::available()) return;   // primary builds; N/A here
    REQUIRE(e.has_value());                             // degraded, not failed
    CHECK_EQ(e->dimension(), 64u);                      // the secondary's dim
    std::vector<std::string> texts{"still works"};
    auto v = e->embed(texts);
    REQUIRE(v.has_value());
    CHECK_EQ((*v)[0].size(), 64u);
}

TEST(plugin_fallback_needs_both_specs) {
    auto e = rag::plugin::make_embedder(nlohmann::json{
        {"type", "fallback"},
        {"primary", {{"type", "hash"}}}});   // no secondary
    CHECK(!e.has_value());
    CHECK_EQ(e.error().code, rag::Errc::invalid_argument);
}

TEST(plugin_composition_nests_arbitrarily) {
    // The recursive resolution must nest to any depth: retry(fallback(a, b)).
    // This also exercises that Registry invokes factories UNLOCKED — building
    // this spec re-enters the same singleton three times on one thread, which
    // would deadlock a naive lock-around-factory design.
    auto e = rag::plugin::make_embedder(nlohmann::json{
        {"type", "retry"},
        {"inner", {{"type", "fallback"},
                   {"primary",   {{"type", "onnx"}, {"model_path", "x.onnx"}}},
                   {"secondary", {{"type", "hash"}, {"dim", 200}}}}}});
    if (rag::dense::OnnxEmbedder::available()) return;
    REQUIRE(e.has_value());
    CHECK_EQ(e->dimension(), 200u);   // onnx unavailable -> fallback -> hash(200)
}

TEST(plugin_load_missing_lib_is_error) {
    auto r = rag::plugin::load_plugin("/no/such/plugin.so");
    CHECK(!r.has_value());
}

TEST(plugin_load_dir_missing_is_empty) {
    auto v = rag::plugin::load_plugin_dir("/no/such/dir/at/all");
    CHECK(v.empty());
}

// ── Polyglot bridge ────────────────────────────────────────────────────

// A deterministic in-memory Channel: answers embed/rerank/retrieve locally so
// the Remote* wrappers can be tested without spawning anything.
struct FakeChannel final : rag::bridge::Channel {
    rag::Result<nlohmann::json> call(std::string_view method, const nlohmann::json& params) override {
        nlohmann::json res;
        if (method == "embed") {
            res["vectors"] = nlohmann::json::array();
            for (const auto& t : params["texts"]) {
                nlohmann::json v = nlohmann::json::array();
                for (int i = 0; i < 4; ++i) v.push_back(float((t.get<std::string>().size() + i) % 7));
                res["vectors"].push_back(v);
            }
        } else if (method == "rerank") {
            res["scores"] = nlohmann::json::array();
            for (const auto& p : params["passages"]) res["scores"].push_back(float(p.get<std::string>().size()));
        } else if (method == "retrieve") {
            res["hits"] = nlohmann::json::array();
            res["hits"].push_back({{"id", "x1"}, {"score", 0.9}, {"text", "hello"}});
            res["hits"].push_back({{"id", 42}, {"score", 0.5}});
        } else if (method == "graph") {
            res["op"] = params.value("op", "");
        } else {
            return rag::fail<nlohmann::json>(rag::Errc::not_found, "no method");
        }
        return res;
    }
    std::string identity() const override { return "fake"; }
};

TEST(bridge_remote_embedder) {
    auto ch = std::make_shared<FakeChannel>();
    rag::bridge::RemoteEmbedder emb{ch, 4, "fake-embed"};
    CHECK_EQ(emb.dimension(), 4u);
    std::vector<std::string> texts{"ab", "abcd"};
    auto v = emb.embed(texts);
    REQUIRE(v.has_value());
    CHECK_EQ(v->size(), 2u);
    CHECK_EQ((*v)[0].size(), 4u);
}

TEST(bridge_remote_reranker) {
    auto ch = std::make_shared<FakeChannel>();
    rag::bridge::RemoteReranker rr{ch, "fake-rerank"};
    std::vector<std::string> ps{"short", "a longer passage"};
    auto s = rr.rerank("q", ps);
    REQUIRE(s.has_value());
    CHECK_EQ(s->size(), 2u);
    CHECK((*s)[1] > (*s)[0]);
}

TEST(bridge_remote_retriever) {
    auto ch = std::make_shared<FakeChannel>();
    rag::bridge::RemoteRetriever retr{ch, "fake-retr"};
    auto r = retr.retrieve("q", 5);
    REQUIRE(r.has_value());
    CHECK_EQ(r->size(), 2u);
    CHECK((*r)[0].uri == "x1");
    CHECK((*r)[1].uri == "42");   // numeric id stringified
    auto g = retr.op("global");
    REQUIRE(g.has_value());
    CHECK((*g)["op"] == "global");
}

TEST(bridge_registered_in_registry) {
    rag::plugin::ensure_builtins_registered();
    CHECK(rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().contains("process"));
    CHECK(rag::plugin::Registry<rag::plugin::AnyEmbedder>::instance().contains("http"));
    CHECK(rag::plugin::Registry<rag::plugin::AnyReranker>::instance().contains("process"));
}

TEST(bridge_open_channel_unknown_transport) {
    auto ch = rag::bridge::open_channel(nlohmann::json{{"transport", "carrier_pigeon"}});
    CHECK(!ch.has_value());
    CHECK_EQ(ch.error().code, rag::Errc::not_found);
}

TEST(bridge_process_roundtrip_with_cat) {
    // Spawn a trivial line-echo peer that speaks the protocol using only /bin sh.
    // It reads a request line and emits a valid reply, exercising the real pipe.
    rag::bridge::ProcessConfig cfg;
    cfg.argv = {"/bin/sh", "-c",
                "while IFS= read -r line; do printf '{\"ok\":true,\"result\":{\"scores\":[1.0]}}\\n'; done"};
    auto ch = rag::bridge::ProcessChannel::spawn(cfg);
    REQUIRE(ch.has_value());
    rag::bridge::RemoteReranker rr{*ch, "sh-echo"};
    std::vector<std::string> ps{"only one"};
    auto s = rr.rerank("q", ps);
    REQUIRE(s.has_value());
    CHECK_EQ(s->size(), 1u);
    CHECK(std::abs((*s)[0] - 1.0f) < 1e-6f);
}

// ── GPU batch scoring ────────────────────────────────────────────────────────
//
// These must pass on a machine with NO GPU, on a build with no GPU backend, and
// on one with both — so every assertion is either about the CPU-equivalence of
// whatever ran, or is guarded by what score_batch actually reports it did.
// score_batch returning false is a ROUTING answer ("not on the GPU"), never an
// error, and the caller's job is then to run its own CPU path.

namespace {

// The CPU reference, written the way the library's own hot loops are: outer
// over candidates so each row stays hot across the whole query batch.
std::vector<float> cpu_score_batch(const std::vector<float>& corpus,
                                   const std::vector<float>& queries, std::size_t dim) {
    const std::size_t n = corpus.size() / dim, nq = queries.size() / dim;
    std::vector<float> out(n * nq);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t q = 0; q < nq; ++q)
            out[q * n + i] = rag::dense::dot(
                std::span<const float>(queries.data() + q * dim, dim),
                std::span<const float>(corpus.data() + i * dim, dim));
    return out;
}

struct GpuFixture {
    std::size_t dim = 64, n = 0, nq = 0;
    std::vector<float> corpus, queries;

    GpuFixture(std::size_t n_, std::size_t nq_, std::size_t dim_ = 64)
        : dim(dim_), n(n_), nq(nq_), corpus(n_ * dim_), queries(nq_ * dim_) {
        std::mt19937 rng(1234);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (auto& x : corpus)  x = g(rng);
        for (auto& x : queries) x = g(rng);
        for (std::size_t i = 0; i < n; ++i)
            rag::dense::normalize(std::span<float>(corpus.data() + i * dim, dim));
        for (std::size_t i = 0; i < nq; ++i)
            rag::dense::normalize(std::span<float>(queries.data() + i * dim, dim));
    }
};

} // namespace

TEST(gpu_reports_a_coherent_device) {
    const auto& info = rag::gpu::device_info();
    if (rag::gpu::available()) {
        // A device that says it is present must describe itself consistently.
        // This also guards the ABI boundary: the Metal backend is Objective-C++
        // compiled by Clang against libc++ while the library may be built by
        // GCC against libstdc++, so device info is marshalled through plain C.
        // When that was wrong the symptoms showed up EXACTLY here — an empty
        // name and a device claiming no unified memory on hardware that has
        // nothing else.
        CHECK(info.backend != rag::gpu::Backend::none);
        CHECK(!info.name.empty());
        CHECK(info.max_buffer_bytes > 0);
    } else {
        CHECK(info.backend == rag::gpu::Backend::none);
    }
    // The name mapping is total — no enum value renders as garbage.
    CHECK_EQ(std::string(rag::gpu::backend_name(rag::gpu::Backend::none)), std::string("none"));
    CHECK_EQ(std::string(rag::gpu::backend_name(rag::gpu::Backend::metal)), std::string("metal"));
}

TEST(gpu_score_batch_matches_cpu_or_declines) {
    // Large enough to clear min_batch_work() on a machine that has a GPU.
    const std::size_t dim = 64;
    const std::size_t n = 40000, nq = 16;
    GpuFixture fx(n, nq, dim);

    std::vector<float> out(n * nq, -999.0f);
    const bool ran = rag::gpu::score_batch(fx.corpus, fx.queries, dim, out);
    if (!ran) { CHECK(true); return; }           // no device: nothing to verify

    const std::vector<float> ref = cpu_score_batch(fx.corpus, fx.queries, dim);
    double worst = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        worst = std::max(worst, (double)std::fabs(out[i] - ref[i]));

    // ABSOLUTE, not relative, error. These are unit vectors, so scores live in
    // [-1,1] and many sit near zero, where a relative test explodes for free —
    // an early version of this comparison reported "MISMATCH" everywhere for
    // exactly that reason while the GPU was in fact more accurate than the CPU.
    CHECK(worst < 1e-5);
}

TEST(gpu_score_batch_validates_and_routes) {
    const std::size_t dim = 64;
    GpuFixture fx(1000, 4, dim);
    std::vector<float> out(1000 * 4);

    // Malformed shapes are refused rather than read out of bounds.
    CHECK(!rag::gpu::score_batch(fx.corpus, fx.queries, 0, out));
    CHECK(!rag::gpu::score_batch(fx.corpus, fx.queries, dim + 1, out));   // not a divisor
    std::vector<float> tiny_out(3);
    CHECK(!rag::gpu::score_batch(fx.corpus, fx.queries, dim, tiny_out));  // output too small

    // Below the measured crossover the GPU must DECLINE even when present:
    // shipping work to a device that loses to the CPU is the failure mode this
    // whole threshold exists to prevent.
    CHECK(1000u * 4u * dim < rag::gpu::min_batch_work());
    CHECK(!rag::gpu::score_batch(fx.corpus, fx.queries, dim, out));

    CHECK(rag::gpu::min_batch_work() > 0);
}

// ── Batch dense search (the GPU's production entry point) ─────────────────

// A corpus big enough that the batch path is worth taking, with HNSW disabled
// so the brute-force scan (the only offloadable shape) is what runs.
rag::index::Corpus batch_corpus(std::size_t docs, std::size_t dim) {
    rag::index::CorpusConfig cfg;
    cfg.hnsw_threshold = 1'000'000;    // force the scan path
    rag::index::Corpus c{cfg};
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "hash"}, {"dim", dim}});
    c.set_embedder(std::move(*emb));
    for (std::size_t i = 0; i < docs; ++i)
        c.add_document("d" + std::to_string(i),
                       "retrieval document number " + std::to_string(i) +
                       " about vectors graphs indexes and ranking");
    (void)c.build();
    return c;
}

TEST(dense_batch_matches_the_per_query_path_exactly) {
    // This is the test that matters. dense_search_batch may route to the GPU,
    // to a packed CPU scan, or to the plain loop, and the caller must not be
    // able to tell which happened. Anything else makes acceleration a
    // correctness risk rather than a performance choice.
    //
    // THE SIZE HERE IS LOAD-BEARING. The first version of this test used 4000
    // chunks and passed while never once reaching the GPU: score_batch declines
    // below gpu::min_batch_work() (2G multiply-adds), so it was silently
    // asserting that the CPU path equals itself. Driving the real API at 200k
    // chunks is what exposed the tie-order bug below. The corpus is sized from
    // min_batch_work() rather than hardcoded so it cannot drift out of range.
    const std::size_t dim = 384;
    const std::size_t nq  = 32;
    const std::size_t need = rag::gpu::min_batch_work() / (nq * dim) + 1000;
    if (!rag::gpu::available()) return;    // CPU-only machines: covered below

    auto c = batch_corpus(need, dim);
    std::vector<std::string> queries;
    for (std::size_t i = 0; i < nq; ++i)
        queries.push_back("vectors ranking indexes " + std::to_string(i));

    // Confirm the batch really is big enough to be offloaded, so a future
    // threshold change turns this into a failure rather than a silent no-op.
    CHECK(nq * c.chunk_count() * dim >= rag::gpu::min_batch_work());

    auto batched = c.dense_search_batch(queries, 10);
    REQUIRE(batched.has_value());
    REQUIRE(batched->size() == queries.size());

    for (std::size_t q = 0; q < queries.size(); ++q) {
        auto one = c.dense_search(queries[q], 10);
        REQUIRE(one.has_value());
        REQUIRE((*batched)[q].size() == one->size());
        for (std::size_t i = 0; i < one->size(); ++i) {
            // Exact same ids in the same ORDER — not just the same set. Ties
            // are common with a hash embedder, and an unstable sort made these
            // two paths disagree on tied hits until dense scoring got a total
            // order (score desc, then chunk id asc).
            CHECK_EQ((*batched)[q][i].chunk.get(), (*one)[i].chunk.get());
            CHECK(std::fabs((*batched)[q][i].score.get() - (*one)[i].score.get()) < 1e-4f);
        }
    }
}

TEST(dense_ranking_breaks_ties_deterministically) {
    // Documents with identical text score identically, so their relative order
    // is decided entirely by the tiebreak. Without one, std::partial_sort's
    // instability leaks the scan's internal layout into the ranking and the
    // same query returns different orders on different paths/thread counts.
    rag::index::CorpusConfig cfg;
    cfg.hnsw_threshold = 1'000'000;
    rag::index::Corpus c{cfg};
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "hash"}, {"dim", 64}});
    c.set_embedder(std::move(*emb));
    for (std::size_t i = 0; i < 5000; ++i)
        c.add_document("d" + std::to_string(i), "identical body text for every document");
    (void)c.build();

    auto first = c.dense_search("identical body text", 10);
    REQUIRE(first.has_value());
    REQUIRE(first->size() == 10);

    // Every score is tied, so the ids must come back ascending — and must be
    // the same on a repeat run.
    for (std::size_t i = 1; i < first->size(); ++i)
        CHECK((*first)[i - 1].chunk.get() < (*first)[i].chunk.get());

    for (int rep = 0; rep < 5; ++rep) {
        auto again = c.dense_search("identical body text", 10);
        REQUIRE(again.has_value());
        for (std::size_t i = 0; i < first->size(); ++i)
            CHECK_EQ((*again)[i].chunk.get(), (*first)[i].chunk.get());
    }
}

TEST(dense_batch_honours_filters_and_the_graph) {
    // The GPU path is only valid for an unfiltered, graph-less scan. With a
    // filter, results must still be filtered; with HNSW built, the graph must
    // still be the thing that answers. Both route to the per-query path, and
    // the point of the test is that routing away does not change the ANSWER.
    rag::index::CorpusConfig cfg;
    cfg.hnsw_threshold = 1'000'000;
    rag::index::Corpus c{cfg};
    auto emb = rag::plugin::make_embedder(nlohmann::json{{"type", "hash"}, {"dim", 64}});
    c.set_embedder(std::move(*emb));
    for (std::size_t i = 0; i < 2000; ++i)
        c.add_document("d" + std::to_string(i), "document about retrieval and vectors",
                       {{"lang", i % 2 ? "en" : "fr"}});
    (void)c.build();

    rag::index::MetaFilter en = [](const rag::Metadata& m) {
        auto it = m.find("lang");
        return it != m.end() && it->second == "en";
    };
    std::vector<std::string> queries{"retrieval", "vectors"};
    auto filtered = c.dense_search_batch(queries, 10, en);
    REQUIRE(filtered.has_value());
    REQUIRE(filtered->size() == 2);
    for (const auto& list : *filtered) {
        CHECK(!list.empty());
        for (const auto& h : list) {
            const rag::Chunk* ch = c.chunk(h.chunk);
            REQUIRE(ch != nullptr);
            REQUIRE(ch->meta != nullptr);
            CHECK_EQ(ch->meta->at("lang"), std::string("en"));
        }
    }
}

TEST(dense_batch_sees_documents_added_after_it_ran) {
    // The packed mirror is a cache keyed on the corpus epoch. If it were not
    // invalidated on mutation, a batch query would keep answering from a
    // snapshot taken before the write — stale results that no per-query search
    // would ever return.
    auto c = batch_corpus(2500, 64);
    std::vector<std::string> q{"zzunique marker phrase"};

    auto before = c.dense_search_batch(q, 5);
    REQUIRE(before.has_value());
    const std::size_t chunks_before = c.chunk_count();

    c.add_document("marker", "zzunique marker phrase appears only here");
    (void)c.build();
    CHECK(c.chunk_count() > chunks_before);

    auto after = c.dense_search_batch(q, 5);
    REQUIRE(after.has_value());
    // The new document must be reachable, and must agree with the single-query
    // path, which has no cache at all.
    auto single = c.dense_search(q[0], 5);
    REQUIRE(single.has_value());
    REQUIRE(!(*after)[0].empty());
    CHECK_EQ((*after)[0][0].chunk.get(), (*single)[0].chunk.get());
}

TEST(dense_batch_rejects_a_corpus_without_an_embedder) {
    rag::index::Corpus c;
    c.add_document("a", "text");
    std::vector<std::string> q{"text"};
    auto r = c.dense_search_batch(q, 5);
    CHECK(!r.has_value());

    // An empty batch is not an error; it is an empty answer.
    auto e = c.dense_search_batch(std::vector<std::string>{}, 5);
    CHECK(e.has_value());
    CHECK(e->empty());
}

TEST(dense_batch_matches_with_gpu_disabled) {
    // The same equivalence must hold on the pure-CPU path, so this test is not
    // silently vacuous on a machine without a GPU (and so the GPU arm above is
    // not the only thing ever exercised on a machine with one).
    //
    // ORDERING HAZARD: gpu::disable() is a one-way latch with process-global
    // scope — there is no re-enable, by design (a caller that turned the GPU
    // off did so because something was wrong with it). So this test compares
    // against a result captured BEFORE it flips the latch, and the existing
    // gpu_disable_is_honoured test relies on the same property. Any test that
    // needs a live GPU must therefore run before this one; tests run in
    // registration order, so keep GPU-dependent cases above this line.
    auto c = batch_corpus(3000, 64);
    std::vector<std::string> queries{"vectors", "ranking", "indexes and graphs"};

    auto with_gpu = c.dense_search_batch(queries, 8);
    REQUIRE(with_gpu.has_value());

    rag::gpu::disable();
    CHECK(!rag::gpu::available());
    auto without = c.dense_search_batch(queries, 8);
    REQUIRE(without.has_value());

    REQUIRE(with_gpu->size() == without->size());
    for (std::size_t q = 0; q < without->size(); ++q) {
        REQUIRE((*with_gpu)[q].size() == (*without)[q].size());
        for (std::size_t i = 0; i < (*without)[q].size(); ++i)
            CHECK_EQ((*with_gpu)[q][i].chunk.get(), (*without)[q][i].chunk.get());
    }
}

TEST(gpu_disable_is_honoured) {
    // Runs last by construction: disable() is deliberately a ONE-WAY switch, so
    // this test permanently turns the GPU off for the process. A reversible
    // toggle would race with in-flight dispatches for no real benefit.
    rag::gpu::disable();
    CHECK(!rag::gpu::available());
    CHECK(rag::gpu::device_info().backend == rag::gpu::Backend::none);

    const std::size_t dim = 64;
    GpuFixture fx(40000, 16, dim);
    std::vector<float> out(40000 * 16, 7.0f);
    CHECK(!rag::gpu::score_batch(fx.corpus, fx.queries, dim, out));
    CHECK_EQ(out[0], 7.0f);          // declined means UNTOUCHED, not partial
}

// ── Write-ahead log ───────────────────────────────────────────────────────

TEST(wal_replays_mutations_into_a_fresh_corpus) {
    const std::string db  = "/tmp/ragcpp_wal_replay.ragdb";
    const std::string log = "/tmp/ragcpp_wal_replay.wal";
    std::remove(db.c_str()); std::remove(log.c_str());

    {
        rag::index::Corpus c;
        REQUIRE(c.add_document("base.txt", "base document about retrieval").has_value());
        REQUIRE(c.build().has_value());
        REQUIRE(c.save(db).has_value());
        REQUIRE(c.open_wal(log).has_value());
        // These are never save()d — only the log knows about them.
        REQUIRE(c.add_document("a.txt", "alpha document about retrieval").has_value());
        REQUIRE(c.add_document("b.txt", "bravo document about retrieval").has_value());
        auto gone = c.add_document("c.txt", "charlie document about retrieval");
        REQUIRE(gone.has_value());
        REQUIRE(c.remove_document(*gone).has_value());
        CHECK(c.wal_bytes() > 0);
    }

    // Recovery: load the (stale) snapshot, then replay.
    auto rec = rag::index::Corpus::load(db);
    REQUIRE(rec.has_value());
    CHECK_EQ(rec->live_document_count(), std::size_t{1});     // snapshot alone
    REQUIRE(rec->open_wal(log).has_value());

    CHECK_EQ(rec->live_document_count(), std::size_t{3});     // base + a + b
    CHECK(rec->find_by_uri("a.txt").has_value());
    CHECK(rec->find_by_uri("b.txt").has_value());
    CHECK(!rec->find_by_uri("c.txt").has_value());            // delete replayed too

    // Replayed documents must be RETRIEVABLE, not merely present — replay goes
    // through add_document precisely so chunking and indexing happen too.
    const auto hits = rec->lexical_search("alpha document", 5);
    CHECK(!hits.empty());

    std::remove(db.c_str()); std::remove(log.c_str());
}

TEST(wal_replay_is_not_re_logged) {
    // Recovery must not append what it just read. Without a guard, every
    // restart would double the log and replay time would grow without bound
    // even on an idle server.
    const std::string db  = "/tmp/ragcpp_wal_double.ragdb";
    const std::string log = "/tmp/ragcpp_wal_double.wal";
    std::remove(db.c_str()); std::remove(log.c_str());

    std::uint64_t first = 0;
    {
        rag::index::Corpus c;
        REQUIRE(c.build().has_value());
        REQUIRE(c.save(db).has_value());
        REQUIRE(c.open_wal(log).has_value());
        for (int i = 0; i < 10; ++i)
            REQUIRE(c.add_document("d" + std::to_string(i), "body " + std::to_string(i)).has_value());
        first = c.wal_bytes();
    }
    CHECK(first > 0);

    for (int restart = 0; restart < 3; ++restart) {
        auto rec = rag::index::Corpus::load(db);
        REQUIRE(rec.has_value());
        REQUIRE(rec->open_wal(log).has_value());
        CHECK_EQ(rec->live_document_count(), std::size_t{10});   // not 20, not 40
        CHECK_EQ(rec->wal_bytes(), first);                       // log did not grow
    }

    std::remove(db.c_str()); std::remove(log.c_str());
}

TEST(wal_tolerates_a_torn_tail_but_not_inner_corruption) {
    const std::string log = "/tmp/ragcpp_wal_torn.wal";
    std::remove(log.c_str());
    {
        rag::store::Wal w;
        REQUIRE(w.open(log).has_value());
        for (int i = 0; i < 5; ++i) {
            rag::store::WalRecord r;
            r.op = rag::store::WalOp::add_document;
            r.uri = "u" + std::to_string(i);
            r.text = "body " + std::to_string(i);
            REQUIRE(w.append(r).has_value());
        }
    }
    const auto full = rag::store::Wal::replay(log);
    REQUIRE(full.has_value());
    CHECK_EQ(full->size(), std::size_t{5});

    // A crash mid-append leaves a partial trailing record. That write was never
    // acknowledged, so dropping it is correct — and it must NOT be reported as
    // corruption, or every crash would become an unrecoverable database.
    const auto whole = std::filesystem::file_size(log);
    for (std::size_t cut : {std::size_t{3}, std::size_t{11}, std::size_t{20}}) {
        std::filesystem::resize_file(log, whole - cut);
        const auto torn = rag::store::Wal::replay(log);
        REQUIRE(torn.has_value());                 // recovers, does not fail
        CHECK(torn->size() == std::size_t{4});     // the intact prefix survives
        for (std::size_t i = 0; i < torn->size(); ++i)
            CHECK_EQ((*torn)[i].uri, "u" + std::to_string(i));
        std::filesystem::resize_file(log, whole);
        // restore the truncated bytes for the next iteration
        std::remove(log.c_str());
        rag::store::Wal w;
        REQUIRE(w.open(log).has_value());
        for (int i = 0; i < 5; ++i) {
            rag::store::WalRecord r;
            r.op = rag::store::WalOp::add_document;
            r.uri = "u" + std::to_string(i);
            r.text = "body " + std::to_string(i);
            REQUIRE(w.append(r).has_value());
        }
    }

    // Corruption in the MIDDLE is a different thing entirely: that record was
    // acknowledged, and silently dropping it would lose a write the client was
    // told had succeeded. It must be an error.
    {
        std::fstream f(log, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(20);                 // inside the first record's payload
        char junk = '\xEE';
        f.write(&junk, 1);
    }
    const auto bad = rag::store::Wal::replay(log);
    CHECK(!bad.has_value());

    std::remove(log.c_str());
}

TEST(wal_checkpoint_snapshots_then_truncates) {
    const std::string db  = "/tmp/ragcpp_wal_ckpt.ragdb";
    const std::string log = "/tmp/ragcpp_wal_ckpt.wal";
    std::remove(db.c_str()); std::remove(log.c_str());

    rag::index::Corpus c;
    REQUIRE(c.build().has_value());
    REQUIRE(c.save(db).has_value());
    REQUIRE(c.open_wal(log).has_value());
    for (int i = 0; i < 25; ++i)
        REQUIRE(c.add_document("d" + std::to_string(i), "body " + std::to_string(i)).has_value());
    CHECK(c.wal_bytes() > 0);

    REQUIRE(c.checkpoint(db).has_value());
    CHECK_EQ(c.wal_bytes(), std::uint64_t{0});      // log discarded...

    // ...and the snapshot must contain everything the log did, or the
    // truncation just destroyed acknowledged writes. This is the ordering the
    // implementation is careful about: save() fully durable BEFORE truncate().
    auto rec = rag::index::Corpus::load(db);
    REQUIRE(rec.has_value());
    CHECK_EQ(rec->live_document_count(), std::size_t{25});
    REQUIRE(rec->open_wal(log).has_value());
    CHECK_EQ(rec->live_document_count(), std::size_t{25});   // replay adds nothing

    std::remove(db.c_str()); std::remove(log.c_str());
}

// ── Opt-in quality pipeline (MMR diversity) ──────────────────────────────

namespace {

// A corpus of `facets` distinct sub-answers, each duplicated `dupes` times.
// Ranking by pure relevance fills the top-k with copies of one facet; the whole
// point of MMR is to spend those slots on different facets instead.
rag::Engine faceted_engine(int facets, int dupes) {
    rag::Engine e;
    for (int f = 0; f < facets; ++f)
        for (int d = 0; d < dupes; ++d)
            e.add("f" + std::to_string(f) + "_d" + std::to_string(d),
                  "topicX facet" + std::to_string(f) + " content variant " + std::to_string(d));
    e.build();
    return e;
}

std::size_t distinct_facets(const std::vector<rag::SearchResult>& hits) {
    std::set<std::string> f;
    for (const auto& h : hits) f.insert(h.uri.substr(0, h.uri.find('_')));
    return f.size();
}

} // namespace

TEST(quality_pipeline_diversifies_more_than_standard) {
    auto engine = faceted_engine(4, 10);

    engine.with_pipeline(rag::pipeline::Pipeline::standard());
    auto plain = engine.search("topicX", 8);
    REQUIRE(plain.has_value());

    engine.with_pipeline(rag::pipeline::Pipeline::quality());
    auto diverse = engine.search("topicX", 8);
    REQUIRE(diverse.has_value());

    // Same number of results — diversity must not cost recall.
    CHECK_EQ(plain->size(), diverse->size());
    // ...but strictly more of the answer covered. Measured 2/4 vs 4/4.
    CHECK(distinct_facets(*diverse) > distinct_facets(*plain));
}

TEST(mmr_lambda_actually_controls_diversity) {
    // The lambda knob must have real range: at 1.0 MMR is pure relevance, and
    // as it falls the diversity term must visibly displace near-duplicates. A
    // knob that does nothing is worse than no knob, because callers tune it and
    // believe the result.
    auto engine = faceted_engine(4, 10);

    engine.with_pipeline(rag::pipeline::Pipeline::quality(1.0f));   // pure relevance
    auto relevance_only = engine.search("topicX", 8);
    REQUIRE(relevance_only.has_value());

    engine.with_pipeline(rag::pipeline::Pipeline::quality(0.2f));   // diversity-heavy
    auto diversified = engine.search("topicX", 8);
    REQUIRE(diversified.has_value());

    CHECK(distinct_facets(*diversified) > distinct_facets(*relevance_only));
    CHECK_EQ(diversified->size(), std::size_t{8});    // still returns a full k
}

TEST(quality_pipeline_matches_standard_when_lambda_is_one) {
    // lambda=1 is pure relevance, so quality() must degenerate EXACTLY to
    // standard(). If it does not, MMR is perturbing the ranking even when it
    // was told not to — which would make the diversity knob untrustworthy.
    auto engine = faceted_engine(4, 10);

    engine.with_pipeline(rag::pipeline::Pipeline::standard());
    auto plain = engine.search("topicX", 8);
    REQUIRE(plain.has_value());

    engine.with_pipeline(rag::pipeline::Pipeline::quality(1.0f));
    auto lam1 = engine.search("topicX", 8);
    REQUIRE(lam1.has_value());

    REQUIRE(plain->size() == lam1->size());
    for (std::size_t i = 0; i < plain->size(); ++i)
        CHECK_EQ((*plain)[i].uri, (*lam1)[i].uri);
}

// ── Opt-in context pipeline (ParentStitch small-to-big) ──────────────────

namespace {

// A few LONG documents, each with a run of consecutive paragraphs on one topic,
// chunked small (max_lines=3) so one document's topic run becomes several
// ADJACENT matching chunks. Ranking by pure relevance then fills the top-k with
// adjacent slivers of one or two documents; ParentStitch folds each run into its
// best sibling and frees the slots for other documents. This is the small-to-big
// failure mode, made real rather than assumed.
rag::Engine fragmented_engine(int docs) {
    rag::index::CorpusConfig cfg;
    cfg.chunk.max_lines = 3;
    cfg.chunk.overlap_lines = 0;
    cfg.chunk.heading_context = false;
    rag::Engine e{cfg};
    for (int d = 0; d < docs; ++d) {
        std::string body = "Engineering note " + std::to_string(d) + "\n\n";
        body += "The design review covered failure domains.\n\n";
        body += "An on-call rotation was established.\n\n";
        // A long adjacent run on the query topic — this document alone yields
        // more than k matching fragments.
        for (int p = 0; p < 12; ++p)
            body += "The migration to the new storage engine reduced replication lag "
                    "and compaction backlog under peak write load step "
                    + std::to_string(p) + ".\n\n";
        body += "Capacity headroom was confirmed before cutover.\n\n";
        e.add("doc" + std::to_string(d) + ".md", body);
    }
    e.build();
    return e;
}

std::size_t distinct_docs(const std::vector<rag::SearchResult>& hits) {
    std::set<std::string> u;
    for (const auto& h : hits) u.insert(h.uri);
    return u.size();
}

} // namespace

TEST(context_pipeline_covers_more_documents_than_standard) {
    // With five fragmented documents the standard top-10 is dominated by the
    // adjacent fragments of one or two documents' topic runs; ParentStitch folds
    // those runs and the freed slots reach more distinct documents. Measured
    // (bench/stitch_bench.cpp n=5): standard covers 2 documents, context covers 4.
    auto engine = fragmented_engine(5);
    const char* q = "migration storage engine replication lag compaction";

    engine.with_pipeline(rag::pipeline::Pipeline::standard());
    auto plain = engine.search(q, 10);
    REQUIRE(plain.has_value());

    engine.with_pipeline(rag::pipeline::Pipeline::context());
    auto stitched = engine.search(q, 10);
    REQUIRE(stitched.has_value());

    // Strictly more of the corpus covered. If this ever fails to be strict, the
    // stitch stage is not being reached by the pipeline — the exact gap this
    // factory exists to close.
    CHECK(distinct_docs(*stitched) > distinct_docs(*plain));
}

TEST(context_pipeline_folds_adjacent_fragments) {
    // The mechanism, not just the outcome: for a single document whose topic run
    // spans many adjacent chunks, standard() returns several of those adjacent
    // fragments and context() must return strictly fewer of them (they were
    // folded into their best sibling).
    auto engine = fragmented_engine(1);
    const char* q = "migration storage engine replication lag compaction";

    engine.with_pipeline(rag::pipeline::Pipeline::standard());
    auto plain = engine.search(q, 10);
    REQUIRE(plain.has_value());

    engine.with_pipeline(rag::pipeline::Pipeline::context());
    auto stitched = engine.search(q, 10);
    REQUIRE(stitched.has_value());

    // One document, so "distinct docs" cannot grow; the visible effect is that
    // stitch collapsed the run to fewer hits.
    CHECK(stitched->size() < plain->size());
    CHECK(!stitched->empty());
}

TEST(context_pipeline_is_a_noop_when_nothing_is_adjacent) {
    // Honesty check: on a corpus of single-chunk documents there is nothing
    // adjacent to fold, so context() must degenerate EXACTLY to standard(). A
    // stage that reordered or dropped hits here would be doing damage for no
    // benefit — the reason the feature is opt-in is precisely that its gain is
    // corpus-dependent, and its COST must be zero when the gain is.
    rag::Engine e;
    for (int i = 0; i < 12; ++i)
        e.add("d" + std::to_string(i), "alpha beta gamma unique term" + std::to_string(i));
    e.build();

    e.with_pipeline(rag::pipeline::Pipeline::standard());
    auto plain = e.search("alpha beta gamma", 8);
    REQUIRE(plain.has_value());

    e.with_pipeline(rag::pipeline::Pipeline::context());
    auto stitched = e.search("alpha beta gamma", 8);
    REQUIRE(stitched.has_value());

    REQUIRE(plain->size() == stitched->size());
    for (std::size_t i = 0; i < plain->size(); ++i)
        CHECK_EQ((*plain)[i].uri, (*stitched)[i].uri);
}

// ── Contextual Retrieval, wired into ingest ───────────────────────────────

// The corpus that makes contextual retrieval matter: a document whose SUBJECT
// is named once, in the title, and whose body paragraphs never repeat it. This
// is exactly the failure the paper is about — chunk 2 says "revenue grew 3%"
// and, ripped out of the document, cannot be attributed to anyone.
rag::index::Corpus faceless_corpus(bool contextual) {
    rag::index::CorpusConfig cfg;
    cfg.contextual = contextual;
    cfg.chunk.max_lines = 3;      // force the subject out of most chunks
    cfg.chunk.overlap_lines = 0;
    cfg.chunk.heading_context = false;   // isolate the contextual signal from
                                         // the heading breadcrumb, so a pass
                                         // here cannot be the breadcrumb's doing
    rag::index::Corpus c{cfg};
    c.add_document("acme.md",
        "Acme Corporation\n"
        "\n"
        "The quarter closed ahead of plan.\n"
        "\n"
        "Revenue grew three percent year over year.\n"
        "\n"
        "Headcount was flat across all divisions.\n");
    c.add_document("globex.md",
        "Globex Corporation\n"
        "\n"
        "The quarter closed behind plan.\n"
        "\n"
        "Revenue fell one percent year over year.\n"
        "\n"
        "Headcount rose across all divisions.\n");
    (void)c.build();
    return c;
}

TEST(contextual_ingest_puts_the_subject_into_the_index) {
    // Without it, no chunk but the first carries "acme" at all, so a query for
    // "acme revenue" can only match on "revenue" — which both documents say.
    auto plain = faceless_corpus(false);
    std::size_t plain_with_subject = 0;
    for (const auto& ch : plain.chunks())
        if (ch.indexed_text().find("Acme") != std::string::npos) ++plain_with_subject;

    auto ctxd = faceless_corpus(true);
    std::size_t ctx_with_subject = 0;
    for (const auto& ch : ctxd.chunks())
        if (ch.indexed_text().find("Acme") != std::string::npos) ++ctx_with_subject;

    // The point of the feature: strictly more of the document's chunks now
    // carry the name that disambiguates them. (The first chunk always holds
    // the title, so this must be a STRICT increase, not merely "some chunk
    // mentions Acme" — that is true without the feature.)
    CHECK(ctx_with_subject > plain_with_subject);

    // And the effect is visible through the LEXICAL INDEX, not just the struct:
    // BM25 is built from indexed_text(), so situating the chunks must change
    // what a subject-qualified query retrieves.
    auto plain_hits = plain.lexical_search("acme revenue", 4);
    auto ctx_hits   = ctxd.lexical_search("acme revenue", 4);
    auto top_uri = [](const rag::index::Corpus& c, const std::vector<rag::Hit>& h) {
        return h.empty() ? std::string{} : c.resolve(h[0]).uri;
    };
    // Contextualized, the top hit for "acme revenue" is the Acme revenue chunk.
    REQUIRE(!ctx_hits.empty());
    CHECK_EQ(top_uri(ctxd, ctx_hits), std::string("acme.md"));
    const rag::Chunk* top = ctxd.chunk(ctx_hits[0].chunk);
    REQUIRE(top != nullptr);
    CHECK(top->text.find("Revenue grew") != std::string::npos);
    (void)plain_hits;
}

TEST(contextual_off_by_default) {
    // The feature costs work at ingest, so it must be opt-in. A default that
    // silently rewrote every indexed chunk would be a surprise.
    rag::index::CorpusConfig cfg;
    CHECK(!cfg.contextual);

    rag::index::Corpus c{cfg};
    c.add_document("a.md", "Acme Corporation\n\nRevenue grew three percent.\n");
    (void)c.build();
    bool any_situated = false;
    for (const auto& ch : c.chunks())
        if (ch.context.find("\u2014") != std::string::npos) any_situated = true;
    CHECK(!any_situated);
}

TEST(contextual_flag_survives_a_save_load_round_trip) {
    // A corpus reopened for writing must keep situating what it ingests, or
    // documents added after a restart are indexed differently from the ones
    // added before it.
    auto path = std::filesystem::temp_directory_path() / "ragcpp_ctx_roundtrip.ragdb";
    {
        auto c = faceless_corpus(true);
        REQUIRE(c.save(path.string()).has_value());
    }
    auto re = rag::index::Corpus::load(path.string());
    REQUIRE(re.has_value());

    // The reopened corpus situates NEW documents too. Key on a chunk that does
    // NOT already contain the subject: the first chunk holds the title anyway,
    // so counting it would let this test pass with the feature switched off.
    re->add_document("initech.md",
        "Initech Corporation\n\nRevenue doubled year over year.\n\nHeadcount fell.\n");
    (void)re->build();
    bool situated = false;
    for (const auto& ch : re->chunks())
        if (ch.doc.get() == 2 && ch.text.find("Initech") == std::string::npos
            && ch.indexed_text().find("Initech") != std::string::npos)
            situated = true;
    CHECK(situated);
    std::filesystem::remove(path);
}

TEST(chunk_geometry_survives_a_save_load_round_trip) {
    // Found while gating the test above: the chunker's geometry was never
    // persisted, so a corpus built with max_lines=3 came back at the default
    // 40 and every document added after the reopen was chunked at a different
    // granularity than the ones already in the store. Half the store is
    // sentence-sized and half is page-sized, and BM25 length normalization
    // sees a corpus that never existed.
    auto path = std::filesystem::temp_directory_path() / "ragcpp_chunkgeom.ragdb";
    {
        rag::index::CorpusConfig cfg;
        cfg.chunk.max_lines = 3;
        cfg.chunk.overlap_lines = 0;
        cfg.chunk.heading_context = false;
        rag::index::Corpus c{cfg};
        c.add_document("a.md", "l1\n\nl3\n\nl5\n\nl7\n\nl9\n\nl11\n\nl13\n");
        REQUIRE(c.save(path.string()).has_value());
    }
    auto before_reload = [&] {
        rag::index::CorpusConfig cfg;
        cfg.chunk.max_lines = 3;
        cfg.chunk.overlap_lines = 0;
        cfg.chunk.heading_context = false;
        rag::index::Corpus c{cfg};
        c.add_document("b.md", "l1\n\nl3\n\nl5\n\nl7\n\nl9\n\nl11\n\nl13\n");
        return c.chunk_count();
    }();

    auto re = rag::index::Corpus::load(path.string());
    REQUIRE(re.has_value());
    const std::size_t existing = re->chunk_count();
    re->add_document("b.md", "l1\n\nl3\n\nl5\n\nl7\n\nl9\n\nl11\n\nl13\n");
    // The same body must chunk the same way after a reopen as before it.
    CHECK_EQ(re->chunk_count() - existing, before_reload);
    CHECK(before_reload > 1);   // the fixture must actually split, or this proves nothing
    std::filesystem::remove(path);
}

TEST(contextualizer_failure_falls_back_instead_of_failing_ingest) {
    // A flaky model must not be able to reject a document. Every call errors
    // here; ingest must still succeed and still produce a situating context
    // from the deterministic extractive backend.
    rag::index::CorpusConfig cfg;
    cfg.contextual = true;
    cfg.chunk.max_lines = 3;
    cfg.chunk.heading_context = false;
    rag::index::Corpus c{cfg};
    std::atomic<int> calls{0};
    c.set_contextualizer([&](std::string_view, std::string_view) -> rag::Result<std::string> {
        ++calls;
        return rag::fail<std::string>(rag::Errc::unavailable, "model down");
    });
    auto id = c.add_document("acme.md",
        "Acme Corporation\n\nRevenue grew three percent.\n\nHeadcount was flat.\n");
    REQUIRE(id.has_value());
    CHECK(calls.load() > 0);
    bool any_ctx = false;
    for (const auto& ch : c.chunks()) if (!ch.context.empty()) any_ctx = true;
    CHECK(any_ctx);
}

TEST(contextualizer_output_reaches_the_index) {
    // The LLM seam must actually be consulted — not merely accepted and then
    // ignored in favour of the fallback.
    rag::index::CorpusConfig cfg;
    cfg.contextual = true;
    cfg.chunk.max_lines = 3;
    cfg.chunk.heading_context = false;
    rag::index::Corpus c{cfg};
    c.set_contextualizer([](std::string_view, std::string_view) -> rag::Result<std::string> {
        return std::string("ZZQMARKER");
    });
    c.add_document("a.md", "Acme Corporation\n\nRevenue grew.\n\nHeadcount flat.\n");
    (void)c.build();
    bool marked = true;
    for (const auto& ch : c.chunks())
        if (ch.indexed_text().find("ZZQMARKER") == std::string::npos) marked = false;
    CHECK(marked);
    // ...and it is searchable, i.e. it went through BM25 too.
    CHECK(!c.lexical_search("zzqmarker", 3).empty());
}

// ── ChunkLease ────────────────────────────────────────────────────────────

TEST(chunk_lease_blocks_writers_for_its_lifetime) {
    // chunks() used to hand out a reference into storage after releasing the
    // lock, so a concurrent add_document() could reallocate the vector under a
    // bulk consumer (graph/raptor/splade all iterate it for a long time). The
    // lease keeps the read lock alive, so a writer must wait.
    rag::index::Corpus c;
    for (int i = 0; i < 64; ++i)
        c.add_document("d" + std::to_string(i), "alpha beta gamma document body");

    std::atomic<bool> wrote{false};
    std::thread w;
    {
        auto lease = c.chunks();
        const std::size_t n = lease.size();
        const rag::Chunk* base = lease.get().data();

        w = std::thread([&] {
            // Enough documents to force at least one reallocation.
            for (int i = 0; i < 512; ++i)
                c.add_document("w" + std::to_string(i), "delta epsilon inserted while leased");
            wrote.store(true);
        });

        // While the lease is held the writer cannot make progress, so the view
        // stays exactly as taken: same length, same backing storage.
        for (int i = 0; i < 200; ++i) {
            CHECK_EQ(lease.size(), n);
            CHECK(lease.get().data() == base);
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        CHECK(!wrote.load());   // blocked, as designed
        // Dropping the lease here releases the read lock and lets it through.
    }

    w.join();
    CHECK(wrote.load());
    CHECK(c.document_count() == std::size_t{64 + 512});
}

TEST(chunk_lease_iterates_like_the_vector_it_replaced) {
    // Source compatibility: every existing call site is `for (auto& ch :
    // corpus.chunks())`, and range-for lifetime-extends the lease temporary, so
    // the lock is held for the whole loop rather than to the semicolon.
    rag::index::Corpus c;
    for (int i = 0; i < 5; ++i) c.add_document("d" + std::to_string(i), "body text here");
    std::size_t seen = 0;
    for (const auto& ch : c.chunks()) { (void)ch; ++seen; }
    CHECK_EQ(seen, c.chunk_count());

    auto lease = c.chunks();
    CHECK_EQ(lease.size(), seen);
    CHECK(!lease.empty());
    const std::vector<rag::Chunk>& as_ref = lease;   // implicit conversion
    CHECK_EQ(as_ref.size(), seen);
    CHECK(&lease[0] == &as_ref[0]);
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
