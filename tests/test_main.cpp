// tests/test_main.cpp — a minimal, dependency-free test harness + all tests.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <rag/rag.hpp>
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
            for (std::size_t d = 0; d < dim; ++d) v[d] = cents[c][d] + 0.06f * g(rng);
            rag::dense::normalize(v);
        }
    }

    // Mean recall@k of `idx` against exact brute-force search.
    double recall(const rag::index::HnswIndex& idx, std::size_t k, std::size_t queries) const {
        std::mt19937 qr(23);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(nc) - 1);
        double acc = 0;
        for (std::size_t qi = 0; qi < queries; ++qi) {
            int c = pick(qr);
            std::vector<float> q(dim);
            for (std::size_t d = 0; d < dim; ++d) q[d] = cents[c][d] + 0.06f * g(qr);
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
            for (const auto& h : idx.search(q, k))
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
