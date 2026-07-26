// bench/code_retrieval_bench.cpp — does structure-aware chunking actually
// RETRIEVE better on real code, or does it only keep definitions intact?
//
// structure_bench.cpp answered a narrower question: of the definitions in a
// file, how many land whole inside one chunk (integrity). Integrity is a
// necessary condition for good code retrieval but not a sufficient one — a
// chunker could keep every function whole and still lose the retrieval race if
// its chunks are worse targets for a natural-language query. The only way to
// know is to MEASURE retrieval, end to end, against ground truth.
//
// This bench does that on CodeSearchNet (python test split, real repositories):
//   * reconstruct real source files from the dataset (done offline; this reads
//     the .py files and a manifest of {query -> the function it documents,
//     located by line span})
//   * chunk every file TWO ways over the same corpus and embedder:
//        source   — rag::loaders::chunk_code   (definition-aligned)
//        windows  — rag::text::chunk_document   (fixed prose windows, the
//                   fallback every system without a code chunker uses)
//   * embed all chunks and all queries with one model, exact cosine top-k
//   * a hit is when the retrieved chunk's line span OVERLAPS the ground-truth
//     function's line span in the SAME file — i.e. the chunk that comes back
//     actually contains the function the query documents
//
// Metric: MRR@10 and Recall@{1,5,10}. Same corpus, same embedder, same queries;
// the ONLY thing that varies between the two arms is how files were cut into
// chunks. That isolates chunking, which is the claim.
//
// Usage:
//   ragcpp_code_retrieval_bench <corpus-dir> <manifest.json> \
//       --model=model.onnx --tokenizer=tokenizer.json [--limit=N]
//
// Requires -DRAGCPP_WITH_ONNX=ON (otherwise there is no embedder and nothing to
// measure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "rag/core/document.hpp"
#include "rag/dense/local_embedder.hpp"
#include "rag/loaders/code_chunker.hpp"
#include "rag/text/chunker.hpp"

namespace fs = std::filesystem;
using rag::Chunk;
using rag::DocId;

namespace {

const char* opt(int argc, char** argv, const char* key, const char* dflt) {
    const std::size_t kl = std::strlen(key);
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return argv[i] + kl + 1;
    return dflt;
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One evaluated arm. chunks[] belong to files; each chunk knows its file index
// and line span so overlap against a ground-truth span is a range test.
struct ChunkRef {
    std::uint32_t file;        // index into file list
    std::uint32_t start_line;
    std::uint32_t end_line;
    std::string   text;        // what we embed and score
};

struct Query {
    std::string   text;
    std::uint32_t file;
    std::uint32_t start_line;
    std::uint32_t end_line;
};

float cosine(const rag::Vector& a, const rag::Vector& b) {
    // Embedder L2-normalizes, so cosine is a dot product.
    float s = 0.f;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

bool overlaps(std::uint32_t a0, std::uint32_t a1, std::uint32_t b0, std::uint32_t b1) {
    return a0 <= b1 && b0 <= a1;
}

// Evaluate one chunking arm: embed its chunks, then score every query against
// ALL chunks in the whole corpus (the realistic setting — a user searches the
// corpus, not one known file). A hit is the retrieved chunk overlapping the
// ground-truth function's span in the correct file. Report MRR@10 and
// Recall@{1,5,10}.
void run_arm(const char* name,
             const std::vector<ChunkRef>& chunks,
             const std::vector<Query>& queries,
             const std::vector<rag::Vector>& qvecs,
             const rag::dense::OnnxEmbedder& emb,
             std::size_t /*nfiles*/) {
    // Embed chunk texts in batches.
    std::vector<rag::Vector> cvecs(chunks.size());
    {
        std::vector<std::string> batch;
        std::vector<std::size_t> idx;
        const std::size_t B = 256;
        auto flush = [&]() {
            if (batch.empty()) return;
            auto r = emb.embed(batch);
            if (!r) { std::printf("embed error: %s\n", r.error().message.c_str()); std::exit(1); }
            for (std::size_t j = 0; j < idx.size(); ++j) cvecs[idx[j]] = std::move((*r)[j]);
            batch.clear(); idx.clear();
        };
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            batch.push_back(chunks[i].text);
            idx.push_back(i);
            if (batch.size() >= B) flush();
        }
        flush();
    }

    double mrr = 0.0;
    std::size_t hit1 = 0, hit5 = 0, hit10 = 0, scored = 0;
    std::size_t total_chunks = chunks.size();

    for (std::size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& q = queries[qi];
        // Score against EVERY chunk in the corpus, take top-10.
        std::vector<std::pair<float, std::uint32_t>> scoredv;
        scoredv.reserve(chunks.size());
        for (std::uint32_t ci = 0; ci < chunks.size(); ++ci)
            scoredv.push_back({cosine(qvecs[qi], cvecs[ci]), ci});
        std::partial_sort(
            scoredv.begin(),
            scoredv.begin() + std::min<std::size_t>(10, scoredv.size()),
            scoredv.end(),
            [](auto& a, auto& b) { return a.first > b.first; });

        ++scored;
        const std::size_t topn = std::min<std::size_t>(10, scoredv.size());
        int rank = -1;
        for (std::size_t r = 0; r < topn; ++r) {
            const auto& c = chunks[scoredv[r].second];
            if (c.file == q.file &&
                overlaps(c.start_line, c.end_line, q.start_line, q.end_line)) { rank = (int)r; break; }
        }
        if (rank >= 0) {
            mrr += 1.0 / (rank + 1);
            if (rank < 1) ++hit1;
            if (rank < 5) ++hit5;
            if (rank < 10) ++hit10;
        }
    }

    std::printf("  %-9s chunks=%-7zu MRR@10=%.4f  R@1=%.4f  R@5=%.4f  R@10=%.4f\n",
                name, total_chunks,
                scored ? mrr / scored : 0.0,
                scored ? (double)hit1 / scored : 0.0,
                scored ? (double)hit5 / scored : 0.0,
                scored ? (double)hit10 / scored : 0.0);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <corpus-dir> <manifest.json> --model=M --tokenizer=T [--limit=N]\n", argv[0]);
        return 2;
    }
    const fs::path corpus_dir = argv[1];
    const fs::path manifest_path = argv[2];
    const std::string model = opt(argc, argv, "--model", "");
    const std::string tok   = opt(argc, argv, "--tokenizer", "");
    const std::size_t limit = std::strtoul(opt(argc, argv, "--limit", "0"), nullptr, 10);
    if (model.empty()) { std::printf("need --model (this bench measures retrieval, which needs an embedder)\n"); return 2; }

    // Load the model.
    rag::dense::LocalEmbedderConfig ec;
    ec.model_path = model; ec.tokenizer_path = tok; ec.max_tokens = 256;
    auto emb = rag::dense::OnnxEmbedder::load(ec);
    if (!emb) { std::printf("embedder error: %s\n", emb.error().message.c_str()); return 1; }
    std::printf("embedder: %s (dim %zu)\n", model.c_str(), emb->dimension());

    // Load manifest.
    nlohmann::json manifest;
    { std::ifstream f(manifest_path); f >> manifest; }
    if (limit && manifest.size() > limit)
        manifest.erase(manifest.begin() + limit, manifest.end());

    // Assign each referenced file an index; slurp its body once.
    std::unordered_map<std::string, std::uint32_t> file_idx;
    std::vector<std::string> file_name, file_body;
    auto intern = [&](const std::string& fn) -> std::uint32_t {
        auto it = file_idx.find(fn);
        if (it != file_idx.end()) return it->second;
        std::uint32_t id = (std::uint32_t)file_name.size();
        file_idx[fn] = id;
        file_name.push_back(fn);
        file_body.push_back(slurp(corpus_dir / fn));
        return id;
    };

    std::vector<Query> queries;
    for (auto& m : manifest) {
        Query q;
        q.text = m["query"].get<std::string>();
        q.file = intern(m["file"].get<std::string>());
        q.start_line = m["start_line"].get<std::uint32_t>();
        q.end_line = m["end_line"].get<std::uint32_t>();
        queries.push_back(std::move(q));
    }
    const std::size_t nfiles = file_name.size();
    std::printf("files=%zu  queries=%zu\n\n", nfiles, queries.size());

    // Embed all queries once (shared by both arms).
    std::vector<rag::Vector> qvecs(queries.size());
    {
        std::vector<std::string> batch; std::vector<std::size_t> idx;
        const std::size_t B = 256;
        auto flush = [&]() {
            if (batch.empty()) return;
            auto r = emb->embed(batch);
            if (!r) { std::printf("query embed error: %s\n", r.error().message.c_str()); return; }
            for (std::size_t j = 0; j < idx.size(); ++j) qvecs[idx[j]] = std::move((*r)[j]);
            batch.clear(); idx.clear();
        };
        for (std::size_t i = 0; i < queries.size(); ++i) {
            batch.push_back(queries[i].text); idx.push_back(i);
            if (batch.size() >= B) flush();
        }
        flush();
    }

    // Build the two chunk sets over the same files.
    std::vector<ChunkRef> source_chunks, window_chunks;
    for (std::uint32_t fi = 0; fi < nfiles; ++fi) {
        const std::string& body = file_body[fi];
        // source: definition-aligned.
        for (auto& c : rag::loaders::chunk_code(DocId::invalid(), ".py", body)) {
            source_chunks.push_back({fi, c.start_line, c.end_line, c.text});
        }
        // windows: fixed prose windows (default options — what a non-code system does).
        for (auto& c : rag::text::chunk_document(DocId::invalid(), body)) {
            window_chunks.push_back({fi, c.start_line, c.end_line, c.text});
        }
    }

    std::printf("Same corpus, same embedder, same queries. Only chunking differs.\n"
                "A hit = the retrieved chunk's line span overlaps the documented\n"
                "function's span in the same file.\n\n");
    run_arm("source",  source_chunks, queries, qvecs, *emb, nfiles);
    run_arm("windows", window_chunks, queries, qvecs, *emb, nfiles);
    return 0;
}
