// rag/eval/beir.cpp — BEIR-format loader + trec_eval metrics.

#include "rag/eval/beir.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace rag::eval {
namespace {

Result<std::vector<nlohmann::json>> read_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f) return std::unexpected(Error{Errc::not_found, "cannot open " + path});
    std::vector<nlohmann::json> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try { out.push_back(nlohmann::json::parse(line)); }
        catch (...) { return std::unexpected(Error{Errc::parse_error, "bad jsonl line in " + path}); }
    }
    return out;
}

std::string jstr(const nlohmann::json& j, const char* key) {
    if (!j.contains(key)) return {};
    const auto& v = j[key];
    return v.is_string() ? v.get<std::string>() : v.dump();
}

} // namespace

Result<BeirDataset>
BeirDataset::load(const std::string& dir, const std::string& qrels_split) {
    BeirDataset ds;
    auto corpus = read_jsonl(dir + "/corpus.jsonl");
    if (!corpus) return std::unexpected(corpus.error());
    for (auto& j : *corpus)
        ds.corpus.push_back({jstr(j, "_id"), jstr(j, "title"), jstr(j, "text")});

    auto queries = read_jsonl(dir + "/queries.jsonl");
    if (!queries) return std::unexpected(queries.error());
    for (auto& j : *queries)
        ds.queries.push_back({jstr(j, "_id"), jstr(j, "text")});

    // qrels: TSV with a header line "query-id\tcorpus-id\tscore".
    std::string qpath = dir + "/qrels/" + qrels_split + ".tsv";
    std::ifstream qf(qpath);
    if (!qf) return std::unexpected(Error{Errc::not_found, "cannot open " + qpath});
    std::string line;
    bool first = true;
    while (std::getline(qf, line)) {
        if (line.empty()) continue;
        if (first) { first = false; if (line.find("query") != std::string::npos) continue; }
        std::istringstream ss(line);
        std::string qid, did, score;
        if (!std::getline(ss, qid, '\t') || !std::getline(ss, did, '\t') || !std::getline(ss, score, '\t'))
            continue;
        int rel = 0;
        try { rel = std::stoi(score); } catch (...) { continue; }
        ds.qrels[qid][did] = rel;
    }
    return ds;
}

// ── Metric primitives ────────────────────────────────────────────────────────
double ndcg_at_k(const Ranking& run, const std::unordered_map<std::string, int>& rel, std::size_t k) {
    double dcg = 0.0;
    std::size_t n = std::min(k, run.size());
    for (std::size_t i = 0; i < n; ++i) {
        auto it = rel.find(run[i]);
        int g = it != rel.end() ? it->second : 0;
        if (g > 0) dcg += ((1u << g) - 1) / std::log2((double)(i + 2));
    }
    // ideal DCG: sort the graded relevances descending.
    std::vector<int> grades;
    for (auto& [d, g] : rel) if (g > 0) grades.push_back(g);
    std::sort(grades.begin(), grades.end(), std::greater<int>());
    double idcg = 0.0;
    for (std::size_t i = 0; i < std::min(k, grades.size()); ++i)
        idcg += ((1u << grades[i]) - 1) / std::log2((double)(i + 2));
    return idcg > 0.0 ? dcg / idcg : 0.0;
}

double recall_at_k(const Ranking& run, const std::unordered_map<std::string, int>& rel, std::size_t k) {
    std::size_t total_rel = 0;
    for (auto& [d, g] : rel) if (g > 0) ++total_rel;
    if (total_rel == 0) return 0.0;
    std::size_t found = 0, n = std::min(k, run.size());
    for (std::size_t i = 0; i < n; ++i) {
        auto it = rel.find(run[i]);
        if (it != rel.end() && it->second > 0) ++found;
    }
    return (double)found / (double)total_rel;
}

static double precision_at_k(const Ranking& run, const std::unordered_map<std::string, int>& rel, std::size_t k) {
    if (k == 0) return 0.0;
    std::size_t found = 0, n = std::min(k, run.size());
    for (std::size_t i = 0; i < n; ++i) {
        auto it = rel.find(run[i]);
        if (it != rel.end() && it->second > 0) ++found;
    }
    return (double)found / (double)k;
}

double average_precision(const Ranking& run, const std::unordered_map<std::string, int>& rel) {
    std::size_t total_rel = 0;
    for (auto& [d, g] : rel) if (g > 0) ++total_rel;
    if (total_rel == 0) return 0.0;
    double sum = 0.0; std::size_t hits = 0;
    for (std::size_t i = 0; i < run.size(); ++i) {
        auto it = rel.find(run[i]);
        if (it != rel.end() && it->second > 0) {
            ++hits;
            sum += (double)hits / (double)(i + 1);
        }
    }
    return sum / (double)total_rel;
}

double reciprocal_rank(const Ranking& run, const std::unordered_map<std::string, int>& rel) {
    for (std::size_t i = 0; i < run.size(); ++i) {
        auto it = rel.find(run[i]);
        if (it != rel.end() && it->second > 0) return 1.0 / (double)(i + 1);
    }
    return 0.0;
}

// ── Harness ──────────────────────────────────────────────────────────────────
Metrics evaluate(const BeirDataset& ds, const RetrieveFn& retrieve, EvalConfig cfg) {
    Metrics m;
    for (auto k : cfg.cutoffs) { m.ndcg[k] = 0; m.recall[k] = 0; m.precision[k] = 0; }
    std::size_t evaluated = 0;
    for (const auto& q : ds.queries) {
        auto qit = ds.qrels.find(q.id);
        if (qit == ds.qrels.end() || qit->second.empty()) continue;   // no judgments
        Ranking run = retrieve(q.text, cfg.depth);
        const auto& rel = qit->second;
        for (auto k : cfg.cutoffs) {
            m.ndcg[k]      += ndcg_at_k(run, rel, k);
            m.recall[k]    += recall_at_k(run, rel, k);
            m.precision[k] += precision_at_k(run, rel, k);
        }
        m.map += average_precision(run, rel);
        m.mrr += reciprocal_rank(run, rel);
        ++evaluated;
    }
    m.queries = evaluated;
    if (evaluated) {
        for (auto k : cfg.cutoffs) { m.ndcg[k] /= evaluated; m.recall[k] /= evaluated; m.precision[k] /= evaluated; }
        m.map /= evaluated; m.mrr /= evaluated;
    }
    return m;
}

Result<Metrics>
evaluate_corpus(const BeirDataset& ds, index::Corpus& corpus, EvalConfig cfg) {
    // Index the dataset: one doc per BEIR corpus entry, uri = its _id, so the
    // ranking maps chunk → document uri = the BEIR doc id.
    std::unordered_map<std::uint32_t, std::string> doc_uri;  // DocId → beir id
    for (const auto& d : ds.corpus) {
        std::string body = d.title.empty() ? d.text : (d.title + ". " + d.text);
        auto did = corpus.add_document(d.id, std::move(body), {}, d.title);
        if (!did) return std::unexpected(did.error());
        doc_uri[did->get()] = d.id;
    }
    if (auto b = corpus.build(); !b) return std::unexpected(b.error());

    auto retrieve = [&](const std::string& query, std::size_t k) -> Ranking {
        std::vector<Hit> hits;
        if (corpus.has_embedder()) { if (auto d = corpus.dense_search(query, k)) hits = std::move(*d); }
        if (hits.empty()) hits = corpus.lexical_search(query, k);
        Ranking run;
        std::unordered_set<std::string> seen;   // dedupe chunks → their doc id
        for (const auto& h : hits) {
            const Chunk* ch = corpus.chunk(h.chunk);
            if (!ch) continue;
            auto it = doc_uri.find(ch->doc.get());
            if (it == doc_uri.end()) continue;
            if (seen.insert(it->second).second) run.push_back(it->second);
        }
        return run;
    };
    return evaluate(ds, retrieve, cfg);
}

std::string Metrics::report() const {
    std::ostringstream os;
    os << "BEIR evaluation (" << queries << " queries)\n";
    os << "  metric        ";
    for (auto& [k, _] : ndcg) os << "@" << k << "\t";
    os << "\n  nDCG          ";
    for (auto& [k, v] : ndcg) { (void)k; os.precision(4); os << std::fixed << v << "\t"; }
    os << "\n  Recall        ";
    for (auto& [k, v] : recall) { (void)k; os << std::fixed << v << "\t"; }
    os << "\n  Precision     ";
    for (auto& [k, v] : precision) { (void)k; os << std::fixed << v << "\t"; }
    os << "\n  MAP  " << std::fixed << map << "   MRR  " << mrr << "\n";
    return os.str();
}

} // namespace rag::eval
