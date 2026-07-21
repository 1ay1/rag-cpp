// examples/beir_eval.cpp — evaluate the engine on a BEIR dataset.
//
// Usage:
//   ragcpp_beir_eval  <dataset-dir>  [qrels-split]
//
// where <dataset-dir> holds corpus.jsonl, queries.jsonl, and qrels/<split>.tsv
// in the standard BEIR layout (e.g. the NFCorpus download). Prints nDCG@k,
// Recall@k, Precision@k, MAP and MRR for the built-in hybrid/BM25 retriever.
//
// Download a dataset, e.g.:
//   wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
//   unzip nfcorpus.zip
//   ragcpp_beir_eval ./nfcorpus test

#include <cstdio>
#include <string>
#include <rag/rag.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <beir-dataset-dir> [qrels-split=test]\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    std::string split = argc >= 3 ? argv[2] : "test";

    auto ds = rag::eval::BeirDataset::load(dir, split);
    if (!ds) {
        std::printf("load failed: %s\n", ds.error().message.c_str());
        return 1;
    }
    std::printf("Loaded %zu docs, %zu queries, %zu judged queries\n",
                ds->corpus.size(), ds->queries.size(), ds->qrels.size());

    // Lexical (BM25) baseline. Attach an embedder before build() for hybrid:
    //   corpus.set_embedder(rag::dense::AnyEmbedder{ /* your embedder */ });
    rag::index::Corpus corpus;
    rag::eval::EvalConfig cfg;
    cfg.cutoffs = {1, 3, 5, 10, 100};
    cfg.depth   = 100;

    auto m = rag::eval::evaluate_corpus(*ds, corpus, cfg);
    if (!m) { std::printf("eval failed: %s\n", m.error().message.c_str()); return 1; }
    std::printf("\n%s\n", m->report().c_str());
    return 0;
}
