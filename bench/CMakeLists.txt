add_executable(ragcpp_bench bench.cpp)
target_link_libraries(ragcpp_bench PRIVATE ragcpp)

# Recall/QPS curve against standard ANN datasets (SIFT, GloVe). Needs data
# files that are far too large to vendor, so it is never run by the test suite;
# it exists so the recall numbers quoted in HnswConfig can be reproduced.
add_executable(ragcpp_ann_bench ann_bench.cpp)
target_link_libraries(ragcpp_ann_bench PRIVATE ragcpp)

# GPU vs CPU batch-scoring throughput. Runs on any machine: without a GPU every
# row reports "declined" and the CPU timings still print.
add_executable(ragcpp_gpu_bench gpu_bench.cpp)
target_link_libraries(ragcpp_gpu_bench PRIVATE ragcpp)

# What Contextual Retrieval costs at ingest vs what it buys at retrieval.
# Self-contained (synthesizes its own corpus), so the numbers quoted in
# CorpusConfig::contextual can be reproduced on any machine.
add_executable(ragcpp_contextual_bench contextual_bench.cpp)
target_link_libraries(ragcpp_contextual_bench PRIVATE ragcpp)

# dense_search_batch vs the per-query loop, through the real Corpus API. Also
# checks the two paths return IDENTICAL rankings, which is how the tied-score
# ordering bug was found. Runs anywhere; without a GPU it reports ~1.0x.
add_executable(ragcpp_batch_bench batch_bench.cpp)
target_link_libraries(ragcpp_batch_bench PRIVATE ragcpp)

# What ParentStitch (small-to-big) folds away and what coverage it buys.
# Self-contained (synthesizes its own fragmented corpus), so the numbers quoted
# in Pipeline::context()'s justification can be reproduced on any machine.
add_executable(ragcpp_stitch_bench stitch_bench.cpp)
target_link_libraries(ragcpp_stitch_bench PRIVATE ragcpp)

# BEIR evaluation of the REAL production path (Engine/Pipeline), not the
# lexical-only shortcut eval::evaluate_corpus takes. A/Bs ranking policies over
# one corpus so a retrieval-quality change is a measured diff, not a claim.
add_executable(ragcpp_beir_bench beir_bench.cpp)
target_link_libraries(ragcpp_beir_bench PRIVATE ragcpp)

# Can we chunk a language nobody has a parser for? Measures definition
# integrity on five real languages (control) and five INVENTED ones that exist
# nowhere on the internet. Self-contained: synthesizes its own corpora, so the
# numbers quoted in structure.hpp can be reproduced on any machine.
add_executable(ragcpp_structure_bench structure_bench.cpp)
target_link_libraries(ragcpp_structure_bench PRIVATE ragcpp)

# Does structure-aware chunking RETRIEVE better on real code, or only keep
# definitions intact? Reconstructs real files from CodeSearchNet, chunks them
# two ways over one embedder, and measures MRR/Recall of retrieving the
# function a docstring documents. Needs an ONNX model (retrieval needs an
# embedder); the corpus + manifest are produced offline, so unlike
# structure_bench this one is NOT self-contained.
add_executable(ragcpp_code_retrieval_bench code_retrieval_bench.cpp)
target_link_libraries(ragcpp_code_retrieval_bench PRIVATE ragcpp)
