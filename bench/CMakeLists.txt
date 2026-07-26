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
