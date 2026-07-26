add_executable(ragcpp_bench bench.cpp)
target_link_libraries(ragcpp_bench PRIVATE ragcpp)

# Recall/QPS curve against standard ANN datasets (SIFT, GloVe). Needs data
# files that are far too large to vendor, so it is never run by the test suite;
# it exists so the recall numbers quoted in HnswConfig can be reproduced.
add_executable(ragcpp_ann_bench ann_bench.cpp)
target_link_libraries(ragcpp_ann_bench PRIVATE ragcpp)
