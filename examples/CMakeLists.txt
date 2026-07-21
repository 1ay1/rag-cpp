add_executable(ragcpp_quickstart quickstart.cpp)
target_link_libraries(ragcpp_quickstart PRIVATE ragcpp)

add_executable(ragcpp_full_pipeline full_pipeline.cpp)
target_link_libraries(ragcpp_full_pipeline PRIVATE ragcpp)

add_executable(ragcpp_graphrag graphrag.cpp)
target_link_libraries(ragcpp_graphrag PRIVATE ragcpp)

add_executable(ragcpp_advanced advanced.cpp)
target_link_libraries(ragcpp_advanced PRIVATE ragcpp)

add_executable(ragcpp_beir_eval beir_eval.cpp)
target_link_libraries(ragcpp_beir_eval PRIVATE ragcpp)
