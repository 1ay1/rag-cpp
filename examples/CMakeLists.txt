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

# ── Out-of-tree plugin demo ──────────────────────────────────────────────────
# A shared-library backend that self-registers on dlopen, and a host that loads
# it and builds its embedder purely by name.
add_library(reverse_hash_plugin SHARED plugin_backend/my_embedder_plugin.cpp)
target_link_libraries(reverse_hash_plugin PRIVATE ragcpp)
set_target_properties(reverse_hash_plugin PROPERTIES PREFIX "lib")

add_executable(ragcpp_plugin_host plugin_backend/host_demo.cpp)
target_link_libraries(ragcpp_plugin_host PRIVATE ragcpp)

# A real LOCAL (in-process) embedder shipped as a plugin: character n-gram
# feature-hashing, no external deps or network. Demonstrates that local
# embedders are just another registry entry, and that config can compose them
# (fallback/retry) declaratively. Host also shows the built-in onnx/gguf names.
add_library(local_ngram_plugin SHARED plugin_backend/local_embedder_plugin.cpp)
target_link_libraries(local_ngram_plugin PRIVATE ragcpp)
set_target_properties(local_ngram_plugin PROPERTIES PREFIX "lib")

add_executable(ragcpp_local_plugin_host plugin_backend/local_host_demo.cpp)
target_link_libraries(ragcpp_local_plugin_host PRIVATE ragcpp)

# ── Polyglot (Python / any-language) backend demo ─────────────────────────
add_executable(ragcpp_python_backend polyglot/python_backend.cpp)
target_link_libraries(ragcpp_python_backend PRIVATE ragcpp)

# ── RCP/1 protocol server front-end ───────────────────────────────────────────
# rag-cpp exposed over the Retrieval Context Protocol (stdio + HTTP). Requires
# the RCP C++ SDK, wired by the top-level CMake when RAGCPP_WITH_RCP is ON.
if(TARGET ragcpp::rcp)
    add_executable(ragcpp_rcp_server rcp_server.cpp)
    target_link_libraries(ragcpp_rcp_server PRIVATE ragcpp::rcp)
endif()
