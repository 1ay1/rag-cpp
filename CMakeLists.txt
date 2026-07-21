cmake_minimum_required(VERSION 3.24)

project(rag-cpp
    VERSION 0.1.0
    DESCRIPTION "A type-theoretic, production-grade retrieval (RAG) engine in modern C++"
    LANGUAGES CXX)

option(RAGCPP_BUILD_TESTS    "Build the test suite"         ON)
option(RAGCPP_BUILD_BENCH    "Build the benchmark harness"  ON)
option(RAGCPP_BUILD_EXAMPLES "Build the example programs"   ON)
option(RAGCPP_BUILD_CLI      "Build the ragcpp CLI tool"     ON)
option(RAGCPP_USE_SIMD       "Enable SIMD distance kernels" ON)
option(RAGCPP_WERROR         "Treat warnings as errors"     OFF)
option(RAGCPP_WITH_ONNX      "Build in-process ONNX Runtime embedder" OFF)
option(RAGCPP_WITH_LLAMA     "Build in-process GGUF (llama.cpp) embedder" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

file(GLOB_RECURSE RAGCPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(ragcpp ${RAGCPP_SOURCES})
add_library(ragcpp::ragcpp ALIAS ragcpp)

target_include_directories(ragcpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(ragcpp PUBLIC nlohmann_json::nlohmann_json)

# ── Optional in-process embedder backends ────────────────────────────────────
if(RAGCPP_WITH_ONNX)
    # Point CMake at your ONNX Runtime install via -Donnxruntime_ROOT=... .
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        target_link_libraries(ragcpp PUBLIC onnxruntime::onnxruntime)
    else()
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATH_SUFFIXES onnxruntime onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIB onnxruntime)
        if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
            message(FATAL_ERROR "RAGCPP_WITH_ONNX=ON but ONNX Runtime not found. "
                "Set onnxruntime_ROOT or ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIB.")
        endif()
        target_include_directories(ragcpp PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(ragcpp PUBLIC ${ONNXRUNTIME_LIB})
    endif()
    target_compile_definitions(ragcpp PUBLIC RAGCPP_WITH_ONNX)
endif()

if(RAGCPP_WITH_LLAMA)
    # Provide llama.cpp via find_package or add_subdirectory before this.
    find_package(llama QUIET)
    if(NOT llama_FOUND)
        message(FATAL_ERROR "RAGCPP_WITH_LLAMA=ON but llama.cpp not found. "
            "Install llama.cpp with CMake config, or add_subdirectory it first.")
    endif()
    target_link_libraries(ragcpp PUBLIC llama)
    target_compile_definitions(ragcpp PUBLIC RAGCPP_WITH_LLAMA)
endif()
target_compile_features(ragcpp PUBLIC cxx_std_23)

if(RAGCPP_USE_SIMD)
    target_compile_definitions(ragcpp PUBLIC RAGCPP_USE_SIMD=1)
endif()

if(NOT MSVC)
    target_compile_options(ragcpp PRIVATE -Wall -Wextra -Wpedantic)
    if(RAGCPP_WERROR)
        target_compile_options(ragcpp PRIVATE -Werror)
    endif()
endif()

if(RAGCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(RAGCPP_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(RAGCPP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(RAGCPP_BUILD_CLI)
    add_subdirectory(cli)
endif()

include(GNUInstallDirs)
install(TARGETS ragcpp EXPORT ragcppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
