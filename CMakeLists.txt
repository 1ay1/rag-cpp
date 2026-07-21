cmake_minimum_required(VERSION 3.24)

project(rag-cpp
    VERSION 0.1.0
    DESCRIPTION "A type-theoretic, production-grade retrieval (RAG) engine in modern C++"
    LANGUAGES CXX)

option(RAGCPP_BUILD_TESTS    "Build the test suite"         ON)
option(RAGCPP_BUILD_BENCH    "Build the benchmark harness"  ON)
option(RAGCPP_BUILD_EXAMPLES "Build the example programs"   ON)
option(RAGCPP_USE_SIMD       "Enable SIMD distance kernels" ON)
option(RAGCPP_WERROR         "Treat warnings as errors"     OFF)

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

include(GNUInstallDirs)
install(TARGETS ragcpp EXPORT ragcppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
