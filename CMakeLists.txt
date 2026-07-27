cmake_minimum_required(VERSION 3.24)

project(rag-cpp
    VERSION 0.1.0
    DESCRIPTION "A type-theoretic, production-grade retrieval (RAG) engine in modern C++"
    LANGUAGES CXX)

# When rag-cpp is pulled in via add_subdirectory() (e.g. as a submodule), the
# consumer owns install/export and usually wants only the library. Detect that
# and (a) default the tests/bench/examples/CLI OFF, (b) skip the install +
# export rules entirely — those reference nlohmann_json, which a subdirectory
# build doesn't put in rag-cpp's export set and CMake rightly rejects.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    set(RAGCPP_IS_TOPLEVEL ${PROJECT_IS_TOP_LEVEL})
else()
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(RAGCPP_IS_TOPLEVEL ON)
    else()
        set(RAGCPP_IS_TOPLEVEL OFF)
    endif()
endif()

option(RAGCPP_BUILD_TESTS    "Build the test suite"         ${RAGCPP_IS_TOPLEVEL})
option(RAGCPP_BUILD_BENCH    "Build the benchmark harness"  ${RAGCPP_IS_TOPLEVEL})
option(RAGCPP_BUILD_EXAMPLES "Build the example programs"   ${RAGCPP_IS_TOPLEVEL})
option(RAGCPP_BUILD_CLI      "Build the ragcpp CLI tool"     ${RAGCPP_IS_TOPLEVEL})
option(RAGCPP_USE_SIMD       "Enable SIMD distance kernels" ON)
option(RAGCPP_WERROR         "Treat warnings as errors"     OFF)
option(RAGCPP_WITH_ONNX      "Build in-process ONNX Runtime embedder" OFF)
option(RAGCPP_WITH_LLAMA     "Build in-process GGUF (llama.cpp) embedder" OFF)
# The pinned RCP SDK uses POSIX socket headers, so it cannot be built by the
# MSYS2/MinGW Windows toolchain yet. Keep the core engine and its Windows CI
# build usable; an explicit request for the front-end fails clearly below.
set(_ragcpp_rcp_default ${RAGCPP_IS_TOPLEVEL})
if(WIN32)
    set(_ragcpp_rcp_default OFF)
endif()
option(RAGCPP_WITH_RCP       "Build the Retrieval Context Protocol (RCP) server front-end" ${_ragcpp_rcp_default})
# GPU acceleration for BATCH scoring (Apple/Metal). Defaults to ON on Apple
# platforms because it needs no SDK, no toolchain component and no download:
# shaders are compiled from source at runtime by the driver. It is also safe to
# leave on — availability is decided at RUNTIME, so a binary built with it still
# runs correctly on a machine with no usable GPU.
if(APPLE)
    option(RAGCPP_WITH_METAL "Build the Metal GPU backend for batch scoring" ON)
else()
    option(RAGCPP_WITH_METAL "Build the Metal GPU backend for batch scoring" OFF)
endif()
set(RAGCPP_RCP_SDK_DIR "" CACHE PATH "Path to a local RCP C++ SDK checkout (sdk/cpp). Empty = FetchContent from GitHub.")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Prefer a system/installed nlohmann_json so the exported package can simply
# find_dependency(nlohmann_json). Only vendor it via FetchContent when absent;
# in that case we add it to our own export set so install(EXPORT) stays valid.
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install ON CACHE INTERNAL "")   # so it can join our export set
    FetchContent_MakeAvailable(nlohmann_json)
    set(RAGCPP_VENDORED_JSON ON)
endif()

file(GLOB_RECURSE RAGCPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Backend translation units are added conditionally by their own blocks below;
# the glob must not pull them in when the corresponding option is off, or the
# build fails on a machine with no OpenCL headers.
list(REMOVE_ITEM RAGCPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/opencl_backend.cpp")
# src/bridge/process.cpp is the subprocess Channel behind open_channel()'s
# `"transport":"process"`. It USED to be dropped from the glob on WIN32 because
# it was POSIX-only (fork/exec, <sys/wait.h>) — but src/bridge/register.cpp calls
# ProcessChannel::spawn() and is always compiled, so removing the TU just moved
# the failure from compile time to link time ('undefined reference to
# rag::bridge::ProcessChannel::spawn'), and only in downstream executables.
# It now carries a CreateProcess backend, so it builds everywhere. Do not
# re-add an exclusion here without also stubbing the transport in register.cpp.

add_library(ragcpp ${RAGCPP_SOURCES})
add_library(ragcpp::ragcpp ALIAS ragcpp)

target_include_directories(ragcpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(ragcpp PUBLIC nlohmann_json::nlohmann_json)

# The plugin loader uses dlopen/dlsym (POSIX) — link the platform's dl library.
target_link_libraries(ragcpp PUBLIC ${CMAKE_DL_LIBS})

# The HTTP transport uses Winsock on Windows. mingw does not honour MSVC's
# `#pragma comment(lib, ...)`, so link ws2_32 explicitly here.
if(WIN32)
    target_link_libraries(ragcpp PUBLIC ws2_32)
endif()

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

# ── Metal GPU backend ─────────────────────────────────────────────────
if(RAGCPP_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "RAGCPP_WITH_METAL=ON but Metal exists only on Apple platforms.")
    endif()
    # Objective-C++ is a CLANG dialect. This project is routinely built with
    # GCC (Homebrew g++-15, because Apple Clang's libc++ still lacks pieces of
    # C++23 that the rest of the library uses), and GCC cannot compile .mm at
    # all. So pin the OBJCXX compiler to the system clang++ regardless of what
    # CXX is — the two object files link together fine, since the .mm exposes
    # only a plain C++ function boundary and no Objective-C types cross it.
    if(NOT CMAKE_OBJCXX_COMPILER)
        find_program(RAGCPP_CLANGXX clang++)
        if(NOT RAGCPP_CLANGXX)
            message(FATAL_ERROR "RAGCPP_WITH_METAL=ON but no clang++ found to compile Objective-C++.")
        endif()
        set(CMAKE_OBJCXX_COMPILER "${RAGCPP_CLANGXX}" CACHE FILEPATH "" FORCE)
    endif()
    enable_language(OBJCXX)
    target_sources(ragcpp PRIVATE src/gpu/metal_backend.mm)
    set_source_files_properties(src/gpu/metal_backend.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc -std=c++20")
    find_library(METAL_LIB Metal REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)
    target_link_libraries(ragcpp PUBLIC ${METAL_LIB} ${FOUNDATION_LIB})
    target_compile_definitions(ragcpp PRIVATE RAGCPP_WITH_METAL=1)
endif()

# ── OpenCL GPU backend (cross-platform) ───────────────────────────────
# Metal is faster on Apple silicon and stays preferred there, but it is
# Apple-only. OpenCL is what makes GPU acceleration available on NVIDIA, AMD and
# Intel: the kernel is compiled at RUNTIME by whichever driver is installed, so
# one binary runs on all of them and no vendor SDK is needed at build time.
#
# AUTO-DETECTED rather than defaulted ON: a machine without OpenCL headers must
# still build, and silently requiring a dependency the user never asked for is
# how a portable library stops being portable. Set RAGCPP_WITH_OPENCL=ON to make
# a missing OpenCL a hard error instead.
if(NOT DEFINED RAGCPP_WITH_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        option(RAGCPP_WITH_OPENCL "Build the OpenCL GPU backend (NVIDIA/AMD/Intel)" ON)
    else()
        option(RAGCPP_WITH_OPENCL "Build the OpenCL GPU backend (NVIDIA/AMD/Intel)" OFF)
    endif()
endif()
if(RAGCPP_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    target_sources(ragcpp PRIVATE src/gpu/opencl_backend.cpp)
    target_link_libraries(ragcpp PUBLIC OpenCL::OpenCL)
    target_compile_definitions(ragcpp PRIVATE RAGCPP_WITH_OPENCL=1)
    message(STATUS "rag-cpp: OpenCL GPU backend enabled")
endif()

if(RAGCPP_USE_SIMD)
    target_compile_definitions(ragcpp PUBLIC RAGCPP_USE_SIMD=1)
endif()

if(NOT MSVC)
    target_compile_options(ragcpp PRIVATE -Wall -Wextra -Wpedantic)
    if(RAGCPP_WERROR)
        target_compile_options(ragcpp PRIVATE -Werror)
    endif()
endif()

# ── RCP: the Retrieval Context Protocol server front-end ─────────────────────
# rag-cpp speaks RCP/1 (github.com/1ay1/rcp) so any RCP client can drive the
# engine over stdio or HTTP. The SDK is header-only C++23. Prefer a local
# checkout (RAGCPP_RCP_SDK_DIR, or the sibling ~/projects/rcp/sdk/cpp) so the
# two repos co-develop; fall back to FetchContent for a standalone build.
set(RAGCPP_RCP_SDK_TAG "v1.0.1" CACHE STRING
    "Git tag of the RCP SDK to fetch when no local checkout is present")
if(RAGCPP_WITH_RCP)
    if(WIN32)
        message(FATAL_ERROR
            "RAGCPP_WITH_RCP=ON is not supported on Windows: the pinned RCP SDK "
            "requires POSIX socket headers (<netinet/in.h>).")
    endif()
    if(RAGCPP_RCP_SDK_DIR AND EXISTS "${RAGCPP_RCP_SDK_DIR}/include/rcp.hpp")
        add_subdirectory(${RAGCPP_RCP_SDK_DIR} ${CMAKE_BINARY_DIR}/rcp-sdk)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../rcp/sdk/cpp/include/rcp.hpp")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rcp/sdk/cpp ${CMAKE_BINARY_DIR}/rcp-sdk)
    else()
        # PINNED to a tag, not `main`.
        #
        # Tracking a branch means the protocol layer of every clean build is
        # whatever the SDK happened to look like that morning — a build that
        # succeeds today can fail, or worse silently change wire behaviour,
        # tomorrow, with nothing in this repository having changed. rag-cpp is
        # the reference RCP implementation, so a reproducible build of it is
        # part of what it demonstrates.
        #
        # v1.0.1 is the first tag containing the C++ SDK fix for a malformed
        # request terminating the server process; v1.0.0 still has that bug, so
        # do NOT lower this pin. Raise it deliberately, when the SDK cuts a
        # release worth taking.
        FetchContent_Declare(rcp
            GIT_REPOSITORY https://github.com/1ay1/rcp.git
            GIT_TAG        ${RAGCPP_RCP_SDK_TAG}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  sdk/cpp)
        FetchContent_MakeAvailable(rcp)
    endif()
    add_library(ragcpp_rcp INTERFACE)
    add_library(ragcpp::rcp ALIAS ragcpp_rcp)
    target_link_libraries(ragcpp_rcp INTERFACE ragcpp rcp::rcp)
    target_compile_definitions(ragcpp_rcp INTERFACE RAGCPP_WITH_RCP=1)
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
include(CMakePackageConfigHelpers)

# Install + export a relocatable CMake package — ONLY for a standalone build.
# A subdirectory consumer already links the ragcpp::ragcpp alias (created next
# to the target above) and owns its own install rules.
if(RAGCPP_IS_TOPLEVEL)
# Install the library and headers, and export a relocatable CMake package so a
# downstream project can simply:
#     find_package(ragcpp CONFIG REQUIRED)
#     target_link_libraries(app PRIVATE ragcpp::ragcpp)
install(TARGETS ragcpp EXPORT ragcppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# An alias so in-tree consumers (add_subdirectory) and installed consumers
# (find_package) both spell the target the same way: ragcpp::ragcpp.
if(NOT TARGET ragcpp::ragcpp)
    add_library(ragcpp::ragcpp ALIAS ragcpp)
endif()

set(RAGCPP_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ragcpp)

install(EXPORT ragcppTargets
    FILE ragcppTargets.cmake
    NAMESPACE ragcpp::
    DESTINATION ${RAGCPP_INSTALL_CMAKEDIR})

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ragcppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ragcppConfig.cmake
    INSTALL_DESTINATION ${RAGCPP_INSTALL_CMAKEDIR})

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ragcppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ragcppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ragcppConfigVersion.cmake
    DESTINATION ${RAGCPP_INSTALL_CMAKEDIR})

# Make the build tree itself find_package-able (no install step needed for
# superbuilds / FetchContent consumers that point CMAKE_PREFIX_PATH here).
export(EXPORT ragcppTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/ragcppTargets.cmake
    NAMESPACE ragcpp::)
endif()  # RAGCPP_IS_TOPLEVEL
