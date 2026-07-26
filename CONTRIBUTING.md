# Contributing to rag-cpp

Thanks for your interest. rag-cpp is a C++23 library with a strong emphasis on
**correctness you can measure** and **a type system that makes illegal states
unrepresentable**. A few conventions keep it that way.

## Building and testing

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                 # C++ + C-API + RCP suites
./build/tests/ragcpp_tests             # the main suite, direct
```

Optional backends are off by default and gated behind flags
(`-DRAGCPP_WITH_ONNX=ON`, `-DRAGCPP_WITH_LLAMA=ON`, `-DRAGCPP_WITH_METAL` — on by
default on Apple). The library builds and passes without any of them.

CI runs on macOS (clang, with and without Metal), Linux (gcc), a `-Werror` job,
and a "clean clone" job that verifies the build against the pinned RCP SDK with
no sibling checkout. Please make sure `ctest` is green locally before opening a
PR; the CI matrix will catch platform-specific issues (a real x86-64 `-Werror`
diagnostic and a flaky clang race were both caught this way).

## The bar for a change

- **Measure before you claim.** Any performance or quality claim must be backed
  by a benchmark against the code that *actually ships* — not a strawman. The
  `bench/` targets are self-contained and reproducible; add one if your change
  touches a hot path or a ranking behaviour.
- **Gate every fix with a test, and validate the test by reverting the fix.** A
  test that passes whether or not the bug is present gates nothing. Confirm it
  *fails* against the unpatched code, then keep the fix.
- **Totality.** Fallible operations return `Result<T> = std::expected<T, Error>`;
  do not throw for expected failure. Errors are the closed `Errc` sum type.
- **Strong types.** `DocId`, `ChunkId`, `Score`, etc. are nominally distinct.
  Don't erase them back to primitives to save a line.
- **Comments explain _why_**, and name the dataset/config behind any number.

## Extending the framework

You usually don't need to touch the core. There are three extension axes
(see [`PLUGINS.md`](PLUGINS.md)):

1. **Compile-time** — model a concept (`Embedder`, `Reranker`, …).
2. **Link-time** — type-erase into an `AnyX`.
3. **Load/config-time** — register a backend by name, or ship it as a plugin
   `.so`. Adding an embedder is one `register_embedder(name, description, fn)`
   call (`rag/plugin/builder.hpp`).

## Commit messages

Lead with the measurement or the root cause, record wrong turns and
self-corrections honestly, and end with the test/bench/conformance status. The
existing history is the style guide.

## Reporting bugs

Open an issue with a minimal reproduction, the platform/compiler, and the CMake
flags you configured with. If it's a ranking or recall regression, include the
corpus shape and query so it can be turned into a bench.
