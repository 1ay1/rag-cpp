#!/usr/bin/env python3
"""ragcpp.py — minimal ctypes bindings for rag-cpp's stable C ABI.

Build the shared library first:
    cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build
Then point RAGCPP_LIB at build/libragcpp.{so,dylib} (or install it).

This is a thin, illustrative wrapper — not a packaged module — showing that the
C ABI (include/rag/c/rag.h) is sufficient to drive the whole engine from Python.
"""
import ctypes as C
import os

_lib_path = os.environ.get("RAGCPP_LIB", "build/libragcpp.dylib")
_lib = C.CDLL(_lib_path)

# ── prototypes ────────────────────────────────────────────────────────────────
_lib.rag_version.restype = C.c_char_p
_lib.rag_last_error.restype = C.c_char_p
_lib.rag_engine_new.restype = C.c_void_p
_lib.rag_embedder_hash.restype = C.c_void_p
_lib.rag_embedder_hash.argtypes = [C.c_size_t]
_lib.rag_engine_set_embedder.argtypes = [C.c_void_p, C.c_void_p]
_lib.rag_engine_add.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p,
                                C.POINTER(C.c_char_p), C.POINTER(C.c_char_p),
                                C.c_size_t, C.POINTER(C.c_uint32)]
_lib.rag_engine_build.argtypes = [C.c_void_p]
_lib.rag_engine_search.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t,
                                   C.POINTER(C.c_char_p), C.POINTER(C.c_char_p),
                                   C.c_size_t, C.POINTER(C.c_void_p)]
_lib.rag_results_count.argtypes = [C.c_void_p]; _lib.rag_results_count.restype = C.c_size_t
_lib.rag_results_score.argtypes = [C.c_void_p, C.c_size_t]; _lib.rag_results_score.restype = C.c_float
_lib.rag_results_uri.argtypes = [C.c_void_p, C.c_size_t]; _lib.rag_results_uri.restype = C.c_char_p
_lib.rag_results_text.argtypes = [C.c_void_p, C.c_size_t]; _lib.rag_results_text.restype = C.c_char_p


class Engine:
    def __init__(self, dim=256):
        self.h = _lib.rag_engine_new()
        _lib.rag_engine_set_embedder(self.h, _lib.rag_embedder_hash(dim))

    def add(self, uri, text, title=""):
        _lib.rag_engine_add(self.h, uri.encode(), text.encode(), title.encode(),
                            None, None, 0, None)

    def build(self):
        _lib.rag_engine_build(self.h)

    def search(self, query, k=5):
        res = C.c_void_p()
        _lib.rag_engine_search(self.h, query.encode(), k, None, None, 0, C.byref(res))
        out = []
        for i in range(_lib.rag_results_count(res)):
            out.append({
                "uri": _lib.rag_results_uri(res, i).decode(),
                "score": _lib.rag_results_score(res, i),
                "text": _lib.rag_results_text(res, i).decode(),
            })
        _lib.rag_results_free(res)
        return out


if __name__ == "__main__":
    print("linked rag-cpp", _lib.rag_version().decode())
    e = Engine()
    e.add("intro.md", "rag-cpp fuses BM25 with dense vector search using RRF.")
    e.add("hnsw.md", "HNSW gives logarithmic nearest-neighbour search over embeddings.")
    e.add("cooking.md", "To make risotto, toast the rice before adding stock.")
    e.build()
    for r in e.search("how does approximate nearest neighbour search work", 2):
        print(f"  [{r['score']:.3f}] {r['uri']}: {r['text'][:60]}...")
