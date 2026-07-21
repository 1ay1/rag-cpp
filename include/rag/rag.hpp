#pragma once
// rag/rag.hpp — umbrella header. Include this to get the whole library.
//
//   #include <rag/rag.hpp>
//   rag::Engine engine;
//   engine.with_embedder(rag::dense::AnyEmbedder{rag::dense::HashEmbedder{256}});
//   engine.add("doc1", "the text ...");
//   engine.build();
//   auto results = engine.search("query", 5);

#include "rag/core/concepts.hpp"
#include "rag/core/document.hpp"
#include "rag/core/types.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/dense/simd.hpp"
#include "rag/engine.hpp"
#include "rag/fusion/fuse.hpp"
#include "rag/graph/graph.hpp"
#include "rag/index/corpus.hpp"
#include "rag/index/hnsw.hpp"
#include "rag/lexical/bm25.hpp"
#include "rag/loaders/loaders.hpp"
#include "rag/loaders/code_chunker.hpp"
#include "rag/pipeline/pipeline.hpp"
#include "rag/ralm/ralm.hpp"
#include "rag/rerank/reranker.hpp"
#include "rag/store/container.hpp"
#include "rag/store/format.hpp"
#include "rag/text/chunker.hpp"
#include "rag/text/tokenizer.hpp"
