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
#include "rag/bridge/bridge.hpp"
#include "rag/cache/cache.hpp"
#include "rag/cascade/cascade.hpp"
#include "rag/crag/crag.hpp"
#include "rag/dense/backends.hpp"
#include "rag/dense/embedder.hpp"
#include "rag/dense/local_embedder.hpp"
#include "rag/dense/simd.hpp"
#include "rag/engine.hpp"
#include "rag/eval/beir.hpp"
#include "rag/fusion/fuse.hpp"
#include "rag/graph/graph.hpp"
#include "rag/index/corpus.hpp"
#include "rag/index/hnsw.hpp"
#include "rag/index/pq.hpp"
#include "rag/late/colbert.hpp"
#include "rag/lexical/bm25.hpp"
#include "rag/loaders/loaders.hpp"
#include "rag/loaders/code_chunker.hpp"
#include "rag/pipeline/pipeline.hpp"
#include "rag/plugin/plugin.hpp"
#include "rag/query/hyde.hpp"
#include "rag/ralm/ralm.hpp"
#include "rag/raptor/raptor.hpp"
#include "rag/rerank/reranker.hpp"
#include "rag/rerank/mmr.hpp"
#include "rag/sparse/splade.hpp"
#include "rag/store/container.hpp"
#include "rag/store/format.hpp"
#include "rag/text/chunker.hpp"
#include "rag/text/contextual.hpp"
#include "rag/text/semantic_chunker.hpp"
#include "rag/text/tokenizer.hpp"
