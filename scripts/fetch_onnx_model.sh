#!/usr/bin/env bash
# scripts/fetch_onnx_model.sh — download a tiny sentence-transformer ONNX model
# to exercise the in-process OnnxEmbedder + its integration test.
#
# Downloads all-MiniLM-L6-v2 (22M params, 384-dim) exported to ONNX from the
# HuggingFace hub, plus its WordPiece vocab. Then build with -DRAGCPP_WITH_ONNX=ON
# and point the test at it:
#
#   scripts/fetch_onnx_model.sh ./models
#   cmake -B build -DRAGCPP_WITH_ONNX=ON -Donnxruntime_ROOT=/path/to/onnxruntime
#   cmake --build build -j
#   RAGCPP_TEST_ONNX_MODEL=./models/model.onnx \
#   RAGCPP_TEST_ONNX_TOKENIZER=./models/vocab.txt \
#   ctest --test-dir build --output-on-failure
set -euo pipefail
OUT="${1:-./models}"
mkdir -p "$OUT"
BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
echo "Fetching all-MiniLM-L6-v2 ONNX + vocab into $OUT ..."
curl -fL "$BASE/onnx/model.onnx" -o "$OUT/model.onnx"
curl -fL "$BASE/vocab.txt"        -o "$OUT/vocab.txt"
echo "Done. model=$OUT/model.onnx  tokenizer=$OUT/vocab.txt"
