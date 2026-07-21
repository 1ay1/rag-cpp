// rag/dense/local_embedder.cpp — in-process ONNX + GGUF embedders.
//
// The bodies are compiled in two flavours selected by RAGCPP_WITH_ONNX /
// RAGCPP_WITH_LLAMA. Without the macro, load() returns Errc::unavailable and
// the class is an inert stub — the core stays dependency-free. With the macro,
// the real onnxruntime / llama.cpp calls are compiled in.

#include "rag/dense/local_embedder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>

#include "rag/dense/simd.hpp"

#ifdef RAGCPP_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#include "rag/dense/wordpiece.hpp"   // provided only in the ONNX build
#endif
#ifdef RAGCPP_WITH_LLAMA
#include <llama.h>
#endif

namespace rag::dense {

// ═════════════════════════════════════════════════════════════════════════════
// OnnxEmbedder
// ═════════════════════════════════════════════════════════════════════════════
struct OnnxEmbedder::Impl {
    LocalEmbedderConfig cfg;
    std::size_t         dim = 0;
    std::string         id;
#ifdef RAGCPP_WITH_ONNX
    Ort::Env           env{ORT_LOGGING_LEVEL_WARNING, "ragcpp"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    WordPieceTokenizer  tok;
    bool                has_token_type = false;
#endif
};

OnnxEmbedder::OnnxEmbedder() : impl_(std::make_shared<Impl>()) {}
OnnxEmbedder::OnnxEmbedder(OnnxEmbedder&&) noexcept = default;
OnnxEmbedder& OnnxEmbedder::operator=(OnnxEmbedder&&) noexcept = default;
OnnxEmbedder::~OnnxEmbedder() = default;

std::size_t      OnnxEmbedder::dimension() const { return impl_->dim; }
std::string_view OnnxEmbedder::identity()  const { return impl_->id; }

Result<OnnxEmbedder> OnnxEmbedder::load(LocalEmbedderConfig cfg) {
#ifndef RAGCPP_WITH_ONNX
    (void)cfg;
    return std::unexpected(Error{Errc::unavailable,
        "ragcpp built without ONNX support (configure -DRAGCPP_WITH_ONNX=ON)"});
#else
    OnnxEmbedder e;
    auto& im = *e.impl_;
    im.cfg = std::move(cfg);
    im.id  = im.cfg.identity_tag.empty() ? "onnx" : im.cfg.identity_tag;
    int threads = im.cfg.threads > 0 ? im.cfg.threads
                : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    im.opts.SetIntraOpNumThreads(threads);
    im.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        auto tk = WordPieceTokenizer::load(im.cfg.tokenizer_path);
        if (!tk) return std::unexpected(tk.error());
        im.tok = std::move(*tk);
#ifdef _WIN32
        std::wstring wpath(im.cfg.model_path.begin(), im.cfg.model_path.end());
        im.session = std::make_unique<Ort::Session>(im.env, wpath.c_str(), im.opts);
#else
        im.session = std::make_unique<Ort::Session>(im.env, im.cfg.model_path.c_str(), im.opts);
#endif
        // token_type_ids is present in BERT-family models but not all.
        im.has_token_type = im.session->GetInputCount() >= 3;
        // Probe output dim by embedding a trivial input.
        std::array<std::string, 1> probe{"probe"};
        auto v = e.embed(probe);
        if (!v) return std::unexpected(v.error());
        im.dim = (*v)[0].size();
    } catch (const std::exception& ex) {
        return std::unexpected(Error{Errc::corrupt_index,
            std::string("onnx load failed: ") + ex.what()});
    }
    return e;
#endif
}

Result<std::vector<Vector>>
OnnxEmbedder::embed(std::span<const std::string> texts) const {
#ifndef RAGCPP_WITH_ONNX
    (void)texts;
    return std::unexpected(Error{Errc::unavailable, "onnx embedder unavailable"});
#else
    auto& im = *impl_;
    std::vector<Vector> out;
    out.reserve(texts.size());
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Input/output names (owned copies).
    std::vector<std::string> in_names, out_names;
    for (std::size_t i = 0; i < im.session->GetInputCount(); ++i)
        in_names.push_back(im.session->GetInputNameAllocated(i, alloc).get());
    for (std::size_t i = 0; i < im.session->GetOutputCount(); ++i)
        out_names.push_back(im.session->GetOutputNameAllocated(i, alloc).get());
    std::vector<const char*> in_c, out_c;
    for (auto& s : in_names)  in_c.push_back(s.c_str());
    for (auto& s : out_names) out_c.push_back(s.c_str());

    try {
        for (const auto& text : texts) {
            auto enc = im.tok.encode(text, im.cfg.max_tokens);   // ids + mask
            const std::int64_t L = static_cast<std::int64_t>(enc.ids.size());
            std::array<std::int64_t, 2> shape{1, L};
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                mem, enc.ids.data(), enc.ids.size(), shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                mem, enc.mask.data(), enc.mask.size(), shape.data(), 2));
            std::vector<std::int64_t> ttids;
            if (im.has_token_type) {
                ttids.assign(enc.ids.size(), 0);
                inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
                    mem, ttids.data(), ttids.size(), shape.data(), 2));
            }
            auto res = im.session->Run(Ort::RunOptions{nullptr},
                in_c.data(), inputs.data(), inputs.size(), out_c.data(), 1);
            // res[0] = last_hidden_state [1, L, H]
            auto info = res[0].GetTensorTypeAndShapeInfo();
            auto dims = info.GetShape();
            const std::int64_t H = dims.back();
            const float* h = res[0].GetTensorData<float>();
            std::vector<float> pooled(static_cast<std::size_t>(H), 0.0f);
            if (im.cfg.pooling == Pooling::cls) {
                for (std::int64_t d = 0; d < H; ++d) pooled[d] = h[d];
            } else if (im.cfg.pooling == Pooling::max) {
                for (std::size_t d = 0; d < pooled.size(); ++d) pooled[d] = -1e30f;
                for (std::int64_t t = 0; t < L; ++t)
                    if (enc.mask[t])
                        for (std::int64_t d = 0; d < H; ++d)
                            pooled[d] = std::max(pooled[d], h[t * H + d]);
            } else {  // mean pooling over masked tokens
                std::int64_t n = 0;
                for (std::int64_t t = 0; t < L; ++t) {
                    if (!enc.mask[t]) continue;
                    ++n;
                    for (std::int64_t d = 0; d < H; ++d) pooled[d] += h[t * H + d];
                }
                if (n > 0) for (auto& x : pooled) x /= static_cast<float>(n);
            }
            if (im.cfg.normalize) normalize(pooled);
            out.push_back(std::move(pooled));
        }
    } catch (const std::exception& ex) {
        return std::unexpected(Error{Errc::transport_error,
            std::string("onnx run failed: ") + ex.what()});
    }
    return out;
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
// GgufEmbedder  (llama.cpp)
// ═════════════════════════════════════════════════════════════════════════════
struct GgufEmbedder::Impl {
    LocalEmbedderConfig cfg;
    std::size_t         dim = 0;
    std::string         id;
#ifdef RAGCPP_WITH_LLAMA
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    ~Impl() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }
#endif
};

GgufEmbedder::GgufEmbedder() : impl_(std::make_shared<Impl>()) {}
GgufEmbedder::GgufEmbedder(GgufEmbedder&&) noexcept = default;
GgufEmbedder& GgufEmbedder::operator=(GgufEmbedder&&) noexcept = default;
GgufEmbedder::~GgufEmbedder() = default;

std::size_t      GgufEmbedder::dimension() const { return impl_->dim; }
std::string_view GgufEmbedder::identity()  const { return impl_->id; }

Result<GgufEmbedder> GgufEmbedder::load(LocalEmbedderConfig cfg) {
#ifndef RAGCPP_WITH_LLAMA
    (void)cfg;
    return std::unexpected(Error{Errc::unavailable,
        "ragcpp built without GGUF/llama.cpp support (configure -DRAGCPP_WITH_LLAMA=ON)"});
#else
    GgufEmbedder e;
    auto& im = *e.impl_;
    im.cfg = std::move(cfg);
    im.id  = im.cfg.identity_tag.empty() ? "gguf" : im.cfg.identity_tag;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    im.model = llama_model_load_from_file(im.cfg.model_path.c_str(), mp);
    if (!im.model)
        return std::unexpected(Error{Errc::corrupt_index, "gguf model load failed"});
    llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.n_ubatch = cp.n_batch = 2048;
    cp.n_threads = im.cfg.threads > 0 ? im.cfg.threads
                 : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    im.ctx = llama_init_from_model(im.model, cp);
    if (!im.ctx)
        return std::unexpected(Error{Errc::unavailable, "gguf context init failed"});
    im.dim = static_cast<std::size_t>(llama_model_n_embd(im.model));
    return e;
#endif
}

Result<std::vector<Vector>>
GgufEmbedder::embed(std::span<const std::string> texts) const {
#ifndef RAGCPP_WITH_LLAMA
    (void)texts;
    return std::unexpected(Error{Errc::unavailable, "gguf embedder unavailable"});
#else
    auto& im = *impl_;
    const llama_vocab* vocab = llama_model_get_vocab(im.model);
    std::vector<Vector> out;
    out.reserve(texts.size());
    for (const auto& text : texts) {
        int n_max = static_cast<int>(text.size()) + 2;
        std::vector<llama_token> toks(n_max);
        int n = llama_tokenize(vocab, text.c_str(), (int)text.size(),
                               toks.data(), n_max, /*add_special=*/true, /*parse_special=*/false);
        if (n < 0) { n = -n; toks.resize(n); llama_tokenize(vocab, text.c_str(), (int)text.size(),
                               toks.data(), n, true, false); }
        toks.resize(std::min<std::size_t>(n, im.cfg.max_tokens));

        llama_memory_clear(llama_get_memory(im.ctx), true);
        llama_batch batch = llama_batch_init((int)toks.size(), 0, 1);
        for (int i = 0; i < (int)toks.size(); ++i) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
            ++batch.n_tokens;
        }
        if (llama_decode(im.ctx, batch) != 0) {
            llama_batch_free(batch);
            return std::unexpected(Error{Errc::transport_error, "gguf decode failed"});
        }
        const float* emb = llama_get_embeddings_seq(im.ctx, 0);
        if (!emb) emb = llama_get_embeddings(im.ctx);
        std::vector<float> v(emb, emb + im.dim);
        llama_batch_free(batch);
        if (im.cfg.normalize) normalize(v);
        out.push_back(std::move(v));
    }
    return out;
#endif
}

} // namespace rag::dense
