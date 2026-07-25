#pragma once
// rag/index/pq.hpp — Product Quantization for vector compression.
//
// Storing every embedding as float32 costs 4·dim bytes/vector — the dominant
// memory cost of a large index. Product Quantization (Jégou et al. 2011) cuts
// that 4–64×: split each vector into `m` contiguous sub-vectors, k-means each
// subspace into 256 centroids, and store each sub-vector as ONE BYTE (its
// nearest centroid id). A vector becomes `m` bytes.
//
// Scoring uses ADC (Asymmetric Distance Computation): the QUERY stays full
// precision; for each subspace we precompute a 256-entry lookup table of
// (query_sub · centroid), then a compressed vector's similarity is just `m`
// table lookups summed — no decompression, cache-friendly, fast.
//
// This is the compression behind FAISS IVFPQ. Here it's a standalone codec you
// can use to (a) shrink a flat brute-force store, or (b) rescore HNSW survivors
// from compressed codes. Trained once on the corpus; serializable.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::index {

struct PqConfig {
    std::size_t   m          = 8;    // subquantizers (dim must be divisible by m)
    std::size_t   ksub       = 256;  // centroids per subspace (1 byte ⇒ 256)
    std::size_t   iters      = 25;   // k-means iterations
    std::uint64_t seed       = 0x9E3779B97F4A7C15ull;
};

// A trained product quantizer + the compressed codes of the vectors it encoded.
class ProductQuantizer {
public:
    ProductQuantizer() = default;

    // Train the codebook on a sample of unit-normalized vectors, then encode
    // `data` (id-parallel). All vectors must share `dim`, divisible by m.
    [[nodiscard]] static Result<ProductQuantizer>
    train(std::span<const Vector> data, PqConfig cfg = {});

    // Train from a FLAT row-major arena (`n` rows of `dim` floats) without
    // materializing a vector-of-vectors. This is the form every caller inside
    // the library already has — HnswIndex::store_ is exactly this layout — and
    // it avoids copying the whole corpus just to hand it over.
    //
    // The m subspaces are independent k-means problems, so they are solved
    // across all cores.
    [[nodiscard]] static Result<ProductQuantizer>
    train_flat(std::span<const float> data, std::size_t n, std::size_t dim, PqConfig cfg = {});

    // Encode one vector to its m-byte code (nearest centroid per subspace).
    [[nodiscard]] std::vector<std::uint8_t> encode(std::span<const float> v) const;

    // Encode into caller-provided storage (exactly m bytes). The hot form when
    // encoding a whole corpus: no per-vector allocation.
    void encode_into(std::span<const float> v, std::span<std::uint8_t> out) const noexcept;

    // Encode `n` rows of a flat arena into a flat `n*m` code block, in parallel.
    void encode_flat(std::span<const float> data, std::size_t n,
                     std::span<std::uint8_t> out) const;

    [[nodiscard]] std::size_t m() const noexcept { return cfg_.m; }
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
    [[nodiscard]] bool empty() const noexcept { return centroids_.empty(); }

    // Approximate reconstruction from a code (centroid concatenation).
    [[nodiscard]] Vector decode(std::span<const std::uint8_t> code) const;

    // Reconstruct into caller-provided storage, resizing it to dim. Used on the
    // rescore path when the exact vectors have been dropped, where a fresh
    // Vector per candidate would be an allocation per hit.
    void decode_into(std::span<const std::uint8_t> code, std::vector<float>& out) const;

    // Build the ADC lookup table for a query: ksub·m floats of (q_sub·centroid).
    [[nodiscard]] std::vector<float> adc_table(std::span<const float> query) const;

    // Similarity (dot ≈ cosine for unit vectors) of a code against a prebuilt
    // ADC table — the hot path: m lookups summed.
    [[nodiscard]] float adc_score(std::span<const std::uint8_t> code,
                                  std::span<const float> table) const noexcept;

    // Search the encoded set (added via add()) for the top-k by ADC.
    void add(std::uint32_t id, std::span<const float> v);
    [[nodiscard]] std::vector<Hit> search(std::span<const float> query, std::size_t k) const;

    [[nodiscard]] std::size_t dimension()  const noexcept { return dim_; }
    [[nodiscard]] std::size_t subspaces()  const noexcept { return cfg_.m; }
    [[nodiscard]] std::size_t code_count() const noexcept { return codes_.size(); }
    // Bytes/vector after compression vs. float32, for reporting.
    [[nodiscard]] float compression_ratio() const noexcept {
        return dim_ ? (float)(dim_ * sizeof(float)) / (float)cfg_.m : 0.0f;
    }

    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static Result<ProductQuantizer> deserialize(std::string_view blob);

private:
    PqConfig                     cfg_{};
    std::size_t                  dim_  = 0;   // full vector dim
    std::size_t                  dsub_ = 0;   // dim / m
    std::vector<float>           centroids_;  // [m][ksub][dsub] flattened
    std::vector<std::uint32_t>   ids_;        // parallel to codes_
    std::vector<std::uint8_t>    codes_;      // [n][m] flattened

    [[nodiscard]] const float* centroid(std::size_t sub, std::size_t c) const {
        return centroids_.data() + (sub * cfg_.ksub + c) * dsub_;
    }
};

} // namespace rag::index
