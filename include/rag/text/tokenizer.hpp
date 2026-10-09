#pragma once
// rag/text/tokenizer.hpp — Unicode-lite tokenization + normalization.
//
// Deterministic, pure, dependency-free. Lowercases ASCII, splits on any
// non-alphanumeric boundary, keeps intra-word ' and digits, and optionally
// drops stopwords and applies the Porter stemmer. Models the `Tokenizer`
// concept.

#include <string>
#include <unordered_map>
#include <string_view>
#include <vector>

namespace rag::text {

// Returns true if `w` is an English stopword (the standard ~180-word list).
[[nodiscard]] bool is_stopword(std::string_view w);

struct TokenizeOptions {
    bool lowercase      = true;
    bool drop_stopwords = true;
    bool stem           = true;
    std::size_t min_len = 2;    // drop 1-char tokens (after stemming)
    std::size_t max_len = 40;   // guard against pathological runs
};

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(TokenizeOptions opts) : opts_(opts) {}

    // A copy takes the options, not the stem memo (it refills on use).
    Tokenizer(const Tokenizer& o) : opts_(o.opts_) {}
    Tokenizer& operator=(const Tokenizer& o) { opts_ = o.opts_; stems_.clear(); return *this; }
    Tokenizer(Tokenizer&&) noexcept            = default;
    Tokenizer& operator=(Tokenizer&&) noexcept = default;

    // Not safe to call on one Tokenizer from two threads at once (it fills
    // its stem memo); copies are independent.
    [[nodiscard]] std::vector<std::string> tokenize(std::string_view text) const;

    [[nodiscard]] const TokenizeOptions& options() const noexcept { return opts_; }

private:
    [[nodiscard]] const std::string& stem_cached(const std::string& word) const;

    TokenizeOptions opts_{};
    // porter_stem memo: a pure function of the word, and word frequency is
    // Zipfian, so a few thousand entries cover most of any text.
    mutable std::unordered_map<std::string, std::string> stems_;
};

// The Porter (1980) stemming algorithm. In-place-ish: returns the stem.
[[nodiscard]] std::string porter_stem(std::string_view word);

} // namespace rag::text
