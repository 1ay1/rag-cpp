#pragma once
// rag/dense/wordpiece.hpp — a minimal WordPiece tokenizer for the ONNX path.
//
// Only compiled into the ONNX build. Loads a HuggingFace-style vocab (either a
// plain `vocab.txt`, one token per line, or the "vocab" object of a
// tokenizer.json) and performs greedy longest-match WordPiece with the standard
// BERT special tokens. Deterministic and dependency-free (json parsing reuses
// nlohmann/json, already a dependency).

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "rag/core/types.hpp"

namespace rag::dense {

struct Encoded {
    std::vector<std::int64_t> ids;
    std::vector<std::int64_t> mask;
    // Segment ids: 0 for the first sequence, 1 for the second. For a single
    // sequence every entry is 0; a cross-encoder pair uses 0 over
    // `[CLS] query [SEP]` and 1 over `passage [SEP]`. BERT-family models read
    // these as `token_type_ids`; models without that input simply ignore them.
    std::vector<std::int64_t> type_ids;
};

class WordPieceTokenizer {
public:
    WordPieceTokenizer() = default;

    static Result<WordPieceTokenizer> load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return std::unexpected(Error{Errc::not_found, "tokenizer file not found: " + path});
        WordPieceTokenizer t;
        if (path.size() > 5 && path.substr(path.size() - 5) == ".json") {
            nlohmann::json j;
            try { f >> j; } catch (...) { return std::unexpected(Error{Errc::corrupt_index, "bad tokenizer.json"}); }
            const auto& vocab = j.contains("model") && j["model"].contains("vocab")
                              ? j["model"]["vocab"] : j["vocab"];
            for (auto it = vocab.begin(); it != vocab.end(); ++it)
                t.vocab_[it.key()] = it.value().get<std::int64_t>();
        } else {
            std::string line; std::int64_t idx = 0;
            while (std::getline(f, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                t.vocab_[line] = idx++;
            }
        }
        auto id = [&](const char* s, std::int64_t def) {
            auto it = t.vocab_.find(s); return it == t.vocab_.end() ? def : it->second;
        };
        t.cls_ = id("[CLS]", 101); t.sep_ = id("[SEP]", 102);
        t.unk_ = id("[UNK]", 100); t.pad_ = id("[PAD]", 0);
        if (t.vocab_.empty()) return std::unexpected(Error{Errc::corrupt_index, "empty vocab"});
        return t;
    }

    // Greedy WordPiece over whitespace/punct-split, lowercased basic tokens.
    Encoded encode(std::string_view text, std::size_t max_tokens) const {
        Encoded e;
        e.ids.push_back(cls_);
        for (auto& word : basic_split(text)) {
            wordpiece(word, e.ids);
            if (e.ids.size() + 1 >= max_tokens) break;
        }
        if (e.ids.size() + 1 > max_tokens) e.ids.resize(max_tokens - 1);
        e.ids.push_back(sep_);
        e.mask.assign(e.ids.size(), 1);
        e.type_ids.assign(e.ids.size(), 0);
        return e;
    }

    // Cross-encoder pair encoding: `[CLS] a [SEP] b [SEP]`, with token_type_ids
    // 0 over the `[CLS] a [SEP]` span and 1 over the `b [SEP]` span. This is the
    // exact sequence a HuggingFace `AutoModelForSequenceClassification` reranker
    // (mono-BERT, bge-reranker, ms-marco cross-encoders) was trained on, so the
    // relevance logit it emits is only meaningful for input shaped this way.
    //
    // The budget is split so the passage never crowds out the query: the query
    // is capped at a third of the window (queries are short; truncating them
    // loses the actual information need), the passage takes the rest. Both share
    // the greedy WordPiece path used by `encode`, so behaviour matches.
    Encoded encode_pair(std::string_view a, std::string_view b, std::size_t max_tokens) const {
        // Reserve room for [CLS] + two [SEP].
        const std::size_t budget = max_tokens > 3 ? max_tokens - 3 : 1;
        const std::size_t a_budget = std::max<std::size_t>(1, budget / 3);

        std::vector<std::int64_t> a_ids, b_ids;
        for (auto& word : basic_split(a)) {
            wordpiece(word, a_ids);
            if (a_ids.size() >= a_budget) { a_ids.resize(a_budget); break; }
        }
        const std::size_t b_budget = budget > a_ids.size() ? budget - a_ids.size() : 1;
        for (auto& word : basic_split(b)) {
            wordpiece(word, b_ids);
            if (b_ids.size() >= b_budget) { b_ids.resize(b_budget); break; }
        }

        Encoded e;
        e.ids.push_back(cls_);
        e.ids.insert(e.ids.end(), a_ids.begin(), a_ids.end());
        e.ids.push_back(sep_);
        const std::size_t seg0 = e.ids.size();          // [CLS] a [SEP]
        e.ids.insert(e.ids.end(), b_ids.begin(), b_ids.end());
        e.ids.push_back(sep_);

        e.mask.assign(e.ids.size(), 1);
        e.type_ids.assign(e.ids.size(), 0);
        for (std::size_t i = seg0; i < e.type_ids.size(); ++i) e.type_ids[i] = 1;
        return e;
    }

private:
    static std::vector<std::string> basic_split(std::string_view s) {
        std::vector<std::string> out; std::string cur;
        auto flush = [&]{ if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
        for (unsigned char c : s) {
            if (std::isspace(c)) { flush(); }
            else if (std::ispunct(c)) { flush(); out.emplace_back(1, (char)c); }
            else cur.push_back((char)std::tolower(c));
        }
        flush();
        return out;
    }

    void wordpiece(const std::string& word, std::vector<std::int64_t>& ids) const {
        std::size_t start = 0; bool bad = false;
        std::vector<std::int64_t> sub;
        while (start < word.size()) {
            std::size_t end = word.size(); std::int64_t cur = -1;
            while (start < end) {
                std::string piece = (start > 0 ? "##" : "") + word.substr(start, end - start);
                if (auto it = vocab_.find(piece); it != vocab_.end()) { cur = it->second; break; }
                --end;
            }
            if (cur < 0) { bad = true; break; }
            sub.push_back(cur); start = end;
        }
        if (bad) ids.push_back(unk_);
        else ids.insert(ids.end(), sub.begin(), sub.end());
    }

    std::unordered_map<std::string, std::int64_t> vocab_;
    std::int64_t cls_ = 101, sep_ = 102, unk_ = 100, pad_ = 0;
};

} // namespace rag::dense
