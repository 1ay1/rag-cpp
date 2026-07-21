#pragma once
// rag/dense/wordpiece.hpp — a minimal WordPiece tokenizer for the ONNX path.
//
// Only compiled into the ONNX build. Loads a HuggingFace-style vocab (either a
// plain `vocab.txt`, one token per line, or the "vocab" object of a
// tokenizer.json) and performs greedy longest-match WordPiece with the standard
// BERT special tokens. Deterministic and dependency-free (json parsing reuses
// nlohmann/json, already a dependency).

#include <cstdint>
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
