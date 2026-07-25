// rag/text/tokenizer.cpp — tokenizer, stopwords, and a full Porter stemmer.

#include "rag/text/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace rag::text {

// ─────────────────────────────────────────────────────────────────────────────
// Stopwords
// ─────────────────────────────────────────────────────────────────────────────
bool is_stopword(std::string_view w) {
    static const std::unordered_set<std::string_view> kStop = {
        "a","about","above","after","again","against","all","am","an","and","any",
        "are","aren't","as","at","be","because","been","before","being","below",
        "between","both","but","by","can","can't","cannot","could","couldn't","did",
        "didn't","do","does","doesn't","doing","don't","down","during","each","few",
        "for","from","further","had","hadn't","has","hasn't","have","haven't","having",
        "he","he'd","he'll","he's","her","here","here's","hers","herself","him",
        "himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into",
        "is","isn't","it","it's","its","itself","let's","me","more","most","mustn't",
        "my","myself","no","nor","not","of","off","on","once","only","or","other",
        "ought","our","ours","ourselves","out","over","own","same","shan't","she",
        "she'd","she'll","she's","should","shouldn't","so","some","such","than","that",
        "that's","the","their","theirs","them","themselves","then","there","there's",
        "these","they","they'd","they'll","they're","they've","this","those","through",
        "to","too","under","until","up","very","was","wasn't","we","we'd","we'll",
        "we're","we've","were","weren't","what","what's","when","when's","where",
        "where's","which","while","who","who's","whom","why","why's","with","won't",
        "would","wouldn't","you","you'd","you'll","you're","you've","your","yours",
        "yourself","yourselves",
    };
    return kStop.contains(w);
}

// ─────────────────────────────────────────────────────────────────────────────
// Porter stemmer (Porter, 1980). Faithful implementation over ASCII lowercase.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

bool is_vowel(const std::string& s, std::size_t i) {
    switch (s[i]) {
        case 'a': case 'e': case 'i': case 'o': case 'u': return true;
        case 'y': return i == 0 ? false : !is_vowel(s, i - 1);
        default:  return false;
    }
}

// measure m: number of VC sequences in [0, j].
int measure(const std::string& s, std::size_t j) {
    int m = 0;
    std::size_t i = 0;
    while (true) {
        if (i > j) return m;
        if (!is_vowel(s, i)) i++;
        else break;
    }
    i++;
    while (true) {
        while (true) {
            if (i > j) return m;
            if (is_vowel(s, i)) i++;
            else break;
        }
        i++;
        m++;
        while (true) {
            if (i > j) return m;
            if (!is_vowel(s, i)) i++;
            else break;
        }
        i++;
    }
}

bool has_vowel(const std::string& s, std::size_t j) {
    for (std::size_t i = 0; i <= j && i < s.size(); ++i)
        if (is_vowel(s, i)) return true;
    return false;
}

bool double_consonant(const std::string& s, std::size_t j) {
    if (j < 1) return false;
    if (s[j] != s[j - 1]) return false;
    return !is_vowel(s, j);
}

// cvc: consonant-vowel-consonant where final consonant isn't w,x,y.
bool cvc(const std::string& s, std::size_t i) {
    if (i < 2) return false;
    if (is_vowel(s, i) || !is_vowel(s, i - 1) || is_vowel(s, i - 2)) return false;
    char c = s[i];
    return c != 'w' && c != 'x' && c != 'y';
}

bool ends_with(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size() &&
           std::string_view(s).substr(s.size() - suf.size()) == suf;
}

// Replace suffix `suf` with `rep` if m(stem) satisfies pred(measure).
template <class Pred>
bool replace(std::string& s, std::string_view suf, std::string_view rep, Pred pred) {
    if (!ends_with(s, suf)) return false;
    std::size_t stem_len = s.size() - suf.size();
    std::string stem = s.substr(0, stem_len);
    int m = stem.empty() ? 0 : measure(stem, stem.size() - 1);
    if (!pred(m)) return false;
    s = stem + std::string(rep);
    return true;
}

auto always = [](int) { return true; };
auto m_gt0  = [](int m) { return m > 0; };
auto m_gt1  = [](int m) { return m > 1; };

} // namespace

std::string porter_stem(std::string_view word) {
    std::string s(word);
    if (s.size() <= 2) return s;

    // Step 1a
    if      (ends_with(s, "sses")) s.resize(s.size() - 2);
    else if (ends_with(s, "ies"))  s.resize(s.size() - 2);
    else if (ends_with(s, "ss"))   { /* keep */ }
    else if (ends_with(s, "s"))    s.resize(s.size() - 1);

    // Step 1b
    bool step1b_extra = false;
    if (ends_with(s, "eed")) {
        std::string stem = s.substr(0, s.size() - 3);
        if (!stem.empty() && measure(stem, stem.size() - 1) > 0) s.resize(s.size() - 1);
    } else if (ends_with(s, "ed")) {
        std::string stem = s.substr(0, s.size() - 2);
        if (has_vowel(stem, stem.empty() ? 0 : stem.size() - 1) && !stem.empty()) {
            s = stem; step1b_extra = true;
        }
    } else if (ends_with(s, "ing")) {
        std::string stem = s.substr(0, s.size() - 3);
        if (has_vowel(stem, stem.empty() ? 0 : stem.size() - 1) && !stem.empty()) {
            s = stem; step1b_extra = true;
        }
    }
    if (step1b_extra) {
        if      (ends_with(s, "at") || ends_with(s, "bl") || ends_with(s, "iz")) s += 'e';
        else if (double_consonant(s, s.size() - 1)) {
            char c = s.back();
            if (c != 'l' && c != 's' && c != 'z') s.resize(s.size() - 1);
        } else if (!s.empty() && measure(s, s.size() - 1) == 1 && cvc(s, s.size() - 1)) {
            s += 'e';
        }
    }

    // Step 1c
    if (ends_with(s, "y")) {
        std::string stem = s.substr(0, s.size() - 1);
        if (has_vowel(stem, stem.empty() ? 0 : stem.size() - 1)) s[s.size() - 1] = 'i';
    }

    // Step 2
    static const std::array<std::pair<const char*, const char*>, 20> step2 = {{
        {"ational","ate"}, {"tional","tion"}, {"enci","ence"}, {"anci","ance"},
        {"izer","ize"}, {"bli","ble"}, {"alli","al"}, {"entli","ent"},
        {"eli","e"}, {"ousli","ous"}, {"ization","ize"}, {"ation","ate"},
        {"ator","ate"}, {"alism","al"}, {"iveness","ive"}, {"fulness","ful"},
        {"ousness","ous"}, {"aliti","al"}, {"iviti","ive"}, {"biliti","ble"},
    }};
    for (auto [suf, rep] : step2) if (replace(s, suf, rep, m_gt0)) break;

    // Step 3
    static const std::array<std::pair<const char*, const char*>, 7> step3 = {{
        {"icate","ic"}, {"ative",""}, {"alize","al"}, {"iciti","ic"},
        {"ical","ic"}, {"ful",""}, {"ness",""},
    }};
    for (auto [suf, rep] : step3) if (replace(s, suf, rep, m_gt0)) break;

    // Step 4
    static const std::array<const char*, 19> step4 = {{
        "al","ance","ence","er","ic","able","ible","ant","ement","ment",
        "ent","ou","ism","ate","iti","ous","ive","ize","tion",
    }};
    for (const char* suf : step4) {
        if (!ends_with(s, suf)) continue;
        std::string_view sv(suf);
        std::string stem = s.substr(0, s.size() - sv.size());
        if (stem.empty()) break;
        // "ion" special-case: only strip after s or t.
        if (sv == "tion") {
            if (measure(stem, stem.size() - 1) > 1) { s = stem + "t"; }
            break;
        }
        if (measure(stem, stem.size() - 1) > 1) s = stem;
        break;
    }

    // Step 5a
    if (ends_with(s, "e")) {
        std::string stem = s.substr(0, s.size() - 1);
        if (!stem.empty()) {
            int m = measure(stem, stem.size() - 1);
            if (m > 1 || (m == 1 && !cvc(stem, stem.size() - 1))) s = stem;
        }
    }
    // Step 5b
    if (measure(s, s.size() - 1) > 1 && double_consonant(s, s.size() - 1) && ends_with(s, "l"))
        s.resize(s.size() - 1);

    (void)always; (void)m_gt1;
    return s;
}

// ──────────────────────────────────────────────────────────────────────
// Tokenizer
// ──────────────────────────────────────────────────────────────────────

namespace {

// Memoized stemming.
//
// porter_stem is a long chain of suffix tests and string rewrites — profiling
// a 20k-document ingest showed it (and the `replace`/`ends_with` helpers it
// calls) as the single dominant cost, above chunking and above BM25 posting
// construction. But it is a PURE function of the word, and word frequency is
// Zipfian: a few thousand distinct tokens cover the overwhelming majority of
// occurrences in real text.
//
// So cache it. Thread-local, so no locking and no sharing between the
// concurrent ingest paths; bounded, so a pathological input (hashes, base64,
// minified JS — text with unbounded distinct "words") cannot grow it without
// limit. On overflow we simply clear and start again rather than implement LRU
// eviction: the access pattern is heavily skewed, so the hot words are back in
// the cache almost immediately, and a clear is O(1) amortized against the
// misses it costs.
const std::string& stem_cached(const std::string& word) {
    static constexpr std::size_t kMaxEntries = 1u << 16;   // ~65k distinct words
    static thread_local std::unordered_map<std::string, std::string> cache;

    auto it = cache.find(word);
    if (it != cache.end()) return it->second;
    if (cache.size() >= kMaxEntries) cache.clear();
    return cache.emplace(word, porter_stem(word)).first->second;
}

} // namespace

std::vector<std::string> Tokenizer::tokenize(std::string_view text) const {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);

    auto flush = [&] {
        if (cur.empty()) return;
        if (cur.size() <= opts_.max_len) {
            if (!(opts_.drop_stopwords && is_stopword(cur))) {
                if (opts_.stem) {
                    const std::string& tok = stem_cached(cur);
                    if (tok.size() >= opts_.min_len) out.push_back(tok);
                } else if (cur.size() >= opts_.min_len) {
                    out.push_back(cur);
                }
            }
        }
        cur.clear();
    };

    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        bool alnum = std::isalnum(c);
        bool intra = (c == '\'');   // keep apostrophes inside words
        if (alnum) {
            cur.push_back(opts_.lowercase ? static_cast<char>(std::tolower(c)) : ch);
        } else if (intra && !cur.empty()) {
            // skip apostrophe; "don't" -> "dont" then stopword/stem handles it
        } else {
            flush();
        }
    }
    flush();
    return out;
}

} // namespace rag::text
