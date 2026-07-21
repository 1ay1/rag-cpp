#pragma once
// rag/cache/cache.hpp — embedding + query result caches.
//
// Two hot-path caches with LRU eviction, both keyed to survive across calls:
//
//   • EmbeddingCache — memoizes text → vector, keyed on (embedder identity,
//     text). Re-embedding the same text (repeated queries, overlapping chunks,
//     PRF expansions) is the single most wasteful cost in a RAG loop; this makes
//     it free after the first call. The identity() in the key guarantees a model
//     swap never returns a stale vector.
//   • QueryCache — memoizes (query, k) → ranked hits, so a repeated query short-
//     circuits the whole pipeline. Invalidate on corpus mutation.
//
// Both are thread-safe (a single mutex; the workloads are read-heavy and the
// critical sections tiny) so they can sit in front of a concurrent query path.

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rag/core/types.hpp"

namespace rag::cache {

// A generic bounded LRU map (string key → value). Move-only values supported.
template <class V>
class LruCache {
public:
    explicit LruCache(std::size_t capacity = 4096) : cap_(capacity ? capacity : 1) {}

    [[nodiscard]] std::optional<V> get(const std::string& key) {
        std::lock_guard lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) { ++misses_; return std::nullopt; }
        order_.splice(order_.begin(), order_, it->second.pos);   // touch (MRU)
        ++hits_;
        return it->second.value;
    }

    void put(const std::string& key, V value) {
        std::lock_guard lk(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second.value = std::move(value);
            order_.splice(order_.begin(), order_, it->second.pos);
            return;
        }
        order_.push_front(key);
        map_.emplace(key, Entry{std::move(value), order_.begin()});
        if (map_.size() > cap_) {
            auto last = order_.end(); --last;
            map_.erase(*last);
            order_.pop_back();
        }
    }

    void clear() { std::lock_guard lk(mu_); map_.clear(); order_.clear(); }
    [[nodiscard]] std::size_t size() const { std::lock_guard lk(mu_); return map_.size(); }
    [[nodiscard]] std::size_t hits()   const { std::lock_guard lk(mu_); return hits_; }
    [[nodiscard]] std::size_t misses() const { std::lock_guard lk(mu_); return misses_; }
    [[nodiscard]] double hit_rate() const {
        std::lock_guard lk(mu_);
        std::size_t t = hits_ + misses_;
        return t ? (double)hits_ / (double)t : 0.0;
    }

private:
    struct Entry { V value; typename std::list<std::string>::iterator pos; };
    mutable std::mutex mu_;
    std::size_t cap_;
    std::list<std::string> order_;                       // MRU front → LRU back
    std::unordered_map<std::string, Entry> map_;
    std::size_t hits_ = 0, misses_ = 0;
};

// Embedding cache: key = identity ⊕ '\0' ⊕ text.
class EmbeddingCache {
public:
    explicit EmbeddingCache(std::size_t capacity = 16384) : lru_(capacity) {}
    [[nodiscard]] std::optional<Vector> get(std::string_view identity, std::string_view text) {
        return lru_.get(key(identity, text));
    }
    void put(std::string_view identity, std::string_view text, Vector v) {
        lru_.put(key(identity, text), std::move(v));
    }
    [[nodiscard]] double hit_rate() const { return lru_.hit_rate(); }
    void clear() { lru_.clear(); }
private:
    static std::string key(std::string_view id, std::string_view text) {
        std::string k; k.reserve(id.size() + text.size() + 1);
        k.append(id); k.push_back('\0'); k.append(text);
        return k;
    }
    LruCache<Vector> lru_;
};

// Query result cache: key = query ⊕ '\0' ⊕ k.
class QueryCache {
public:
    explicit QueryCache(std::size_t capacity = 4096) : lru_(capacity) {}
    [[nodiscard]] std::optional<std::vector<Hit>> get(std::string_view query, std::size_t k) {
        return lru_.get(key(query, k));
    }
    void put(std::string_view query, std::size_t k, std::vector<Hit> hits) {
        lru_.put(key(query, k), std::move(hits));
    }
    void clear() { lru_.clear(); }   // call on any corpus mutation
    [[nodiscard]] double hit_rate() const { return lru_.hit_rate(); }
private:
    static std::string key(std::string_view q, std::size_t k) {
        std::string s(q); s.push_back('\0'); s.append(std::to_string(k));
        return s;
    }
    LruCache<std::vector<Hit>> lru_;
};

} // namespace rag::cache
