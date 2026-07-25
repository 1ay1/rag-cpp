#pragma once
// rag/store/container.hpp — assemble/parse the sectioned .ragdb container.
//
// Corpus serialization goes through here: it packs named sections into the
// documented layout (format.hpp), writes the CRC trailer, and on load verifies
// magic + major version + CRC before handing sections back by tag. Unknown
// tags round-trip through readers that skip them, so the format is forward-
// compatible.

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "rag/core/types.hpp"
#include "rag/store/format.hpp"

namespace rag::store {

class Container {
public:
    // Add a section payload under a tag. Later duplicates overwrite.
    void put(Tag tag, std::string payload) { sections_[static_cast<std::uint32_t>(tag)] = std::move(payload); }
    void put_raw(std::uint32_t tag, std::string payload) { sections_[tag] = std::move(payload); }

    [[nodiscard]] bool has(Tag tag) const { return sections_.count(static_cast<std::uint32_t>(tag)) != 0; }
    [[nodiscard]] const std::string* get(Tag tag) const {
        auto it = sections_.find(static_cast<std::uint32_t>(tag));
        return it == sections_.end() ? nullptr : &it->second;
    }

    void set_flags(std::uint32_t f) { flags_ = f; }
    [[nodiscard]] std::uint32_t flags() const noexcept { return flags_; }
    [[nodiscard]] std::uint16_t format_major() const noexcept { return major_; }
    [[nodiscard]] std::uint16_t format_minor() const noexcept { return minor_; }

    // Serialize to the on-disk byte layout (header + table + payloads + crc).
    [[nodiscard]] std::string serialize() const;

    // Parse + verify a byte blob. Rejects bad magic, mismatched MAJOR version,
    // or CRC failure with a typed Error.
    [[nodiscard]] static Result<Container> parse(std::string_view blob);

    // Convenience: write to / read from a file path.
    [[nodiscard]] Result<void> write_file(const std::string& path) const;

    // Delete `<path>.tmp.<pid>.*` files whose owning process is gone. Called
    // automatically after a successful write_file; exposed so a host can also
    // reclaim space at startup without performing a save.
    static void sweep_orphan_temps(const std::string& path);
    [[nodiscard]] static Result<Container> read_file(const std::string& path);

private:
    std::map<std::uint32_t, std::string> sections_;
    std::uint32_t flags_ = 0;
    std::uint16_t major_ = kFormatMajor;
    std::uint16_t minor_ = kFormatMinor;
};

} // namespace rag::store
