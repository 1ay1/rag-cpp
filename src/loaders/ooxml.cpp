// rag/loaders/ooxml.cpp — ZIP + DEFLATE + OOXML text extraction, self-contained.
//
// The INFLATE implementation is a straightforward canonical-Huffman decoder.
// It is written for clarity and for being obviously bounds-safe on hostile
// input rather than for throughput: office documents are small, and a
// malformed one arriving from a user's corpus must produce an error, never a
// read past the end of the buffer. Every table lookup and every back-reference
// is range-checked, and the output is capped so a zip bomb cannot exhaust
// memory.

#include "rag/loaders/ooxml.hpp"
#include "rag/loaders/loaders.hpp"   // html_to_text, reused for EPUB XHTML parts

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace rag::loaders {

namespace {

// A decompressed office part larger than this is not a document anyone means
// to search; it is a bomb or a corruption. 256 MB is far past any real .docx.
constexpr std::size_t kMaxInflate = 256u * 1024u * 1024u;

std::uint16_t rd16(std::string_view d, std::size_t at) {
    return static_cast<std::uint16_t>(static_cast<unsigned char>(d[at]) |
                                      (static_cast<unsigned char>(d[at + 1]) << 8));
}
std::uint32_t rd32(std::string_view d, std::size_t at) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(d[at])) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d[at + 1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d[at + 2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(d[at + 3])) << 24);
}

// ─── Canonical Huffman decoding ───────────────────────────────────────────────

struct Huffman {
    // counts[len] = number of symbols with that code length; symbols[] holds
    // the symbols ordered by (length, symbol), which is what "canonical" means
    // and what lets decode() walk lengths without storing explicit codes.
    std::array<std::uint16_t, 16> counts{};
    std::vector<std::uint16_t> symbols;

    bool build(const std::uint8_t* lengths, std::size_t n) {
        counts.fill(0);
        for (std::size_t i = 0; i < n; ++i) counts[lengths[i]]++;
        counts[0] = 0;
        // Reject over-subscribed code sets: a corrupt table that would let
        // decode() run off the end of symbols[].
        int left = 1;
        for (std::size_t len = 1; len < 16; ++len) {
            left <<= 1;
            left -= counts[len];
            if (left < 0) return false;
        }
        std::array<std::uint16_t, 16> offs{};
        for (std::size_t len = 1; len < 15; ++len) offs[len + 1] = static_cast<std::uint16_t>(offs[len] + counts[len]);
        symbols.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i)
            if (lengths[i]) symbols[offs[lengths[i]]++] = static_cast<std::uint16_t>(i);
        return true;
    }
};

class BitReader {
public:
    explicit BitReader(std::string_view d) : d_(d) {}

    // Returns -1 on exhaustion rather than throwing: every caller checks, and
    // truncated input is expected (it is what a partially-downloaded file
    // looks like), not exceptional.
    int bit() {
        if (nbits_ == 0) {
            if (pos_ >= d_.size()) { ok_ = false; return -1; }
            cur_ = static_cast<unsigned char>(d_[pos_++]);
            nbits_ = 8;
        }
        int b = cur_ & 1;
        cur_ >>= 1;
        --nbits_;
        return b;
    }

    int bits(int n) {
        int v = 0;
        for (int i = 0; i < n; ++i) {
            int b = bit();
            if (b < 0) return -1;
            v |= b << i;
        }
        return v;
    }

    void align() { nbits_ = 0; }

    int decode(const Huffman& h) {
        int code = 0, first = 0, index = 0;
        for (std::size_t len = 1; len < 16; ++len) {
            int b = bit();
            if (b < 0) return -1;
            code |= b;
            int count = h.counts[len];
            if (code - first < count) {
                std::size_t at = static_cast<std::size_t>(index + (code - first));
                if (at >= h.symbols.size()) return -1;
                return h.symbols[at];
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        return -1;
    }

    [[nodiscard]] bool ok() const noexcept { return ok_; }
    [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
    [[nodiscard]] std::size_t size() const noexcept { return d_.size(); }
    [[nodiscard]] std::string_view data() const noexcept { return d_; }
    void seek(std::size_t p) { pos_ = p; nbits_ = 0; }

private:
    std::string_view d_;
    std::size_t pos_ = 0;
    unsigned cur_ = 0;
    int nbits_ = 0;
    bool ok_ = true;
};

const std::uint16_t kLenBase[29] = {
    3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
const std::uint8_t kLenExtra[29] = {
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
const std::uint16_t kDistBase[30] = {
    1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,
    4097,6145,8193,12289,16385,24577};
const std::uint8_t kDistExtra[30] = {
    0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

} // namespace

Result<std::string> inflate_raw(std::string_view in, std::size_t expected) {
    std::string out;
    out.reserve(std::min<std::size_t>(expected ? expected : in.size() * 4, 1u << 20));
    BitReader br(in);

    Huffman fixed_lit, fixed_dist;
    {
        std::array<std::uint8_t, 288> l{};
        for (std::size_t i = 0; i < 144; ++i) l[i] = 8;
        for (std::size_t i = 144; i < 256; ++i) l[i] = 9;
        for (std::size_t i = 256; i < 280; ++i) l[i] = 7;
        for (std::size_t i = 280; i < 288; ++i) l[i] = 8;
        fixed_lit.build(l.data(), l.size());
        std::array<std::uint8_t, 30> d{};
        d.fill(5);
        fixed_dist.build(d.data(), d.size());
    }

    for (;;) {
        int final_block = br.bit();
        if (final_block < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated");
        int type = br.bits(2);
        if (type < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated");

        if (type == 0) {
            // Stored: length-prefixed raw bytes, byte-aligned.
            br.align();
            std::size_t p = br.pos();
            if (p + 4 > in.size()) return fail<std::string>(Errc::invalid_argument, "deflate: short stored block");
            std::uint16_t len = rd16(in, p);
            p += 4;
            if (p + len > in.size()) return fail<std::string>(Errc::invalid_argument, "deflate: stored overrun");
            if (out.size() + len > kMaxInflate) return fail<std::string>(Errc::invalid_argument, "deflate: output too large");
            out.append(in.data() + p, len);
            br.seek(p + len);
        } else if (type == 1 || type == 2) {
            Huffman dyn_lit, dyn_dist;
            const Huffman* lit = &fixed_lit;
            const Huffman* dist = &fixed_dist;

            if (type == 2) {
                int hlit = br.bits(5), hdist = br.bits(5), hclen = br.bits(4);
                if (hlit < 0 || hdist < 0 || hclen < 0)
                    return fail<std::string>(Errc::invalid_argument, "deflate: truncated header");
                const std::size_t nlit = static_cast<std::size_t>(hlit) + 257;
                const std::size_t ndist = static_cast<std::size_t>(hdist) + 1;
                const std::size_t ncode = static_cast<std::size_t>(hclen) + 4;
                if (nlit > 286 || ndist > 30)
                    return fail<std::string>(Errc::invalid_argument, "deflate: bad table sizes");

                static const std::uint8_t kOrder[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
                std::array<std::uint8_t, 19> clen{};
                for (std::size_t i = 0; i < ncode; ++i) {
                    int v = br.bits(3);
                    if (v < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated clen");
                    clen[kOrder[i]] = static_cast<std::uint8_t>(v);
                }
                Huffman clh;
                if (!clh.build(clen.data(), clen.size()))
                    return fail<std::string>(Errc::invalid_argument, "deflate: bad code-length table");

                std::vector<std::uint8_t> lens(nlit + ndist, 0);
                for (std::size_t i = 0; i < lens.size();) {
                    int sym = br.decode(clh);
                    if (sym < 0) return fail<std::string>(Errc::invalid_argument, "deflate: bad code length");
                    if (sym < 16) { lens[i++] = static_cast<std::uint8_t>(sym); continue; }
                    std::size_t rep = 0;
                    std::uint8_t val = 0;
                    if (sym == 16) {
                        if (i == 0) return fail<std::string>(Errc::invalid_argument, "deflate: repeat at start");
                        val = lens[i - 1];
                        int e = br.bits(2);
                        if (e < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated repeat");
                        rep = 3 + static_cast<std::size_t>(e);
                    } else if (sym == 17) {
                        int e = br.bits(3);
                        if (e < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated repeat");
                        rep = 3 + static_cast<std::size_t>(e);
                    } else {
                        int e = br.bits(7);
                        if (e < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated repeat");
                        rep = 11 + static_cast<std::size_t>(e);
                    }
                    if (i + rep > lens.size())
                        return fail<std::string>(Errc::invalid_argument, "deflate: repeat overruns table");
                    for (std::size_t k = 0; k < rep; ++k) lens[i++] = val;
                }
                if (!dyn_lit.build(lens.data(), nlit) ||
                    !dyn_dist.build(lens.data() + nlit, ndist))
                    return fail<std::string>(Errc::invalid_argument, "deflate: bad huffman table");
                lit = &dyn_lit;
                dist = &dyn_dist;
            }

            for (;;) {
                int sym = br.decode(*lit);
                if (sym < 0) return fail<std::string>(Errc::invalid_argument, "deflate: bad symbol");
                if (sym < 256) {
                    if (out.size() >= kMaxInflate)
                        return fail<std::string>(Errc::invalid_argument, "deflate: output too large");
                    out.push_back(static_cast<char>(sym));
                    continue;
                }
                if (sym == 256) break;               // end of block
                std::size_t li = static_cast<std::size_t>(sym) - 257;
                if (li >= 29) return fail<std::string>(Errc::invalid_argument, "deflate: bad length code");
                int le = br.bits(kLenExtra[li]);
                if (le < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated length");
                const std::size_t length = kLenBase[li] + static_cast<std::size_t>(le);

                int dsym = br.decode(*dist);
                if (dsym < 0 || dsym >= 30)
                    return fail<std::string>(Errc::invalid_argument, "deflate: bad distance code");
                int de = br.bits(kDistExtra[dsym]);
                if (de < 0) return fail<std::string>(Errc::invalid_argument, "deflate: truncated distance");
                const std::size_t distance = kDistBase[dsym] + static_cast<std::size_t>(de);

                // The check that matters: a back-reference before the start of
                // the output is the classic malformed-stream read.
                if (distance > out.size())
                    return fail<std::string>(Errc::invalid_argument, "deflate: distance before start");
                if (out.size() + length > kMaxInflate)
                    return fail<std::string>(Errc::invalid_argument, "deflate: output too large");
                const std::size_t from = out.size() - distance;
                for (std::size_t k = 0; k < length; ++k) out.push_back(out[from + k]);
            }
        } else {
            return fail<std::string>(Errc::invalid_argument, "deflate: reserved block type");
        }
        if (final_block) break;
    }
    return out;
}

// ─── ZIP central directory ────────────────────────────────────────────────────

bool looks_like_zip(std::string_view b) noexcept {
    return b.size() >= 4 && b[0] == 'P' && b[1] == 'K' &&
           static_cast<unsigned char>(b[2]) == 0x03 && static_cast<unsigned char>(b[3]) == 0x04;
}

Result<std::vector<ZipEntry>> zip_entries(std::string_view d) {
    // Find the End Of Central Directory record by scanning backwards. It is at
    // the end unless there is a trailing comment, which is why this is a scan
    // and not a fixed offset.
    if (d.size() < 22) return fail<std::vector<ZipEntry>>(Errc::invalid_argument, "not a zip (too small)");
    std::size_t eocd = std::string_view::npos;
    const std::size_t limit = std::min<std::size_t>(d.size(), 66000);
    for (std::size_t back = 22; back <= limit; ++back) {
        std::size_t at = d.size() - back;
        if (d[at] == 'P' && d[at + 1] == 'K' &&
            static_cast<unsigned char>(d[at + 2]) == 0x05 &&
            static_cast<unsigned char>(d[at + 3]) == 0x06) { eocd = at; break; }
    }
    if (eocd == std::string_view::npos)
        return fail<std::vector<ZipEntry>>(Errc::invalid_argument, "not a zip (no end-of-central-directory)");

    const std::uint16_t count = rd16(d, eocd + 10);
    std::uint32_t cd_off = rd32(d, eocd + 16);
    if (cd_off >= d.size())
        return fail<std::vector<ZipEntry>>(Errc::invalid_argument, "zip: central directory out of range");

    std::vector<ZipEntry> out;
    out.reserve(count);
    std::size_t p = cd_off;
    for (std::uint16_t i = 0; i < count; ++i) {
        if (p + 46 > d.size()) break;
        if (!(d[p] == 'P' && d[p + 1] == 'K' &&
              static_cast<unsigned char>(d[p + 2]) == 0x01 &&
              static_cast<unsigned char>(d[p + 3]) == 0x02)) break;
        ZipEntry e;
        e.method = rd16(d, p + 10);
        e.comp_size = rd32(d, p + 20);
        e.uncomp_size = rd32(d, p + 24);
        const std::uint16_t nlen = rd16(d, p + 28);
        const std::uint16_t elen = rd16(d, p + 30);
        const std::uint16_t clen = rd16(d, p + 32);
        e.offset = rd32(d, p + 42);
        if (p + 46 + nlen > d.size()) break;
        e.name.assign(d.data() + p + 46, nlen);
        out.push_back(std::move(e));
        p += 46u + nlen + elen + clen;
    }
    if (out.empty()) return fail<std::vector<ZipEntry>>(Errc::invalid_argument, "zip: no entries");
    return out;
}

Result<std::string> zip_read(std::string_view d, const ZipEntry& e) {
    // The local header repeats the name and extra-field lengths, and they can
    // differ from the central directory's, so the data offset must be computed
    // from the LOCAL header. Getting this wrong yields plausible-looking
    // garbage rather than an error, which is why it is worth a comment.
    if (e.offset + 30 > d.size()) return fail<std::string>(Errc::invalid_argument, "zip: bad local offset");
    if (!(d[e.offset] == 'P' && d[e.offset + 1] == 'K' &&
          static_cast<unsigned char>(d[e.offset + 2]) == 0x03 &&
          static_cast<unsigned char>(d[e.offset + 3]) == 0x04))
        return fail<std::string>(Errc::invalid_argument, "zip: bad local header");
    const std::uint16_t nlen = rd16(d, e.offset + 26);
    const std::uint16_t elen = rd16(d, e.offset + 28);
    const std::size_t data = e.offset + 30u + nlen + elen;
    if (data + e.comp_size > d.size()) return fail<std::string>(Errc::invalid_argument, "zip: entry overruns file");

    std::string_view raw = d.substr(data, e.comp_size);
    if (e.method == 0) return std::string(raw);
    if (e.method == 8) return inflate_raw(raw, e.uncomp_size);
    return fail<std::string>(Errc::parse_error, "zip: unsupported compression method");
}

Result<std::string> zip_read_name(std::string_view d, std::string_view name) {
    auto entries = zip_entries(d);
    if (!entries) return std::unexpected(entries.error());
    for (const auto& e : *entries)
        if (e.name == name) return zip_read(d, e);
    return fail<std::string>(Errc::not_found, "zip: entry not found: " + std::string(name));
}

// ─── XML → text ───────────────────────────────────────────────────────────────

namespace {

void decode_entity(std::string_view ent, std::string& out) {
    if (ent == "amp") out += '&';
    else if (ent == "lt") out += '<';
    else if (ent == "gt") out += '>';
    else if (ent == "quot") out += '"';
    else if (ent == "apos") out += '\'';
    else if (ent == "nbsp") out += ' ';
    else if (!ent.empty() && ent[0] == '#') {
        // Numeric character reference. Emitted as UTF-8 so that a document
        // written in any script survives extraction intact.
        unsigned long cp = 0;
        if (ent.size() > 2 && (ent[1] == 'x' || ent[1] == 'X'))
            cp = std::strtoul(std::string(ent.substr(2)).c_str(), nullptr, 16);
        else
            cp = std::strtoul(std::string(ent.substr(1)).c_str(), nullptr, 10);
        if (cp == 0 || cp > 0x10FFFF) return;
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
}

// The tag's local name, without namespace prefix: "w:p" -> "p".
std::string_view local_name(std::string_view tag) {
    std::size_t i = 0;
    while (i < tag.size() && tag[i] != ' ' && tag[i] != '/' && tag[i] != '>') ++i;
    std::string_view name = tag.substr(0, i);
    if (auto c = name.find(':'); c != std::string_view::npos) name.remove_prefix(c + 1);
    return name;
}

} // namespace

std::string ooxml_xml_to_text(std::string_view xml) {
    std::string out;
    out.reserve(xml.size() / 4);
    std::size_t i = 0;
    bool skipping = false;                 // inside a part we do not want
    int table_depth = 0;                   // inside <w:tbl>

    auto newline = [&] {
        if (!out.empty() && out.back() != '\n') out.push_back('\n');
    };
    auto tab = [&] {
        if (!out.empty() && out.back() != '\n' && out.back() != '\t') out.push_back('\t');
    };

    while (i < xml.size()) {
        if (xml[i] == '<') {
            std::size_t end = xml.find('>', i);
            if (end == std::string_view::npos) break;
            std::string_view tag = xml.substr(i + 1, end - i - 1);
            const bool closing = !tag.empty() && tag.front() == '/';
            if (closing) tag.remove_prefix(1);
            std::string_view name = local_name(tag);

            // Structure-bearing elements. Paragraph and line breaks become
            // newlines so the chunker still sees document structure. The set
            // spans BOTH OOXML (w:/a: prefixes) and OpenDocument (text:/table:
            // prefixes): local_name() has already stripped the prefix, and the
            // ODF names do not collide with the OOXML ones, so one pass reads
            // .docx and .odt alike.
            //
            // Tables need care: a cell contains its own paragraph, so treating
            // every paragraph as a newline explodes a 2x2 table into four
            // lines and the row — the unit that actually means something,
            // since it is one record — is destroyed. Inside a table a
            // paragraph is therefore a TAB and only a table ROW ends the line,
            // so a row survives as "Metric\tValue", which reads as a record and
            // retrieves as one.
            //
            //   OOXML:  tbl / tr / tc      ODF: table / table-row / table-cell
            if (name == "tbl" || name == "table")
                { table_depth += closing ? -1 : 1; if (table_depth < 0) table_depth = 0; newline(); }
            else if (name == "tr" || name == "table-row") newline();
            else if (name == "p" || name == "br" || name == "sectPr" ||
                     name == "h" || name == "line-break")
                { if (table_depth > 0) tab(); else newline(); }
            else if (name == "tab" || name == "tc" || name == "table-cell")
                tab();
            // Deleted text and field instructions are not document content.
            // OOXML: delText / instrText.  ODF marks deletions inside
            // <text:tracked-changes>; the visible document does not include
            // them, so skip that subtree.
            else if (name == "delText" || name == "instrText" ||
                     name == "tracked-changes")
                skipping = !closing;

            i = end + 1;
            continue;
        }
        // Character data.
        if (xml[i] == '&') {
            std::size_t semi = xml.find(';', i);
            if (semi != std::string_view::npos && semi - i <= 10) {
                if (!skipping) decode_entity(xml.substr(i + 1, semi - i - 1), out);
                i = semi + 1;
                continue;
            }
        }
        if (!skipping) out.push_back(xml[i]);
        ++i;
    }

    // Collapse the runs of blank lines that OOXML's empty paragraphs produce.
    std::string tidy;
    tidy.reserve(out.size());
    int nl = 0;
    for (char c : out) {
        if (c == '\n') {
            if (++nl > 2) continue;
        } else if (c != ' ' && c != '\t') {
            nl = 0;
        }
        tidy.push_back(c);
    }
    while (!tidy.empty() && (tidy.back() == '\n' || tidy.back() == ' ')) tidy.pop_back();
    return tidy;
}

// ─── Per-format extraction ────────────────────────────────────────────────────

Result<std::string> docx_to_text(std::string_view bytes) {
    auto doc = zip_read_name(bytes, "word/document.xml");
    if (!doc) return std::unexpected(doc.error());
    std::string text = ooxml_xml_to_text(*doc);

    // Headers, footnotes and endnotes carry real content (definitions,
    // citations, disclaimers) and cost one extra part read each.
    auto entries = zip_entries(bytes);
    if (entries) {
        for (const auto& e : *entries) {
            if (e.name != "word/footnotes.xml" && e.name != "word/endnotes.xml") continue;
            if (auto part = zip_read(bytes, e)) {
                std::string t = ooxml_xml_to_text(*part);
                if (!t.empty()) { text += "\n\n"; text += t; }
            }
        }
    }
    return text;
}

Result<std::string> xlsx_to_text(std::string_view bytes) {
    auto entries = zip_entries(bytes);
    if (!entries) return std::unexpected(entries.error());

    // Excel interns repeated strings in a shared table and references them by
    // index from the sheets, so extracting a sheet without resolving the table
    // yields a grid of integers. This is the one place where naive tag
    // stripping produces confidently wrong output rather than no output.
    std::vector<std::string> shared;
    for (const auto& e : *entries) {
        if (e.name != "xl/sharedStrings.xml") continue;
        auto part = zip_read(bytes, e);
        if (!part) break;
        std::string_view x = *part;
        std::size_t p = 0;
        while ((p = x.find("<si", p)) != std::string_view::npos) {
            std::size_t close = x.find("</si>", p);
            if (close == std::string_view::npos) break;
            shared.push_back(ooxml_xml_to_text(x.substr(p, close - p)));
            p = close + 5;
        }
        break;
    }

    std::string out;
    for (const auto& e : *entries) {
        if (e.name.rfind("xl/worksheets/sheet", 0) != 0) continue;
        auto part = zip_read(bytes, e);
        if (!part) continue;
        std::string_view x = *part;

        if (!out.empty()) out += "\n";
        out += "# " + e.name.substr(std::string("xl/worksheets/").size()) + "\n";

        // Walk cells in document order, emitting one line per row.
        std::size_t p = 0;
        std::string row;
        while (p < x.size()) {
            std::size_t c = x.find("<c ", p);
            std::size_t r = x.find("<row", p);
            if (r != std::string_view::npos && (c == std::string_view::npos || r < c)) {
                if (!row.empty()) { out += row; out += '\n'; row.clear(); }
                p = r + 4;
                continue;
            }
            if (c == std::string_view::npos) break;
            std::size_t cend = x.find("</c>", c);
            std::size_t selfclose = x.find("/>", c);
            if (cend == std::string_view::npos ||
                (selfclose != std::string_view::npos && selfclose < cend)) { p = c + 3; continue; }
            std::string_view cell = x.substr(c, cend - c);
            const bool is_shared = cell.find("t=\"s\"") != std::string_view::npos;
            std::string v = ooxml_xml_to_text(cell);
            if (is_shared) {
                char* endp = nullptr;
                long idx = std::strtol(v.c_str(), &endp, 10);
                v = (idx >= 0 && static_cast<std::size_t>(idx) < shared.size())
                        ? shared[static_cast<std::size_t>(idx)] : std::string{};
            }
            if (!v.empty()) { if (!row.empty()) row += '\t'; row += v; }
            p = cend + 4;
        }
        if (!row.empty()) { out += row; out += '\n'; }
    }
    if (out.empty()) return fail<std::string>(Errc::parse_error, "xlsx: no worksheets found");
    return out;
}

Result<std::string> pptx_to_text(std::string_view bytes) {
    auto entries = zip_entries(bytes);
    if (!entries) return std::unexpected(entries.error());

    // Slides are numbered parts; sort them so the extracted deck reads in
    // presentation order rather than central-directory order.
    std::vector<const ZipEntry*> slides;
    for (const auto& e : *entries)
        if (e.name.rfind("ppt/slides/slide", 0) == 0 && e.name.size() > 5 &&
            e.name.compare(e.name.size() - 4, 4, ".xml") == 0)
            slides.push_back(&e);
    std::sort(slides.begin(), slides.end(), [](const ZipEntry* a, const ZipEntry* b) {
        auto num = [](const std::string& n) {
            std::size_t i = std::string("ppt/slides/slide").size();
            return std::strtol(n.c_str() + i, nullptr, 10);
        };
        return num(a->name) < num(b->name);
    });

    std::string out;
    int n = 0;
    for (const ZipEntry* e : slides) {
        auto part = zip_read(bytes, *e);
        if (!part) continue;
        std::string t = ooxml_xml_to_text(*part);
        if (t.empty()) continue;
        // A slide number is real context: "what was on slide 12" is a question
        // people actually ask, and the chunk should be able to answer it.
        out += "## Slide " + std::to_string(++n) + "\n";
        out += t;
        out += "\n\n";
    }
    // Speaker notes are often where the substance is.
    for (const auto& e : *entries) {
        if (e.name.rfind("ppt/notesSlides/notesSlide", 0) != 0) continue;
        if (auto part = zip_read(bytes, e)) {
            std::string t = ooxml_xml_to_text(*part);
            if (!t.empty()) { out += "### Notes\n"; out += t; out += "\n\n"; }
        }
    }
    if (out.empty()) return fail<std::string>(Errc::parse_error, "pptx: no slides found");
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

Result<std::string> ooxml_to_text(std::string_view bytes) {
    auto entries = zip_entries(bytes);
    if (!entries) return std::unexpected(entries.error());
    bool word = false, excel = false, ppt = false;
    for (const auto& e : *entries) {
        if (e.name == "word/document.xml") word = true;
        else if (e.name.rfind("xl/worksheets/", 0) == 0) excel = true;
        else if (e.name.rfind("ppt/slides/slide", 0) == 0) ppt = true;
    }
    if (word) return docx_to_text(bytes);
    if (excel) return xlsx_to_text(bytes);
    if (ppt) return pptx_to_text(bytes);
    return fail<std::string>(Errc::parse_error, "zip is not an OOXML document");
}

// ─── OpenDocument ────────────────────────────────────────────────────────
Result<std::string> odf_to_text(std::string_view bytes) {
    // The whole body lives in content.xml for .odt, .ods and .odp alike. The
    // shared xml_to_text understands text:p / text:h / table:* so paragraphs,
    // headings and table rows survive; a spreadsheet's rows come through the
    // same table path as a Word table, i.e. one record per line.
    auto content = zip_read_name(bytes, "content.xml");
    if (!content) return std::unexpected(content.error());
    std::string text = ooxml_xml_to_text(*content);
    if (text.empty())
        return fail<std::string>(Errc::parse_error, "odf: content.xml held no text");
    return text;
}

// ─── EPUB ────────────────────────────────────────────────────────────
namespace {

// Pull an attribute value out of a start tag: attr("idref=\"x\"", "idref") -> x.
std::string attr_value(std::string_view tag, std::string_view name) {
    std::size_t p = 0;
    while ((p = tag.find(name, p)) != std::string_view::npos) {
        std::size_t after = p + name.size();
        // Require the match to be a whole attribute name (preceded by space).
        if (p != 0 && tag[p - 1] != ' ' && tag[p - 1] != '\t') { p = after; continue; }
        while (after < tag.size() && (tag[after] == ' ' || tag[after] == '=')) ++after;
        if (after >= tag.size()) return {};
        char q = tag[after];
        if (q != '"' && q != '\'') { p = after; continue; }
        std::size_t end = tag.find(q, after + 1);
        if (end == std::string_view::npos) return {};
        return std::string(tag.substr(after + 1, end - after - 1));
    }
    return {};
}

// Resolve a spine href relative to the OPF's directory (hrefs are relative to
// the package file, which is usually under OEBPS/ or similar).
std::string join_path(std::string_view base_dir, std::string_view href) {
    if (href.empty()) return {};
    if (href.front() == '/') return std::string(href.substr(1));
    if (base_dir.empty()) return std::string(href);
    std::string out = std::string(base_dir);
    if (out.back() != '/') out.push_back('/');
    out += href;
    // Normalise a single leading "../" against base if present (rare in EPUBs).
    return out;
}

} // namespace

Result<std::string> epub_to_text(std::string_view bytes) {
    auto entries = zip_entries(bytes);
    if (!entries) return std::unexpected(entries.error());

    // Prefer reading order from the OPF package: META-INF/container.xml points
    // to the .opf, whose <spine> lists <itemref idref> in order, each idref
    // resolving through <manifest><item id href> to an XHTML file. When any of
    // that is missing or malformed, fall back to every (x)html part sorted by
    // name, which is close enough for retrieval.
    std::vector<std::string> order;
    std::string opf_path;
    if (auto container = zip_read_name(bytes, "META-INF/container.xml")) {
        std::string_view c = *container;
        std::size_t rf = c.find("<rootfile");
        if (rf != std::string_view::npos) {
            std::size_t end = c.find('>', rf);
            if (end != std::string_view::npos)
                opf_path = attr_value(c.substr(rf, end - rf), "full-path");
        }
    }
    if (!opf_path.empty()) {
        if (auto opf = zip_read_name(bytes, opf_path)) {
            std::string_view o = *opf;
            std::string base_dir;
            if (auto slash = opf_path.rfind('/'); slash != std::string::npos)
                base_dir = opf_path.substr(0, slash);
            // id -> href from the manifest.
            std::unordered_map<std::string, std::string> manifest;
            std::size_t p = 0;
            while ((p = o.find("<item ", p)) != std::string_view::npos) {
                std::size_t end = o.find('>', p);
                if (end == std::string_view::npos) break;
                std::string_view tag = o.substr(p, end - p);
                std::string id = attr_value(tag, "id"), href = attr_value(tag, "href");
                if (!id.empty() && !href.empty()) manifest[id] = href;
                p = end + 1;
            }
            // spine itemrefs give the order.
            std::size_t sp = o.find("<spine");
            std::size_t send = o.find("</spine>", sp == std::string_view::npos ? 0 : sp);
            p = (sp == std::string_view::npos) ? 0 : sp;
            while (p < o.size() && (send == std::string_view::npos || p < send)) {
                std::size_t ir = o.find("<itemref", p);
                if (ir == std::string_view::npos || (send != std::string_view::npos && ir >= send)) break;
                std::size_t end = o.find('>', ir);
                if (end == std::string_view::npos) break;
                std::string idref = attr_value(o.substr(ir, end - ir), "idref");
                if (auto it = manifest.find(idref); it != manifest.end())
                    order.push_back(join_path(base_dir, it->second));
                p = end + 1;
            }
        }
    }
    if (order.empty()) {
        for (const auto& e : *entries) {
            std::string_view n = e.name;
            auto ends = [&](const char* s){ std::string_view suf(s); return n.size() >= suf.size() && n.compare(n.size()-suf.size(), suf.size(), suf) == 0; };
            if (ends(".xhtml") || ends(".html") || ends(".htm")) order.push_back(e.name);
        }
        std::sort(order.begin(), order.end());
    }

    std::string out;
    for (const auto& name : order) {
        auto part = zip_read_name(bytes, name);
        if (!part) continue;
        std::string t = html_to_text(*part);
        if (t.empty()) continue;
        if (!out.empty()) out += "\n\n";
        out += t;
    }
    if (out.empty()) return fail<std::string>(Errc::parse_error, "epub: no readable content documents");
    return out;
}

// ─── One door for every ZIP-container document ──────────────────────────────
Result<std::string> zip_document_to_text(std::string_view bytes) {
    auto entries = zip_entries(bytes);
    if (!entries) return std::unexpected(entries.error());
    bool ooxml = false, odf = false, epub = false, mimetype_epub = false;
    for (const auto& e : *entries) {
        if (e.name == "word/document.xml" || e.name.rfind("xl/worksheets/", 0) == 0 ||
            e.name.rfind("ppt/slides/slide", 0) == 0) ooxml = true;
        else if (e.name == "content.xml") odf = true;
        else if (e.name == "META-INF/container.xml") epub = true;
        else if (e.name == "mimetype") mimetype_epub = true;
    }
    // EPUB and OOXML are the most specific signatures; ODF's content.xml is
    // generic enough that OOXML wins if both somehow appear.
    if (ooxml) return ooxml_to_text(bytes);
    if (epub || mimetype_epub) {
        if (auto r = epub_to_text(bytes)) return r;   // else fall through
    }
    if (odf) return odf_to_text(bytes);
    if (epub || mimetype_epub) return epub_to_text(bytes);
    return fail<std::string>(Errc::parse_error,
        "zip is not a recognised document (not OOXML, OpenDocument, or EPUB)");
}

} // namespace rag::loaders
