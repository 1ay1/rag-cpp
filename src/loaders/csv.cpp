// src/loaders/csv.cpp — tabular ingestion: one document per row.
//
// See loaders.hpp for why rows, not token windows, are the right unit here.

#include "rag/loaders/loaders.hpp"

#include <fstream>
#include <sstream>

namespace rag::loaders {
namespace {

// Split one CSV record into fields, honouring RFC 4180 quoting.
//
// `in` is consumed from `pos`; on return `pos` points just past the record's
// terminating newline (or at the end). Returns false when there is nothing left
// to read. A newline INSIDE a quoted field does not end the record, which is why
// this cannot be a simple getline() loop — that bug is the reason most hand
// rolled CSV readers corrupt real-world exports.
bool next_record(std::string_view in, std::size_t& pos, char delim,
                 std::vector<std::string>& out) {
    out.clear();
    if (pos >= in.size()) return false;

    std::string field;
    bool in_quotes = false;
    bool any = false;

    while (pos < in.size()) {
        const char c = in[pos];
        if (in_quotes) {
            if (c == '"') {
                // "" inside a quoted field is a literal quote.
                if (pos + 1 < in.size() && in[pos + 1] == '"') { field.push_back('"'); pos += 2; continue; }
                in_quotes = false; ++pos; continue;
            }
            field.push_back(c); ++pos; continue;
        }
        if (c == '"') { in_quotes = true; any = true; ++pos; continue; }
        if (c == delim) { out.push_back(std::move(field)); field.clear(); any = true; ++pos; continue; }
        if (c == '\r') { ++pos; continue; }          // tolerate CRLF
        if (c == '\n') { ++pos; out.push_back(std::move(field)); return true; }
        field.push_back(c); any = true; ++pos;
    }
    out.push_back(std::move(field));
    return any || !out.empty();
}

std::string trim(std::string_view s) {
    std::size_t a = s.find_first_not_of(" \t");
    if (a == std::string_view::npos) return {};
    std::size_t b = s.find_last_not_of(" \t");
    return std::string(s.substr(a, b - a + 1));
}

} // namespace

Result<std::vector<LoadedDoc>>
load_csv_text(std::string_view csv, const std::string& uri_prefix, const CsvOptions& opts) {
    std::vector<LoadedDoc> out;
    std::size_t pos = 0;
    std::vector<std::string> fields;

    // Header (or synthetic column names col1..colN when has_header is false).
    std::vector<std::string> headers;
    if (opts.has_header) {
        if (!next_record(csv, pos, opts.delimiter, fields))
            return fail<std::vector<LoadedDoc>>(Errc::invalid_argument, "csv: empty input");
        for (auto& f : fields) headers.push_back(trim(f));
    }

    // Which columns feed the searchable text. Resolved to indices once, so the
    // per-row loop is a lookup rather than a string comparison per column.
    auto index_of = [&](const std::string& name) -> std::size_t {
        for (std::size_t i = 0; i < headers.size(); ++i) if (headers[i] == name) return i;
        return static_cast<std::size_t>(-1);
    };

    std::size_t row = 0;
    while (next_record(csv, pos, opts.delimiter, fields)) {
        ++row;
        // A trailing newline yields one empty field; that is not a record.
        if (fields.size() == 1 && fields[0].empty()) continue;

        if (!headers.empty() && fields.size() != headers.size())
            return fail<std::vector<LoadedDoc>>(
                Errc::invalid_argument,
                "csv: row " + std::to_string(row) + " has " + std::to_string(fields.size()) +
                " fields, header has " + std::to_string(headers.size()));

        // Synthesize names on the first data row when there is no header.
        if (headers.empty())
            for (std::size_t i = 0; i < fields.size(); ++i)
                headers.push_back("col" + std::to_string(i + 1));

        // Text: the named columns, else everything. Rendered as "name: value"
        // per column so the COLUMN NAME is searchable too — a query for
        // "status open" should match a row whose status column says open, and
        // without the name in the text only "open" would be indexed.
        std::string text;
        auto append_col = [&](std::size_t i) {
            if (i >= fields.size() || fields[i].empty()) return;
            if (!text.empty()) text += "\n";
            text += headers[i] + ": " + fields[i];
        };
        if (opts.text_columns.empty()) {
            for (std::size_t i = 0; i < fields.size(); ++i) append_col(i);
        } else {
            for (const auto& name : opts.text_columns) {
                std::size_t i = index_of(name);
                if (i != static_cast<std::size_t>(-1)) append_col(i);
            }
        }

        LoadedDoc d;
        d.text = std::move(text);

        if (!opts.title_column.empty()) {
            std::size_t i = index_of(opts.title_column);
            if (i != static_cast<std::size_t>(-1) && i < fields.size()) d.title = fields[i];
        }

        std::string id;
        if (!opts.id_column.empty()) {
            std::size_t i = index_of(opts.id_column);
            if (i != static_cast<std::size_t>(-1) && i < fields.size()) id = fields[i];
        }
        if (id.empty()) id = std::to_string(row);
        d.uri = uri_prefix.empty() ? id : uri_prefix + "#" + id;

        // Every column as filterable metadata: this is the point of ingesting a
        // table as a table rather than as prose.
        if (opts.meta_all_columns)
            for (std::size_t i = 0; i < fields.size() && i < headers.size(); ++i)
                d.meta[headers[i]] = fields[i];
        d.meta["row"] = std::to_string(row);

        out.push_back(std::move(d));
    }
    return out;
}

Result<std::vector<LoadedDoc>>
load_csv(const std::filesystem::path& path, const CsvOptions& opts) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail<std::vector<LoadedDoc>>(Errc::io_error, "cannot open '" + path.string() + "'");
    std::ostringstream ss; ss << f.rdbuf();

    CsvOptions o = opts;
    // A .tsv whose delimiter was left at the default comma parses as one giant
    // column; picking it from the extension makes the common case just work.
    if (o.delimiter == ',') {
        auto ext = path.extension().string();
        if (ext == ".tsv" || ext == ".tab") o.delimiter = '\t';
    }
    return load_csv_text(ss.str(), path.filename().string(), o);
}

} // namespace rag::loaders
