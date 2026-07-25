#pragma once
// rag/rcp/error.hpp — the error algebra seam between rag-cpp and RCP.
//
// rag-cpp speaks `rag::Error{Errc, message}` (a small closed enum of domain
// failures); the wire speaks JSON-RPC integer codes (`rcp::errc::*`, spec §12).
// This header is the ONE total mapping between them, so every fallible engine
// call lifts cleanly into a spec-correct wire error with the right retryability
// and structured `data` — and nothing downstream ever hand-writes a code.
//
// The mapping is a total function over Errc (a switch the compiler checks is
// exhaustive), in the house style: no default case that silently swallows a new
// variant, no stringly-typed comparisons.

#include "rag/core/types.hpp"

#include <rcp/types.hpp>

#include <string>
#include <utility>

namespace rag::rcp {

// Lift a rag-cpp domain error onto its canonical RCP wire error (spec §12).
// Retryable/structured-data defaults come from the RCP code (see Error::retryable).
[[nodiscard]] inline ::rcp::Error to_wire(const rag::Error& e) {
    int code = ::rcp::errc::InternalError;
    switch (e.code) {
        case Errc::ok:                 code = ::rcp::errc::InternalError;      break; // ok is never an error
        case Errc::not_found:          code = ::rcp::errc::NotFound;           break;
        case Errc::invalid_argument:   code = ::rcp::errc::InvalidParams;      break;
        case Errc::dimension_mismatch: code = ::rcp::errc::InvalidParams;      break;
        case Errc::parse_error:        code = ::rcp::errc::InvalidParams;      break;
        case Errc::already_exists:     code = ::rcp::errc::Conflict;           break;
        case Errc::unavailable:        code = ::rcp::errc::BackendUnavailable; break;
        case Errc::transport_error:    code = ::rcp::errc::BackendUnavailable; break;
        case Errc::io_error:           code = ::rcp::errc::InternalError;      break;
        case Errc::empty_corpus:       code = ::rcp::errc::InternalError;      break;
        case Errc::corrupt_index:      code = ::rcp::errc::InternalError;      break;
    }
    return ::rcp::Error{code, e.message.empty() ? std::string(to_string(e.code)) : e.message};
}

// Convenience: fold a failed rag Result into a failed rcp Result<Json>.
template <class T>
[[nodiscard]] ::rcp::Result<::rcp::Json> forward_error(const rag::Result<T>& r) {
    return std::unexpected(to_wire(r.error()));
}

// A wire failure straight from an RCP code (for front-end-side validation that
// never touched the engine, e.g. a malformed param the SDK didn't catch).
[[nodiscard]] inline std::unexpected<::rcp::Error>
wire_fail(int code, std::string message, ::rcp::Json data = ::rcp::Json(nullptr)) {
    return std::unexpected(::rcp::Error{code, std::move(message), std::move(data)});
}

} // namespace rag::rcp
