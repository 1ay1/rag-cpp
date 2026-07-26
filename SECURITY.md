# Security Policy

## Supported versions

rag-cpp is pre-1.0; security fixes land on `main` and are included in the next
tagged release. The latest release is the supported one.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1.0 | ❌        |

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** rather than opening a
public issue:

- Use GitHub's [private vulnerability reporting](https://github.com/1ay1/rag-cpp/security/advisories/new)
  ("Report a vulnerability" under the repository's **Security** tab), or
- open a minimal, non-public channel with the maintainer.

Include a description, a reproduction (or PoC), the affected version/commit, and
the platform. You can expect an acknowledgement within a few days.

## Scope

rag-cpp is a library and a local/served retrieval engine. Areas of particular
interest:

- **Deserialization** — the `.ragdb` container is CRC-checked and rejects unknown
  major versions, but parsing untrusted files is inherently sensitive.
- **The RCP server** (`ragcpp serve`) — it parses untrusted JSON-RPC over stdio
  or HTTP. Filter trees are validated against advertised fields by the RCP SDK
  before evaluation.
- **Network embedder/reranker backends** — they issue outbound requests to
  configured endpoints; treat backend URLs and API keys as sensitive input.

## Good to know

- No expected-failure path throws; errors are typed `Result<T>` values, which
  makes fuzzing the parsers tractable.
- Optional native backends (ONNX, llama.cpp) are off by default and pull in
  third-party code only when explicitly enabled at build time.
