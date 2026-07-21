#!/usr/bin/env python3
"""ragcpp_server.py — reference polyglot backend for rag-cpp.

Speak the rag-cpp bridge protocol so an engine/retriever/reranker written in
PYTHON becomes a first-class rag-cpp backend. Two interchangeable transports:

  * stdio (default): read one compact JSON object per line on stdin, print one
    JSON object per line on stdout. This is what rag::bridge::ProcessChannel
    drives — the host spawns `python3 ragcpp_server.py`.

  * http (`--http PORT`): serve the same methods over POST /rag/<method>, which
    rag::bridge::HttpChannel drives.

Request:  {"method": "embed"|"rerank"|"retrieve"|"graph", "params": {...}}
Reply:    {"ok": true, "result": {...}}   or   {"error": {"message": "..."}}

Replace the toy implementations below with your real model / index / graph.
Everything here is pure-stdlib so it runs anywhere Python 3.8+ does; swap in
sentence-transformers, FAISS, networkx, an LLM, etc.
"""
import sys
import json
import math
import hashlib


# ── Your models / indexes go here ────────────────────────────────────────────
# Toy, deterministic stand-ins so the demo runs with zero dependencies. Replace
# with real ones (the wire contract is all rag-cpp cares about).

EMBED_DIM = 384

def embed_one(text: str):
    """Deterministic hashing embedder. Swap for sentence-transformers:
        return model.encode(text).tolist()
    """
    v = [0.0] * EMBED_DIM
    for tok in text.lower().split():
        h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
        v[h % EMBED_DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# A tiny in-Python "index" so `retrieve` has something to return. Replace with
# FAISS / Elasticsearch / a graph engine.
DOCS = [
    ("d1", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
    ("d2", "Photosynthesis converts light energy into chemical energy in plants."),
    ("d3", "The Great Wall of China stretches thousands of kilometres."),
]
_DOC_VECS = [(uri, text, embed_one(text)) for uri, text in DOCS]

def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# ── Method handlers ──────────────────────────────────────────────────────────

def m_embed(params):
    texts = params.get("texts", [])
    return {"vectors": [embed_one(t) for t in texts]}

def m_rerank(params):
    query = params.get("query", "")
    passages = params.get("passages", [])
    qv = embed_one(query)
    # Toy cross-encoder: cosine of independent embeddings. Swap for a real
    # cross-encoder (e.g. a HF AutoModelForSequenceClassification).
    return {"scores": [_cosine(qv, embed_one(p)) for p in passages]}

def m_retrieve(params):
    query = params.get("query", "")
    k = int(params.get("k", 5))
    qv = embed_one(query)
    scored = sorted(
        ((_cosine(qv, v), uri, text) for uri, text, v in _DOC_VECS),
        reverse=True,
    )
    hits = [{"id": uri, "score": float(s), "text": text} for s, uri, text in scored[:k]]
    return {"hits": hits}

def m_graph(params):
    """GraphRAG-style escape hatch. `op` selects the operation; return anything.
    Wire your networkx / LLM community summaries here.
    """
    op = params.get("op", "")
    if op == "local":
        return m_retrieve(params)
    if op == "global":
        return {"summary": "global graph summary would go here", "communities": 0}
    return {"op": op, "note": "unimplemented in reference server"}

HANDLERS = {
    "embed": m_embed,
    "rerank": m_rerank,
    "retrieve": m_retrieve,
    "graph": m_graph,
}


def dispatch(req):
    method = req.get("method", "")
    params = req.get("params", {}) or {}
    fn = HANDLERS.get(method)
    if fn is None:
        return {"error": {"code": "not_found", "message": f"unknown method '{method}'"}}
    try:
        return {"ok": True, "result": fn(params)}
    except Exception as e:  # totality: never crash the pipe
        return {"error": {"code": "internal", "message": str(e)}}


# ── stdio transport (ProcessChannel) ─────────────────────────────────────────

def run_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            reply = {"error": {"code": "bad_json", "message": str(e)}}
        else:
            reply = dispatch(req)
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()


# ── http transport (HttpChannel) ─────────────────────────────────────────────

def run_http(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            method = self.path.rsplit("/", 1)[-1]
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                reply = {"error": {"code": "bad_json", "message": str(e)}}
            else:
                # Accept both {"method","params"} and a bare params body routed by path.
                if "method" not in body:
                    body = {"method": method, "params": body.get("params", body)}
                reply = dispatch(body)
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"ragcpp_server: listening on http://127.0.0.1:{port}/rag/<method>\n")
    srv.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--http":
        run_http(int(sys.argv[2]))
    else:
        run_stdio()
