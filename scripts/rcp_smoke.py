#!/usr/bin/env python3
import subprocess, json, sys

reqs = [
    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"client":{"name":"smoke","version":"0"}}},
    {"jsonrpc":"2.0","id":2,"method":"retrieve","params":{"query":"how does nearest neighbour search work","k":2}},
    {"jsonrpc":"2.0","id":3,"method":"retrieve","params":{"query":"paris landmark","k":3,"filter":{"field":"topic","op":"eq","value":"landmarks"}}},
    {"jsonrpc":"2.0","id":4,"method":"embed","params":{"input":["hello world"],"encoding":"float"}},
    {"jsonrpc":"2.0","id":5,"method":"info","params":{}},
    {"jsonrpc":"2.0","id":6,"method":"ping","params":{}},
    {"jsonrpc":"2.0","id":7,"method":"index/add","params":{"documents":[{"id":"doc://new","text":"A brand new document about vector databases.","meta":{"lang":"en","topic":"retrieval"}}]}},
]
inp = "\n".join(json.dumps(r) for r in reqs) + "\n"
p = subprocess.run([sys.argv[1]], input=inp, capture_output=True, text=True, timeout=20)
for line in p.stdout.splitlines():
    line=line.strip()
    if not line: continue
    o=json.loads(line)
    tag = "ERR "+json.dumps(o["error"]) if "error" in o else "ok "+str(list(o.get("result",{}).keys()))
    print("id", o.get("id"), tag)
    r = o.get("result",{})
    if o.get("id")==2:
        for h in r["hits"]: print("     ", round(h["score"],3), h["id"], "| cite:", h.get("citation"))
    if o.get("id")==3:
        print("      filtered:", [h["id"] for h in r["hits"]])
    if o.get("id")==4:
        print("      dim", r.get("dimension"), "model", r.get("model"))
    if o.get("id")==5:
        caps = r.get("capabilities",{})
        print("      caps:", list(caps.keys()))
if p.stderr.strip():
    print("STDERR:", p.stderr[:500], file=sys.stderr)
