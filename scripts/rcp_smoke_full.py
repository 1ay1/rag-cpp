#!/usr/bin/env python3
import subprocess, json, sys
reqs = [
 {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"client":{"name":"s","version":"0"}}},
 {"jsonrpc":"2.0","id":2,"method":"retrieve","params":{"query":"nearest neighbour search","k":2,"rerank":{"topN":4}}},
 {"jsonrpc":"2.0","id":3,"method":"retrieve","params":{"query":"paris tower","k":2,"mode":"sparse"}},
 {"jsonrpc":"2.0","id":4,"method":"retrieve","params":{"query":"plants energy","k":2,"rewrite":"multi-query"}},
 {"jsonrpc":"2.0","id":5,"method":"rerank","params":{"query":"vector search","documents":["HNSW graph index for vectors","risotto rice recipe"]}},
 {"jsonrpc":"2.0","id":6,"method":"query/transform","params":{"query":"how does rrf work","method":"hyde"}},
 {"jsonrpc":"2.0","id":7,"method":"embed/sparse","params":{"texts":["nearest neighbour search"],"kind":"query"}},
 {"jsonrpc":"2.0","id":8,"method":"embed/multi","params":{"inputs":["vector search"]}},
 {"jsonrpc":"2.0","id":9,"method":"graph","params":{"query":"retrieval","op":"local","k":2}},
 {"jsonrpc":"2.0","id":10,"method":"memory/build","params":{}},
 {"jsonrpc":"2.0","id":11,"method":"memory/recall","params":{"query":"retrieval","n":3}},
 {"jsonrpc":"2.0","id":12,"method":"info","params":{}},
]
inp="\n".join(json.dumps(r) for r in reqs)+"\n"
p=subprocess.run([sys.argv[1]],input=inp,capture_output=True,text=True,timeout=25)
for line in p.stdout.splitlines():
    line=line.strip()
    if not line: continue
    o=json.loads(line)
    i=o.get("id"); r=o.get("result",{})
    if "error" in o: print(f"id {i} ERR {o['error']['code']} {o['error']['message']}"); continue
    if i==2: print("id 2 retrieve+rerank:", [round(h['score'],2) for h in r['hits']], "reranked=",r['usage'].get('reranked'))
    elif i==3: print("id 3 sparse mode:", [h['id'] for h in r['hits']], "mode=",r['usage']['mode'])
    elif i==4: print("id 4 rewrite:", [h['id'] for h in r['hits']], "mode=",r['usage']['mode'])
    elif i==5: print("id 5 rerank results:", [(x['index'],round(x['score'],2)) for x in r['results']])
    elif i==6: print("id 6 transform queries:", r['queries'])
    elif i==7: print("id 7 sparse dims:", len(r['sparse'][0]['indices']))
    elif i==8: print("id 8 multi matrix rows x dim:", len(r['matrices'][0]),"x",r['dimension'])
    elif i==9: print("id 9 graph hits:", [h['id'] for h in r['hits']])
    elif i==10: print("id 10 memory/build:", r)
    elif i==11: print("id 11 recall clues:", len(r['clues']),"clue types")
    elif i==12: print("id 12 CAPS:", sorted(r['capabilities'].keys()))
if p.stderr.strip(): print("STDERR:",p.stderr[:400],file=sys.stderr)
