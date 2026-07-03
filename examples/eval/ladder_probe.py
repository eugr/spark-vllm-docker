import json, time, urllib.request, sys

URL = "http://localhost:8000/v1/chat/completions"
SENT = "The archive records describe seasonal patterns in coastal trade routes. "

def probe(n_tokens, timeout=200):
    reps = max(1, n_tokens // 16)
    prompt = SENT * reps + "\nReply with the single word 'ok'."
    t0 = time.time()
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4",
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 60}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        dt = time.time() - t0
        u = r["usage"]
        print(f"PASS prompt={u['prompt_tokens']:>5} wall={dt:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"WEDGE at ~{n_tokens} requested tokens: {type(e).__name__} after {time.time()-t0:.0f}s", flush=True)
        return False

for n in [50, 80, 120, 250, 1000, 1500]:
    if not probe(n):
        sys.exit(1)
print("ALL LADDER STEPS PASSED (?!)", flush=True)
