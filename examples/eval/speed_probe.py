import json, time, urllib.request

URL = "http://localhost:8000/v1/chat/completions"

def run(max_tokens=400):
    t0 = time.time()
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4",
            "messages": [{"role": "user", "content": "Write a detailed multi-paragraph explanation of how tides work."}],
            "max_tokens": max_tokens, "temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    u = r["usage"]
    return u["completion_tokens"], dt

rates = []
for i in range(3):
    toks, dt = run()
    rate = toks / dt
    rates.append(rate)
    print(f"run {i+1}: {toks} tokens in {dt:.1f}s -> {rate:.2f} tok/s", flush=True)
print(f"DECODE RATE: {sum(rates)/len(rates):.2f} tok/s (avg of 3)", flush=True)
