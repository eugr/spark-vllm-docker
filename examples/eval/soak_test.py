import json, time, random, sys, threading, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
random.seed(42)
FILLER = ["The archive records describe seasonal trade patterns.",
          "Engineers documented calibration anomalies in the subsystem.",
          "Quarterly reports showed steady regional growth.",
          "The observatory logged unusual atmospheric readings.",
          "Council minutes recorded the infrastructure proposal."]

def make_prompt(n_tokens):
    s = []
    while sum(len(x.split()) for x in s) < n_tokens * 0.75:
        s.append(random.choice(FILLER))
    return " ".join(s) + f" [marker {random.randint(1000,9999)}] Summarize in one sentence."

def chat(prompt, max_tokens, timeout=300):
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["usage"]["completion_tokens"], time.time() - t0

count, t_start = 0, time.time()
def log(tag, toks, dt):
    global count
    count += 1
    print(f"[{count:3}] {time.time()-t_start:6.0f}s {tag:<12} {toks:4} tok {dt:5.1f}s", flush=True)

try:
    # Phase A: 20 sequential mixed-size
    for i in range(20):
        n = random.choice([20, 50, 150, 400, 800])
        toks, dt = chat(make_prompt(n), random.choice([100, 300, 600]))
        log(f"seq-{n}", toks, dt)

    # Phase B: 5 rounds of 4-way concurrent bursts
    for rnd in range(5):
        results, errs = [], []
        def worker(n):
            try:
                toks, dt = chat(make_prompt(n), 300)
                results.append((toks, dt))
            except Exception as e:
                errs.append(e)
        threads = [threading.Thread(target=worker, args=(random.choice([30, 200, 1000, 3000]),)) for _ in range(4)]
        [t.start() for t in threads]
        [t.join(timeout=400) for t in threads]
        if errs or len(results) < 4:
            print(f"CONCURRENT ROUND {rnd+1} FAILED: {len(results)}/4 ok, errs={errs[:1]}", flush=True)
            sys.exit(1)
        for toks, dt in results:
            log(f"conc-r{rnd+1}", toks, dt)

    # Phase C: 5 long-context
    for n in [5000, 8000, 12000, 16000, 20000]:
        toks, dt = chat(make_prompt(n), 200, timeout=500)
        log(f"long-{n}", toks, dt)

    # Phase D: 15 more sequential mixed
    for i in range(15):
        n = random.choice([20, 100, 500, 1500, 4000])
        toks, dt = chat(make_prompt(n), random.choice([100, 400]))
        log(f"seq2-{n}", toks, dt)

    print(f"\nSOAK PASSED: {count} requests, {(time.time()-t_start)/60:.1f} minutes, zero failures", flush=True)
except Exception as e:
    print(f"\nSOAK FAILED at request {count+1}: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
