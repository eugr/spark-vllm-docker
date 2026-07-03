import json, time, random, sys, urllib.request

URL = "http://localhost:8000/v1/chat/completions"

def chat(prompt, max_tokens, timeout=560, temperature=0.0):
    t0 = time.time()
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    c = r["choices"][0]["message"]
    return c.get("content") or "", r["usage"], dt

random.seed(11)
topics = ["The maritime survey recorded {} conditions across the {} sector.",
          "Committee notes describe {} adjustments to the {} framework.",
          "Field engineers logged {} variance in the {} assembly."]
words = ["nominal","elevated","cyclical","marginal","sustained","irregular",
         "coastal","inland","northern","auxiliary","primary","thermal"]
lines = [random.choice(topics).format(random.choice(words), random.choice(words))
         for _ in range(1150)]

# Three needles at ~10%, ~50%, ~90% depth
needles = {
    "shallow": ("The project codename at the north site is CRIMSON-FALCON-3311.", 115),
    "middle":  ("The vault combination for sector B is 74-19-86-52.", 575),
    "deep":    ("The lead auditor's badge number is Q-88041-ZETA.", 1035),
}
for name, (text, pos) in needles.items():
    lines.insert(pos, text)
doc = "\n".join(lines)

fails = 0

# Needle retrievals (greedy)
tests = [
    ("shallow", "What is the project codename at the north site? Answer with just the codename.", "CRIMSON-FALCON-3311"),
    ("middle",  "What is the vault combination for sector B? Answer with just the numbers.", "74-19-86-52"),
    ("deep",    "What is the lead auditor's badge number? Answer with just the badge number.", "Q-88041-ZETA"),
]
for name, q, expect in tests:
    t0 = time.time()
    ans, u, dt = chat(doc + "\n\n" + q, 1024)
    ok = expect in ans
    fails += 0 if ok else 1
    print(f"[needle-{name:<7}] {'OK  ' if ok else 'FAIL'} prompt={u['prompt_tokens']} wall={dt:.1f}s -> {ans.strip()[:60]!r}", flush=True)

# Factual greedy canaries
canaries = [
    ("What is 127 * 43? Answer with just the number.", "5461"),
    ("What is the 10th Fibonacci number if F(1)=1, F(2)=1? Just the number.", "55"),
    ("Spell 'accuracy' backwards. Just the reversed string.", "ycarucca"),
]
for q, expect in canaries:
    ans, u, dt = chat(q, 2048)
    ok = expect.lower() in ans.lower()
    fails += 0 if ok else 1
    print(f"[canary] {'OK  ' if ok else 'FAIL'} -> {ans.strip()[:50]!r} (want {expect})", flush=True)

print(f"\nACCURACY PROBE: {'PASSED' if fails == 0 else f'{fails} FAILURES'}", flush=True)
sys.exit(1 if fails else 0)
