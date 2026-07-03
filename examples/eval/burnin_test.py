import json, time, urllib.request, random

URL = "http://localhost:8000/v1/chat/completions"

def chat(messages, max_tokens, timeout=560):
    t0 = time.time()
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4", "messages": messages,
                         "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    c = r["choices"][0]
    return c["message"].get("content"), c["finish_reason"], r["usage"], dt

# Phase 1: 10 short requests — must sail past the ~6-request wedge envelope
prompts = [
    "What is 17 * 23? Reply with just the number.",
    "Name the capital of Australia in one word.",
    "Write a haiku about mountains.",
    "What year did the Apollo 11 moon landing happen? Just the year.",
    "Give one synonym for 'quick'.",
    "What is the chemical symbol for gold? Just the symbol.",
    "Complete: 'To be or not to be, that is the ...' (one word)",
    "How many sides does a hexagon have? Just the number.",
    "Name any prime number between 30 and 40.",
    "What language is spoken in Brazil? One word.",
]
for i, p in enumerate(prompts, 1):
    content, fin, u, dt = chat([{"role": "user", "content": p}], 1500)
    print(f"[{i}/10] {dt:5.1f}s fin={fin} tokens={u['completion_tokens']} -> {(content or '')[:60]!r}", flush=True)

print("\nPhase 1 complete — no wedge through 10 requests.\n", flush=True)

# Phase 2: long-context needle test (~25K tokens) — exercises DSA sparse path
random.seed(7)
topics = ["The quarterly report showed steady growth in the {} sector, with analysts noting {} trends across regional markets.",
          "Historical records from the {} archive describe {} patterns in trade routes along the coast.",
          "The engineering team documented {} anomalies in the {} subsystem during routine calibration."]
words = ["maritime","agricultural","semiconductor","logistics","textile","municipal","astronomical","geothermal","pharmaceutical","orchestral"]
lines = [random.choice(topics).format(random.choice(words), random.choice(words)) for _ in range(1200)]
lines.insert(600, "IMPORTANT FACT: The secret access code for the vault is MAGENTA-7742-OTTER.")
haystack = "\n".join(lines)

t0 = time.time()
content, fin, u, dt = chat([{"role": "user", "content": haystack +
    "\n\nWhat is the secret access code for the vault? Reply with just the code."}], 2048)
print(f"[needle] prompt_tokens={u['prompt_tokens']} completion={u['completion_tokens']} "
      f"wall={dt:.1f}s fin={fin}", flush=True)
print("answer:", repr(content), flush=True)
ok = content and "MAGENTA-7742-OTTER" in content
print("NEEDLE FOUND" if ok else "NEEDLE MISSED", flush=True)

# Phase 3: two more short requests after the long one (post-long-context health)
for i in range(2):
    content, fin, u, dt = chat([{"role": "user", "content": "What is 6*7? Just the number."}], 800)
    print(f"[post-{i+1}] {dt:5.1f}s fin={fin} -> {(content or '')[:40]!r}", flush=True)

print("\nBURN-IN PASSED", flush=True)
