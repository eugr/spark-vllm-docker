import json, time, random, urllib.request

URL = "http://localhost:8000/v1/chat/completions"

def chat(messages, max_tokens, timeout=1500, temperature=0.0):
    t0 = time.time()
    req = urllib.request.Request(URL,
        data=json.dumps({"model": "GLM-5.2-NVFP4", "messages": messages,
                         "max_tokens": max_tokens, "temperature": temperature}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    c = r["choices"][0]["message"]
    return (c.get("content") or ""), r["usage"], dt

# ~320K-token document: filler logs with 5 story-facts planted at
# 10/30/50/70/90% depth, all about one fictional expedition.
random.seed(77)
words = ["survey","ledger","archive","manifest","registry","bulletin"]
lines = [f"Log {i}: routine {random.choice(words)} entry, sector {random.randint(1,999)}, no anomalies." for i in range(19400)]
facts = [
    (1940,  "EXPEDITION FACT 1: The expedition ship is named the Peregrine Star."),
    (5820,  "EXPEDITION FACT 2: Captain Ilse Moreau leads the expedition."),
    (9700,  "EXPEDITION FACT 3: The destination is the Vetlanda Trench."),
    (13580, "EXPEDITION FACT 4: The mission carries exactly 47 crew members."),
    (17460, "EXPEDITION FACT 5: The expedition departs on March 9th."),
]
for pos, text in facts:
    lines.insert(pos, text)
doc = "\n".join(lines)

# 1. Prefill timing + decode-speed measurement (long generation at depth)
msgs = [{"role": "user", "content": doc + "\n\nWrite a one-paragraph mission briefing for this expedition, using every EXPEDITION FACT in the log. Then add a short risk assessment."}]
out, u, dt = chat(msgs, 600)
print(f"[gen-at-depth] prompt={u['prompt_tokens']} completion={u['completion_tokens']} wall={dt:.0f}s", flush=True)

# 2. Follow-up on cached prefix: pure decode speed at 320K context
msgs.append({"role": "assistant", "content": out})
msgs.append({"role": "user", "content": "Now rewrite the briefing as exactly ten numbered bullet points."})
out2, u2, dt2 = chat(msgs, 500)
gen_rate = u2["completion_tokens"] / dt2
print(f"[decode-at-depth] completion={u2['completion_tokens']} wall={dt2:.1f}s -> {gen_rate:.1f} tok/s (prefix cached)", flush=True)

# 3. Coherence scoring: are all 5 facts present in the briefing?
checks = ["Peregrine Star", "Moreau", "Vetlanda", "47", "March 9"]
found = [c for c in checks if c.lower() in out.lower()]
print(f"\n[coherence] facts used in briefing: {len(found)}/5 -> {found}", flush=True)
print("\n===== BRIEFING (first response) =====", flush=True)
print(out.strip()[:1500], flush=True)
print("\n===== BULLETS (second response) =====", flush=True)
print(out2.strip()[:1200], flush=True)
