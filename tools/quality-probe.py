#!/usr/bin/env python3
"""Detect quality regressions between two vLLM serving configurations.

Throughput work on these recipes changes attention backends, KV headroom and
RoPE scaling. Some of those changes are supposed to be numerically neutral and
some are not, and the difference matters more than the tok/s:

  - An attention backend swap (FlashInfer -> TRITON_ATTN) should be neutral.
    Greedy output ought to match token for token. It will not match bit for bit
    in the logits, but the argmax path should be stable, and a divergence is a
    genuine finding rather than noise.
  - Raising gpu_memory_utilization should be exactly neutral.
  - Static YaRN is NOT neutral. It rescales RoPE for every request including
    short ones, so the 1M recipe is expected to score below the 262k recipe on
    short-context tasks. The question is how much.

So this records, for a fixed greedy prompt set, the exact completion and a
digest per prompt. Run it against config A, then against config B, then diff:

  ./quality-probe.py --model qwen3.8-27b   --out /tmp/q-triton.json
  ./quality-probe.py --model qwen3.8-27b-1m --out /tmp/q-yarn.json
  ./quality-probe.py --compare /tmp/q-triton.json /tmp/q-yarn.json

Image input is part of the served contract, not an extra. The checkpoint is
Qwen3_5ForConditionalGeneration with a 27-layer vision tower that the NVFP4
quantization deliberately leaves in bf16, and all three qwen3.8 recipes pass
--limit-mm-per-prompt, so a config change can regress the image path just as
easily as the text one. The vision probes below cover it. Use --no-vision when
the served model is text-only, or those probes will score it as broken.

Graded tasks have checkable answers, so a divergence can be labelled better or
worse instead of merely different. The set is small and is a regression tripwire,
not a benchmark suite -- it will not rank models, only tell you whether a config
change moved the output.
"""

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# Each probe is (id, prompt, expected-substring or None).
# Prompts are phrased to keep answers short and greedy-stable; long free-form
# generation diverges on trivia and would produce false alarms.
PROBES = [
    ("arith-1", "Compute 17 * 23. Reply with only the number.", "391"),
    ("arith-2", "Compute 2^12. Reply with only the number.", "4096"),
    ("arith-3", "What is 1234 + 8766? Reply with only the number.", "10000"),
    ("logic-1", "If all bloops are razzies and all razzies are lazzies, are all "
                "bloops lazzies? Answer yes or no only.", "yes"),
    ("logic-2", "A bat and ball cost $1.10 total. The bat costs $1.00 more than "
                "the ball. How much does the ball cost? Reply with only the amount.", "0.05"),
    ("code-1", "Write a Python one-liner that returns the number of vowels in a "
               "string s. Reply with only the expression.", "aeiou"),
    ("code-2", "In Python, what does list(range(3, 9, 2)) evaluate to? Reply with "
               "only the literal.", "[3, 5, 7]"),
    ("fact-1", "What is the chemical symbol for tungsten? Reply with only the symbol.", "W"),
    ("fact-2", "In what year did the Chernobyl disaster occur? Reply with only the year.", "1986"),
    ("format-1", "List exactly three primes greater than 50, comma separated, "
                 "nothing else.", None),
    ("recall-1", "Name the author of 'The Left Hand of Darkness'. Reply with only "
                 "the name.", "Le Guin"),
    ("struct-1", 'Return a JSON object with keys "a" and "b" set to 1 and 2. '
                 "Reply with only the JSON.", '"a"'),
]

# A 128x128 two-colour PNG: a blue triangle with a white "42" on it. Kept inline
# as base64 so this stays a single stdlib-only file with no binary fixture to
# lose. Small on purpose -- preprocessor_config.json sets a 65536-px^2
# (256x256) shortest_edge floor, so 96px, 128px and 160px sources all cost an
# identical 103 prompt tokens; only above the floor does cost move (a 320px
# source costs 139). Regenerate with:
#
#   im = Image.new("P", (128, 128))
#   im.putpalette([255, 255, 255, 30, 90, 220] + [0] * 762)
#   d = ImageDraw.Draw(im)
#   d.polygon([(16, 112), (64, 24), (112, 112)], fill=1)
#   d.text((64, 86), "42", fill=0, anchor="mm",
#          font=ImageFont.truetype("DejaVuSans-Bold.ttf", 44))
#   im.save("v128.png", optimize=True)
VISION_PROBE_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAADAFBMVEX///8eWtwAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACU"
    "oMXfAAABXklEQVR42u3X0a6DIBBFUfb///R9rREFdOhOc895beOsMIhDa0mSJEmSJEmSJC+D"
    "XR/+NwBkgQ0AWWADQBbYAJAFNgBcAcgCGwCywAaALLABIAtsAMgCGwCuAGSBDQBZYANAFtgA"
    "kAU2AGSBDQBXALLABoAssAEgC2wAyAIbALLABixU3CK4KtOvtUFwB7j4/+b6A0C1oFNqtN9K"
    "ATwCsHUBToBOj7bWPwJ6r0ThEowA/fOgDnBc6ek7et1ZcOz09wGfz2URQHkDVgFUN2AMuN25"
    "7xuwDKB6AUaA08/V9QeAmY/1UwATgFYNOC/ALaA/P70G9DfVeCgrAJwbcAO4nh9rOjAaVW/m"
    "19qvYBtMJYUzOguANvd2bgO0yePp2VVkoUmlF6V5wMT1qfAuNjsVvz4MbADFAH4NQDmA3wKw"
    "I3b9FYENAFlgA0AW2ACQBTYAZIENAH0JkiRJkiT5bv4ATo4NdhQcjKgAAAAASUVORK5CYII="
)

# Same (id, prompt, expected) shape as PROBES, all against VISION_PROBE_IMAGE.
# Three questions rather than one so a partial failure is legible: glyph
# recognition, colour, and both-plus-instruction-following degrade separately.
VISION_PROBES = [
    ("vision-1", "What number is written in this image? Reply with only the "
                 "number.", "42"),
    ("vision-2", "What colour is the shape in this image? Reply with only the "
                 "colour.", "blue"),
    ("vision-3", "Reply with exactly two words separated by a comma: the shape "
                 "you see, then the number written on it. Nothing else.",
                 "triangle, 42"),
]


def api_key():
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    path = os.environ.get("VLLM_API_KEY_FILE", "/home/paul/.vllm-api-key")
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("VLLM_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(f"no VLLM_API_KEY in {path}")


def ask(base, key, model, prompt, max_tokens, no_think, image=None):
    """One greedy chat completion. Returns (visible_text, reasoning_text)."""
    if image is None:
        content = prompt
    else:
        # vLLM accepts a data URL here; PNG and WebP both decode. The text block
        # goes first so the question is not split across the visual tokens.
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    if no_think:
        # The card's instruct path. Without this the model spends most of the
        # token budget in a reasoning block and short answers get truncated.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
    )
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Connection", "close")
    with urllib.request.urlopen(req, context=CTX, timeout=900) as resp:
        out = json.loads(resp.read())
    msg = out["choices"][0]["message"]
    return (msg.get("content") or "").strip(), (msg.get("reasoning_content") or "").strip()


def graded(text, expect):
    if expect is None:
        return None
    return expect.lower() in text.lower()


def run(args):
    key = api_key()
    results = {}
    passed = total = 0
    probes = [(pid, prompt, expect, None) for pid, prompt, expect in PROBES]
    if not args.no_vision:
        probes += [(pid, prompt, expect, VISION_PROBE_IMAGE)
                   for pid, prompt, expect in VISION_PROBES]
    for pid, prompt, expect, image in probes:
        text, reasoning = ask(args.base, key, args.model, prompt,
                              args.max_tokens, args.no_think, image)
        ok = graded(text, expect)
        if ok is not None:
            total += 1
            passed += int(ok)
        results[pid] = {
            "prompt": prompt,
            "answer": text,
            "digest": hashlib.sha256(text.encode()).hexdigest()[:16],
            "reasoning_chars": len(reasoning),
            "image": image is not None,
            "expected": expect,
            "correct": ok,
        }
        mark = {True: "PASS", False: "FAIL", None: "----"}[ok]
        shown = text.replace("\n", " ")[:58]
        print(f"  [{mark}] {pid:10s} {shown}", flush=True)

    out = {
        "label": args.label,
        "model": args.model,
        "no_think": args.no_think,
        "vision": not args.no_vision,
        "score": {"passed": passed, "graded": total},
        "results": results,
    }
    print(f"\ngraded {passed}/{total}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


def compare(path_a, path_b):
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    print(f"A = {a['label'] or a['model']}   graded "
          f"{a['score']['passed']}/{a['score']['graded']}")
    print(f"B = {b['label'] or b['model']}   graded "
          f"{b['score']['passed']}/{b['score']['graded']}\n")

    same = diff = 0
    regress, improve = [], []
    for pid in a["results"]:
        if pid not in b["results"]:
            continue
        ra, rb = a["results"][pid], b["results"][pid]
        if ra["digest"] == rb["digest"]:
            same += 1
            continue
        diff += 1
        print(f"DIVERGED {pid}")
        print(f"    A: {ra['answer'][:100]!r}")
        print(f"    B: {rb['answer'][:100]!r}")
        if ra["correct"] is True and rb["correct"] is False:
            regress.append(pid)
        elif ra["correct"] is False and rb["correct"] is True:
            improve.append(pid)

    print(f"\nidentical {same}/{same + diff}, diverged {diff}")
    if regress:
        print(f"REGRESSED (A correct, B wrong): {', '.join(regress)}")
    if improve:
        print(f"improved  (A wrong, B correct): {', '.join(improve)}")
    if diff and not regress and not improve:
        print("divergences are all on ungraded or equally-scored probes")
    # Non-zero only on a scored regression: a pure wording change between two
    # configs is expected on some of these and should not fail a script.
    return 1 if regress else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://localhost:8000")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-think", action="store_true",
                    help="send enable_thinking=false (the card's instruct path)")
    ap.add_argument("--no-vision", action="store_true",
                    help="skip the image probes (for a text-only served model)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.model:
        ap.error("--model is required unless --compare is used")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
