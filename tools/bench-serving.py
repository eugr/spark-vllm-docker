#!/usr/bin/env python3
"""Benchmark a running vLLM server: decode rate, concurrency scaling, prefill rate.

Runs on the cluster head node against the local server. Reads the API key from
/home/paul/.vllm-api-key so the key never has to be passed on a command line.

Every number it prints is derived from the server's own token accounting
(usage.completion_tokens, usage.prompt_tokens) rather than a client-side token
estimate, and speculative-decoding acceptance is read from /metrics deltas
across the run rather than inferred.

  ./bench-serving.py --model qwen3.8-27b-1m --suite decode
  ./bench-serving.py --model qwen3.8-27b-1m --suite all --json out.json
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# Prompt families. Decode rate is nearly content-independent, but acceptance
# length is not: a drafter does much better on boilerplate than on prose, so a
# single prompt would over- or under-state speculative decoding's benefit.
PROMPTS = {
    "prose": "Write a detailed essay about the history of the printing press "
             "and its effect on literacy in early modern Europe.",
    "code": "Write a complete Python implementation of a red-black tree with "
            "insert, delete, and search. Include docstrings.",
    "chat": "I'm planning a two-week trip to Japan in the spring. Walk me "
            "through an itinerary and explain your reasoning for each stop.",
    "repetitive": "Count from 1 to 500, writing each number on its own line as "
                  "'Line N: value N'.",
}


def api_key():
    path = os.environ.get("VLLM_API_KEY_FILE", "/home/paul/.vllm-api-key")
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(f"no VLLM_API_KEY in {path}")


class Client:
    def __init__(self, base, key):
        self.base = base.rstrip("/")
        self.key = key

    def _req(self, path, payload=None, timeout=1800):
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data)
        req.add_header("Authorization", f"Bearer {self.key}")
        # Ask the server to close rather than hold the socket open after the
        # final SSE frame; a kept-alive idle socket is what makes a finished
        # stream look like a hung one.
        req.add_header("Connection", "close")
        if data:
            req.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(req, context=CTX, timeout=timeout)

    def complete(self, prompt, max_tokens, temperature=0.0):
        """Non-streaming completion. Returns (elapsed_s, usage dict)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.perf_counter()
        with self._req("/v1/completions", payload) as resp:
            body = json.loads(resp.read())
        return time.perf_counter() - t0, body["usage"], body["choices"][0]["text"]

    def stream_ttft(self, prompt, max_tokens, temperature=0.0, timeout=1800):
        """Streaming completion. Returns (ttft_s, total_s, completion_tokens, prompt_tokens).

        TTFT isolates prefill: for a long prompt it is essentially the prefill
        time, so prompt_tokens/ttft is the prefill rate.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t0 = time.perf_counter()
        ttft = None
        usage = None
        with self._req("/v1/completions", payload, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                obj = json.loads(chunk)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if ttft is None and choices and choices[0].get("text"):
                    ttft = time.perf_counter() - t0
                if usage and choices and choices[0].get("finish_reason"):
                    # Everything needed is in hand; do not wait on [DONE].
                    break
        total = time.perf_counter() - t0
        return ttft, total, usage

    def snapshot(self):
        """One /metrics scrape -> (flat counters, per-draft-position counters)."""
        flat, per_pos = {}, {}
        with self._req("/metrics") as resp:
            text = resp.read().decode()
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            labels, _, value = line.rpartition(" ")
            try:
                val = float(value)
            except ValueError:
                continue
            name = labels.split("{")[0].strip()
            flat[name] = flat.get(name, 0.0) + val
            if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
                pos = int(labels.split('position="')[1].split('"')[0])
                per_pos[pos] = per_pos.get(pos, 0.0) + val
        return flat, per_pos


def positional(before, after, drafts):
    """Per-draft-position acceptance rate.

    This is what actually decides k. Position i's rate is the probability that
    the i-th drafted token survives verification, and it falls monotonically
    because a position is only reached if every earlier one was accepted. Once
    a position's rate no longer pays for the drafter pass that produces it,
    every deeper position is dead weight -- which is why k=8 lost to k=3 here
    despite a higher mean acceptance length.
    """
    if drafts <= 0:
        return None
    rates = {}
    for pos, count in sorted(after.items()):
        delta = count - before.get(pos, 0.0)
        rates[pos] = round(delta / drafts, 3)
    return rates


def acceptance(before, after):
    """Mean accepted tokens per step, from /metrics deltas.

    vLLM emits one 'draft' per speculation step. Each step always commits the
    verified base token plus however many drafted tokens were accepted, so the
    mean acceptance length is 1 + accepted/drafts. A value of 1.0 means every
    draft was rejected, i.e. speculation is pure overhead.
    """
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - \
        before.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - \
        before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    draft_toks = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - \
        before.get("vllm:spec_decode_num_draft_tokens_total", 0)
    if drafts <= 0:
        return None
    return {
        "drafts": drafts,
        "accepted_tokens": accepted,
        "draft_tokens": draft_toks,
        "mean_acceptance_length": round(1 + accepted / drafts, 3),
        "per_token_accept_rate": round(accepted / draft_toks, 3) if draft_toks else None,
    }


def suite_decode(cli, repeats, max_tokens):
    """Single-stream decode rate per prompt family."""
    results = []
    for name, prompt in PROMPTS.items():
        # One warm-up per family so prefix caching and any lazy kernel
        # selection are settled before the measured passes.
        cli.complete(prompt, 32)
        for i in range(repeats):
            before, before_pos = cli.snapshot()
            elapsed, usage, _ = cli.complete(prompt, max_tokens)
            after, after_pos = cli.snapshot()
            n = usage["completion_tokens"]
            acc = acceptance(before, after)
            pos = positional(before_pos, after_pos, acc["drafts"] if acc else 0)
            results.append({
                "prompt": name,
                "run": i,
                "completion_tokens": n,
                "elapsed_s": round(elapsed, 3),
                "tok_per_s": round(n / elapsed, 2),
                "acceptance": acc,
                "positional_accept_rate": pos,
            })
            al = acc["mean_acceptance_length"] if acc else float("nan")
            print(f"  decode/{name} run{i}: {n / elapsed:6.2f} tok/s "
                  f"({n} tok in {elapsed:5.2f}s)  accept_len {al:.2f}  "
                  f"per_pos {pos}", flush=True)
    return results


def suite_concurrency(cli, levels, max_tokens):
    """Aggregate throughput as concurrent streams increase.

    Single-stream tok/s is the latency number; this is the capacity number, and
    the two move differently — a config can win one and lose the other.
    """
    results = []
    prompts = list(PROMPTS.values())
    for c in levels:
        # Distinct prompts per slot: identical prompts would share prefix-cache
        # blocks and overstate throughput.
        work = [f"{prompts[i % len(prompts)]} (variant {i})" for i in range(c)]
        before, before_pos = cli.snapshot()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            got = list(pool.map(lambda p: cli.complete(p, max_tokens), work))
        wall = time.perf_counter() - t0
        after, after_pos = cli.snapshot()
        total_tok = sum(u["completion_tokens"] for _, u, _ in got)
        per_stream = [u["completion_tokens"] / e for e, u, _ in got]
        results.append({
            "concurrency": c,
            "wall_s": round(wall, 3),
            "total_completion_tokens": total_tok,
            "aggregate_tok_per_s": round(total_tok / wall, 2),
            "mean_per_stream_tok_per_s": round(sum(per_stream) / len(per_stream), 2),
            "acceptance": acceptance(before, after),
            "positional_accept_rate": positional(
                before_pos, after_pos,
                (acceptance(before, after) or {}).get("drafts", 0)),
        })
        print(f"  concurrency {c}: {total_tok / wall:7.2f} tok/s aggregate, "
              f"{sum(per_stream) / len(per_stream):6.2f} tok/s per stream", flush=True)
    return results


def suite_prefill(cli, sizes):
    """Prefill rate at increasing prompt lengths, from TTFT.

    Prompts are built from a non-repeating filler so prefix caching cannot
    shortcut the prefill, and each size is requested once.
    """
    results = []
    for idx, target in enumerate(sizes):
        # This filler runs ~14 tokens per item. Each size gets its own salt so
        # the prompts are not nested prefixes of one another -- otherwise the
        # larger sizes score against a warm prefix cache and read far faster
        # than a genuine cold prefill.
        salt = f"s{idx}x{target}"
        words = " ".join(f"{salt}item{i:07d} value{i * 7 % 9973:05d}"
                         for i in range(max(1, target // 14)))
        prompt = f"Read the following log and reply with only the word OK.\n{words}\nReply:"
        ttft, total, usage = cli.stream_ttft(prompt, 8)
        ptok = usage["prompt_tokens"] if usage else None
        rate = round(ptok / ttft, 1) if (ptok and ttft) else None
        results.append({
            "target_tokens": target,
            "prompt_tokens": ptok,
            "ttft_s": round(ttft, 3) if ttft else None,
            "total_s": round(total, 3),
            "prefill_tok_per_s": rate,
        })
        shown = f"{ttft:.2f}s" if ttft else "n/a"
        print(f"  prefill {ptok} tok: TTFT {shown} -> {rate} tok/s "
              f"(total {total:.1f}s)", flush=True)
    return results


def suite_depth(cli, depths, gen_tokens):
    """Decode rate as a function of how much context is already in the KV cache.

    Prefill rate and decode rate answer different questions. Prefill says how
    long you wait to start; this says what the session feels like once you are
    deep into it, which is the number that decides whether a large context
    window is usable or merely available.

    Each depth is measured by timing the whole request and subtracting the
    measured TTFT, so the prefill is excluded rather than amortised into the
    decode figure.
    """
    results = []
    for idx, depth in enumerate(depths):
        salt = f"d{idx}x{depth}"
        words = " ".join(f"{salt}item{i:07d} value{i * 7 % 9973:05d}"
                         for i in range(max(1, depth // 14)))
        prompt = (f"Here is a log:\n{words}\n\n"
                  "Ignore the log. Write a detailed paragraph about tidal "
                  "patterns.")
        before, before_pos = cli.snapshot()
        ttft, total, usage = cli.stream_ttft(prompt, gen_tokens)
        after, after_pos = cli.snapshot()
        if not (ttft and usage):
            print(f"  depth {depth}: no usable timing", flush=True)
            continue
        acc = acceptance(before, after)
        pos = positional(before_pos, after_pos, acc["drafts"] if acc else 0)
        n = usage["completion_tokens"]
        decode_s = total - ttft
        rate = n / decode_s if decode_s > 0 else None
        results.append({
            "context_tokens": usage["prompt_tokens"],
            "completion_tokens": n,
            "ttft_s": round(ttft, 3),
            "decode_s": round(decode_s, 3),
            "decode_tok_per_s": round(rate, 2) if rate else None,
            "acceptance": acc,
            "positional_accept_rate": pos,
        })
        al = acc["mean_acceptance_length"] if acc else float("nan")
        print(f"  depth {usage['prompt_tokens']:>7} tok: decode "
              f"{rate:5.2f} tok/s (TTFT {ttft:6.1f}s)  accept_len {al:.2f}  "
              f"per_pos {pos}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="all",
                    choices=["all", "decode", "concurrency", "prefill", "depth"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--prefill-sizes", default="8000,32000,100000")
    ap.add_argument("--depths", default="1000,16000,64000,200000")
    ap.add_argument("--depth-gen-tokens", type=int, default=128)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", dest="json_out", default="")
    args = ap.parse_args()

    cli = Client(args.base, api_key())
    cli.model = args.model

    out = {"label": args.label, "model": args.model, "suites": {}}
    want = (("decode", "concurrency", "prefill", "depth")
            if args.suite == "all" else (args.suite,))

    if "decode" in want:
        print("== decode (single stream) ==", flush=True)
        out["suites"]["decode"] = suite_decode(cli, args.repeats, args.max_tokens)
    if "concurrency" in want:
        print("== concurrency ==", flush=True)
        levels = [int(x) for x in args.concurrency.split(",")]
        out["suites"]["concurrency"] = suite_concurrency(cli, levels, args.max_tokens)
    if "prefill" in want:
        print("== prefill ==", flush=True)
        sizes = [int(x) for x in args.prefill_sizes.split(",")]
        out["suites"]["prefill"] = suite_prefill(cli, sizes)

    if "depth" in want:
        print("== decode at context depth ==", flush=True)
        depths = [int(x) for x in args.depths.split(",")]
        out["suites"]["depth"] = suite_depth(cli, depths, args.depth_gen_tokens)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
