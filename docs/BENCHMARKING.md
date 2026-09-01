# Benchmarking vLLM recipes on the Spark cluster

Model-agnostic mechanics for measuring a live recipe: TLS trust, the
llama-benchy / lm-eval / spec-decode invocations, and the harness
landmines that cost real time. Per-recipe results, tuning history, and
model-specific deviations live in that recipe's own docs page (e.g.
`docs/GLM53_FLASH.md`), which references this file rather than repeating
it. Everything here applies equally to the qwen, ds4f, and GLM lanes.

## TLS: the pfSense CA (one-time per bench host)

The `vllm@` units serve HTTPS with a leaf cert issued by the local
**pfSense CA**, and the server presents only the leaf. Every verifying
client (llama-benchy, lm-eval, requests) then fails with
`unable to get local issuer certificate`. `curl -k` works, but those
tools have no skip-verify flag, so append the CA to a bundle:

```bash
# CA lives at ~/.config/certs/pfsense-ca.pem on the workstation
cat /etc/ssl/certs/ca-certificates.crt ~/.config/certs/pfsense-ca.pem \
  > ~/.config/certs/ca-bundle-plus-pfsense.pem
```

Then export `SSL_CERT_FILE=~/.config/certs/ca-bundle-plus-pfsense.pem`
(and `REQUESTS_CA_BUNDLE=` the same, for requests-based tools). Do NOT
point `SSL_CERT_FILE` at the CA alone - it replaces the system bundle and
breaks uv's own PyPI fetches. `tools/bench-serving.py` sidesteps all of
this with `CERT_NONE` (verification off); the probe scripts take
`--insecure` for the same reason.

The API key is `docker exec vllm_node printenv VLLM_API_KEY` (the host
`~/.vllm-api-key` may not match). Never echo it into a shared log.

## Host setup

uv/uvx via the standard `astral.sh/uv/install.sh` (installs to
`~/.local/bin`). A cluster node is a fine bench host and already has the
model snapshot in its HF cache; the workstation often cannot reach
huggingface.co, so run evals on a node and pass LOCAL paths.

## Speed - llama-benchy

```bash
KEY=$(docker exec vllm_node printenv VLLM_API_KEY)
SNAP=$(ls -d ~/.cache/huggingface/hub/models--<ORG>--<MODEL>/snapshots/*/)
SSL_CERT_FILE=~/.config/certs/ca-bundle-plus-pfsense.pem \
uvx llama-benchy@0.4.0 \
  --base-url https://<node>:8000/v1 --api-key "$KEY" \
  --model <served-name> --tokenizer "$SNAP" \
  --pp 2048 --tg 128 --runs 3 \
  --save-result ~/bench.json --format json
```

- `--base-url` MUST include the `/v1` suffix (404 without it).
- `--tokenizer` MUST be the local snapshot path - the bare HF id triggers
  a re-download that fails offline.
- Decode tok/s is noisy (CV ~13% on GB10); for a stable number or an A/B,
  raise `--runs` (15+) and use `--tg 256 --exact-tg`. Runs are seconds
  each - the engine boot is the real cost, so over-sample runs freely.

## Quality - lm-eval

Dependencies: `--from "lm_eval[api]" --with transformers` is the minimum
(the `[api]` extra does NOT pull transformers; the run dies on import
without it). For RULER add `,ruler` to the extra plus
`--with wonderwords --with nltk`.

```bash
SSL_CERT_FILE=~/.config/certs/ca-bundle-plus-pfsense.pem \
REQUESTS_CA_BUNDLE=~/.config/certs/ca-bundle-plus-pfsense.pem \
OPENAI_API_KEY="$KEY" \
uvx --from "lm_eval[api]" --with transformers lm_eval \
  --model local-completions \
  --model_args "model=<served-name>,base_url=https://<node>:8000/v1/completions,num_concurrent=8,max_retries=3,tokenizer=$SNAP,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --limit 200 \
  --output_path ~/lm-eval-results/gsm8k
```

RULER: `--tasks niah_single_2,niah_multikey_1,ruler_vt`, one length via
`--metadata '{"max_seq_lengths":[<LEN>]}'`, and **`--gen_kwargs
max_gen_toks=256`** (mandatory - the default budget truncates chat-model
ruler_vt answers and invalidates the score). 25 samples/length
(`--limit 25`).

### lm-eval landmines (each cost real time, 2026-08-31)

- **Validate the request count before trusting the table.** A partial or
  dropped run looks like a clean pass. Confirm fired ==
  tasks x lengths x limit (e.g. 3 x 1 x 25 = 75).
- **One length per RULER invocation.** A multi-value
  `max_seq_lengths:[65536,131072]` silently ran only the FIRST length
  (75/75, one length; the second absent from the table, no sentinel row).
- **Long context (>=64K): `num_concurrent<=2` AND `timeout=3600`.**
  lm-eval's aiohttp session timeout defaults to 300s. At high concurrency
  the multi-minute long-context prefills queue and later requests wait
  past 300s, so the session tears down with `RuntimeError: Session is
  closed` - a CLIENT-side failure; the server never receives the requests
  (its queue stays 0). At concurrency 1-2 each request finishes inside the
  raised budget. (Short-context tasks like GSM8K are fine at
  num_concurrent=8.)
- **Never pipe an lm-eval run through `head`.** `... | grep X | head -N`
  sends SIGPIPE up the chain when head closes and kills the eval
  mid-run - it exits 0 (head's status) with no results written. Use
  `tail`, or redirect to a file.
- **RULER long-context build is a CPU-bound false-stall.** Context
  construction is single-threaded token-by-token noise insertion (one
  core ~100%, minutes); the GPU/server queue sits at 0 the whole time.
  Do not mistake the idle server for a hang.

## Spec-decode acceptance (free, from /metrics)

After a few varied requests, read off `/metrics`:
`sum(vllm:spec_decode_num_accepted_tokens_total) /
vllm:spec_decode_num_drafts_total` gives tokens-per-step (+1 bonus token
per draft). `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=N}`
shows the per-depth acceptance decay - the lever for tuning
`num_speculative_tokens` (k): if deep positions rarely accept, a lower k
wastes less draft compute. A fresh engine boot zeroes these counters, so
one boot per variant gives a clean per-variant reading.

## Refusal / abliteration probe

For abliterated checkpoints, `tools/refusal-probe.py` is the ONLY gate
that fails if the config silently served the stock checkpoint (bench,
GSM8K, RULER all pass either way). It scores refuse-vs-comply on short
greedy generations over a benign set (over-refusal check) and a mild,
category-level sensitive set (non-operational by design - it measures
whether the refusal circuit fires, not content). Runs with `--insecure`
to skip the CA bundle.

## Provenance caveat

Record the exact invocation (concurrency, timeout, limit) alongside any
published number. The original GLM-5.3-Flash 131K RULER figures are in
`docs/GLM53_FLASH.md` but the command that produced them is recorded
nowhere (not git, not session transcripts) - do not assume a prior run's
parameters; measure and write them down.
