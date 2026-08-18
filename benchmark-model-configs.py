#!/usr/bin/env python3
"""Reproducibly compare configurations exposed through a vLLM server.

The script is designed for a single accelerator: start one configuration,
record it under a label, restart the server with the next configuration, and
record that label into the same result file.  The first run freezes the request
workload.  Later runs reuse it and reject incompatible overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
DEFAULT_PROMPT = """Explain how speculative decoding improves autoregressive
language-model inference. Discuss draft acceptance, verification cost, mean
accepted length, time to first token, and output-token throughput. Include a
worked numerical example and practical guidance for choosing the number of
speculative tokens."""
DEFAULT_SYSTEM = "You are a precise technical writer. Give a detailed, self-contained answer."

SPEC_COUNTERS = {
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "draft": "vllm:spec_decode_num_draft_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "accepted_by_pos": "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}
PROM_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)"
    r"(?:\s+\d+)?$"
)


class BenchmarkError(RuntimeError):
    """An actionable benchmark failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def api_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def request_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise BenchmarkError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise BenchmarkError(f"Could not reach {url}: {exc}") from exc


def discover_model(base_url: str, headers: dict[str, str], timeout: float) -> str:
    payload = request_json(f"{base_url}/models", headers=headers, timeout=timeout)
    models = payload.get("data", [])
    if not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise BenchmarkError(f"{base_url}/models did not return a model ID")
    return str(models[0]["id"])


def metrics_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/metrics", "", ""))


def parse_spec_metrics(text: str) -> dict[str, Any]:
    scalar = {key: 0.0 for key in ("accepted", "draft", "drafts")}
    positions: dict[str, float] = {}
    found: set[str] = set()
    names_to_keys = {value: key for key, value in SPEC_COUNTERS.items()}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE.match(line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        key = names_to_keys.get(name)
        if key is None:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        found.add(key)
        if key == "accepted_by_pos":
            position_match = re.search(r'(?:^|,)position="([^"]+)"', labels or "")
            if position_match:
                position = position_match.group(1)
                positions[position] = positions.get(position, 0.0) + value
        else:
            scalar[key] += value

    return {
        "available": bool(found),
        **scalar,
        "accepted_by_pos": positions,
    }


def fetch_spec_metrics(
    base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any] | None:
    request = Request(metrics_url(base_url), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return parse_spec_metrics(response.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, TimeoutError):
        return None


def spec_metric_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not before or not after or not before.get("available") or not after.get("available"):
        return None

    delta = {
        key: max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0)))
        for key in ("accepted", "draft", "drafts")
    }
    all_positions = set(before.get("accepted_by_pos", {})) | set(
        after.get("accepted_by_pos", {})
    )
    delta["accepted_by_pos"] = {
        position: max(
            0.0,
            float(after.get("accepted_by_pos", {}).get(position, 0.0))
            - float(before.get("accepted_by_pos", {}).get(position, 0.0)),
        )
        for position in all_positions
    }
    if delta["draft"] > 0:
        delta["acceptance_percent"] = 100.0 * delta["accepted"] / delta["draft"]
    if delta["drafts"] > 0:
        delta["mean_accepted_length"] = 1.0 + delta["accepted"] / delta["drafts"]
        delta["acceptance_by_position_percent"] = {
            position: 100.0 * value / delta["drafts"]
            for position, value in sorted(
                delta["accepted_by_pos"].items(), key=lambda item: int(item[0])
            )
        }
    return delta


def delta_text_pieces(delta: dict[str, Any]) -> list[str]:
    """Return visible and reasoning text from old and new vLLM stream schemas."""
    pieces: list[str] = []
    reasoning = delta.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        pieces.append(reasoning)

    content = delta.get("content")
    if isinstance(content, str) and content:
        pieces.append(content)
    return pieces


def streamed_chat_completion(
    base_url: str,
    *,
    headers: dict[str, str],
    workload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": workload["model"],
        "messages": [
            {"role": "system", "content": workload["system"]},
            {"role": "user", "content": workload["prompt"]},
        ],
        "max_tokens": workload["max_tokens"],
        "temperature": workload["temperature"],
        "top_p": workload["top_p"],
        "seed": workload["seed"],
        "ignore_eos": workload["ignore_eos"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if workload.get("chat_template_kwargs") is not None:
        payload["chat_template_kwargs"] = workload["chat_template_kwargs"]
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    first_text_at: float | None = None
    last_text_at: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    observed_delta_fields: set[str] = set()

    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"Invalid streaming JSON: {data[:300]}") from exc
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if isinstance(delta, dict):
                        observed_delta_fields.update(str(field) for field in delta)
                    event_pieces = delta_text_pieces(delta)
                    if event_pieces:
                        now = time.perf_counter()
                        if first_text_at is None:
                            first_text_at = now
                        last_text_at = now
                        pieces.extend(event_pieces)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise BenchmarkError(f"HTTP {exc.code} from chat completions: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise BenchmarkError(f"Chat completion failed: {exc}") from exc

    finished = time.perf_counter()
    if usage is None:
        raise BenchmarkError(
            "The server did not return streaming usage. Ensure vLLM supports "
            "stream_options.include_usage."
        )
    if first_text_at is None or last_text_at is None:
        fields = ", ".join(sorted(observed_delta_fields)) or "none"
        raise BenchmarkError(
            "The response contained no recognized streamed text "
            f"(observed delta fields: {fields})"
        )

    completion_tokens = int(usage.get("completion_tokens", 0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    token_span = max(0.0, last_text_at - first_text_at)
    decode_tokens = max(0, completion_tokens - 1)
    decode_tps = decode_tokens / token_span if token_span > 0 else None
    response_text = "".join(pieces)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_seconds": finished - started,
        "ttft_seconds": first_text_at - started,
        "decode_seconds": token_span,
        "decode_tokens_per_second": decode_tps,
        "end_to_end_output_tokens_per_second": (
            completion_tokens / (finished - started) if finished > started else None
        ),
        "response_characters": len(response_text),
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise BenchmarkError("No measured runs were recorded")

    def values(name: str) -> list[float]:
        return [float(run[name]) for run in runs if run.get(name) is not None]

    decode_rates = values("decode_tokens_per_second")
    ttfts = values("ttft_seconds")
    totals = values("total_seconds")
    completion_counts = values("completion_tokens")
    hashes = [str(run["response_sha256"]) for run in runs]
    if not decode_rates:
        raise BenchmarkError(
            "No decode rate could be calculated. Use at least two output tokens; "
            "for fixed-length comparisons, leave --ignore-eos enabled."
        )
    return {
        "runs": len(runs),
        "median_decode_tokens_per_second": statistics.median(decode_rates),
        "mean_decode_tokens_per_second": statistics.fmean(decode_rates),
        "p10_decode_tokens_per_second": percentile(decode_rates, 0.10),
        "p90_decode_tokens_per_second": percentile(decode_rates, 0.90),
        "median_ttft_seconds": statistics.median(ttfts),
        "median_total_seconds": statistics.median(totals),
        "median_completion_tokens": statistics.median(completion_counts),
        "response_sha256": hashes[0] if len(set(hashes)) == 1 else None,
        "outputs_stable_within_config": len(set(hashes)) == 1,
    }


def format_number(value: Any, decimals: int = 2, missing: str = "-") -> str:
    if value is None:
        return missing
    return f"{float(value):.{decimals}f}"


def print_comparison(document: dict[str, Any]) -> None:
    configs = document.get("configs", [])
    if not configs:
        print("No configurations have been recorded.")
        return

    baseline_rate = configs[0]["summary"]["median_decode_tokens_per_second"]
    baseline_hash = configs[0]["summary"].get("response_sha256")
    headers = (
        "Config",
        "Runs",
        "Decode tok/s",
        "Speedup",
        "TTFT s",
        "Total s",
        "MTP accept",
        "Mean len",
        "Same output",
    )
    rows: list[tuple[str, ...]] = []
    for config in configs:
        summary = config["summary"]
        rate = summary["median_decode_tokens_per_second"]
        spec = config.get("speculative_metrics") or {}
        output_hash = summary.get("response_sha256")
        if not summary.get("outputs_stable_within_config"):
            same_output = "mixed"
        elif baseline_hash is None:
            same_output = "unknown"
        else:
            same_output = "yes" if output_hash == baseline_hash else "no"
        rows.append(
            (
                str(config["label"]),
                str(summary["runs"]),
                format_number(rate),
                format_number(rate / baseline_rate, 3) + "x",
                format_number(summary["median_ttft_seconds"], 3),
                format_number(summary["median_total_seconds"], 2),
                (
                    format_number(spec.get("acceptance_percent"), 1) + "%"
                    if spec.get("acceptance_percent") is not None
                    else "-"
                ),
                format_number(spec.get("mean_accepted_length"), 2),
                same_output,
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"Result file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid result JSON in {path}: {exc}") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError(
            f"Unsupported result schema {document.get('schema_version')!r} in {path}"
        )
    return document


def save_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_prompt(args: argparse.Namespace) -> str | None:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def resolve_workload(
    args: argparse.Namespace,
    existing: dict[str, Any] | None,
    *,
    discovered_model: str,
) -> dict[str, Any]:
    provided = {
        "prompt": selected_prompt(args),
        "system": args.system,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
    }
    if existing is not None:
        for key, value in provided.items():
            if value is not None and value != existing.get(key):
                raise BenchmarkError(
                    f"The existing session fixes {key}={existing.get(key)!r}, "
                    f"but this invocation requested {value!r}. Use a new result file."
                )
        return existing

    workload = {
        "prompt": provided["prompt"] if provided["prompt"] is not None else DEFAULT_PROMPT,
        "system": provided["system"] if provided["system"] is not None else DEFAULT_SYSTEM,
        "model": provided["model"] if provided["model"] is not None else discovered_model,
        "max_tokens": provided["max_tokens"] if provided["max_tokens"] is not None else 512,
        "temperature": (
            provided["temperature"] if provided["temperature"] is not None else 0.0
        ),
        "top_p": provided["top_p"] if provided["top_p"] is not None else 1.0,
        "seed": provided["seed"] if provided["seed"] is not None else 42,
        "ignore_eos": (
            provided["ignore_eos"] if provided["ignore_eos"] is not None else True
        ),
    }
    workload["prompt_sha256"] = hashlib.sha256(
        workload["prompt"].encode("utf-8")
    ).hexdigest()
    return workload


def command_run(args: argparse.Namespace) -> None:
    result_path: Path = args.results
    document = load_document(result_path) if result_path.exists() else None
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = api_headers(api_key)
    base_url = normalize_base_url(args.base_url)
    parsed_base_url = urlsplit(base_url)
    if parsed_base_url.username or parsed_base_url.password:
        raise BenchmarkError(
            "Do not embed credentials in --base-url; provide the API key through "
            "--api-key-env instead."
        )
    discovered_model = discover_model(base_url, headers, args.timeout)
    workload = resolve_workload(
        args,
        document.get("workload") if document else None,
        discovered_model=discovered_model,
    )
    if workload["model"] != discovered_model and args.model is None:
        print(
            f"Note: session requests model {workload['model']!r}; server advertises "
            f"{discovered_model!r}.",
            file=sys.stderr,
        )

    if document is None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "workload": workload,
            "configs": [],
        }
    if any(config.get("label") == args.label for config in document["configs"]):
        raise BenchmarkError(
            f"Configuration label {args.label!r} already exists in {result_path}. "
            "Choose a new label or a new result file."
        )

    print(
        f"Benchmarking {args.label}: {args.warmups} warm-up(s), "
        f"{args.runs} measured run(s), {workload['max_tokens']} output tokens"
    )
    for index in range(args.warmups):
        print(f"  warm-up {index + 1}/{args.warmups}", flush=True)
        streamed_chat_completion(
            base_url, headers=headers, workload=workload, timeout=args.timeout
        )

    metrics_before = fetch_spec_metrics(base_url, headers, args.timeout)
    runs: list[dict[str, Any]] = []
    for index in range(args.runs):
        print(f"  measured {index + 1}/{args.runs}", end="", flush=True)
        run = streamed_chat_completion(
            base_url, headers=headers, workload=workload, timeout=args.timeout
        )
        runs.append(run)
        print(
            f"  {format_number(run['decode_tokens_per_second'])} decode tok/s, "
            f"TTFT {format_number(run['ttft_seconds'], 3)}s",
            flush=True,
        )
    metrics_after = fetch_spec_metrics(base_url, headers, args.timeout)

    config_result = {
        "label": args.label,
        "recorded_at": utc_now(),
        "base_url": base_url,
        "server_model": discovered_model,
        "runs": runs,
        "summary": summarize_runs(runs),
        "speculative_metrics": spec_metric_delta(metrics_before, metrics_after),
    }
    document["configs"].append(config_result)
    save_document(result_path, document)
    print(f"\nSaved {result_path}")
    print_comparison(document)


def command_compare(args: argparse.Namespace) -> None:
    print_comparison(load_document(args.results))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare vLLM configurations with one frozen request workload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Single-GPU MTP example:
  # Start the MTP-3 server, then:
  ./benchmark-model-configs.py run --label mtp3 --results /tmp/qwen-mtp.json

  # Restart with MTP-4, then append the same frozen workload:
  ./benchmark-model-configs.py run --label mtp4 --results /tmp/qwen-mtp.json

  ./benchmark-model-configs.py compare --results /tmp/qwen-mtp.json

The default workload uses greedy decoding, a fixed seed, ignore_eos=true, and
512 output tokens. Exact response hashes are compared across configurations.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Benchmark the currently running server")
    run.set_defaults(handler=command_run)
    run.add_argument("--label", required=True, help="Unique configuration label")
    run.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Session JSON file (default: benchmark-results.json)",
    )
    run.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible v1 URL",
    )
    run.add_argument("--model", help="Model ID; defaults to the first /v1/models entry")
    prompt_group = run.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="User prompt for a new session")
    prompt_group.add_argument("--prompt-file", type=Path, help="Read the user prompt from a file")
    run.add_argument("--system", help="System prompt for a new session")
    run.add_argument("--max-tokens", type=int, help="Fixed output-token limit (default: 512)")
    run.add_argument("--temperature", type=float, help="Sampling temperature (default: 0)")
    run.add_argument("--top-p", type=float, help="Top-p sampling value (default: 1)")
    run.add_argument("--seed", type=int, help="Sampling seed (default: 42)")
    run.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force exactly max-tokens output (default: enabled)",
    )
    run.add_argument("--runs", type=int, default=5, help="Measured requests (default: 5)")
    run.add_argument("--warmups", type=int, default=1, help="Unmeasured warm-ups (default: 1)")
    run.add_argument("--timeout", type=float, default=600, help="Per-request timeout seconds")
    run.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; its value is never saved",
    )

    compare = subparsers.add_parser("compare", help="Print a saved comparison")
    compare.set_defaults(handler=command_compare)
    compare.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Session JSON file (default: benchmark-results.json)",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    if args.runs < 1:
        raise BenchmarkError("--runs must be at least 1")
    if args.warmups < 0:
        raise BenchmarkError("--warmups cannot be negative")
    if args.max_tokens is not None and args.max_tokens < 2:
        raise BenchmarkError("--max-tokens must be at least 2")
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    if args.temperature is not None and args.temperature < 0:
        raise BenchmarkError("--temperature cannot be negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise BenchmarkError("--top-p must be greater than 0 and at most 1")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        args.handler(args)
    except (BenchmarkError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
