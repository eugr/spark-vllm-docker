#!/usr/bin/env python3
"""Compare vLLM configurations with a frozen code-generation task suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_SPEC = importlib.util.spec_from_file_location(
    "benchmark_model_configs", SCRIPT_DIR / "benchmark-model-configs.py"
)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise RuntimeError("Could not load benchmark-model-configs.py")
CORE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE)

SCHEMA_VERSION = 1
BENCHMARK_TYPE = "code-generation"
DEFAULT_SYSTEM = (
    "You are an expert software engineer. Output only the requested source code "
    "without Markdown fences or explanatory prose. Produce complete, executable code."
)
DEFAULT_TASKS = [
    {
        "name": "python_ttl_lru",
        "language": "Python",
        "system": DEFAULT_SYSTEM,
        "prompt": """Write a Python 3 module implementing a thread-safe, bounded LRU
cache with per-entry TTL. Use only the standard library. Include type hints,
get, set, delete, clear, capacity eviction, lazy expiry, cache statistics, and
comprehensive unittest tests with a controllable clock. Return only the full
module source code.""",
    },
    {
        "name": "typescript_async_map",
        "language": "TypeScript",
        "system": DEFAULT_SYSTEM,
        "prompt": """Write a strict TypeScript implementation of an asynchronous
map function with a configurable concurrency limit. It must preserve input
order, stop scheduling after the first failure, support AbortSignal, clean up
listeners, and handle synchronous callback exceptions. Include dependency-free
tests and return only the complete source code.""",
    },
    {
        "name": "rust_log_aggregator",
        "language": "Rust",
        "system": DEFAULT_SYSTEM,
        "prompt": """Write a Rust program using only the standard library that parses
streaming web-server log lines, rejects malformed records without panicking,
and aggregates request counts, byte totals, status classes, and the ten busiest
paths. Define clear data types, avoid unnecessary allocation, include unit
tests, and return only the complete source code.""",
    },
    {
        "name": "postgres_event_analytics",
        "language": "SQL",
        "system": DEFAULT_SYSTEM,
        "prompt": """Write a PostgreSQL migration for users and immutable product
events, including constraints, partitioning by month, appropriate indexes, and
safe creation of the next partition. Then provide SQL queries for daily active
users, a seven-day rolling average, and weekly signup-cohort retention. Include
SQL comments and return only executable SQL.""",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_tasks(tasks: Any) -> list[dict[str, str]]:
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise CORE.BenchmarkError("The code suite must contain a non-empty task list")

    validated: list[dict[str, str]] = []
    names: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise CORE.BenchmarkError(f"Task {index + 1} must be a JSON object")
        name = task.get("name")
        prompt = task.get("prompt")
        language = task.get("language", "Code")
        system = task.get("system", DEFAULT_SYSTEM)
        for field, value in (
            ("name", name),
            ("prompt", prompt),
            ("language", language),
            ("system", system),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CORE.BenchmarkError(
                    f"Task {index + 1} requires a non-empty {field} string"
                )
        if name in names:
            raise CORE.BenchmarkError(f"Duplicate task name: {name}")
        names.add(name)
        validated.append(
            {
                "name": name,
                "language": language,
                "system": system,
                "prompt": prompt,
            }
        )
    return validated


def load_suite(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return validate_tasks(DEFAULT_TASKS)
    try:
        return validate_tasks(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise CORE.BenchmarkError(f"Suite file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CORE.BenchmarkError(f"Invalid suite JSON in {path}: {exc}") from exc


def load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CORE.BenchmarkError(f"Result file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CORE.BenchmarkError(f"Invalid result JSON in {path}: {exc}") from exc
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("benchmark_type") != BENCHMARK_TYPE
    ):
        raise CORE.BenchmarkError(f"Unsupported code benchmark document: {path}")
    return document


def save_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate_task_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [run for task in task_results for run in task["runs"]]
    rates = [
        float(run["decode_tokens_per_second"])
        for run in runs
        if run.get("decode_tokens_per_second") is not None
    ]
    if not rates:
        raise CORE.BenchmarkError(
            "No decode rate could be calculated. Use at least two output tokens; "
            "for fixed-length comparisons, leave --ignore-eos enabled."
        )
    ttfts = [float(run["ttft_seconds"]) for run in runs]
    totals = [float(run["total_seconds"]) for run in runs]
    return {
        "tasks": len(task_results),
        "samples": len(runs),
        "median_decode_tokens_per_second": statistics.median(rates),
        "mean_decode_tokens_per_second": statistics.fmean(rates),
        "p10_decode_tokens_per_second": CORE.percentile(rates, 0.10),
        "p90_decode_tokens_per_second": CORE.percentile(rates, 0.90),
        "median_ttft_seconds": statistics.median(ttfts),
        "median_request_seconds": statistics.median(totals),
        "outputs_stable_within_config": all(
            task["summary"]["outputs_stable_within_config"] for task in task_results
        ),
        "response_hashes": {
            task["name"]: task["summary"]["response_sha256"] for task in task_results
        },
    }


def same_outputs(config: dict[str, Any], baseline: dict[str, Any]) -> str:
    summary = config["summary"]
    baseline_summary = baseline["summary"]
    if not summary["outputs_stable_within_config"]:
        return "mixed"
    if not baseline_summary["outputs_stable_within_config"]:
        return "unknown"
    return (
        "yes"
        if summary["response_hashes"] == baseline_summary["response_hashes"]
        else "no"
    )


def print_config_comparison(document: dict[str, Any]) -> None:
    configs = document.get("configs", [])
    if not configs:
        print("No configurations have been recorded.")
        return
    baseline = configs[0]
    baseline_rate = baseline["summary"]["median_decode_tokens_per_second"]
    headers = (
        "Config",
        "Samples",
        "Decode tok/s",
        "Speedup",
        "TTFT s",
        "Request s",
        "MTP accept",
        "Mean len",
        "Same output",
    )
    rows: list[tuple[str, ...]] = []
    for config in configs:
        summary = config["summary"]
        spec = config.get("speculative_metrics") or {}
        rate = summary["median_decode_tokens_per_second"]
        rows.append(
            (
                config["label"],
                str(summary["samples"]),
                CORE.format_number(rate),
                CORE.format_number(rate / baseline_rate, 3) + "x",
                CORE.format_number(summary["median_ttft_seconds"], 3),
                CORE.format_number(summary["median_request_seconds"], 2),
                (
                    CORE.format_number(spec.get("acceptance_percent"), 1) + "%"
                    if spec.get("acceptance_percent") is not None
                    else "-"
                ),
                CORE.format_number(spec.get("mean_accepted_length"), 2),
                same_outputs(config, baseline),
            )
        )
    print_table(headers, rows)


def print_task_results(config: dict[str, Any]) -> None:
    headers = ("Task", "Language", "Decode tok/s", "TTFT s", "MTP accept", "Mean len")
    rows: list[tuple[str, ...]] = []
    for task in config["task_results"]:
        summary = task["summary"]
        spec = task.get("speculative_metrics") or {}
        rows.append(
            (
                task["name"],
                task["language"],
                CORE.format_number(summary["median_decode_tokens_per_second"]),
                CORE.format_number(summary["median_ttft_seconds"], 3),
                (
                    CORE.format_number(spec.get("acceptance_percent"), 1) + "%"
                    if spec.get("acceptance_percent") is not None
                    else "-"
                ),
                CORE.format_number(spec.get("mean_accepted_length"), 2),
            )
        )
    print_table(headers, rows)


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def resolve_session(
    args: argparse.Namespace,
    existing: dict[str, Any] | None,
    discovered_model: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if existing is not None:
        settings = existing["settings"]
        checks = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "runs_per_task": args.runs_per_task,
            "warmups": args.warmups,
            "enable_thinking": args.enable_thinking,
        }
        for key, value in checks.items():
            if value is not None and value != settings[key]:
                raise CORE.BenchmarkError(
                    f"The existing session fixes {key}={settings[key]!r}, but "
                    f"this invocation requested {value!r}. Use a new result file."
                )
        if args.suite_file is not None and load_suite(args.suite_file) != existing["tasks"]:
            raise CORE.BenchmarkError(
                "The supplied suite differs from the frozen session; use a new result file."
            )
        return existing["tasks"], settings

    tasks = load_suite(args.suite_file)
    settings = {
        "model": args.model or discovered_model,
        "max_tokens": args.max_tokens if args.max_tokens is not None else 512,
        "runs_per_task": args.runs_per_task if args.runs_per_task is not None else 3,
        "warmups": args.warmups if args.warmups is not None else 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "ignore_eos": True,
        "enable_thinking": (
            args.enable_thinking if args.enable_thinking is not None else False
        ),
    }
    return tasks, settings


def task_workload(
    task: dict[str, str], settings: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": settings["model"],
        "system": task["system"],
        "prompt": task["prompt"],
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "seed": settings["seed"],
        "ignore_eos": settings["ignore_eos"],
        "chat_template_kwargs": {"enable_thinking": settings["enable_thinking"]},
    }


def command_run(args: argparse.Namespace) -> None:
    existing = load_document(args.results) if args.results.exists() else None
    base_url = CORE.normalize_base_url(args.base_url)
    parsed_url = urlsplit(base_url)
    if parsed_url.username or parsed_url.password:
        raise CORE.BenchmarkError(
            "Do not embed credentials in --base-url; use --api-key-env instead."
        )
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = CORE.api_headers(api_key)
    discovered_model = CORE.discover_model(base_url, headers, args.timeout)
    tasks, settings = resolve_session(args, existing, discovered_model)

    if existing is None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_type": BENCHMARK_TYPE,
            "created_at": utc_now(),
            "tasks": tasks,
            "settings": settings,
            "configs": [],
        }
    else:
        document = existing
    if any(config["label"] == args.label for config in document["configs"]):
        raise CORE.BenchmarkError(
            f"Configuration label {args.label!r} already exists in {args.results}"
        )

    print(
        f"Benchmarking {args.label}: {len(tasks)} code tasks, "
        f"{settings['runs_per_task']} run(s) each, "
        f"{settings['max_tokens']} output tokens"
    )
    warmup_workload = task_workload(tasks[0], settings)
    for index in range(settings["warmups"]):
        print(f"  warm-up {index + 1}/{settings['warmups']}", flush=True)
        CORE.streamed_chat_completion(
            base_url, headers=headers, workload=warmup_workload, timeout=args.timeout
        )

    config_metrics_before = CORE.fetch_spec_metrics(base_url, headers, args.timeout)
    task_results: list[dict[str, Any]] = []
    for task in tasks:
        print(f"\n{task['name']} ({task['language']})", flush=True)
        workload = task_workload(task, settings)
        task_metrics_before = CORE.fetch_spec_metrics(base_url, headers, args.timeout)
        runs: list[dict[str, Any]] = []
        for index in range(settings["runs_per_task"]):
            print(
                f"  measured {index + 1}/{settings['runs_per_task']}",
                end="",
                flush=True,
            )
            run = CORE.streamed_chat_completion(
                base_url, headers=headers, workload=workload, timeout=args.timeout
            )
            runs.append(run)
            print(
                f"  {CORE.format_number(run['decode_tokens_per_second'])} decode tok/s, "
                f"TTFT {CORE.format_number(run['ttft_seconds'], 3)}s",
                flush=True,
            )
        task_metrics_after = CORE.fetch_spec_metrics(base_url, headers, args.timeout)
        task_results.append(
            {
                "name": task["name"],
                "language": task["language"],
                "runs": runs,
                "summary": CORE.summarize_runs(runs),
                "speculative_metrics": CORE.spec_metric_delta(
                    task_metrics_before, task_metrics_after
                ),
            }
        )
    config_metrics_after = CORE.fetch_spec_metrics(base_url, headers, args.timeout)

    config = {
        "label": args.label,
        "recorded_at": utc_now(),
        "base_url": base_url,
        "server_model": discovered_model,
        "task_results": task_results,
        "summary": aggregate_task_results(task_results),
        "speculative_metrics": CORE.spec_metric_delta(
            config_metrics_before, config_metrics_after
        ),
    }
    document["configs"].append(config)
    save_document(args.results, document)
    print(f"\nSaved {args.results}\n")
    print("Per-task results")
    print_task_results(config)
    print("\nConfiguration comparison")
    print_config_comparison(document)


def command_compare(args: argparse.Namespace) -> None:
    document = load_document(args.results)
    print_config_comparison(document)
    if args.details:
        for config in document["configs"]:
            print(f"\n{config['label']} per-task results")
            print_task_results(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM configurations on a frozen code-generation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Single-GPU example:
  ./benchmark-code-generation.py run --label mtp3 --results /tmp/qwen-code.json
  # Restart the server with MTP-4.
  ./benchmark-code-generation.py run --label mtp4 --results /tmp/qwen-code.json
  ./benchmark-code-generation.py compare --details --results /tmp/qwen-code.json

The first run freezes the suite and settings. Later configurations reuse them.
Default cost per configuration: 1 warm-up plus 4 tasks x 3 runs x 512 tokens.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Benchmark the currently running server")
    run.set_defaults(handler=command_run)
    run.add_argument("--label", required=True, help="Unique configuration label")
    run.add_argument(
        "--results",
        type=Path,
        default=Path("code-benchmark-results.json"),
        help="Session JSON file (default: code-benchmark-results.json)",
    )
    run.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible v1 URL",
    )
    run.add_argument("--model", help="Model ID; defaults to the first /v1/models entry")
    run.add_argument("--suite-file", type=Path, help="Custom JSON task suite for a new session")
    run.add_argument("--max-tokens", type=int, help="Tokens generated per task (default: 512)")
    run.add_argument("--runs-per-task", type=int, help="Measured runs per task (default: 3)")
    run.add_argument("--warmups", type=int, help="Warm-ups per configuration (default: 1)")
    run.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include model thinking tokens (default: disabled)",
    )
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
        default=Path("code-benchmark-results.json"),
        help="Session JSON file (default: code-benchmark-results.json)",
    )
    compare.add_argument("--details", action="store_true", help="Show per-task results")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    if args.max_tokens is not None and args.max_tokens < 2:
        raise CORE.BenchmarkError("--max-tokens must be at least 2")
    if args.runs_per_task is not None and args.runs_per_task < 1:
        raise CORE.BenchmarkError("--runs-per-task must be at least 1")
    if args.warmups is not None and args.warmups < 0:
        raise CORE.BenchmarkError("--warmups cannot be negative")
    if args.timeout <= 0:
        raise CORE.BenchmarkError("--timeout must be positive")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        args.handler(args)
    except (CORE.BenchmarkError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
