#!/usr/bin/env python3
"""Render a memory profile card as Markdown and a CPU/CUDA startup chart."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

import yaml

from profile_card import SCHEMA

GIB = 1024 ** 3
MIB = 1024 ** 2


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def amount(value, divisor=GIB, unit="GiB"):
    return f"{value / divisor:,.3f} {unit}" if number(value) else "unavailable"


def difference(value, baseline):
    return value - baseline if number(value) and number(baseline) else None


def cell(value):
    """Keep profile labels from changing Markdown table structure or adding HTML."""
    if value is None:
        return "unavailable"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers, rows):
    return "\n".join("| " + " | ".join(cell(value) for value in row) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def load_card(path):
    with Path(path).open() as stream:
        card = yaml.safe_load(stream)
    if not isinstance(card, dict) or card.get("profile_schema") != SCHEMA:
        raise ValueError(f"Expected a {SCHEMA} profile card")
    for key in ("hosts", "ranks", "api_processes"):
        if not isinstance(card.get(key), list) or any(not isinstance(row, dict) for row in card[key]):
            raise ValueError(f"Expected a list of mappings in {key}")
    hosts = [host.get("host_id") for host in card["hosts"]]
    if any(not isinstance(host, str) for host in hosts) or len(set(hosts)) != len(hosts):
        raise ValueError("Expected unique host IDs")
    for process in card["ranks"] + card["api_processes"]:
        if process.get("host_id") not in hosts:
            raise ValueError("Process refers to a host absent from the card")
        points = process.get("checkpoints", [])
        if not isinstance(points, list) or any(not isinstance(point, dict) or not number(point.get("elapsed_seconds")) for point in points):
            raise ValueError("Expected checkpoints with finite elapsed_seconds")
    return card


def process_label(process, api=False):
    name = "API" if api else process.get("rank_key", "rank unknown")
    return f"{name} (PID {process.get('pid', '?')})"


def startup_points(process, host):
    cutoff = host.get("startup_cutoff_elapsed_seconds")
    return sorted((point for point in process.get("checkpoints", [])
                   if not number(cutoff) or point["elapsed_seconds"] <= cutoff + 0.0005),
                  key=lambda point: point["elapsed_seconds"])


def series(points, field, *, relative=False, baseline=None):
    """Missing observations remain gaps; never fabricate pre-measurement zeros."""
    observed = [point[field] for point in points if number(point.get(field))]
    if not observed:
        return [], []
    if relative and baseline is None:
        baseline = observed[0]
    if relative and not number(baseline):
        return [], []
    times = [point["elapsed_seconds"] for point in points]
    values = [(point[field] - (baseline if relative else 0)) / GIB
              if number(point.get(field)) else float("nan") for point in points]
    return times, values


def host_points(card, host):
    rows = [point for process in card["ranks"] + card["api_processes"]
            if process["host_id"] == host["host_id"] for point in startup_points(process, host)]
    # A host snapshot observed by several processes is still one observation,
    # never a sum. Preserve distinct observations even at rounded equal times.
    pairs = sorted({(row["elapsed_seconds"], row["host_unavailable_bytes"])
                    for row in rows if number(row.get("host_unavailable_bytes"))})
    return [{"elapsed_seconds": time, "host_unavailable_bytes": used} for time, used in pairs]


def cpu_stack_series(processes, host):
    """Align CPU checkpoints with last-observation carry-forward within startup.

    No process has a value before its first sample. An explicit missing sample
    invalidates its value until a later successful observation. These sums are
    estimates from asynchronous samples, not simultaneous host measurements.
    """
    rows = [startup_points(process, host) for process in processes]
    times = sorted({point["elapsed_seconds"] for points in rows for point in points})
    components = []
    for points in rows:
        cursor, current, values = 0, float("nan"), []
        for time in times:
            while cursor < len(points) and points[cursor]["elapsed_seconds"] <= time:
                value = points[cursor].get("cpu_pss_bytes")
                current = value / GIB if number(value) else float("nan")
                cursor += 1
            values.append(current)
        components.append(values)
    total = [sum(values) if all(math.isfinite(value) for value in values) else float("nan")
             for values in zip(*components)]
    return times, components, total


def plot_cpu_stack(ax, processes, host, rank_count, colors):
    times, components, total = cpu_stack_series(processes, host)
    bottom = [0.0] * len(times)
    for pos, (process, values) in enumerate(zip(processes, components)):
        top = [base + value for base, value in zip(bottom, values)]
        color = colors(pos % 10)
        ax.fill_between(times, bottom, top, step="post", color=color, alpha=.65,
                        label=process_label(process, pos >= rank_count))
        ax.plot(times, top, drawstyle="steps-post", color=color, linewidth=.8)
        bottom = top
    if times:
        ax.plot(times, total, drawstyle="steps-post", color="#242424", linewidth=1.8,
                marker="o", markersize=3, label="Estimated total (recorded processes)")
        # Shade times at which any contributor is unknown, including startup
        # before the first observation. Never substitute zero for missing PSS.
        unknown_label = "Total unavailable"
        for (left, value), right in zip([(0, float("nan")), *zip(times, total)], times):
            if not math.isfinite(value) and right > left:
                ax.axvspan(left, right, color="#808080", alpha=.10, linewidth=0, label=unknown_label)
                unknown_label = None
        if math.isfinite(total[-1]):
            ax.annotate(f"Latest-sample sum: ~{total[-1]:.2f} GiB", xy=(times[-1], total[-1]),
                        xytext=(.55, .82), textcoords="axes fraction", fontsize=10,
                        arrowprops={"arrowstyle": "->", "color": "#444444"})
    ax.set_title("CPU RAM · stacked PSS, using each recorded process's latest sample", loc="left")


def markdown_report(card, graph=None, *, relative=False):
    coverage = card.get("coverage") or {}
    lines = ["# vLLM startup memory report", "",
             f"**Recipe:** {cell(card.get('recipe'))}  ",
             f"**Model:** {cell(card.get('model'))}  ",
             f"**Run:** {cell(card.get('run_id'))} · **Status:** {cell(card.get('status'))}", "",
             f"Ready ranks: {len(coverage.get('ready_ranks', []))}/{len(coverage.get('expected_ranks', []))}. "
             f"API readiness observed: {'yes' if coverage.get('api_readiness_observed') else 'no'}."]
    for key in ("missing_ranks", "duplicate_ranks", "instrumentation_errors"):
        if coverage.get(key):
            lines += [f"**{key.replace('_', ' ').capitalize()}:** {cell(coverage[key])}"]
    lines += ["", "This is an observed startup profile; it does not establish capacity for maximum serving load.", ""]
    if graph:
        lines += [f"![CPU and GPU startup memory]({graph})", ""]
    lines += ["Chart points are recorded checkpoints; connecting lines only guide the eye. "
              "Host sampled peaks can occur between these checkpoints. Each host has its own elapsed clock."]
    if relative:
        lines += ["The chart shows host growth from its pre-vLLM baseline and each process counter's "
                  "growth from its first available measurement. Measurements before that point are unknown. "
                  "CPU growth curves remain separate because their baselines occur at different times. "
                  "Tables retain absolute measurements."]
    else:
        lines += ["CPU PSS bands are stacked using each process's last available sample. Their top edge "
                  "is an estimated sum for the recorded processes, not a simultaneous host measurement. "
                  "The total is unavailable (gray background) until every contributor has been sampled, "
                  "or after an explicit missing sample until it recovers. Known lower bands can remain visible."]
    lines += ["", "## Host memory", "",
              "Host usage is MemTotal − MemAvailable, including unrelated activity. "
              "On unified memory systems, CPU and GPU share this memory; do not add these counters together.", ""]
    rows = []
    for host in card["hosts"]:
        baseline = (host.get("baseline_host_memory") or {}).get("unavailable_bytes")
        ready = (host.get("ready_host_memory") or {}).get("unavailable_bytes")
        total = (host.get("baseline_host_memory") or {}).get("MemTotal")
        percent = f" ({100 * ready / total:.1f}%)" if number(ready) and number(total) and total > 0 else ""
        rows.append([host["host_id"], amount(total), amount(baseline), amount(ready) + percent,
                     amount(difference(ready, baseline)), amount(host.get("startup_peak_unavailable_bytes")),
                     amount(host.get("startup_peak_increment_bytes"))])
    lines += [table(["Host", "Total RAM", "Baseline used", "Ready used", "Ready growth", "Sampled peak used", "Peak growth"], rows), "",
              "## Rank storage and CUDA memory", "",
              "Model and allocator values are measured at readiness. Incomplete ranks may show an earlier KV allocation. "
              "Model and KV values count unique backing storage. Other CUDA allocations include graph/workspace "
              "allocations visible to PyTorch; native/driver allocations can be outside these counters.", ""]
    rows = []
    for rank in card["ranks"]:
        kv = rank.get("kv_cache") or {}
        model = rank.get("model_storage_at_ready") or {}
        cuda = rank.get("cuda_allocator_at_ready") or {}
        residual = rank.get("non_kv_torch_at_ready") or {}
        rows.append([process_label(rank), rank["host_id"], "yes" if rank.get("worker_ready") else "no", amount(model.get("cuda_storage_bytes")),
                     amount((kv.get("storage") or {}).get("cuda_storage_bytes")),
                     amount(residual.get("other_allocated_bytes_including_graphs")),
                     amount(cuda.get("allocated_bytes")), amount(cuda.get("reserved_bytes"))])
    lines += [table(["Rank", "Host", "Ready", "Model CUDA", "KV CUDA", "Other CUDA", "CUDA allocated", "CUDA reserved"], rows), "",
              "## CPU memory at readiness", "",
              "PSS apportions shared CPU pages between processes. PSS/RSS can omit GPU-backed pages. "
              "Free glibc heap is allocator-held space, not a promise that all of it can be released.", ""]
    rows = []
    for api, processes in ((False, card["ranks"]), (True, card["api_processes"])):
        for process in processes:
            cpu = process.get("cpu_at_ready") or {}
            heap = process.get("native_heap_at_ready") or {}
            rows.append([process_label(process, api), process["host_id"], amount(cpu.get("Pss")),
                         amount(cpu.get("Rss")), amount(heap.get("uordblks")), amount(heap.get("fordblks"))])
    lines += [table(["Process", "Host", "CPU PSS", "CPU RSS", "Used glibc heap", "Free glibc heap"], rows)]
    totals = []
    for host in card["hosts"]:
        processes = [process for process in card["ranks"] + card["api_processes"] if process["host_id"] == host["host_id"]]
        values = [(process.get("cpu_at_ready") or {}).get("Pss") for process in processes]
        total = sum(values) if values and all(number(value) for value in values) else None
        totals.append([host["host_id"], f"{sum(number(value) for value in values)}/{len(values)}", amount(total)])
    lines += ["", table(["Host", "Processes with ready PSS", "Estimated combined ready PSS"], totals), "",
              "Combined PSS sums the recorded rank/API readiness snapshots, which occur at different times. "
              "Other processes are excluded; an incomplete profile can also omit ranks or API processes."]
    for rank in card["ranks"]:
        kv = rank.get("kv_cache") or {}
        utilization = rank.get("utilization_check") or {}
        snapshot = utilization.get("snapshot") or {}
        graphs = rank.get("graphs") or {}
        lines += ["", f"## {cell(process_label(rank))} — {cell(rank['host_id'])}", "",
                  table(["Measurement", "Value"], [
                      ["Free memory at utilization check", amount(snapshot.get("free_memory"))],
                      ["Total memory at utilization check", amount(snapshot.get("total_memory"))],
                      ["gpu_memory_utilization", utilization.get("gpu_memory_utilization")],
                      ["Requested memory at check", amount(utilization.get("requested_memory_bytes"))],
                      ["KV budget", amount(rank.get("kv_budget_bytes"))],
                      ["KV equivalent capacity tokens", kv.get("equivalent_capacity_tokens")],
                      ["KV per 1,000 equivalent capacity tokens", amount(kv.get("effective_bytes_per_1000_capacity_tokens"), MIB, "MiB")],
                      ["KV CPU storage", amount((kv.get("storage") or {}).get("cpu_storage_bytes"))],
                      ["Graph profiling estimate", amount(graphs.get("profiling_estimate_bytes"))],
                  ]), "", "The KV capacity ratio depends on this configuration, especially for hybrid models; "
                  "it is not a universal marginal cost per user token."]
        if graphs.get("captures"):
            lines += ["", "Graph capture deltas (not isolated graph-object sizes):", "",
                      table(["Capture", "Reported free-memory delta", "CUDA allocated delta", "CUDA reserved delta"],
                            [[item.get("phase"), amount(item.get("reported_free_memory_delta_bytes")),
                              amount(item.get("torch_allocated_delta_bytes")), amount(item.get("torch_reserved_delta_bytes"))]
                             for item in graphs["captures"]])]
        if rank.get("failures"):
            lines += ["", f"**Worker failures:** {cell(rank['failures'])}"]
    lines += ["", "## Startup checkpoints", "",
              "Times are seconds since the profiler started on that host. Missing values are unavailable, not zero. "
              "CPU measurements start at each process's first hook; the API hook occurs late in startup."]
    for host in card["hosts"]:
        entries = []
        for api, processes in ((False, card["ranks"]), (True, card["api_processes"])):
            for process in processes:
                if process["host_id"] == host["host_id"]:
                    entries.extend((point["elapsed_seconds"], process_label(process, api), point)
                                   for point in startup_points(process, host))
        lines += ["", f"### Host {cell(host['host_id'])}", "",
                  table(["Seconds", "Process", "Phase", "CPU PSS", "CUDA allocated", "CUDA reserved", "Host used"],
                        [[f"{time:.3f}", label, point.get("phase", "unknown").replace("_", " "),
                          amount(point.get("cpu_pss_bytes")), amount(point.get("cuda_allocated_bytes")),
                          amount(point.get("cuda_reserved_bytes")), amount(point.get("host_unavailable_bytes"))]
                         for time, label, point in sorted(entries, key=lambda entry: (entry[0], entry[1]))])]
    return "\n".join(lines) + "\n"


def plot_card(card, output, *, relative=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ValueError("Plotting requires matplotlib; install it in your Python environment or use --no-plot") from error
    hosts = card["hosts"]
    if not hosts:
        raise ValueError("No hosts to plot; use --no-plot for a text-only report")
    fig, axes = plt.subplots(3 * len(hosts), 1, figsize=(12, 9 * len(hosts)), squeeze=False, layout="constrained")
    fig.suptitle(f"vLLM startup memory · {card.get('recipe') or card.get('run_id', '')} · {card.get('status', 'unknown')}\n"
                 + ("Relative growth from recorded baselines" if relative else "Recorded memory at startup checkpoints"), fontsize=15)
    colors = plt.get_cmap("tab10")
    try:
        for index, host in enumerate(hosts):
            host_ax, cpu_ax, gpu_ax = axes[index * 3:index * 3 + 3, 0]
            ranks = [rank for rank in card["ranks"] if rank["host_id"] == host["host_id"]]
            apis = [api for api in card["api_processes"] if api["host_id"] == host["host_id"]]
            baseline = (host.get("baseline_host_memory") or {}).get("unavailable_bytes")
            points = host_points(card, host)
            # Relative host RAM requires the actual pre-vLLM baseline. Never
            # silently substitute the first worker checkpoint for that baseline.
            if not relative or number(baseline):
                times, values = series(points, "host_unavailable_bytes", relative=relative, baseline=baseline)
                if times:
                    host_ax.plot(times, values, "o-", color="#5865a8", markersize=4, label="Host RAM unavailable")
            references = [("Sampled startup peak", host.get("startup_peak_unavailable_bytes"), "#ad5c24")]
            if not relative:
                references += [("Pre-vLLM baseline", baseline, "#738073")]
            for label, value, color in references:
                if number(value) and (not relative or number(baseline)):
                    host_ax.axhline((value - (baseline if relative else 0)) / GIB,
                                    color=color, linestyle=":", linewidth=1.2, label=label)
            host_ax.set_title(f"Host {host.get('hostname') or host['host_id']} · {host['host_id']} — system RAM", loc="left")
            for pos, process in enumerate(ranks + apis):
                is_api = pos >= len(ranks)
                label, color = process_label(process, is_api), colors(pos % 10)
                points = startup_points(process, host)
                times, values = series(points, "cpu_pss_bytes", relative=relative)
                if times and relative:
                    cpu_ax.plot(times, values, "o-", markersize=4, color=color, label=label)
                for field, style, suffix in (("cuda_allocated_bytes", "-", "allocated"), ("cuda_reserved_bytes", "--", "reserved")):
                    # Include API CUDA only if it actually allocated memory.
                    if is_api and not any(number(p.get(field)) and p[field] != 0 for p in points):
                        continue
                    times, values = series(points, field, relative=relative)
                    if times:
                        gpu_ax.plot(times, values, linestyle=style, marker="o", markersize=3,
                                    color=color, label=f"{label} {suffix}")
                        if field == "cuda_allocated_bytes":
                            for phase, marker in (("model_loaded", "D"), ("kv_allocated", "s"), ("worker_ready", "*")):
                                selected = [(t, v) for p, t, v in zip(points, times, values)
                                            if p.get("phase") == phase and math.isfinite(v)]
                                if selected:
                                    gpu_ax.scatter(*zip(*selected), marker=marker, s=75, color=color, edgecolors="white", linewidths=.7, zorder=4)
            if relative:
                cpu_ax.set_title("CPU PSS growth by process · separate measurement baselines", loc="left")
            else:
                plot_cpu_stack(cpu_ax, ranks + apis, host, len(ranks), colors)
            gpu_ax.set_title("GPU memory by process · PyTorch CUDA allocator", loc="left")
            end = max([point["elapsed_seconds"] for point in host_points(card, host)] +
                      [point["elapsed_seconds"] for process in ranks + apis for point in startup_points(process, host)] + [1])
            for ax in (host_ax, cpu_ax, gpu_ax):
                ax.set_xlim(0, max(1, end) * 1.03)
                ax.set_ylabel("Change (GiB)" if relative else "Memory (GiB)")
                ax.set_xlabel("Seconds since profiler start on this host")
                ax.grid(alpha=.18)
                ax.spines[["right", "top"]].set_visible(False)
                if ax.lines:
                    ax.legend(loc="upper left", fontsize=8, ncol=2)
                    if not relative:
                        ax.set_ylim(bottom=0)
                else:
                    ax.text(.5, .5, "No measurements available", transform=ax.transAxes, ha="center")
            if gpu_ax.lines:
                handles, _ = gpu_ax.get_legend_handles_labels()
                for marker, label in (("D", "Model loaded"), ("s", "KV allocated"), ("*", "Worker ready")):
                    handles.append(Line2D([], [], color="#555555", marker=marker, linestyle="None", label=label))
                gpu_ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
        cpu_note = ("CPU growth uses separate process baselines." if relative else
                    "CPU stack carries samples forward; the estimated total requires every recorded process.")
        fig.supxlabel(cpu_note + "\n"
                       "On unified memory systems, do not add host RAM and CUDA memory. No cross-host totals.", fontsize=9)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, facecolor="white")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="YAML (or JSON) profile card; raw traces are not required")
    parser.add_argument("--output", "-o", type=Path, help="Write Markdown here; omit for stdout")
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument("--plot", type=Path, help="PNG, SVG or PDF chart; defaults to OUTPUT.png with --output")
    plotting.add_argument("--no-plot", action="store_true", help="Text only, without matplotlib")
    parser.add_argument("--relative", action="store_true", help="Plot growth from recorded baselines (tables stay absolute)")
    args = parser.parse_args()
    plot = args.plot or (args.output.with_suffix(".png") if args.output and not args.no_plot else None)
    if plot and plot.suffix.lower() not in (".png", ".svg", ".pdf"):
        parser.error("--plot must end in .png, .svg or .pdf")
    paths = [path.resolve() for path in (args.card, args.output, plot) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("Input card, report and plot must have different paths")
    try:
        card = load_card(args.card)
        if plot:
            plot_card(card, plot, relative=args.relative)
        graph = None
        if plot:
            graph = quote(os.path.relpath(plot.resolve(), args.output.resolve().parent), safe="/.") if args.output else quote(str(plot.resolve()), safe="/.")
        report = markdown_report(card, graph, relative=args.relative)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
            print(f"Report: {args.output}" + (f"\nChart: {plot}" if plot else ""), file=sys.stderr)
        else:
            print(report, end="")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"memory-profile report: {error}\n")


if __name__ == "__main__":
    main()
