#!/usr/bin/env python3
"""Estimate startup RAM requirements and check a profiled Spark deployment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import yaml

from report import GIB, amount, cell, load_card, number, table

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT = 128 * 1024


class TopologyError(ValueError):
    """The configured cluster cannot supply the profiled topology."""


def positive(value):
    return number(value) and value > 0


def config(rank):
    return (rank.get('metadata') or {}).get('configuration') or {}


def profile_problems(card):
    """Fail closed: a failed or partially collected run cannot establish fit."""
    problems = []
    coverage = card.get('coverage') or {}
    keys = [r.get('rank_key') for r in card['ranks']]
    expected = coverage.get('expected_ranks', [])
    if (card.get('status') != 'startup_observed' or not keys or len(set(keys)) != len(keys)
            or set(keys) != set(expected) or not coverage.get('api_readiness_observed')
            or coverage.get('instrumentation_errors') or coverage.get('missing_ranks')
            or coverage.get('duplicate_ranks')):
        problems.append('A complete startup_observed card with every rank and API is required.')
    if not any(p.get('ready') for p in card['api_processes']):
        problems.append('No ready API process recorded.')
    head_ranks = [r for r in card['ranks'] if r.get('rank') == 0]
    if len(head_ranks) != 1 or not any(p.get('ready') and p['host_id'] == head_ranks[0]['host_id']
                                     for p in card['api_processes']):
        problems.append('Requires the ready API on the rank-zero head host.')
    for host in card['hosts']:
        ranks = [r for r in card['ranks'] if r['host_id'] == host['host_id']]
        if len(ranks) != 1:
            problems.append('Live estimates currently support one GPU rank per physical host.')
        if not host.get('sampling_covers_startup'):
            problems.append('Host sampling did not cover startup.')
        for key in ('startup_peak_increment_bytes', 'before_first_serving_kv_allocation_peak_increment_bytes',
                    'from_first_serving_kv_allocation_peak_increment_bytes'):
            if not number(host.get(key)) or host[key] < 0:
                problems.append(f'Missing or invalid host measurement: {key}.')
        baseline = host.get('baseline_host_memory') or {}
        if not positive(baseline.get('MemTotal')) or not positive(baseline.get('MemAvailable')):
            problems.append('Missing host baseline memory.')
    topologies = set()
    for rank in card['ranks']:
        cfg, hw = config(rank), (rank.get('metadata') or {}).get('hardware') or {}
        parallel = cfg.get('parallel_config') or {}
        topology = tuple(parallel.get(k, 1) for k in ('tensor_parallel_size', 'pipeline_parallel_size',
                                                    'data_parallel_size', 'decode_context_parallel_size',
                                                    'prefill_context_parallel_size'))
        topologies.add(topology)
        if (not all(positive(n) and int(n) == n for n in topology) or topology[2:] != (1, 1, 1)
                or topology[0] * topology[1] != len(card['ranks'])):
            problems.append('Requires a consistent TP/PP topology with DP=DCP=PCP=1.')
        if not rank.get('worker_ready') or rank.get('failures'):
            problems.append('A worker is not ready or has recorded failures.')
        if hw.get('integrated') is not True or not hw.get('name') or not hw.get('compute_capability'):
            problems.append('This estimator requires a native Linux unified-memory GPU profile.')
        model = cfg.get('model_config') or {}
        if not positive(model.get('max_model_len')):
            problems.append('Missing resolved maximum context length.')
        if (cfg.get('cache_config') or {}).get('kv_cache_memory_bytes') is not None:
            problems.append('Utilization minima require an automatic-KV profile; explicit KV bytes bypass sizing-cost measurement.')
        cache = rank.get('kv_cache') or {}
        storage = cache.get('storage') or {}
        if not positive(storage.get('cuda_storage_bytes')) or storage.get('cpu_storage_bytes') != 0:
            problems.append('Requires measured GPU KV storage without CPU KV offload.')
        utilization = rank.get('utilization_check') or {}
        snapshot = utilization.get('snapshot') or {}
        if (not positive(utilization.get('gpu_memory_utilization'))
                or utilization['gpu_memory_utilization'] > 1
                or not positive(snapshot.get('total_memory')) or not positive(snapshot.get('free_memory'))
                or not positive(utilization.get('requested_memory_bytes')) or not positive(rank.get('kv_budget_bytes'))):
            problems.append('Missing utilization admission or KV budget measurements.')
        elif rank['kv_budget_bytes'] > utilization['requested_memory_bytes']:
            problems.append('KV budget exceeds the recorded request; custom sizing is unsupported.')
        if not any(p.get('phase') == 'utilization_check' for p in rank.get('checkpoints', [])):
            problems.append('Missing utilization checkpoint.')
    if len(topologies) != 1:
        problems.append('Conflicting rank topologies.')
    return sorted(set(problems))


def cache_model(rank):
    """Calibrate pool-block demand from the recorded group-aware concurrency.

    Do not scale hybrid cache bytes/token as a straight line. Only ordinary
    FullAttentionSpec blocks shrink; retain the observed demand of other groups.
    Validate the shared-pool geometry before using that inference.
    """
    kv = rank.get('kv_cache') or {}
    blocks, concurrency = kv.get('num_blocks'), kv.get('max_concurrency')
    storage = (kv.get('storage') or {}).get('cuda_storage_bytes')
    groups = kv.get('group_layouts') or []
    if not all(positive(n) for n in (blocks, concurrency, storage)) or int(blocks) != blocks or not groups:
        raise ValueError('KV block/concurrency measurements are unavailable.')
    allowed = {'FullAttentionSpec', 'MambaSpec', 'SlidingWindowSpec'}
    for group in groups:
        if (group.get('spec_type') not in allowed or group.get('host_resident')
                or not all(positive(group.get(k)) and int(group[k]) == group[k]
                           for k in ('count', 'block_size', 'page_size_bytes', 'layers'))):
            raise ValueError('KV layout cannot be safely resized from this card.')
    pages = {g['page_size_bytes'] for g in groups}
    pool_bytes = storage / blocks
    if len(pages) != 1 or pool_bytes != max(g['layers'] for g in groups) * next(iter(pages)):
        raise ValueError('KV backing storage does not match a uniform shared pool.')
    per_request = blocks / concurrency
    if not math.isclose(per_request, round(per_request), abs_tol=1e-5, rel_tol=1e-8):
        raise ValueError('Recorded concurrency does not resolve to whole request blocks.')
    context = config(rank)['model_config']['max_model_len']
    full = [g for g in groups if g['spec_type'] == 'FullAttentionSpec']
    full_blocks = sum(g['count'] * math.ceil(context / g['block_size']) for g in full)
    residual = round(per_request) - full_blocks
    if residual < 0:
        raise ValueError('Group geometry disagrees with measured capacity.')
    # A null pool block and lookahead/alignment slack for each group. Keep the
    # profiled scheduling/graph overhead even when recommending one sequence.
    speculative = (config(rank).get('speculative_config') or {}).get('num_speculative_tokens') or 0
    slack = 1 + sum(g['count'] * (1 + math.ceil(speculative / g['block_size'])) for g in groups)
    return {'pool_bytes': int(pool_bytes), 'blocks': int(blocks), 'context': int(context),
            'full_groups': full, 'residual_blocks': residual, 'slack_blocks': slack,
            'minimum_context': max(1, speculative + 1)}


def kv_breakdown(model, context, sequences=1):
    if not model['minimum_context'] <= context <= model['context']:
        raise ValueError('Context is outside the profiled/speculative bounds; a new profile is needed.')
    if not isinstance(sequences, int) or isinstance(sequences, bool) or sequences < 1:
        raise ValueError('Sequence count must be a positive integer.')
    full = sum(
        g['count'] * math.ceil(context / g['block_size']) for g in model['full_groups'])
    # Null block is shared by the pool; state and lookahead space is per request.
    blocks = sequences * (model['residual_blocks'] + full + model['slack_blocks'] - 1) + 1
    return {'context_tokens': context, 'sequences': sequences,
            'full_attention_blocks_per_sequence': full,
            'retained_other_blocks_per_sequence': model['residual_blocks'],
            'slack_blocks_per_sequence': model['slack_blocks'] - 1, 'null_blocks': 1,
            'total_blocks': blocks, 'pool_bytes_per_block': model['pool_bytes'],
            'kv_bytes': blocks * model['pool_bytes']}


def kv_for_context(model, context, sequences=1):
    return kv_breakdown(model, context, sequences)['kv_bytes']


def shared_kv_budget(models, demands):
    # TP/PP uses the minimum block count across ranks. All stages must fit the
    # largest request block count, even when their bytes per pool block differ.
    blocks = max(math.ceil(demand / model['pool_bytes']) for model, demand in zip(models, demands))
    return blocks * max(model['pool_bytes'] for model in models)


def peak_requirement(host, old_kv, new_kv):
    # Never subtract serving KV from the loading/profiling peak.
    before = host['before_first_serving_kv_allocation_peak_increment_bytes']
    after = host['from_first_serving_kv_allocation_peak_increment_bytes']
    return max(before, after - old_kv + new_kv, 0)


def init_growth(host, rank):
    baseline = host['baseline_host_memory']['MemAvailable']
    return max(0, baseline - rank['utilization_check']['snapshot']['free_memory'])


def budget_overhead(rank):
    return rank['utilization_check']['requested_memory_bytes'] - rank['kv_budget_bytes']


def utilization_estimate(ranks, totals, kv_bytes):
    """Invert vLLM's automatic KV budget, not the sampled host RAM peak."""
    rows = []
    for rank, total, kv in zip(ranks, totals, kv_bytes):
        overhead = budget_overhead(rank)
        budget = overhead + kv
        rows.append({'rank_key': rank['rank_key'], 'device_total_bytes': total,
                     'inferred_non_kv_budget_bytes': overhead, 'kv_bytes': kv,
                     'required_vllm_budget_bytes': budget, 'minimum_utilization': budget / total})
    limiting = max(rows, key=lambda row: row['minimum_utilization'])
    minimum = limiting['minimum_utilization']
    return {'minimum': minimum, 'setting': math.ceil(minimum * 1000) / 1000,
            'limiting_rank': limiting['rank_key'], 'ranks': rows,
            'basis': 'automatic KV sizing with the profiled non-KV overhead; not a host-RAM cap'}


def tuned_utilization(ranks, totals, kv_bytes):
    return utilization_estimate(ranks, totals, [kv_bytes] * len(ranks))['setting']


def resized_requirement(host, rank, kv_bytes, reserve, request=None):
    old = rank['kv_cache']['storage']['cuda_storage_bytes']
    # Reusing the measured non-KV budget keeps admission consistent with a
    # smaller utilization setting; explicit KV bytes keep allocation predictable.
    if request is None:
        request = budget_overhead(rank) + kv_bytes
    return max(peak_requirement(host, old, kv_bytes), init_growth(host, rank) + request) + reserve


def analyze(card, context=DEFAULT_CONTEXT, reserve=4 * GIB):
    result = {'recipe': card.get('recipe'), 'run_id': card.get('run_id'), 'target_context': context,
              'reserve_bytes': reserve, 'problems': profile_problems(card), 'hosts': []}
    if result['problems']:
        return result
    ranks = sorted(card['ranks'], key=lambda r: r['rank'])
    result['configuration'] = config(ranks[0])
    totals = [r['utilization_check']['snapshot']['total_memory'] for r in ranks]
    result['utilization_estimates'] = {
        'profiled_cache': utilization_estimate(ranks, totals, [r['kv_cache']['storage']['cuda_storage_bytes'] for r in ranks])}
    for rank in ranks:
        host = next(h for h in card['hosts'] if h['host_id'] == rank['host_id'])
        baseline = host['baseline_host_memory']['MemTotal'] - host['baseline_host_memory']['MemAvailable']
        original = max(host['startup_peak_increment_bytes'], init_growth(host, rank)
                       + rank['utilization_check']['requested_memory_bytes']) + reserve
        row = {'host_id': host['host_id'], 'hostname': host.get('hostname'), 'rank_key': rank['rank_key'],
               'hardware': rank['metadata']['hardware'], 'baseline_bytes': baseline,
               'profiled_available_bytes': original, 'profiled_total_bytes': baseline + original,
               'profiled_kv_bytes': rank['kv_cache']['storage']['cuda_storage_bytes']}
        try:
            model = cache_model(rank)
            row['cache_model'] = model
            row['target_kv_breakdown'] = kv_breakdown(model, context)
            row['target_kv_bytes'] = row['target_kv_breakdown']['kv_bytes']
        except ValueError as error:
            row['resize_error'] = str(error)
        result['hosts'].append(row)
    # vLLM's explicit KV byte option is one common per-rank value.
    if all('target_kv_bytes' in row for row in result['hosts']):
        budget = shared_kv_budget([row['cache_model'] for row in result['hosts']],
                                 [row['target_kv_bytes'] for row in result['hosts']])
        result['target_kv_budget_bytes'] = budget
        estimate = utilization_estimate(ranks, totals, [budget] * len(ranks))
        result['utilization_estimates']['target_context'] = estimate
        util = estimate['setting']
        result['target_gpu_memory_utilization'] = util
        for row, rank in zip(result['hosts'], ranks):
            host = next(h for h in card['hosts'] if h['host_id'] == row['host_id'])
            request = math.ceil(util * rank['utilization_check']['snapshot']['total_memory'])
            row['target_available_bytes'] = resized_requirement(host, rank, budget, reserve, request)
            row['target_total_bytes'] = row['baseline_bytes'] + row['target_available_bytes']
            row['target_host_memory_breakdown'] = {
                'background_bytes': row['baseline_bytes'],
                'pre_kv_startup_peak_increment_bytes': host['before_first_serving_kv_allocation_peak_increment_bytes'],
                'post_kv_startup_peak_increment_bytes': host['from_first_serving_kv_allocation_peak_increment_bytes'],
                'removed_profiled_kv_bytes': row['profiled_kv_bytes'], 'added_target_kv_bytes': budget,
                'adjusted_startup_peak_increment_bytes': peak_requirement(host, row['profiled_kv_bytes'], budget),
                'initial_gate_available_bytes': init_growth(host, rank) + request,
                'reserve_bytes': reserve,
            }
    sequences = (result['configuration'].get('scheduler_config') or {}).get('max_num_seqs')
    if isinstance(sequences, int) and sequences > 0 and all('cache_model' in row for row in result['hosts']):
        models = [row['cache_model'] for row in result['hosts']]
        budget = shared_kv_budget(models, [kv_for_context(model, model['context'], sequences) for model in models])
        result['full_concurrency_kv_budget_bytes'] = budget
        result['utilization_estimates']['full_concurrency'] = utilization_estimate(ranks, totals, [budget] * len(ranks))
    return result


def select_hosts(count, config_path, explicit=None):
    if explicit:
        if len(explicit) != count or explicit[0] != 'local' or len(set(explicit)) != count:
            raise ValueError('--host must give local first, then exactly one distinct host per remaining rank.')
        return explicit
    if count == 1:
        return ['local']
    spec = importlib.util.spec_from_file_location('memory_capacity_runner', ROOT / 'run-recipe.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ENV_FILE = config_path
    env = runner.load_env_file()
    nodes = runner.parse_nodes(env.get('CLUSTER_NODES'))
    if len(nodes) < count:
        raise TopologyError(f'Profile requires {count} nodes; only {len(nodes)} configured. Topology cannot be reduced from this card.')
    if not env.get('LOCAL_IP') or nodes[0] != env['LOCAL_IP']:
        raise ValueError('Saved configuration must identify this head with LOCAL_IP as the first CLUSTER_NODES entry.')
    # Do not mistake running this command on a worker for running on the head.
    try:
        addresses = json.loads(subprocess.run(['ip', '-j', '-4', 'address', 'show'], check=True,
                                             capture_output=True, text=True, timeout=5).stdout)
        local = {a['local'] for link in addresses for a in link.get('addr_info', [])}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError('Cannot verify the configured head address; use explicit --host mappings.') from error
    if nodes[0] not in local:
        raise ValueError('Run this check on the configured head node.')
    selected = ['local', *nodes[1:count]]
    if len(set(nodes[:count])) != count:
        raise ValueError('Duplicate configured nodes.')
    return selected


def probe_host(host, timeout=20):
    source = Path(__file__).with_name('host_probe.py').read_text()
    command = [sys.executable, '-']
    if host != 'local':
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:%+-]*', host):
            return {'error': 'Invalid SSH host identifier.'}
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '--', host, 'python3 -']
    try:
        process = subprocess.run(command, input=source, capture_output=True, text=True, timeout=timeout)
        if process.returncode:
            return {'error': 'Host probe failed (check Python, driver and SSH access).'}
        data = json.loads(process.stdout)
        if not isinstance(data, dict):
            raise ValueError('Invalid probe result')
        return data
    except (OSError, ValueError, subprocess.SubprocessError):
        return {'error': 'Host probe unavailable or timed out.'}


def check_cluster(card, analysis, inventories):
    """Evaluate each host independently; spare RAM on a peer cannot compensate."""
    result = {'status': 'unknown', 'hosts': []}
    if analysis['problems']:
        result['reason'] = 'Profile is incomplete or unsupported.'
        return result
    if len(inventories) != len(analysis['hosts']):
        result['reason'] = 'The profiled number of hosts is unavailable.'
        return result
    ids = [h.get('host_id') for h in inventories if h.get('host_id')]
    duplicates = len(ids) != len(set(ids))
    ranks = sorted(card['ranks'], key=lambda r: r['rank'])
    compatible = []
    for row, rank, live in zip(analysis['hosts'], ranks, inventories):
        entry = {'rank_key': rank['rank_key'], 'hostname': live.get('hostname', row['hostname'])}
        result['hosts'].append(entry)
        hardware = rank['metadata']['hardware']
        gpus, memory = live.get('gpus') or [], live.get('memory') or {}
        error = live.get('error') or live.get('gpu_error')
        if not live.get('host_id'):
            error = error or 'Physical host identity unavailable.'
        elif duplicates:
            error = 'Two target entries refer to the same physical host.'
        elif live.get('wsl'):
            error = 'WSL memory accounting is unsupported.'
        elif not all(positive(memory.get(k)) for k in ('MemTotal', 'MemAvailable')):
            error = error or 'Host memory measurements unavailable.'
        elif memory['MemAvailable'] > memory['MemTotal']:
            error = 'Available memory exceeds total host memory.'
        elif len(gpus) != 1:
            error = error or 'Exactly one GPU per host is required by this checker.'
        elif any(gpus[0].get(k) != hardware.get(k) for k in ('name', 'compute_capability', 'integrated')):
            error = 'GPU differs from the profiled hardware; reprofile for a fit verdict.'
        elif abs(gpus[0].get('total_memory_bytes', 0) - memory['MemTotal']) > memory['MemTotal'] * .05:
            error = 'CUDA and host totals do not agree with native UMA accounting.'
        if error:
            entry.update(status='unknown', reason=error)
        else:
            compatible.append((row, rank, live, entry))
    if len(compatible) != len(ranks):
        result['reason'] = 'At least one host could not be evaluated.'
        return result
    reserve = analysis['reserve_bytes']
    totals = [live['memory']['MemTotal'] for _, _, live, _ in compatible]
    target_budget = analysis.get('target_kv_budget_bytes')
    estimates = {'profiled_cache': utilization_estimate(ranks, totals, [row['profiled_kv_bytes'] for row in analysis['hosts']])}
    if target_budget is not None:
        estimates['target_context'] = utilization_estimate(ranks, totals, [target_budget] * len(ranks))
    if analysis.get('full_concurrency_kv_budget_bytes') is not None:
        estimates['full_concurrency'] = utilization_estimate(
            ranks, totals, [analysis['full_concurrency_kv_budget_bytes']] * len(ranks))
    result['utilization_estimates'] = estimates
    target_util = estimates['target_context']['setting'] if 'target_context' in estimates else None
    result['target_gpu_memory_utilization'] = target_util
    budgets = []
    for row, rank, live, entry in compatible:
        total = live['memory']['MemTotal']
        cache = config(rank)['cache_config']
        request = math.ceil(total * rank['utilization_check']['gpu_memory_utilization'])
        budget = cache.get('kv_cache_memory_bytes')
        if budget is None:
            budget = request - budget_overhead(rank)
        budgets.append(budget)
        entry.update(available_bytes=live['memory']['MemAvailable'], requested_bytes=request)
    # Native TP/PP uses the minimum number of cache blocks across workers.
    resize_known = all('cache_model' in row for row in analysis['hosts'])
    blocks = min(math.floor(b / row['cache_model']['pool_bytes']) for b, row in zip(budgets, analysis['hosts'])) if resize_known else None
    for row, rank, live, entry in compatible:
        host = next(h for h in card['hosts'] if h['host_id'] == row['host_id'])
        available = entry['available_bytes']
        old_kv = row['profiled_kv_bytes']
        # Without pool geometry, only the unchanged physical-memory size can
        # reuse a measured automatic allocation.
        if blocks is None:
            same_total = live['memory']['MemTotal'] == rank['utilization_check']['snapshot']['total_memory']
            new_kv = old_kv if same_total else None
        else:
            new_kv = max(0, blocks) * row['cache_model']['pool_bytes']
        if new_kv is None:
            entry.update(status='unknown', reason='Automatic KV resizing needs a supported cache layout.')
        else:
            needed = max(peak_requirement(host, old_kv, new_kv), init_growth(host, rank) + entry['requested_bytes']) + reserve
            entry['profiled_required_bytes'] = needed
            entry['estimated_automatic_kv_bytes'] = new_kv
            capacity_ok = new_kv >= old_kv
            entry['status'] = 'fits' if available >= needed and capacity_ok else 'does_not_fit'
            if not capacity_ok:
                entry['reason'] = 'The original utilization setting would allocate less KV than the profile.'
            elif available < needed:
                entry['reason'] = 'Insufficient available RAM for startup and the utilization gate, including reserve.'
        target = resized_requirement(host, rank, target_budget, reserve, math.ceil(target_util * live['memory']['MemTotal'])) if target_util is not None else None
        entry['target_required_bytes'] = target
        entry['target_status'] = ('fits' if available >= target and target_util <= 1 else 'does_not_fit') if target is not None else 'unknown'
        floor = peak_requirement(host, old_kv, 0) + reserve
        entry['non_kv_floor_bytes'] = floor
        entry['any_kv_status'] = 'unknown'
        if available <= floor:
            entry['any_kv_status'] = 'does_not_fit'
        if 'cache_model' in row:
            model = row['cache_model']
            minimum = model['minimum_context']
            demand = kv_for_context(model, minimum)
            # A common budget must fit all stages, checked below.
            entry['minimum_kv_budget_bytes'] = demand
    if resize_known:
        minimum_context = max(r['cache_model']['minimum_context'] for r in analysis['hosts'])
        models = [r['cache_model'] for r in analysis['hosts']]
        minimum_budget = shared_kv_budget(models, [kv_for_context(m, minimum_context) for m in models])
        minimum_util = tuned_utilization(ranks, totals, minimum_budget)
        result['minimum_kv_budget_bytes'] = minimum_budget
        result['minimum_gpu_memory_utilization'] = minimum_util
        result['minimum_context'] = minimum_context
        for row, rank, live, entry in compatible:
            host = next(h for h in card['hosts'] if h['host_id'] == row['host_id'])
            required = resized_requirement(host, rank, minimum_budget, reserve, math.ceil(minimum_util * live['memory']['MemTotal']))
            if entry['available_bytes'] >= required and minimum_util <= 1:
                entry['any_kv_status'] = 'fits'
            # Between the startup floor and the conservative small-cache bound,
            # do not claim no KV can fit: unrecorded window/state details matter.
    def aggregate(key):
        states = [e.get(key, 'unknown') for e in result['hosts']]
        return 'does_not_fit' if 'does_not_fit' in states else 'unknown' if 'unknown' in states else 'fits'
    result.update(status=aggregate('status'), target_status=aggregate('target_status'), any_kv_status=aggregate('any_kv_status'))
    return result


def console_cell(value):
    if value is None:
        return 'unavailable'
    return ''.join(char if char.isprintable() else ' ' for char in str(value))


def console_table(headers, rows):
    values = [[console_cell(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[index]) for row in values) for index in range(len(headers))]
    values.insert(1, ['-' * width for width in widths])
    return '\n'.join('  '.join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
                     for row in values)


def amount_range(values):
    low, high = min(values), max(values)
    return amount(low) if low == high else f'{low / GIB:.3f} .. {high / GIB:.3f} GiB'


def render(card, analysis, live=None, *, console=False):
    clean = console_cell if console else cell
    grid = console_table if console else table
    title = f"Memory capacity: {clean(card.get('recipe') or card.get('model'))}"
    lines = [title if console else '# ' + title, '']
    if analysis['problems']:
        verdict = 'Verdict: UNKNOWN' if console else '**Verdict: UNKNOWN**'
        return '\n'.join(lines + [verdict, '', *['- ' + clean(p) for p in analysis['problems']]]) + '\n'
    cfg = analysis['configuration']
    p, s = cfg['parallel_config'], cfg['scheduler_config']
    context = cfg['model_config']['max_model_len']
    lines += [f"Profile: {len(analysis['hosts'])} host(s), TP={p['tensor_parallel_size']}, PP={p['pipeline_parallel_size']}; "
              f"context {context:,} tokens, max sequences {s.get('max_num_seqs')}, batch tokens {s.get('max_num_batched_tokens')}.",
              f"GPU: {clean(analysis['hosts'][0]['hardware']['name'])}; KV dtype: {clean(cfg['cache_config'].get('cache_dtype'))}; "
              f"speculation: {clean((cfg.get('speculative_config') or {}).get('method') or 'none')}.", '',
              f"RAM estimates per host, including {amount(analysis['reserve_bytes'])} reserve. "
              "Required available RAM is the estimated capacity needed before starting the model. "
              "Total RAM adds the profile's existing OS/background usage to that requirement.",
              "Available now, shown in a live check, is Linux MemAvailable: an estimate of RAM that can be "
              "used without swapping, including unused pages and reclaimable portions of caches. "
              "MemFree counts only unused pages. Swap is not included.",
              "Keeping the same cache on a different RAM size requires retuning utilization or setting explicit KV bytes.", '',
              grid(['Profile host / rank', 'Profiled KV', 'Profiled: required available / total',
                     f"{analysis['target_context']:,} tokens: required available / total"],
                    [[f"{r['hostname']} / {r['rank_key']}", amount(r['profiled_kv_bytes']),
                      f"{amount(r['profiled_available_bytes'])} / {amount(r['profiled_total_bytes'])}",
                      f"{amount(r.get('target_available_bytes'))} / {amount(r.get('target_total_bytes'))}"] for r in analysis['hosts']]), '']
    estimates = (live or {}).get('utilization_estimates') or analysis['utilization_estimates']
    basis = 'checked hosts' if (live or {}).get('utilization_estimates') else 'profiled GPUs'
    names = {'profiled_cache': f"Recorded KV / max-num-seqs={s.get('max_num_seqs')}",
             'full_concurrency': f"{s.get('max_num_seqs')} full {context:,}-token sequences",
             'target_context': f"1 x {analysis['target_context']:,}-token sequence"}
    lines += [f"Automatic KV sizing: minimum utilization estimates on {basis} "
              f"(device total {amount_range([r['device_total_bytes'] for r in estimates['profiled_cache']['ranks']])} per GPU).",
              'These reuse the profiled non-KV sizing cost; lowering sequence count may change that cost. '
              'The setting column rounds the estimate upward to 0.001.', '',
              grid(['Configuration', 'KV budget / rank', 'vLLM sizing budget / rank', 'Min util', 'Setting'],
                   [[names[key], amount_range([r['kv_bytes'] for r in estimates[key]['ranks']]),
                     amount_range([r['required_vllm_budget_bytes'] for r in estimates[key]['ranks']]),
                     f"{estimates[key]['minimum']:.6f}",
                     f"{estimates[key]['setting']:.3f}" if estimates[key]['setting'] <= 1 else 'unavailable (>1)']
                    for key in ('profiled_cache', 'full_concurrency', 'target_context') if key in estimates]), '',
              'Sizing cost = original requested bytes - measured KV budget. '
              'Min util = max across ranks of (sizing cost + target KV) / device total. '
              'Host RAM also includes startup peaks, background/API memory and the reserve; '
              'gpu-memory-utilization is not a cap on total host RAM.', '']
    concurrency = [(r.get('kv_cache') or {}).get('max_concurrency') for r in card['ranks']]
    if all(positive(value) for value in concurrency):
        lines += [f"Recorded cache capacity: about {min(concurrency):.2f} full {context:,}-token sequences. "
                  f"max-num-seqs={s.get('max_num_seqs')} is a scheduling limit, not a promise that all sequences fit at full context.", '']
    budget = analysis.get('target_kv_budget_bytes')
    if budget is not None:
        target = estimates['target_context']
        util = target['setting']
        limiting = next(r for r in target['ranks'] if r['rank_key'] == target['limiting_rank'])
        lines += [f"Target calculation ({clean(limiting['rank_key'])}): "
                  f"({amount(limiting['inferred_non_kv_budget_bytes'])} sizing cost + {amount(limiting['kv_bytes'])} KV) "
                  f"/ {amount(limiting['device_total_bytes'])} = {target['minimum']:.6f}.", '',
                  f"Fixed-cache configuration for one {analysis['target_context']:,}-token sequence "
                  f"(same TP/PP and hardware; {amount(budget)} KV budget per rank):", '',
                  *([] if console else ['```text']), f"--max-model-len {analysis['target_context']} --max-num-seqs 1",
                  f"--kv-cache-memory-bytes {budget}" + (f" --gpu-memory-utilization {util:.3f}" if util <= 1 else ''),
                  '' if console else '```',
                  'With explicit KV bytes, utilization does not size the cache; in this profiled worker '
                  'it still controls the initial admission gate. The number above is an automatic-sizing '
                  'equivalent, not a minimum needed for this explicit-KV command. To use automatic sizing, '
                  'omit --kv-cache-memory-bytes and use the listed utilization estimate; '
                  'rounding and new profiling measurements may change the allocated KV size.', '']
        if util > 1:
            lines += ['No valid utilization setting covers this target on the profiled RAM size; '
                      'it requires more RAM or a new profile with lower startup overhead.', '']
    for row in analysis['hosts']:
        if row.get('resize_error'):
            lines += [f"Resize estimate unavailable for {clean(row['rank_key'])}: {clean(row['resize_error'])}"]
    if live is not None:
        labels = {'fits': 'LIKELY FITS', 'does_not_fit': 'DOES NOT FIT', 'unknown': 'UNKNOWN'}
        verdict = f"Current cluster {'-' if console else '—'} profiled cache/settings: {labels[live['status']]}"
        lines += [verdict if console else f"**{verdict}**",
                  f"{analysis['target_context']:,} tokens, reduced KV: {labels.get(live.get('target_status'), 'UNKNOWN')}. "
                  f"Any usable KV with smaller context: {labels.get(live.get('any_kv_status'), 'UNKNOWN')}.", '']
        if live.get('reason'):
            lines += [live['reason'], '']
        if live.get('any_kv_status') == 'fits' and live.get('status') != 'fits':
            lines += [f"Small-cache fallback estimate: --max-model-len {live['minimum_context']} --max-num-seqs 1 "
                      f"--kv-cache-memory-bytes {live['minimum_kv_budget_bytes']} "
                      f"--gpu-memory-utilization {live['minimum_gpu_memory_utilization']:.3f}. "
                      'This demonstrates room for usable KV, not a useful long-context configuration.', '']
        if live['hosts']:
            lines += [grid(['Current host / rank', 'Available now (MemAvailable)', 'Required available at startup', 'Result'],
                            [[f"{r['hostname']} / {r['rank_key']}", amount(r.get('available_bytes')),
                              amount(r.get('profiled_required_bytes')), labels.get(r.get('status'), 'UNKNOWN')
                              + (': ' + r['reason'] if r.get('reason') else '')] for r in live['hosts']]), '']
    lines += ['Estimates retain profiled loading/warmup overhead. Reduced caches keep the measured non-full-attention '
              'state/window cost and add block/alignment slack; they are conservative estimates, not exact minima. '
              'A TP profile cannot establish a smaller topology.',
              'Memory fit assumes the same model revision, image, kernels and other settings. It does not validate '
              'model files, network transport, software compatibility or maximum serving load. '
              'Existing workloads are not subtracted from current usage. Swap is not counted. '
              'CPU and GPU share RAM; their counters are not added.']
    if console:
        # Keep aligned tables and command flags intact; wrap narrative text for
        # stdout/pagers without terminal detection, colors or new dependencies.
        lines = [line if '\n' in line else '  ' + line if line.startswith('--') else
                 textwrap.fill(line, width=100, break_long_words=False, break_on_hyphens=False)
                 for line in lines]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('card', type=Path)
    parser.add_argument('--context', type=int, default=DEFAULT_CONTEXT, help='Reduced-cache target, default 131072 (128K)')
    parser.add_argument('--reserve-gib', type=float, default=4, help='Extra RAM reserve per host, default 4 GiB')
    parser.add_argument('--check-host', action='store_true', help='Check this head and saved cluster peers; solo profiles check only local')
    parser.add_argument('--config', type=Path, default=ROOT / '.env')
    parser.add_argument('--host', action='append', help='Explicit host mapping: local first, then SSH peers in rank order')
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument('--format', choices=('markdown', 'console', 'json'), default='markdown',
                         help='Output format (default: markdown); console prints plain aligned text')
    formats.add_argument('--json', dest='format', action='store_const', const='json',
                         help='Alias for --format json')
    parser.add_argument('--output', '-o', type=Path)
    args = parser.parse_args()
    if args.context < 1 or not number(args.reserve_gib) or args.reserve_gib < 0:
        parser.error('Context must be positive and reserve must be finite and nonnegative.')
    if args.host and not args.check_host:
        parser.error('--host requires --check-host')
    try:
        card = load_card(args.card)
        analysis = analyze(card, args.context, math.ceil(args.reserve_gib * GIB))
        live = None
        if args.check_host:
            try:
                if analysis['problems']:
                    raise ValueError('Incomplete or unsupported profile; host probing skipped.')
                hosts = select_hosts(len(analysis['hosts']), args.config, args.host)
                with ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
                    inventories = list(pool.map(probe_host, hosts))
                live = check_cluster(card, analysis, inventories)
            except TopologyError as error:
                live = {'status': 'does_not_fit', 'target_status': 'does_not_fit', 'any_kv_status': 'does_not_fit',
                        'reason': str(error), 'hosts': []}
            except ValueError as error:
                live = {'status': 'unknown', 'reason': str(error), 'hosts': []}
        output = (json.dumps({'requirements': analysis, 'live': live}, indent=2) + '\n'
                  if args.format == 'json' else render(card, analysis, live, console=args.format == 'console'))
        if args.output:
            args.output.write_text(output)
        else:
            print(output, end='')
        return 2 if analysis['problems'] or (live and live['status'] == 'unknown') else 1 if live and live['status'] == 'does_not_fit' else 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(2, f'memory-capacity: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
