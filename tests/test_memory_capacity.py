#!/usr/bin/env python3
"""CPU-only capacity, admission, topology and fail-closed regression tests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'mods/memory-profile'
sys.path.insert(0, str(MOD))
import capacity as C

G = C.GIB


def fixture(count=2):
    hosts, ranks = [], []
    for i in range(count):
        cfg = {
            'model_config': {'model': 'example/model', 'max_model_len': 262144},
            'parallel_config': {'tensor_parallel_size': count, 'pipeline_parallel_size': 1,
                                'data_parallel_size': 1, 'decode_context_parallel_size': 1,
                                'prefill_context_parallel_size': 1},
            'cache_config': {'gpu_memory_utilization': .6, 'kv_cache_memory_bytes': None, 'cache_dtype': 'fp8'},
            'scheduler_config': {'max_num_seqs': 8, 'max_num_batched_tokens': 4096},
            'speculative_config': {'num_speculative_tokens': 8, 'method': 'test'},
        }
        hosts.append({'host_id': f'h{i}', 'hostname': f'node{i}', 'sampling_covers_startup': True,
                      'baseline_host_memory': {'MemTotal': 128*G, 'MemAvailable': 123*G},
                      'startup_peak_increment_bytes': 90*G,
                      'before_first_serving_kv_allocation_peak_increment_bytes': 35*G,
                      'from_first_serving_kv_allocation_peak_increment_bytes': 90*G})
        ranks.append({'rank_key': f'dp0/rank{i}', 'rank': i, 'host_id': f'h{i}', 'worker_ready': True,
                      'metadata': {'configuration': cfg, 'hardware': {'name': 'NVIDIA GB10', 'compute_capability': [12, 1],
                                                                    'integrated': True, 'total_memory_bytes': 128*G}},
                      'checkpoints': [{'phase': 'utilization_check', 'elapsed_seconds': 5}],
                      'utilization_check': {'snapshot': {'total_memory': 128*G, 'free_memory': 118*G},
                                            'gpu_memory_utilization': .6, 'requested_memory_bytes': .6*128*G},
                      'kv_budget_bytes': 40*G,
                      'kv_cache': {'num_blocks': 320, 'max_concurrency': 20,
                                   'storage': {'cuda_storage_bytes': 40*G, 'cpu_storage_bytes': 0},
                                   'group_layouts': [{'count': 1, 'block_size': 32768, 'page_size_bytes': G//8,
                                                     'layers': 1, 'spec_type': name, 'host_resident': False}
                                                    for name in ['FullAttentionSpec', 'MambaSpec']]}})
    return {'profile_schema': 'spark-vllm-memory-profile/v1', 'status': 'startup_observed',
            'recipe': 'example', 'model': 'example/model', 'run_id': 'test', 'hosts': hosts, 'ranks': ranks,
            'api_processes': [{'host_id': 'h0', 'ready': True}],
            'coverage': {'expected_ranks': [r['rank_key'] for r in ranks], 'api_readiness_observed': True,
                         'instrumentation_errors': []}}


def inventory(card, available=110*G):
    return [{'host_id': h['host_id'], 'hostname': h['hostname'], 'wsl': False,
             'gpus': [copy.deepcopy(r['metadata']['hardware'])],
             'memory': {'MemTotal': 128*G, 'MemAvailable': available}}
            for h, r in zip(card['hosts'], card['ranks'])]


class RequirementsTests(unittest.TestCase):
    def test_profiled_and_128k_requirements_include_background_and_reserve(self):
        result = C.analyze(fixture())
        self.assertFalse(result['problems'])
        row = result['hosts'][0]
        self.assertEqual(row['profiled_available_bytes'], 94*G)
        self.assertEqual(row['profiled_total_bytes'], 99*G)
        self.assertEqual(result['target_kv_budget_bytes'], 17*G//8)
        self.assertEqual(row['target_available_bytes'], (50+4)*G + 17*G//8)

    def test_hybrid_fixed_cost_is_not_halved_with_context(self):
        model = C.cache_model(fixture()['ranks'][0])
        self.assertEqual(model['residual_blocks'], 8)
        full = C.kv_for_context(model, 262144)
        half = C.kv_for_context(model, 131072)
        self.assertGreater(half, full / 2)
        self.assertEqual(full - half, 4*G//8)

    def test_full_attention_layout_resizes_in_whole_blocks(self):
        rank = fixture(1)['ranks'][0]
        rank['kv_cache']['group_layouts'].pop()
        rank['kv_cache']['max_concurrency'] = 40
        model = C.cache_model(rank)
        self.assertEqual(model['residual_blocks'], 0)
        self.assertEqual(C.kv_for_context(model, 131072), 7*G//8)
        self.assertEqual(C.kv_for_context(model, 131073), 8*G//8)

    def test_sequence_count_scales_requests_but_not_the_shared_null_block(self):
        model = C.cache_model(fixture()['ranks'][0])
        one = C.kv_breakdown(model, 131072)
        eight = C.kv_breakdown(model, 131072, 8)
        self.assertEqual(eight['total_blocks'], 8 * (one['total_blocks'] - 1) + 1)
        self.assertEqual(eight['sequences'], 8)

    def test_utilization_is_sizing_budget_not_whole_host_peak(self):
        card = fixture()
        analysis = C.analyze(card)
        estimates = analysis['utilization_estimates']
        self.assertAlmostEqual(estimates['profiled_cache']['minimum'], .6)
        target = estimates['target_context']
        self.assertAlmostEqual(target['minimum'], (.6*128 - 40 + 17/8) / 128)
        self.assertLess(target['minimum'], analysis['hosts'][0]['target_total_bytes'] / (128*G))
        # Extra OS usage, startup transient memory and user reserve must affect
        # host requirements without masquerading as automatic KV-sizing costs.
        card['hosts'][0]['baseline_host_memory']['MemAvailable'] -= 10*G
        card['hosts'][0]['from_first_serving_kv_allocation_peak_increment_bytes'] += 10*G
        larger = C.analyze(card, reserve=8*G)
        self.assertEqual(larger['utilization_estimates'], estimates)
        self.assertGreater(larger['hosts'][0]['target_total_bytes'], analysis['hosts'][0]['target_total_bytes'])

    def test_measured_qwen_pool_math_and_utilization(self):
        # Recorded numeric geometry, independent of local profile artifacts.
        card = fixture()
        for rank, budget in zip(card['ranks'], [46569391514, 46370448794]):
            rank['utilization_check']['requested_memory_bytes'] = 71864741274
            rank['utilization_check']['snapshot']['total_memory'] = 130663165952
            rank['kv_budget_bytes'] = budget
            rank['kv_cache'].update(num_blocks=5495, max_concurrency=7.201834862385321,
                                   storage={'cuda_storage_bytes': 46365491200, 'cpu_storage_bytes': 0},
                                   group_layouts=[{'count': count, 'layers': layers, 'block_size': 1648,
                                                   'page_size_bytes': 1687552, 'spec_type': spec}
                                                  for count, layers, spec in [(4, 4, 'FullAttentionSpec'),
                                                                              (2, 4, 'MambaSpec'),
                                                                              (8, 5, 'MambaSpec'),
                                                                              (1, 5, 'SlidingWindowSpec')]])
        result = C.analyze(card)
        detail = result['hosts'][0]['target_kv_breakdown']
        self.assertEqual(detail['full_attention_blocks_per_sequence'], 320)
        self.assertEqual(detail['retained_other_blocks_per_sequence'], 123)
        self.assertEqual(detail['slack_blocks_per_sequence'] + detail['null_blocks'], 31)
        self.assertEqual(detail['total_blocks'], 474)
        self.assertEqual(result['target_kv_budget_bytes'], 3999498240)
        estimates = result['utilization_estimates']
        self.assertAlmostEqual(estimates['profiled_cache']['minimum'], .5499620582161477)
        self.assertEqual(estimates['profiled_cache']['setting'], .55)
        self.assertAlmostEqual(estimates['target_context']['minimum'], .2257238335311326)
        self.assertEqual(estimates['target_context']['setting'], .226)
        self.assertEqual(estimates['target_context']['limiting_rank'], 'dp0/rank1')
        self.assertEqual(estimates['full_concurrency']['setting'], .605)

    def test_pp_shared_budget_covers_largest_block_demand_on_every_rank(self):
        card = fixture()
        for rank in card['ranks']:
            cfg = rank['metadata']['configuration']['parallel_config']
            cfg.update(tensor_parallel_size=1, pipeline_parallel_size=2)
        smaller = card['ranks'][1]['kv_cache']
        smaller['storage']['cuda_storage_bytes'] = 20*G
        smaller['max_concurrency'] = 10
        for group in smaller['group_layouts']:
            group.update(page_size_bytes=G//16, count=2)
        result = C.analyze(card)
        self.assertFalse(result['problems'])
        budget = result['target_kv_budget_bytes']
        self.assertEqual(budget, 33*G//8)
        pool_blocks = min(budget // row['cache_model']['pool_bytes'] for row in result['hosts'])
        self.assertTrue(all(pool_blocks >= row['target_kv_breakdown']['total_blocks'] for row in result['hosts']))

    def test_explicit_kv_profile_cannot_infer_automatic_sizing_cost(self):
        card = fixture()
        for rank in card['ranks']:
            rank['metadata']['configuration']['cache_config']['kv_cache_memory_bytes'] = 40*G
        result = C.analyze(card)
        self.assertTrue(any('automatic-KV profile' in problem for problem in result['problems']))

    def test_pre_kv_peak_is_never_reduced(self):
        card = fixture()
        card['hosts'][0]['before_first_serving_kv_allocation_peak_increment_bytes'] = 89*G
        self.assertEqual(C.analyze(card)['hosts'][0]['target_available_bytes'], 93*G)

    def test_context_above_profile_limit_has_no_resize_estimate(self):
        result = C.analyze(fixture(), context=524288)
        self.assertNotIn('target_kv_budget_bytes', result)
        self.assertIn('resize_error', result['hosts'][0])

    def test_unknown_layout_and_aliased_pool_do_not_produce_target(self):
        for change in ['unknown', 'storage', 'concurrency']:
            with self.subTest(change=change):
                card = fixture()
                kv = card['ranks'][0]['kv_cache']
                if change == 'unknown': kv['group_layouts'][0]['spec_type'] = 'FutureCacheSpec'
                elif change == 'storage': kv['storage']['cuda_storage_bytes'] += 1024
                else: kv['max_concurrency'] = 13.7
                result = C.analyze(card)
                self.assertNotIn('target_kv_budget_bytes', result)

    def test_incomplete_failed_missing_and_duplicate_ranks_fail_closed(self):
        for kind in ['status', 'missing', 'duplicate', 'sampling', 'instrumentation', 'failed', 'api']:
            with self.subTest(kind=kind):
                card = fixture()
                if kind == 'status': card['status'] = 'incomplete'
                elif kind == 'missing': card['ranks'].pop()
                elif kind == 'duplicate': card['ranks'].append(copy.deepcopy(card['ranks'][0]))
                elif kind == 'sampling': card['hosts'][0]['sampling_covers_startup'] = False
                elif kind == 'instrumentation': card['coverage']['instrumentation_errors'] = [['cuda', 'error']]
                elif kind == 'failed': card['ranks'][0]['failures'] = [{}]
                else: card['api_processes'] = []
                self.assertTrue(C.analyze(card)['problems'])

    def test_unsupported_discrete_and_context_parallel_profiles(self):
        card = fixture()
        card['ranks'][0]['metadata']['hardware']['integrated'] = False
        self.assertTrue(C.analyze(card)['problems'])
        card = fixture()
        card['ranks'][0]['metadata']['configuration']['parallel_config']['decode_context_parallel_size'] = 2
        self.assertTrue(C.analyze(card)['problems'])

    def test_profile_not_mutated(self):
        card = fixture()
        before = copy.deepcopy(card)
        C.analyze(card)
        self.assertEqual(card, before)


class LiveTests(unittest.TestCase):
    def run_check(self, available, card=None):
        card = card or fixture()
        return C.check_cluster(card, C.analyze(card), inventory(card, available))

    def test_all_hosts_fit(self):
        result = self.run_check(110*G)
        self.assertEqual(result['status'], 'fits')
        self.assertEqual(result['target_status'], 'fits')
        self.assertEqual(result['any_kv_status'], 'fits')

    def test_full_cache_fails_but_128k_fits(self):
        result = self.run_check(60*G)
        self.assertEqual(result['status'], 'does_not_fit')
        self.assertEqual(result['target_status'], 'fits')
        self.assertEqual(result['any_kv_status'], 'fits')

    def test_128k_fails_but_small_cache_fits(self):
        result = self.run_check(56*G)
        self.assertEqual(result['target_status'], 'does_not_fit')
        self.assertEqual(result['any_kv_status'], 'fits')

    def test_below_non_kv_startup_floor(self):
        result = self.run_check(50*G)
        self.assertEqual(result['any_kv_status'], 'does_not_fit')

    def test_uncertain_small_cache_is_not_false_negative(self):
        result = self.run_check(55*G)
        self.assertEqual(result['any_kv_status'], 'unknown')

    def test_cannot_borrow_another_hosts_ram(self):
        card = fixture()
        live = inventory(card)
        live[1]['memory']['MemAvailable'] = 40*G
        result = C.check_cluster(card, C.analyze(card), live)
        self.assertEqual(result['status'], 'does_not_fit')
        self.assertEqual(result['any_kv_status'], 'does_not_fit')

    def test_initial_utilization_gate_can_fail_despite_peak_fitting(self):
        card = fixture()
        for r in card['ranks']:
            r['utilization_check']['gpu_memory_utilization'] = .95
            r['utilization_check']['requested_memory_bytes'] = .95*128*G
        result = self.run_check(110*G, card)
        self.assertEqual(result['status'], 'does_not_fit')
        self.assertGreater(result['hosts'][0]['profiled_required_bytes'], 110*G)

    def test_automatic_cache_growth_on_larger_host_is_accounted_for(self):
        card = fixture()
        live = inventory(card, 100*G)
        for node in live:
            node['memory']['MemTotal'] = 256*G
            node['gpus'][0]['total_memory_bytes'] = 256*G
        result = C.check_cluster(card, C.analyze(card), live)
        self.assertEqual(result['status'], 'does_not_fit')
        self.assertGreater(result['hosts'][0]['estimated_automatic_kv_bytes'], 100*G)

    def test_tuned_admission_uses_common_utilization_on_unequal_hosts(self):
        card = fixture()
        live = inventory(card, 65*G)
        live[1]['memory']['MemTotal'] = 256*G
        live[1]['gpus'][0]['total_memory_bytes'] = 256*G
        result = C.check_cluster(card, C.analyze(card), live)
        self.assertEqual(result['hosts'][0]['target_status'], 'fits')
        self.assertEqual(result['hosts'][1]['target_status'], 'does_not_fit')
        estimates = result['utilization_estimates']
        self.assertAlmostEqual(estimates['profiled_cache']['ranks'][0]['minimum_utilization'], .6)
        self.assertAlmostEqual(estimates['profiled_cache']['ranks'][1]['minimum_utilization'], .3)
        self.assertEqual(estimates['target_context']['setting'], result['target_gpu_memory_utilization'])

    def test_probe_failure_mismatch_wsl_and_duplicate_physical_hosts_unknown(self):
        for kind in ['offline', 'gpu', 'wsl', 'duplicate']:
            with self.subTest(kind=kind):
                card = fixture()
                live = inventory(card)
                if kind == 'offline': live[1] = {'error': 'unreachable'}
                elif kind == 'gpu': live[1]['gpus'][0]['name'] = 'Different GPU'
                elif kind == 'wsl': live[1]['wsl'] = True
                else: live[1]['host_id'] = live[0]['host_id']
                self.assertEqual(C.check_cluster(card, C.analyze(card), live)['status'], 'unknown')

    def test_existing_workload_is_not_credited_as_free_ram(self):
        self.assertEqual(self.run_check(10*G)['status'], 'does_not_fit')


class DiscoveryAndCliTests(unittest.TestCase):
    def test_solo_ignores_cluster_configuration(self):
        with patch.object(C.subprocess, 'run', side_effect=AssertionError('No subprocess needed')):
            self.assertEqual(C.select_hosts(1, Path('/nonexistent')), ['local'])

    def test_saved_cluster_trims_extra_peers_and_does_not_execute_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'cluster.env'
            config.write_text('LOCAL_IP=10.1.0.1\nCLUSTER_NODES=10.1.0.1,10.1.0.2,10.1.0.3\nTOKEN=$(false)\n')
            probe = subprocess.CompletedProcess([], 0, json.dumps([{'addr_info': [{'local': '10.1.0.1'}]}]))
            with patch.object(C.subprocess, 'run', return_value=probe):
                self.assertEqual(C.select_hosts(2, config), ['local', '10.1.0.2'])
            with self.assertRaises(C.TopologyError):
                C.select_hosts(4, config)

    def test_wrong_head_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'cluster.env'
            config.write_text('LOCAL_IP=10.1.0.1\nCLUSTER_NODES=10.1.0.1,10.1.0.2\n')
            with patch.object(C.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '[]')):
                with self.assertRaisesRegex(ValueError, 'head node'):
                    C.select_hosts(2, config)

    def test_ssh_is_bounded_and_host_is_not_shell_code(self):
        with patch.object(C.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '{"hostname":"worker"}')) as run:
            self.assertEqual(C.probe_host('worker')['hostname'], 'worker')
            args, kwargs = run.call_args
            self.assertIn('BatchMode=yes', args[0])
            self.assertIn('--', args[0])
            self.assertEqual(kwargs['timeout'], 20)
            self.assertNotIn('shell', kwargs)
        with patch.object(C.subprocess, 'run') as run:
            self.assertIn('error', C.probe_host('-oProxyCommand=bad'))
            self.assertIn('error', C.probe_host('host;bad'))
            run.assert_not_called()

    def test_cli_offline_json_report_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'card.yaml'
            path.write_text(yaml.safe_dump(fixture()))
            command = [sys.executable, str(MOD/'capacity.py'), str(path)]
            good = subprocess.run([*command, '--json'], capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)['requirements']['target_context'], 131072)
            no_nodes = subprocess.run([*command, '--check-host', '--config', '/dev/null'], capture_output=True, text=True)
            self.assertEqual(no_nodes.returncode, 1, no_nodes.stderr)
            self.assertIn('DOES NOT FIT', no_nodes.stdout)
            bad = subprocess.run([*command, '--reserve-gib', 'nan'], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            card = fixture()
            card['status'] = 'incomplete'
            path.write_text(yaml.safe_dump(card))
            incomplete = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(incomplete.returncode, 2)
            self.assertIn('UNKNOWN', incomplete.stdout)
            path.write_text('hosts: [broken')
            malformed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(malformed.returncode, 2)
            self.assertNotIn('Traceback', malformed.stderr)


if __name__ == '__main__':
    unittest.main()
