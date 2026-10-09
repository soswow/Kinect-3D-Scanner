"""Source/stdlib contracts for the unexecuted combined-sync research helper.

These tests do not import numerical libraries or simulate mathematical GPU
results. They exercise source containment, counters, memory limits and the
actual flagged/malformed orchestration paths using instrumented transport.
Fresh synthetic/all-nine numerical proofs remain mandatory.
"""

import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from scripts.research import research_combined_sync_icp as helper
from scripts.research import validate_combined_sync_proof as proof


class SourceContracts(unittest.TestCase):
    def test_imports_do_not_load_numerical_libraries(self):
        for name in ("numpy", "open3d", "cupy", "cv2"):
            self.assertNotIn(name, sys.modules)

    def test_guard_inverse_recovers_original_math_bytes(self):
        original = helper.original_gpu_source()
        guarded = helper.guarded_gpu_source(original)
        for before, after in reversed(helper._EDITS):
            self.assertEqual(guarded.count(after), 1)
            guarded = guarded.replace(after, before)
        self.assertEqual(guarded.encode(), original.encode())

    def test_guard_precedes_barriers_and_partial_reads(self):
        text = helper.guarded_gpu_source()
        normal = text.split('void combined_guarded_normal_partials(', 1)[1]
        self.assertLess(normal.index('if (control[0] || control[7]) return;'), normal.index('__shared__'))
        collapse = text.split('void combined_guarded_collapse_partials(', 1)[1]
        self.assertLess(collapse.index('if (control[0] || control[7])'), collapse.index('partials['))
        self.assertIn('if (threadIdx.x < 30) totals[threadIdx.x] = 0.;', collapse)

    def test_wrong_or_duplicated_original_signature_fails_closed(self):
        original = helper.original_gpu_source()
        for modified in (original.replace('void normal_partials(', 'void altered('),
                         original + 'void normal_partials('):
            with self.assertRaises(RuntimeError):
                helper.guarded_gpu_source(modified)

    def test_original_transform_is_byte_identical_and_no_new_solver(self):
        original = helper.original_gpu_source().split('extern "C" __global__ void normal_partials(', 1)[0]
        actual = helper.guarded_gpu_source().split('extern "C" __global__ void combined_guarded_normal_partials(', 1)[0]
        self.assertEqual(original, actual)
        source = Path(helper.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CombinedSyncResidentICP')
        names = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        self.assertFalse({'_transform', 'match', '_equations', '_original_cpu_match'} & names)
        self.assertNotIn('linalg.solve', source)

    def test_compile_contract_has_no_fma_and_no_graph_claim(self):
        value = helper.source_contract()
        self.assertEqual(value['options'], ['--fmad=false'])
        self.assertFalse(value['graph_capture'])
        self.assertEqual(value['original_gpu_source_sha256'],
                         hashlib.sha256(helper.original_gpu_source().encode()).hexdigest())


class CounterContracts(unittest.TestCase):
    def test_original_unambiguous_hits_and_misses(self):
        values = [0, 7, 3, 0, 0, 0, 55, 0, 0, 0]
        original = copy.copy(values)
        value = helper.validate_provisional_counters(values, 10, 12, direct_miss=True)
        self.assertEqual(value['hits'], 7)
        self.assertEqual(value['misses'], 3)
        self.assertEqual(values, original)

    def test_supported_ambiguous_rows_request_original_cpu_resolution(self):
        value = helper.validate_provisional_counters([2, 8, 0, 2, 1, 2, 55, 0, 0, 0], 10, 12, direct_miss=True)
        self.assertEqual(value['flagged'], 2)
        self.assertEqual(value['fallback'], 2)

    def test_missing_original_cpu_policy_is_flagged(self):
        value = helper.validate_provisional_counters([3, 7, 0, 3, 0, 0, 55, 0, 0, 0], 10, 12, direct_miss=False)
        self.assertEqual(value['flagged'], 3)
        with self.assertRaises(RuntimeError):
            helper.validate_provisional_counters([0, 7, 3, 0, 0, 0, 55, 0, 0, 0], 10, 12, direct_miss=False)

    def test_malformed_never_converts_to_cpu_recovery(self):
        for values in ([0, 10, 0, 0, 0, 0, 55, 1, 0, 0],
                       [2, 8, 0, 2, 3, 0, 55, 0, 0, 0],
                       [0, 9, 0, 0, 0, 0, 55, 0, 0, 0],
                       [0, 10, 0, 0, 0, 0, 121, 0, 0, 0],
                       [0, 10, 0, 0, 0, 0, 55, 0, 1, 0]):
            with self.assertRaises(RuntimeError):
                helper.validate_provisional_counters(values, 10, 12, direct_miss=True)

    def test_negative_wrong_dtype_boolean_shape_and_count_rejected(self):
        valid = [0, 10, 0, 0, 0, 0, 55, 0, 0, 0]
        for values in (valid[:-1], [False] + valid[1:], [0.0] + valid[1:], [-1] + valid[1:], [2**64] + valid[1:]):
            with self.assertRaises(RuntimeError):
                helper.validate_provisional_counters(values, 10, 12, direct_miss=True)
        for count in (0, -1, True, helper.MAX_POINTS + 1):
            with self.assertRaises(RuntimeError):
                helper.validate_provisional_counters(valid, count, 12, direct_miss=True)


class MemoryContracts(unittest.TestCase):
    def test_320_byte_joint_transport_and_additional_owned_buffers(self):
        self.assertEqual(helper.COMBINED_BYTES, 320)
        forecast = helper.scratch_forecast(128, 256)
        self.assertEqual(forecast['persistent_fast_bytes'], 20 * 128 + 320)
        self.assertEqual(forecast['peak_query_bytes'], 156 * 128 + 400)
        self.assertGreaterEqual(forecast['peak_query_bytes'], forecast['provisional_query_bytes'])
        self.assertEqual(forecast['combined_array_bytes'], forecast['resident_owned_bytes'] + forecast['peak_query_bytes'])

    def test_original_limit_enlargement_is_reported_not_hidden(self):
        value = helper.scratch_forecast(1_000_000, 1_000_000)
        self.assertGreater(value['peak_query_bytes'], 136 * 1024**2)
        self.assertLess(value['combined_array_bytes'], 256 * 1024**2)
        for args in ((-1, 2), (True, 2), (2, helper.MAX_POINTS + 1)):
            with self.assertRaises(ValueError):
                helper.scratch_forecast(*args)

    def test_old_timing_authority_and_disabled_audits_rejected_before_imports(self):
        for changes in ({'audit_nearest':False}, {'audit_misses':False}, {'proof_authority':object()}):
            values = {'audit_nearest':True, 'audit_misses':True, 'proof_authority':None}
            values.update(changes)
            with self.assertRaises(ValueError):
                helper.CombinedSyncResidentICP(types.SimpleNamespace(**values), lambda *_: None, gpu_timing=False)
        with self.assertRaises(ValueError):
            helper.CombinedSyncResidentICP(types.SimpleNamespace(audit_nearest=True, audit_misses=True,
                proof_authority=None), lambda *_: None, gpu_timing=True)


def complete_common_statistics(count=129):
    keys = ('iteration_calls', 'primary_counter_term_syncs', 'primary_download_bytes', 'provisional_queries',
        'provisional_candidate_visits', 'common_calls', 'common_queries', 'flagged_calls', 'flagged_queries',
        'discarded_placeholder_calls', 'malformed_results', 'full_query_audited_calls', 'full_query_audited_rows',
        'id_bit_comparison_rows', 'metric_bit_comparison_rows', 'filtered_bit_comparison_rows',
        'term_bit_comparison_calls', 'equation_bit_mismatches', 'nearest_bit_mismatches', 'peak_combined_owned_query_bytes')
    values = {key:0 for key in keys}
    for key in ('iteration_calls','primary_counter_term_syncs','common_calls','full_query_audited_calls','term_bit_comparison_calls'):
        values[key] = 1
    for key in ('provisional_queries','common_queries','full_query_audited_rows','id_bit_comparison_rows',
                'metric_bit_comparison_rows','filtered_bit_comparison_rows'):
        values[key] = count
    values['primary_download_bytes'] = 320
    return values


class AuthorityContracts(unittest.TestCase):
    def test_distinct_immutable_token_defensive_runtime(self):
        token = proof.CombinedSyncProofAuthority('a','a'*64,'b','b'*64,'{"gpu":"actual"}',
            'c'*64,frozenset({'d'*64}),(('source','e'*64),))
        self.assertNotIsInstance(token, proof.original.DeviceFlatGridProofAuthority)
        from scripts.research.validate_bulk_resident_audit import BulkResidentAuditAuthority
        self.assertNotIsInstance(token, BulkResidentAuditAuthority)
        value = token.runtime_binding
        value['gpu'] = 'mutated'
        self.assertEqual(token.runtime_binding['gpu'], 'actual')
        with self.assertRaises(AttributeError):
            token.bridge_report_sha256 = 'f'*64

    def test_complete_common_counts_pass_without_numerical_imports(self):
        proof.combined_counts(complete_common_statistics(), 129)

    def test_query_coverage_term_or_bit_omission_fails(self):
        original = complete_common_statistics()
        for key in ('full_query_audited_rows', 'primary_counter_term_syncs', 'primary_download_bytes',
                    'term_bit_comparison_calls', 'id_bit_comparison_rows', 'metric_bit_comparison_rows',
                    'filtered_bit_comparison_rows'):
            value = copy.copy(original)
            value[key] -= 1
            with self.assertRaises(proof.GridProofError):
                proof.combined_counts(value, 129)

    def test_hidden_malformed_mismatch_and_boolean_counter_fail(self):
        original = complete_common_statistics()
        for key in ('malformed_results', 'nearest_bit_mismatches', 'equation_bit_mismatches'):
            value = copy.copy(original)
            value[key] = 1
            with self.assertRaises(proof.GridProofError):
                proof.combined_counts(value, 129)
        value = copy.copy(original)
        value['common_calls'] = True
        with self.assertRaises(proof.GridProofError):
            proof.combined_counts(value, 129)

    def test_undiscarded_flagged_placeholders_rejected(self):
        value = complete_common_statistics()
        for key in ('iteration_calls', 'primary_counter_term_syncs', 'full_query_audited_calls'):
            value[key] += 1
        value['primary_download_bytes'] += 320
        value['flagged_calls'] = 1
        with self.assertRaises(proof.GridProofError):
            proof.combined_counts(value, 129)

    def test_old_kind_refused_before_disk_or_numerical_checks(self):
        for kind in ('standalone-original-double-device-flat-resident', 'complete-old-resident-scalar-bulk-dual-audit',
                     'offline-checkpoint-resident-finish-v2'):
            with self.assertRaisesRegex(proof.GridProofError, 'Old device/Finish/bulk'):
                proof.common_checks({'kind':kind}, {}, {}, {})

    def test_original_nine_proposal_loop_ast_exact(self):
        from scripts.research.benchmark_combined_sync_icp import original_nine_loop_binding
        self.assertTrue(original_nine_loop_binding()['original_nine_proposal_loop_unchanged'])



class FakeArray:
    def __init__(self, count=10, *, columns=None, dtype='f8', events=None):
        self.shape = (count, columns) if columns is not None else (count,)
        self.ndim, self.dtype = len(self.shape), dtype
        self.flags, self.device = types.SimpleNamespace(c_contiguous=True), types.SimpleNamespace(id=0)
        self.events = events if events is not None else []

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, key, value):
        self.events.append('control_reset')

    def __getitem__(self, key):
        return self

    def view(self, _):
        return self


class FakeHost:
    nbytes = 320
    def __init__(self, counters):
        self.counters = counters
    def __getitem__(self, _):
        return self.counters


def instrumented_iteration(counters):
    events = []
    solver = object.__new__(helper.CombinedSyncResidentICP)
    cp = types.SimpleNamespace(ndarray=FakeArray, float64='f8', int32='i4',
        asnumpy=lambda _: (events.append('joint_copy') or FakeHost(counters)))
    np = types.SimpleNamespace(uint32=int, int32=int, float64=float)
    nearest, distances = FakeArray(dtype='i4'), FakeArray()
    def audit(*_):
        events.append('original_full_cpu_audit')
        return nearest, distances
    retrieval = types.SimpleNamespace(miss_policy='direct-miss-research-v1',
        _raw_device=lambda *_: (events.append('original_raw_nn') or FakeArray()),
        classify=lambda *_: events.append('original_classifier'), nearest_device=audit)
    solver.cp, solver.np, solver.retrieval = cp, np, retrieval
    solver.device_id, solver.max_points = 0, helper.MAX_POINTS
    solver._check_configuration = lambda: None
    solver.statistics = {'nn_wall_s':0.0}
    keys = ('iteration_calls', 'primary_counter_term_syncs', 'primary_download_bytes', 'provisional_queries',
        'provisional_candidate_visits', 'malformed_results', 'full_query_audited_calls', 'full_query_audited_rows',
        'flagged_calls', 'flagged_queries', 'discarded_placeholder_calls', 'common_calls', 'common_queries',
        'primary_enqueue_s', 'primary_copy_wait_s', 'original_full_audit_s')
    solver.combined_statistics = {key:0 for key in keys}
    control = FakeArray(40, dtype='u8', events=events)
    solver._buffers = lambda _: {'combined':control, 'ids':FakeArray(dtype='i4'), 'squared':FakeArray(),
        'reasons':FakeArray(dtype='u4'), 'flagged':FakeArray(dtype='u4')}
    solver.combined_partial_kernel = lambda *_: events.append('guarded_original_equations')
    solver.combined_collapse_kernel = lambda *_: events.append('guarded_collapse')
    moving = FakeArray(columns=3)
    item = {'target':types.SimpleNamespace(points=[None]*12), 'data':FakeArray(12, columns=3)}
    return solver, events, (moving, item, FakeArray(12, columns=3), FakeArray(dtype='i4'), .12,
        FakeArray(1, columns=30), FakeArray(30))


class OrchestrationContracts(unittest.TestCase):
    def test_flagged_placeholders_discarded_original_cpu_resolves_before_math(self):
        solver, events, args = instrumented_iteration([2, 8, 0, 2, 1, 2, 55, 0, 0, 0])
        result = object()
        def original_math(*_):
            events.append('authoritative_original_equations')
            return result
        with patch.object(helper.DeviceGridResidentICP, '_equations', original_math):
            self.assertIs(solver._iteration_data(*args), result)
        self.assertEqual(events, ['control_reset', 'original_raw_nn', 'original_classifier',
            'guarded_original_equations', 'guarded_collapse', 'joint_copy',
            'original_full_cpu_audit', 'authoritative_original_equations'])
        stats = solver.combined_statistics
        self.assertEqual(stats['discarded_placeholder_calls'], 1)
        self.assertEqual(stats['full_query_audited_rows'], 10)
        self.assertEqual(stats['primary_counter_term_syncs'], 1)
        self.assertEqual(stats['primary_download_bytes'], 320)

    def test_malformed_hard_fault_before_original_cpu_or_solve(self):
        solver, events, args = instrumented_iteration([0, 10, 0, 0, 0, 0, 55, 1, 0, 0])
        with self.assertRaisesRegex(RuntimeError, 'CPU recovery is prohibited'):
            solver._iteration_data(*args)
        self.assertNotIn('original_full_cpu_audit', events)
        self.assertEqual(solver.combined_statistics['malformed_results'], 1)

    def test_gpu_execution_fault_does_not_invoke_cpu_resolution(self):
        solver, events, args = instrumented_iteration([0, 10, 0, 0, 0, 0, 55, 0, 0, 0])
        def failure(*_):
            raise RuntimeError('injected device execution fault')
        solver.combined_partial_kernel = failure
        with self.assertRaisesRegex(RuntimeError, 'device execution fault'):
            solver._iteration_data(*args)
        self.assertNotIn('original_full_cpu_audit', events)
        self.assertNotIn('joint_copy', events)

    def test_complete_cpu_audit_failure_never_reaches_equations(self):
        solver, events, args = instrumented_iteration([2, 8, 0, 2, 1, 2, 55, 0, 0, 0])
        solver.retrieval.nearest_device = lambda *_: (_ for _ in ()).throw(RuntimeError('original CPU audit mismatch'))
        with patch.object(helper.DeviceGridResidentICP, '_equations') as equations:
            with self.assertRaisesRegex(RuntimeError, 'CPU audit mismatch'):
                solver._iteration_data(*args)
            equations.assert_not_called()

    def test_policy_mutation_fails_before_any_launch(self):
        solver, events, args = instrumented_iteration([0, 10, 0, 0, 0, 0, 55, 0, 0, 0])
        solver._check_configuration = lambda: (_ for _ in ()).throw(RuntimeError('policy changed'))
        with self.assertRaisesRegex(RuntimeError, 'policy changed'):
            solver._iteration_data(*args)
        self.assertEqual(events, [])


class CompleteICPShadowContracts(unittest.TestCase):
    def setUp(self):
        from scripts.research import benchmark_combined_sync_icp as producer
        from scripts.research import finish_resident_registration as observation
        from scripts.research import validate_finish_resident_proof as checks
        self.producer, self.observation, self.checks = producer, observation, checks
        tree = ast.parse(Path(producer.__file__).read_text(encoding='utf-8'))
        factory = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='bridge_class')
        cls = next(node for node in factory.body if isinstance(node,ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='match')
        code = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        self.events = []
        self.result,self.shadow = object(),object()
        self.source,self.target,self.initial = object(),object(),object()
        self.resident = types.SimpleNamespace(statistics={'cpu_fallback_calls':0})
        def match(a,b,c):
            self.events.append('resident')
            return self.result
        self.resident.match = match
        def native(a,b,c):
            self.assertIs(a,self.source); self.assertIs(b,self.target); self.assertIs(c,self.initial)
            self.events.append('original_cpu_full_icp')
            return self.shadow
        namespace = {'time':__import__('time'),'os':os,'original_cpu_match':native,
            'CombinedComponentFailure':producer.CombinedComponentFailure}
        exec(compile(code,producer.__file__,'exec'),namespace)
        self.bridge_type = type('RecordedBridge',(),{'match':namespace['match']})
        self.bridge = self.bridge_type()
        self.bridge.resident,self.bridge.shadow_rows,self.bridge.shadow_failure = self.resident,[],None
        self.bridge.statistics = {'icp_calls':0,'icp_wall_s':0.0}
        self.patchers = [
            patch.dict(sys.modules,{'cv2':types.SimpleNamespace(getNumThreads=lambda:20),
                'open3d':types.SimpleNamespace(utility=types.SimpleNamespace(get_max_threads=lambda:20))}),
            patch.dict(os.environ,{'OMP_NUM_THREADS':'8','KINECT_CUDA_REGISTRATION':'cpu'}),
            patch.object(observation,'capture_inputs',lambda *_:{'original':True}),
            patch.object(observation,'summarize_result',lambda value,*_:{'owner':id(value)}),
            patch.object(observation,'compare_results',lambda *_:{'passed':True}),
            patch.object(checks,'call_signature',lambda _:'a'*64),
            patch.object(checks,'validate_shadow',lambda _:self.events.append('full_shadow_validated'))]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_original_cpu_shadow_once_and_identical_resident_return(self):
        value = self.bridge.match(self.source,self.target,self.initial)
        self.assertIs(value,self.result)
        self.assertEqual(self.events,['resident','original_cpu_full_icp','full_shadow_validated'])
        self.assertTrue(self.bridge.shadow_rows[0]['complete'])
        self.assertEqual(self.bridge.statistics['icp_calls'],1)

    def test_shadow_fault_latched_outside_ordinary_rejection_and_not_retried(self):
        with patch.object(self.checks,'validate_shadow',side_effect=ValueError('CPU correspondence mismatch')):
            with self.assertRaises(self.producer.CombinedComponentFailure) as first:
                self.bridge.match(self.source,self.target,self.initial)
        self.assertNotIsInstance(first.exception,Exception)
        events = copy.copy(self.events)
        with self.assertRaises(self.producer.CombinedComponentFailure) as second:
            self.bridge.match(self.source,self.target,self.initial)
        self.assertIs(first.exception,second.exception)
        self.assertEqual(self.events,events)
        self.assertFalse(self.bridge.shadow_rows[0]['complete'])

    def test_resident_device_failure_preserved_and_no_cpu_pose_substitution(self):
        primary = RuntimeError('actual device fault')
        self.resident.match = lambda *_: (_ for _ in ()).throw(primary)
        with self.assertRaises(self.producer.CombinedComponentFailure) as caught:
            self.bridge.match(self.source,self.target,self.initial)
        self.assertIs(caught.exception.__cause__,primary)
        self.assertNotIn('original_cpu_full_icp',self.events)
        self.assertFalse(self.bridge.shadow_rows[0]['complete'])

    def test_wrong_thread_policy_rejected_before_resident_or_cpu(self):
        with patch.dict(sys.modules,{'cv2':types.SimpleNamespace(getNumThreads=lambda:1)}):
            with self.assertRaises(self.producer.CombinedComponentFailure):
                self.bridge.match(self.source,self.target,self.initial)
        self.assertEqual(self.events,[])


if __name__ == '__main__':
    unittest.main()
