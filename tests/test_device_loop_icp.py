"""Source/control/ownership contracts only; these are not numerical GPU proof."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import device_loop_icp as loop


class SourceContracts(unittest.TestCase):
    def test_complete_source_injects_uniform_guards_and_recovers_original_math(self):
        result = loop.source_contract()
        self.assertTrue(result["original_math_guard_inverse"])
        self.assertTrue(result["graph_option_supported"])
        self.assertEqual(result["stages"], [[.12, 40], [.06, 30], [.03, 20]])
        source = loop.generated_source()
        self.assertIn("if(control[0]!=SOLVE || totals[28]==0.)return;", source)
        self.assertIn("if(control[0]!=EQUATIONS) return;", source)
        self.assertIn("if (control[0]!=PREPARE && control[0]!=QUERY) return;", source)
        self.assertIn("for(int block=0;", source.replace("for (int block=0;", "for(int block=0;"))

    def test_guard_refuses_changed_original_signatures(self):
        with self.assertRaises(loop.DeviceLoopError):
            loop.edited("void x(){}", (("void y()", "void guarded_y()"),))

    def test_guard_refuses_repeated_or_preexisting_seam(self):
        for value in ("old old", "old new"):
            with self.subTest(value=value), self.assertRaises(loop.DeviceLoopError):
                loop.edited(value, (("old", "new"),))

    def test_no_numeric_import_at_module_load(self):
        tree = ast.parse(Path(loop.__file__).read_text(encoding="utf-8"))
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        modules = [n.module if isinstance(n, ast.ImportFrom) else a.name
                   for n in imports for a in (n.names if isinstance(n, ast.Import) else [None])]
        self.assertFalse(any(name and name.split(".")[0] in ("numpy", "cupy", "open3d") for name in modules))

    def test_query_guards_precede_shared_barriers(self):
        source = loop.generated_source()
        for name, guard in (("loop_normal_partials", "if(control[0]!=EQUATIONS) return;"),
                ("loop_flat_grid_nearest_two", "if(control[0]!=PREPARE && control[0]!=QUERY) return;"),
                ("loop_classify_flat_results", "if(control[0]!=PREPARE && control[0]!=QUERY) return;")):
            body = source[source.index("void "+name):]
            self.assertLess(body.index(guard), body.index("__syncthreads()"))

    def test_no_host_per_iteration_equation_solve_or_pose_rounding(self):
        tree = ast.parse(Path(loop.__file__).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeviceLoopICP")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_enqueue_step")
        body = ast.unparse(method)
        self.assertNotIn("asnumpy", body)
        self.assertNotIn("round(", body)
        self.assertNotIn("EigenSolve", body)


class BoundedContracts(unittest.TestCase):
    def test_worst_case_allocation_includes_packet_and_controls(self):
        n, m = 1_000_000, 1_000_000
        f = loop.scratch_forecast(n, m)
        self.assertEqual(f["lane_bytes"], 184*n+240*((n+127)//128)+4096)
        self.assertEqual(f["combined_bytes"], f["lane_bytes"]+48*m)
        self.assertEqual(f["host_exception_packet_bytes"], 64*n)
        self.assertEqual(f["small_control_packet_bytes"], 256)

    def test_bounds_reject_bool_negative_and_excessive_count(self):
        for bad in (True, -1, 1_000_001, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                loop.scratch_forecast(bad, 1)

    def test_chunks_are_small_exact_integers(self):
        for count in (1, 2, 4):
            self.assertEqual(loop.checked_chunk(count), count)
        for bad in (True, 0, 3, 8, 1.):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                loop.checked_chunk(bad)

    def test_invalid_constructor_fails_before_numeric_imports(self):
        for kwargs in ({"device": True}, {"audit_nearest": False}, {"max_points": 0},
                {"max_total_bytes": -1}, {"audit_nearest": False, "audit_misses": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                loop.DeviceLoopICP(SimpleNamespace(device_id=0), **kwargs)

    def test_exhaustive_hit_miss_packet_and_legitimate_ambiguity(self):
        counters = [10, 5, 3, 2, 1, 1, 300, 0, 5, 3]
        self.assertEqual(loop.check_packet(counters, 10, 30, audit_hits=True, audit_misses=True), 10)
        self.assertEqual(loop.check_packet([2, 5, 3, 2, 1, 1, 300, 0, 0, 0], 10, 30,
            audit_hits=False, audit_misses=False), 2)

    def test_malformed_or_missing_shadow_never_authorizes_equations(self):
        valid = [10, 5, 3, 2, 1, 1, 300, 0, 5, 3]
        for index, replacement in ((0, 9), (1, 6), (4, 3), (6, 301), (7, 1), (8, 4), (9, 2)):
            values = valid.copy(); values[index] = replacement
            with self.subTest(index=index), self.assertRaises(loop.DeviceLoopError):
                loop.check_packet(values, 10, 30, audit_hits=True, audit_misses=True)

    def test_terminal_control_requires_all_three_stages(self):
        state = loop.control_state([6, 3, 0, 90, 0, 93, 2, 0])
        self.assertEqual(state["phase"], "done")
        for values in ([6, 2, 0, 90, 0, 93, 2, 0], [1, 3, 0, 90, 0, 93, 2, 0],
                [1, 2, 21, 90, 0, 93, 2, 0], [1, 0, 40, 91, 0, 93, 2, 0]):
            with self.subTest(values=values), self.assertRaises(loop.DeviceLoopError):
                loop.control_state(values)

    def test_transport_masks_bind_every_classifier_reason(self):
        counts = [2, 5, 3, 1, 1, 0, 0, 0]
        counters = [10, 5, 3, 2, 1, 1, 300, 0, 5, 3]
        loop.check_reason_counts(counts, counters)
        for index in range(8):
            changed = counts.copy(); changed[index] += 1
            with self.subTest(index=index), self.assertRaises(loop.DeviceLoopError):
                loop.check_reason_counts(changed, counters)


def fake_lane():
    lane = object.__new__(loop.DeviceLoopICP)
    lane.device, lane.max_points = 0, 100
    lane.max_scratch_bytes, lane.max_total_bytes = 4096, 8192
    lane.audit_nearest = lane.audit_misses = True
    lane.cuda_graph = False
    lane.configuration = (0, 100, 4096, 8192, True, True, False)
    lane.graph, lane.graph_chunk = None, None
    lane.cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()), asnumpy=lambda x, **_: x)
    lane.stream = nullcontext()
    lane.statistics = {"cleanup_s": 0., "chunk_wall_s": 0., "chunks": 0, "enqueued_steps": 0,
                       "solve_blocks": 0, "updates": 0, "queries": 0, "query_rows": 0}
    lane.statistics.update({"graph_capture_s": 0., "graph_captures": 0,
                            "maximum_graph_nodes": 0, "graph_launches": 0})
    lane.started, lane.closed, lane.failure = True, False, None
    lane.buffers, lane.lease = {"pose": object(), "matrix": object(), "gradient": object()}, {"owned": object()}
    lane.n = 10
    return lane


class FailureOwnershipContracts(unittest.TestCase):
    def test_nn_fault_latches_and_never_resolves_cpu_or_returns_result(self):
        lane = fake_lane()
        state = {"phase": "fault", "error": 1}
        with patch.object(lane, "_enqueue_step") as enqueue, patch.object(lane, "_read_control", return_value=state), \
                patch.object(lane, "_resolve_nn") as resolve:
            with self.assertRaises(loop.DeviceLoopError) as failure:
                lane.advance(2)
            self.assertIs(lane.failure, failure.exception)
            self.assertEqual(enqueue.call_count, 2)
            resolve.assert_not_called()
            with self.assertRaises(loop.DeviceLoopError):
                lane.advance(1)

    def test_solve_rejection_carries_actual_system_and_preserves_pose_owner(self):
        lane = fake_lane(); original = lane.buffers["pose"]
        with patch.object(lane, "_enqueue_step"), \
                patch.object(lane, "_read_control", return_value={"phase": "solve_block", "error": 4}):
            with self.assertRaises(loop.DeviceLoopBlocked) as error:
                lane.advance(1)
            self.assertIs(error.exception.evidence["previous_pose"], original)
            self.assertIs(error.exception.evidence["matrix"], lane.buffers["matrix"])
            self.assertIs(lane.buffers["pose"], original)
            self.assertEqual(lane.statistics["solve_blocks"], 1)

    def test_audit_failure_cannot_reach_equations_or_success(self):
        lane = fake_lane(); primary = ValueError("wrong actual NN ID")
        with patch.object(lane, "_enqueue_step"), \
                patch.object(lane, "_read_control", return_value={"phase": "nn_block"}), \
                patch.object(lane, "_resolve_nn", side_effect=primary):
            with self.assertRaises(ValueError) as error:
                lane.advance(4)
            self.assertIs(error.exception, primary)
            self.assertIs(lane.failure, primary)

    def test_cleanup_failure_retains_all_async_owners_and_primary(self):
        lane = fake_lane(); primary = ValueError("primary work fault"); cleanup = RuntimeError("sync fault")
        lane.failure = primary
        lane.stream = SimpleStream(cleanup)
        original_buffers, original_lease = lane.buffers, lane.lease
        with self.assertRaises(ValueError) as error:
            lane.close()
        self.assertIs(error.exception, primary)
        self.assertIs(error.exception.__cause__, cleanup)
        self.assertIs(lane.buffers, original_buffers)
        self.assertIs(lane.lease, original_lease)
        self.assertFalse(lane.closed)

    def test_successful_close_synchronizes_before_releasing_lease(self):
        lane = fake_lane(); events = []
        lane.stream = SimpleStream(None, lambda: events.append((bool(lane.buffers), lane.lease is not None)))
        lane.close()
        self.assertEqual(events, [(True, True)])
        self.assertEqual(lane.buffers, {})
        self.assertIsNone(lane.lease)
        self.assertTrue(lane.closed)

    def test_changed_audit_policy_is_rejected_before_enqueue(self):
        lane = fake_lane(); lane.audit_nearest = False
        with patch.object(lane, "_enqueue_step") as enqueue, self.assertRaises(loop.DeviceLoopError):
            lane.advance(1)
        enqueue.assert_not_called()

    def test_equation_pipeline_follows_classifier_blocking_transition(self):
        lane = fake_lane()
        lane.np = SimpleNamespace(int32=int, uint32=int)
        lane.blocks, lane.m = 1, 20
        names = ("totals matrix gradient status update composed pose control step pivots diagonal "
                 "original moving tables shifts raw nearest squared reasons flagged counters cumulative "
                 "target normals filtered partials metrics").split()
        lane.buffers = {name: name for name in names}
        calls = []
        with patch.object(lane, "_launch", side_effect=lambda name, *args: calls.append(name)):
            lane._enqueue_step()
        self.assertLess(calls.index("loop_classified"), calls.index("loop_normal_partials"))
        self.assertLess(calls.index("loop_device_ldlt_solve"), calls.index("loop_commit_update"))
        self.assertLess(calls.index("loop_commit_update"), calls.index("loop_transform_points"))
        self.assertEqual(calls[-1], "loop_metrics")

    def test_graph_capture_has_only_fixed_device_steps_and_reuses_one_graph(self):
        lane = fake_lane(); events = []
        class Graph:
            def launch(self, *, stream): events.append("launch")
        class Stream(SimpleStream):
            def begin_capture(self): events.append("begin")
            def end_capture(self): events.append("end"); return Graph()
        lane.stream = Stream(None, lambda: events.append("sync"))
        with patch.object(lane, "_enqueue_step", side_effect=lambda: events.append("step")), \
                patch.object(lane, "_read_control") as copy:
            lane._launch_graph(4)
            lane._launch_graph(4)
        self.assertEqual(events, ["sync", "begin", "step", "step", "step", "step", "end", "launch", "launch"])
        copy.assert_not_called()
        self.assertEqual(lane.statistics["maximum_graph_nodes"], 44)
        self.assertEqual(lane.statistics["graph_captures"], 1)
        with self.assertRaises(loop.DeviceLoopError): lane._launch_graph(2)

    def test_graph_capture_primary_failure_survives_end_capture_failure(self):
        lane = fake_lane(); primary = ValueError("enqueue fault"); cleanup = RuntimeError("capture fault")
        class Stream(SimpleStream):
            def begin_capture(self): pass
            def end_capture(self): raise cleanup
        lane.stream = Stream()
        with patch.object(lane, "_enqueue_step", side_effect=primary):
            with self.assertRaises(ValueError) as caught: lane._launch_graph(2)
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)

    def test_failed_sync_retains_graph_executable(self):
        lane = fake_lane(); owner = object(); lane.graph = owner
        lane.stream = SimpleStream(RuntimeError("pending graph"))
        with self.assertRaises(RuntimeError): lane.close()
        self.assertIs(lane.graph, owner)


class SimpleStream:
    def __init__(self, failure=None, on_sync=lambda: None):
        self.failure, self.on_sync = failure, on_sync
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def synchronize(self):
        self.on_sync()
        if self.failure: raise self.failure


if __name__ == "__main__":
    unittest.main()
