"""Observe exact global-proposal inputs without changing their computation.

This is a diagnostic supervisor for the frozen checkpoint runner. Its distinct
report kind deliberately cannot authorize that runner's production timing
route. Private snapshots explain where CPU/resident Finish paths first differ;
they are neither pose seeds nor a speed/quality proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KIND = "offline-finish-global-proposal-input-diagnostic-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("finish_arguments", nargs=argparse.REMAINDER,
                        help="After --, the complete original checkpoint-runner arguments")
    args = parser.parse_args()
    arguments = args.finish_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments or "--run-allocated" not in arguments:
        parser.error("Supply original allocated Finish arguments after --")
    if (arguments.count("--mode") != 1 or arguments.index("--mode") + 1 >= len(arguments)
            or arguments[arguments.index("--mode") + 1] not in ("native", "audit")):
        parser.error("Diagnostic observation supports native/audit only; it authorizes no timing")
    args.diagnostics = args.diagnostics.resolve()
    try:
        args.diagnostics.relative_to(ROOT / "benchmark-output")
    except ValueError:
        parser.error("Keep private snapshots inside ignored benchmark-output")
    snapshot = args.diagnostics.with_suffix(".inputs.npz")
    if args.diagnostics.exists() or snapshot.exists():
        parser.error("Preserve old diagnostic artifacts; use fresh output paths")

    from scripts.research import profile_checkpoint_resident_finish as base
    from scripts.research.validate_checkpoint_finish_proof import FINISH_ARTIFACTS
    from scripts import process_metrics

    pins = {name: digest(ROOT / name) for name in (*FINISH_ARTIFACTS,
            "scripts/research/profile_finish_ransac_diagnostics.py", "scripts/process_metrics.py")}
    report = {"kind": KIND, "status": "running", "artifacts_sha256": pins,
              "rows": [], "performance_authority": False,
              "scope": "Output-only copies of actual original global RANSAC input arrays. "
                       "Original proposal seeds, algorithms, thread protocol and gates delegated. "
                       "Observation overhead remains inside the supervised Finish; no timing claim."}
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def save():
        args.diagnostics.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")

    original_scope, original_kind = base.GlobalSeedThreadScope, base.KIND
    original_worker = process_metrics.finish_cuda_worker
    deferred_worker_calls = []
    previous_argv = sys.argv[:]

    def defer_worker():
        deferred_worker_calls.append(True)

    def fragment_values(fragment):
        return (("train_points", fragment.train.points),
                ("train_normals", fragment.train.normals),
                ("coarse_points", fragment.coarse.points),
                ("coarse_normals", fragment.coarse.normals),
                ("fpfh", fragment.fpfh.data))

    class DiagnosticScope(original_scope):
        def call(self, source, target, seed):
            import numpy as np
            row = {"invocation": len(report["rows"]), "source": source.index,
                   "target": target.index, "original_seed": seed, "complete": False,
                   "inputs": {}}
            report["rows"].append(row)
            observed = []
            try:
                for side, fragment in (("source", source), ("target", target)):
                    for name, value in fragment_values(fragment):
                        value = np.asarray(value)
                        if value.dtype != np.float64 or not np.isfinite(value).all():
                            self.fail("Diagnostic requires finite original FP64 proposal arrays")
                        label = f"r{row['invocation']:04d}_{side}_{name}"
                        arrays[label] = np.array(value, copy=True, order="C")
                        binding = {"shape": list(value.shape), "dtype": value.dtype.str,
                                   "sha256": hashlib.sha256(arrays[label].tobytes()).hexdigest(),
                                   "snapshot_key": label}
                        row["inputs"][side + "_" + name] = binding
                        observed.append((fragment, name, binding))
            except BaseException as error:
                self.fail("Original proposal input observation failed", error)
            try:
                result = super().call(source, target, seed)
            except BaseException as error:
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                raise
            try:
                if result is None:
                    row["proposal"] = None
                else:
                    copied = np.array(result, dtype=np.float64, copy=True)
                    row["proposal"] = copied.tolist()
                    row["proposal_sha256"] = hashlib.sha256(copied.tobytes()).hexdigest()
                row["complete"] = True
                return result
            except BaseException as error:
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                self.fail("Original proposal output observation failed", error)
            finally:
                try:
                    for fragment, name, binding in observed:
                        value = np.asarray(dict(fragment_values(fragment))[name])
                        if (list(value.shape) != binding["shape"] or value.dtype.str != binding["dtype"]
                                or hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != binding["sha256"]):
                            self.fail("Original global proposal input changed during observation")
                except BaseException as error:
                    self.fail("Original proposal input closure failed", error)

        def report(self):
            value = super().report()
            value["diagnostic_supervisor"] = {"kind": KIND, "artifacts_sha256": pins,
                                               "path": str(args.diagnostics)}
            return value

    save()
    failure = None
    try:
        base.GlobalSeedThreadScope, base.KIND = DiagnosticScope, KIND
        process_metrics.finish_cuda_worker = defer_worker
        sys.argv = [str(Path(base.__file__).resolve()), *arguments]
        base.main()
        if any(digest(ROOT / name) != value for name, value in pins.items()):
            raise RuntimeError("Diagnostic dependencies changed during observation")
        if not report["rows"] or not all(row["complete"] for row in report["rows"]):
            raise RuntimeError("Diagnostic proposal coverage incomplete")
        if sys.platform == "win32" and deferred_worker_calls != [True]:
            raise RuntimeError("Expected exactly one deferred terminal worker request")
        report["status"] = "complete"
    except BaseException as error:
        failure = error
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        base.GlobalSeedThreadScope, base.KIND = original_scope, original_kind
        process_metrics.finish_cuda_worker = original_worker
        sys.argv = previous_argv
        try:
            if arrays:
                import numpy as np
                np.savez_compressed(snapshot, **arrays)
                report["snapshot"] = {"path": str(snapshot), "sha256": digest(snapshot)}
            report["supervisor_restored"] = True
            report["deferred_terminal_worker_calls"] = len(deferred_worker_calls)
            save()
        except BaseException as error:
            report.update(status="failed", closure_failure={"type": type(error).__name__, "message": str(error)})
            try:
                save()
            except BaseException as secondary:
                error.add_note(f"Failure-status write also failed: {secondary}")
            if failure is not None:
                failure.add_note(f"Diagnostic snapshot/report also failed: {error}")
            else:
                raise
    if deferred_worker_calls:
        original_worker()


if __name__ == "__main__":
    main()
