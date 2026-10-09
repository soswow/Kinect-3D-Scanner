"""Attribute Python boundaries in a complete-bridge timing run.

This observational launcher adds profiling overhead. Its timings must never be
published as uninstrumented performance measurements. It leaves the frozen
producer and numerical helpers unchanged, and the producer still requires its
own fresh audit permit and exact terminal/gate checks. A sidecar is written at
the existing completed-worker exit boundary, before Windows terminates the
isolated CUDA process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class BoundaryProfile:
    def __init__(self, output):
        self.output = output
        self.stack = []
        self.rows = {}
        self.started = time.perf_counter()
        self.saved = False
        self.active = False
        self.code_owners = {}
        self.prefix = str(ROOT).replace("\\", "/").casefold() + "/"

    def event(self, frame, event, arg):
        if event not in ("call", "return"):
            return
        code = frame.f_code
        if not self.active:
            if (event != "call" or code.co_name != "__call__"
                    or not code.co_filename.endswith("microbatch_bridge_driver.py")
                    or getattr(frame.f_locals.get("self"), "shared", None) is None):
                return
            self.active = True
            self.started = time.perf_counter()
        if (event == "call" and code.co_name == "finish_cuda_worker"
                and code.co_filename.endswith("process_metrics.py")):
            self.save("completed-producer-worker-boundary")
            return
        code_id = id(code)
        if code_id not in self.code_owners:
            filename = code.co_filename.replace("\\", "/")
            selected = filename.casefold().startswith(self.prefix) and filename != __file__.replace("\\", "/")
            self.code_owners[code_id] = ((filename[len(self.prefix):], code.co_firstlineno, code.co_name)
                                        if selected else None)
        key = self.code_owners[code_id]
        if key is None:
            return
        now = time.perf_counter()
        if event == "call":
            self.stack.append([id(frame), key, now, 0.0])
        elif self.stack and self.stack[-1][0] == id(frame):
            _, key, begin, children = self.stack.pop()
            elapsed = now - begin
            row = self.rows.setdefault(key, [0, 0.0, 0.0])
            row[0] += 1
            row[1] += elapsed
            row[2] += max(0.0, elapsed - children)
            if self.stack:
                self.stack[-1][3] += elapsed

    def save(self, boundary):
        if self.saved:
            return
        sys.setprofile(None)
        rows = [{"file": k[0], "line": k[1], "function": k[2],
                 "calls": v[0], "inclusive_wall_s": v[1], "exclusive_within_selected_files_wall_s": v[2]}
                for k, v in self.rows.items()]
        rows.sort(key=lambda r: r["inclusive_wall_s"], reverse=True)
        report = {"kind": "complete-bridge-observational-python-boundaries-v1",
                  "instrumented": True, "performance_authority": False,
                  "boundary": boundary, "started_at": "first GPU alignment call; imports, preflight, native controls and shared setup excluded",
                  "profiled_wall_s": time.perf_counter() - self.started,
                  "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "timer_scope": "Inclusive clocks overlap. Exclusive means selected repository Python children removed; native/library calls and profiler overhead remain. Do not subtract these clocks to predict scanner speed.",
                  "rows": rows}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self.saved = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("producer_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.profile_output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "benchmark-output"):
        parser.error("Require a fresh private benchmark-output sidecar")
    from scripts.research import microbatch_bridge_driver as producer
    rest = args.producer_arguments
    if rest[:1] == ["--"]:
        rest = rest[1:]
    sys.argv = [producer.__file__] + rest
    configured = producer.parse()
    if configured.mode != "timing":
        parser.error("Only separately audited timing mode can be profiled")
    profile = BoundaryProfile(output)
    sys.setprofile(profile.event)
    try:
        producer.run(configured)
    finally:
        profile.save("returned-or-failed-producer")


if __name__ == "__main__":
    main()
