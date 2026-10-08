"""Pure CPU check of conservative float AABB enclosure; no GPU imports.

This checks the bounds/query-rounding premise. Actual OptiX traversal, FP64
selection, native CPU index agreement and speed require the separate proof.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "benchmark-output/cuda-pipeline/optix-nearest/cpu-enclosure-check.json")
    args = parser.parse_args()
    source_path = ROOT / "scripts/cuda_optix_registration.py"
    source = source_path.read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "conservative_aabbs")
    function_source = ast.get_source_segment(source, function)
    namespace = {"np": np}
    # Extract only the pure NumPy helper. Importing the GPU module is unnecessary.
    exec(compile(function_source, str(source_path), "exec"), namespace)
    enclosure = namespace["conservative_aabbs"]
    rng = np.random.default_rng(8174)
    inside_count = 0
    for scale in (0., 1., 1000., 100000., 999900.):
        points = rng.uniform(-1., 1., (32, 3)) * scale
        for radius in (1e-6, 1e-5, .003, .03, .12, 3.):
            boxes = enclosure(points, radius)
            assert boxes is not None
            for factor in (0., .5, np.nextafter(1., 0.), 1., np.nextafter(1., np.inf)):
                for axis in range(3):
                    for sign in (-1., 1.):
                        queries = points.copy()
                        queries[:, axis] += sign * factor * radius
                        squared = np.sum((queries - points)**2, axis=1)
                        inside = squared < radius * radius
                        rounded = queries.astype(np.float32)
                        contained = np.all((rounded > boxes[:, :3]) &
                                           (rounded < boxes[:, 3:]), axis=1)
                        assert np.all(contained[inside]), (scale, radius, factor, axis, sign)
                        inside_count += int(np.count_nonzero(inside))
    assert enclosure(np.array([[1e7, 0., 0.]]), .03) is None
    assert enclosure(np.array([[0., 0., 0.]]), 1e-9) is None
    assert enclosure(np.array([[np.inf, 0., 0.]]), .03) is None
    result = {"kind": "pure-cpu-conservative-float-enclosure-check",
              "contained_strict_radius_queries": inside_count,
              "passed": True, "gpu_initialization": False,
              "helper_sha256": hashlib.sha256(function_source.encode()).hexdigest(),
              "module_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__,
              "scope": "Outward float bounds and monotone query rounding only; actual OptiX traversal and nearest search remain untested."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
