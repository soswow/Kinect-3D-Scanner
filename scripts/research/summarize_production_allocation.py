"""Publish scalar installed-Final allocation evidence from closed local reports.

Standard library only. Historical configured allowances are not observations of
old allocation. Source transitions and single-run walls cannot establish causal
speed. No arrays, calibration inventories, camera serials or private absolute
paths are copied, and this publication never authorizes a numerical experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.profile_session import source_hash

KIND = "installed-weighted-final-allocation-publication-v1"
BASELINE = "9331dee7c6ec7731eff9ebfeab76b83cc5484b8936dbd1d1d0f62fc570cef60a"
CURRENT = "07a948e81127dc742a330ec6bc24c85ceb7a9996128703c2a1cb7f9d0593741c"
STUDY = ROOT / "benchmark-output/field-cuda-study"
PAIRS = {
    "chest-5": ("raw-baselines-v1/chest-5-scan-session-cuda-0.json",
        "production-memory-raw-v1/chest-5-scan-session-full.json",
        "raw-baselines-v1/execution.json", "production-memory-raw-v1/execution.json"),
    "chest-6": ("raw-baselines-v1/chest-6-scan-session_20261009_002019-cuda-0.json",
        "production-memory-raw-v1/chest-6-scan-session_20261009_002019-full.json",
        "raw-baselines-v1/execution.json", "production-memory-raw-v1/execution.json"),
    "chest-7": ("raw-baselines-20k-cuda-v1/chest-7-scan-session_20261009_003259-cuda-0.json",
        "production-memory-raw-20k-v1/chest-7-scan-session_20261009_003259-full.json",
        "raw-baselines-20k-cuda-v1/execution.json", "production-memory-raw-20k-v1/execution.json"),
}
PRODUCER_FILES = ("scripts/research/summarize_production_allocation.py",
    "tests/test_production_allocation_summary.py", "scripts/profile_session.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def sha256(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Require a lowercase SHA256 digest")
    return value


def number(value, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0
        and (not positive or value > 0), "Invalid finite nonnegative metric")
    return value


def count(value, positive=False):
    require(type(value) is int and value >= int(positive), "Invalid integer counter")
    return value


def relative(name):
    require(type(name) is str and bool(name) and not name.startswith(("/", "\\"))
        and not Path(name).is_absolute() and "\\" not in name and ":" not in name
        and ".." not in Path(name).parts, "Require a bounded relative evidence path")
    return name


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def pins():
    return {name: file_digest(ROOT / name) for name in PRODUCER_FILES}


class Reports:
    def __init__(self, root=STUDY):
        self.root = Path(root)
        self.values, self.files = {}, {}

    def read(self, name):
        relative(name)
        if name not in self.values:
            path = self.root / name
            require(path.stat().st_size <= 32*1024**2, "Unbounded JSON evidence")
            raw = path.read_bytes()
            value = json.loads(raw)
            require(type(value) is dict, "Require a JSON object")
            self.values[name] = value
            self.files[name] = {"path": "benchmark-output/field-cuda-study/"+name,
                "sha256": digest(raw), "bytes": len(raw), "reported_status": value.get("status", "profile")}
        return self.values[name]

    def ref(self, name):
        self.read(name)
        return dict(self.files[name])

    def matches_path(self, name, recorded):
        relative(name)
        require(type(recorded) is str and Path(recorded).resolve() == (self.root/name).resolve(),
            "Evidence path does not identify the expected local report")

    def attachment(self, name, expected_digest):
        relative(name)
        expected_digest = sha256(expected_digest)
        if name not in self.files:
            path = self.root/name
            require(file_digest(path) == expected_digest, "Payload/log content digest differs")
            self.files[name] = {"path": "benchmark-output/field-cuda-study/"+name,
                "sha256": expected_digest, "bytes": path.stat().st_size, "reported_status": "payload"}
        require(self.files[name]["sha256"] == expected_digest, "Conflicting evidence digest")
        return dict(self.files[name])

    def close(self):
        require(all(file_digest(self.root/name) == record["sha256"] for name, record in self.files.items()),
            "Evidence bytes changed during publication")


def artifact_map(value):
    require(type(value) is dict and bool(value), "Missing source artifact inventory")
    normalized = set()
    for name, content in value.items():
        # Historical Windows producers recorded repository-relative Path keys.
        # Validate their normalized spelling; never change measured dictionaries
        # or relax exact before/after equality, and never publish these keys.
        require(type(name) is str, "Artifact identity must be text")
        safe_name = relative(name.replace("\\", "/"))
        require(safe_name not in normalized, "Ambiguous artifact path aliases")
        normalized.add(safe_name)
        sha256(content)
    return value


def exact_count(value, expected):
    return count(value) == expected


def closed_current(value):
    require(value.get("source_sha256") == value.get("source_sha256_after") == CURRENT,
        "Current source before/after identity differs")
    require(artifact_map(value["artifacts_sha256"]) == value["artifacts_sha256_after"],
        "Experiment helper bytes changed")
    require(value.get("cleanup_failures") == [] and value.get("environment_restored") is True,
        "Experiment cleanup/environment closure failed")


def actual_cuda(profile):
    backend = profile["backend"]
    require(backend.get("requested") == "cuda" and backend.get("device") == "CUDA:0"
        and backend.get("tracking") == "legacy" and backend.get("tracking_device") == "CPU:0"
        and backend.get("cuda_available") is True and backend.get("fallback_reason") is None,
        "Original actual CUDA fusion/CPU tracking backend required")
    native = backend["native_kernels"]
    require(native.get("active") is True and native.get("requested") == "on" and native.get("api_version") == 2,
        "Original native CPU kernels unavailable")
    require(backend["geometric_verification"]["implementation"] == "legacy"
        and backend["confidence_cuda"]["requested"] == backend["confidence_cuda"]["implementation"] == "fused"
        and backend["confidence_cuda"].get("fallback_reason") is None,
        "Actual geometry/fused weighted backend differs")
    inp = backend["cuda_input"]
    require(inp.get("requested") == "auto" and inp.get("implementation") == "cuda"
        and inp.get("device") == "CUDA:0" and inp.get("probe_passed") is True
        and count(inp["gpu_batches"], True) > 0 and exact_count(inp["cpu_batches"],0)
        and exact_count(inp["fallback_batches"],0),
        "Actual exact CUDA input path/probe differs")
    matching = backend["descriptor_matching"]
    require(matching.get("requested") == matching.get("implementation") == "cuda"
        and count(matching["cuda_batches"], True) > 0 and exact_count(matching["cpu_batches"],0)
        and matching.get("fallback_reason") is None, "Actual descriptor backend differs")
    confidence = backend["depth_confidence"]
    require(confidence.get("requested") == "off" and confidence.get("implementation") == "cpu"
        and confidence.get("device") == "CPU:0" and exact_count(confidence["gpu_calls"],0)
        and exact_count(confidence["fallback_calls"],0)
        and count(confidence["cpu_calls"], True) > 0 and profile["settings"]["confidence_fusion"] is True,
        "CUDA confidence off must retain the CPU confidence algorithm")


def full_profile(value, source):
    require(value.get("schema_version") == 1 and value.get("source_sha256") == source
        and value.get("source_changed_during_profile") is False and value.get("input_changed_during_profile") is False
        and value.get("pose_seeds_used") is False and value.get("finish_requested") is True
        and value.get("mesh_built") is True and value.get("experimental_visual_fallback") is False
        and not value.get("checkpoint") and not value.get("research_finish"), "Completed unseeded full raw profile required")
    sha256(value["input_sha256"])
    frames = count(value["frames"], True)
    selection = value["selected_indices"]
    require(frames <= 1000 and type(selection) is list and selection == list(range(frames))
        and all(type(i) is int for i in selection) and len(value["live_diagnostics"]) == frames,
        "All stored raw views in original order required")
    retained = value["accepted_indices"]
    require(type(retained) is list and retained == sorted(set(retained))
        and all(type(i) is int and 0 <= i < frames for i in retained)
        and len(retained) == count(value["accepted"], True), "Final coverage is malformed")
    live = value["accepted_indices_before_finish"]
    require(type(live) is list and live == sorted(set(live))
        and all(type(i) is int and 0 <= i < frames for i in live)
        and len(live) == count(value["accepted_before_finish"], True), "Live coverage counter/list differs")
    poses = value["poses"]
    require(type(poses) is list and [row.get("index") for row in poses] == retained,
        "Final pose membership differs from view coverage")
    for row in poses:
        matrix = row.get("camera_to_world")
        require(type(matrix) is list and len(matrix) == 4 and all(type(r) is list and len(r) == 4 for r in matrix)
            and all(type(x) in (int,float) and math.isfinite(x) for r in matrix for x in r),
            "Finite full-precision Final pose entries required")
    require(value["native_mode"] == "on" and value["native_extension"]["changed_during_profile"] is False,
        "Native input changed during profile")
    sha256(value["native_extension"]["sha256"])
    require(value["settings"]["confidence_fusion"] is True and value["settings"]["final_voxel_m"] == .005,
        "Weighted 5mm Final settings required")
    actual_cuda(value)
    for name in ("live_s", "finish_s", "processing_s"):
        number(value[name], True)
    require(math.isclose(value["processing_s"], value["live_s"]+value["finish_s"], abs_tol=1e-6),
        "Processing wall does not equal its phases")


def pair_scope(old, new):
    full_profile(old, BASELINE)
    full_profile(new, CURRENT)
    for key in ("input_sha256", "session", "frames", "selected_indices", "seed", "settings", "settings_overrides",
            "original_settings_sha256", "versions", "native_mode", "initial_blocks", "thread_policy", "omp_threads",
            "pipeline_options", "gpu_hardware", "accepted_indices", "accepted_indices_before_finish"):
        require(old[key] == new[key], "Historical/current raw settings/selection/runtime/coverage differ: "+key)
    require(old["native_extension"]["sha256"] == new["native_extension"]["sha256"], "Native extension transition differs")
    require(new["pipeline_options"]["KINECT_LIVE_RECOVERY"] == "full"
        and new["pipeline_options"]["KINECT_CUDA_CONFIDENCE"] == "off" and new["seed"] == 0,
        "Original full recovery/CPU confidence policy required")
    return {"input_sha256":new["input_sha256"], "settings_sha256":canonical(new["settings"]),
        "selected_indices_sha256":canonical(new["selected_indices"]), "pipeline_options_sha256":canonical(new["pipeline_options"]),
        "runtime_scope_sha256":canonical({key:new[key] for key in ("versions", "thread_policy", "omp_threads", "gpu_hardware")}),
        "native_extension_sha256":new["native_extension"]["sha256"], "frames":new["frames"],
        "live_views":count(new["accepted_before_finish"], True), "final_views":new["accepted"],
        "confidence_fusion":True, "cuda_confidence_acceleration":"off", "confidence_algorithm":"cpu"}


def allocation(old, new):
    a, b = old["final_reconstruction"], new["final_reconstruction"]
    limit = count(new["settings"]["final_block_count"], True)
    require(a.get("applied") is True and b.get("applied") is True and a.get("voxel_m") == b.get("voxel_m") == .005,
        "Successful requested Final model required")
    require(a.get("block_limit") == b.get("block_limit") == limit, "Final logical allowance differs")
    required = count(b["required_blocks"], True)
    require(required <= limit and all(count(b[key], True) == required for key in
        ("blocks", "requested_block_capacity", "allocated_blocks", "initial_block_capacity"))
        and b.get("allocation_strategy") == "exact missing-key activation" and a.get("blocks") == required,
        "Actual Final count/initial capacity/final capacity differs or grew")
    # Five float32 values per 16^3 voxel block: TSDF, weight and RGB.
    floor_per_block = 16**3 * 5 * 4 / 1024**2
    configured = number(a["attribute_budget_mib"], True)
    require(configured == limit*floor_per_block
        and b["configured_attribute_budget_mib"] == configured
        and number(b["attribute_budget_mib"], True) == required*floor_per_block,
        "Attribute floor/declared allowance arithmetic differs")
    require(not any(key in a for key in ("allocated_blocks", "initial_block_capacity")),
        "Historical schema unexpectedly observes capacity; revise rather than relabel it")
    return {"logical_block_limit":limit, "required_blocks":required,
        "historical_configured_attribute_budget_mib":configured, "historical_actual_block_capacity":None,
        "current_initial_block_capacity":required, "current_actual_block_capacity":required,
        "current_attribute_allocation_floor_mib":b["attribute_budget_mib"],
        "planning_elapsed_ms":number(b["planning_elapsed_ms"]), "capacity_unchanged":True,
        "memory_scope":"Configured historical allowance versus measured current attributes; not peak GPU/process memory."}


def worker(reports, execution_name, profile_name, profile, *, current):
    execution = reports.read(execution_name)
    expected_source = CURRENT if current else BASELINE
    require(execution.get("status") == "complete" and execution.get("source_sha256") == expected_source,
        "Raw execution is incomplete or belongs to another source")
    artifact_map(execution["artifacts_sha256"])
    key = "profile" if current else "report"
    digest_key = "profile_sha256" if current else "report_sha256"
    rows = [row for row in execution["runs"] if Path(row.get(key, "")).name == Path(profile_name).name]
    require(len(rows) == 1, "Expected one closed raw worker for each profile")
    row = rows[0]
    reports.matches_path(profile_name, row[key])
    require(row.get("status") == "complete" and row.get("exit_code") == 0
        and row.get("child_wait_completed") is True and row.get(digest_key) == reports.ref(profile_name)["sha256"]
        and row.get("input_sha256") == profile["input_sha256"] and row.get("mesh_built") is True,
        "Raw child/report/input completion is not closed")
    require(row["live_s"] == profile["live_s"] and row["finish_s"] == profile["finish_s"], "Worker phase wall differs")
    require(row.get("accepted_before_finish") == profile["accepted_before_finish"]
        and row.get("accepted_after_finish" if current else "accepted") == profile["accepted"],
        "Worker Live/Final view counters differ from the closed profile")
    log_name = profile_name.removesuffix(".json")+".log"
    reports.matches_path(log_name, row["log"])
    reports.attachment(log_name, row["log_sha256"])
    result = {"execution":reports.ref(execution_name), "exit_code":0, "child_wait_completed":True,
        "owned_tree_closed":None, "historical_tree_ownership_recorded":False}
    if current:
        require(execution.get("kind") == "raw-field-existing-recovery-policy-experiment-v1"
            and row.get("owned_tree_closed") is True and row.get("process_cleanup_failures") == []
            and row.get("process_ownership") == "windows-kill-on-close-job"
            and count(row.get("child_pid"), True) > 0, "Current owned process tree/cleanup is not closed")
        require(execution["expected_thread_policy"] == profile["thread_policy"]
            and execution["native_extension"]["sha256"] == profile["native_extension"]["sha256"],
            "Current controller actual runtime differs")
        geometry_name = profile_name.removesuffix(".json")+".geometry.npz"
        reports.matches_path(geometry_name, profile["geometry"]["artifact"])
        result["geometry"] = reports.attachment(geometry_name, row["geometry_sha256"])
        result.update(owned_tree_closed=True, historical_tree_ownership_recorded=None)
    return result


def quality(reports, name, old_name, new_name, old, new):
    value = reports.read(name)
    require(value.get("baseline_report_sha256") == reports.ref(old_name)["sha256"]
        and value.get("candidate_report_sha256") == reports.ref(new_name)["sha256"]
        and value.get("same_mesh_success") is True and value.get("same_accepted_indices") is True
        and value.get("common_poses") == len(new["accepted_indices"]), "Surface/pose report reference or coverage differs")
    require(value.get("preview_resolution_comparison") is False, "Changed preview comparison is outside this validation")
    translation = number(value["max_pose_translation_delta_m"])
    rotation = number(value["max_pose_rotation_delta_deg"])
    require(translation <= .0005 and rotation <= .1, "Declared final-pose limits failed")
    surface = value["triangle_surface_metrics_vs_cpu"]
    metric = {key:number(surface[key]) for key in ("threshold_m", "samples_per_surface", "precision", "completeness",
        "surface_p95_m", "surface_rmse_m")}
    require(metric["threshold_m"] == .005 and type(surface["samples_per_surface"]) is int
        and metric["samples_per_surface"] == 30000 and .999 <= metric["precision"] <= 1
        and .999 <= metric["completeness"] <= 1 and metric["surface_p95_m"] <= .0005
        and surface.get("alignment") == "fixed input coordinate frame; no scale or trajectory fitting",
        "Fixed physical surface contract failed")
    metric["alignment"] = "fixed input coordinate frame; no fitting"
    return {"evidence":reports.ref(name), "final_pose_translation_max_m":translation,
        "final_pose_rotation_max_deg":rotation, "surface":metric,
        "vertex_distance_max_m":number(value["symmetric_vertex_distance_m"]["max"]),
        "geometry_passed":True, "bit_identical_pose_claim":False, "performance_authority":False}


def quality_execution(reports, name, observations_name, label, quality_name, old_name, new_name, old, new):
    """Fresh comparer inputs are observed before/after, not retroactively inferred."""
    value = reports.read(name)
    require(value.get("kind") == "production-allocation-fixed-surface-quality-execution-v1"
        and value.get("status") == "complete" and value.get("source_sha256") == value.get("source_sha256_after") == CURRENT,
        "Fresh physical quality controller/source is not closed")
    inputs = value["inputs_sha256"]
    require(type(inputs) is dict and bool(inputs) and inputs == value["inputs_sha256_after"],
        "Quality raw/mesh/profile/native inputs changed")
    for content in inputs.values():sha256(content)
    normalized = {str(Path(path).resolve()):content for path,content in inputs.items()}
    require(len(normalized) == len(inputs), "Ambiguous quality input path aliases")
    artifacts = artifact_map(value["artifacts_sha256"])
    require(artifacts == value["artifacts_sha256_after"] and set(artifacts) == {
        "scripts/compare_session_profiles.py", "scripts/profile_session.py", "shared/surface_metrics.py"},
        "Physical quality source helpers changed or scope differs")
    rows = [row for row in value["runs"] if row.get("label") == label]
    require(len(rows) == 1, "Exactly one physical quality child per session required")
    row = rows[0]
    require(row.get("exit_code") == 0 and row.get("child_wait_completed") is True,
        "Physical quality child did not exit successfully")
    for path_key, report_name, hash_key in (("baseline",old_name,"baseline_report_sha256"),
            ("candidate",new_name,"candidate_report_sha256"), ("quality",quality_name,"quality_sha256")):
        reports.matches_path(report_name,row[path_key])
        require(row[hash_key] == reports.ref(report_name)["sha256"], "Quality execution report bytes differ")
        if path_key != "quality":
            require(normalized[str((reports.root/report_name).resolve())] == row[hash_key],
                "Quality input report was not observed before/after")
    geometry_refs = {}
    def bound_raw_input(profile):
        # profile_session intentionally records args.session.name, while the
        # controller records actual full input paths. A basename authorizes no
        # other input: exactly one closed path with this name and raw digest is
        # required. Older full-path profiles still bind exact resolved identity.
        recorded = profile["session"]
        require(type(recorded) is str and bool(recorded), "Missing raw archive identity")
        if Path(recorded).name == recorded:
            require(recorded.endswith(".zip"), "Unexpected raw archive basename")
            matches = [content for path,content in normalized.items() if Path(path).name == recorded]
            require(len(matches) == 1 and matches[0] == profile["input_sha256"],
                "Quality raw archive basename/digest is absent or ambiguous")
        else:
            require(normalized[str(Path(recorded).resolve())] == profile["input_sha256"],
                "Quality raw archive full path/digest differs")
    for role, profile_name, profile in (("baseline",old_name,old),("candidate",new_name,new)):
        geometry_name = profile_name.removesuffix(".json")+".geometry.npz"
        reports.matches_path(geometry_name,profile["geometry"]["artifact"])
        content = row[role+"_geometry_sha256"]
        require(normalized[str((reports.root/geometry_name).resolve())] == content,
            "Quality geometry payload was not observed before/after")
        geometry_refs[role] = reports.attachment(geometry_name,content)
        bound_raw_input(profile)
        require(normalized[str(Path(profile["native_extension"]["path"]).resolve())]
                == profile["native_extension"]["sha256"], "Quality raw ZIP/native binary scope differs")
    log_name = quality_name.removesuffix(".json")+".log"
    reports.matches_path(log_name,row["log"])
    reports.attachment(log_name,row["log_sha256"])
    controller = reports.read(observations_name)
    require(controller.get("geometry_execution") == {"path":name,"sha256":reports.ref(name)["sha256"]},
        "Root completion does not bind the fresh geometry execution")
    array_equal = row["mesh_arrays_equal"]
    require(type(array_equal) is dict and set(array_equal) == {"points","faces"}
        and all(type(v) is bool for v in array_equal.values()), "Malformed ordered-mesh diagnostics")
    return {"evidence":reports.ref(name), "inputs_before_after_sha256":canonical(inputs),
        "geometry_payloads":geometry_refs, "exit_code":0, "child_wait_completed":True,
        "ordered_mesh_arrays_equal":dict(array_equal), "ordered_mesh_array_identity_required":False,
        "scope":"Observed raw/profile/geometry/native/helper bytes unchanged before and after the fixed-frame comparison."}


def smoke(reports, name, current_profile):
    value = reports.read(name)
    require(value.get("kind") == "production-weighted-final-missing-activation-physical-smoke-v1"
        and value.get("status") == "passed" and value.get("performance_claim") is False
        and value.get("whole_session_quality_proven") is False, "Installed physical smoke did not pass")
    closed_current(value)
    runtime = value["runtime_binding"]
    require(runtime == value["runtime_binding_after"] and runtime["open3d_threads"] == 20
        and all(runtime["versions"][k] == current_profile["versions"][k] for k in ("numpy", "open3d"))
        and runtime["gpu"] == current_profile["gpu_hardware"], "Physical loaded runtime/device closure differs")
    native = [item for path, item in runtime["binaries"].items() if Path(path).name.startswith("_kinect_native")]
    require(len(native) == 1 and native[0]["sha256"] == current_profile["native_extension"]["sha256"],
        "Physical native extension identity differs from raw replay")
    pairs = value["pairs"]
    require(type(pairs) is list and len(pairs) == 3 and [p.get("backend") for p in pairs]
        == ["cpu-original", "cuda-tensor", "cuda-fused"], "All original weighted backends required")
    rows = []
    for pair in pairs:
        require(pair.get("device") == ("CPU:0" if pair["backend"] == "cpu-original" else "CUDA:0"),
            "Physical selected backend device differs")
        require(pair.get("complete") is True and pair.get("inputs_unchanged") is True
            and pair["inputs"] == pair["inputs_after"] and count(pair["fractional_confidence_pixels"], True) > 0
            and count(pair["nonzero_weight_voxels"], True) > 0 and count(pair["fractional_weight_voxels"], True) > 0,
            "Physical input/fractional-confidence/nonvacuous coverage failed")
        old, new = pair["ordinary"], pair["candidate"]
        require(old["initial_capacity"] == new["initial_capacity"] == new["final_capacity"] == 4
            and old["final_capacity"] == 8 and old["final_blocks"] == new["final_blocks"] == 4
            and [f["capacity"] for f in old["frames"]] == [4,8,8]
            and [f["capacity"] for f in new["frames"]] == [4,4,4]
            and all([f["active_keys"] for f in mode["frames"]] == [3,4,4] for mode in (old,new)),
            "Physical causal capacity/active key progression differs")
        for attribute, elements in (("tsdf",16384), ("weight",16384), ("color",49152)):
            comparison = pair["comparison"][attribute]
            require(type(comparison["elements"]) is int and comparison["elements"] == elements
                and type(comparison["bit_mismatches"]) is int and comparison["bit_mismatches"] == 0
                and old["attribute_snapshots"][attribute] == new["attribute_snapshots"][attribute],
                "Installed per-key attribute bit comparison failed")
            descriptor = new["attribute_snapshots"][attribute]
            channels = 3 if attribute == "color" else 1
            require(descriptor.get("shape") == [4,4096,channels] and descriptor.get("dtype") == "<f4"
                and descriptor.get("bytes") == elements*4, "Physical original float32 attribute descriptor differs")
            sha256(descriptor["sha256"])
        rows.append({"backend":pair["backend"], "device":pair["device"], "original_final_capacity":8,
            "candidate_final_capacity":4, "float32_values_compared":81920, "bit_mismatches":0,
            "nonzero_weight_voxels":pair["nonzero_weight_voxels"], "fractional_weight_voxels":pair["fractional_weight_voxels"]})
    return {"evidence":reports.ref(name), "runtime_binding_sha256":canonical(runtime), "backends":rows,
        "capacity_and_attribute_bits_passed":True, "whole_session_authority":False}


def budget_probe(reports, name, profile_name, profile):
    value = reports.read(name)
    require(value.get("kind") == "current-full-raw-final-budget-boundary-probe-v2" and value.get("status") == "passed",
        "Fresh corrected budget boundary probe required")
    closed_current(value)
    require(value.get("geometry_authority") is False and value.get("registration_authority") is False
        and value.get("performance_claim") is False and value.get("owner_state_unchanged") is True,
        "Budget probe scope/owner closure differs")
    inputs = value["inputs_sha256"]
    require(inputs == value["inputs_sha256_after"] and inputs["profile"] == reports.ref(profile_name)["sha256"]
        and inputs["raw_zip"] == profile["input_sha256"], "Budget probe uses another raw replay")
    require(value["native_extension_sha256"] == value["native_extension_sha256_after"]
        == value["actual_native_sha256"] == profile["native_extension"]["sha256"]
        and value["gpu_hardware"] == value["gpu_hardware_after"] == profile["gpu_hardware"]
        and value["actual_versions"] == profile["versions"], "Budget native/device runtime changed")
    boundary = value["boundary"]
    require(boundary.get("candidate_allocated") is False and boundary.get("tracking_or_fusion_called") is False
        and boundary.get("hooks_restored") is True and boundary.get("required_blocks") == 13302
        and boundary.get("logical_limit") == 10000, "Logical10k boundary did not stop before candidate allocation")
    creations = boundary["create_calls"]
    require(type(creations) is list and len(creations) == 1 and creations[0].get("requested_blocks") == 1
        and creations[0].get("actual_capacity") == 1 and creations[0].get("complete") is True,
        "Probe did not use exactly one native block1 planner scratch")
    plans = boundary["planner_calls"]
    require(type(plans) is list and len(plans) == 1 and plans[0].get("complete") is True
        and plans[0].get("required_blocks") == 13302 and plans[0].get("stage") == "final_capacity_preflight"
        and plans[0].get("views") == profile["accepted"] and boundary.get("original_error_type") == "ValueError",
        "Original single allocation planner/typed early error closure differs")
    require(value.get("actual_device") == profile["backend"]["device"]
        and value.get("actual_threads") == {"open3d":profile["thread_policy"]["open3d_threads"],
            "opencv":profile["thread_policy"]["opencv_threads"], "omp":profile["omp_threads"]},
        "Budget actual selected device/thread policy differs")
    preflight = value["preflight"]
    representation = value["settings_reconstruction"]
    require(preflight.get("frames") == profile["frames"] and preflight.get("final_views") == profile["accepted"]
        and preflight.get("required_blocks") == 13302 and preflight.get("settings_sha256") == canonical(profile["settings"])
        and representation.get("sha256") == canonical(profile["settings"])
        and representation.get("numeric_tolerance") == 0,
        "Budget current raw/settings zero-tolerance provenance differs")
    serializer = value["pose_serializer"]
    require(serializer.get("original_p_tolist_expression") is True and serializer.get("loaded_code_checked") is True,
        "Fresh raw full-precision allocation pose provenance missing")
    return {"evidence":reports.ref(name), "fresh_profile":reports.ref(profile_name), "logical_limit":10000,
        "required_blocks":13302, "candidate_allocated":False, "scratch_block_capacity":1,
        "owner_and_hooks_unchanged":True, "pose_scope":"Completed current raw replay poses used for allocation only; never tracking seeds.",
        "geometry_authority":False, "registration_authority":False, "performance_authority":False}


def observed_completion(reports, observations_name, report_name):
    """Root tool completion is explicitly separate from owned replay trees."""
    controller = reports.read(observations_name)
    require(controller.get("kind") == "root-completed-exec-observations-v1" and controller.get("status") == "complete",
        "Closed root tool observations required")
    matches = [row for row in controller["observations"] if row.get("report",{}).get("path") == report_name]
    require(len(matches) == 1, "Missing/duplicated completed tool observation")
    row = matches[0]
    require(row["report"].get("sha256") == reports.ref(report_name)["sha256"]
        and row.get("exit_code") == 0 and row.get("tool_completed") is True and row.get("pid_recorded") is False
        and type(row.get("tool_chunk_id")) is str and bool(row["tool_chunk_id"])
        and type(row.get("executable")) is str and bool(row["executable"])
        and type(row.get("argv")) is list and bool(row["argv"]), "Actual root wait/report/exit observation failed")
    if row.get("log"):
        reports.attachment(relative(row["log"]["path"]), row["log"]["sha256"])
    return {"controller":reports.ref(observations_name), "exit_code":0, "tool_completed":True,
        "pid_recorded":False, "owned_process_tree_claim":False}


def failure(reports, name):
    value = reports.read(name)
    require(value.get("status") == "failed" and type(value.get("failure")) is dict,
        "Prior failure must remain failed")
    failure_type = value["failure"]["type"]
    require(type(failure_type) is str and failure_type.isidentifier(), "Malformed failure type")
    return {"evidence":reports.ref(name), "reported_status":"failed", "failure_type":value["failure"]["type"],
        "message_omitted_for_privacy":True, "numerical_authority":False}


def summarize(reports, *, current_source, smoke_name="production-allocation-smoke-v2/report.json",
        probe_name="production-final-budget-v2/report.json", observations_name="root-completed-exec-observations-v1.json",
        quality_directory="production-memory-quality-v2"):
    require(current_source == CURRENT, "Installed source differs from the declared production evidence")
    rows = []
    for label, (old_name, new_name, old_exec, new_exec) in PAIRS.items():
        old, new = reports.read(old_name), reports.read(new_name)
        scoped = pair_scope(old, new)
        row = {"session":label, "baseline":reports.ref(old_name), "current":reports.ref(new_name),
            "matched_raw_scope":scoped, "allocation":allocation(old, new),
            "baseline_worker":worker(reports,old_exec,old_name,old,current=False),
            "current_worker":worker(reports,new_exec,new_name,new,current=True)}
        quality_name = relative(quality_directory)+"/"+label+".json"
        row["geometry"] = quality(reports,quality_name,old_name,new_name,old,new)
        row["geometry"]["physical_input_closure"] = quality_execution(reports,
            relative(quality_directory)+"/execution.json",observations_name,label,quality_name,old_name,new_name,old,new)
        row["geometry"]["completion"] = observed_completion(reports,observations_name,quality_name)
        row["observed_phase_walls_s"] = {"historical":{k:number(old[k],True) for k in ("live_s","finish_s","processing_s")},
            "current":{k:number(new[k],True) for k in ("live_s","finish_s","processing_s")}}
        row["causal_speed_claim"] = False
        rows.append(row)
    current_five = reports.read(PAIRS["chest-5"][1])
    physical = smoke(reports,smoke_name,current_five)
    physical["completion"] = observed_completion(reports,observations_name,smoke_name)
    c7_name = PAIRS["chest-7"][1]
    probe = budget_probe(reports,probe_name,c7_name,reports.read(c7_name))
    probe["completion"] = observed_completion(reports,observations_name,probe_name)
    result = {"kind":KIND, "status":"complete", "baseline_source_sha256":BASELINE,
        "installed_source_sha256":CURRENT, "sessions":rows, "installed_physical_proof":physical,
        "allocation_only_early_failure":probe, "preserved_failures":[
            failure(reports,"production-allocation-smoke-v1/report.json"),
            failure(reports,"production-final-budget-v1/report.json")],
        "geometry_authority":False, "registration_authority":False, "performance_authority":False,
        "causal_speed_claim":False, "memory_scope":"Historical configured Final allowance versus measured installed allocation; attribute floor, not peak RSS/VRAM.",
        "limits":["Private local raw ZIPs and geometry payloads are required to reproduce numerical measurements.",
            "Stored selected-view replay is not continuous camera FPS.",
            "Historical raw profiles do not record actual old native capacity or owned process-tree closure.",
            "Root completion observations lack a returned PID and do not establish owned-tree identity.",
            "The source transition and single-run phase walls are descriptive; no causal speed gain is inferred.",
            "Surface/pose bounds are declared observable comparisons; no bit-identical trajectory or ground-truth claim."],
        "evidence":list(reports.files.values())}
    reports.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--smoke-report", default="production-allocation-smoke-v2/report.json")
    parser.add_argument("--probe-report", default="production-final-budget-v2/report.json")
    parser.add_argument("--observations", default="root-completed-exec-observations-v1.json")
    parser.add_argument("--quality-directory", default="production-memory-quality-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve published/failed reports; require a fresh output")
    args.output.resolve().relative_to(ROOT/"benchmark-output")
    before = pins()
    result = summarize(Reports(args.study_root), current_source=source_hash(), smoke_name=args.smoke_report,
        probe_name=args.probe_report, observations_name=args.observations, quality_directory=args.quality_directory)
    require(source_hash() == CURRENT and pins() == before, "Installed core/publication producer changed")
    result["publication_artifacts_sha256"] = before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output), "status":result["status"], "performance_authority":False}))


if __name__ == "__main__":
    main()
