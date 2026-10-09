"""Separate component timing authority; no old proof is relabelled or extended.

The original combined proof certifies changed orchestration and equation bits.
The original Device proof certifies the unchanged resident retrieval interface.
New timing source pins and the exact shared runtime/fixture are required. No
whole-Finish authority or production registration policy is created here.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from scripts.research import validate_combined_sync_proof as combined
from scripts.research import validate_device_flat_grid_proof as device
from scripts.research.archive.validate_uniform_grid_proof import GridProofError, canonical_hash, require, file_hash

ROOT = Path(__file__).resolve().parents[2]
POLICY = "original-flat-normal-single-copy-unaudited-component-v1"
ARTIFACTS = ("scripts/research/combined_sync_timing_adapter.py",
    "scripts/research/benchmark_combined_sync_timing.py",
    "scripts/research/validate_combined_sync_timing.py",
    "tests/test_combined_sync_timing_contract.py")
SHARED_BINDING_KEYS = ("source_sha256", "component_source_sha256", "domain", "thread_policy",
    "resident_configuration", "resident_math", "cache_policy", "gpu", "versions")


@dataclass(frozen=True)
class CombinedSyncTimingAuthority:
    combined_authority: combined.CombinedSyncProofAuthority
    device_authority: device.DeviceFlatGridProofAuthority
    runtime_binding_json: str
    timing_artifact_sha256: tuple
    fixture_binding_sha256: str
    proof_report_files: tuple
    policy: str = POLICY

    @property
    def runtime_binding(self):
        return json.loads(self.runtime_binding_json)


_ACTIVE = {}


def validate_timing_authority(combined_synthetic, combined_bridge, device_synthetic, device_bridge,
        expected_combined_binding, expected_device_binding, expected_fixture, expected_artifacts):
    """Freshly validate both complete proofs against actual rebuilt configuration."""
    a = combined.validate_combined_sync_proof(combined_synthetic, combined_bridge,
        expected_combined_binding, expected_fixture)
    b = device.validate_grid_proof(device_synthetic, device_bridge, expected_device_binding, expected_fixture)
    require(type(a) is combined.CombinedSyncProofAuthority and type(b) is device.DeviceFlatGridProofAuthority,
            "Require separate exact combined and unchanged Device component authorities")
    require(all(expected_combined_binding[key] == expected_device_binding[key] for key in SHARED_BINDING_KEYS)
            and a.target_digests == b.target_digests,
            "Original math/configuration/raw fixture/target membership differs between independent proofs")
    pins = device.normalized_artifacts(expected_artifacts)
    require(set(pins) == set(ARTIFACTS) and all(file_hash(ROOT/name) == value for name,value in pins.items()),
            "Current separate timing adapter/driver/guard/contract sources differ")
    value = CombinedSyncTimingAuthority(a,b,json.dumps(expected_combined_binding,sort_keys=True,
        separators=(",", ":"),allow_nan=False),tuple(sorted(pins.items())),canonical_hash(expected_fixture),
        tuple((str(Path(path).resolve()),digest) for path,digest in (
            (combined_synthetic,a.synthetic_report_sha256),(combined_bridge,a.bridge_report_sha256),
            (device_synthetic,b.synthetic_report_sha256),(device_bridge,b.bridge_report_sha256))))
    # Immutable caller-constructed dataclasses do not grant activation. Each
    # successful complete proof validation owns a separate registered token.
    while len(_ACTIVE) >= 4:
        del _ACTIVE[next(iter(_ACTIVE))]
    _ACTIVE[id(value)] = value
    return value


def validate_activation(authority, device_authority, actual_configuration, actual_cache):
    require(type(authority) is CombinedSyncTimingAuthority and _ACTIVE.get(id(authority)) is authority,
            "Only a freshly validated new component timing authority can activate")
    require(type(device_authority) is device.DeviceFlatGridProofAuthority
            and device_authority == authority.device_authority and authority.policy == POLICY,
            "Unaudited retrieval must use the exact independently validated original Device authority")
    runtime = authority.runtime_binding
    require(actual_configuration == runtime["resident_configuration"] and actual_cache == {
        key:runtime["cache_policy"][key] for key in ("max_clouds", "retained_gpu_bytes")},
        "Timing actual selected device/query/cache/scratch/point/event configuration changed")
    require(all(file_hash(Path(path)) == digest for path,digest in authority.proof_report_files),
            "Current component proof bytes changed before timing activation")
    for token in (authority.combined_authority, authority.device_authority):
        for name, value in token.artifact_sha256:
            require(file_hash(ROOT/name) == value, "Inherited audited source bytes changed before activation")
    require(all(file_hash(ROOT/name) == value for name,value in authority.timing_artifact_sha256),
            "Separate timing source bytes changed before activation")
    return authority
