"""Research method: bounded first-representative reuse of original CPU ICP.

Actual seeds are never rounded. Exact directed cloud value bytes define a
bucket. A hit reconstructs a new result from an immutable original result;
fresh original same-input CPU shadows are mandatory during the audit phase.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import math
import struct
from types import SimpleNamespace, MappingProxyType
from typing import NamedTuple
from scripts.research import microbatch_bridge_scope as scope

POLICY = "exact-cloud-first-representative-near-seed-cpu-reuse-v1"
THRESHOLDS = (1e-10, 1e-8)


def require(ok, message):
    if not ok: raise scope.BridgeFailure(message)


def seed_distance(a, b):
    require(len(a) == len(b) == 128, "Original FP64 4x4 seed bits required")
    x, y = struct.unpack("<16d", a), struct.unpack("<16d", b)
    require(all(math.isfinite(v) for v in x+y), "Finite unrounded seed bits required")
    return max(abs(p-q) for p, q in zip(x, y))


class FrozenResult(NamedTuple):
    pose: bytes
    raw_pairs: bytes
    pair_dtype: str
    pair_shape: tuple
    evidence_json: str
    fitness: float
    rmse: float

    @property
    def owned_bytes(self):
        return len(self.pose)+len(self.raw_pairs)+len(self.evidence_json.encode())


def freeze_result(np, result, n, m):
    evidence = scope.result_evidence(np, result, n, m)
    pose, pairs = np.asarray(result.transformation), np.asarray(result.correspondence_set)
    return FrozenResult(pose.tobytes(order="C"), pairs.tobytes(order="C"), pairs.dtype.str,
        tuple(pairs.shape), scope.canonical(evidence), float(result.fitness), float(result.inlier_rmse))


def restore_result(np, value):
    # Both arrays are new, writable owners. Gates cannot mutate cached bytes.
    return SimpleNamespace(transformation=np.frombuffer(value.pose, dtype="<f8").reshape(4, 4).copy(),
        correspondence_set=np.frombuffer(value.raw_pairs, dtype=value.pair_dtype).reshape(value.pair_shape).copy(),
        fitness=value.fitness, inlier_rmse=value.rmse)


class RepresentativeCache:
    """Bounded LRU eviction; representatives retain immutable insertion order."""
    def __init__(self, threshold=1e-10, max_entries=256, max_clouds=64, max_bytes=64*1024**2):
        require(type(threshold) is float and threshold in THRESHOLDS, "Declared near-seed threshold required")
        require(all(type(v) is int and v > 0 for v in (max_entries, max_clouds, max_bytes)), "Positive explicit cache caps required")
        self.configuration = MappingProxyType({"threshold": threshold, "max_entries": max_entries, "max_clouds": max_clouds, "max_bytes": max_bytes})
        self.configuration_owner = self.configuration; self.configuration_snapshot = tuple(sorted(self.configuration.items()))
        self.entries, self.buckets = OrderedDict(), OrderedDict()
        self.next_id, self.bytes, self.peak_bytes, self.evictions = 0, 0, 0, 0
        self.closed = False

    def healthy(self):
        require(not self.closed and self.configuration is self.configuration_owner
            and tuple(sorted(self.configuration.items())) == self.configuration_snapshot,
            "Cache effective configuration changed")
        require(0 <= self.bytes <= self.configuration["max_bytes"] and len(self.entries) <= self.configuration["max_entries"]
            and len(self.buckets) <= self.configuration["max_clouds"], "Bounded cache accounting failed")

    def _drop(self, index):
        bucket, seed, payload, size = self.entries.pop(index)
        self.bytes -= size; self.evictions += 1
        self.buckets[bucket].remove(index)
        if not self.buckets[bucket]: del self.buckets[bucket]

    def find(self, bucket, seed):
        self.healthy()
        for index in self.buckets.get(bucket, ()):
            _, representative, payload, _ = self.entries[index]
            delta = seed_distance(seed, representative)
            if delta <= self.configuration["threshold"]:
                self.entries.move_to_end(index); self.buckets.move_to_end(bucket)
                return index, representative, payload, delta
        return None

    def insert(self, bucket, seed, payload):
        self.healthy()
        require(type(seed) is bytes and type(payload) is FrozenResult and type(payload.pose) is bytes
            and type(payload.raw_pairs) is bytes and type(payload.evidence_json) is str and type(payload.pair_shape) is tuple,
            "Owned immutable representative required")
        seed_distance(seed, seed)
        size = len(bucket.encode())+len(seed)+payload.owned_bytes
        if size > self.configuration["max_bytes"]: return None
        if bucket not in self.buckets:
            while len(self.buckets) >= self.configuration["max_clouds"]:
                for index in list(next(iter(self.buckets.values()))): self._drop(index)
        while self.entries and (len(self.entries) >= self.configuration["max_entries"] or self.bytes+size > self.configuration["max_bytes"]):
            self._drop(next(iter(self.entries)))
        index = self.next_id; self.next_id += 1
        self.entries[index] = (bucket, seed, payload, size)
        self.buckets.setdefault(bucket, []).append(index)
        self.buckets.move_to_end(bucket); self.bytes += size; self.peak_bytes = max(self.peak_bytes, self.bytes)
        return index

    def close(self):
        self.entries.clear(); self.buckets.clear(); self.bytes = 0; self.closed = True


class NearSeedMatcher:
    def __init__(self, np, original, route, *, audit=True, timing_authorizer=None, **configuration):
        require(type(audit) is bool, "Explicit audit mode required")
        require(audit or callable(timing_authorizer), "Timing requires own closed-audit actual-input authorizer")
        self.np, self.original, self.route, self.audit = np, original, route, audit
        self.timing_authorizer = timing_authorizer
        self.owners = (np, original, route, audit, timing_authorizer)
        self.cache = RepresentativeCache(**configuration)
        self.cache_owner = self.cache
        self.configuration_snapshot = tuple(sorted(self.cache.configuration.items()))
        self.previous_peak, self.previous_evictions, self.cleared_pairs = 0, 0, 0
        self.failure = None; self.last_receipt = None
        self.statistics = {"calls": 0, "hits": 0, "misses": 0, "audited_hits": 0, "original_cpu_calls": 0, "unretained_misses": 0}

    def check_configuration(self):
        require(self.cache is self.cache_owner
            and tuple(sorted(self.cache.configuration.items())) == self.configuration_snapshot,
            "Method effective cache configuration/owner changed")
        self.cache.healthy()

    def match(self, source, target, initial):
        if self.failure is not None: raise scope.BridgeFailure("Failed near-seed method cannot resume") from self.failure
        try:
            require(all(value is owner for value, owner in zip((self.np, self.original, self.route, self.audit, self.timing_authorizer), self.owners)), "Method owner/audit configuration changed")
            self.check_configuration()
            self.route(); self.original.check()
            before = scope.input_binding(self.np, source, target, initial)
            if not self.audit: self.timing_authorizer(before)
            require(before["seed"]["dtype"] == "<f8" and before["seed"]["shape"] == [4, 4], "Original FP64 seed required")
            require(all(v["dtype"] == "<f8" for role in ("source", "target") for v in before[role].values()), "Exact original FP64 geometry precision required")
            seed = self.np.asarray(initial).tobytes(order="C")
            bucket = scope.canonical({key: before[key] for key in ("source", "target")})
            hit = self.cache.find(bucket, seed); self.statistics["calls"] += 1
            if hit is None:
                result = self.original.cpu_match(source, target, initial)
                self.statistics["original_cpu_calls"] += 1; self.statistics["misses"] += 1
                payload = freeze_result(self.np, result, len(source.points), len(target.points))
                index = self.cache.insert(bucket, seed, payload)
                self.statistics["unretained_misses"] += index is None
                representative, delta, kind = seed, 0., "miss"
                shadow = {"collected": False, "reason": "Original CPU result executed on actual input"}
            else:
                index, representative, payload, delta = hit
                result = restore_result(self.np, payload); self.statistics["hits"] += 1; kind = "hit"
                shadow = {"collected": False}
                if self.audit:
                    native = self.original.cpu_match(source, target, initial)
                    self.statistics["original_cpu_calls"] += 1
                    shadow = scope.result_shadow(self.np, native, result, len(source.points), len(target.points))
                    shadow["collected"] = True
                    shadow["passed"] = (all(type(shadow[k]) in (int, float) and math.isfinite(shadow[k]) and shadow[k] >= 0
                        for k in ("transform_max_abs_delta", "fitness_abs_delta", "rmse_abs_delta"))
                        and shadow["correspondence_ids_equal"] and shadow["transform_max_abs_delta"] <= 1e-10
                        and shadow["fitness_abs_delta"] <= 1e-12 and shadow["rmse_abs_delta"] <= 1e-12)
                    require(shadow["passed"], "Fresh same-input original CPU hit shadow rejected reuse")
                    self.statistics["audited_hits"] += 1
            require(before == scope.input_binding(self.np, source, target, initial), "Near-seed method changed input/seed owners")
            self.check_configuration()
            self.route(); self.original.check()
            self.last_receipt = {"kind": kind, "class_id": index, "cloud_value_sha256": hashlib.sha256(bucket.encode()).hexdigest(),
                "representative_seed_sha256": hashlib.sha256(representative).hexdigest(), "seed_max_abs_delta": delta,
                "threshold": self.cache.configuration["threshold"], "evictions": self.cache.evictions, "shadow": shadow}
            return result
        except BaseException as error:
            self.failure = error
            raise scope.BridgeFailure("Near-seed method failed; no original-call retry") from error

    def close(self): self.cache.close()

    def clear_pair(self):
        require(self.failure is None, "Damaged method cannot start another pair")
        self.check_configuration()
        configuration = dict(self.cache.configuration)
        self.previous_peak = max(self.previous_peak, self.cache.peak_bytes)
        self.previous_evictions += self.cache.evictions; self.cleared_pairs += 1
        self.cache.close(); self.cache = RepresentativeCache(**configuration); self.cache_owner = self.cache

    def report(self):
        return {"policy": POLICY, "audit": self.audit, "configuration": dict(self.cache.configuration),
            "statistics": dict(self.statistics), "closed": self.cache.closed, "failure": None if self.failure is None else repr(self.failure),
            "peak_payload_bytes": max(self.previous_peak, self.cache.peak_bytes), "owned_payload_bytes": self.cache.bytes,
            "evictions": self.previous_evictions+self.cache.evictions, "cleared_pairs": self.cleared_pairs,
            "byte_cap_scope": "Seed/result/key/evidence byte payloads; Python object overhead independently bounded by entry/cloud caps. Original native ICP workspace and input clouds excluded."}
