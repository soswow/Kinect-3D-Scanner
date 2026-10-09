"""Offline marker seeds for an absent original SIFT local proposal only.

This file has no numerical imports at module load. Marker identities are
proposal inputs; original reciprocal ICP, motion, held-out, graph, information
and independent-camera gates remain the pose authorities. No board geometry
or marker PnP result is accepted as a camera pose.
"""

from __future__ import annotations

import hashlib
import time

POLICY = "native-marker-absent-sift-local-icp-seed-v1"
MAX_FEATURE_CACHE_BYTES = 32 * 1024**2
MAX_CORNERS_PER_VIEW = 1000


class MarkerProposalFailure(BaseException):
    """A latched research error that ordinary Finish rejection must not hide."""


class NativeMarkerProvider:
    """Lazy native RGB decoding with the original calibrated depth support.

    Temporary projection buffers are not cached. Only bounded marker features
    and integer identities are retained; original SIFT features are untouched.
    CUDA queries use the unchanged, fully CPU-shadowed corner helper. Audit
    work is charged to the caller, never subtracted from a Finish speed claim.
    """

    def __init__(self, engine, *, lookup_mode="cuda-audit", device=0):
        if lookup_mode not in ("cpu", "cuda-audit"):
            raise ValueError("Marker lookup must be cpu or cuda-audit")
        import cv2
        import numpy as np
        from scanner_server.appearance import Features, propose_transform
        from shared.calibration import _rectified_depth, _color_maps_numpy, prepare_metric_depth
        from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS
        from shared.visual_tracking import sampled_points, feature_agreement

        if engine.settings.sensor_calibration is None:
            raise ValueError("Require actual native RGB/depth calibration")
        if (engine.unprocessed_count or len(engine.raw_frames) != len(engine.frame_metadata)
                or not 0 < len(engine.raw_frames) <= 500):
            raise ValueError("Require a bounded, fully processed fresh Live checkpoint")
        self.engine, self.np, self.cv2 = engine, np, cv2
        self.Features, self.propose_transform = Features, propose_transform
        self.rectify, self.color_maps, self.prepare_depth = _rectified_depth, _color_maps_numpy, prepare_metric_depth
        self.sampled_points, self.feature_agreement = sampled_points, feature_agreement
        self.maximum_lag_ms = RGB_DEPTH_ASSISTANCE_LIMIT_MS
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        self.lookup, self.cache, self.cache_bytes, self.rows, self.pairs = None, {}, 0, [], []
        self.lookup_mode, self.device = lookup_mode, device
        self.statistics = {"views_prepared": 0, "proposal_calls": 0, "proposals": 0,
            "decode_s": 0., "prepare_projection_s": 0., "lookup_and_shadow_s": 0.,
            "depth_support_s": 0., "proposal_s": 0., "cache_peak_bytes": 0}
        self.provenance = {"lookup_mode": lookup_mode, "dictionary": "DICT_4X4_250",
            "key": "decoded marker ID and canonical corner index, never board position",
            "maximum_native_rgb_distance_px": 3., "maximum_feature_cache_bytes": MAX_FEATURE_CACHE_BYTES,
            "maximum_corners_per_view": MAX_CORNERS_PER_VIEW,
            "cache_memory_scope": "Feature numeric buffers plus 24-byte logical integer key/index records; Python object/report heap overhead separately captured by process RSS. Temporary per-view projection/decoder and CuPy pool allocations are not retained feature-cache bytes.",
            "projection": "Original _rectified_depth/prepare_metric_depth/_color_maps_numpy; every visible actual depth-grid pixel in original row-major order",
            "support": "Original sampled_points measured center/3x3 support; reused depth pixels and duplicate marker IDs rejected",
            "scope": "Proposal only. All setup, decoding, projection, lookup, CPU shadows and support remain inside Finish."}

    def array_hash(self, array):
        np = self.np
        value = np.asarray(array)
        return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}

    def features_hash(self, features):
        return {name: None if getattr(features, name) is None else self.array_hash(getattr(features, name))
                for name in ("pixels", "points", "descriptors")}

    def _features(self, index):
        if index in self.cache:
            return self.cache[index]
        np, cv2, engine = self.np, self.cv2, self.engine
        if type(index) is not int or not 0 <= index < len(engine.raw_frames):
            raise ValueError("Marker view index is outside the original checkpoint")
        rgb, raw = engine.raw_frames[index]
        original = [self.array_hash(rgb), self.array_hash(raw)]
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or raw.dtype != np.uint16 or raw.ndim != 2):
            raise ValueError("Require original native uint8 RGB and uint16 raw depth")
        row = {"index": index, "raw_rgb": original[0], "raw_depth": original[1]}
        lag = engine.frame_metadata[index].get("rgb_depth_delta_ms")
        if lag is not None and (not np.isfinite(lag) or abs(lag) > self.maximum_lag_ms):
            features = self.Features(np.empty((0, 2)), np.empty((0, 3)), None)
            result = features, {}
            row.update(reason="Original RGB/depth assistance lag gate", supported_corners=0)
        else:
            started = time.perf_counter()
            corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            elapsed = time.perf_counter() - started
            self.statistics["decode_s"] += elapsed
            values = [] if ids is None else [int(value) for value in ids.ravel()]
            duplicates = {value for value in values if values.count(value) > 1}
            keys, uv = [], []
            for marker, polygon in zip(values, corners):
                if marker in duplicates:
                    continue
                for corner, point in enumerate(np.asarray(polygon).reshape(4, 2)):
                    keys.append((marker, corner))
                    uv.append(point)
            queries = np.asarray(uv, np.float64).reshape(-1, 2)
            if len(queries) > MAX_CORNERS_PER_VIEW or not np.isfinite(queries).all():
                raise ValueError("Decoded corner count/coordinates exceed declared finite domain")
            row.update(decoded_ids=values, duplicate_ids_rejected=sorted(duplicates), decode_s=elapsed)
            if len(queries):
                started = time.perf_counter()
                metric = self.rectify(raw, engine.settings)
                depth = self.prepare_depth(raw, engine.settings)
                map_x, map_y, visible = self.color_maps(metric, depth, engine.settings, rgb.shape)
                ys, xs = np.nonzero(visible)
                projections = np.column_stack((map_x[ys, xs], map_y[ys, xs])).astype(np.float64)
                elapsed = time.perf_counter() - started
                self.statistics["prepare_projection_s"] += elapsed
                row["prepare_projection_s"] = elapsed
                started = time.perf_counter()
                if len(projections):
                    if self.lookup_mode == "cuda-audit":
                        if self.lookup is None:
                            from scripts.research.native_corner_cuda import NativeCornerCuda
                            self.lookup = NativeCornerCuda(device=self.device)
                            self.provenance["cuda_lookup"] = self.lookup.provenance
                            self.provenance["cuda_self_check"] = self.lookup.self_check()
                        winners, squared, lookup_record = self.lookup.query(queries, projections)
                        row["cuda_lookup"] = lookup_record
                    else:
                        winners, squared = [], []
                        for query in queries:
                            distances = np.sum((projections-query)**2, axis=1)
                            winner = int(np.argmin(distances))
                            winners.append(winner)
                            squared.append(distances[winner])
                    selected_keys, pixels = [], []
                    for key, winner, distance in zip(keys, winners, squared):
                        if distance <= 9.:
                            selected_keys.append(key)
                            pixels.append((int(xs[winner]), int(ys[winner])))
                else:
                    selected_keys, pixels = [], []
                elapsed = time.perf_counter() - started
                self.statistics["lookup_and_shadow_s"] += elapsed
                row["lookup_and_shadow_s"] = elapsed
                started = time.perf_counter()
                reused = {pixel for pixel in pixels if pixels.count(pixel) > 1}
                pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
                points, supported = self.sampled_points(depth, pixels, engine.settings.camera)
                for offset, pixel in enumerate(map(tuple, pixels.tolist())):
                    if pixel in reused:
                        supported[offset] = False
                selected = np.flatnonzero(supported)
                features = self.Features(pixels[selected], points[selected], None)
                identities = {selected_keys[int(offset)]: destination for destination, offset in enumerate(selected)}
                result = features, identities
                elapsed = time.perf_counter() - started
                self.statistics["depth_support_s"] += elapsed
                row.update(depth_support_s=elapsed, supported_corners=len(selected),
                    reused_depth_pixels_rejected=len(reused), mapped_corners=len(pixels))
            else:
                result = self.Features(np.empty((0, 2)), np.empty((0, 3)), None), {}
                row.update(supported_corners=0, reason="No unique decoded marker corners")
        if [self.array_hash(rgb), self.array_hash(raw)] != original:
            raise ValueError("Original raw RGB/depth inputs were mutated")
        features, identities = result
        # Account both numeric arrays and two-int64 identity/index records.
        retained = features.pixels.nbytes + features.points.nbytes + 24*len(identities)
        if self.cache_bytes + retained > MAX_FEATURE_CACHE_BYTES:
            raise ValueError("Bounded marker feature cache exhausted; research does not silently evict/recompute")
        features.pixels.flags.writeable = features.points.flags.writeable = False
        row["feature_binding"] = self.features_hash(features)
        row["identities"] = [{"id": key[0], "corner": key[1], "row": value} for key, value in sorted(identities.items())]
        self.cache[index], self.cache_bytes = result, self.cache_bytes + retained
        self.statistics["views_prepared"] += 1
        self.statistics["cache_peak_bytes"] = max(self.statistics["cache_peak_bytes"], self.cache_bytes)
        self.rows.append(row)
        return result

    def proposal(self, source, target, camera):
        started = time.perf_counter()
        self.statistics["proposal_calls"] += 1
        a, a_ids = self._features(source.index)
        b, b_ids = self._features(target.index)
        common = sorted(a_ids.keys() & b_ids.keys())
        row = {"source": source.index, "target": target.index, "common_identity_corners": len(common)}
        pose = None
        if len(common) >= 40:
            matches = self.np.asarray([(a_ids[key], b_ids[key]) for key in common], int)
            pose = self.propose_transform(a, b, camera, matches)
            if pose is not None:
                left, right = matches.T
                supported, evidence = self.feature_agreement(a.points[left], b.points[right], b.pixels[right], pose, camera)
                row["original_feature_support"] = evidence
                if not supported:
                    pose = None
        row.update(proposal=pose is not None, wall_s=time.perf_counter()-started)
        self.statistics["proposal_s"] += row["wall_s"]
        if pose is not None:
            if (pose.shape != (4, 4) or not self.np.isfinite(pose).all()
                    or not self.np.array_equal(pose[3], [0., 0., 0., 1.])):
                raise ValueError("Malformed marker proposal")
            self.statistics["proposals"] += 1
            row["seed"] = self.array_hash(pose)
            row["seed_matrix"] = pose.tolist()
        self.pairs.append(row)
        return pose

    def report(self):
        return {"provenance": self.provenance, "statistics": dict(self.statistics),
            "view_rows": self.rows, "proposal_rows": self.pairs,
            "counter_scope": "Nested wall timers. proposal_s includes lazy decode/projection/lookup/support; do not add it to those substage times."}


class MarkerSeedScope:
    """Transactional local-only initialization seam around original gates."""

    def __init__(self, fragments, provider, *, trace=None):
        self.module, self.provider, self.trace = fragments, provider, trace
        self.original_local = fragments._local_match
        self.original_proposal = fragments.propose_transform
        self.original_matches = fragments._matches
        self.failure, self.pending, self.active, self.restored = None, None, False, True
        self.records, self.calls, self.injected, self.absent, self.accepted = [], 0, 0, 0, 0

    def fail(self, message, cause=None):
        if self.failure is None:
            self.failure = MarkerProposalFailure(message)
            if cause is not None:
                self.failure.__cause__ = cause
        raise self.failure

    def healthy(self):
        if self.failure is not None:
            raise self.failure

    def _checked_proposal(self, source, target, camera, matches=None):
        self.healthy()
        result = self.original_proposal(source, target, camera, matches)
        if self.pending is not None and (source is self.pending[0] and target is self.pending[1]):
            self.pending[2] += 1
            if result is not None:
                self.fail("Original SIFT absence changed while delegating marker-seeded ICP")
        return result

    def _local(self, source, target, camera, settings, initial=None, *, measured_first=False):
        self.healthy()
        if self.pending is not None:
            self.fail("Nested local marker dispatch is unsupported")
        self.calls += 1
        started = time.perf_counter()
        row = {"type": "marker_seed", "sequence": self.calls, "source": source.index,
            "target": target.index, "marker_injected": False, "complete": False}
        try:
            before = [self.provider.features_hash(source.features), self.provider.features_hash(target.features)]
            check_started = time.perf_counter()
            matches = self.original_matches(source, target)
            sift = self.original_proposal(source.features, target.features, camera, matches)
            row["duplicate_original_sift_proposal_check_s"] = time.perf_counter()-check_started
            row["original_sift_proposal_absent"] = sift is None
            seed = None
            if sift is None:
                self.absent += 1
                seed = self.provider.proposal(source, target, camera)
            row["marker_injected"] = seed is not None
            if seed is not None:
                self.injected += 1
                row["marker_seed"] = self.provider.array_hash(seed)
                self.pending = [source.features, target.features, 0]
        except MarkerProposalFailure:
            raise
        except BaseException as error:
            self.fail("Marker proposal/preflight failed", error)
        try:
            result = self.original_local(source, target, camera, settings,
                initial=initial if seed is None else seed, measured_first=measured_first)
        except MarkerProposalFailure:
            raise
        except Exception as error:
            # Original gate exceptions keep their original rejection semantics.
            row["original_gate_exception"] = {"type": type(error).__name__, "message": str(error)}
            row["wall_s"] = time.perf_counter()-started
            try:
                self.records.append(row)
                if self.trace is not None:
                    self.trace(row)
            except BaseException as recording_error:
                self.fail("Original local gate exception trace failed", recording_error)
            raise
        finally:
            delegated_checks = 0 if self.pending is None else self.pending[2]
            self.pending = None
            try:
                after = [self.provider.features_hash(source.features), self.provider.features_hash(target.features)]
                if after != before:
                    self.fail("Original SIFT Features changed during marker proposal dispatch")
                if seed is not None and delegated_checks != 1:
                    self.fail("Marker seed did not pass the unchanged original SIFT-absence branch exactly once")
                if seed is not None and self.provider.array_hash(seed) != row["marker_seed"]:
                    self.fail("Marker PnP seed was mutated during original ICP")
                row["sift_features_unchanged"] = True
                row["delegated_sift_checks"] = delegated_checks
            except MarkerProposalFailure:
                raise
            except BaseException as error:
                self.fail("Marker dispatch evidence failed", error)
        try:
            row.update(complete=True, accepted=result is not None, wall_s=time.perf_counter()-started)
            if result is not None:
                row["accepted_transform"] = self.provider.array_hash(result)
                if seed is not None:
                    self.accepted += 1
            self.records.append(row)
            if self.trace is not None:
                self.trace(row)
        except BaseException as error:
            self.fail("Marker dispatch trace failed", error)
        return result

    def __enter__(self):
        if self.active:
            raise MarkerProposalFailure("Marker scope cannot be reentered")
        self.healthy()
        try:
            self.module._local_match = self._local
            self.module.propose_transform = self._checked_proposal
            self.active, self.restored = True, False
        except BaseException as error:
            for name, original in (("_local_match", self.original_local), ("propose_transform", self.original_proposal)):
                try:
                    setattr(self.module, name, original)
                except BaseException as cleanup:
                    error.add_note(f"Marker scope partial-entry restore {name} also failed: {cleanup}")
            raise
        return self

    def __exit__(self, kind, error, traceback):
        cleanup = []
        for name, original in (("propose_transform", self.original_proposal), ("_local_match", self.original_local)):
            try:
                setattr(self.module, name, original)
            except BaseException as secondary:
                cleanup.append(secondary)
        self.active = False
        self.restored = not cleanup
        primary = self.failure if self.failure is not None else error
        if cleanup:
            if primary is not None:
                primary.add_note(f"Marker scope restore also failed: {cleanup}")
            else:
                self.fail("Marker scope restore failed", cleanup[0])
        if self.failure is not None:
            raise self.failure
        return False

    def report(self):
        failure = None
        if self.failure is not None:
            failure = {"type": type(self.failure).__name__, "message": str(self.failure)}
            if self.failure.__cause__ is not None:
                failure["cause"] = {"type": type(self.failure.__cause__).__name__,
                    "message": str(self.failure.__cause__)}
        return {"policy": POLICY, "local_calls": self.calls, "absent_sift_calls": self.absent,
            "injected_local_seeds": self.injected, "accepted_marker_seeded_local_icp": self.accepted,
            "failure": failure, "hooks_restored": self.restored,
            "scope": "Local initialization only. Original SIFT features/matches/witnesses and global proposals/independent-camera ambiguity gates unchanged. Existing sequential camera edges retain their original separate authority.",
            "records": self.records, "provider": self.provider.report()}
