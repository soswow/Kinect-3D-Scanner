"""Optional measured appearance and continuous sensor constraints for depth poses.

Visual trajectories propose alignment; raw depth still validates it. Acceleration
supplies a reliability-weighted gravity constraint, never integrated position.
Engineering weights and veto bounds are not calibrated probabilities.
"""

import math
from bisect import bisect_left, bisect_right
import numpy as np
import cv2

from shared.visual_tracking import feature_agreement, sampled_points, paired_depth_scale
from shared.inertial import GravityEstimator
from .appearance import Features, correspondences, propose_transform
from .visual_refinement import measured_pose
from .geometry_registration import rigid, pose_distance


def visual_seed(source, target):
    a, b = (m.get("visual_tracking", {}) for m in (source, target))
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if not (a.get("valid") and b.get("valid") and a.get("segment")
            and a.get("segment") == b.get("segment")
            and source.get("capture_generation") == target.get("capture_generation")):
        return None
    try:
        pa, pb = np.asarray(a["camera_to_local"], float), np.asarray(b["camera_to_local"], float)
        if rigid(pa) and rigid(pb):
            return np.linalg.inv(pb) @ pa
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        pass
    return None


def extract_tracks(metadata, depth, camera):
    """Validate bounded client identities and measure points from server depth."""
    tracking = metadata.get("visual_tracking", {})
    observation = tracking.get("measured_tracks") if isinstance(tracking, dict) else None
    # A fresh reference has measured pixels before it has a verified pose.
    # Its identities may establish a pose with a later raw observation. Failed
    # frames export no current tracks, so they cannot inherit stale reference
    # evidence. Recorded pose validity never authorizes these correspondences.
    if not isinstance(observation, dict) or not tracking.get("segment"):
        return None
    try:
        if observation.get("version") != 1 or observation.get("image_size") != [camera.width, camera.height]:
            return None
        identities = observation["ids"]
        if not 40 <= len(identities) <= 500 or any(type(i) is not int or i < 0 or i > 2**53 for i in identities):
            return None
        ids = np.asarray(identities, np.int64)
        pixels = np.asarray(observation["pixels"], np.float32)
        if pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all() or len(np.unique(ids)) != len(ids):
            return None
        if np.any(pixels < 0) or np.any(pixels >= [camera.width, camera.height]):
            return None
        points, keep = sampled_points(depth, pixels, camera)
        if keep.sum() < 40:
            return None
        return ids[keep], Features(pixels[keep], points[keep], None)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def gravity_observation(metadata):
    try:
        return _gravity_observation(metadata)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def _gravity_observation(metadata):
    """Use a local window of reliable reads, retaining coarse legacy evidence.

    Packet-to-exposure offset is unknown. Even good readings provide a broad
    constraint, with additional tolerance for host association uncertainty.
    """
    stamp = metadata.get("depth_host_monotonic_s")
    samples = metadata.get("motion_history", {}).get("accelerometer", ())
    selected = []
    if stamp is not None:
        for sample in samples:
            if (sample.get("valid") and abs(sample.get("host_monotonic_s", -math.inf) - stamp) <= .15
                    and sample.get("capture_generation") == metadata.get("capture_generation")):
                selected.append(sample.get("gravity", {}))
    basis = "continuous_host_reads"
    if not selected:
        observation = metadata.get("accelerometer", {})
        if observation.get("valid"):
            selected = [observation.get("gravity", {})]
            basis = "associated_read"
        else:
            observation = metadata.get("orientation_accelerometer", {})
            if observation.get("valid") and 0 <= observation.get("sample_age_ms", math.inf) <= 150:
                selected = [observation.get("gravity", {})]
                basis = "coarse_host_read"
    selected = [g for g in selected if g.get("valid") and .4 <= g.get("confidence", 0) <= 1]
    if not selected or len({g.get("calibration_id") for g in selected}) != 1:
        return None
    try:
        directions = np.asarray([g["up_camera"] for g in selected], float)
        if (directions.shape != (len(selected), 3) or not np.isfinite(directions).all()
                or np.any(abs(np.linalg.norm(directions, axis=1) - 1) > .01)):
            return None
        weights = np.asarray([g["confidence"] for g in selected])
        up = np.average(directions, axis=0, weights=weights)
        length = np.linalg.norm(up)
        if not np.isfinite(length) or length < .1:
            return None
        up /= length
        spread = np.degrees(np.arccos(np.clip(directions @ up, -1, 1))).max()
        if spread > 15:
            return None
        return {"up": up, "confidence": float(weights.mean()), "spread_deg": float(spread),
                "calibration_id": selected[0].get("calibration_id"), "basis": basis,
                "verified": all(g.get("calibration_verified") for g in selected),
                "mapping_uncertainty_s": metadata.get("host_mapping_uncertainty_s", 0.)}
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return None


class MotionEvidence:
    def __init__(self, metadata, camera, appearance=(), *, gravity=True, visual=True, journal=None, tracked=()):
        self.metadata, self.camera = metadata, camera
        self.appearance = appearance
        self.visual = visual
        self.tracked = tracked
        self.journal = journal
        visual_reads = {}
        for m in metadata:
            for row in m.get("motion_history", {}).get("visual", ()):
                stamp = row.get("host_monotonic_s")
                if isinstance(stamp, (float, int)) and np.isfinite(stamp):
                    visual_reads[row.get("capture_generation"), stamp] = row
        self.visual_reads = sorted(visual_reads.values(), key=lambda row: row["host_monotonic_s"])
        pools = {}
        for segment in (journal or {}).get("segments",()):
            estimator = GravityEstimator(segment["calibration"])
            pool = pools.setdefault(segment["capture_generation"],{})
            for sample in segment["samples"]:
                pool[sample["sequence"]] = {**sample,"gravity":estimator.update(sample)}
        indexed = {}
        for generation,pool in pools.items():
            samples = sorted(pool.values(),key=lambda s:s["host_monotonic_s"])
            indexed[generation] = ([s["host_monotonic_s"] for s in samples],samples)
        self.gravity = []
        for m in metadata:
            pool = indexed.get(m.get("capture_generation"))
            stamp = m.get("depth_host_monotonic_s")
            if pool and stamp is not None:
                times,samples = pool
                nearby = samples[bisect_left(times,stamp-.15):bisect_right(times,stamp+.15)]
                m = {**m,"motion_history":{**m.get("motion_history",{}),"accelerometer":nearby}}
            self.gravity.append(gravity_observation(m) if gravity else None)
        self.pairs = {}
        self.pair_features = {}

    def features_for_pair(self, a, b):
        self.appearance_pair(a, b)
        return self.pair_features.get((a, b))

    def visual_uncertainty(self, a, b):
        """Explicit engineering scales for accumulated measured visual motion.

        Successful short steps broaden with chain length. Failed observations
        broaden faster. These weights do not certify the camera trajectory.
        """
        steps = abs(self.metadata[a]["visual_tracking"].get("steps", 0) - self.metadata[b]["visual_tracking"].get("steps", 0))
        start, end = sorted([self.metadata[i].get("captured_monotonic_s", 0) for i in (a, b)])
        generation = self.metadata[a].get("capture_generation")
        missed = sum(not row.get("visual_tracking", {}).get("valid", False) for row in self.visual_reads
                     if start < row["host_monotonic_s"] <= end and row.get("capture_generation") == generation)
        return {"translation_scale_m": .004 + .004 * math.sqrt(steps) + .008 * missed,
                "rotation_scale_deg": .4 + .35 * math.sqrt(steps) + .5 * missed,
                "unverified_intermediate_frames": missed}

    def appearance_pair(self, a, b):
        if (a, b) not in self.pairs:
            value = None
            candidates = []
            if self.visual and self.tracked and self.tracked[a] is not None and self.tracked[b] is not None:
                ma, mb = self.metadata[a], self.metadata[b]
                ta, tb = ma.get("visual_tracking", {}), mb.get("visual_tracking", {})
                if (ma.get("capture_generation") and ma.get("capture_generation") == mb.get("capture_generation")
                        and ta.get("segment") == tb.get("segment")):
                    _, x, y = np.intersect1d(self.tracked[a][0], self.tracked[b][0], assume_unique=True, return_indices=True)
                    if len(x) >= 40:
                        candidates.append((self.tracked[a][1], self.tracked[b][1], np.column_stack((x, y)), "tracked_rgbd_identities"))
            if self.appearance and self.appearance[a] is not None and self.appearance[b] is not None:
                source, target = self.appearance[a], self.appearance[b]
                matches = correspondences(source, target)
                candidates.append((source, target, matches, "sift_rgbd_identities"))
            for source, target, matches, basis in candidates:
                pose = propose_transform(source, target, self.camera, matches)
                if pose is not None:
                    x, y = matches.T
                    moved = source.points[x] @ pose[:3, :3].T + pose[:3, 3]
                    supported = np.linalg.norm(moved - target.points[y], axis=1) <= np.maximum(.03, 3*paired_depth_scale(source.points[x], target.points[y]))
                    inliers = matches[supported]
                    x, y = inliers.T
                    good, _ = feature_agreement(source.points[x], target.points[y], target.pixels[y], pose, self.camera)
                    if good:
                        joint = measured_pose(source, target, inliers, pose, self.camera, minimum_matches=35)
                        if joint is not None and feature_agreement(source.points[x], target.points[y], target.pixels[y], joint, self.camera)[0]:
                            pose = joint
                        value = (pose, inliers)
                        self.pair_features[a, b] = (source, target, basis)
                        break
            self.pairs[a, b] = value
        return self.pairs[a, b]

    def reciprocal_pose(self, a, b, initial):
        """Fit the reverse measurement with fixed color/depth identities.

        Anonymous depth ICP must not erase the directions that measured texture
        constrains. The reverse fit uses the other image's pixels and raw depth.
        """
        pair = self.appearance_pair(a, b)
        if pair is None:
            return None
        source, target, _ = self.features_for_pair(a, b)
        pose = measured_pose(target, source, pair[1][:, ::-1], initial, self.camera, minimum_matches=35)
        if pose is None:
            return None
        x, y = pair[1].T
        good, _ = feature_agreement(target.points[y], source.points[x], source.pixels[x], pose, self.camera)
        return pose if good else None

    def seeds(self, a, b):
        output = []
        if self.visual:
            pose = visual_seed(self.metadata[a], self.metadata[b])
            if pose is not None:
                output.append(("camera_visual_history", pose))
        appearance = self.appearance_pair(a, b)
        if appearance is not None:
            output.append(("measured_rgbd_features", appearance[0]))
        return output

    def gravity_pair(self, a, b):
        ga, gb = self.gravity[a], self.gravity[b]
        if ga is None or gb is None or ga["calibration_id"] != gb["calibration_id"]:
            return None
        limit = (30. if ga["verified"] and gb["verified"] else 60.) + max(ga["spread_deg"],gb["spread_deg"])
        if ("coarse_host_read" in (ga["basis"], gb["basis"])
                or max(ga["mapping_uncertainty_s"],gb["mapping_uncertainty_s"]) > .03):
            limit += 15.
        return ga["up"],gb["up"],limit

    def revisit_pairs(self, groups, *, pool_per_view=12, budget_per_view=3):
        """Retrieve color candidates across maps before blind geometry search.

        Descriptor summaries only order a bounded search; mutual feature/depth
        matches still establish every proposed connection.
        """
        if not self.appearance:
            return []
        available = [i for i,f in enumerate(self.appearance) if f is not None
                     and f.descriptors is not None and len(f.points) >= 40]
        if len(available) < 2:
            return []
        floating = all(self.appearance[i].descriptors.dtype.kind == "f" for i in available)
        votes = None
        if floating:
            # Actual local descriptor identities retrieve places. Percentiles
            # of an image's SIFT values discard those identities and miss real
            # revisits with different framing. Approximate neighbours order
            # candidates only; mutual raw RGB-D verification stays mandatory.
            descriptors = np.concatenate([self.appearance[i].descriptors for i in available]).astype(np.float32)
            labels = np.concatenate([np.full(len(self.appearance[i].descriptors),i,int) for i in available])
            cv2.setRNGSeed(0)
            index = cv2.flann_Index(descriptors,dict(algorithm=1,trees=4))
            votes = {}
            for a in available:
                values = self.appearance[a].descriptors
                sample = values[np.linspace(0,len(values)-1,min(300,len(values)),dtype=int)].astype(np.float32)
                neighbours,_ = index.knnSearch(sample,min(48,len(descriptors)),params=dict(checks=64))
                # Several orientations or neighbours from one image cannot
                # turn one query feature into many place-retrieval votes.
                identities = np.unique(np.column_stack((np.repeat(np.arange(len(sample)),neighbours.shape[1]),
                                                        labels[neighbours.reshape(-1)])),axis=0)
                votes[a] = np.bincount(identities[:,1],minlength=len(self.metadata))
        else:
            summaries = np.array([np.percentile(self.appearance[i].descriptors,(25,50,75),axis=0).reshape(-1) for i in available])
            summaries /= np.maximum(np.linalg.norm(summaries,axis=1,keepdims=True),1e-8)
            distances = 2 - 2 * summaries @ summaries.T
        owner = {i:g for g,indices in enumerate(groups) for i in indices}
        candidates = {}
        for row,a in enumerate(available):
            def eligible(b):
                if owner[a] != owner[b]:
                    return True
                if abs(a-b) <= 3:
                    return False
                ta,tb = (self.metadata[i].get("captured_monotonic_s") for i in (a,b))
                return abs(ta-tb) >= 3. if ta is not None and tb is not None else abs(a-b) >= 8
            ranking = sorted(available,key=lambda b:(-votes[a][b],b)) if votes is not None else [available[j] for j in np.argsort(distances[row])]
            other = [b for b in ranking if eligible(b)]
            supported = []
            for b in other[:pool_per_view]:
                pair = (max(a,b),min(a,b))
                evidence = self.appearance_pair(*pair)
                if evidence is not None:
                    supported.append((len(evidence[1]),pair))
            for score,pair in sorted(supported,reverse=True)[:budget_per_view]:
                candidates[pair] = score
        return sorted(candidates,key=lambda p:(-candidates[p],p))

    def check(self, a, b, pose, *, require_distributed=True):
        report = {"accepted": True}
        prediction = visual_seed(self.metadata[a],self.metadata[b]) if self.visual else None
        if prediction is not None:
            distance,angle = pose_distance(prediction,pose)
            uncertainty = self.visual_uncertainty(a, b)
            translation_scale, rotation_scale = uncertainty["translation_scale_m"], uncertainty["rotation_scale_deg"]
            report["visual_history"] = {"translation_difference_m": distance,"rotation_difference_deg":angle,
                **uncertainty}
            # A soft search weight, never an acceptance decision or a calibrated
            # posterior. Accumulated/temporarily lost chains get broader scales.
            report["score_weight"] = 1/(1+(distance/translation_scale)**2+(angle/rotation_scale)**2)
        ga, gb = self.gravity[a], self.gravity[b]
        if ga is not None and gb is not None and ga["calibration_id"] == gb["calibration_id"]:
            angle = float(np.degrees(np.arccos(np.clip((pose[:3, :3] @ ga["up"]) @ gb["up"], -1, 1))))
            verified = ga["verified"] and gb["verified"]
            limit = self.gravity_pair(a,b)[2]
            confidence = min(ga["confidence"], gb["confidence"])
            scale = 10. if verified else 25.
            report["gravity"] = {"disagreement_deg": angle, "limit_deg": limit,
                "confidence": confidence, "source_basis": ga["basis"], "target_basis": gb["basis"],
                "calibration_verified": verified, "accepted": angle <= limit}
            report["score_weight"] = report.get("score_weight",1.) / (1 + confidence * (angle / scale)**2)
            if angle > limit:
                report.update(accepted=False, reason="gravity_direction_conflict")
        appearance = self.appearance_pair(a, b)
        if appearance is not None:
            source, target, basis = self.features_for_pair(a, b)
            x, y = appearance[1].T
            good, evidence = feature_agreement(source.points[x], target.points[y], target.pixels[y], pose, self.camera,
                                               require_distributed=require_distributed)
            report["appearance"] = {"accepted": good, "basis": basis, **evidence}
            if not good:
                report.update(accepted=False, reason="measured_rgbd_feature_conflict")
        return report
