"""Conditional compact tag initialization with measured recovery and depth audit."""

import time
import numpy as np

from shared.apriltag import TagFrame, tag_motion, without_repeated
from .joint_depth_bundle import refine_motion_graph
from .motion_evidence import MotionEvidence


def recover_tag_graph(views, frames, camera, motion=None, notify=None, device="cpu"):
    from .depth_graph import components, _tree, scene_visibility
    from .geometry_registration import register_pair, verify_candidates

    started = time.monotonic()
    frames, repeated = without_repeated(frames)
    report = {"attempted": False, "applied": False,
              "repeated_identities": [{"dictionary": key[0], "id": key[1]} for key in sorted(repeated)]}
    if motion:
        motion.tag_frames = frames
    bank, pairs = {}, set()
    for a, frame in enumerate(frames):
        for key in frame.tags:
            previous = bank.setdefault(key, [])
            pairs.update((a, b) for b in set(previous[:5] + previous[::4][-27:] + previous[-8:]))
            previous.append(a)
    measured, proposals = [], []
    for a, b in sorted(pairs):
        pose, support = tag_motion(frames[a], frames[b], camera)
        if pose is None:
            continue
        score = support["inlier_tags"] / (1 + support["median_pixel_error"])
        measured.append({"source": a, "target": b, "transform": pose, "score": score,
                         "method": "apriltag", "apriltag_constraint": True})
        proposals.append({"source": a, "target": b, "method": "apriltag", "accepted": True,
                          "proposal_only": True, "score": score, "transform": pose.tolist(),
                          "apriltag_support": support, "reason": "shared_measured_tag_corners"})
    groups = components(len(views), measured)
    report.update(attempted=bool(measured), shared_label_pairs=len(pairs), pose_pairs=len(measured),
                  initial_component_sizes=[len(g) for g in groups])
    if not groups or len(groups[0]) < 3:
        report.update(reason="insufficient_connected_tag_measurements", elapsed_s=time.monotonic()-started)
        return None, report
    group = groups[0]
    seeds = [e for e in measured if e["source"] in group and e["target"] in group]
    poses, tree = _tree(group, seeds)
    compact = MotionEvidence(motion.metadata if motion else [{} for _ in views], camera,
                             gravity=False, visual=False)
    compact.tag_frames = frames
    if notify:
        notify(f"AprilTag map: jointly fitting {len(poses)} cameras from shared corners")
    poses, corner_report = refine_motion_graph(poses, [], compact, max_evaluations=80)
    poses, depth_report = refine_motion_graph(poses, [], compact, views=views, max_evaluations=80)
    supported = {i for i, support in depth_report.get("apriltag_camera_support", {}).items()
                 if support["accepted"]}
    report.update(corner_refinement=corner_report, depth_refinement=depth_report,
                  tag_supported_cameras=sorted(supported))
    if len(supported) < 3:
        report.update(reason="tag_map_not_measurement_supported", elapsed_s=time.monotonic()-started)
        return None, report
    # Unsupported cameras are proposals for fallback, never fusion authority.
    if motion:
        motion.tag_pose_seeds = dict(poses)
    poses = {i: poses[i] for i in supported}
    active = [e for e in seeds if (e["source"], e["target"]) in tree
              and e["source"] in poses and e["target"] in poses]
    missing = sorted(set(range(len(views))) - set(poses))
    if len(missing) > max(8, len(views)//4):
        report.update(reason="tag_coverage_needs_general_recovery", remaining_cameras=missing,
                      elapsed_s=time.monotonic()-started)
        return None, report
    fallback = []
    for a in missing:
        if not motion:
            break
        if notify:
            notify(f"AprilTag map: checking RGB/depth recovery for capture {a+1}")
        candidates = sorted(poses, key=lambda b: (abs(a-b), b))[:8]
        for b in candidates:
            initial = None
            if a in motion.tag_pose_seeds:
                initial = np.linalg.inv(poses[b]) @ motion.tag_pose_seeds[a]
            extra = motion.seeds(a, b)
            relative, evidence = register_pair(views[a], views[b], method="tag_gap", device=device,
                initial=initial, extra_seeds=extra, pose_check=lambda p: motion.check(a, b, p),
                reciprocal_refine=lambda p: motion.reciprocal_pose(a, b, p))
            proposals.append(evidence)
            if relative is None:
                continue
            poses[a] = poses[b] @ relative
            active.append({"source": a, "target": b, "transform": relative, "score": evidence["score"],
                           "method": "tag_gap", "measured_appearance": bool(evidence.get("evidence", {})
                               .get("motion_evidence", {}).get("appearance", {}).get("accepted"))})
            fallback.append(a)
            break
    # A partial map supplies seeds to the complete recovery path. It must not
    # silently strand other tag components or previously recoverable captures.
    if len(poses) != len(views):
        report.update(reason="uncovered_captures_need_general_recovery", recovered_cameras=fallback,
                      remaining_cameras=sorted(set(range(len(views))) - set(poses)), elapsed_s=time.monotonic()-started)
        return None, report
    def fit_recovered(current):
        # Keep ordinary features bounded to recovered views and their measured
        # anchors. Avoid expansion into all intermediate camera observations.
        compact = MotionEvidence(motion.metadata, camera, motion.appearance, tracked=motion.tracked,
                                 gravity=False, visual=False)
        # A corner fit may reject a complete marker through depth or pixel
        # disagreement. Do not let it pull RGB-recovered cameras away from their
        # measurements in the final fit. Retain whole supported markers only.
        trusted = depth_report.get("apriltag_camera_support", {})
        compact.tag_frames = []
        for i, frame in enumerate(frames):
            identities = trusted.get(i, {}).get("inlier_identities", []) if i in supported and i not in fallback else []
            keys = {(key["dictionary"], key["id"]) for key in identities}
            compact.tag_frames.append(TagFrame({key: tag for key, tag in frame.tags.items() if key in keys}))
        return refine_motion_graph(current, [e for e in active if e["method"] == "tag_gap"],
                                   compact, views=views, max_evaluations=80)
    poses, final_report = fit_recovered(poses) if fallback else (poses, depth_report)
    # Recheck every recovered pose after the joint fit. A bounded correction
    # pass replaces a weak tag observation with independently measured RGB-D.
    for attempt in range(3):
        final_support = final_report.get("apriltag_camera_support", {})
        unsupported = [i for i in poses if not final_support.get(i, {}).get("accepted")]
        confirmed = set(poses)-set(unsupported)
        final_recovery = []
        for a in unsupported:
            if not motion:
                break
            anchors = sorted(confirmed, key=lambda b: (abs(a-b), b))[:8]
            for b in anchors:
                relative = np.linalg.inv(poses[b]) @ poses[a]
                verified, evidence = verify_candidates(views[a], views[b], [{"method": "tag_map_final", "pose": relative}],
                    pose_check=lambda p: motion.check(a, b, p),
                    reciprocal_refine=lambda p: motion.reciprocal_pose(a, b, p))
                if verified is not None:
                    confirmed.add(a)
                    final_recovery.append({"source": a, "target": b, "evidence": evidence})
                    break
        unsupported = sorted(set(poses)-confirmed)
        if not unsupported or not motion or attempt == 2:
            break
        corrected = []
        for a in unsupported:
            if notify:
                notify(f"AprilTag map: refreshing measured recovery for capture {a+1}")
            for b in sorted(confirmed, key=lambda b: (abs(a-b), b))[:8]:
                initial = np.linalg.inv(poses[b]) @ poses[a]
                relative, evidence = register_pair(views[a], views[b], method="tag_gap", device=device,
                    initial=initial, extra_seeds=motion.seeds(a, b), pose_check=lambda p: motion.check(a, b, p),
                    reciprocal_refine=lambda p: motion.reciprocal_pose(a, b, p))
                proposals.append(evidence)
                if relative is None:
                    continue
                poses[a] = poses[b] @ relative
                active = [e for e in active if e["source"] != a or e["method"] != "tag_gap"]
                active.append({"source": a, "target": b, "transform": relative, "score": evidence["score"],
                               "method": "tag_gap", "measured_appearance": bool(evidence.get("evidence", {})
                                   .get("motion_evidence", {}).get("appearance", {}).get("accepted"))})
                if a not in fallback:
                    fallback.append(a)
                corrected.append(a)
                break
        if not corrected:
            break
        poses, final_report = fit_recovered(poses)
    report.update(recovered_cameras=fallback, final_measured_recovery=final_recovery)
    # Require a gauge-connected graph of measurements that still agree with
    # the final poses. An internally consistent isolated tag island cannot
    # acquire authority merely through its original initializer.
    bank = {}
    for i, support in final_report.get("apriltag_camera_support", {}).items():
        if support["accepted"]:
            for key in support["inlier_identities"]:
                bank.setdefault((key["dictionary"], key["id"]), []).append(i)
    proof = {}
    for observers in bank.values():
        observers.sort()
        for a in observers[1:]:
            proof.setdefault((a, observers[0]), {"source": a, "target": observers[0],
                "method": "apriltag", "score": 1., "apriltag_constraint": True,
                "shared_landmark_refined": True})
    for item in final_recovery:
        a, b = item["source"], item["target"]
        proof[a, b] = {"source": a, "target": b, "method": "tag_gap",
                       "score": item["evidence"]["score"], "shared_landmark_refined": True}
    groups = components(len(views), list(proof.values()))
    report["final_measurement_component_sizes"] = [len(group) for group in groups]
    if groups and len(groups[0]) != len(views):
        unsupported = sorted(set(unsupported) | (set(poses)-set(groups[0])))
    # Tags remain fixed corner measurements. Raw sampled depth independently
    # checks every camera against the other retained views before fusion.
    validation = scene_visibility(views, poses, notify)
    if not validation["accepted"] or unsupported:
        if notify:
            notify(f"AprilTag map validation: depth conflicts {validation['free_space_fraction']:.2%}, "
                   f"{len(unsupported)} cameras need fresh tag/RGB support")
        report.update(reason="final_tag_or_depth_validation_failed", validation=validation,
                      unsupported_cameras=unsupported, elapsed_s=time.monotonic()-started)
        return None, report
    active = list(proof.values())
    for edge in active:
        edge["transform"] = np.linalg.inv(poses[edge["target"]]) @ poses[edge["source"]]
    report.update(applied=True, reason="tag_map_and_measured_gap_recovery_validated",
                  recovered_cameras=fallback, validation=validation, elapsed_s=time.monotonic()-started)
    solved = [{"frame_indices": sorted(poses), "poses": poses, "validation": validation,
               "joint_refinement": final_report}]
    return (solved, proposals, active), report
