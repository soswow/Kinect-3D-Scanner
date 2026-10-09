"""Choose bounded raw-view work from retained fragment evidence, never poses."""

import numpy as np


def fragment_pairs(engine, *, temporal_only=False):
    """Supporting camera pairs are retrieval hints, not authoritative transforms.

    Only constraints retained by a committed fragment reconstruction qualify.
    Every consumer still estimates and validates against raw observations.
    """
    report = getattr(engine, "fragment_reconnection", {})
    if not report.get("applied") or report.get("failed"):
        return []
    positions = {index: position for position, (index, _) in enumerate(engine.poses)}
    groups = []
    for edge in report.get("verified_bridges", []):
        if not edge.get("connected_to_scan"):
            continue
        if temporal_only and not (edge.get("temporal_constraint")
                                  or edge.get("validation_scope") == "sequential camera pair"):
            continue
        pairs = []
        for a, b in edge.get("support", []):
            if a in positions and b in positions and a != b:
                pairs.append(tuple(sorted((positions[a], positions[b]))))
        if pairs:
            groups.append(sorted(set(pairs), key=lambda p: (-(p[1] - p[0]), p)))
    # Cover each bridge before spending work on its additional witnesses.
    groups.sort(key=lambda group: (-(group[0][1] - group[0][0]), group[0]))
    ordered = []
    seen = set()
    for rank in range(max(map(len, groups), default=0)):
        for group in groups:
            if rank < len(group) and group[rank] not in seen:
                seen.add(group[rank])
                ordered.append(group[rank])
    return ordered


def choose_keyframes(count, limit, hints=()):
    """Preserve temporal coverage while including exact measured bridge views."""
    limit = min(count, max(2, limit))
    if count <= limit:
        return np.arange(count, dtype=int)
    if not hints:
        return np.linspace(0, count - 1, limit, dtype=int)
    # Reserve half the capacity for coverage; spend the rest on paired evidence.
    chosen = set(np.linspace(0, count - 1, max(2, limit // 2), dtype=int))
    for pair in hints:
        addition = set(pair) - chosen
        if len(chosen) + len(addition) <= limit:
            chosen.update(addition)
    for position in np.linspace(0, count - 1, limit, dtype=int):
        if len(chosen) >= limit:
            break
        chosen.add(int(position))
    # Duplicates in the two uniform grids may leave capacity; choose largest gaps.
    while len(chosen) < limit:
        ordered = sorted(chosen)
        a, b = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
        chosen.add((a + b) // 2)
    return np.array(sorted(chosen), dtype=int)


def validate_fragment_boundaries(engine, proposals, budget_check=lambda: True):
    """Later corrections must preserve raw measured temporal/storage ties.

    Fragment transforms are deliberately not imported into the camera solve.
    The original pair identities select fresh visual and held-out measurements.
    """
    report = {"validated_fragment_boundaries": 0, "fragment_boundary_checks": []}
    evidence = getattr(engine, "fragment_reconnection", {})
    if not evidence.get("applied") or evidence.get("failed"):
        return True, report
    from collections import OrderedDict
    from types import SimpleNamespace

    import open3d as o3d
    from .bundle_adjustment import _timely
    from .engine import ScanEngine
    from .fragments import _heldout, _rigid, _view, _visual_witness

    original = dict(engine.poses)
    proposed = dict(proposals)
    camera = engine.settings.camera
    adapter = SimpleNamespace(settings=engine.settings, raw_frames=engine.raw_frames,
        frame_metadata=engine.frame_metadata, max_depth_m=engine.settings.far_m,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(camera.width, camera.height,
            camera.fx, camera.fy, camera.cx, camera.cy))
    adapter._make_rgbd = lambda rgb, depth: ScanEngine._make_rgbd(adapter, rgb, depth)
    cache = OrderedDict()

    def view(index):
        if index not in cache:
            if len(cache) >= 8:
                cache.popitem(last=False)
            cache[index] = _view(adapter, index)
        cache.move_to_end(index)
        return cache[index]

    for edge in evidence.get("verified_bridges", []):
        scope = edge.get("validation_scope")
        if not edge.get("connected_to_scan") or not (
                edge.get("temporal_constraint") or scope == "sequential camera pair"):
            continue
        for a, b in edge.get("support", []):
            if not budget_check():
                return False, report
            if a not in original or b not in original or a not in proposed or b not in proposed:
                report["fragment_boundary_failure"] = "Missing retained boundary camera"
                return False, report
            old = np.linalg.inv(original[b]) @ original[a]
            new = np.linalg.inv(proposed[b]) @ proposed[a]
            correction = np.linalg.inv(old) @ new
            translation = float(np.linalg.norm(correction[:3, 3]))
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1))))
            row = {"source_index": a, "target_index": b,
                   "relative_correction_m": translation, "relative_correction_deg": rotation}
            report["fragment_boundary_checks"].append(row)
            if not _rigid(new) or translation > 0.03 or rotation > 3:
                row["reason"] = "boundary_correction_bounds"
                return False, report
            if not (_timely(engine.frame_metadata[a]) and _timely(engine.frame_metadata[b])):
                row["reason"] = "boundary_rgbd_not_synchronized"
                return False, report
            source, target = view(a), view(b)
            if not budget_check():
                return False, report
            if source is None or target is None:
                row["reason"] = "missing_boundary_depth"
                return False, report
            valid, stats = _heldout(source.heldout, target.heldout, new,
                                   0.4 if scope == "sequential camera pair" else 0.45)
            row.update(stats)
            if valid and edge.get("visual_constraint"):
                valid, stats = _visual_witness(source, target, new, camera)
                row.update(stats)
            row.update(accepted=bool(valid), reason="accepted_raw_boundary" if valid else "boundary_measurement_disagreement")
            if not valid:
                return False, report
        report["validated_fragment_boundaries"] += 1
    return True, report
