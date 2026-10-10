"""Offline camera estimation from every retained depth view.

Live poses, timestamp gaps and fragment counts do not authorize edges. Optional
AprilTag corner identities supplement independently verified depth evidence.
Disconnected components retain their own coordinate systems in the report.
Only the largest measured component is offered to the single-volume engine.
"""

from collections import deque
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree

from .geometry_registration import (
    DepthView, evaluate, features, global_seeds, ppf_seeds, refine, pose_distance,
    prepare_view, register_pair, verify_candidates, _visibility,
)


def scene_visibility(views, poses, notify=None):
    """Audit every ordered view pair for surfaces in measured empty space.

    This reports a geometric diagnostic, not a probability of correctness.
    It includes pairs that were never proposed as graph edges.
    """
    conflicts, tested, free = [], 0, 0.
    indices = sorted(poses)
    if not any(len(views[i].cloud.points) >= 100 for i in indices):
        return {"accepted":False,"reason":"insufficient_depth","tested_surface_projections":0,
                "free_space_fraction":0.,"conflicting_view_pairs":0,"worst_pairs":[]}
    for a in indices:
        if notify and indices.index(a) % 10 == 0:
            notify(f"Checking complete depth component: view {indices.index(a)+1}/{len(indices)}")
        for b in indices:
            if a == b:
                continue
            evidence = _visibility(views[a], views[b], np.linalg.inv(poses[b]) @ poses[a])
            tested += evidence["tested"]
            free += evidence["free_space_fraction"] * evidence["tested"]
            if evidence["tested"] >= 500 and evidence["free_space_fraction"] > .15:
                conflicts.append({"source": a, "target": b, **evidence})
    return {"accepted": free/max(1,tested) <= .12 and not any(e["free_space_fraction"] > .4 for e in conflicts),
            "tested_surface_projections": tested, "free_space_fraction": free/max(1,tested),
            "conflicting_view_pairs": len(conflicts),
            "worst_pairs": sorted(conflicts, key=lambda e: -e["free_space_fraction"])[:30]}

REG = o3d.pipelines.registration


def components(count, edges):
    parents = list(range(count))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for e in edges:
        parents[root(e["source"])] = root(e["target"])
    groups = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def edge_from_report(report):
    if not report.get("accepted"):
        return None
    transform = report.get("transform")
    if transform is None:  # Earlier benchmark files store all candidates.
        candidates = [c for c in report["candidates"] if c["accepted"]]
        transform = max(candidates, key=lambda c: c["score"])["pose"]
    return {"source": report["source"], "target": report["target"],
            "transform": np.array(transform), "score": report["score"],
            "method": report["method"]}


def _tree(nodes, edges):
    """Maximum-evidence tree, with no privileged first camera/recorded pose."""
    selected = []
    parent = {i: i for i in nodes}
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for edge in sorted(edges, key=lambda e: (-e["score"], e["source"], e["target"])):
        a, b = edge["source"], edge["target"]
        if root(a) != root(b):
            parent[root(a)] = root(b)
            selected.append(edge)
    links = {i: [] for i in nodes}
    for e in selected:
        a, b, t = e["source"], e["target"], e.get("initial_transform",e["transform"])
        links[a].append((b, np.linalg.inv(t)))
        links[b].append((a, t))
    world, pending = {min(nodes): np.eye(4)}, deque([min(nodes)])
    while pending:
        i = pending.popleft()
        for j, t in links[i]:
            if j not in world:
                world[j] = world[i] @ t
                pending.append(j)
    return world, {(e["source"], e["target"]) for e in selected}


def bridge_visibility(views, edges, candidate):
    """Use every available view on both sides, with no minimum view count.

    Two isolated frames can establish a bridge. When a component contains more
    depth observations, a bridge must also explain their measured empty space.
    A partial pair fit cannot override contradictory room geometry.
    """
    groups = components(len(views), edges)
    a, b = candidate["source"], candidate["target"]
    left = next(g for g in groups if a in g)
    right = next(g for g in groups if b in g)
    if left is right:
        return {"accepted": True, "reason": "existing_component_cycle"}
    wa, _ = _tree(left, [e for e in edges if e["source"] in left and e["target"] in left])
    wb, _ = _tree(right, [e for e in edges if e["source"] in right and e["target"] in right])
    transform = wb[b] @ candidate["transform"] @ np.linalg.inv(wa[a])
    wa = {i: transform @ p for i,p in wa.items()}
    projected, free, conflicts = 0, 0., []
    for source_group, target_group in ((wa,wb),(wb,wa)):
        for i, p in source_group.items():
            for j, q in target_group.items():
                evidence = _visibility(views[i],views[j],np.linalg.inv(q) @ p)
                projected += evidence["tested"]
                free += evidence["tested"] * evidence["free_space_fraction"]
                if evidence["tested"] >= 500 and evidence["free_space_fraction"] > .4:
                    conflicts.append({"source": i, "target": j, **evidence})
                    # Passing bridges examine every cross-view pair. A veto
                    # already proved by measured empty space needs no further
                    # projection work; this does not change acceptance.
                    return {"accepted": False,"reason": "component_free_space_conflict",
                            "complete": False,"tested_surface_projections": projected,
                            "free_space_fraction": free/max(1,projected),
                            "conflicting_view_pairs": 1,"worst_pairs": conflicts}
    fraction = free/max(1,projected)
    good = fraction <= .12 and not conflicts
    return {"accepted": good, "complete":True, "reason": "component_depth_supported" if good else "component_free_space_conflict",
            "tested_surface_projections": projected, "free_space_fraction": fraction,
            "conflicting_view_pairs": len(conflicts),
            "worst_pairs": sorted(conflicts,key=lambda e:-e["free_space_fraction"])[:5]}


def aggregate_view(views, indices, edges):
    """Accumulate unfused geometry for descriptor search, retaining raw views."""
    world, _ = _tree(indices,[e for e in edges if e["source"] in indices and e["target"] in indices])
    train, witness = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    import copy
    for i in indices:
        train += copy.deepcopy(views[i].cloud).transform(world[i])
        witness += copy.deepcopy(views[i].heldout).transform(world[i])
    # Keep fine clouds for refinement. Only descriptor proposals use larger
    # cells; downsampling the refinement cloud can remove the few corners that
    # constrain a camera's translation along a largely planar scene.
    if len(indices) > 1:
        train, witness = train.voxel_down_sample(.025), witness.voxel_down_sample(.02)
    return DepthView(min(indices),None,train,witness,None,
                     tree=cKDTree(np.asarray(witness.points)),proposal_voxel=.05)


def aggregate_bridge(views, left, right, edges, device):
    """Search the entire available geometry on each side, not one small patch."""
    a, b = aggregate_view(views,left,edges), aggregate_view(views,right,edges)
    if min(len(a.cloud.points),len(b.cloud.points)) < 100:
        return None, {"source": a.index,"target": b.index,"method": "aggregate","accepted": False,"reason": "insufficient_depth"}
    seeds = [("aggregate_identity",np.eye(4))]+global_seeds(a,b,"multiscale",device)+ppf_seeds(a,b)
    tested, supported = [], []
    for name, seed in seeds:
        pose = refine(a,b,seed)
        if pose is None or any(max(pose_distance(pose,p)[0]/.03,pose_distance(pose,p)[1]/3) < 1 for p in tested):
            continue
        tested.append(pose)
        evidence = evaluate(a,b,pose,minimum_overlap=.08,check_visibility=False)
        if not evidence["accepted"]:
            continue
        reverse = refine(b,a,np.linalg.inv(pose))
        if reverse is None:
            continue
        cycle = pose_distance(np.eye(4),reverse @ pose)
        if cycle[0] > .025 or cycle[1] > 3:
            continue
        candidate = {"source": a.index,"target": b.index,"transform": pose,
                     "score": evidence["score"],"method": "aggregate"}
        component_evidence = bridge_visibility(views,edges,candidate)
        if component_evidence["accepted"]:
            supported.append((candidate,evidence,component_evidence,name))
    report = {"source": a.index,"target": b.index,"method": "aggregate", "accepted": False,
              "source_views": left,"target_views": right,"tested_hypotheses": len(tested)}
    if not supported:
        report["reason"] = "no_supported_component_pose"
        return None, report
    supported.sort(key=lambda row:-row[0]["score"])
    best, evidence, component_evidence, name = supported[0]
    if any(row[0]["score"] >= .85*best["score"] and
           max(pose_distance(best["transform"],row[0]["transform"])[0]/.05,
               pose_distance(best["transform"],row[0]["transform"])[1]/5) > 1 for row in supported[1:]):
        report["reason"] = "competing_component_geometry"
        return None, report
    information = REG.get_information_matrix_from_point_clouds(a.cloud,b.cloud,.03,best["transform"])
    best.update(information=information.tolist(),aggregate_evidence=True)
    report.update(accepted=True,reason="supported_component_geometry",score=best["score"],
                  transform=best["transform"].tolist(),evidence=evidence,
                  component_validation=component_evidence,seed_method=name)
    return best, report


def solve_graph(views, edges, notify=None, *, tag_frames=None, camera=None):
    """Optimize then recheck depth; reject contradictions without forced fusion.

    A tree bridge can stand on one pair when that pair is geometrically
    determined. Redundant constraints remain evidence and must agree with the
    final poses. If optimization cannot explain them, the conflicting edges
    are removed and all remaining components are solved again.
    """
    active, rejected = list(edges), []
    while True:
        solved, conflicts = [], []
        groups = components(len(views), active)
        for group in groups:
            subset = [e for e in active if e["source"] in group and e["target"] in group]
            world, tree = _tree(group, subset)
            graph = REG.PoseGraph()
            ids = {i: n for n, i in enumerate(group)}
            for i in group:
                graph.nodes.append(REG.PoseGraphNode(world[i]))
            for e in subset:
                a, b = e["source"], e["target"]
                information = (np.asarray(e["information"]) if "information" in e else
                    REG.get_information_matrix_from_point_clouds(views[a].cloud, views[b].cloud, .03, e["transform"]))
                graph.edges.append(REG.PoseGraphEdge(ids[a], ids[b], e["transform"], information,
                    (a, b) not in tree))
            if len(group) > 1:
                REG.global_optimization(graph, REG.GlobalOptimizationLevenbergMarquardt(),
                    REG.GlobalOptimizationConvergenceCriteria(), REG.GlobalOptimizationOption(
                        max_correspondence_distance=.03, edge_prune_threshold=.25, reference_node=0))
            optimized = {i: graph.nodes[ids[i]].pose.copy() for i in group}
            for e in subset:
                a, b = e["source"], e["target"]
                predicted = np.linalg.inv(optimized[b]) @ optimized[a]
                deviation = pose_distance(e["transform"], predicted)
                evidence = ({"accepted": True} if e.get("aggregate_evidence") else
                    evaluate(views[a], views[b], predicted,
                             minimum_condition=0 if e.get("apriltag_constraint") else 1e-4))
                if e.get("apriltag_constraint"):
                    from shared.apriltag import tag_agreement
                    if tag_frames is None or not tag_agreement(tag_frames[a], tag_frames[b], predicted, camera)[0]:
                        evidence = {"accepted": False}
                # A pruned graph edge is still a measured contradiction. It
                # cannot be ignored simply because the optimizer disliked it.
                if not evidence["accepted"] or deviation[0] > .06 or deviation[1] > 6:
                    conflicts.append(e)
            solved.append({"frame_indices": group, "poses": optimized})
        if not conflicts:
            return solved, active, rejected
        # Removing contradictory edges is explicit, and cannot connect a new
        # component. Re-solving prevents rejected edges retaining authority.
        bad = {id(e) for e in conflicts}
        rejected += [{"source": e["source"], "target": e["target"],
                      "reason": "inconsistent_optimized_depth_or_cycle"} for e in conflicts]
        active = [e for e in active if id(e) not in bad]
        if notify:
            notify(f"Rechecking graph after excluding {len(conflicts)} contradictory links")


def recover_depth_graph(views, *, device="cpu", notify=None, cached_pairs=(), checkpoint=None,
                        tag_frames=None, camera=None):
    count = len(views)
    if [v.index for v in views] != list(range(count)):
        raise ValueError("Depth graph requires capture-order view indices")
    cache = {(r["source"], r["target"], r["method"]): r for r in cached_pairs}
    reports, edges, tested, ambiguous = [], {}, set(), set()
    def say(message):
        if notify:
            notify(message)
    def match(a, b, method, initial=None):
        key = (a, b, method)
        if key in tested:
            return None
        tested.add(key)
        report = cache.get(key)
        if method == "apriltag":
            from shared.apriltag import tag_motion
            pose, support = tag_motion(tag_frames[a], tag_frames[b], camera)
            if pose is None:
                return None
            # Measured tag identities constrain the axes that a plane's depth
            # normals cannot. Depth still independently checks overlap, range,
            # reciprocal residuals and empty space; optimization rechecks tags.
            evidence = evaluate(views[a], views[b], pose, minimum_condition=0)
            report = {"source": a, "target": b, "method": method, **evidence,
                      "apriltag_support": support, "transform": pose.tolist(),
                      "candidates": [{"method": "apriltag", "pose": pose.tolist()}]}
        elif report is None:
            _, report = register_pair(views[a], views[b], method=method, device=device, initial=initial)
        else:
            _, report = verify_candidates(views[a],views[b],report.get("candidates",()),method=method)
            report["revalidated_cache"] = True
        reports.append(report)
        if checkpoint and len(reports) % 10 == 0:
            checkpoint(reports)
        edge = edge_from_report(report)
        if edge is not None:
            if (a,b) in ambiguous:
                return None
            old = edges.get((a, b))
            if old is not None and max(pose_distance(old["transform"], edge["transform"])[0] / .05,
                                      pose_distance(old["transform"], edge["transform"])[1] / 5) > 1:
                edges.pop((a, b))
                ambiguous.add((a,b))
                report["cross_method_conflict"] = True
                return None
            if method != "local":
                component_evidence = bridge_visibility(views,list(edges.values()),edge)
                report["component_validation"] = component_evidence
                if not component_evidence["accepted"]:
                    return None
            edges[a, b] = edge if old is None or edge["score"] > old["score"] else old
            if method == "apriltag" or old is not None and old.get("apriltag_constraint"):
                edges[a, b]["apriltag_constraint"] = True
        return edge
    if tag_frames is not None:
        # Search a bounded bank of every shared family/ID, including revisits.
        bank = {}
        tag_pairs = set()
        for a, frame in enumerate(tag_frames):
            for identity in frame.tags:
                previous = bank.setdefault(identity, [])
                tag_pairs.update((a, b) for b in set(previous[:5] + previous[::4][-27:] + previous[-8:]))
                previous.append(a)
        for a, b in sorted(tag_pairs):
            match(a, b, "apriltag")
        say(f"AprilTag registration: checked {len(tag_pairs)} shared-label pairs")
    pairs = [(i, i-step) for i in range(count) for step in (1, 2, 3) if i >= step]
    for n, (a, b) in enumerate(pairs):
        match(a, b, "local")
        if n % 20 == 0:
            say(f"Depth registration: local evidence {n+1}/{len(pairs)}")
    say("Optimizing local depth maps before testing their global placement")
    local_solved, local_active, local_rejected = solve_graph(views,list(edges.values()),say,
        tag_frames=tag_frames, camera=camera)
    local_world = {i:p for c in local_solved for i,p in c["poses"].items()}
    edges = {(e["source"],e["target"]):e for e in local_active}
    for e in edges.values():
        # These poses initialize geometry accumulation only. Original measured
        # transforms remain graph constraints and are revalidated after fusion
        # proposals; optimization does not manufacture new evidence.
        e["initial_transform"] = np.linalg.inv(local_world[e["target"]]) @ local_world[e["source"]]
    # Search order only: temporal distance never rules a pose in/out.
    for a, b in sorted(pairs, key=lambda pair: (pair[0]-pair[1], pair[0])):
        groups = components(count, list(edges.values()))
        owner = {i: j for j, group in enumerate(groups) for i in group}
        if owner[a] == owner[b]:
            continue
        say(f"Depth registration: global recovery {a+1} <-> {b+1}; {len(groups)} components")
        if match(a,b,"gicp") is not None:
            continue
        recovered = match(a, b, "multiscale")
        # A different geometry descriptor also challenges an accepted FPFH
        # pose; agreement is useful, but two algorithms are not ground truth.
        ppf = match(a, b, "ppf")
        if recovered is None and ppf is None:
            match(a, b, "projective")
    # Descriptor summaries retrieve revisits without an image or stored pose.
    # Evaluate a bounded candidate list per frame, not a fixed frame/fragment
    # cap. It is a search policy; omitted candidates remain explicitly unsolved.
    if count > 1 and len(components(count, list(edges.values()))) > 1:
        summaries = []
        for v in views:
            _, f = features(v, .16)
            values = np.asarray(f.data)
            summary = np.percentile(values, (25, 50, 75), axis=1).reshape(-1) if values.shape[1] else np.zeros(99)
            summaries.append(summary / max(np.linalg.norm(summary), 1e-8))
        summaries = np.array(summaries)
        distance = np.sum((summaries[:, None] - summaries[None, :]) ** 2, axis=2)
        candidates = set()
        groups = components(count, list(edges.values()))
        owner = {i: j for j, group in enumerate(groups) for i in group}
        for a in range(count):
            other = [int(b) for b in np.argsort(distance[a]) if owner[a] != owner[int(b)] and abs(a-int(b)) > 3]
            for b in other[:3]:
                candidates.add((max(a,b), min(a,b)))
        for n, (a,b) in enumerate(sorted(candidates, key=lambda p: (distance[p], p))):
            groups = components(count, list(edges.values()))
            owner = {i: j for j, group in enumerate(groups) for i in group}
            if owner[a] == owner[b]:
                continue
            say(f"Depth registration: revisit {a+1} <-> {b+1}; candidate {n+1}/{len(candidates)}")
            match(a,b,"multiscale")
            match(a,b,"ppf")
    # Larger shared surface context can distinguish a room bridge that a small
    # frame patch cannot. Candidate budgets grow with retained components.
    groups = components(count,list(edges.values()))
    if len(groups) > 1:
        summaries = []
        for group in groups:
            combined = aggregate_view(views,group,list(edges.values()))
            _, f = features(combined,.16)
            data = np.asarray(f.data)
            summary = np.percentile(data,(25,50,75),axis=1).reshape(-1) if data.shape[1] else np.zeros(99)
            summaries.append(summary/max(np.linalg.norm(summary),1e-8))
        distance = np.sum((np.array(summaries)[:,None]-np.array(summaries)[None,:])**2,axis=2)
        candidates = {(min(a,int(b)),max(a,int(b))) for a in range(len(groups))
                      for b in np.argsort(distance[a]) if int(b) != a}
        # All small component sets; otherwise three closest summaries per map.
        if len(groups) > 8:
            candidates = {(min(a,int(b)),max(a,int(b))) for a in range(len(groups))
                          for b in [i for i in np.argsort(distance[a]) if i != a][:3]}
        for n,(a,b) in enumerate(sorted(candidates,key=lambda p:distance[p])):
            now = components(count,list(edges.values()))
            left = next(g for g in now if groups[a][0] in g)
            right = next(g for g in now if groups[b][0] in g)
            if left is right:
                continue
            say(f"Searching accumulated depth components {n+1}/{len(candidates)}")
            edge, report = aggregate_bridge(views,left,right,list(edges.values()),device)
            reports.append(report)
            if edge is not None:
                edges[edge["source"],edge["target"]] = edge
            if checkpoint:
                checkpoint(reports)
    say("Optimizing and checking all measured depth constraints")
    solved, active, rejected = solve_graph(views, list(edges.values()), say, tag_frames=tag_frames, camera=camera)
    rejected = local_rejected + rejected
    edges = {(e["source"],e["target"]):e for e in active}
    # Camera order offers a hypothesis after global geometry has established
    # a map. It supplies no motion ceiling, accepted pose, or timestamp gate.
    if solved and len(solved[0]["frame_indices"]) > 1:
        world = solved[0]["poses"]
        known = sorted(world)
        for a in sorted(set(range(count)) - set(known)):
            before, after = [i for i in known if i < a], [i for i in known if i > a]
            if not before or not after:
                continue
            left, right = before[-1], after[0]
            amount = (a-left)/(right-left)
            rotation = Slerp([0.,1.], Rotation.from_matrix([world[left][:3,:3],world[right][:3,:3]]))([amount]).as_matrix()[0]
            prediction = np.eye(4)
            prediction[:3,:3] = rotation
            prediction[:3,3] = (1-amount)*world[left][:3,3]+amount*world[right][:3,3]
            say(f"Testing interpolated depth hypothesis for capture {a+1}")
            for b in (left,right):
                edge = match(a,b,"interpolated_depth", np.linalg.inv(world[b]) @ prediction)
                if edge is not None:
                    active.append(edge)
        if any(r["method"] == "interpolated_depth" and r["accepted"] for r in reports):
            solved, active, extra_rejected = solve_graph(views,active,say, tag_frames=tag_frames, camera=camera)
            rejected += extra_rejected
    for component in solved:
        component["validation"] = scene_visibility(views,component["poses"],say)
    solved.sort(key=lambda c: (not c["validation"]["accepted"], -len(c["frame_indices"]), c["frame_indices"][0]))
    return solved, {"algorithm": "offline_depth_graph_v1", "device": device,
                    "pairs": reports, "graph_rejected_edges": rejected,
                    "ambiguous_camera_pairs": [list(p) for p in sorted(ambiguous)],
                    "verified_bridges": [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in e.items()} for e in active],
                    "component_sizes": [len(c["frame_indices"]) for c in solved],
                    "validated_component_sizes": [len(c["frame_indices"]) for c in solved if c["validation"]["accepted"]],
                    "rejected_component_sizes": [len(c["frame_indices"]) for c in solved if not c["validation"]["accepted"]],
                    "validation": "sampled depth, visibility, six-direction conditioning or measured AprilTags, competing poses, reciprocal fit and graph revalidation",
                    "thresholds_are_calibrated_probabilities": False}


def propose_depth_poses(engine, progress_cb=None):
    # Current Open3D builds expose an explicit thread setting; OMP alone may
    # not cap descriptor/normal work. Restore the server's previous setting.
    getter = getattr(o3d.utility,"get_max_threads",None)
    setter = getattr(o3d.utility,"set_max_threads",None)
    previous = getter() if getter and setter else None
    if previous is not None:
        setter(max(1,int(os.environ.get("OMP_NUM_THREADS","4"))))
    try:
        return _propose_depth_poses(engine,progress_cb)
    finally:
        if previous is not None:
            setter(previous)


def _propose_depth_poses(engine, progress_cb=None):
    def notify(message):
        if progress_cb:
            progress_cb(0, len(engine.raw_frames), {"stage": "fragment_reconnection", "message": message})
    views = []
    for i, (_, raw) in enumerate(engine.raw_frames):
        notify(f"Preparing depth view {i+1}/{len(engine.raw_frames)}")
        views.append(prepare_view(i, raw, engine.settings))
    device = "cuda" if str(engine.device).startswith("CUDA") else "cpu"
    cached = getattr(engine, "_offline_depth_pair_cache", ())
    tag_options = {}
    if engine.settings.apriltag_tracking:
        from .apriltag_tracking import observation
        tag_options = {"tag_frames": [observation(engine, i) for i in range(len(views))],
                       "camera": engine.settings.camera}
    solved, report = recover_depth_graph(views, device=device, notify=notify, cached_pairs=cached, **tag_options)
    report["cached_pairs_revalidated"] = len(cached)
    if not solved or not solved[0]["validation"]["accepted"]:
        report.update(applied=False, failed=True, reason="No connected, geometrically determined depth views")
        return None, report
    selected = solved[0]
    proposals = sorted(selected["poses"].items())
    baseline = dict(engine.poses)
    included = set(selected["frame_indices"])
    report.update(applied=False, recovered_frames=len(included - set(baseline)),
                  corrected_frames=sum(i in baseline and max(pose_distance(baseline[i], p)[0]/.01,
                      pose_distance(baseline[i], p)[1]) > 1 for i,p in proposals),
                  excluded_frames=sorted(set(range(len(views))) - included),
                  fragments=[{"id": n, "frame_indices": c["frame_indices"], "validation": c["validation"],
                      "poses": [{"index": i, "camera_to_component": p.tolist()} for i,p in sorted(c["poses"].items())]}
                      for n,c in enumerate(solved)],
                  unconnected_fragments=list(range(1, len(solved))),
                  selection="Largest measured component; other components retained with independent poses",
                  budget_limited=False)
    return proposals, report
