"""Try the original verified ORB policy, then a bounded SIFT fallback bank."""


def select_policy(settings, requested, experimental="off"):
    """Keep adaptive recovery within the resolution pair tested on both sessions."""
    if requested not in ("orb", "sift", "adaptive"):
        raise ValueError("KINECT_VISUAL_FEATURES must be orb, sift, or adaptive")
    if experimental not in ("off", "on"):
        raise ValueError("KINECT_ADAPTIVE_EXPERIMENTAL must be off or on")
    policy = requested
    reason = f"Requested {requested.upper()} recovery"
    if requested == "adaptive":
        final_voxel = settings.final_voxel_m or settings.voxel_m
        if settings.voxel_m == .01 and final_voxel == .005:
            policy = "orb_then_sift"
            reason = "Adaptive enabled for 10 mm Live / 5 mm Final"
        elif experimental == "on":
            policy = "orb_then_sift"
            reason = "Experimental adaptive override enabled at an unvalidated resolution"
        else:
            policy = "orb"
            reason = "Adaptive requires 10 mm Live / 5 mm Final; using ORB"
    return {
        "visual_features": "orb" if requested == "adaptive" else requested,
        "visual_policy": policy,
        "visual_policy_requested": requested,
        "visual_policy_reason": reason,
        "adaptive_experimental": experimental,
    }


def register(engine, source, rgbd, primary_register):
    primary = primary_register(source, rgbd)
    orb_cache = engine._visual_cache
    engine._sift_visual_cache = {i: value for i, value in engine._sift_visual_cache.items()
                               if i in orb_cache}
    if primary is not None:
        if engine._visual_evidence is not None:
            engine._visual_evidence["feature_method"] = "orb"
        return primary
    if not engine.settings.color_recovery or rgbd is None or not engine.poses:
        return None
    stats = engine.backend["sift_fallback"]
    stats["attempts"] += 1
    for index, (feature, cloud) in engine._sift_visual_cache.items():
        primary_cloud = orb_cache[index][1]
        if primary_cloud is not None:
            engine._sift_visual_cache[index] = feature, primary_cloud
    engine._visual_cache = engine._sift_visual_cache
    engine.backend["visual_features"] = "sift"
    try:
        result = primary_register(source, rgbd)
        if result is not None:
            stats["verified"] += 1
            if engine._visual_evidence is not None:
                engine._visual_evidence["feature_method"] = "sift"
        return result
    finally:
        engine._sift_visual_cache = engine._visual_cache
        engine._visual_cache = orb_cache
        engine.backend["visual_features"] = "orb"
