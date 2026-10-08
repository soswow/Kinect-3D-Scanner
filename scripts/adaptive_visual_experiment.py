"""Research ORB-first proposals with independently verified SIFT fallback.

The original engine authorizes every proposal. This changes only which feature
bank is tried after ORB fails; no pose, verification or recovery is cached.
"""

from scanner_server.engine import ScanEngine


class AdaptiveVisualEngine(ScanEngine):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self.backend["visual_features"] != "orb":
            raise ValueError("Adaptive visual experiment requires the ORB primary recipe")
        self._sift_visual_cache = {}
        self.backend["experimental_visual_policy"] = "orb_then_sift"
        self.backend["sift_fallback"] = {"attempts": 0, "verified": 0}

    def _visual_register(self, source, rgbd):
        primary = super()._visual_register(source, rgbd)
        orb_cache = self._visual_cache
        # Both banks retain only the engine's current observed keyframes.
        self._sift_visual_cache = {i: value for i, value in self._sift_visual_cache.items()
                                   if i in orb_cache}
        if primary is not None:
            if self._visual_evidence is not None:
                self._visual_evidence["feature_method"] = "orb"
            return primary
        if not self.settings.color_recovery or rgbd is None or not self.poses:
            return None
        stats = self.backend["sift_fallback"]
        stats["attempts"] += 1
        # Reuse original immutable raw clouds where a SIFT feature entry exists.
        for index, (feature, cloud) in self._sift_visual_cache.items():
            primary_cloud = orb_cache[index][1]
            if primary_cloud is not None:
                self._sift_visual_cache[index] = feature, primary_cloud
        self._visual_cache = self._sift_visual_cache
        self.backend["visual_features"] = "sift"
        try:
            result = super()._visual_register(source, rgbd)
            if result is not None:
                stats["verified"] += 1
                if self._visual_evidence is not None:
                    self._visual_evidence["feature_method"] = "sift"
            return result
        finally:
            self._sift_visual_cache = self._visual_cache
            self._visual_cache = orb_cache
            self.backend["visual_features"] = "orb"
