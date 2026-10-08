"""Reuse immutable source levels only during one registration decision."""

from functools import wraps


class SourcePyramid:
    def __init__(self, source):
        self.source = source
        self.levels = {}
        self.tensors = {}

    def level(self, voxel):
        if voxel not in self.levels:
            self.levels[voxel] = self.source.voxel_down_sample(voxel)
        return self.levels[voxel]

    def tensor(self, voxel, convert):
        if voxel not in self.tensors:
            self.tensors[voxel] = convert(self.level(voxel))
        return self.tensors[voxel]


class TargetPyramid(SourcePyramid):
    """Prepared levels for an immutable raw observation in the keyframe bank."""

    def __init__(self, source):
        super().__init__(source)
        self._raw_normals_ready = source.has_normals()

    def _check_normals(self):
        # Experimental proposals can defer raw normal preparation until a
        # geometric fallback needs it. Voxel levels inherit their orientation;
        # invalidate both forms when that derivative input becomes available.
        ready = self.source.has_normals()
        if ready != self._raw_normals_ready:
            self.levels.clear()
            self.tensors.clear()
            self._raw_normals_ready = ready

    def level(self, voxel):
        self._check_normals()
        if voxel not in self.levels:
            import open3d as o3d

            level = self.source.voxel_down_sample(voxel)
            level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
            self.levels[voxel] = level
        return self.levels[voxel]

    def tensor(self, voxel, convert):
        self._check_normals()
        return super().tensor(voxel, convert)


def reuse_icp_source(register):
    """Release source levels on success, rejection or exception.

    Several verified proposals can run ICP on the same camera observation.
    Only source downsampling/conversion is shared; fresh model/target data,
    pose guesses, ICP budgets and all verification remain per attempt.
    """
    @wraps(register)
    def wrapped(engine, source_pcd, rgbd=None):
        previous = getattr(engine, "_icp_source_pyramid", None)
        engine._icp_source_pyramid = SourcePyramid(source_pcd)
        try:
            return register(engine, source_pcd, rgbd)
        finally:
            if previous is None:
                del engine._icp_source_pyramid
            else:
                engine._icp_source_pyramid = previous
    return wrapped
