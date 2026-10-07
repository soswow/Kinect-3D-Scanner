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
