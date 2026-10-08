"""Standalone flat candidate iterator; no production selection.

All cache, original-double geometry, launch/output layout, CPU ambiguity/miss
resolver, estimator and convergence methods are inherited from the immutable
original uniform-grid adapter. Only construction selects the separate shader.
Unaudited research misses require newly measured flat-specific proof authority.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


import hashlib
import time

from scripts.research.archive.cuda_uniform_grid_registration import UniformGridICP, DOMAIN as ORIGINAL_DOMAIN
from scripts.research.archive.validate_flat_grid_proof import FlatGridProofAuthority

CUDA_SOURCE = Path(__file__).with_name("research_flat_grid_nn.cu")
DOMAIN = dict(ORIGINAL_DOMAIN,
    version="flat-dyadic27-original-double-v1",
    traversal="27 parallel cell lookups; exclusive shared prefix of disjoint ranges; ordinal tid+128*k; upper_bound maps every original candidate exactly once",
    output="unchanged original five-column first/second IDs, FP64 squared distances and visits; fixed128 threads")


class FlatUniformGridICP(UniformGridICP):
    def __init__(self, device="CUDA:0", max_clouds=64, max_cache_bytes=256*1024**2,
                 audit_nearest=False, audit_misses=False, miss_policy="cpu-fallback", proof_authority=None):
        if proof_authority is not None:
            if not isinstance(proof_authority, FlatGridProofAuthority):
                raise ValueError("Flat traversal requires separate flat-specific measured proof authority")
            expected = dict(proof_authority.artifact_sha256)
            root = Path(__file__).parents[3]
            for path in (Path(__file__), CUDA_SOURCE, Path(__file__).with_name("validate_flat_grid_proof.py"),
                         ROOT/"scripts/tool_paths.py", ROOT/"scripts/tool-catalog.json"):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(str(path.relative_to(root))):
                    raise ValueError("Flat adapter/kernel/validator bytes differ from the independent proofs")
        started = time.perf_counter()
        super().__init__(device, max_clouds, max_cache_bytes, audit_nearest, audit_misses,
                         miss_policy, proof_authority)
        self.inherited_adapter_setup_s = time.perf_counter()-started
        started = time.perf_counter()
        with self.cp.cuda.Device(self.device_id), self.cp.cuda.Stream.null:
            self.kernel = self.cp.RawKernel(CUDA_SOURCE.read_text(), "flat_grid_nearest_two",
                options=("--std=c++11", "--fmad=false"))
            self.kernel.compile()
        self.flat_shader_setup_s = time.perf_counter()-started
        self.kernel_sha256 = hashlib.sha256(CUDA_SOURCE.read_bytes()).hexdigest()
