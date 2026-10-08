"""Separate metadata/budget wrapper; original resident math remains inherited.

No GPU transforms, reductions, Huber terms, convergence/updates or Eigen solve
are reimplemented here. The roundoff-changing original ResidentICP still needs
fresh complete original CPU bridge gates on the NEW device-grid trajectory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hashlib
from scripts.research.archive.research_resident_icp import ResidentICP,ResidentFallback,GPU_SOURCE,CPU_BRIDGE_SOURCE,TERMS,BLOCK



def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class DeviceGridResidentICP(ResidentICP):
    def __init__(self,retrieval,solve,**kwargs):
        authority=getattr(retrieval,"proof_authority",None)
        if authority is not None:
            expected=authority.runtime_binding["resident_configuration"]
            actual={"device":f"CUDA:{retrieval.device_id}","max_points":kwargs.get("max_points",1000000),
                "max_scratch_bytes":kwargs.get("max_scratch_bytes",256*1024**2),
                "max_query_bytes":retrieval.max_query_bytes,"gpu_timing":kwargs.get("gpu_timing",True)}
            if actual!=expected:
                raise ValueError("Resident math configuration differs from freshly audited trajectory")
        super().__init__(retrieval,solve,**kwargs)
        self.retrieval_paths=[ROOT/"scripts/tool_paths.py",ROOT/"scripts/tool-catalog.json",
            ROOT/"scripts/research/cuda_device_flat_grid_registration.py",
            ROOT/"scripts/research/research_device_flat_grid_nn.cu",ROOT/"scripts/research/archive/cuda_flat_grid_registration.py",
            ROOT/"scripts/research/archive/research_flat_grid_nn.cu",ROOT/"scripts/research/validate_device_flat_grid_proof.py"]
        self.resident_math_paths=[ROOT/"scripts/research/archive/research_resident_icp.py",Path(__file__)]
        # Parent's metadata assumes OptiX. Replace it before any actual calls;
        # report() below never invokes its hardcoded OptiX provenance branch.
        self.provenance.pop("retrieval_script_sha256",None)
        self.provenance.update(retrieval="true-device original-double flat grid",
            actual_retrieval_sha256={str(path.relative_to(ROOT)):file_hash(path) for path in self.retrieval_paths},
            original_resident_math_sha256={str(path.relative_to(ROOT)):file_hash(path) for path in self.resident_math_paths},
            authority="Research-only device-flat trajectory; unaudited timing requires fresh synthetic/all-nine trajectory authority",
            roundoff="Original resident transforms/reductions/Eigen bridge inherited unchanged; this remains roundoff-changing research")
        self.statistics["peak_combined_resident_and_query_scratch_bytes"]=0

    def _match_resident(self,source,target,initial):
        # Only a stronger memory bound wraps the unchanged parent body.
        n=len(source.points);target_n=len(target.points)
        blocks=(n+BLOCK-1)//BLOCK
        resident_owned=2*n*24+target_n*24+n*4+blocks*TERMS*8+TERMS*8+16*8
        query_worst=n*136+80
        combined=resident_owned+query_worst
        self.statistics["peak_combined_resident_and_query_scratch_bytes"]=max(
            self.statistics["peak_combined_resident_and_query_scratch_bytes"],combined)
        if query_worst>self.retrieval.max_query_bytes:
            raise ResidentFallback("Device query scratch exceeds separate retrieval byte cap")
        if combined>self.max_scratch_bytes:
            raise ResidentFallback("Combined resident math and device query scratch exceeds configured byte cap")
        return super()._match_resident(source,target,initial)

    def report(self):
        from scripts.research.archive.research_resident_icp import runtime_source_hash
        unchanged_retrieval=all(file_hash(ROOT/path)==digest for path,digest in self.provenance["actual_retrieval_sha256"].items())
        unchanged_math=all(file_hash(ROOT/path)==digest for path,digest in self.provenance["original_resident_math_sha256"].items())
        return {"provenance":dict(self.provenance,source_unchanged=runtime_source_hash()==self.provenance["source_sha256"],
            actual_retrieval_unchanged=unchanged_retrieval,original_resident_math_unchanged=unchanged_math,
            original_gpu_source_sha256=hashlib.sha256(GPU_SOURCE.encode()).hexdigest(),
            original_cpu_bridge_source_sha256=hashlib.sha256(CPU_BRIDGE_SOURCE.encode()).hexdigest()),
            "device":f"CUDA:{self.device_id}","gpu_timing":self.gpu_timing,
            "statistics":{key:value for key,value in self.statistics.items()
                if self.gpu_timing or key not in ("gpu_transform_ms","gpu_equations_ms")},
            "uncollected_statistics":[] if self.gpu_timing else ["gpu_transform_ms","gpu_equations_ms"],
            "fallback_reasons":dict(self.fallback_reasons),
            "retrieval_statistics":dict(self.retrieval.statistics),"scratch_limit_bytes":self.max_scratch_bytes,
            "query_scratch_limit_bytes":self.retrieval.max_query_bytes,"max_points":self.max_points,
            "memory_scope":"Original-order target XYZ and sorted indexes under one separate retained cache cap; wrapper bounds math-owned plus worst-case query temporaries together. Pools and host packets separately observed.",
            "timer_scope":"device_total_wall_s and device_raw_classify_counter_s are inclusive selected-stream wall, absorbing queued earlier transforms/uploads; flagged-copy and CPU-resolution are nested. Correction/XYZ enqueue timers do not wait completion. NN host/control sync, device events and later equation copies overlap. device_query_download_bytes counts full64-byte packets, including audit metadata, rather than XYZ alone."}
