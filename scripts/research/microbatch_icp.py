"""Research-only independent seed scheduling with original resident ICP math.

No numerical library is imported until construction. The initial experiment
retains the CPU Eigen solver boundary. A prepared pair owns immutable shared
source/target buffers; each lane owns a nonblocking stream and mutable ICP state.
No original Finish hook or old proof token is installed or minted here.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import sys
import time
from types import CodeType

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

POLICY="independent-seed-explicit-stream-original-resident-cpu-solve-v1"
STAGES=((.12,40),(.06,30),(.03,20))
SOURCE_FILES=("scripts/research/microbatch_icp.py",
    "scripts/research/cuda_device_flat_grid_registration.py",
    "scripts/research/research_device_flat_grid_nn.cu",
    "scripts/research/archive/cuda_flat_grid_registration.py",
    "scripts/research/archive/research_flat_grid_nn.cu",
    "scripts/research/archive/cuda_uniform_grid_registration.py",
    "scripts/research/archive/research_uniform_grid_nn.cu",
    "scripts/research/archive/research_resident_icp.py",
    "scanner_server/fragments.py","scripts/tool_paths.py","scripts/tool-catalog.json",
    "scripts/profile_session.py","tests/test_microbatch_icp.py",
    "scripts/research/gpu_icp_experiment_driver.py","scripts/research/gpu_icp_experiment_protocol.py",
    "scripts/research/gpu_icp_experiment_capture.py","tests/test_gpu_icp_experiment_protocol.py",
    "tests/test_gpu_icp_experiment_capture.py")


class MicrobatchFailure(RuntimeError):
    """A failed batch never supplies partial successful registration results."""


def require(value,message):
    if not value:raise MicrobatchFailure(message)


def file_pins():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def numerical_modules(o3d):
    backend=sys.modules.get(o3d.geometry.PointCloud.__module__.rsplit(".",1)[0])
    numpy_core=sys.modules.get("numpy._core._multiarray_umath")
    if numpy_core is None:numpy_core=sys.modules.get("numpy.core._multiarray_umath")
    require(backend is not None and getattr(backend,"__file__",None)
        and numpy_core is not None and getattr(numpy_core,"__file__",None),
        "Installed search/arithmetic binary layout cannot be bound")
    return backend,numpy_core


def numerical_runtime(np,o3d,cp,*,hash_binaries=True):
    """Bind binary bytes at setup; keep boundary metadata checks inexpensive."""
    backend,numpy_core=numerical_modules(o3d)
    configuration=json.dumps(getattr(np.__config__,"CONFIG",{}),sort_keys=True,
        separators=(",",":"),allow_nan=False)
    result={"numpy_build_configuration_sha256":hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
        "numpy_cpu_features":{key:bool(value) for key,value in getattr(numpy_core,"__cpu_features__",{}).items()},
        "nvrtc_version":list(cp.cuda.nvrtc.getVersion())}
    if hash_binaries:
        result.update(open3d_backend_binary_sha256=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest(),
            numpy_core_binary_sha256=hashlib.sha256(Path(numpy_core.__file__).read_bytes()).hexdigest())
    return result


def current_thread_policy(o3d,cv2):
    return {"open3d_threads":o3d.utility.get_max_threads(),"opencv_threads":cv2.getNumThreads(),
        "omp_threads":os.environ.get("OMP_NUM_THREADS")}


def code_state(code):
    return (code.co_code,code.co_names,code.co_varnames,code.co_argcount,
        code.co_kwonlyargcount,code.co_posonlyargcount,code.co_freevars,code.co_cellvars,
        code.co_flags,tuple(code_state(v) if isinstance(v,CodeType) else v for v in code.co_consts))


def source_method(function,path,owner):
    """Fail before enqueue if a loaded numerical body differs from its source."""
    module=ast.parse(Path(path).read_text(encoding="utf-8"))
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name==owner)
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==function.__name__)
    compiled=compile(module,function.__code__.co_filename,"exec",dont_inherit=True)
    class_code=next(v for v in compiled.co_consts if isinstance(v,CodeType) and v.co_name==owner)
    method_code=next(v for v in class_code.co_consts if isinstance(v,CodeType) and v.co_name==function.__name__)
    require(code_state(function.__code__)==code_state(method_code),"Loaded ICP/NN body differs from original source")
    return method


def derive_lane_methods(nn_class,resident_class):
    """Only stream/permit names and readonly shared uploads change in the AST."""
    nearest=source_method(nn_class.nearest_device,ROOT/SOURCE_FILES[1],"DeviceFlatGridICP")
    body=source_method(resident_class._match_resident,ROOT/SOURCE_FILES[7],"ResidentICP")
    edits={"stream":0,"permit":0,"source":0,"normals":0,"upload_counter":0}

    class ReplaceNearest(ast.NodeTransformer):
        def visit_Attribute(self,node):
            self.generic_visit(node)
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse("cp.cuda.Stream.null",mode="eval").body,include_attributes=False):
                edits["stream"]+=1
                return ast.copy_location(ast.parse("self.stream",mode="eval").body,node)
            if isinstance(node.value,ast.Name) and node.value.id=="self" and node.attr=="proof_authority":
                edits["permit"]+=1
                return ast.copy_location(ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),attr="batch_permit",ctx=node.ctx),node)
            return node

    class ReplaceUploads(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple)
                    and [getattr(v,"id",None) for v in node.targets[0].elts]==["original","moving"]):
                require(isinstance(node.value,ast.Tuple) and ast.unparse(node.value.elts[0])=="cp.asarray(points)","Original source upload seam changed")
                node.value.elts[0]=ast.copy_location(ast.parse("self.shared_original",mode="eval").body,node.value.elts[0]);edits["source"]+=1
            elif len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="normals":
                require(ast.unparse(node.value)=="cp.asarray(normal_points)","Original normal upload seam changed")
                node.value=ast.copy_location(ast.parse("self.shared_normals",mode="eval").body,node.value);edits["normals"]+=1
            return self.generic_visit(node)
        def visit_AugAssign(self,node):
            if ast.unparse(node.target)=="self.statistics['host_to_device_bytes']":
                require(ast.unparse(node.value)=="original.nbytes + normals.nbytes","Original upload accounting seam changed")
                node.value=ast.copy_location(ast.Constant(value=0),node.value);edits["upload_counter"]+=1
            return self.generic_visit(node)

    nearest=ReplaceNearest().visit(nearest);body=ReplaceUploads().visit(body)
    require(edits=={"stream":1,"permit":1,"source":1,"normals":1,"upload_counter":1},"Unexpected original orchestration seams")
    def build(node,original):
        namespace=dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),"microbatch-derived-original-body","exec",dont_inherit=True),namespace)
        return namespace[node.name]
    return build(nearest,nn_class.nearest_device),build(body,resident_class._match_resident),edits


def run_ordered(jobs,schedule):
    """Consume seed results in input order, and join all workers on failure."""
    require(len(jobs) in (2,4),"Require exactly two or four independent seeds")
    require(schedule in ("serial","concurrent"),"Unknown seed schedule")
    if schedule=="serial":return [job() for job in jobs]
    with ThreadPoolExecutor(max_workers=len(jobs),thread_name_prefix="microbatch-icp") as pool:
        futures=[pool.submit(job) for job in jobs]
        return [future.result() for future in futures]


def matrix_record(np,array,*,contiguous=True):
    require(array.dtype==np.float64 and (not contiguous or array.flags.c_contiguous),"Require original FP64 matrices and declared cloud layout")
    return {"shape":list(array.shape),"dtype":array.dtype.str,"nbytes":array.nbytes,
        "sha256":hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def cloud_record(np,cloud):
    arrays={name:np.asarray(getattr(cloud,name)) for name in ("points","normals","colors")}
    require(all(array.ndim==2 and array.shape[1]==3 and np.isfinite(array).all() for array in arrays.values()),
        "Require finite original cloud arrays")
    return {name:matrix_record(np,array) for name,array in arrays.items()}


def scratch_bound(rows,source_bytes,normal_bytes,cache_bytes,lanes):
    require(type(rows) is int and 0<rows<=1000000 and type(lanes) is int and lanes in (2,4),"Invalid batch scratch domain")
    blocks=(rows+127)//128
    per_lane=rows*24+rows*4+blocks*30*8+30*8+16*8+rows*136+80
    return {"per_lane_bytes":per_lane,"query_bytes":rows*136+80,
        "combined_bytes":cache_bytes+source_bytes+normal_bytes+lanes*per_lane}


class MicrobatchICP:
    def __init__(self,solve_factory,*,device="CUDA:0",audit=True,timing_permit=None,
                 max_points=1000000,max_query_bytes=136*1024**2,
                 max_cache_bytes=256*1024**2,max_total_bytes=512*1024**2):
        require(callable(solve_factory) and type(audit) is bool,"Require an original solve factory and explicit audit mode")
        require(str(device).startswith("CUDA:") and str(device).split(":")[1].isdigit(),"Require explicit CUDA device")
        require(all(type(v) is int and v>0 for v in (max_points,max_query_bytes,max_cache_bytes,max_total_bytes)),"Require bounded memory policy")
        require(max_points<=1000000 and max_query_bytes<=136*1024**2 and max_cache_bytes<=256*1024**2
            and max_total_bytes<=1024*1024**2,"Memory/count policy exceeds experiment domain")
        require(audit or timing_permit is not None,"Unaudited batches require a fresh batch-specific timing permit")
        if not audit:
            protocol=importlib.import_module("scripts.research.gpu_icp_experiment_protocol")
            protocol.assert_registered_microbatch_permit(timing_permit)
        self.source_before=file_pins();self.device_id=int(str(device).split(":")[1])
        self.audit,self.timing_permit=audit,timing_permit
        self.max_points,self.max_query_bytes,self.max_cache_bytes,self.max_total_bytes=max_points,max_query_bytes,max_cache_bytes,max_total_bytes
        self.solve_factory=solve_factory;self.lanes=[];self.lease=None;self.failure=None;self.closed=False;self.active=False
        self.batches=[];self.cleanup_failures=[];self.shared=None;self.runtime=None
        self.statistics={"cold_setup_s":0.,"shared_prepare_s":0.,"batch_wall_s":0.,"cleanup_s":0.,
            "shared_upload_bytes":0,"peak_combined_owned_bytes":0,"successful_batches":0,
            "source_checks_s":0.,"source_checks":0}
        started=time.perf_counter()
        try:
            self.cp=importlib.import_module("cupy");self.np=importlib.import_module("numpy")
            nn=importlib.import_module("scripts.research.cuda_device_flat_grid_registration")
            resident=importlib.import_module("scripts.research.archive.research_resident_icp")
            require(resident.STAGES==STAGES and resident.BLOCK==128 and resident.TERMS==30,"Original ICP recipe changed")
            self.resident=resident
            self.nearest_body,self.resident_body,self.edits=derive_lane_methods(nn.DeviceFlatGridICP,resident.ResidentICP)
            methods=((nn.DeviceFlatGridICP._nearest_device_impl,ROOT/SOURCE_FILES[1],"DeviceFlatGridICP"),
                (nn.DeviceFlatGridICP._raw_device,ROOT/SOURCE_FILES[5],"UniformGridICP"))
            methods+=tuple((getattr(resident.ResidentICP,name),ROOT/SOURCE_FILES[7],"ResidentICP")
                for name in ("_transform","_equations","_iteration_data"))
            for function,path,owner in methods:source_method(function,path,owner)
            original_ast=ast.parse((ROOT/SOURCE_FILES[7]).read_text(encoding="utf-8"))
            gpu_literal=next(node.value for node in original_ast.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=="GPU_SOURCE" for target in node.targets))
            require(resident.GPU_SOURCE==ast.literal_eval(gpu_literal),"Loaded original math shader differs from source")
            self.nn_class=nn.DeviceFlatGridICP
            self.shared=nn.DeviceFlatGridICP(device,max_clouds=1,max_cache_bytes=max_cache_bytes,
                audit_nearest=True,audit_misses=True,miss_policy="direct-miss-research-v1",max_query_bytes=max_query_bytes)
            with self.cp.cuda.Device(self.device_id),self.cp.cuda.Stream.null:
                self.kernels=tuple(self.cp.RawKernel(resident.GPU_SOURCE,name,options=("--fmad=false",))
                    for name in ("transform_points","normal_partials","collapse_partials"))
                for kernel in self.kernels:kernel.compile()
            self.prototype_solve=solve_factory()
            require(isinstance(self.prototype_solve,resident.EigenSolve),"Require the original pinned EigenSolve bridge")
            self.solve_metadata=json.loads(json.dumps(self.prototype_solve.metadata,sort_keys=True,allow_nan=False))
            from scripts.profile_session import source_hash
            self.core_sha256=source_hash()
            o3d=importlib.import_module("open3d")
            cv2=importlib.import_module("cv2")
            properties=self.cp.cuda.runtime.getDeviceProperties(self.device_id)
            name=properties["name"]
            if isinstance(name,bytes):name=name.decode("utf-8")
            self.runtime={"python":platform.python_version(),"numpy":self.np.__version__,"cupy":self.cp.__version__,
                "open3d":o3d.__version__,"device":f"CUDA:{self.device_id}","gpu_name":name,
                "compute_capability":[properties["major"],properties["minor"]],
                "cuda_driver_version":self.cp.cuda.runtime.driverGetVersion(),
                "cuda_runtime_version":self.cp.cuda.runtime.runtimeGetVersion(),
                "thread_policy":current_thread_policy(o3d,cv2),
                "solve_library_sha256":hashlib.sha256(self.prototype_solve.path.read_bytes()).hexdigest()}
            self.o3d,self.cv2=o3d,cv2
            self.numeric_runtime=numerical_runtime(self.np,o3d,self.cp)
            self.numeric_modules=numerical_modules(o3d)
            self.numeric_module_paths=tuple(str(Path(module.__file__).resolve()) for module in self.numeric_modules)
            self.runtime.update(self.numeric_runtime)
            self.function_states=tuple((function,function.__code__,function.__defaults__,function.__kwdefaults__)
                for function in (nn.DeviceFlatGridICP._nearest_device_impl,nn.DeviceFlatGridICP._raw_device,
                    resident.ResidentICP._transform,resident.ResidentICP._equations,resident.ResidentICP._iteration_data))
        except BaseException as error:
            self.failure=error
            if self.shared is not None:
                try:self.shared.close()
                except BaseException as cleanup:error.add_note(f"Partial shared-cache cleanup failed: {cleanup}")
            raise
        finally:self.statistics["cold_setup_s"]+=time.perf_counter()-started

    def binding(self):
        return {"method_policy":POLICY,"source_sha256":self.core_sha256,"artifacts_sha256":dict(self.source_before),
            "runtime":dict(self.runtime),"solve_metadata":dict(self.solve_metadata),
            "configuration":{"device":f"CUDA:{self.device_id}","max_points":self.max_points,
                "max_query_bytes":self.max_query_bytes,"max_cache_bytes":self.max_cache_bytes,
                "max_total_bytes":self.max_total_bytes,"stages":[list(stage) for stage in STAGES],
                "gpu_timing":False,"lane_streams":"private nonblocking; no forced-null query execution"}}

    def _healthy(self):
        started=time.perf_counter()
        try:
            require(not self.closed and self.failure is None,"Batch helper is closed or failed; no retry")
            require(file_pins()==self.source_before,"Batch or inherited sources changed")
            require(hashlib.sha256(self.prototype_solve.path.read_bytes()).hexdigest()==self.runtime["solve_library_sha256"],
                "Original solve library changed")
            require(self.prototype_solve.metadata==self.solve_metadata,"Original solve metadata changed")
            modules=numerical_modules(self.o3d)
            require(all(current is old for current,old in zip(modules,self.numeric_modules))
                and tuple(str(Path(module.__file__).resolve()) for module in modules)==self.numeric_module_paths,
                "Loaded search/arithmetic module ownership changed")
            environment={key:value for key,value in self.numeric_runtime.items() if key not in (
                "open3d_backend_binary_sha256","numpy_core_binary_sha256")}
            require(numerical_runtime(self.np,self.o3d,self.cp,hash_binaries=False)==environment,
                "Installed search/arithmetic configuration changed")
            require(current_thread_policy(self.o3d,self.cv2)==self.runtime["thread_policy"],
                "Actual native/OpenCV/OMP thread policy changed")
            require(self.resident.STAGES==STAGES and self.resident.BLOCK==128 and self.resident.TERMS==30,
                "Original resident numerical recipe changed")
            for function,code,defaults,kwdefaults in self.function_states:
                require(function.__code__ is code and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults,
                    "Inherited function configuration changed")
        finally:
            self.statistics["source_checks_s"]+=time.perf_counter()-started
            self.statistics["source_checks"]+=1

    def _sync_lanes(self):
        failures=[]
        with self.cp.cuda.Device(self.device_id):
            for lane in self.lanes:
                try:lane.stream.synchronize()
                except BaseException as error:failures.append(error)
        if failures:
            for error in failures[1:]:failures[0].add_note(str(error))
            raise failures[0]

    def _lane(self,index):
        owner=self;resident=self.resident;cp=self.cp
        class LaneNN(self.nn_class):
            nearest_device=owner.nearest_body
            def _dataset(self,target,radius):
                require(owner.lease is not None and target is owner.lease["target"],"Foreign shared pair")
                return owner.lease["items"][radius]
        retrieval=object.__new__(LaneNN)
        # Cache mutation/ownership remains exclusively with the shared parent;
        # concurrent lanes only inherit raw launches and exact CPU resolution.
        for key in ("cp","device_id","kernel","classify","pack","scatter","scatter_metrics","max_query_bytes"):
            setattr(retrieval,key,getattr(self.shared,key))
        retrieval.statistics={key:(0. if isinstance(value,float) else 0) for key,value in self.shared.statistics.items()}
        retrieval.audit_nearest=retrieval.audit_misses=self.audit
        retrieval.miss_policy="direct-miss-research-v1";retrieval.batch_permit=self.timing_permit
        with cp.cuda.Device(self.device_id):stream=cp.cuda.Stream(non_blocking=True)
        retrieval.stream=stream
        class LaneICP(resident.ResidentICP):
            _match_resident=owner.resident_body
            def match(self,source,target,initial):
                started=time.perf_counter()
                try:
                    require(self.retrieval.stream is self.stream and self.stream.device_id==owner.device_id
                        and self.retrieval.audit_nearest is owner.audit and self.retrieval.audit_misses is owner.audit
                        and self.retrieval.batch_permit is owner.timing_permit
                        and self.retrieval.miss_policy=="direct-miss-research-v1", "Lane query policy/stream changed")
                    with cp.cuda.Device(owner.device_id),self.stream:
                        result=self._match_resident(source,target,initial)
                        self.stream.synchronize()
                        return result
                finally:
                    self.statistics["calls"]+=1;self.statistics["wall_s"]+=time.perf_counter()-started
        solve=self.prototype_solve if index==0 else self.solve_factory()
        require(isinstance(solve,resident.EigenSolve) and solve.metadata==self.solve_metadata,"Lane solve bridge differs")
        with cp.cuda.Device(self.device_id),stream:
            lane=LaneICP(retrieval,solve,gpu_timing=False,owns_retrieval=False,max_points=self.max_points,
                max_scratch_bytes=self.max_total_bytes)
        lane.stream=stream;lane.index=index
        lane.transform_kernel,lane.partial_kernel,lane.collapse_kernel=self.kernels
        lane.provenance.pop("retrieval_script_sha256",None)
        return lane

    def prepare_pair(self,source,target,*,lanes):
        """Prepare an owned readonly lease; caller must finish lanes before release."""
        require(lanes in (2,4) and not self.active,"Prepare a pair only outside active workers")
        self._healthy();np,cp=self.np,self.cp
        inputs={"source":cloud_record(np,source),"target":cloud_record(np,target)}
        points=np.asarray(source.points);normals=np.asarray(target.normals)
        require(len(points)>0 and len(target.points)>0 and len(points)<=self.max_points
            and len(target.points)<=self.max_points and normals.shape==np.asarray(target.points).shape,
            "Unsupported original shared point/normal counts")
        bound=scratch_bound(len(points),points.nbytes,normals.nbytes,self.max_cache_bytes,lanes)
        require(bound["query_bytes"]<=self.max_query_bytes and bound["combined_bytes"]<=self.max_total_bytes,
            "Concurrent batch worst-case byte cap exceeded before allocation")
        self._sync_lanes()
        if self.lease is not None:
            self.release_pair(self.lease)
        with cp.cuda.Device(self.device_id),cp.cuda.Stream.null:
            items={radius:self.shared._dataset(target,radius) for radius,_ in STAGES}
            require(all(item["data"] is not None and not item["cpu_only"] for item in items.values()),
                "Shared grid unsupported/cache-budget bypass; whole-call fallback prohibited")
            original=cp.asarray(points);normal_data=cp.asarray(normals)
            cp.cuda.Stream.null.synchronize()
        combined=scratch_bound(len(points),original.nbytes,normal_data.nbytes,self.shared.cache_bytes,lanes)["combined_bytes"]
        self.statistics["peak_combined_owned_bytes"]=max(self.statistics["peak_combined_owned_bytes"],combined)
        self.statistics["shared_upload_bytes"]+=original.nbytes+normal_data.nbytes
        self.lease={"target":target,"items":items,"source_points":original,"normals":normal_data,
            "inputs":inputs,"device_id":self.device_id,"owned_bytes":combined,"closed":False,"streams":[],
            "owner":self}
        return self.lease

    def register_lease_stream(self,lease,stream):
        require(lease is self.lease and not lease["closed"],"Foreign/closed pair lease")
        require(stream.device_id==self.device_id,"Lease stream belongs to another device")
        if all(stream is not old for old in lease["streams"]):
            require(len(lease["streams"])<4,"Pair lease stream cap exceeded")
            lease["streams"].append(stream)

    def release_pair(self,lease):
        require(lease is self.lease and not self.active,"Cannot release a foreign or executing pair lease")
        with self.cp.cuda.Device(self.device_id):
            errors=[]
            for stream in lease["streams"]:
                try:stream.synchronize()
                except BaseException as error:errors.append(error)
            if errors:
                if self.failure is None:self.failure=errors[0]
                for error in errors[1:]:errors[0].add_note(str(error))
                if self.failure is errors[0]:raise self.failure
                self.failure.add_note(f"Lease completion also failed: {errors[0]}")
                raise self.failure from errors[0]
        for lane in self.lanes:lane.shared_original=lane.shared_normals=None
        lease["closed"]=True
        lease["source_points"]=lease["normals"]=None
        lease["items"]={}
        self.lease=None

    def match_seeds(self,source,target,initials,*,schedule="concurrent"):
        self._healthy();require(not self.active,"A shared helper cannot run overlapping pair batches")
        np,cp=self.np,self.cp
        require(len(initials) in (2,4),"Require exactly two or four independent original seeds")
        require(schedule in ("serial","concurrent"),"Unknown seed schedule")
        points=np.asarray(source.points);target_points=np.asarray(target.points);normals=np.asarray(target.normals)
        arrays=(points,target_points,normals)
        require(all(a.dtype==np.float64 and a.ndim==2 and a.shape[1]==3 and a.flags.c_contiguous
            and len(a)>0 and len(a)<=self.max_points and np.isfinite(a).all() for a in arrays)
            and normals.shape==target_points.shape,"Unsupported original shared cloud/normal data")
        seeds=[np.asarray(seed) for seed in initials]
        require(all(seed.dtype==np.float64 and seed.shape==(4,4) and np.isfinite(seed).all()
            for seed in seeds),"Require exact finite original FP64 seeds; original C/F layout is retained")
        before={"source":cloud_record(np,source),"target":cloud_record(np,target),
            "seeds":[matrix_record(np,seed,contiguous=False) for seed in seeds]}
        if not self.audit:
            protocol=importlib.import_module("scripts.research.gpu_icp_experiment_protocol")
            protocol.validate_microbatch_permit(self.timing_permit,binding=self.binding(),pair_binding=before,schedule=schedule)
        started=time.perf_counter();primary=None
        try:
            prepare=time.perf_counter()
            while len(self.lanes)<len(seeds):self.lanes.append(self._lane(len(self.lanes)))
            lease=self.prepare_pair(source,target,lanes=len(seeds))
            selected=self.lanes[:len(seeds)]
            stats_before=[(dict(lane.statistics),dict(lane.retrieval.statistics)) for lane in selected]
            for lane in selected:
                lane.shared_original=lease["source_points"];lane.shared_normals=lease["normals"]
                self.register_lease_stream(lease,lane.stream)
            self.statistics["shared_prepare_s"]+=time.perf_counter()-prepare
            self.active=True
            jobs=[lambda lane=lane,seed=seed:lane.match(source,target,seed) for lane,seed in zip(selected,seeds)]
            results=run_ordered(jobs,schedule)
            self._sync_lanes()
            after={"source":cloud_record(np,source),"target":cloud_record(np,target),
                "seeds":[matrix_record(np,seed,contiguous=False) for seed in seeds]}
            require(before==after,"Original pair/seed input bytes changed")
            require(all(lane.statistics["cpu_fallback_calls"]==0 and lane.retrieval.statistics["audit_index_mismatches"]==0
                and lane.retrieval.statistics["audit_false_misses"]==0 and lane.retrieval.statistics["device_malformed_results"]==0
                for lane in selected),"A lane fallback/audit/malformed fault cannot certify a batch")
            self._healthy();self.statistics["successful_batches"]+=1
            lane_deltas=[{"index":lane.index,"statistics":{key:value-old[0][key] for key,value in lane.statistics.items()},
                "retrieval_statistics":{key:value-old[1][key] for key,value in lane.retrieval.statistics.items()}}
                for lane,old in zip(selected,stats_before)]
            self.batches.append({"schedule":schedule,"lanes":len(seeds),"inputs":before,"wall_s":time.perf_counter()-started,
                "cpu_solver_boundary":True,"full_call_fallbacks":0,"input_bytes_unchanged":True,"lane_deltas":lane_deltas})
            return results
        except BaseException as error:self.failure=primary=error;raise
        finally:
            try:self._sync_lanes()
            except BaseException as cleanup:
                self.cleanup_failures.append(str(cleanup))
                if primary is not None:primary.add_note(f"Batch stream completion failed: {cleanup}")
                else:self.failure=cleanup;raise
            finally:self.active=False;self.statistics["batch_wall_s"]+=time.perf_counter()-started

    def report(self):
        return {"policy":POLICY,"binding":self.binding(),"audit":self.audit,"closed":self.closed,
            "statistics":dict(self.statistics),"shared_retrieval_statistics":dict(self.shared.statistics),
            "lanes":[{"index":lane.index,"stream_ptr":int(lane.stream.ptr),"stream_device":lane.stream.device_id,
                "statistics":dict(lane.statistics),"retrieval_statistics":dict(lane.retrieval.statistics),
                "solve_metadata":getattr(lane.solve,"metadata",None)} for lane in self.lanes],
            "batches":list(self.batches),"source_unchanged":file_pins()==self.source_before,
            "failure":None if self.failure is None else {"type":type(self.failure).__name__,"message":str(self.failure)},
            "cleanup_failures":list(self.cleanup_failures),"stream_scope":"Explicit nonblocking lane streams; shared setup/null stream completes before workers; join/sync all before eviction",
            "memory_scope":"Shared original source/normals and target cache plus all perlane live math/query scratch; CUDA pools/host audit packets separate",
            "timing_scope":"Driver batch wall includes input/permit preflight, preparation, joins and completion; adapter batch_wall_s starts after input/permit preflight. source_checks_s is nested boundary validation, not additive; lane walls overlap, never sum. Binary bytes hash at cold construction and driver before/after closure; boundary checks retain loaded module ownership/configuration. Cold construction/close separate",
            "whole_finish_authority":False}

    def close(self):
        if self.closed:return
        require(not self.active,"Cannot release an active shared pair")
        started=time.perf_counter()
        try:
            # Native target buffers cannot be released until both internal and
            # external registered lease streams have proved completion. A
            # failed completion retains owners and permits only cleanup retry.
            self._sync_lanes()
            if self.lease is not None:self.release_pair(self.lease)
            self.shared.close()
            self.closed=True
        except BaseException as error:
            self.cleanup_failures.append(str(error))
            if self.failure is None:self.failure=error
            if self.failure is error:raise
            self.failure.add_note(f"Batch close also failed: {error}")
            raise self.failure from error
        finally:self.statistics["cleanup_s"]+=time.perf_counter()-started
