"""No NumPy/Open3D/CUDA imports: scheduling, source seams and lifetime contracts."""
import ast
from contextlib import nullcontext
import sys
import subprocess
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CodeType,FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import microbatch_icp as batch


def original_function(path,owner,name):
    module=ast.parse(path.read_text(encoding="utf-8"))
    code=compile(module,str(path),"exec",dont_inherit=True)
    class_code=next(value for value in code.co_consts if isinstance(value,CodeType) and value.co_name==owner)
    method_code=next(value for value in class_code.co_consts if isinstance(value,CodeType) and value.co_name==name)
    return FunctionType(method_code,{})


class SourceContracts(unittest.TestCase):
    def classes(self):
        nearest=original_function(batch.ROOT/batch.SOURCE_FILES[1],"DeviceFlatGridICP","nearest_device")
        resident=original_function(batch.ROOT/batch.SOURCE_FILES[7],"ResidentICP","_match_resident")
        return SimpleNamespace(nearest_device=nearest),SimpleNamespace(_match_resident=resident)

    def test_only_declared_stream_permit_readonly_upload_seams_change(self):
        nn,resident=self.classes()
        nearest,body,edits=batch.derive_lane_methods(nn,resident)
        self.assertEqual(edits,{"stream":1,"permit":1,"source":1,"normals":1,"upload_counter":1})
        self.assertIn("stream",nearest.__code__.co_names)
        self.assertIn("batch_permit",nearest.__code__.co_names)
        self.assertNotIn("proof_authority",nearest.__code__.co_names)
        self.assertNotIn("null",nearest.__code__.co_names)
        self.assertIn("shared_original",body.__code__.co_names)
        self.assertIn("shared_normals",body.__code__.co_names)
        for name in ("_iteration_data","_transform","solve"):
            self.assertIn(name,body.__code__.co_names)
        result_keywords=next(value for value in body.__code__.co_consts
            if isinstance(value,tuple) and "correspondence_set" in value)
        self.assertIn("transformation",result_keywords)

    def test_loaded_body_mutation_rejected_even_same_function_owner(self):
        nn,_=self.classes();function=nn.nearest_device
        function.__code__=(lambda *args:None).__code__
        with self.assertRaisesRegex(batch.MicrobatchFailure,"Loaded"):
            batch.source_method(function,batch.ROOT/batch.SOURCE_FILES[1],"DeviceFlatGridICP")

    def test_numerical_modules_absent_from_fresh_import(self):
        check="from scripts.research import microbatch_icp; import sys; assert not ({'cupy','numpy','open3d'} & set(sys.modules))"
        child=subprocess.run([sys.executable,"-S","-c",check],cwd=batch.ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(child.returncode,0,child.stderr)

    def test_no_old_token_or_null_query_dispatch_remains_in_generated_lane(self):
        nn,resident=self.classes();nearest,_,_=batch.derive_lane_methods(nn,resident)
        self.assertNotIn("proof_authority",nearest.__code__.co_names)
        self.assertNotIn("null",nearest.__code__.co_names)


class RuntimeContracts(unittest.TestCase):
    def fake_runtime(self,root):
        backend=root/"backend.pyd";backend.write_bytes(b"original search")
        core=root/"numpy.pyd";core.write_bytes(b"original arithmetic")
        modules={"mock.backend":SimpleNamespace(__file__=str(backend)),
            "numpy._core._multiarray_umath":SimpleNamespace(__file__=str(core),__cpu_features__={"AVX":True})}
        pointcloud=SimpleNamespace(__module__="mock.backend.geometry")
        np=SimpleNamespace(__config__=SimpleNamespace(CONFIG={"compiler":"same","bits":64}))
        o3d=SimpleNamespace(geometry=SimpleNamespace(PointCloud=pointcloud))
        cp=SimpleNamespace(cuda=SimpleNamespace(nvrtc=SimpleNamespace(getVersion=lambda:(12,9))))
        return np,o3d,cp,modules,backend

    def test_binary_change_changes_exact_runtime_fingerprint(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,backend=self.fake_runtime(Path(directory))
            with patch.dict(sys.modules,modules):
                original=batch.numerical_runtime(np,o3d,cp)
                backend.write_bytes(b"different search")
                changed=batch.numerical_runtime(np,o3d,cp)
            self.assertNotEqual(original["open3d_backend_binary_sha256"],changed["open3d_backend_binary_sha256"])
            self.assertEqual(original["numpy_cpu_features"],{"AVX":True})
            self.assertEqual(original["nvrtc_version"],[12,9])

    def test_build_configuration_ignores_key_order_and_preserves_value_types(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,_=self.fake_runtime(Path(directory))
            with patch.dict(sys.modules,modules):
                original=batch.numerical_runtime(np,o3d,cp)
                np.__config__.CONFIG={"bits":64,"compiler":"same"}
                self.assertEqual(original,batch.numerical_runtime(np,o3d,cp))
                np.__config__.CONFIG["bits"]=64.0
                self.assertNotEqual(original,batch.numerical_runtime(np,o3d,cp))

    def test_unknown_loaded_backend_fails_closed(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,_=self.fake_runtime(Path(directory))
            modules["mock.backend"]=None
            with patch.dict(sys.modules,modules),self.assertRaisesRegex(batch.MicrobatchFailure,"binary layout"):
                batch.numerical_runtime(np,o3d,cp)

    def test_boundary_metadata_check_does_not_reread_large_binaries(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,_=self.fake_runtime(Path(directory))
            with patch.dict(sys.modules,modules),patch.object(Path,"read_bytes",side_effect=AssertionError("Unexpected binary read")):
                value=batch.numerical_runtime(np,o3d,cp,hash_binaries=False)
            self.assertEqual(set(value),{"numpy_build_configuration_sha256","numpy_cpu_features","nvrtc_version"})

    def test_loaded_binary_owner_replacement_rejected_with_same_filename(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,backend=self.fake_runtime(Path(directory))
            helper=object.__new__(batch.MicrobatchICP)
            helper.np,helper.o3d,helper.cp=np,o3d,cp
            helper.closed=False;helper.failure=None;helper.source_before={}
            helper.statistics={"source_checks_s":0.,"source_checks":0}
            helper.prototype_solve=SimpleNamespace(path=backend,metadata={});helper.solve_metadata={}
            helper.runtime={"solve_library_sha256":batch.hashlib.sha256(backend.read_bytes()).hexdigest()}
            with patch.dict(sys.modules,modules):
                helper.numeric_modules=batch.numerical_modules(o3d)
                helper.numeric_module_paths=tuple(str(Path(m.__file__).resolve()) for m in helper.numeric_modules)
                helper.numeric_runtime=batch.numerical_runtime(np,o3d,cp)
                sys.modules["mock.backend"]=SimpleNamespace(__file__=str(backend))
                with patch.object(batch,"file_pins",return_value={}),self.assertRaisesRegex(batch.MicrobatchFailure,"module ownership"):
                    helper._healthy()
            self.assertEqual(helper.statistics["source_checks"],1)

    def test_actual_open3d_opencv_and_omp_changes_stop_boundary_check(self):
        with TemporaryDirectory() as directory:
            np,o3d,cp,modules,backend=self.fake_runtime(Path(directory))
            values={"o3d":20,"cv2":20}
            o3d.utility=SimpleNamespace(get_max_threads=lambda:values["o3d"])
            cv2=SimpleNamespace(getNumThreads=lambda:values["cv2"])
            helper=object.__new__(batch.MicrobatchICP)
            helper.np,helper.o3d,helper.cp,helper.cv2=np,o3d,cp,cv2
            helper.closed=False;helper.failure=None;helper.source_before={}
            helper.statistics={"source_checks_s":0.,"source_checks":0}
            helper.prototype_solve=SimpleNamespace(path=backend,metadata={});helper.solve_metadata={}
            helper.resident=SimpleNamespace(STAGES=batch.STAGES,BLOCK=128,TERMS=30);helper.function_states=()
            with patch.dict(sys.modules,modules),patch.dict(batch.os.environ,{"OMP_NUM_THREADS":"8"}),patch.object(batch,"file_pins",return_value={}):
                helper.numeric_modules=batch.numerical_modules(o3d)
                helper.numeric_module_paths=tuple(str(Path(m.__file__).resolve()) for m in helper.numeric_modules)
                helper.numeric_runtime=batch.numerical_runtime(np,o3d,cp)
                helper.runtime={"solve_library_sha256":batch.hashlib.sha256(backend.read_bytes()).hexdigest(),
                    "thread_policy":batch.current_thread_policy(o3d,cv2)}
                helper._healthy()
                for key in ("o3d","cv2","omp"):
                    if key=="omp":batch.os.environ["OMP_NUM_THREADS"]="4"
                    else:values[key]=1
                    with self.assertRaisesRegex(batch.MicrobatchFailure,"thread policy"):
                        helper._healthy()
                    values["o3d"]=values["cv2"]=20;batch.os.environ["OMP_NUM_THREADS"]="8"


class SchedulerContracts(unittest.TestCase):
    def test_serial_consumption_preserves_seed_order(self):
        entered=[]
        jobs=[lambda i=i:(entered.append(i),i)[1] for i in range(4)]
        self.assertEqual(batch.run_ordered(jobs,"serial"),[0,1,2,3])
        self.assertEqual(entered,[0,1,2,3])

    def test_four_concurrent_lanes_are_live_but_return_in_seed_order(self):
        gate=threading.Barrier(4);entered=[];lock=threading.Lock()
        def job(index):
            with lock:entered.append(index)
            gate.wait(timeout=2)
            return index
        self.assertEqual(batch.run_ordered([lambda i=i:job(i) for i in range(4)],"concurrent"),[0,1,2,3])
        self.assertEqual(set(entered),set(range(4)))

    def test_failure_waits_for_other_lane_before_propagation(self):
        gate=threading.Barrier(2);finished=threading.Event();primary=OSError("lane fault")
        def bad():gate.wait(timeout=2);raise primary
        def good():gate.wait(timeout=2);finished.set();return 1
        with self.assertRaises(OSError) as error:batch.run_ordered([bad,good],"concurrent")
        self.assertIs(error.exception,primary)
        self.assertTrue(finished.is_set())

    def test_unsupported_schedule_or_lane_count_stops_before_jobs(self):
        called=[]
        for jobs,schedule in (([lambda:called.append(1)],"serial"),([lambda:called.append(1)]*2,"unknown")):
            with self.assertRaises(batch.MicrobatchFailure):batch.run_ordered(jobs,schedule)
        self.assertEqual(called,[])


class OwnershipContracts(unittest.TestCase):
    def helper(self):
        helper=object.__new__(batch.MicrobatchICP)
        helper.cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda device:nullcontext()))
        helper.device_id=0;helper.active=False;helper.failure=None
        helper.lanes=[SimpleNamespace(shared_original=object(),shared_normals=object())]
        helper.lease={"closed":False,"streams":[],"source_points":object(),"normals":object(),"items":{.12:object()}}
        return helper

    def test_release_synchronizes_before_dropping_shared_owners(self):
        helper=self.helper();lease=helper.lease;observed=[]
        stream=SimpleNamespace(device_id=0,synchronize=lambda:observed.append((lease["source_points"] is not None,lease["normals"] is not None)))
        helper.register_lease_stream(lease,stream)
        helper.release_pair(lease)
        self.assertEqual(observed,[(True,True)])
        self.assertTrue(lease["closed"]);self.assertIsNone(helper.lease)
        self.assertIsNone(helper.lanes[0].shared_original)

    def test_release_fault_attempts_every_stream_and_retains_buffers(self):
        helper=self.helper();lease=helper.lease;primary=OSError("sync");attempted=[]
        def bad():attempted.append(0);raise primary
        def good():attempted.append(1)
        lease["streams"]=[SimpleNamespace(synchronize=bad),SimpleNamespace(synchronize=good)]
        with self.assertRaises(OSError) as error:helper.release_pair(lease)
        self.assertIs(error.exception,primary);self.assertEqual(attempted,[0,1])
        self.assertIs(helper.lease,lease);self.assertFalse(lease["closed"])
        self.assertIsNotNone(lease["source_points"])

    def test_close_does_not_release_cache_when_internal_completion_fails(self):
        helper=self.helper();lease=helper.lease;primary=OSError("lane synchronization")
        helper.closed=False;helper.statistics={"cleanup_s":0.};helper.cleanup_failures=[];released=[]
        helper.shared=SimpleNamespace(close=lambda:released.append(True))
        helper._sync_lanes=lambda:(_ for _ in ()).throw(primary)
        with self.assertRaises(OSError) as error:helper.close()
        self.assertIs(error.exception,primary);self.assertEqual(released,[])
        self.assertFalse(helper.closed);self.assertIs(helper.lease,lease)
        self.assertIsNotNone(lease["source_points"])
        # Completion may later succeed for cleanup; failed math cannot retry.
        helper._sync_lanes=lambda:None
        helper.close()
        self.assertTrue(helper.closed);self.assertEqual(released,[True]);self.assertIs(helper.failure,primary)

    def test_close_does_not_release_cache_when_external_lease_stream_fails(self):
        helper=self.helper();lease=helper.lease;primary=OSError("external stream")
        helper.closed=False;helper.statistics={"cleanup_s":0.};helper.cleanup_failures=[];released=[]
        helper.shared=SimpleNamespace(close=lambda:released.append(True));helper._sync_lanes=lambda:None
        lease["streams"]=[SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(primary))]
        with self.assertRaises(OSError) as error:helper.close()
        self.assertIs(error.exception,primary);self.assertEqual(released,[])
        self.assertFalse(helper.closed);self.assertIs(helper.lease,lease)

    def test_lease_completion_failure_preserves_earlier_math_fault(self):
        helper=self.helper();lease=helper.lease
        primary=ValueError("original numeric fault");secondary=OSError("completion")
        helper.failure=primary
        lease["streams"]=[SimpleNamespace(synchronize=lambda:(_ for _ in ()).throw(secondary))]
        with self.assertRaises(ValueError) as error:helper.release_pair(lease)
        self.assertIs(error.exception,primary);self.assertIs(primary.__cause__,secondary)
        self.assertIs(helper.failure,primary);self.assertIsNotNone(lease["source_points"])

    def test_active_foreign_wrong_device_or_fifth_lane_cannot_take_ownership(self):
        helper=self.helper();lease=helper.lease
        with self.assertRaises(batch.MicrobatchFailure):helper.register_lease_stream(lease,SimpleNamespace(device_id=1))
        with self.assertRaises(batch.MicrobatchFailure):helper.release_pair(dict(lease))
        helper.active=True
        with self.assertRaises(batch.MicrobatchFailure):helper.release_pair(lease)
        helper.active=False;lease["streams"]=[object()]*4
        with self.assertRaises(batch.MicrobatchFailure):helper.register_lease_stream(lease,SimpleNamespace(device_id=0))

    def test_existing_registered_stream_is_idempotent_at_four_lane_cap(self):
        helper=self.helper();lease=helper.lease
        streams=[SimpleNamespace(device_id=0) for _ in range(4)]
        for stream in streams:helper.register_lease_stream(lease,stream)
        helper.register_lease_stream(lease,streams[0])
        self.assertEqual(len(lease["streams"]),4)

    def test_owned_scratch_includes_all_lanes_and_query_temporaries(self):
        two=batch.scratch_bound(9000,9000*24,10000*24,2*1024**2,2)
        four=batch.scratch_bound(9000,9000*24,10000*24,2*1024**2,4)
        self.assertEqual(four["combined_bytes"]-two["combined_bytes"],2*two["per_lane_bytes"])
        self.assertEqual(two["query_bytes"],136*9000+80)
        with self.assertRaises(batch.MicrobatchFailure):batch.scratch_bound(1000001,0,0,0,4)

    def test_unvalidated_timing_and_bad_limits_stop_before_imports(self):
        with patch.object(batch.importlib,"import_module") as imports:
            for kwargs in ({"audit":False},{"max_points":1000001},{"max_total_bytes":0}):
                with self.assertRaises(batch.MicrobatchFailure):batch.MicrobatchICP(lambda:None,**kwargs)
            imports.assert_not_called()

    def test_unregistered_nonnull_permit_is_rejected_before_cupy_import(self):
        protocol=SimpleNamespace(assert_registered_microbatch_permit=lambda token:batch.require(False,"unregistered permit"))
        with patch.object(batch.importlib,"import_module",return_value=protocol) as imports:
            with self.assertRaisesRegex(batch.MicrobatchFailure,"unregistered"):
                batch.MicrobatchICP(lambda:None,audit=False,timing_permit=object())
            imports.assert_called_once_with("scripts.research.gpu_icp_experiment_protocol")

    def test_original_fortran_seed_hash_uses_c_value_order_without_mutation(self):
        dtype=SimpleNamespace(str="<f8");requested=[]
        array=SimpleNamespace(dtype=dtype,shape=(4,4),nbytes=128,
            flags=SimpleNamespace(c_contiguous=False),tobytes=lambda order:(requested.append(order),b"original FP64 values")[1])
        record=batch.matrix_record(SimpleNamespace(float64=dtype),array,contiguous=False)
        self.assertEqual(requested,["C"])
        self.assertEqual(record["nbytes"],128)
        self.assertFalse(array.flags.c_contiguous)
        with self.assertRaises(batch.MicrobatchFailure):batch.matrix_record(SimpleNamespace(float64=dtype),array)


if __name__=="__main__":unittest.main()
