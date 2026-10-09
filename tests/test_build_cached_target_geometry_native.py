"""Mocked Windows builder contracts; never execute compiler or native code."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.research import build_cached_target_geometry_native as build


def fixture(folder):
    root=Path(folder)
    for name in ("cl.exe","VsDevCmd.bat","cmd.exe","eigen.tar.gz","nano.tar.gz"):
        (root/name).write_bytes(name.encode())
    (root/"eigen").mkdir();(root/"nano").mkdir()
    return SimpleNamespace(output_dir=root/"fresh",compiler=root/"cl.exe",vsdevcmd=root/"VsDevCmd.bat",cmd=root/"cmd.exe",
        eigen_root=root/"eigen",eigen_archive=root/"eigen.tar.gz",nanoflann_root=root/"nano",nanoflann_archive=root/"nano.tar.gz",run_allocated=True)


def fake_recipe(args):
    return {"kind":build.helper.KIND,"stage":"recipe only","source_contract":build.helper.source_contract(),
        "dependencies":{"eigen":{"files":["COPYING.MPL2"],"archive_sha256":"e","headers_sha256":"eh"},
                        "nanoflann":{"files":["COPYING"],"archive_sha256":"n","headers_sha256":"nh"}},
        "compiler":{"path":str(args.compiler),"sha256":build.sha(args.compiler)},"output":str(args.output_dir/"cached-target.dll"),
        "command":[str(args.compiler),"/nologo","/O2","/EHsc","/LD","/std:c++17","/fp:strict","/DEIGEN_DONT_PARALLELIZE",
            str(build.ROOT/"scripts/research/cached_target_geometry_native.cpp"),"/link","/OUT:"+str(args.output_dir/"cached-target.dll")]}


class Contracts(unittest.TestCase):
    def test_environment_never_forwards_secrets_or_cl_injection(self):
        raw="PATH=tools\nINCLUDE=headers\nLIB=libraries\nSYSTEMROOT=windows\nTOKEN=secret\nCL=/fp:fast\n_LINK_=bad\nLIBPATH=more\nLIBPATH=more\n"
        value=build.filtered_environment(raw)
        self.assertEqual(value["LIBPATH"],"more")
        self.assertTrue(all(k not in value for k in ("TOKEN","CL","_LINK_")))
        with self.assertRaises(build.BuildFailure):build.filtered_environment(raw+"PATH=foreign\n")
        with self.assertRaises(build.BuildFailure):build.filtered_environment("PATH=tools\n")

    def test_batch_paths_allow_spaces_but_refuse_shell_expansion(self):
        self.assertIn("with spaces",build.command_path("with spaces/VsDevCmd.bat"))
        for char in ('&','|','%','!','^','\n','"','\u00e9'):
            with self.assertRaises(build.BuildFailure):build.command_path("bad"+char+"path")

    def test_developer_capture_is_not_a_compiler_or_file_operation_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder);args.output_dir.mkdir()
            calls=[]
            def runner(command,**kwargs):
                calls.append((command,kwargs))
                return SimpleNamespace(returncode=0,stdout="PATH=tools\nINCLUDE=headers\u00e9\nLIB=libraries\nSYSTEMROOT=windows\nTOKEN=never-saved\n".encode("utf-16le"))
            env,receipt=build.capture_environment(args.vsdevcmd,args.output_dir,args.cmd,runner=runner)
            self.assertEqual(calls[0][0][1:5],["/d","/u","/s","/c"])
            self.assertEqual(calls[0][0][5:],["call","capture-developer-environment.cmd"])
            self.assertEqual(calls[0][1]["cwd"],str(args.output_dir))
            self.assertNotIn('\\"',subprocess.list2cmdline(calls[0][0]))
            self.assertEqual(calls[0][1]["stdout"],subprocess.PIPE)
            self.assertEqual(env["INCLUDE"],"headers\u00e9")
            self.assertNotIn("TOKEN",env)
            text=(args.output_dir/"capture-developer-environment.cmd").read_text()
            self.assertNotIn("never-saved",text)
            self.assertFalse(receipt["capture_stdout_saved"])
            self.assertNotIn("&&",text)

    def test_failed_environment_does_not_expose_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder);args.output_dir.mkdir()
            def runner(*a,**k):return SimpleNamespace(returncode=9,stdout=b"secret credential")
            with self.assertRaises(build.BuildFailure) as caught:
                build.capture_environment(args.vsdevcmd,args.output_dir,args.cmd,runner=runner)
            self.assertNotIn("secret credential",str(caught.exception))

    def test_preflight_nonwindows_and_existing_output_refuse_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder)
            with self.assertRaises(build.BuildFailure):build.preflight(args,platform="posix")
            args.output_dir.mkdir()
            with self.assertRaises(build.BuildFailure):build.preflight(args,platform="nt")

    def test_fp_and_tie_policy_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder);recipe=fake_recipe(args)
            for extra in ("/fp:fast","/DNANOFLANN_FIRST_MATCH"):
                changed=copy.deepcopy(recipe);changed["command"].append(extra)
                with patch.object(build.helper,"build_recipe",return_value=changed):
                    with self.assertRaises(build.BuildFailure):build.preflight(args,platform="nt")

    def execute_fixture(self,folder,code=0,mutate=None,create=True):
        args=fixture(folder);recipe=fake_recipe(args)
        records=[]
        def runner(command,**kwargs):
            records.append((command,kwargs))
            if kwargs["stdout"] == subprocess.PIPE:
                return SimpleNamespace(returncode=0,stdout="PATH=tools\nINCLUDE=headers\nLIB=libraries\nSYSTEMROOT=windows\nCL=bad\nSECRET=bad\n".encode("utf-16le"))
            kwargs["stdout"].write(b"actual compiler diagnostic\n")
            if create:(args.output_dir/"cached-target.dll").write_bytes(b"fake DLL for mocked contract only")
            if mutate:mutate(args)
            return SimpleNamespace(returncode=code)
        def verify(archive,*rest):
            return recipe["dependencies"]["eigen" if archive == args.eigen_archive else "nanoflann"]
        with patch.object(build.helper,"build_recipe",return_value=recipe),patch.object(build.helper,"_verify_archive",side_effect=verify):
            if code != 0 or mutate or not create:
                with self.assertRaises(build.BuildFailure):build.build(args,runner=runner,platform="nt")
            else:build.build(args,runner=runner,platform="nt")
        return args,records,json.loads((args.output_dir/"cached-target.build.json").read_text())

    def test_success_direct_compiler_argv_and_compatible_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            args,calls,receipt=self.execute_fixture(folder)
            self.assertEqual(receipt["stage"],"built");self.assertEqual(receipt["actual_exit_code"],0)
            self.assertTrue(receipt["root_waited"])
            self.assertEqual(calls[1][0][0],str(args.compiler))
            self.assertNotIn("shell",calls[1][1])
            self.assertNotIn("CL",calls[1][1]["env"]);self.assertNotIn("SECRET",calls[1][1]["env"])
            self.assertEqual(receipt["library_sha256"],build.sha(args.output_dir/"cached-target.dll"))
            self.assertEqual(receipt["dependencies"],receipt["dependencies_after"])
            self.assertFalse(receipt["environment_values_saved"])

    def test_compiler_actual_nonzero_and_log_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            args,_,receipt=self.execute_fixture(folder,code=7)
            self.assertEqual(receipt["stage"],"failed");self.assertEqual(receipt["actual_exit_code"],7)
            self.assertTrue(receipt["root_waited"])
            self.assertEqual((args.output_dir/"compiler.log").read_bytes(),b"actual compiler diagnostic\n")

    def test_exit_zero_without_dll_is_not_built(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,receipt=self.execute_fixture(folder,create=False)
            self.assertEqual(receipt["actual_exit_code"],0);self.assertEqual(receipt["stage"],"failed")

    def test_compiler_mutation_demotes_build_without_changing_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            _,_,receipt=self.execute_fixture(folder,mutate=lambda a:a.compiler.write_bytes(b"replaced compiler"))
            self.assertEqual(receipt["actual_exit_code"],0);self.assertEqual(receipt["stage"],"failed")
            self.assertTrue(receipt["cleanup_failures"])


if __name__ == "__main__":unittest.main()
