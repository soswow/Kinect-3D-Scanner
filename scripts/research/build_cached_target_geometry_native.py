"""Explicit Windows research build of the pinned cached-target lookup helper.

No download, compiler execution or DLL load occurs at import. --run-allocated
builds in a fresh directory, writes a NativeLibrary-compatible receipt and keeps
failed compiler logs. A successful build is not a nearest-neighbour parity or
performance qualification; run the separate allocated parity benchmark.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research import cached_target_geometry_native as helper

SOURCES = ("scripts/research/build_cached_target_geometry_native.py",
           "tests/test_build_cached_target_geometry_native.py")
# Only compiler/Windows paths are retained in the child environment. CL/LINK
# injection variables and unrelated credentials are neither forwarded nor saved.
ENV_NAMES = ("PATH","INCLUDE","LIB","LIBPATH","SYSTEMROOT","WINDIR","TEMP","TMP",
             "COMSPEC","PATHEXT","PROGRAMFILES","PROGRAMFILES(X86)","PROGRAMW6432",
             "VCTOOLSINSTALLDIR","VCTOOLSVERSION","VCINSTALLDIR","VSINSTALLDIR",
             "WINDOWSSDKDIR","WINDOWSSDKVERSION","UNIVERSALCRTSDKDIR","UCRTVERSION")


class BuildFailure(RuntimeError):
    pass


def require(value,message):
    if not value: raise BuildFailure(message)


def sha(path):
    return helper.file_hash(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_contract():
    return {"kind":"research-cached-target-msvc-builder-v1",
        "artifacts_sha256":{name:sha(ROOT/name) for name in SOURCES},
        "helper_source_contract":helper.source_contract(),
        "environment_policy":"allowlisted build/Windows paths only; no arbitrary environment values printed or saved",
        "capture_encoding_policy":"ASCII batch-call paths; cmd /u internal SET output UTF-16LE; direct compiler Unicode argv/environment",
        "compiler_policy":"direct cl.exe argv, /fp:strict, C++17, original pinned leaf/tie source; fresh output"}


def command_path(path):
    """A quoted batch call accepts spaces, never shell expansion/operators."""
    value=str(Path(path).resolve())
    require(value.isascii() and not any(c in value for c in '\r\n\x00"%!?&|<>^'),
            "Batch-call paths must be ASCII without expansion/operator characters")
    return value


def filtered_environment(text):
    result={}
    for line in text.splitlines():
        if "=" not in line: continue
        key,value=line.split("=",1)
        key=key.upper()
        if key in ENV_NAMES:
            # `set LIB` also lists LIBPATH. Later explicit prefixes may repeat
            # that same value; contradictory values are never accepted.
            require((key not in result or result[key] == value) and "\x00" not in value,
                    "Contradictory/invalid compiler environment variable")
            result[key]=value
    require(all(result.get(key) for key in ("PATH","INCLUDE","LIB","SYSTEMROOT")),
            "Developer environment lacks required compiler/Windows paths")
    return result


def capture_environment(vsdevcmd,output_dir,cmd,*,runner=subprocess.run):
    """The only shell step calls VS's batch file; it performs no file mutation."""
    vs=command_path(vsdevcmd)
    script=Path(output_dir)/"capture-developer-environment.cmd"
    command_path(script)
    # stdout goes to an internal pipe, is filtered and discarded. No set-all
    # command is used; even similarly prefixed variables never reach cl.exe.
    lines=['@echo off',f'call "{vs}" -no_logo -arch=x64 -host_arch=x64 >nul',
           'if errorlevel 1 exit /b 1']+["set "+key for key in ENV_NAMES]+['exit /b 0']
    with script.open("x",encoding="ascii",newline="\r\n") as stream:
        stream.write("\n".join(lines)+"\n")
    # The owned basename contains no spaces or expansion characters. Passing
    # a quoted absolute command as a list argument makes list2cmdline insert
    # backslash escapes that cmd.exe does not interpret as quote escapes.
    completed=runner([str(cmd),"/d","/u","/s","/c","call",script.name],cwd=str(output_dir),
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=False)
    require(type(completed.returncode) is int,"Developer environment process has no actual exit")
    require(completed.returncode == 0,"Developer environment process failed (output intentionally not saved)")
    raw=completed.stdout
    require(type(raw) is bytes and len(raw) <= 4*1024**2,"Bounded private compiler environment required")
    # cmd /u gives its internal SET commands UTF-16LE output independently of
    # console OEM/ANSI pages. The ASCII script safely retains Unicode values
    # for the subsequent direct Unicode compiler argv/environment.
    text=raw.decode("utf-16le",errors="strict")
    return filtered_environment(text),{"actual_exit_code":completed.returncode,
        "script_sha256":sha(script),"capture_stdout_saved":False,"capture_stderr_saved":False}


def preflight(args,*,platform=os.name):
    require(platform == "nt" and args.run_allocated,"Explicit exclusive Windows build allocation required")
    args.output_dir=Path(args.output_dir).resolve()
    require(not args.output_dir.exists(),"Preserve existing builds; output directory must be fresh")
    for name in ("eigen_root","eigen_archive","nanoflann_root","nanoflann_archive","compiler","vsdevcmd","cmd"):
        path=Path(getattr(args,name)).resolve(strict=True)
        setattr(args,name,path)
    require(args.compiler.is_file() and args.compiler.name.lower() == "cl.exe"
        and args.vsdevcmd.is_file() and args.vsdevcmd.name.lower() == "vsdevcmd.bat"
        and args.cmd.is_file() and args.cmd.name.lower() == "cmd.exe", "Explicit cl.exe, VsDevCmd.bat and cmd.exe required")
    command_path(args.vsdevcmd);command_path(args.output_dir)
    # This uses the held verifier: every selected staged header/license must
    # match its pinned archive byte-for-byte, not merely a caller's label.
    recipe=helper.build_recipe(eigen_root=args.eigen_root,eigen_archive=args.eigen_archive,
        nanoflann_root=args.nanoflann_root,nanoflann_archive=args.nanoflann_archive,
        compiler=args.compiler,output=args.output_dir/"cached-target.dll")
    require([v for v in recipe["command"] if v.lower().startswith("/fp:")] == ["/fp:strict"]
        and "/std:c++17" in recipe["command"] and "/DEIGEN_DONT_PARALLELIZE" in recipe["command"]
        and not any("nanoflann_first_match" in v.lower() or "/fp:fast" in v.lower() for v in recipe["command"]),
        "Require original floating/tie compiler policy")
    return recipe


def build(args,*,runner=subprocess.run,platform=os.name):
    started=time.perf_counter()
    recipe=preflight(args,platform=platform)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    output=args.output_dir/"cached-target.dll"
    receipt_path=output.with_suffix(".build.json")
    log_path=args.output_dir/"compiler.log"
    initial=source_contract()
    fixed={str(path):sha(path) for path in (args.compiler,args.vsdevcmd,args.cmd,Path(sys.executable),
        args.eigen_archive,args.nanoflann_archive)}
    report=dict(recipe,stage="failed",actual_exit_code=None,root_waited=False,started_utc=utc(),
        builder_source=initial,fixed_files=fixed,failure=None,cleanup_failures=[],
        developer_environment=str(args.vsdevcmd),environment_values_saved=False,
        qualification="none; build only, fresh original-API parity and independent timing still required")
    primary=None
    try:
        environment,env_receipt=capture_environment(args.vsdevcmd,args.output_dir,args.cmd,runner=runner)
        report["environment_capture"]=env_receipt
        report["child_environment_names"]=sorted(environment)
        begin=time.perf_counter()
        with log_path.open("xb") as log:
            completed=runner(recipe["command"],cwd=str(args.output_dir),env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=False)
        report.update(actual_exit_code=completed.returncode,root_waited=True,compiler_wall_s=time.perf_counter()-begin)
        require(type(completed.returncode) is int and completed.returncode == 0,"Compiler failed; retained actual exit and compiler.log")
        require(output.is_file() and output.stat().st_size > 0,"Compiler exited zero without its DLL")
        report["library_sha256"]=sha(output)
        report["stage"]="built"
    except BaseException as error:
        primary=error
        report["failure"]={"type":type(error).__name__,"message":str(error)}
    finally:
        try:
            report["builder_source_after"]=source_contract()
            report["fixed_files_after"]={path:sha(path) for path in fixed}
            # Re-read each archive AND selected staged header/license after cl
            # completes; recorded provenance cannot be swapped during compile.
            dependencies={"eigen":helper._verify_archive(args.eigen_archive,args.eigen_root,helper.EIGEN_ARCHIVE_SHA256,
                lambda p:p.parts[0] == "Eigen" or str(p) == "COPYING.MPL2"),
                "nanoflann":helper._verify_archive(args.nanoflann_archive,args.nanoflann_root,helper.NANOFLANN_ARCHIVE_SHA256,
                lambda p:str(p) in ("include/nanoflann.hpp","COPYING"))}
            report["dependencies_after"]=dependencies
            require(report["builder_source_after"] == initial and report["fixed_files_after"] == fixed
                and dependencies == recipe["dependencies"],"Build source/compiler/dependency/license bytes changed")
            if log_path.exists():report["compiler_log"]={"path":str(log_path),"sha256":sha(log_path)}
        except BaseException as secondary:
            report["cleanup_failures"].append({"type":type(secondary).__name__,"message":str(secondary)})
            primary=primary or secondary
        if primary is not None:report["stage"]="failed"
        report.update(ended_utc=utc(),whole_build_wall_s=time.perf_counter()-started)
        try:
            with receipt_path.open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
        except BaseException as secondary:
            if primary is not None:raise primary from secondary
            raise
    if primary is not None:raise primary
    return report


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("eigen-root","eigen-archive","nanoflann-root","nanoflann-archive","compiler","vsdevcmd","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    default=Path(os.environ.get("SystemRoot","C:/Windows"))/"System32/cmd.exe"
    p.add_argument("--cmd",type=Path,default=default)
    p.add_argument("--run-allocated",action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    build(parse(argv))


if __name__ == "__main__":main()
