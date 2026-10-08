# Build on macOS with the Python environment containing the Kinect bindings.
import importlib.util
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

if sys.platform != "darwin":
    raise SystemExit("The client bundle must be built on macOS.")
root = Path(SPECPATH).parent
if importlib.util.find_spec("freenect") is None:
    raise SystemExit("Install working freenect bindings in the build environment first.")

datas = collect_data_files("open3d", includes=["resources/*"])
datas += [(str(root / "calibration/default.json"), "calibration"),
          (str(root / "calibration/raw-to-mm.bin"), "calibration"),
          (str(root / "kinect_scanner/gui/assets"), "kinect_scanner/gui/assets")]
hiddenimports = ["freenect", "open3d.pybind"]
if importlib.util.find_spec("_kinect_native") is not None:
    hiddenimports.append("_kinect_native")

a = Analysis(
    [str(root / "packaging/client_entry.py")], pathex=[str(root)],
    # ML operation plugins require unrelated Torch/TensorFlow runtimes.
    binaries=[item for item in collect_dynamic_libs("open3d")
              if Path(item[0]).name not in ("open3d_tf_ops.dylib", "open3d_torch_ops.dylib")], datas=datas,
    hiddenimports=hiddenimports,
    excludes=["scanner_server", "torch", "tensorflow", "IPython", "jupyter",
              "matplotlib", "open3d.ml.torch", "open3d.ml.tf", "open3d.visualization.tensorboard_plugin"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Kinect 3D Scanner",
          console=False, argv_emulation=False)
collection = COLLECT(exe, a.binaries, a.datas, name="Kinect 3D Scanner")
app = BUNDLE(collection, name="Kinect 3D Scanner.app",
             bundle_identifier="org.kinect3dscanner.client",
             info_plist={"CFBundleShortVersionString": "0.1.0",
                         "CFBundleVersion": "1", "NSHighResolutionCapable": True})
