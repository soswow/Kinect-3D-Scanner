"""Build and check a standalone Mac client with this environment's dependencies."""

from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != "darwin":
        raise SystemExit("Build the client on macOS using its configured Python environment.")
    root = Path(__file__).resolve().parents[1]
    from install_macos_client import require_stopped

    # PyInstaller replaces dist bundles too; running a development build must
    # not lose its embedded Python archive or helper executable during rebuild.
    require_stopped(root / "dist/Kinect 3D Scanner.app")
    subprocess.run([sys.executable, str(root / "scripts/build_macos_icon.py")],
                   check=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    str(root / "packaging/macos-client.spec"),
                    "--distpath", str(root / "dist"), "--workpath", str(root / "build/macos-client")],
                   cwd=root, check=True)
    bundle = root / "dist/Kinect 3D Scanner.app"
    subprocess.run([str(bundle / "Contents/MacOS/Kinect 3D Scanner"), "--check"],
                   cwd="/", check=True, timeout=60)
    print(f"Client ready: {bundle}\nQuit the client and viewers, then install with:\n"
          f"{sys.executable} scripts/install_macos_client.py --source '{bundle}'")


if __name__ == "__main__":
    main()
