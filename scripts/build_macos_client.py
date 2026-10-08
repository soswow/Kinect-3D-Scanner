"""Build and check a standalone Mac client with this environment's dependencies."""

from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != "darwin":
        raise SystemExit("Build the client on macOS using its configured Python environment.")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/build_macos_icon.py")],
                   check=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    str(root / "packaging/macos-client.spec"),
                    "--distpath", str(root / "dist"), "--workpath", str(root / "build/macos-client")],
                   cwd=root, check=True)
    bundle = root / "dist/Kinect 3D Scanner.app"
    subprocess.run([str(bundle / "Contents/MacOS/Kinect 3D Scanner"), "--check"],
                   cwd="/", check=True, timeout=60)
    print(f"Client ready: {bundle}\nCopy this entire .app to Applications or open it in Finder.")


if __name__ == "__main__":
    main()
