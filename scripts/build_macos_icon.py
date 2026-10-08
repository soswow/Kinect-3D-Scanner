"""Compile the Tahoe icon and render PNG/ICNS fallbacks from Icon Composer."""

from pathlib import Path
import subprocess
import sys
import shutil
import tempfile


def main():
    if sys.platform != "darwin":
        raise SystemExit("Generate the Mac icon using Xcode 26 or later on macOS.")
    root = Path(__file__).resolve().parents[1]
    icons = root / "assets/icons"
    document = icons / "kinect-scanner-client.icon"
    developer = Path(subprocess.check_output(["/usr/bin/xcode-select", "-p"], text=True).strip())
    renderer = developer.parent / "Applications/Icon Composer.app/Contents/Executables/ictool"
    if not renderer.is_file():
        raise SystemExit("Select a full Xcode 26+ installation with Icon Composer first.")
    with tempfile.TemporaryDirectory(prefix="kinect-icon-") as directory:
        output = Path(directory)
        subprocess.run(["/usr/bin/xcrun", "actool", str(document), "--compile", str(output),
                        "--platform", "macosx", "--minimum-deployment-target", "15.0",
                        "--app-icon", document.stem, "--include-all-app-icons",
                        "--target-device", "mac", "--output-partial-info-plist",
                        str(output / "icon-info.plist")], check=True)
        shutil.copy2(output / "Assets.car", icons / "Assets.car")
    source = icons / "kinect-scanner-client.png"
    subprocess.run([str(renderer), str(document), "--export-image", "--output-file", str(source),
                    "--platform", "macOS", "--rendition", "Default",
                    "--width", "1024", "--height", "1024", "--scale", "1"], check=True)
    iconset = icons / "kinect-scanner-client.iconset"
    iconset.mkdir(exist_ok=True)
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = points * scale
            suffix = "@2x" if scale == 2 else ""
            output = iconset / f"icon_{points}x{points}{suffix}.png"
            subprocess.run(["/usr/bin/sips", "--resampleHeightWidth",
                            str(pixels), str(pixels), str(source), "--out", str(output)],
                           check=True, stdout=subprocess.DEVNULL)
    output = icons / "kinect-scanner-client.icns"
    subprocess.run(["/usr/bin/iconutil", "--convert", "icns",
                    str(iconset), "--output", str(output)], check=True)
    print(f"Client icon ready: {output}")


if __name__ == "__main__":
    main()
