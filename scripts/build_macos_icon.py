"""Generate the Mac iconset and ICNS from the client PNG master."""

from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != "darwin":
        raise SystemExit("Generate the Mac icon using sips and iconutil on macOS.")
    root = Path(__file__).resolve().parents[1]
    icons = root / "assets/icons"
    source = icons / "kinect-scanner-client.png"
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
