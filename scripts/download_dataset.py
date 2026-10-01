"""Download official public RGB-D examples into the ignored datasets directory."""

import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["redwood", "tum-xyz", "tum-desk"])
    args = parser.parse_args()
    target = ROOT / "datasets"
    target.mkdir(exist_ok=True)
    if args.dataset == "redwood":
        import open3d as o3d

        sample = o3d.data.SampleRedwoodRGBDImages(data_root=str(target / "redwood"))
        print(f"{len(sample.color_paths)} Redwood frames available")
        return
    name = args.dataset.removeprefix("tum-")
    archive = target / f"tum-{name}.tgz"
    if not archive.exists():
        url = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_{name}.tgz"
        partial = archive.with_suffix(".tgz.part")
        print(f"Downloading TUM freiburg1_{name} (~350–450 MB)", flush=True)
        urllib.request.urlretrieve(url, partial)
        # A failed download never leaves a file that looks complete.
        partial.replace(archive)
    with tarfile.open(archive) as source:
        source.extractall(target, filter="data")
    print(target / f"rgbd_dataset_freiburg1_{name}")


if __name__ == "__main__":
    main()
