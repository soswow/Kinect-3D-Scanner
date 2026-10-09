"""Check all registered depth views against every other measured view."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import open3d as o3d
from scanner_server.depth_graph import scene_visibility
from scripts.benchmarks.benchmark_offline_geometry import load_views


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    args = parser.parse_args()
    o3d.utility.set_max_threads(4)
    result=json.loads(args.report.read_text())["results"][0]
    views=load_views(result["path"],result.get("registration_far_m"))
    poses={int(i):np.array(p) for i,p in result["components"][0]["poses"].items()}
    report=scene_visibility(views,poses)
    args.report.with_name(args.report.stem+"-visibility.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
