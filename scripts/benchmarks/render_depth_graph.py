"""Render measured depth geometry and camera paths without using RGB."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scripts.benchmarks.benchmark_offline_geometry import load_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--component-index", type=int, default=0)
    args = parser.parse_args()
    o3d.utility.set_max_threads(4)
    report = json.loads(args.report.read_text())
    result = report["results"][args.session_index]
    component = result["components"][args.component_index]
    views = load_views(result["path"],result.get("registration_far_m"))
    poses = {int(i): np.asarray(p) for i,p in component["poses"].items()}
    points, colors, cameras = [], [], []
    palette = plt.get_cmap("turbo")
    for i in sorted(poses):
        pose = poses[i]
        points.append(np.asarray(views[i].cloud.voxel_down_sample(.06).points) @ pose[:3,:3].T + pose[:3,3])
        colors.extend([palette(i / max(1, len(views)-1))] * len(points[-1]))
        cameras.append(pose[:3,3])
    points, colors, cameras = np.vstack(points), np.asarray(colors), np.asarray(cameras)
    stride = max(1, len(points)//100000)
    points, colors = points[::stride], colors[::stride]
    fig = plt.figure(figsize=(14,10), layout="constrained")
    for panel, (axes, title) in enumerate((((0,2),"Plan (X/Z)"), ((0,1),"Elevation (X/Y)"), ((2,1),"Elevation (Z/Y)"))):
        ax = fig.add_subplot(2,2,panel+1)
        ax.scatter(points[:,axes[0]], points[:,axes[1]], c=colors, s=.5, alpha=.2, rasterized=True)
        ax.plot(cameras[:,axes[0]], cameras[:,axes[1]], color="black", linewidth=.7, alpha=.7)
        ax.scatter(cameras[:,axes[0]], cameras[:,axes[1]], c=np.array([palette(i/max(1,len(views)-1)) for i in sorted(poses)]), s=16, edgecolors="black", linewidths=.3)
        ax.set(title=title, xlabel="metres", ylabel="metres", aspect="equal")
        ax.grid(alpha=.15)
    ax = fig.add_subplot(2,2,4,projection="3d")
    ax.scatter(points[::3,0],points[::3,2],-points[::3,1], c=colors[::3], s=.5, alpha=.3)
    ax.plot(cameras[:,0], cameras[:,2], -cameras[:,1], color="black", linewidth=.7)
    ax.set(xlabel="X (m)",ylabel="Z (m)",zlabel="-Y (m)",title="Depth surfaces and camera path")
    ax.set_box_aspect(np.maximum(.1, np.ptp(points[:,[0,2,1]],axis=0)))
    status = "validated" if component["validation"]["accepted"] else "rejected"
    fig.suptitle(f"Depth-only registration: {len(poses)}/{len(views)} captures in selected {status} component\nColor indicates capture order; black line joins camera positions in time", fontsize=15)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)


if __name__ == "__main__":
    main()
