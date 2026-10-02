"""Surface error and completeness in a fixed metric frame, without fitting scale."""

import numpy as np
import open3d as o3d


def sample_surface(mesh, count=20000, seed=11):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(faces) or not np.all(np.isfinite(vertices)):
        raise ValueError("Surface needs finite vertices and triangles")
    triangle = vertices[faces]
    area = np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
        axis=1,
    )
    if area.sum() <= 0:
        raise ValueError("Surface has zero area")
    rng = np.random.default_rng(seed)
    selected = triangle[rng.choice(len(faces), count, p=area / area.sum())]
    a = np.sqrt(rng.random((count, 1)))
    b = rng.random((count, 1))
    return (
        selected[:, 0] * (1 - a) + selected[:, 1] * a * (1 - b) + selected[:, 2] * a * b
    )


def surface_metrics(mesh, reference, threshold_m=0.01, samples=20000):
    if not 0.0001 <= threshold_m <= 0.2 or not 100 <= samples <= 1000000:
        raise ValueError("Invalid surface scoring budget or metric threshold")

    def distance(points, target):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target))
        return scene.compute_distance(
            o3d.core.Tensor(points.astype(np.float32))
        ).numpy()

    estimated = sample_surface(mesh, samples)
    measured = sample_surface(reference, samples, seed=12)
    error = distance(estimated, reference)
    missing = distance(measured, mesh)
    precision, completeness = (
        float(np.mean(error <= threshold_m)),
        float(np.mean(missing <= threshold_m)),
    )
    return {
        "surface_rmse_m": float(np.sqrt(np.mean(error**2))),
        "surface_p95_m": float(np.percentile(error, 95)),
        "precision": precision,
        "completeness": completeness,
        "fscore": 2 * precision * completeness / max(precision + completeness, 1e-12),
        "threshold_m": threshold_m,
        "samples_per_surface": samples,
        "alignment": "fixed input coordinate frame; no scale or trajectory fitting",
    }
