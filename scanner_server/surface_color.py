"""Bounded low-frequency photo matching in physical surface coordinates.

Only a smoothly interpolated gain modifies the selected original photograph.
Depth visibility, photo labels, geometry, UVs and high-frequency photo sampling
are unchanged. One field is shared by all atlas pages and fallback photos.
"""

from dataclasses import dataclass
import copy
import time

import numpy as np


@dataclass
class PhotoColorField:
    vertices: np.ndarray
    triangles: np.ndarray
    log_gains: np.ndarray  # vertex, global photo, RGB
    face_indices: np.ndarray
    view_indices: np.ndarray

    def subset(self, faces, views):
        """Share the fitted arrays; remap page faces and conservatively culled views."""
        return PhotoColorField(
            self.vertices, self.triangles, self.log_gains,
            self.face_indices[np.asarray(faces)], self.view_indices[np.asarray(views, int)],
        )

    def evaluate(self, positions, face_ids, view):
        from .texturing import _interpolate_face_corrections

        corners = self.triangles[self.face_indices[face_ids]]
        return _interpolate_face_corrections(
            positions, np.arange(len(face_ids)), self.vertices[corners],
            self.log_gains[corners, self.view_indices[view]],
        )


def _surface_kernel(mesh):
    """Short, normal-aware physical edges; never connect disconnected sheets."""
    from scipy.sparse import coo_matrix, diags

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    faces = np.asarray(mesh.triangles)
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    a, b = edges.T
    length = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    keep = (length < .035) & (np.abs(np.einsum('ij,ij->i', normals[a], normals[b])) > .8)
    a, b, length = a[keep], b[keep], length[keep]
    weight = np.exp(-length**2 / .02**2).astype(np.float32)
    n = len(vertices)
    adjacent = coo_matrix((np.r_[weight, weight, np.full(n, 3, np.float32)],
                          (np.r_[a, b, np.arange(n)], np.r_[b, a, np.arange(n)])), shape=(n, n)).tocsr()
    return diags(1 / np.asarray(adjacent.sum(axis=1)).ravel()) @ adjacent


def _normal_axes(vertices, faces):
    """Area-weighted incident normal axes survive contradictory face winding."""
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    covariance = cross[:, :, None] * cross[:, None, :] / np.maximum(np.linalg.norm(cross, axis=1)[:, None, None], 1e-12)
    incident = np.zeros((len(vertices), 3, 3))
    mean = np.zeros((len(vertices), 3))
    for corner in range(3):
        np.add.at(incident, faces[:, corner], covariance)
        np.add.at(mean, faces[:, corner], cross)
    eigenvalues, axes = np.linalg.eigh(incident)
    result = axes[:, :, -1]
    # At a genuine corner, several normal axes can be equally strong. Use the
    # consistently oriented bisector there; choosing one perpendicular axis
    # would falsely reject all front-facing observations of a coarse box.
    length = np.linalg.norm(mean, axis=1)
    ambiguous = (eigenvalues[:, -1] < 2 * eigenvalues[:, -2]) & (length > 1e-12)
    result[ambiguous] = mean[ambiguous] / length[ambiguous, None]
    return result


def _consensus(colors, weights):
    """Capped geometric weights and robust residuals resist isolated highlights."""
    output = np.empty((len(colors), 3), np.float32)
    for start in range(0, len(colors), 4096):
        values = colors[start:start + 4096]
        weight = weights[start:start + 4096]
        weight = np.minimum(weight / np.maximum(weight.max(axis=1, keepdims=True), 1e-8), .5)
        current = weight
        for _ in range(5):
            target = np.einsum('nvc,nv->nc', values, current) / np.maximum(current.sum(axis=1)[:, None], 1e-8)
            residual = np.max(np.abs(values - target[:, None]), axis=2)
            current = weight * np.minimum(1, .035 / np.maximum(residual, 1e-8))
        output[start:start + 4096] = target
    return output


def _bounded_log_gains(target, source):
    # Near-black channels carry little color information. In particular, raising
    # the blue channel of red wood magnifies sensor noise. Shrink those RGB gains
    # towards the luminance gain instead of excluding all dark-colored surfaces.
    luma = np.array([.2126, .7152, .0722], np.float32)
    scalar = np.log(np.maximum(target @ luma, .02)[:, None] / np.maximum(source @ luma, .02))
    rgb = np.log(np.maximum(target[:, None], .01) / np.maximum(source, .01))
    reliability = np.minimum(target[:, None], source)**2
    reliability /= reliability + .06**2
    return np.clip(scalar[:, :, None] + reliability * (rgb - scalar[:, :, None]), np.log(.5), np.log(2)).astype(np.float32)


def estimate_color_field(mesh, prepared, settings, gains=None, *, native_sigma=8):
    """Fit one all-photo field, with held-out interior continuity verification.

    At most 200,000 surface vertices and 64 photos are retained. The largest
    RGB array is 154 MB at that budget; several color/support buffers coexist.
    Robust fitting/gain temporaries are chunked and only one photo is blurred
    at a time. Input photographs and Open3D baking require additional memory.
    """
    from .texturing import _project
    import cv2

    started = time.monotonic()
    report = dict(applied=False, method='surface_low_frequency_photo_gains',
                  correction_mode='log_gain', native_lowpass_sigma_pixels=native_sigma,
                  physical_kernel_passes=2, gain_bounds=[.5, 2], dark_channel_floor=.06)
    n, views = len(mesh.vertices), len(prepared)
    if views < 2 or not n or not len(mesh.triangles):
        report['reason'] = 'Fewer than two source photos or empty mesh'
        return None, report
    if n > 200000 or views > 64:
        report['reason'] = 'Surface color fit exceeds 200000 vertices or 64 photos'
        return None, report
    gains = np.ones((views, 3), np.float32) if gains is None else np.asarray(gains)
    if gains.shape != (views, 3) or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError('One finite positive RGB gain is required per photo')
    if not np.isfinite(native_sigma) or native_sigma <= 0:
        raise ValueError('A positive native low-pass sigma is required')
    scoring = copy.deepcopy(mesh)
    scoring.orient_triangles()
    vertices = np.asarray(scoring.vertices).copy()
    faces = np.asarray(scoring.triangles).copy()
    normals = _normal_axes(vertices, faces)
    from open3d import utility
    scoring.vertex_normals = utility.Vector3dVector(normals)
    report['normal_method'] = 'incident_face_normal_axis'
    # These interior points do not train the vertex fields. Compare corrected
    # observations at identical world positions, including their depth tests.
    held_faces = np.linspace(0, len(faces) - 1, min(2048, len(faces)), dtype=int)
    triangles = vertices[faces[held_faces]]
    held_world = np.einsum('k,fkj->fj', [.23, .31, .46], triangles)
    held_normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    held_normal /= np.maximum(np.linalg.norm(held_normal, axis=1, keepdims=True), 1e-12)
    colors = np.zeros((n, views, 3), np.float32)
    quality = np.zeros((n, views), np.float32)
    held_colors = np.zeros((len(held_faces), views, 3), np.float32)
    held_weights = np.zeros((len(held_faces), views), np.float32)
    for view, (pose, rgb, depth) in enumerate(prepared):
        blurred = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), native_sigma)
        for world, normal, values_out, weights_out in (
            (vertices, normals, colors[:, view], quality[:, view]),
            (held_world, held_normal, held_colors[:, view], held_weights[:, view]),
        ):
            for start in range(0, len(world), 65536):
                stop = start + 65536
                values, weight = _project(world[start:stop], normal[start:stop], blurred,
                    depth, pose, settings.camera, settings.near_m, settings.far_m,
                    settings.sensor_calibration, settings.rgb_camera)
                values_out[start:stop] = np.clip(values * gains[view], 0, 1)
                weights_out[start:stop] = weight
    support = (quality > 0).astype(np.float32)
    colors *= support[:, :, None]
    kernel = _surface_kernel(scoring)
    for _ in range(2):
        support = kernel @ support
        colors = (kernel @ colors.reshape(n, -1)).reshape(n, views, 3)
        quality = kernel @ quality
    colors /= np.maximum(support[:, :, None], 1e-8)
    visible = support > .1
    quality *= visible
    target = _consensus(colors, quality)
    del quality
    logs = np.empty_like(colors)
    for start in range(0, n, 4096):
        logs[start:start + 4096] = _bounded_log_gains(target[start:start + 4096], colors[start:start + 4096])
    del colors
    logs[~visible] = 0
    # A lone observation cannot establish a relative color correction.
    logs[(visible.sum(axis=1) < 2)] = 0
    field = PhotoColorField(vertices, faces, logs, np.arange(len(faces)), np.arange(views))
    corrected = held_colors.copy()
    for view in range(views):
        corrected[:, view] *= np.exp(field.evaluate(held_world, held_faces, view))
    valid = (held_weights > 0).sum(axis=1) >= 2
    if valid.any():
        before_target = _consensus(held_colors, held_weights)
        after_target = _consensus(corrected, held_weights)
        w = held_weights * valid[:, None]
        before_error = np.mean(np.abs(held_colors - before_target[:, None]), axis=2)
        after_error = np.mean(np.abs(corrected - after_target[:, None]), axis=2)
        # Independent disconnected surfaces must not hide one another's failure.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        _, components = connected_components(graph, directed=False)
        held_components = components[faces[held_faces, 0]]
        rejected = []
        for component in np.unique(held_components[valid]):
            ids = valid & (held_components == component)
            if ids.sum() < 4:
                continue
            point_w = w[ids]
            before_points = np.sum(before_error[ids] * point_w, axis=1) / point_w.sum(axis=1)
            after_points = np.sum(after_error[ids] * point_w, axis=1) / point_w.sum(axis=1)
            if after_points.mean() > before_points.mean() + .002 and np.median(after_points) > np.median(before_points) + .002:
                logs[components == component] = 0
                corrected[ids] = held_colors[ids]
                after_error[ids] = before_error[ids]
                rejected.append(int(component))
        before = np.sum(before_error * w) / max(w.sum(), 1e-8)
        after = np.sum(after_error * w) / max(w.sum(), 1e-8)
        report.update(validation_points=int(valid.sum()), validation_before=float(before), validation_after=float(after))
        report['validation_clipped_observations'] = int(np.sum((held_weights > 0) & np.any(corrected > 1, axis=2)))
        report.update(rejected_components=len(rejected), validation_regressed_points=int(np.sum(valid &
            ((np.sum(after_error * w, axis=1) - np.sum(before_error * w, axis=1)) > .005 * w.sum(axis=1)))))
        if (before > .002 and after > before * .98) or after > before + .002:
            report['reason'] = 'Held-out low-frequency continuity did not improve'
            return None, report
    else:
        report['reason'] = 'No jointly visible held-out surface points'
        return None, report
    report.update(applied=bool(np.any(np.abs(logs) > 1e-5)),
                  reason='Depth-visible shared surface color fit', surface_vertices=n,
                  source_photos=views, fit_seconds=time.monotonic() - started,
                  field_bytes=int(logs.nbytes))
    return field, report
