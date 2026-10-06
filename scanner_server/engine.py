"""ScanEngine — CPU/CUDA 3D scanning pipeline using Open3D tensor API.

Two-phase workflow:
  Phase 1 (capture): Store raw RGB+depth frames as fast as possible.
  Phase 2 (process): On demand, process stored frames through the
      registration + TSDF integration pipeline, then extract mesh.

Processing is incremental — only unprocessed frames are run through ICP/TSDF,
so mid-scan previews don't re-process already-integrated frames.

TSDF integration and extraction use the selected CPU/CUDA VoxelBlockGrid.
Geometric ICP can use the tensor backend; feature/color recovery remains on CPU.
"""

import copy
import logging
import os
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import open3d.core as o3c
import trimesh

from shared.calibration import prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS, RGB_DEPTH_CAPTURE_LIMIT_MS
from shared.config import LIVE_MAX_POINTS, PRESET_DEFAULT, ScanPreset
from shared.settings import ScanSettings

from .backend import select_backend

_REG = o3d.pipelines.registration
logger = logging.getLogger("scanner_server")


class ScanEngine:
    MODEL_REFRESH_INTERVAL = 3  # re-extract model_pcd every N integrations
    MAX_FRAMES = int(os.environ.get("KINECT_MAX_FRAMES", "500"))
    BLOCK_COUNT = int(os.environ.get("KINECT_BLOCK_COUNT", "50000"))

    def __init__(self, device=None, tracking=None):
        self.preset = PRESET_DEFAULT
        self.settings = ScanSettings()
        self.device, self.backend = select_backend(device, tracking)
        logger.info("ScanEngine using device: %s", self.device)
        self.reset()

    # ── VBG creation ──────────────────────────────────────────────────

    def _create_vbg(self, block_count=None):
        """Create a fresh VoxelBlockGrid on self.device."""
        return o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=self.voxel_size,
            block_resolution=16,
            block_count=self.BLOCK_COUNT if block_count is None else block_count,
            device=self.device,
        )

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self, preset: ScanPreset | None = None, settings: ScanSettings | None = None
    ):
        if preset is not None:
            self.preset = preset
        if settings is not None:
            self.settings = settings
        elif preset is not None:
            self.settings = ScanSettings(
                voxel_m=preset.voxel_size,
                truncation_m=preset.sdf_trunc,
                near_m=preset.depth_near_mm / 1000,
                far_m=preset.max_depth_m,
            )
        p = self.settings
        self.backend["fusion"] = (
            "confidence_weighted" if p.confidence_fusion else "uniform"
        )
        self.voxel_size = p.voxel_m
        self.sdf_trunc = p.truncation_m
        self.max_depth_m = float(p.far_m)
        self.reg_voxel = max(p.voxel_m * 1.5, 0.0075)
        c = p.camera
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            c.width, c.height, c.fx, c.fy, c.cx, c.cy
        )
        self.intrinsic_tensor = o3c.Tensor(
            self.intrinsic.intrinsic_matrix, dtype=o3c.float64
        )

        self.vbg = self._create_vbg()

        self.frame_count = 0
        self.session_id = uuid.uuid4().hex
        self.stage_totals_ms = {}
        self._frame_timings = {}
        self._stage_children = []
        self.refinement = {"applied": False, "reason": "Not requested"}
        self.original_poses = None
        self._refined_count = None
        self._reconnection_count = None
        self._pose_seeds_only = False
        self.fragment_reconnection = {"applied": False, "reason": "Not requested"}
        self.model_pcd = None
        self._live_points = np.empty((0, 3), dtype=np.float32)
        self._live_colors = np.empty((0, 3), dtype=np.float32)
        self._last_rgbd = None
        self._last_reg_pcd = None
        self._model_fpfh = None
        self._model_feature_cloud = None
        self._model_pyramid = {}
        self._tensor_model_pyramid = {}
        self._integrations_since_model = 0
        self.cumulative_T = np.eye(4)
        self.mesh = None
        self.point_cloud = None
        self._final_vbg = None
        self._fusion_block_limit = None
        self.final_reconstruction = {"applied": False, "reason": "Uses live volume"}

        # Raw frame storage
        self.raw_frames: list[tuple[np.ndarray, np.ndarray]] = []
        self._processed_count = 0
        self.poses = []
        self.diagnostics = []
        self.frame_metadata = []
        self._frame_ids = set()
        self._stored_monotonic = []
        self._appearance_cache = {}
        self._visual_cache = {}
        self._visual_evidence = None
        self.tracking_edges = []
        self._tracking_lost_frames = 0
        self._lost_at_index = None
        self._recovery_preview = None
        self._world_up = np.array([0.0, -1.0, 0.0])
        self._up_estimated = False
        self._last_fusion_stats = {}

    # ── helpers ────────────────────────────────────────────────────────

    @contextmanager
    def _stage(self, name):
        # CUDA kernels are asynchronous; synchronize to report real wall time.
        if str(self.device).startswith("CUDA"):
            o3c.cuda.synchronize()
        start = time.monotonic()
        self._stage_children.append(0.0)
        try:
            yield
        finally:
            if str(self.device).startswith("CUDA"):
                o3c.cuda.synchronize()
            elapsed = (time.monotonic() - start) * 1000
            exclusive = max(0.0, elapsed - self._stage_children.pop())
            if self._stage_children:
                self._stage_children[-1] += elapsed
            self._frame_timings[name] = self._frame_timings.get(name, 0) + exclusive
            self.stage_totals_ms[name] = self.stage_totals_ms.get(name, 0) + exclusive

    def reconstruction_report(self):
        return {
            "version": 1,
            "session_id": self.session_id,
            "pose_convention": "camera_to_world",
            "length_unit": "metres",
            "settings": self.settings.to_dict(),
            "backend": self.backend,
            "frames": list(self.diagnostics),
            "poses": [
                {"index": i, "camera_to_world": p.tolist()} for i, p in self.poses
            ],
            "original_poses": None
            if self.original_poses is None
            else [
                {"index": i, "camera_to_world": p.tolist()}
                for i, p in self.original_poses
            ],
            "stage_totals_ms": dict(self.stage_totals_ms),
            "refinement": self.refinement,
            "final_reconstruction": self.final_reconstruction,
            "fragment_reconnection": self.fragment_reconnection,
            "tracking_edges": list(self.tracking_edges),
            "tracking": {
                "state": "recovering" if self._tracking_lost_frames else "tracking" if self.poses else "waiting",
                "last_tracked_index": self.poses[-1][0] if self.poses else None,
                "lost_at_index": self._lost_at_index,
            },
        }

    def live_snapshot(self, max_points=LIVE_MAX_POINTS, *, array_geometry=False):
        """Bounded immutable view of cached fused geometry; no extra extraction."""
        if max_points < 1:
            raise ValueError("Live point budget must be positive")
        max_points = min(max_points, LIVE_MAX_POINTS)
        points, colors = self._live_points, self._live_colors
        step = max(1, int(np.ceil(len(points) / max_points)))
        pending_age = (
            time.monotonic() - self._stored_monotonic[self._processed_count]
            if self.unprocessed_count
            else 0.0
        )
        result = self.diagnostics[-1] if self.diagnostics else {}
        points = points[::step].astype(np.float32)
        colors = np.clip(colors[::step], 0, 1).astype(np.float32)
        if self.mesh is not None:
            excluded = len(self.fragment_reconnection.get("excluded_frames", []))
            guidance = f"Final model retains {self.frame_count} of {self.stored_count} captures."
            if excluded:
                guidance += f" {excluded} previously fused views were removed because their positions could not be verified."
            guidance += " Save Session preserves all captured views."
        elif result and not result.get("success"):
            guidance = (
                "STOP — model paused. Return to the highlighted camera and match "
                "the last good image. Recovery frames are checked without adding geometry."
                if self.poses else
                "STOP — model paused. Include more measured surface within depth range "
                "to establish the first tracked view."
            )
        elif self.unprocessed_count >= 5 or pending_age > 2:
            guidance = (
                "Pause movement or reduce capture rate while reconstruction catches up"
            )
        elif result.get("valid_depth_fraction", 1) < 0.2:
            guidance = "Move within depth range and include more measured surface"
        else:
            guidance = "Move slowly with overlap; revisit textured corners"
        return {
            "type": "live",
            "session_id": self.session_id,
            "stored_count": self.stored_count,
            "frame_count": self.frame_count,
            "processed_count": self._processed_count,
            "processing_interval_s": (
                sum(r.get("elapsed_ms", 0) for r in self.diagnostics[-12:])
                / min(12, len(self.diagnostics)) / 1000
                if self.diagnostics else 0.0
            ),
            "pending_count": self.unprocessed_count,
            "pending_age_s": round(max(0, pending_age), 3),
            "skipped_count": sum(not r["success"] for r in self.diagnostics),
            "guidance": guidance,
            "geometry_frame_count": self.frame_count - self._integrations_since_model,
            "points": points if array_geometry else points.tolist(),
            "colors": colors if array_geometry else colors.tolist(),
            "camera_to_world": self.cumulative_T.tolist(),
            **self.tracking_snapshot(),
            "camera": asdict(self.settings.camera),
            "color_assistance_requested": self.settings.color_recovery or self.settings.relocalize,
            "surface_description": (
                "Fused preview includes tentative surface. Inspect shows the final mesh."
                if self.mesh is not None else
                "Live preview includes tentative surface. Finish may remove weak or unconnected areas."
            ),
            "backend": self.backend,
            "result": self.diagnostics[-1] if self.diagnostics else {},
        }

    def tracking_snapshot(self):
        """Accepted poses only: the camera's current pose is unknown while lost."""
        lost = self._tracking_lost_frames > 0
        anchor = self.poses[-1] if self.poses else None
        if lost and anchor is not None and self._recovery_preview is None:
            import base64

            import cv2

            rgb = self.raw_frames[anchor[0]][0]
            height = max(1, round(rgb.shape[0] * 240 / rgb.shape[1]))
            preview = cv2.resize(rgb, (240, height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if ok:
                self._recovery_preview = base64.b64encode(encoded).decode("ascii")
        return {
            "tracking_state": "recovering" if lost else "tracking" if anchor else "waiting",
            "fusion_paused": lost,
            "lost_at_index": self._lost_at_index,
            "last_tracked_index": anchor[0] if anchor else None,
            "last_tracked_rgb_png": self._recovery_preview if lost else None,
            "trajectory": [
                {"index": i, "camera_to_world": p.tolist()} for i, p in self.poses
            ],
            "world_up": self._world_up.tolist(),
            "up_estimated": self._up_estimated,
        }

    def _make_rgbd(self, rgb, depth):
        """Create legacy Open3D RGBDImage for registration point cloud.

        Input depth is native raw disparity or legacy registered millimetres.
        """
        depth_f = depth.astype(np.float32)
        depth_f[(depth == 0) | (depth > self.max_depth_m * 1000)] = 0.0

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_f, dtype=np.float32))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=self.max_depth_m,
            convert_rgb_to_intensity=False,
        )

    def _integrate_vbg(self, rgb, depth, extrinsic, volume=None):
        """Run TSDF integration via VoxelBlockGrid.

        Creates tensor images, integrates, and frees them immediately.
        """
        volume = self.vbg if volume is None else volume
        depth_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth))).to(
            self.device
        )
        extrinsic_t = o3c.Tensor(extrinsic, dtype=o3c.float64)

        frustum_block_coords = volume.compute_unique_block_coordinates(
            depth_img,
            self.intrinsic_tensor,
            extrinsic_t,
            depth_scale=1000.0,
            depth_max=self.max_depth_m,
            trunc_voxel_multiplier=self.sdf_trunc / self.voxel_size,
        )
        if self._fusion_block_limit is not None:
            _, found = volume.hashmap().find(frustum_block_coords)
            added = int(np.count_nonzero(~found.cpu().numpy()))
            if volume.hashmap().size() + added > self._fusion_block_limit:
                raise ValueError(
                    f"Final volume exceeds {self._fusion_block_limit} blocks; "
                    "use a coarser final voxel or increase the final block budget"
                )
        if self.settings.confidence_fusion:
            from .weighted_fusion import integrate_weighted

            self._last_fusion_stats = integrate_weighted(
                self, volume, frustum_block_coords, rgb, depth, extrinsic
            )
            return
        color_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb))).to(
            self.device
        )
        volume.integrate(
            frustum_block_coords,
            depth_img,
            color_img,
            self.intrinsic_tensor,
            self.intrinsic_tensor,
            extrinsic_t,
            depth_scale=1000.0,
            depth_max=self.max_depth_m,
            trunc_voxel_multiplier=self.sdf_trunc / self.voxel_size,
        )
        del depth_img, color_img, frustum_block_coords, extrinsic_t

    def _make_reg_pcd(self, rgbd):
        """Build a coarser point cloud for registration (not integration)."""
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd = pcd.voxel_down_sample(self.reg_voxel)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.reg_voxel * 4, max_nn=30)
        )
        return pcd

    def _extract_model_pcd(self):
        """Extract and downsample a model point cloud from the TSDF volume.

        Cache registration scales; compute FPFH lazily if recovery needs it.
        """
        t_pcd = self.vbg.extract_point_cloud(
            weight_threshold=0.01 if self.settings.confidence_fusion else 0.5
        )
        pcd = t_pcd.to_legacy()
        del t_pcd
        # Reuse this extraction before tracking downsamples it. Keep only a
        # bounded copy for feedback, independent of the registration model.
        points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        step = max(1, int(np.ceil(len(points) / LIVE_MAX_POINTS)))
        self._live_points = points[::step].astype(np.float32)
        self._live_colors = colors[::step].astype(np.float32)
        pcd = pcd.voxel_down_sample(self.reg_voxel)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.reg_voxel * 4, max_nn=30)
        )
        self.model_pcd = pcd

        self._model_feature_cloud = None
        self._model_fpfh = None  # Compute only when recovery actually needs it.
        self._model_pyramid = {}
        self._tensor_model_pyramid = {}
        for scale in (4, 2, 1):
            voxel = self.reg_voxel * scale
            level = pcd.voxel_down_sample(voxel)
            level.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
            )
            self._model_pyramid[scale] = level
            if self.backend["tracking"] == "tensor":
                self._tensor_model_pyramid[scale] = (
                    o3d.t.geometry.PointCloud.from_legacy(
                        level, dtype=o3c.float32, device=self.device
                    )
                )

        self._integrations_since_model = 0

    def _icp(self, source, target, init=None):
        """Coarse-to-fine robust point-to-plane tracking in camera-to-world space."""
        pose = self.cumulative_T if init is None else init
        for scale, iterations in ((4, 40), (2, 30), (1, 20)):
            voxel = self.reg_voxel * scale
            src = source.voxel_down_sample(voxel)
            if target is self.model_pcd:
                tgt = self._model_pyramid[scale]
            else:
                tgt = target.voxel_down_sample(voxel)
                tgt.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
                )
            if self.backend["tracking"] == "tensor":
                source_tensor = o3d.t.geometry.PointCloud.from_legacy(
                    src, dtype=o3c.float32, device=self.device
                )
                target_tensor = (
                    self._tensor_model_pyramid[scale]
                    if target is self.model_pcd
                    else o3d.t.geometry.PointCloud.from_legacy(
                        tgt, dtype=o3c.float32, device=self.device
                    )
                )
                reg = o3d.t.pipelines.registration
                kernel = reg.robust_kernel.RobustKernel(
                    reg.robust_kernel.RobustKernelMethod.HuberLoss, max(0.005, voxel)
                )
                native = reg.icp(
                    source_tensor,
                    target_tensor,
                    voxel * 3,
                    o3c.Tensor(pose, dtype=o3c.float64),
                    reg.TransformationEstimationPointToPlane(kernel),
                    reg.ICPConvergenceCriteria(max_iteration=iterations),
                )
                match = native.correspondence_set.cpu().numpy().reshape(-1)
                valid = np.flatnonzero(match >= 0)
                result = SimpleNamespace(
                    transformation=native.transformation.cpu().numpy(),
                    fitness=native.fitness,
                    inlier_rmse=native.inlier_rmse,
                    correspondence_set=np.column_stack((valid, match[valid])),
                )
            else:
                result = _REG.registration_icp(
                    src,
                    tgt,
                    max_correspondence_distance=voxel * 3,
                    init=pose,
                    estimation_method=_REG.TransformationEstimationPointToPlane(
                        _REG.HuberLoss(k=max(0.005, voxel))
                    ),
                    criteria=_REG.ICPConvergenceCriteria(max_iteration=iterations),
                )
            pose = result.transformation
        return result

    def _alignment_error(self, result, target, check_motion=True):
        s = self.settings
        if not np.all(np.isfinite(result.transformation)):
            return "Non-finite tracking pose"
        if result.fitness < s.min_fitness or result.inlier_rmse > s.max_rmse_m:
            return (
                f"Weak alignment: overlap {result.fitness:.2f}, "
                f"RMSE {result.inlier_rmse * 1000:.1f} mm"
            )
        relative = np.linalg.inv(self.cumulative_T) @ result.transformation
        translation = np.linalg.norm(relative[:3, 3])
        angle = np.degrees(
            np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))
        )
        if check_motion and (
            translation > s.max_translation_m or angle > s.max_rotation_deg
        ):
            return f"Pose jump: {translation:.2f} m / {angle:.1f} degrees; move slower"
        # Normal diversity detects translation ambiguity on a single plane.
        matched_target = self._model_pyramid[1] if target is self.model_pcd else target
        normals = np.asarray(matched_target.normals)
        correspondences = np.asarray(result.correspondence_set)
        if len(correspondences):
            normals = normals[correspondences[:, 1]]
        if (
            len(normals)
            and np.linalg.eigvalsh(normals.T @ normals / len(normals))[0] < 0.002
        ):
            return "Tracking is ambiguous on a flat surface; include corners or curved geometry"
        return None

    def _relocalize(self, source, rgbd):
        """Appearance is only a seed; require reciprocal geometry and model checks."""
        if (
            not self.settings.relocalize
            or self._tracking_lost_frames < 2
            or rgbd is None
        ):
            return None
        from .appearance import correspondences, extract_features, propose_transform
        from .refinement import _match, _trustworthy, motion

        lag = self.frame_metadata[self._processed_count].get("rgb_depth_delta_ms")
        if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
            return None
        features = extract_features(
            np.asarray(rgbd.color),
            np.rint(np.asarray(rgbd.depth) * 1000).astype(np.uint16),
            self.settings.camera,
        )
        selected = np.unique(
            np.linspace(0, len(self.poses) - 1, min(32, len(self.poses)), dtype=int)
        )
        candidates = []
        keep = {self.poses[j][0] for j in selected}
        self._appearance_cache = {
            i: f for i, f in self._appearance_cache.items() if i in keep
        }
        for j in selected:
            index, pose = self.poses[j]
            lag = self.frame_metadata[index].get("rgb_depth_delta_ms")
            if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                continue
            if index not in self._appearance_cache:
                rgb, depth = prepare_rgbd(*self.raw_frames[index], self.settings)
                self._appearance_cache[index] = extract_features(
                    rgb, depth, self.settings.camera
                )
            matched = correspondences(features, self._appearance_cache[index])
            if len(matched) >= 40:
                candidates.append((len(matched), index, pose, matched))
        for _, index, pose, matches in sorted(candidates, key=lambda r: -r[0])[:3]:
            initial = propose_transform(
                features, self._appearance_cache[index], self.settings.camera, matches
            )
            if initial is None:
                continue
            rgb, depth = prepare_rgbd(*self.raw_frames[index], self.settings)
            target = self._make_reg_pcd(self._make_rgbd(rgb, depth))
            forward = _match(source, target, initial)
            reverse = _match(target, source, np.linalg.inv(forward.transformation))
            translation, angle = motion(reverse.transformation @ forward.transformation)
            if not (
                _trustworthy(forward, target)
                and _trustworthy(reverse, source)
                and translation < 0.01
                and angle < 2
            ):
                continue
            result = self._icp(
                source, self.model_pcd, init=pose @ forward.transformation
            )
            correction_m, correction_deg = motion(
                np.linalg.inv(pose @ forward.transformation) @ result.transformation)
            if (
                result.fitness >= 0.6
                and correction_m <= 0.03 and correction_deg <= 3
                and self._alignment_error(result, self.model_pcd, False) is None
            ):
                return result
        return None

    def _fpfh_fallback(self, source, target):
        """Fast Global Registration using cached model FPFH, refined with ICP."""
        # Global feature matching must not compare every fine model point.
        # A bounded, coarse cloud prevents very long recovery on large scans.
        feature_voxel = max(self.reg_voxel * 4, 0.03)

        def coarse(cloud):
            cloud = cloud.voxel_down_sample(feature_voxel)
            if len(cloud.points) > 5000:
                cloud = cloud.uniform_down_sample(
                    int(np.ceil(len(cloud.points) / 5000))
                )
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=feature_voxel * 2, max_nn=30
                )
            )
            return cloud

        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_voxel * 5, max_nn=100
        )
        coarse_source = coarse(source)
        src_feat = _REG.compute_fpfh_feature(coarse_source, search_param)
        if self._model_fpfh is None:
            self._model_feature_cloud = coarse(target)
            self._model_fpfh = _REG.compute_fpfh_feature(
                self._model_feature_cloud, search_param
            )
        fgr = _REG.registration_fgr_based_on_feature_matching(
            coarse_source,
            self._model_feature_cloud,
            src_feat,
            self._model_fpfh,
            _REG.FastGlobalRegistrationOption(
                maximum_correspondence_distance=feature_voxel * 2, iteration_number=32
            ),
        )

        # Refine with ICP
        refined = self._icp(source, target, init=fgr.transformation)
        return refined

    def _tracking_initial_guess(self):
        """Predict smooth camera motion; bound extrapolation when frames are skipped."""
        continuous = self._continuous_motion_guess()
        if continuous is not None:
            return continuous
        if len(self.poses) < 2:
            return self.cumulative_T
        previous_index, previous = self.poses[-2]
        last_index, last = self.poses[-1]
        current_index = self._processed_count

        def timestamp(index):
            value = self.frame_metadata[index].get("timestamp_s")
            return float(index) if value is None else value

        previous_time, last_time, current_time = map(
            timestamp, (previous_index, last_index, current_index)
        )
        interval = last_time - previous_time
        gap = current_time - last_time
        if not (
            np.isfinite(interval) and np.isfinite(gap) and interval > 0 and gap > 0
        ):
            return last
        ratio = min(gap / interval, 2.0)
        delta = np.linalg.inv(previous) @ last
        extrapolation = np.eye(4)
        rotation = delta[:3, :3]
        angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
        if angle > 1e-8:
            if abs(np.sin(angle)) < 1e-8:
                return last
            axis = np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]
            ) / (2 * np.sin(angle))
            extrapolation[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis * angle * ratio
            )
        extrapolation[:3, 3] = delta[:3, 3] * ratio
        if np.linalg.norm(extrapolation[:3, 3]) > self.settings.max_translation_m:
            return last
        return last @ extrapolation

    def _continuous_motion_guess(self):
        """Camera-side short-baseline motion is a seed, never fusion authority."""
        from .fragments import _rigid

        if not self.settings.color_recovery or not self.poses:
            return None
        if not 0 <= self._processed_count < len(self.frame_metadata):
            return None
        if any(self.frame_metadata[i].get("rgb_depth_delta_ms") is not None
               and abs(self.frame_metadata[i]["rgb_depth_delta_ms"]) > RGB_DEPTH_ASSISTANCE_LIMIT_MS
               for i in (self._processed_count, self.poses[-1][0])):
            return None
        current = self.frame_metadata[self._processed_count].get("visual_tracking", {})
        last = self.frame_metadata[self.poses[-1][0]].get("visual_tracking", {})
        if not (isinstance(current, dict) and isinstance(last, dict)
                and current.get("valid") and last.get("valid")
                and current.get("segment") and current.get("segment") == last.get("segment")):
            return None
        try:
            a, b = np.asarray(last.get("camera_to_local"), float), np.asarray(current.get("camera_to_local"), float)
            if not (_rigid(a) and _rigid(b)):
                return None
            from .refinement import motion

            relative = np.linalg.inv(a) @ b
            translation, angle = motion(relative)
            if translation > self.settings.max_translation_m or angle > self.settings.max_rotation_deg:
                return None
            return self.cumulative_T @ relative
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _visual_register(self, source, rgbd):
        """Use measured keyframes and retain their visual constraint after ICP."""
        from shared.visual_tracking import feature_agreement

        from .appearance import correspondences, extract_features, propose_transform
        from .refinement import motion

        if not self.settings.color_recovery or rgbd is None or not self.poses:
            return None
        index = self._processed_count
        lag = self.frame_metadata[index].get("rgb_depth_delta_ms")
        if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
            return None
        features = extract_features(np.asarray(rgbd.color),
                                    np.rint(np.asarray(rgbd.depth) * 1000).astype(np.uint16),
                                    self.settings.camera)
        recent = self.poses[-8:]
        historical = [self.poses[j] for j in np.unique(np.linspace(
            0, len(self.poses) - 1, min(4, len(self.poses)), dtype=int))]
        candidates = {i: pose for i, pose in historical + recent}
        self._visual_cache = {i: value for i, value in self._visual_cache.items() if i in candidates}
        for target_index in sorted(candidates, reverse=True):
            target_pose = candidates[target_index]
            lag = self.frame_metadata[target_index].get("rgb_depth_delta_ms")
            if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                continue
            if target_index not in self._visual_cache:
                rgb, depth = prepare_rgbd(*self.raw_frames[target_index], self.settings)
                target_rgbd = self._make_rgbd(rgb, depth)
                self._visual_cache[target_index] = (
                    extract_features(rgb, depth, self.settings.camera),
                    self._make_reg_pcd(target_rgbd))
            target_features, target = self._visual_cache[target_index]
            matches = correspondences(features, target_features)
            proposal = propose_transform(features, target_features, self.settings.camera, matches)
            if proposal is None:
                continue
            a, b = matches.T
            # Feature identities survive refinement: anonymous geometric overlap
            # cannot move a textured surface to another plausible model location.
            pose = target_pose @ proposal
            translation, angle = motion(np.linalg.inv(self.cumulative_T) @ pose)
            if not self._tracking_lost_frames and (
                    translation > self.settings.max_translation_m or angle > self.settings.max_rotation_deg):
                continue
            native = self._icp(source, target, proposal)
            correction, correction_angle = motion(np.linalg.inv(proposal) @ native.transformation)
            good, stats = feature_agreement(features.points[a], target_features.points[b],
                                           target_features.pixels[b], native.transformation,
                                           self.settings.camera)
            if not good or correction > 0.03 or correction_angle > 3:
                # Retain the observed proposal if geometry independently agrees.
                native = _REG.evaluate_registration(source, target, 0.0225, proposal)
                good, stats = feature_agreement(features.points[a], target_features.points[b],
                                               target_features.pixels[b], proposal, self.settings.camera)
            if not good or native.fitness < 0.6 or native.inlier_rmse > 0.015:
                continue
            moved = copy.deepcopy(source).transform(native.transformation)
            reverse = _REG.evaluate_registration(target, moved, 0.0225, np.eye(4))
            if reverse.fitness < 0.45 or reverse.inlier_rmse > 0.015:
                continue
            self._visual_evidence = {"target_index": target_index, "feature_support": stats,
                                     "forward_overlap": float(native.fitness),
                                     "reverse_overlap": float(reverse.fitness),
                                     "relative_pose": native.transformation.tolist()}
            return SimpleNamespace(transformation=target_pose @ native.transformation,
                                   fitness=native.fitness, inlier_rmse=native.inlier_rmse,
                                   correspondence_set=native.correspondence_set)
        return None

    def _color_recovery(self, source_pcd, rgbd):
        """Use synchronized RGB-D motion to seed geometry; never bypass its gates."""
        if not self.settings.color_recovery or self._last_rgbd is None or rgbd is None:
            return None
        lag = self.frame_metadata[self._processed_count].get("rgb_depth_delta_ms")
        previous_lag = self.frame_metadata[self.poses[-1][0]].get("rgb_depth_delta_ms")
        if any(value is not None and abs(value) > RGB_DEPTH_ASSISTANCE_LIMIT_MS
               for value in (lag, previous_lag)):
            return None

        def intensity(frame):
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                frame.color,
                frame.depth,
                depth_scale=1.0,
                depth_trunc=self.max_depth_m,
                convert_rgb_to_intensity=True,
            )

        current = intensity(rgbd)
        previous = intensity(self._last_rgbd)
        image = np.asarray(current.color)
        gy, gx = np.gradient(image)
        if np.mean(np.hypot(gx, gy) > 0.015) < 0.05:
            return None  # Insufficient texture; a successful solve is not evidence.
        option = o3d.pipelines.odometry.OdometryOption(
            iteration_number_per_pyramid_level=o3d.utility.IntVector([40, 20, 10]),
            depth_diff_max=0.05,
            depth_min=float(self.settings.near_m),
            depth_max=self.max_depth_m,
        )
        guess = np.linalg.inv(self.cumulative_T) @ self._tracking_initial_guess()
        success, relative, information = o3d.pipelines.odometry.compute_rgbd_odometry(
            current,
            previous,
            self.intrinsic,
            guess,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        if not success or not np.all(np.isfinite(information)):
            return None
        pose = self.cumulative_T @ relative
        refined = self._icp(source_pcd, self.model_pcd, init=pose)
        from .refinement import motion

        correction, angle = motion(np.linalg.inv(pose) @ refined.transformation)
        if correction <= 0.03 and angle <= 3 and self._alignment_error(refined, self.model_pcd) is None:
            return refined
        return None

    def _register(self, source_pcd, rgbd=None):
        """Register source (camera coords) against model_pcd (world coords).

        Uses cumulative_T as initial guess so ICP starts near the true pose.
        Returns (result, method_str) or (None, error_str).
        """
        self._visual_evidence = None
        visual = self._visual_register(source_pcd, rgbd)
        if visual is not None:
            return visual, "keyframe+visual"
        if self._tracking_lost_frames:
            # ICP against a large accumulated model can snap onto another side
            # of a box. Resume only after verification against the last actual
            # camera observation, or verified appearance relocalization.
            recovered = self._recover_anchor(source_pcd)
            if recovered is not None:
                return recovered, "anchor+icp"
            try:
                recovered = self._relocalize(source_pcd, rgbd)
                if recovered is not None:
                    return recovered, "appearance+icp"
            except (RuntimeError, ValueError):
                logger.debug("Appearance relocalization failed", exc_info=True)
            return None, "Tracking lost; match the last good view to resume fusion"
        if self.settings.color_recovery:
            try:
                recovered = self._color_recovery(source_pcd, rgbd)
                if recovered is not None:
                    return recovered, "rgbd+icp"
            except (RuntimeError, ValueError):
                logger.debug("Color-assisted tracking failed", exc_info=True)
        result = self._icp(source_pcd, self.model_pcd, init=self.cumulative_T)
        error = self._alignment_error(result, self.model_pcd)
        if error is None:
            return result, "icp"
        # A turning camera can see surfaces from recently accepted frames that
        # are not in the cached model yet. Refresh before attempting recovery.
        if self._integrations_since_model:
            with self._stage("model_refresh"):
                self._extract_model_pcd()
            result = self._icp(source_pcd, self.model_pcd, init=self.cumulative_T)
            error = self._alignment_error(result, self.model_pcd)
            if error is None:
                return result, "icp"
        if len(self.poses) >= 2:
            predicted = self._icp(
                source_pcd, self.model_pcd, init=self._tracking_initial_guess()
            )
            if self._alignment_error(predicted, self.model_pcd) is None:
                return predicted, "motion+icp"
        try:
            recovered = self._color_recovery(source_pcd, rgbd)
            if recovered is not None:
                return recovered, "rgbd+icp"
        except (RuntimeError, ValueError):
            logger.debug("Color recovery failed", exc_info=True)
        # Global recovery is subject to the same confidence and motion limits.
        try:
            recovered = self._fpfh_fallback(source_pcd, self.model_pcd)
            if self._alignment_error(recovered, self.model_pcd) is None:
                return recovered, "global"
        except Exception:
            logger.debug("Global recovery failed", exc_info=True)
        try:
            recovered = self._relocalize(source_pcd, rgbd)
            if recovered is not None:
                return recovered, "appearance+icp"
        except (RuntimeError, ValueError):
            logger.debug("Appearance relocalization failed", exc_info=True)
        return None, error

    def _recover_anchor(self, source):
        """Require nearby, reciprocal overlap with the last accepted raw view."""
        from .refinement import _match, _trustworthy, motion

        if self._last_rgbd is None or not self.poses:
            return None
        if self._last_reg_pcd is None:
            self._last_reg_pcd = self._make_reg_pcd(self._last_rgbd)
        target = self._last_reg_pcd
        forward = _match(source, target, np.eye(4))
        if not _trustworthy(forward, target):
            return None
        translation, angle = motion(forward.transformation)
        if translation > min(0.15, self.settings.max_translation_m) or angle > min(
            15, self.settings.max_rotation_deg
        ):
            return None
        reverse = _match(target, source, np.linalg.inv(forward.transformation))
        translation, angle = motion(reverse.transformation @ forward.transformation)
        if not _trustworthy(reverse, source) or translation > 0.01 or angle > 2:
            return None
        if self._integrations_since_model:
            with self._stage("model_refresh"):
                self._extract_model_pcd()
        result = self._icp(
            source, self.model_pcd, self.poses[-1][1] @ forward.transformation
        )
        # Model refinement must agree with the independently observed anchor.
        translation, angle = motion(
            np.linalg.inv(self.poses[-1][1] @ forward.transformation)
            @ result.transformation
        )
        if (
            result.fitness < 0.6
            or translation > 0.03
            or angle > 3
            or self._alignment_error(result, self.model_pcd) is not None
        ):
            return None
        return result

    def _verify_tracking_transition(self, source, result):
        """A gap or large step needs actual camera-to-camera evidence."""
        from .fragments import capture_gap_limit
        from .refinement import _match, _trustworthy, motion

        if self._last_rgbd is None or not self.poses:
            return None
        index, last = self.poses[-1]
        relative = np.linalg.inv(last) @ result.transformation
        translation, angle = motion(relative)
        current_stamp = self.frame_metadata[self._processed_count].get("timestamp_s")
        last_stamp = self.frame_metadata[index].get("timestamp_s")
        gap = (current_stamp - last_stamp if current_stamp is not None and last_stamp is not None else 0)
        gap_limit = capture_gap_limit(self.frame_metadata[max(0, index - 8):self._processed_count + 1])
        if gap <= gap_limit and translation <= 0.1 and angle <= 8:
            return None
        if self._last_reg_pcd is None:
            self._last_reg_pcd = self._make_reg_pcd(self._last_rgbd)
        target = self._last_reg_pcd
        forward = _match(source, target, relative)
        reverse = _match(target, source, np.linalg.inv(forward.transformation))
        cycle_m, cycle_deg = motion(reverse.transformation @ forward.transformation)
        difference_m, difference_deg = motion(np.linalg.inv(relative) @ forward.transformation)
        if (_trustworthy(forward, target) and _trustworthy(reverse, source)
                and cycle_m < 0.01 and cycle_deg < 2
                and difference_m <= 0.03 and difference_deg <= 3):
            return None
        return "Unverified tracking transition after a gap or large movement; return to the last good view"

    # ── public API: capture (fast) ─────────────────────────────────────
    def store_frame(self, rgb: np.ndarray, depth: np.ndarray, metadata=None) -> dict:
        """Store a raw frame for later processing. Very fast — no ICP/TSDF."""
        c = self.settings.rgb_camera
        if rgb.shape != (c.height, c.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"RGB must be uint8 {c.height}x{c.width}x3 for the selected calibration")
        if depth.shape != (480, 640) or depth.dtype != np.uint16:
            raise ValueError("Depth must be uint16 480x640")
        metadata = dict(metadata or {})
        if metadata.get("depth_encoding", self.settings.depth_encoding) != self.settings.depth_encoding:
            raise ValueError("Frame depth encoding does not match the session calibration")
        if self.settings.sensor_calibration and np.any(depth > 2047):
            raise ValueError("Raw 11-bit depth codes must be in 0..2047")
        stamp = metadata.get("timestamp_s")
        if stamp is not None and (
            not isinstance(stamp, (int, float)) or not np.isfinite(stamp)
        ):
            raise ValueError("timestamp_s must be a finite number")
        frame_id = metadata.get("frame_id")
        if frame_id is not None:
            if not isinstance(frame_id, (str, int)):
                raise ValueError("frame_id must be a string or integer")
            if frame_id in self._frame_ids:
                return {
                    "success": False,
                    "stored_count": self.stored_count,
                    "message": "Duplicate capture skipped",
                }
        lag = metadata.get("rgb_depth_delta_ms")
        if lag is not None and (not np.isfinite(lag) or abs(lag) > RGB_DEPTH_CAPTURE_LIMIT_MS):
            return {
                "success": False,
                "stored_count": self.stored_count,
                "message": f"RGB/depth timestamps differ by more than {RGB_DEPTH_CAPTURE_LIMIT_MS} ms",
            }
        if self.stored_count >= self.MAX_FRAMES:
            return {
                "success": False,
                "stored_count": self.stored_count,
                "message": f"Session limit ({self.MAX_FRAMES} frames); build or start a new scan",
            }
        idx = len(self.raw_frames)
        self.raw_frames.append((rgb.copy(), depth.copy()))
        self._stored_monotonic.append(time.monotonic())
        self._final_vbg = None
        self.final_reconstruction = {"applied": False, "reason": "Awaiting final build"}
        self.mesh = None
        self.point_cloud = None
        self._refined_count = None
        self._reconnection_count = None
        self.fragment_reconnection = {"applied": False, "reason": "Awaiting final build"}
        self.refinement = {"applied": False, "reason": "Awaiting final build"}
        self.original_poses = None
        self.frame_metadata.append(metadata)
        if frame_id is not None:
            self._frame_ids.add(frame_id)
        return {
            "success": True,
            "stored_count": idx + 1,
            "index": idx,
            "session_id": self.session_id,
            "message": f"Frame {idx + 1} stored",
        }

    @property
    def stored_count(self) -> int:
        return len(self.raw_frames)

    @property
    def unprocessed_count(self) -> int:
        return len(self.raw_frames) - self._processed_count

    # ── public API: process on demand ──────────────────────────────────
    def process_frames(self, progress_cb=None, max_frames=None) -> dict:
        """Process all unprocessed stored frames through ICP + TSDF.

        This is incremental — only frames after _processed_count are processed.
        progress_cb(current_idx, total, result_dict) is called after each frame.

        Returns summary dict.
        """
        total = len(self.raw_frames)
        start = self._processed_count
        processed = 0
        errors = 0

        end = total if max_frames is None else min(total, start + max_frames)
        for i in range(start, end):
            rgb, depth = self.raw_frames[i]
            self._frame_timings = {}
            frame_started = time.monotonic()
            try:
                result = self._process_single_frame(rgb, depth)
            except Exception as exc:
                logger.exception("Frame %d failed", i)
                result = {"success": False, "message": f"Frame failed: {exc}"}
            self._processed_count = i + 1
            result["frame_count"] = self.frame_count
            result["index"] = i
            result["metadata"] = self.frame_metadata[i]
            result["session_id"] = self.session_id
            result["timings_ms"] = dict(self._frame_timings)
            if self.settings.confidence_fusion:
                result["fusion_confidence"] = (
                    dict(self._last_fusion_stats) if result["success"] else {}
                )
            result["elapsed_ms"] = (time.monotonic() - frame_started) * 1000
            self.diagnostics.append(result)
            if not result["success"] and self._tracking_lost_frames == 0:
                self._lost_at_index = i
                self._recovery_preview = None
            if result["success"]:
                self._lost_at_index = None
                self._recovery_preview = None
            self._tracking_lost_frames = (
                0 if result["success"] else self._tracking_lost_frames + 1
            )
            result["fusion_paused"] = bool(self._tracking_lost_frames)
            result["last_tracked_index"] = self.poses[-1][0] if self.poses else None
            processed += 1
            if not result["success"]:
                errors += 1
            if progress_cb:
                progress_cb(i + 1, total, result)

        return {
            "processed": processed,
            "errors": errors,
            "skipped_count": sum(not r["success"] for r in self.diagnostics),
            "total": total,
            "frame_count": self.frame_count,
        }

    def _process_single_frame(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
        """Process a single frame through ICP registration + TSDF integration."""
        t0 = time.monotonic()

        with self._stage("depth_filter"):
            rgb, depth = prepare_rgbd(rgb, depth, self.settings)
        valid_fraction = np.count_nonzero(depth) / depth.size
        if np.count_nonzero(depth) < 1000:
            return {
                "success": False,
                "message": "Too few valid depth pixels after clipping",
            }
        # Legacy RGBD + registration point cloud (CPU)
        with self._stage("registration_cloud"):
            rgbd = self._make_rgbd(rgb, depth)
            current_pcd = self._make_reg_pcd(rgbd)

        if len(current_pcd.points) < 100:
            return {
                "success": False,
                "frame_count": self.frame_count,
                "message": "Too few depth points. Skipped.",
            }

        # First frame — integrate directly (need model_pcd immediately)
        if self.frame_count == 0:
            # A dominant plane provides an estimated up direction for the map.
            # It is an orientation aid, not a measured gravity sensor.
            plane, inliers = current_pcd.segment_plane(0.015, 3, 100)
            normal = np.asarray(plane[:3])
            if len(inliers) >= 100 and abs(normal[1]) > 0.3:
                self._world_up = normal * (-1 if normal[1] > 0 else 1)
                self._up_estimated = True
            extrinsic = np.linalg.inv(self.cumulative_T)
            with self._stage("fusion"):
                self._integrate_vbg(rgb, depth, extrinsic)
            with self._stage("model_refresh"):
                self._extract_model_pcd()
            self.frame_count = 1
            self._last_rgbd = rgbd
            self._last_reg_pcd = current_pcd
            self.poses.append((self._processed_count, self.cumulative_T.copy()))
            elapsed_ms = (time.monotonic() - t0) * 1000
            return {
                "success": True,
                "frame_count": 1,
                "message": f"Reference frame ({elapsed_ms:.0f}ms)",
                "pose": self.cumulative_T.tolist(),
                "valid_depth_fraction": valid_fraction,
            }

        # Register current → model
        try:
            with self._stage("tracking"):
                result, method = self._register(current_pcd, rgbd)
        except Exception as e:
            return {
                "success": False,
                "frame_count": self.frame_count,
                "message": f"Registration failed: {e}",
            }

        if result is None:
            return {
                "success": False,
                "frame_count": self.frame_count,
                "message": method,
            }

        independently_verified = method == "keyframe+visual" and self._visual_evidence is not None
        if method not in ("anchor+icp", "appearance+icp") and not independently_verified:
            with self._stage("tracking_verification"):
                error = self._verify_tracking_transition(current_pcd, result)
            if error is not None:
                return {"success": False, "frame_count": self.frame_count, "message": error}

        # Commit tracking state only after integration succeeds.
        pose = result.transformation
        with self._stage("fusion"):
            self._integrate_vbg(rgb, depth, np.linalg.inv(pose))
        self.cumulative_T = pose
        self._last_rgbd = rgbd
        self._last_reg_pcd = current_pcd
        self.frame_count += 1
        self._integrations_since_model += 1
        self.poses.append((self._processed_count, pose.copy()))
        if self._visual_evidence is not None:
            self.tracking_edges.append({"source_index": self._processed_count,
                                        **self._visual_evidence})

        # Refresh model periodically
        should_extract = (
            method in ("global", "appearance+icp")
            or self._integrations_since_model >= self.MODEL_REFRESH_INTERVAL
        )
        if should_extract:
            with self._stage("model_refresh"):
                self._extract_model_pcd()

        elapsed_ms = (time.monotonic() - t0) * 1000

        if method == "global":
            msg = (
                f"Frame {self.frame_count} via global "
                f"(fitness={result.fitness:.3f}, {elapsed_ms:.0f}ms)"
            )
        else:
            msg = (
                f"Frame {self.frame_count} "
                f"(fitness={result.fitness:.3f}, "
                f"RMSE={result.inlier_rmse:.4f}, {elapsed_ms:.0f}ms)"
            )

        return {
            "success": True,
            "frame_count": self.frame_count,
            "message": msg,
            "fitness": float(result.fitness),
            "rmse_m": float(result.inlier_rmse),
            "valid_depth_fraction": valid_fraction,
            "pose": self.cumulative_T.tolist(),
            "method": method,
            "visual_evidence": self._visual_evidence,
        }

    def extract_preview(self, progress_cb=None):
        """Process unprocessed frames, then extract current mesh/point cloud.

        Returns (mesh, point_cloud, process_result) or (None, None, result).
        """
        process_result = self.process_frames(progress_cb=progress_cb)
        if self.frame_count == 0:
            return None, None, process_result
        try:
            t_mesh = self.vbg.extract_triangle_mesh(weight_threshold=0.5)
            mesh = t_mesh.to_legacy()
            del t_mesh
            mesh.compute_vertex_normals()
            t_pcd = self.vbg.extract_point_cloud(weight_threshold=0.5)
            pcd = t_pcd.to_legacy()
            del t_pcd
            return mesh, pcd, process_result
        except Exception:
            traceback.print_exc()
            return None, None, process_result

    def _reconnect_volume(self, progress_cb=None):
        """Commit connected raw fragments only after a complete fresh fusion."""
        if self._reconnection_count == self.stored_count:
            return not self.fragment_reconnection.get("failed", False)
        from .fragments import propose_fragment_poses

        started = time.monotonic()
        report = {"applied": False}
        try:
            proposals, report = propose_fragment_poses(self, progress_cb)
            if proposals is not None:
                candidate = copy.copy(self)
                candidate._fusion_block_limit = self.settings.final_block_count
                required = self._required_fusion_blocks(proposals, progress_cb)
                report.update(fusion_required_blocks=required,
                              fusion_block_limit=candidate._fusion_block_limit,
                              fusion_voxel_m=self.voxel_size)
                if required > candidate._fusion_block_limit:
                    raise ValueError(
                        f"Verified reconstruction needs {required} blocks at "
                        f"{self.voxel_size * 1000:g} mm; increase the final block "
                        f"budget from {candidate._fusion_block_limit} to at least {required}"
                    )
                candidate.vbg = self._create_vbg(block_count=max(1, required))
                for completed, (index, pose) in enumerate(proposals, 1):
                    rgb, depth = prepare_rgbd(*self.raw_frames[index], self.settings)
                    candidate._integrate_vbg(rgb, depth, np.linalg.inv(pose))
                    if progress_cb:
                        progress_cb(completed, len(proposals), {
                            "stage": "fragment_reconnection",
                            "message": f"Fusing connected fragments {completed}/{len(proposals)} views",
                        })
                candidate._extract_model_pcd()
                if candidate.model_pcd is None or len(candidate.model_pcd.points) < 100:
                    raise ValueError("Reconnected volume has insufficient geometry")
                diagnostics = [dict(result) for result in self.diagnostics]
                connected = {i for i, _ in proposals}
                fragment_by_frame = {i: f["id"] for f in report["fragments"] for i in f["frame_indices"]}
                for result in diagnostics:
                    if result["success"] and result["index"] not in connected:
                        result.update(success=False, excluded_offline=True,
                                      message_before_reconnection=result.get("message"),
                                      pose_before_reconnection=result.get("pose"),
                                      message="Excluded: no verified connection to the anchored reconstruction")
                        result.pop("pose", None)
                for index, pose in proposals:
                    result = diagnostics[index]
                    if result.get("pose") is not None:
                        result["pose_before_reconnection"] = result["pose"]
                    if not result["success"]:
                        result.update(success=True, recovered_offline=True, method="fragment+graph",
                                      message_before_reconnection=result.get("message"),
                                      message="Recovered through verified fragment registration")
                    result.update(pose=pose.tolist(), fragment_id=fragment_by_frame.get(index))
                count, lost_at, last_tracked = 0, None, None
                for result in diagnostics:
                    count += bool(result["success"])
                    if result["success"]:
                        last_tracked = result["index"]
                    lost_at = None if result["success"] else result["index"] if lost_at is None else lost_at
                    result.update(frame_count=count, fusion_paused=not result["success"],
                                  last_tracked_index=last_tracked)
                last_index, last_pose = proposals[-1]
                last_rgbd = self._make_rgbd(*prepare_rgbd(*self.raw_frames[last_index], self.settings))
                report.update(applied=True, reason=(f"Reconnected {report['recovered_frames']} views; "
                              f"{report['corrected_frames']} poses corrected, {len(report['excluded_frames'])} excluded"),
                              fusion_blocks=int(candidate.vbg.hashmap().size()),
                              fusion_block_limit=candidate._fusion_block_limit,
                              fusion_voxel_m=self.voxel_size)
                # Native work has succeeded; commit the candidate as one state.
                self.original_poses = [(i, p.copy()) for i, p in self.poses]
                for name in ("vbg", "model_pcd", "_live_points", "_live_colors", "_model_feature_cloud",
                             "_model_fpfh", "_model_pyramid", "_tensor_model_pyramid", "_integrations_since_model"):
                    setattr(self, name, getattr(candidate, name))
                self.poses, self.diagnostics = proposals, diagnostics
                self._pose_seeds_only = False
                self.frame_count = len(proposals)
                self.cumulative_T = last_pose.copy()
                self._last_rgbd = last_rgbd
                # Offline reconnection can change the last accepted observation.
                self._last_reg_pcd = None
                self._tracking_lost_frames = self._processed_count - last_index - 1
                self._lost_at_index = lost_at
                self._recovery_preview = None
                self._refined_count = None
                self._final_vbg = None
            self.fragment_reconnection = report
            self._reconnection_count = self.stored_count
        except Exception as exc:
            logger.exception("Fragment reconnection failed; original reconstruction retained")
            report.update(applied=False, failed=True, reason=f"Reconnection failed: {exc}")
            self.fragment_reconnection = report
        elapsed = (time.monotonic() - started) * 1000
        self.fragment_reconnection["elapsed_ms"] = elapsed
        self.stage_totals_ms["fragment_reconnection"] = self.stage_totals_ms.get("fragment_reconnection", 0) + elapsed
        return not self.fragment_reconnection.get("failed", False)

    def _required_fusion_blocks(self, proposals, progress_cb=None):
        """Measure allocation before touching a candidate or existing volume.

        A one-block grid computes the exact frustum coordinates without voxel
        activation. The configured block limit remains a hard memory budget.
        """
        scratch = self._create_vbg(block_count=1)
        blocks = set()
        for completed, (index, pose) in enumerate(proposals, 1):
            _, depth = prepare_rgbd(*self.raw_frames[index], self.settings)
            image = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth))).to(self.device)
            coordinates = scratch.compute_unique_block_coordinates(
                image, self.intrinsic_tensor,
                o3c.Tensor(np.linalg.inv(pose), dtype=o3c.float64),
                depth_scale=1000.0, depth_max=self.max_depth_m,
                trunc_voxel_multiplier=self.sdf_trunc / self.voxel_size,
            ).cpu().numpy()
            blocks.update(map(tuple, coordinates))
            if progress_cb:
                progress_cb(completed, len(proposals), {
                    "stage": "fragment_reconnection",
                    "message": f"Planning verified fusion {completed}/{len(proposals)} views: {len(blocks)} blocks",
                })
        return len(blocks)

    def _refine_volume(self):
        """Transactional refinement: never rewrite poses in an existing TSDF."""
        if self._refined_count == self.stored_count:
            return
        from .refinement import propose_poses

        started = time.monotonic()
        try:
            proposals, report = propose_poses(self)
            if proposals is not None:
                candidate = copy.copy(self)
                candidate.vbg = self._create_vbg()
                for index, pose in proposals:
                    rgb, depth = self.raw_frames[index]
                    rgb, depth = prepare_rgbd(rgb, depth, self.settings)
                    candidate._integrate_vbg(rgb, depth, np.linalg.inv(pose))
                candidate._extract_model_pcd()
                if candidate.model_pcd is None or len(candidate.model_pcd.points) < 100:
                    raise ValueError("Refined volume has insufficient geometry")
                diagnostics = [dict(result) for result in self.diagnostics]
                for index, pose in proposals:
                    diagnostics[index]["pose_before_refinement"] = diagnostics[
                        index
                    ].get("pose")
                    diagnostics[index]["pose"] = pose.tolist()
                original_poses = [(i, p.copy()) for i, p in self.poses]
                cumulative = proposals[-1][1].copy()
                report.update(
                    applied=True,
                    reason="Validated poses committed after fresh TSDF fusion",
                )
                # All native work and report preparation succeeded. Commit.
                self._final_vbg = None
                if self.original_poses is None:
                    self.original_poses = original_poses
                for name in (
                    "vbg",
                    "model_pcd",
                    "_live_points",
                    "_live_colors",
                    "_model_feature_cloud",
                    "_model_fpfh",
                    "_model_pyramid",
                    "_tensor_model_pyramid",
                    "_integrations_since_model",
                ):
                    setattr(self, name, getattr(candidate, name))
                self.poses = proposals
                self.cumulative_T = cumulative
                self.diagnostics = diagnostics
            self.refinement = report
        except Exception as exc:
            logger.exception("Pose refinement failed; original reconstruction retained")
            self.refinement = {"applied": False, "reason": f"Refinement failed: {exc}"}
        self.refinement["elapsed_ms"] = (time.monotonic() - started) * 1000
        self.stage_totals_ms["final_refinement"] = (
            self.stage_totals_ms.get("final_refinement", 0)
            + self.refinement["elapsed_ms"]
        )
        self._refined_count = self.stored_count

    def _final_volume(self, progress_cb=None):
        """Fresh bounded fusion of accepted poses, leaving live tracking intact."""
        if self.settings.final_voxel_m is None:
            self.final_reconstruction = {
                "applied": False,
                "reason": "Uses live volume",
                "voxel_m": self.voxel_size,
            }
            return self.vbg
        if self._final_vbg is not None:
            return self._final_vbg
        started = time.monotonic()
        candidate = copy.copy(self)
        candidate.settings = replace(self.settings, voxel_m=self.settings.final_voxel_m)
        candidate.voxel_size = candidate.settings.voxel_m
        candidate._fusion_block_limit = self.settings.final_block_count
        candidate.vbg = candidate._create_vbg(block_count=candidate._fusion_block_limit)
        for completed, (index, pose) in enumerate(self.poses, 1):
            rgb, depth = prepare_rgbd(*self.raw_frames[index], self.settings)
            candidate._integrate_vbg(rgb, depth, np.linalg.inv(pose))
            if progress_cb:
                progress_cb(
                    completed,
                    len(self.poses),
                    {
                        "message": f"Final fusion {completed}/{len(self.poses)} accepted views"
                    },
                )
        elapsed = (time.monotonic() - started) * 1000
        self.stage_totals_ms["final_reintegration"] = (
            self.stage_totals_ms.get("final_reintegration", 0) + elapsed
        )
        self.final_reconstruction = {
            "applied": False,
            "reason": "Awaiting final surface validation",
            "voxel_m": candidate.voxel_size,
            "blocks": int(candidate.vbg.hashmap().size()),
            "block_limit": candidate._fusion_block_limit,
            "attribute_budget_mib": candidate._fusion_block_limit * 4096 * 20 / 2**20,
            "elapsed_ms": elapsed,
        }
        return candidate.vbg

    def build_mesh(self, progress_cb=None) -> tuple[bool, dict]:
        """Build transactionally; a failed final build preserves existing geometry."""
        process_result = self.process_frames(progress_cb=progress_cb)
        if self.frame_count == 0:
            return False, process_result
        if self.settings.reconnect_fragments:
            if not self._reconnect_volume(progress_cb):
                process_result.update(success=False, message=self.fragment_reconnection["reason"],
                                      fragment_reconnection=self.fragment_reconnection)
                return False, process_result
        else:
            self.fragment_reconnection = {"applied": False, "reason": "Not requested"}
        process_result.update(frame_count=self.frame_count,
                              skipped_count=sum(not d["success"] for d in self.diagnostics),
                              fragment_reconnection=self.fragment_reconnection)
        if self.settings.refine_poses:
            self._refine_volume()
        else:
            self.refinement = {"applied": False, "reason": "Not requested"}
        process_result["refinement"] = self.refinement
        try:
            volume = self._final_volume(progress_cb)
            mesh = volume.extract_triangle_mesh(
                weight_threshold=self.settings.final_weight
            ).to_legacy()
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            if len(mesh.triangles) and self.settings.min_component_triangles:
                labels, sizes, _ = mesh.cluster_connected_triangles()
                mask = (
                    np.asarray(sizes)[np.asarray(labels)]
                    < self.settings.min_component_triangles
                )
                mesh.remove_triangles_by_mask(mask)
                mesh.remove_unreferenced_vertices()
            if not len(mesh.triangles):
                raise ValueError("No confident surface: capture more overlapping views")
            if mesh.has_vertex_colors():
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.clip(np.asarray(mesh.vertex_colors), 0, 1)
                )
            mesh.compute_vertex_normals()
            pcd = volume.extract_point_cloud(
                weight_threshold=self.settings.final_weight
            ).to_legacy()
            # Commit only after fusion and both native extractions succeed.
            self.mesh, self.point_cloud = mesh, pcd
            if self.settings.final_voxel_m is not None:
                self._final_vbg = volume
                self.final_reconstruction.update(
                    applied=True, reason="Fresh bounded final fusion"
                )
            process_result["final_reconstruction"] = dict(self.final_reconstruction)
            return True, process_result
        except Exception as exc:
            logger.exception("Final build failed; live volume retained")
            self.final_reconstruction = {"applied": False, "reason": str(exc)}
            process_result["final_reconstruction"] = dict(self.final_reconstruction)
            process_result["message"] = str(exc)
            return False, process_result

    def export_ply(self, filepath: str, as_mesh: bool = True) -> bool:
        try:
            if as_mesh and self.mesh is not None:
                o3d.io.write_triangle_mesh(filepath, self.mesh)
            elif self.point_cloud is not None:
                o3d.io.write_point_cloud(filepath, self.point_cloud)
            else:
                return False
            return True
        except Exception:
            traceback.print_exc()
            return False

    def export_obj(self, filepath: str) -> bool:
        if self.mesh is None:
            return False
        try:
            verts = np.asarray(self.mesh.vertices)
            faces = np.asarray(self.mesh.triangles)
            colors = np.asarray(self.mesh.vertex_colors)
            tm = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_colors=(colors * 255).astype(np.uint8)
                if len(colors) > 0
                else None,
            )
            tm.export(filepath)
            return True
        except Exception:
            traceback.print_exc()
            return False

    def shutdown(self):
        """Clean up resources. Call on application exit."""
