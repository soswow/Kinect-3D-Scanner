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

from shared.calibration import prepare_metric_depth, prepare_rgbd
from shared.capture import RGB_DEPTH_ASSISTANCE_LIMIT_MS, RGB_DEPTH_CAPTURE_LIMIT_MS
from shared.config import LIVE_MAX_POINTS, PRESET_DEFAULT, ScanPreset
from shared.settings import ScanSettings

from .backend import select_backend
from .tracking_cache import TargetPyramid, reuse_icp_source
from .cuda_registration import registration_scope
from .cuda_fusion import FusionUpdateError
from .cuda_input import CudaInputError, InputPreparation
from .cuda_confidence import ConfidencePreparation
from .fusion_memory import plan_fusion

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
        self._discard_project_archive()
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
        self._input_preparation = InputPreparation(self.device, self.backend)
        self._confidence_preparation = ConfidencePreparation(self.device, self.backend)
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

        self.fusion_failure = None
        self.input_failure = None
        self._live_fusion_generation = 0
        self.frame_count = 0
        self.session_id = uuid.uuid4().hex
        self.stage_totals_ms = {}
        self._frame_timings = {}
        self._stage_children = []
        self.refinement = {"applied": False, "reason": "Not requested"}
        self.bundle_adjustment = {"applied": False, "reason": "Not requested"}
        self._bundle_count = None
        self.original_poses = None
        self._refined_count = None
        self._reconnection_count = None
        self._pose_seeds_only = False
        self._project_processing_paused = False
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
        self._integrations_since_preview = 0
        self._pending_model_cloud = None
        self._pending_model_frame_count = None
        model_refresh = os.environ.get("KINECT_MODEL_REFRESH", "eager").lower()
        if model_refresh not in ("eager", "lazy"):
            raise ValueError("KINECT_MODEL_REFRESH must be eager or lazy")
        self.backend["model_preparation"] = model_refresh
        cache_mode = os.environ.get("KINECT_KEYFRAME_CACHE", "off").lower()
        if cache_mode not in ("on", "off"):
            raise ValueError("KINECT_KEYFRAME_CACHE must be on or off")
        self.backend["keyframe_pyramid_cache"] = cache_mode
        recovery_mode = os.environ.get("KINECT_LIVE_RECOVERY", "full").lower()
        if recovery_mode not in ("full", "deferred"):
            raise ValueError("KINECT_LIVE_RECOVERY must be full or deferred")
        self.backend["live_recovery"] = recovery_mode
        feature_method = os.environ.get("KINECT_VISUAL_FEATURES", "orb").lower()
        from .adaptive_visual import select_policy
        self.backend.update(select_policy(p, feature_method,
            os.environ.get("KINECT_ADAPTIVE_EXPERIMENTAL", "off").lower()))
        self.backend["sift_fallback"] = {"attempts": 0, "verified": 0}
        visual_refinement = os.environ.get("KINECT_VISUAL_REFINEMENT", "icp").lower()
        if visual_refinement not in ("icp", "measured", "measured-fine"):
            raise ValueError("KINECT_VISUAL_REFINEMENT must be icp, measured, or measured-fine")
        self.backend["visual_refinement"] = visual_refinement
        self.backend["stage_devices"]["measured_visual_refinement"] = "CPU:0"
        final_visual_first = os.environ.get("KINECT_FINAL_VISUAL_FIRST", "off").lower()
        if final_visual_first not in ("off", "on"):
            raise ValueError("KINECT_FINAL_VISUAL_FIRST must be off or on")
        self.backend["final_visual_first"] = final_visual_first
        final_local_refinement = os.environ.get("KINECT_FINAL_LOCAL_REFINEMENT", "icp").lower()
        if final_local_refinement not in ("icp", "measured"):
            raise ValueError("KINECT_FINAL_LOCAL_REFINEMENT must be icp or measured")
        self.backend["final_local_refinement"] = final_local_refinement
        self.backend["visual_refinement_statistics"] = {"attempts": 0, "icp_fallbacks": 0}
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
        self._sift_visual_cache = {}
        self._visual_target_pyramids = {}
        self._visual_rgbd_cache = {}
        self._cuda_rgbd_odometry = None
        self._cuda_descriptor_matcher = None
        self._cuda_final_descriptor_matcher = None
        self.backend["projective_statistics"] = {"visual_attempts": 0, "visual_icp_fallbacks": 0,
                                                  "recovery_attempts": 0, "recovery_cpu_fallbacks": 0}
        self.backend["descriptor_matching"].update(cuda_batches=0, cpu_batches=0, final_cuda_batches=0)
        self._visual_evidence = None
        self.tracking_edges = []
        self._tracking_lost_frames = 0
        self._lost_at_index = None
        self._recovery_preview = None
        self._world_up = np.array([0.0, -1.0, 0.0])
        self._up_estimated = False
        self._last_fusion_stats = {}

    # ── helpers ────────────────────────────────────────────────────────

    def _prepare_input(self, rgb, raw, settings=None, *, cpu_prepare=None):
        return self._input_preparation.prepare(
            rgb, raw, self.settings if settings is None else settings,
            prepare_rgbd if cpu_prepare is None else cpu_prepare)

    @contextmanager
    def _stage(self, name):
        # CUDA kernels are asynchronous; synchronize to report real wall time.
        if str(self.device).startswith("CUDA"):
            o3c.cuda.synchronize()
        start = time.perf_counter()
        self._stage_children.append(0.0)
        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                if str(self.device).startswith("CUDA"):
                    o3c.cuda.synchronize()
            except Exception as exc:
                if isinstance(body_error, (FusionUpdateError, CudaInputError)):
                    # A failing cleanup must not disguise a partial update or
                    # hard input failure as an ordinary rejected registration.
                    logger.warning("Synchronization also failed after CUDA processing failure: %s", exc)
                elif name == "fusion":
                    raise FusionUpdateError(f"CUDA fusion synchronization failed: {exc}") from exc
                else:
                    raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
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
            "fusion_failure": self.fusion_failure,
            "input_failure": self.input_failure,
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
            "bundle_adjustment": self.bundle_adjustment,
            "final_reconstruction": self.final_reconstruction,
            "fragment_reconnection": self.fragment_reconnection,
            "tracking_edges": list(self.tracking_edges),
            "tracking": {
                "state": "error" if self.fusion_failure or self.input_failure else "recovering" if self._tracking_lost_frames else "tracking" if self.poses else "waiting",
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
        if self.fusion_failure is not None:
            guidance = self.fusion_failure["message"]
        elif self.input_failure is not None:
            guidance = self.input_failure["message"]
        elif self.mesh is not None:
            excluded = len(self.fragment_reconnection.get("excluded_frames", []))
            guidance = f"Final model retains {self.frame_count} of {self.stored_count} captures."
            if excluded:
                guidance += f" {excluded} previously fused views were removed because their positions could not be verified."
            loops = len(self.fragment_reconnection.get("loop_closures", [])) + (self.refinement.get("loops", 0) if self.refinement.get("applied") else 0)
            guidance += f" {loops} verified loop constraint{'s' if loops != 1 else ''}."
            guidance += " Save Session preserves all captured views."
        elif result and not result.get("success"):
            guidance = (
                "View retained for Finish. Live geometry includes only verified views; maintain overlap while capturing."
                if self.backend["live_recovery"] == "deferred" else
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
            "geometry_frame_count": self.frame_count - self._integrations_since_preview,
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
            from shared.inertial import rotate_display
            rotation = self.frame_metadata[anchor[0]].get("orientation", {}).get("rotation_cw_degrees", 0)
            if rotation in (90, 180, 270):
                rgb = rotate_display(rgb, rotation)
            height = max(1, round(rgb.shape[0] * 240 / rgb.shape[1]))
            preview = cv2.resize(rgb, (240, height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if ok:
                self._recovery_preview = base64.b64encode(encoded).decode("ascii")
        return {
            "tracking_state": "error" if self.fusion_failure or self.input_failure else "recovering" if lost else "tracking" if anchor else "waiting",
            "fusion_paused": lost or self.fusion_failure is not None or self.input_failure is not None,
            "volume_requires_reset": self.fusion_failure is not None,
            "fusion_failure": self.fusion_failure,
            "input_requires_reset": self.input_failure is not None,
            "input_failure": self.input_failure,
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
                    "Fusion extent changed after allocation planning; "
                    "the previous reconstruction is retained. Retry the build."
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

    def _make_reg_pcd(self, rgbd, *, normals=True):
        """Build a coarser point cloud for registration (not integration)."""
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd = pcd.voxel_down_sample(self.reg_voxel)
        if normals:
            self._ensure_raw_normals(pcd)
        return pcd

    def _ensure_raw_normals(self, cloud):
        if cloud is not None and not cloud.has_normals():
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.reg_voxel * 4, max_nn=30))

    def _refresh_live_points(self):
        """Extract the current surface once; defer registration levels if unused."""
        t_pcd = self.vbg.extract_point_cloud(
            weight_threshold=0.01 if self.settings.confidence_fusion else 0.5)
        pcd = t_pcd.to_legacy()
        self._pending_model_cloud = pcd
        self._pending_model_frame_count = self.frame_count
        points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        step = max(1, int(np.ceil(len(points) / LIVE_MAX_POINTS)))
        self._live_points = points[::step].astype(np.float32)
        self._live_colors = colors[::step].astype(np.float32)
        self._integrations_since_preview = 0

    def _extract_model_pcd(self, *, from_snapshot=False):
        """Extract and downsample a model point cloud from the TSDF volume.

        Cache registration scales; compute FPFH lazily if recovery needs it.
        """
        if not from_snapshot or self._pending_model_cloud is None:
            self._refresh_live_points()
        pcd = self._pending_model_cloud
        prepared_count = self._pending_model_frame_count
        self._pending_model_cloud = None
        self._pending_model_frame_count = None
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

        self._integrations_since_model = self.frame_count - prepared_count

    def _icp(self, source, target, init=None, *, fine_only=False):
        """Coarse-to-fine robust point-to-plane tracking in camera-to-world space."""
        if not fine_only:
            self._ensure_raw_normals(target)
        pose = self.cumulative_T if init is None else init
        pyramid = getattr(self, "_icp_source_pyramid", None)
        if pyramid is not None and pyramid.source is not source:
            pyramid = None
        target_pyramid = None
        for index, (_, cloud) in (self._visual_cache.items()
                                  if self.backend["keyframe_pyramid_cache"] == "on" else ()):
            if cloud is target:
                target_pyramid = self._visual_target_pyramids.get(index)
                if target_pyramid is None or target_pyramid.source is not target:
                    target_pyramid = self._visual_target_pyramids[index] = TargetPyramid(target)
                break
        for scale, iterations in (((1, 20),) if fine_only else ((4, 40), (2, 30), (1, 20))):
            voxel = self.reg_voxel * scale
            src = (source.voxel_down_sample(voxel) if pyramid is None
                   else pyramid.level(voxel))
            if target is self.model_pcd:
                tgt = self._model_pyramid[scale]
            elif target_pyramid is not None:
                tgt = target_pyramid.level(voxel)
            else:
                tgt = target.voxel_down_sample(voxel)
                tgt.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30)
                )
            if self.backend["tracking"] == "tensor":
                def source_tensor_level(level):
                    return o3d.t.geometry.PointCloud.from_legacy(
                        level, dtype=o3c.float32, device=self.device)

                source_tensor = (source_tensor_level(src) if pyramid is None
                                 else pyramid.tensor(voxel, source_tensor_level))
                target_tensor = (
                    self._tensor_model_pyramid[scale]
                    if target is self.model_pcd
                    else target_pyramid.tensor(voxel, source_tensor_level)
                    if target_pyramid is not None
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
            self.settings.camera, method="sift",
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
                rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
                self._appearance_cache[index] = extract_features(
                    rgb, depth, self.settings.camera, method="sift"
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
            rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
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
        from shared.inertial import gravity_seed

        seed = self._motion_initial_guess()
        if self.settings.gravity_assistance and self.poses and 0 <= self._processed_count < len(self.frame_metadata):
            seed, report = gravity_seed(seed, self.cumulative_T,
                                        self.frame_metadata[self.poses[-1][0]],
                                        self.frame_metadata[self._processed_count])
            self.frame_metadata[self._processed_count]["gravity_tracking"] = report
        return seed

    def _motion_initial_guess(self):
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
        if self.backend["visual_policy"] == "orb_then_sift":
            from .adaptive_visual import register

            return register(self, source, rgbd, self._visual_register_primary)
        return self._visual_register_primary(source, rgbd)

    def _visual_register_primary(self, source, rgbd):
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
                                    self.settings.camera, method=self.backend["visual_features"])
        recent = self.poses[-8:]
        # Stable landmarks avoid evicting/repreparing an evenly resampled bank
        # on every accepted frame. Keep the initial five views for a returning
        # loop, plus 27 spaced landmarks and the recent eight raw observations.
        historical = self.poses[:5] + self.poses[::4][-27:]
        candidates = {i: pose for i, pose in historical + recent}
        self._visual_cache = {i: value for i, value in self._visual_cache.items() if i in candidates}
        self._visual_target_pyramids = {i: value for i, value in self._visual_target_pyramids.items() if i in candidates}
        self._visual_rgbd_cache = {i: value for i, value in self._visual_rgbd_cache.items() if i in candidates}
        ranked = []
        prepared = []
        for target_index, target_pose in candidates.items():
            lag = self.frame_metadata[target_index].get("rgb_depth_delta_ms")
            if lag is not None and abs(lag) > RGB_DEPTH_ASSISTANCE_LIMIT_MS:
                continue
            if target_index not in self._visual_cache:
                rgb, depth = self._prepare_input(*self.raw_frames[target_index], self.settings)
                self._visual_cache[target_index] = (
                    extract_features(rgb, depth, self.settings.camera, method=self.backend["visual_features"]), None)
            target_features, target = self._visual_cache[target_index]
            prepared.append((target_index, target_pose, target_features))
        matched_bank = None
        if self.backend["descriptor_matching"]["implementation"] == "cuda":
            if self._cuda_descriptor_matcher is None:
                from .cuda_matching import Matcher

                self._cuda_descriptor_matcher = Matcher(int(str(self.device).split(":")[1]))
            matched_bank = self._cuda_descriptor_matcher.match(features, [row[2] for row in prepared])
        if matched_bank is None:
            self.backend["descriptor_matching"]["cpu_batches"] += 1
            matched_bank = [correspondences(features, row[2]) for row in prepared]
        else:
            self.backend["descriptor_matching"]["cuda_batches"] += 1
        for (target_index, target_pose, target_features), matches in zip(prepared, matched_bank):
            if len(matches) >= 40:
                ranked.append((len(matches), target_index, target_pose, matches))
        # Cheap descriptor retrieval searches the whole bounded keyframe bank;
        # only the five best candidates pay for point-cloud registration.
        for _, target_index, target_pose, matches in sorted(ranked, key=lambda row: (-row[0], -row[1]))[:5]:
            target_features, target = self._visual_cache[target_index]
            proposal = propose_transform(features, target_features, self.settings.camera, matches)
            if proposal is None:
                continue
            if target is None:
                rgb, depth = self._prepare_input(*self.raw_frames[target_index], self.settings)
                target_rgbd = self._make_rgbd(rgb, depth)
                target = self._make_reg_pcd(target_rgbd, normals=self._needs_raw_normals())
                self._visual_cache[target_index] = target_features, target
                if self.backend["projective_odometry"] != "off":
                    self._visual_rgbd_cache[target_index] = target_rgbd
            a, b = matches.T
            # Feature identities survive refinement: anonymous geometric overlap
            # cannot move a textured surface to another plausible model location.
            pose = target_pose @ proposal
            translation, angle = motion(np.linalg.inv(self.cumulative_T) @ pose)
            if not self._tracking_lost_frames and (
                    translation > self.settings.max_translation_m or angle > self.settings.max_rotation_deg):
                continue
            fast_pose_applied = False
            if self.backend["visual_refinement"].startswith("measured"):
                from .visual_refinement import measured_pose

                self.backend["visual_refinement_statistics"]["attempts"] += 1
                relative = measured_pose(features, target_features, matches, proposal, self.settings.camera)
                fast_pose_applied = relative is not None
                if not fast_pose_applied:
                    self.backend["visual_refinement_statistics"]["icp_fallbacks"] += 1
                native = (self._icp(source, target, relative, fine_only=True)
                          if relative is not None and self.backend["visual_refinement"] == "measured-fine"
                          else _REG.evaluate_registration(source, target, 0.0225, relative)
                          if relative is not None else self._icp(source, target, proposal))
            elif self.backend["projective_odometry"] != "off":
                self.backend["projective_statistics"]["visual_attempts"] += 1
                target_rgbd = self._visual_rgbd_cache.get(target_index)
                if target_rgbd is None:
                    rgb, depth = self._prepare_input(*self.raw_frames[target_index], self.settings)
                    target_rgbd = self._visual_rgbd_cache[target_index] = self._make_rgbd(rgb, depth)
                relative = self._projective_pose(rgbd, target_rgbd, proposal)
                fast_pose_applied = relative is not None
                if not fast_pose_applied:
                    self.backend["projective_statistics"]["visual_icp_fallbacks"] += 1
                native = (_REG.evaluate_registration(source, target, 0.0225, relative)
                          if relative is not None else self._icp(source, target, proposal))
            else:
                native = self._icp(source, target, proposal)
            correction, correction_angle = motion(np.linalg.inv(proposal) @ native.transformation)
            good, stats = feature_agreement(features.points[a], target_features.points[b],
                                           target_features.pixels[b], native.transformation,
                                           self.settings.camera)
            if (fast_pose_applied
                    and (not good or correction > 0.03 or correction_angle > 3
                         or native.fitness < 0.6 or native.inlier_rmse > 0.015)):
                if self.backend["visual_refinement"].startswith("measured"):
                    self.backend["visual_refinement_statistics"]["icp_fallbacks"] += 1
                else:
                    self.backend["projective_statistics"]["visual_icp_fallbacks"] += 1
                native = self._icp(source, target, proposal)
                correction, correction_angle = motion(np.linalg.inv(proposal) @ native.transformation)
                good, stats = feature_agreement(features.points[a], target_features.points[b],
                                               target_features.pixels[b], native.transformation, self.settings.camera)
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

    def _needs_raw_normals(self):
        return self.backend["projective_odometry"] == "off" and self.backend["visual_refinement"] == "icp"

    def _projective_pose(self, source, target, initial, *, recovery=False):
        from .cuda_odometry import RGBDOdometry

        if self._cuda_rgbd_odometry is None:
            self._cuda_rgbd_odometry = RGBDOdometry(self.device)
        return self._cuda_rgbd_odometry.estimate(
            source, target, self.intrinsic_tensor, initial, self.max_depth_m,
            "hybrid" if recovery else self.backend["projective_odometry"],
            iterations=(40, 20, 10) if recovery else (20, 10, 5))

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
        def legacy_relative():
            success, relative, information = o3d.pipelines.odometry.compute_rgbd_odometry(
                current, previous, self.intrinsic, guess,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
            return relative if success and np.all(np.isfinite(information)) else None

        from .refinement import motion

        def verify(relative):
            if relative is None:
                return None
            pose = self.cumulative_T @ relative
            refined = self._icp(source_pcd, self.model_pcd, init=pose)
            correction, angle = motion(np.linalg.inv(pose) @ refined.transformation)
            if correction <= 0.03 and angle <= 3 and self._alignment_error(refined, self.model_pcd) is None:
                return refined
            return None

        if self.backend["projective_odometry"] != "off":
            self.backend["projective_statistics"]["recovery_attempts"] += 1
            verified = verify(self._projective_pose(rgbd, self._last_rgbd, guess, recovery=True))
            if verified is not None:
                return verified
            self.backend["projective_statistics"]["recovery_cpu_fallbacks"] += 1
        return verify(legacy_relative())

    @reuse_icp_source
    def _register(self, source_pcd, rgbd=None):
        """Register source (camera coords) against model_pcd (world coords).

        Uses cumulative_T as initial guess so ICP starts near the true pose.
        Returns (result, method_str) or (None, error_str).
        """
        self._visual_evidence = None
        visual = self._visual_register(source_pcd, rgbd)
        if visual is not None:
            return visual, "keyframe+visual"
        if self.backend["live_recovery"] == "deferred" and self.settings.color_recovery:
            return None, "No verified visual alignment; raw view retained for Finish recovery"
        if self._pending_model_cloud is not None:
            with self._stage("model_refresh"):
                self._extract_model_pcd(from_snapshot=True)
        self._ensure_raw_normals(source_pcd)
        self._ensure_raw_normals(self._last_reg_pcd)
        if self._tracking_lost_frames:
            # ICP against a large accumulated model can snap onto another side
            # of a box. Resume only after verification against the last actual
            # camera observations, or verified appearance relocalization.
            recovered = self._recover_anchor(source_pcd)
            if recovered is not None:
                return recovered, "anchor+icp"
            try:
                recovered = self._relocalize(source_pcd, rgbd)
                if recovered is not None:
                    return recovered, "appearance+icp"
            except CudaInputError:
                raise
            except (RuntimeError, ValueError):
                logger.debug("Appearance relocalization failed", exc_info=True)
            return None, "Tracking lost; match the last good view to resume fusion"
        if self.settings.color_recovery:
            try:
                recovered = self._color_recovery(source_pcd, rgbd)
                if recovered is not None:
                    return recovered, "rgbd+icp"
            except CudaInputError:
                raise
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
        except CudaInputError:
            raise
        except (RuntimeError, ValueError):
            logger.debug("Color recovery failed", exc_info=True)
        # Global recovery is subject to the same confidence and motion limits.
        try:
            recovered = self._fpfh_fallback(source_pcd, self.model_pcd)
            if self._alignment_error(recovered, self.model_pcd) is None:
                return recovered, "global"
        except CudaInputError:
            raise
        except Exception:
            logger.debug("Global recovery failed", exc_info=True)
        try:
            recovered = self._relocalize(source_pcd, rgbd)
            if recovered is not None:
                return recovered, "appearance+icp"
        except CudaInputError:
            raise
        except (RuntimeError, ValueError):
            logger.debug("Appearance relocalization failed", exc_info=True)
        return None, error

    def _recover_anchor(self, source):
        """Require nearby, reciprocal overlap with up to five accepted raw views."""
        from .refinement import _match, _trustworthy, motion

        if self._last_rgbd is None or not self.poses:
            return None
        if self._last_reg_pcd is None:
            self._last_reg_pcd = self._make_reg_pcd(self._last_rgbd)
        for index, anchor_pose in reversed(self.poses[-5:]):
            if index == self.poses[-1][0]:
                target = self._last_reg_pcd
            else:
                cached = self._visual_cache.get(index)
                target = cached[1] if cached is not None else None
                if target is None:
                    rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
                    target = self._make_reg_pcd(self._make_rgbd(rgb, depth))
            self._ensure_raw_normals(target)
            forward = _match(source, target, np.eye(4))
            if not _trustworthy(forward, target):
                continue
            translation, angle = motion(forward.transformation)
            if translation > min(0.15, self.settings.max_translation_m) or angle > min(
                15, self.settings.max_rotation_deg
            ):
                continue
            reverse = _match(target, source, np.linalg.inv(forward.transformation))
            translation, angle = motion(reverse.transformation @ forward.transformation)
            if not _trustworthy(reverse, source) or translation > 0.01 or angle > 2:
                continue
            if self._integrations_since_model:
                with self._stage("model_refresh"):
                    self._extract_model_pcd()
            observed = anchor_pose @ forward.transformation
            result = self._icp(source, self.model_pcd, observed)
            translation, angle = motion(np.linalg.inv(observed) @ result.transformation)
            if (result.fitness >= 0.6 and translation <= 0.03 and angle <= 3
                    and self._alignment_error(result, self.model_pcd, False) is None):
                return result
        return None

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
        self._ensure_raw_normals(source)
        self._ensure_raw_normals(target)
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
        self._project_processing_paused = False
        self._stored_monotonic.append(time.monotonic())
        self._final_vbg = None
        self.final_reconstruction = {"applied": False, "reason": "Awaiting final build"}
        self.mesh = None
        self.point_cloud = None
        self._refined_count = None
        self._reconnection_count = None
        self.fragment_reconnection = {"applied": False, "reason": "Awaiting final build"}
        self.refinement = {"applied": False, "reason": "Awaiting final build"}
        self.bundle_adjustment = {"applied": False, "reason": "Awaiting final build"}
        self._bundle_count = None
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
    def _require_safe_volume(self):
        if self.fusion_failure is not None:
            raise FusionUpdateError(self.fusion_failure["message"])
        if self.input_failure is not None:
            raise CudaInputError(self.input_failure["message"])

    @registration_scope
    def process_frames(self, progress_cb=None, max_frames=None) -> dict:
        """Process all unprocessed stored frames through ICP + TSDF.

        This is incremental — only frames after _processed_count are processed.
        progress_cb(current_idx, total, result_dict) is called after each frame.

        Returns summary dict.
        """
        self._require_safe_volume()
        total = len(self.raw_frames)
        start = self._processed_count
        processed = 0
        errors = 0

        end = total if max_frames is None else min(total, start + max_frames)
        for i in range(start, end):
            rgb, depth = self.raw_frames[i]
            self._frame_timings = {}
            frame_started = time.monotonic()
            fusion_generation = self._live_fusion_generation
            try:
                result = self._process_single_frame(rgb, depth)
            except FusionUpdateError as exc:
                logger.exception("CUDA fusion failed; live volume cannot be reused")
                message = ("CUDA fusion failed; the live volume may be partially updated. "
                           "Raw frames are retained. Save Session, then reset and replay the recording.")
                self.fusion_failure = {"index": i, "reason": str(exc), "message": message,
                                       "completed_live_update": self._live_fusion_generation != fusion_generation}
                raise FusionUpdateError(message) from exc
            except CudaInputError as exc:
                if self._live_fusion_generation != fusion_generation:
                    logger.exception("Frame %d input preparation failed after fusion", i)
                    message = ("Reconstruction stopped after fusion because input preparation failed. "
                               "Raw frames are retained. Save Session, then reset and replay the recording.")
                    self.fusion_failure = {"index": i, "reason": str(exc), "message": message,
                                           "completed_live_update": True}
                    raise FusionUpdateError(message) from exc
                logger.exception("Frame %d CUDA preparation failed; processing paused", i)
                message = ("CUDA preparation stopped before fusion: " + str(exc) + ". "
                           "Raw frames are retained. Save Session, then reset and replay the recording. "
                           "For compatibility errors, set KINECT_CUDA_INPUT or KINECT_CUDA_CONFIDENCE "
                           "to auto or off before restarting.")
                self.input_failure = {"index": i, "reason": str(exc), "message": message}
                raise CudaInputError(message) from exc
            except Exception as exc:
                if self._live_fusion_generation != fusion_generation:
                    logger.exception("Frame %d failed after fusion; live volume cannot be reused", i)
                    message = ("Reconstruction stopped after fusion because frame processing failed. "
                               "Raw frames are retained. Save Session, then reset and replay the recording.")
                    self.fusion_failure = {"index": i, "reason": str(exc), "message": message,
                                           "completed_live_update": True}
                    raise FusionUpdateError(message) from exc
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
        t0 = time.perf_counter()

        with self._stage("depth_filter"):
            rgb, depth = self._prepare_input(rgb, depth, self.settings)
        valid_fraction = np.count_nonzero(depth) / depth.size
        if np.count_nonzero(depth) < 1000:
            return {
                "success": False,
                "message": "Too few valid depth pixels after clipping",
            }
        # Legacy RGBD + registration point cloud (CPU)
        with self._stage("registration_cloud"):
            rgbd = self._make_rgbd(rgb, depth)
            current_pcd = self._make_reg_pcd(rgbd, normals=self._needs_raw_normals())

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
            acceleration = self.frame_metadata[self._processed_count].get("accelerometer", {})
            gravity = acceleration.get("gravity", {})
            if acceleration.get("valid") and gravity.get("valid") and gravity.get("calibration_verified") and gravity.get("confidence", 0) >= 0.7:
                up = np.asarray(gravity.get("up_camera"), float)
                if up.shape == (3,) and np.isfinite(up).all() and .99 <= np.linalg.norm(up) <= 1.01:
                    self._world_up = self.cumulative_T[:3, :3] @ up
                    self._up_estimated = True
            extrinsic = np.linalg.inv(self.cumulative_T)
            with self._stage("fusion"):
                self._integrate_vbg(rgb, depth, extrinsic)
                self._live_fusion_generation += 1
            with self._stage("model_refresh"):
                self._extract_model_pcd()
            self.frame_count = 1
            self._last_rgbd = rgbd
            self._last_reg_pcd = current_pcd
            self.poses.append((self._processed_count, self.cumulative_T.copy()))
            elapsed_ms = (time.perf_counter() - t0) * 1000
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
        except (CudaInputError, FusionUpdateError):
            raise
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
            self._live_fusion_generation += 1
        self.cumulative_T = pose
        self._last_rgbd = rgbd
        self._last_reg_pcd = current_pcd
        self.frame_count += 1
        self._integrations_since_model += 1
        self._integrations_since_preview += 1
        self.poses.append((self._processed_count, pose.copy()))
        if self._visual_evidence is not None:
            self.tracking_edges.append({"source_index": self._processed_count,
                                        **self._visual_evidence})

        # Refresh model periodically
        should_extract = (
            method in ("global", "appearance+icp")
            or self._integrations_since_preview >= self.MODEL_REFRESH_INTERVAL
        )
        if should_extract:
            with self._stage("model_refresh"):
                if method == "keyframe+visual" and self.backend["model_preparation"] == "lazy":
                    self._refresh_live_points()
                else:
                    self._extract_model_pcd()

        elapsed_ms = (time.perf_counter() - t0) * 1000

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
        if self.settings.offline_registration == "depth":
            from .depth_graph import propose_depth_poses as propose_fragment_poses
        else:
            from .fragments import propose_fragment_poses

        started = time.monotonic()
        report = {"applied": False}
        try:
            proposals, report = propose_fragment_poses(self, progress_cb)
            if proposals is not None:
                candidate = copy.copy(self)
                required = self._required_fusion_blocks(proposals, progress_cb)
                report.update(fusion_required_blocks=required, fusion_voxel_m=self.voxel_size)
                report.update(plan_fusion(self.device, required, self.voxel_size))
                candidate._fusion_block_limit = report["allocated_blocks"]
                report.update(fusion_required_blocks=required,
                              fusion_block_limit=candidate._fusion_block_limit,
                              fusion_voxel_m=self.voxel_size)
                candidate.vbg = self._create_vbg(block_count=candidate._fusion_block_limit)
                for completed, (index, pose) in enumerate(proposals, 1):
                    rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
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
                while len(diagnostics) < self.stored_count:
                    index = len(diagnostics)
                    diagnostics.append({"index": index, "success": False,
                        "metadata": self.frame_metadata[index], "session_id": self.session_id,
                        "message": "Awaiting offline depth registration"})
                connected = {i for i, _ in proposals}
                fragment_by_frame = {i: f["id"] for f in report["fragments"] for i in f["frame_indices"]}
                for result in diagnostics:
                    if result["index"] not in connected:
                        result.update(success=False, excluded_offline=True,
                                      message_before_reconnection=result.get("message"),
                                      pose_before_reconnection=result.get("pose"),
                                      message="Excluded: no verified connection to the selected reconstruction")
                        result.pop("pose", None)
                for index, pose in proposals:
                    result = diagnostics[index]
                    if result.get("pose") is not None:
                        result["pose_before_reconnection"] = result["pose"]
                    if not result["success"]:
                        result.update(success=True, recovered_offline=True,
                                      method="depth+graph" if self.settings.offline_registration == "depth" else "fragment+graph",
                                      message_before_reconnection=result.get("message"),
                                      message="Registered from validated depth geometry" if self.settings.offline_registration == "depth" else "Recovered through verified fragment registration")
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
                last_rgbd = self._make_rgbd(*self._prepare_input(*self.raw_frames[last_index], self.settings))
                report.update(applied=True, reason=(f"Reconnected {report['recovered_frames']} views; "
                              f"{report['corrected_frames']} poses corrected, {len(report['excluded_frames'])} excluded"),
                              fusion_blocks=int(candidate.vbg.hashmap().size()),
                              fusion_block_limit=candidate._fusion_block_limit,
                              fusion_voxel_m=self.voxel_size)
                if self.settings.offline_registration == "depth":
                    sizes = report["validated_component_sizes"]
                    report["reason"] = (f"Depth registered {sum(sizes)}/{self.stored_count} captures across {len(sizes)} maps; "
                                        f"{len(proposals)} fused in the selected map")
                # Native work has succeeded; commit the candidate as one state.
                self.original_poses = [(i, p.copy()) for i, p in self.poses]
                for name in ("vbg", "model_pcd", "_live_points", "_live_colors", "_model_feature_cloud",
                             "_model_fpfh", "_model_pyramid", "_tensor_model_pyramid", "_integrations_since_model",
                             "_integrations_since_preview", "_pending_model_cloud", "_pending_model_frame_count"):
                    setattr(self, name, getattr(candidate, name))
                self.poses, self.diagnostics = proposals, diagnostics
                self._processed_count = self.stored_count
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
                self._bundle_count = None
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

    def _required_fusion_blocks(self, proposals, progress_cb=None, *, stage="fragment_reconnection"):
        """Measure allocation before touching a candidate or existing volume.

        A one-block grid computes the exact frustum coordinates without voxel
        activation. Allocation follows the measured scene extent.
        """
        scratch = self._create_vbg(block_count=1)
        blocks = set()
        for completed, (index, pose) in enumerate(proposals, 1):
            depth = prepare_metric_depth(self.raw_frames[index][1], self.settings)
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
                    "stage": stage,
                    "message": f"Estimating reconstruction storage {completed}/{len(proposals)} views",
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
                required = self._required_fusion_blocks(proposals, stage="refinement")
                report.update(plan_fusion(self.device, required, self.voxel_size))
                candidate._fusion_block_limit = report["allocated_blocks"]
                candidate.vbg = self._create_vbg(block_count=candidate._fusion_block_limit)
                for index, pose in proposals:
                    rgb, depth = self.raw_frames[index]
                    rgb, depth = self._prepare_input(rgb, depth, self.settings)
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
                    "_integrations_since_preview",
                    "_pending_model_cloud",
                    "_pending_model_frame_count",
                ):
                    setattr(self, name, getattr(candidate, name))
                self.poses = proposals
                self.cumulative_T = cumulative
                self.diagnostics = diagnostics
                self._bundle_count = None
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

    def _bundle_volume(self, progress_cb=None):
        """Commit joint camera/landmark refinement only after bounded fresh fusion."""
        if self._bundle_count == self.stored_count:
            return True
        from .bundle_adjustment import MAX_ROTATION_DEG, MAX_TRANSLATION_M, propose_bundle_poses
        from .fragments import _rigid
        from .refinement import motion

        started = time.monotonic()
        report = {"applied": False}
        try:
            proposals, report = propose_bundle_poses(self, progress_cb)
            report = dict(report)
            if proposals is not None:
                if ([i for i, _ in proposals] != [i for i, _ in self.poses]
                        or not proposals
                        or not all(_rigid(p) for _, p in proposals)
                        or not np.allclose(proposals[0][1], self.poses[0][1], atol=1e-8, rtol=0)):
                    raise ValueError("Bundle proposal changed frame identities or the anchored coordinate system")
                for (_, old), (_, new) in zip(self.poses, proposals):
                    translation, angle = motion(np.linalg.inv(old) @ new)
                    if translation > MAX_TRANSLATION_M or angle > MAX_ROTATION_DEG:
                        raise ValueError("Bundle proposal exceeded camera correction bounds")
                required = self._required_fusion_blocks(proposals, progress_cb, stage="bundle_adjustment")
                report.update(fusion_required_blocks=required, fusion_voxel_m=self.voxel_size)
                report.update(plan_fusion(self.device, required, self.voxel_size))
                limit = report["allocated_blocks"]
                report.update(fusion_required_blocks=required, fusion_block_limit=limit,
                              fusion_voxel_m=self.voxel_size)
                candidate = copy.copy(self)
                candidate._fusion_block_limit = limit
                candidate.vbg = self._create_vbg(block_count=limit)
                for completed, (index, pose) in enumerate(proposals, 1):
                    rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
                    candidate._integrate_vbg(rgb, depth, np.linalg.inv(pose))
                    if progress_cb:
                        progress_cb(completed, len(proposals), {
                            "stage": "bundle_adjustment",
                            "message": f"Fusing joint RGB-D refinement {completed}/{len(proposals)} views",
                        })
                candidate._extract_model_pcd()
                if candidate.model_pcd is None or len(candidate.model_pcd.points) < 100:
                    raise ValueError("Jointly refined volume has insufficient geometry")
                diagnostics = [dict(result) for result in self.diagnostics]
                for index, pose in proposals:
                    diagnostics[index]["pose_before_bundle_adjustment"] = diagnostics[index].get("pose")
                    diagnostics[index]["pose"] = pose.tolist()
                originals = [(i, p.copy()) for i, p in self.poses]
                last_index, last_pose = proposals[-1]
                last_rgbd = self._make_rgbd(*self._prepare_input(*self.raw_frames[last_index], self.settings))
                report.update(applied=True, reason="Validated joint RGB-D refinement committed after fresh fusion",
                              fusion_blocks=int(candidate.vbg.hashmap().size()))
                # Native allocation, fusion, extraction and state preparation succeeded.
                for name in ("vbg", "model_pcd", "_live_points", "_live_colors", "_model_feature_cloud",
                             "_model_fpfh", "_model_pyramid", "_tensor_model_pyramid", "_integrations_since_model",
                             "_integrations_since_preview", "_pending_model_cloud", "_pending_model_frame_count"):
                    setattr(self, name, getattr(candidate, name))
                if self.original_poses is None:
                    self.original_poses = originals
                self.poses, self.diagnostics = proposals, diagnostics
                self.cumulative_T = last_pose.copy()
                self._last_rgbd = last_rgbd
                self._last_reg_pcd = None
                self._final_vbg = None
            self.bundle_adjustment = report
            self._bundle_count = self.stored_count
        except Exception as exc:
            logger.exception("Joint RGB-D refinement failed; previous reconstruction retained")
            report.update(applied=False, failed=True, reason=f"Joint refinement failed: {exc}")
            self.bundle_adjustment = report
        elapsed = (time.monotonic() - started) * 1000
        self.bundle_adjustment["elapsed_ms"] = elapsed
        self.stage_totals_ms["bundle_adjustment"] = self.stage_totals_ms.get("bundle_adjustment", 0) + elapsed
        return not self.bundle_adjustment.get("failed", False)

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
        planning_started = time.monotonic()
        required = candidate._required_fusion_blocks(self.poses, progress_cb, stage="final_reintegration")
        planning_elapsed_ms = (time.monotonic() - planning_started) * 1000
        allocation = plan_fusion(self.device, required, candidate.voxel_size)
        candidate._fusion_block_limit = allocation["allocated_blocks"]
        candidate._final_missing_only_activation = candidate.settings.confidence_fusion is True
        allocated = candidate._fusion_block_limit
        candidate._final_allocated_blocks = allocated
        candidate.vbg = candidate._create_vbg(block_count=allocated)
        initial_capacity = int(candidate.vbg.hashmap().capacity())
        if candidate._final_missing_only_activation and initial_capacity != allocated:
            raise ValueError("Weighted Final native initial capacity differs from the exact plan")
        for completed, (index, pose) in enumerate(self.poses, 1):
            rgb, depth = self._prepare_input(*self.raw_frames[index], self.settings)
            candidate._integrate_vbg(rgb, depth, np.linalg.inv(pose))
            if progress_cb:
                progress_cb(
                    completed,
                    len(self.poses),
                    {"message": f"Final fusion {completed}/{len(self.poses)} accepted views"},
                )
        actual_capacity = int(candidate.vbg.hashmap().capacity())
        actual_blocks = int(candidate.vbg.hashmap().size())
        if candidate._final_missing_only_activation and (
                actual_capacity != allocated or actual_blocks != required):
            raise ValueError("Weighted Final capacity or unique block count differs from the exact plan")
        elapsed = (time.monotonic() - started) * 1000
        self.stage_totals_ms["final_reintegration"] = (
            self.stage_totals_ms.get("final_reintegration", 0) + elapsed
        )
        self.final_reconstruction = {
            **allocation,
            "applied": False,
            "reason": "Awaiting final surface validation",
            "voxel_m": candidate.voxel_size,
            "blocks": actual_blocks,
            "block_limit": candidate._fusion_block_limit,
            "required_blocks": required,
            "requested_block_capacity": allocated,
            "allocated_blocks": actual_capacity,
            "initial_block_capacity": initial_capacity,
            "allocation_strategy": "exact missing-key activation" if candidate._final_missing_only_activation else "automatic native activation",
            "attribute_budget_mib": actual_capacity * 4096 * 20 / 2**20,
            "planned_attribute_budget_mib": allocation["attribute_budget_mib"],
            "planning_elapsed_ms": planning_elapsed_ms,
            "elapsed_ms": elapsed,
        }
        return candidate.vbg

    @registration_scope
    def build_mesh(self, progress_cb=None) -> tuple[bool, dict]:
        """Build transactionally; a failed final build preserves existing geometry."""
        if self.settings.offline_registration == "depth":
            process_result = {"success": True, "frame_count": self.frame_count, "skipped_count": 0}
        else:
            process_result = self.process_frames(progress_cb=progress_cb)
            if self.frame_count == 0:
                return False, process_result
        if self.settings.reconnect_fragments or self.settings.offline_registration == "depth":
            if not self._reconnect_volume(progress_cb):
                process_result.update(success=False, message=self.fragment_reconnection["reason"],
                                      fragment_reconnection=self.fragment_reconnection)
                return False, process_result
        else:
            self.fragment_reconnection = {"applied": False, "reason": "Not requested"}
        if self.frame_count == 0:
            return False, {**process_result, "success": False, "message": "No verified camera poses"}
        process_result.update(frame_count=self.frame_count,
                              skipped_count=sum(not d["success"] for d in self.diagnostics),
                              fragment_reconnection=self.fragment_reconnection)
        if self.settings.refine_poses and self.settings.offline_registration != "depth":
            self._refine_volume()
        else:
            self.refinement = {"applied": False, "reason": "Not requested"}
        process_result["refinement"] = self.refinement
        if self.settings.bundle_adjustment and self.settings.offline_registration != "depth":
            if not self._bundle_volume(progress_cb):
                process_result.update(success=False, message=self.bundle_adjustment["reason"],
                                      bundle_adjustment=self.bundle_adjustment)
                return False, process_result
        else:
            self.bundle_adjustment = {"applied": False, "reason": "Not requested"}
        process_result["bundle_adjustment"] = self.bundle_adjustment
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
                    applied=True, reason="Fresh fusion with automatic allocation"
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
        self._discard_project_archive()

    def _discard_project_archive(self):
        path = getattr(self, "_project_archive_path", None)
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._project_archive_path = None
