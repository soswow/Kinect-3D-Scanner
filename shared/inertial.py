"""Polled Kinect acceleration, conservative gravity fusion, and image orientation.

Confidence is an engineering reliability score, not calibrated probability.
Factory axes are provisional; explicit per-device calibration is supported.
"""

import math
import time
import uuid
import json
import hashlib
from collections import deque

import numpy as np

G = 9.80665
MAX_HOST_MAPPING_UNCERTAINTY_S = 0.03
FACTORY_CALIBRATION = {
    "id": "kinect-v1-factory-axes-unverified",
    "verified": False,
    "bias_m_s2": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
    # XY roll signs follow glview's atan2(y, x)-90 in its Y-up OpenGL
    # projection. Z completes a proper rotation; physical extrinsics need
    # independent calibration, particularly across motor tilt positions.
    "sensor_to_camera": [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
}


def calibration_profile(value=None):
    if value is not None and (not isinstance(value, dict) or not set(FACTORY_CALIBRATION).issubset(value)):
        raise ValueError("Accelerometer calibration requires identity, verification, bias, scale, and sensor-to-camera rotation")
    value = dict(FACTORY_CALIBRATION if value is None else value)
    bias, scale = np.asarray(value["bias_m_s2"], float), np.asarray(value["scale"], float)
    rotation = np.asarray(value["sensor_to_camera"], float)
    if (bias.shape != (3,) or scale.shape != (3,) or rotation.shape != (3, 3)
            or not all(np.isfinite(a).all() for a in (bias, scale, rotation))
            or np.any(np.abs(bias) > G) or np.any((scale < 0.5) | (scale > 2))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
            or not isinstance(value.get("id"), str) or not 1 <= len(value["id"]) <= 128
            or type(value.get("verified")) is not bool):
        raise ValueError("Invalid accelerometer bias, scale, rotation, or identity")
    profile = {"id": value["id"], "verified": value["verified"],
               "bias_m_s2": bias.tolist(), "scale": scale.tolist(),
               "sensor_to_camera": rotation.tolist()}
    if "evidence" in value:
        evidence = json.dumps(value["evidence"], allow_nan=False)
        if len(evidence.encode()) > 64000:
            raise ValueError("Accelerometer calibration evidence exceeds 64 KB")
        profile["evidence"] = json.loads(evidence)
    return profile


def fit_calibration(observations, validation=None, identity=None):
    """Fit bias/scale/alignment to independently known stationary camera-up vectors."""
    def arrays(rows, minimum):
        if len(rows) < minimum:
            raise ValueError(f"Require at least {minimum} independent stationary orientations")
        measured = np.asarray([r["acceleration_m_s2"] for r in rows], float)
        up = np.asarray([r["up_camera"] for r in rows], float)
        if (measured.shape != (len(rows), 3) or up.shape != measured.shape
                or not np.isfinite(measured).all() or not np.isfinite(up).all()
                or np.any(np.abs(np.linalg.norm(up, axis=1) - 1) > .01)):
            raise ValueError("Use finite 3D acceleration and independently measured unit camera-up vectors")
        return measured, G * up

    measured, target = arrays(observations, 6)
    design = np.column_stack((measured, np.ones(len(measured))))
    if np.linalg.matrix_rank(design) < 4 or np.linalg.cond(design) > 100 or np.linalg.matrix_rank(target - target.mean(axis=0)) < 3:
        raise ValueError("Calibration orientations must span all three axes")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    matrix, offset = coefficients[:3].T, coefficients[3]
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Calibration alignment is singular")
    scale = np.linalg.norm(matrix, axis=0)
    u, _, vt = np.linalg.svd(matrix / scale)
    rotation = u @ vt
    bias = -np.linalg.solve(matrix, offset)
    profile = calibration_profile({"id": identity or "kinect-accel-" + hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()[:16],
                                   "verified": False, "bias_m_s2": bias.tolist(), "scale": scale.tolist(),
                                   "sensor_to_camera": rotation.tolist()})

    def errors(a, b):
        predicted = ((a - bias) * scale) @ rotation.T
        distances = np.linalg.norm(predicted - b, axis=1)
        directions = np.sum(predicted * b, axis=1) / (np.linalg.norm(predicted, axis=1) * G)
        return {"rms_m_s2": float(np.sqrt(np.mean(distances**2))),
                "max_m_s2": float(distances.max()),
                "max_angle_deg": float(np.degrees(np.arccos(np.clip(directions, -1, 1))).max())}

    report = {"algorithm": "stationary-affine-polar-v1", "fit": errors(measured, target)}
    if report["fit"]["max_m_s2"] > .25 or report["fit"]["max_angle_deg"] > 2:
        raise ValueError("Stationary calibration residuals exceed 0.25 m/s² or 2°; check references and motion")
    if validation is not None:
        a, b = arrays(validation, 3)
        report["validation"] = errors(a, b)
        if report["validation"]["max_m_s2"] > .25 or report["validation"]["max_angle_deg"] > 2:
            raise ValueError("Held-out accelerometer calibration failed")
        profile["verified"] = True
    profile["evidence"] = {"observations": observations, "validation": validation, "report": report,
                           "reference": "Independent stationary camera-up vectors supplied by operator"}
    return calibration_profile(profile)


class GravityEstimator:
    VERSION = "robust-gravity-v1"

    def __init__(self, calibration=None):
        self.calibration = calibration_profile(calibration)
        self.history = deque(maxlen=8)
        self.filtered = None
        self.last_time = None

    def update(self, sample):
        result = {"valid": False, "confidence": 0.0, "algorithm": self.VERSION,
                  "calibration_id": self.calibration["id"],
                  "calibration_verified": self.calibration["verified"]}
        try:
            if not sample.get("valid"):
                raise ValueError(sample.get("reason", "Acceleration unavailable"))
            stamp = float(sample["host_monotonic_s"])
            latency = float(sample["read_end_s"]) - float(sample["read_start_s"])
            acceleration = np.asarray(sample["acceleration_m_s2"], float)
            if acceleration.shape != (3,) or not np.isfinite(acceleration).all():
                raise ValueError("Invalid acceleration vector")
            if not math.isfinite(stamp) or not 0 <= latency <= 0.03:
                raise ValueError("Acceleration timing uncertain")
            acceleration = (acceleration - self.calibration["bias_m_s2"]) * self.calibration["scale"]
            acceleration = np.asarray(self.calibration["sensor_to_camera"]) @ acceleration
            norm = float(np.linalg.norm(acceleration))
            if not 0.7 * G <= norm <= 1.3 * G:
                raise ValueError("Acceleration inconsistent with gravity")
            gap = stamp - self.last_time if self.last_time is not None else None
            if gap is not None and (gap <= 0 or gap > 0.25):
                self.history.clear()
                self.filtered = None
            self.history.append(acceleration)
            median = np.median(self.history, axis=0)
            alpha = 1 if self.filtered is None else 1 - math.exp(-max(gap or 0.05, 0.001) / 0.12)
            self.filtered = median if self.filtered is None else self.filtered + alpha * (median - self.filtered)
            scatter = float(np.median(np.linalg.norm(np.asarray(self.history) - median, axis=1)))
            confidence = math.exp(-((norm - G) / (0.07 * G))**2 - (scatter / (0.08 * G))**2)
            confidence *= min(1, len(self.history) / 5)
            filtered_norm = np.linalg.norm(self.filtered)
            if filtered_norm < 0.25 * G:
                raise ValueError("Gravity direction is ambiguous")
            up = self.filtered / filtered_norm
            self.last_time = stamp
            result.update(valid=confidence >= 0.15, confidence=confidence,
                          up_camera=up.tolist(), norm_m_s2=norm, scatter_m_s2=scatter,
                          reason="Gravity available" if confidence >= 0.15 else "Acceleration changing")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.history.clear()
            self.filtered = None
            self.last_time = None
            result["reason"] = str(exc)
        return result


class AccelerometerPoller:
    """Copy every read, including errors; never silently treat cached data as fresh."""

    def __init__(self, freenect, device, generation=None, calibration=None, enabled=True):
        self.freenect, self.device = freenect, device
        self.generation = generation or uuid.uuid4().hex
        self.estimator = GravityEstimator(calibration)
        self.samples = deque(maxlen=64)
        self.sequence, self.errors, self.next_poll = 0, 0, 0
        self.enabled = enabled and all(hasattr(freenect, n) for n in
                                      ("update_tilt_state", "get_tilt_state", "get_mks_accel"))
        self.reason = "Waiting for acceleration" if self.enabled else "Accelerometer unavailable or disabled"

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if not self.enabled or now < self.next_poll:
            return None
        self.sequence += 1
        start = time.monotonic()
        sample = {"version": 1, "capture_generation": self.generation,
                  "sequence": self.sequence, "read_start_s": start, "valid": False}
        try:
            code = self.freenect.update_tilt_state(self.device)
            sample["return_code"] = code
            if code < 0:
                raise ValueError(f"Accelerometer read failed ({code})")
            state = self.freenect.get_tilt_state(self.device)
            raw = [int(getattr(state, "accelerometer_" + axis)) for axis in "xyz"]
            for field in ("tilt_angle", "tilt_status"):
                if hasattr(state, field):
                    sample[field + "_raw"] = int(getattr(state, field))
            if hasattr(self.freenect, "get_tilt_degs"):
                tilt = float(self.freenect.get_tilt_degs(state))
                sample["tilt_degrees"] = tilt if math.isfinite(tilt) else None
            acceleration = [float(v) for v in self.freenect.get_mks_accel(state)]
            sample.update(raw_counts=raw, acceleration_m_s2=acceleration if all(math.isfinite(v) for v in acceleration) else None)
            if not all(math.isfinite(v) for v in acceleration) or not 0.1 * G < np.linalg.norm(acceleration) < 4 * G:
                raise ValueError("Accelerometer returned implausible data")
            sample["valid"] = True
            self.errors = 0
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            self.errors += 1
            sample["reason"] = str(exc)
        end = time.monotonic()
        sample.update(read_end_s=end, host_monotonic_s=(start + end) / 2,
                      timestamp_s=time.time(), units="m/s^2", timing_reference="host_read_interval")
        sample["gravity"] = self.estimator.update(sample)
        self.samples.append(sample)
        self.next_poll = end + 0.05  # No catch-up bursts after a late control transfer.
        if self.errors >= 3 or end - start > 0.05:
            self.enabled = False
            self.reason = "Accelerometer disabled after read errors or excessive latency"
        else:
            self.reason = sample.get("reason", "Acceleration available")
        return sample

    def associate(self, image_time, mapping_uncertainty=0):
        if not math.isfinite(mapping_uncertainty) or not 0 <= mapping_uncertainty <= MAX_HOST_MAPPING_UNCERTAINTY_S:
            return {"valid": False, "reason": "Image-to-host timing uncertain"}
        if not self.enabled or not self.samples:
            return {"valid": False, "reason": self.reason}
        sample = min(self.samples, key=lambda s: abs(image_time - s["host_monotonic_s"]))
        delta = image_time - sample["host_monotonic_s"]
        gravity = dict(sample["gravity"])
        valid = bool(sample["valid"] and gravity["valid"] and abs(delta) <= 0.15)
        return {"version": 1, "valid": valid,
                "reason": gravity.get("reason") if valid else "Acceleration missing, unreliable, or stale",
                "capture_generation": self.generation, "sequence": sample["sequence"],
                "sample_delta_ms": delta * 1000, "read_start_s": sample["read_start_s"],
                "read_end_s": sample["read_end_s"], "acceleration_m_s2": sample.get("acceleration_m_s2"),
                "gravity": gravity}


class OrientationTracker:
    """Quarter-turn display only; raw camera geometry never changes."""

    def __init__(self):
        self.rotation = 0
        self.candidate = None
        self.candidate_since = None
        self.generation = None

    def update(self, accelerometer, stamp, mode="auto"):
        fixed = {"landscape": 0, "portrait_left": 90, "portrait_right": 270}
        if mode in fixed:
            self.rotation = fixed[mode]
            self.candidate = None
            return {"rotation_cw_degrees": self.rotation, "mode": mode, "valid": True, "reason": "Manual orientation"}
        try:
            if not accelerometer.get("valid"):
                raise ValueError("Holding orientation: acceleration unavailable")
            gravity = accelerometer["gravity"]
            if not gravity.get("valid") or gravity.get("confidence", 0) < 0.4:
                raise ValueError("Holding orientation: acceleration changing")
            down = -np.asarray(gravity["up_camera"], float)
            if down.shape != (3,) or not np.isfinite(down).all() or np.linalg.norm(down[:2]) < 0.35:
                raise ValueError("Holding orientation: looking up or down")
            generation = accelerometer.get("capture_generation")
            if generation != self.generation:
                self.candidate = None
                self.generation = generation
            angle = math.degrees(math.atan2(down[0], down[1])) % 360
            candidate = (int(math.floor((angle + 45) / 90)) * 90) % 360
            distance = abs((angle - self.rotation + 180) % 360 - 180)
            if candidate == self.rotation or distance < 60:
                self.candidate = None
            elif candidate != self.candidate:
                self.candidate, self.candidate_since = candidate, stamp
            elif stamp - self.candidate_since >= 0.3:
                self.rotation, self.candidate = candidate, None
            return {"rotation_cw_degrees": self.rotation, "mode": mode, "valid": True,
                    "reason": "Automatic orientation", "confidence": gravity["confidence"]}
        except (KeyError, TypeError, ValueError) as exc:
            self.candidate = None
            return {"rotation_cw_degrees": self.rotation, "mode": mode, "valid": False, "reason": str(exc)}


def rotate_display(array, rotation):
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Display rotation must be a clockwise quarter turn")
    return np.ascontiguousarray(np.rot90(array, -rotation // 90))


def gravity_seed(seed, previous_pose, previous_metadata, current_metadata):
    """Blend two gravity observations with the RGB-D prediction, keeping translation.

    This is only an ICP initializer. No acceptance/fusion decision uses it alone.
    """
    report = {"applied": False, "algorithm": "gravity-seed-v1"}
    try:
        a, b = (m.get("accelerometer", {}) for m in (previous_metadata, current_metadata))
        if not (a.get("valid") and b.get("valid") and a.get("capture_generation")
                and a.get("capture_generation") == b.get("capture_generation")):
            raise ValueError("Gravity observations unavailable or cross a reconnect")
        ga, gb = a["gravity"], b["gravity"]
        if not (ga.get("valid") and gb.get("valid") and ga["calibration_id"] == gb["calibration_id"]):
            raise ValueError("Gravity calibration or reliability changed")
        if any(abs(float(s["sample_delta_ms"])) > 150 for s in (a, b)):
            raise ValueError("Gravity observation is stale")
        up_a, up_b = np.asarray(ga["up_camera"], float), np.asarray(gb["up_camera"], float)
        if any(u.shape != (3,) or not np.isfinite(u).all() or not .99 <= np.linalg.norm(u) <= 1.01 for u in (up_a, up_b)):
            raise ValueError("Invalid gravity direction")
        confidence = min(float(ga["confidence"]), float(gb["confidence"]))
        if not math.isfinite(confidence) or not 0.4 <= confidence <= 1:
            raise ValueError("Gravity reliability too low")
        source, target = seed[:3, :3] @ up_b, previous_pose[:3, :3] @ up_a
        cross = np.cross(source, target)
        sine, cosine = np.linalg.norm(cross), np.clip(source @ target, -1, 1)
        angle = math.atan2(sine, cosine)
        report["disagreement_deg"] = math.degrees(angle)
        if angle > math.radians(25):
            raise ValueError("Gravity conflicts with the motion prediction")
        weight = 0.5 * confidence
        if not (ga.get("calibration_verified") and gb.get("calibration_verified")):
            weight *= 0.25
        correction_angle = min(angle * weight, math.radians(3))
        result = seed.copy()
        if sine > 1e-8:
            x, y, z = cross / sine
            skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
            correction = np.eye(3) + math.sin(correction_angle) * skew + (1 - math.cos(correction_angle)) * (skew @ skew)
            result[:3, :3] = correction @ seed[:3, :3]
        report.update(applied=correction_angle > 1e-8, weight=weight,
                      correction_deg=math.degrees(correction_angle))
        return result, report
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        report["reason"] = str(exc)
        return seed, report
