"""Short image-space histories with timestamp and camera-motion quality gates."""
from collections import deque
import math
import cv2
import numpy as np
from src.risk_geometry import overlap

class CameraMotionGuard:
    """Conservative affine sanity check; this does not measure world velocity."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.previous = None

    def update(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (320, max(32, round(h*320/w)))), cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
        if previous is None or previous.shape != gray.shape:
            return False
        points = cv2.goodFeaturesToTrack(previous, 120, .01, 8)
        if points is None or len(points) < 8:
            return False
        following, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if following is None or status is None:
            return False
        selected = status.ravel().astype(bool)
        a, b = points[selected], following[selected]
        if len(a) < 8:
            return False
        transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3)
        if transform is None or not np.isfinite(transform).all() or inliers.mean() < .5:
            return False
        scale = math.hypot(transform[0,0], transform[1,0])
        rotation = abs(math.degrees(math.atan2(transform[1,0], transform[0,0])))
        translation = math.hypot(transform[0,2]/gray.shape[1], transform[1,2]/gray.shape[0])
        return (rotation <= self.cfg["camera_max_rotation_deg"]
                and abs(scale-1) <= self.cfg["camera_max_scale_change"]
                and translation <= self.cfg["camera_max_translation"])

class MotionHistory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.histories = {}

    def prune(self, timestamp):
        self.histories = {key: history for key, history in self.histories.items()
                          if timestamp-history[-1][0] <= self.cfg["history_window_s"]}

    def update(self, detection, geometry, timestamp, valid, corridor_polygon=None):
        result = {"quality": "insufficient", "velocity_norm_per_s": None,
                  "time_to_corridor_s": None, "time_to_path_s": None, "ttc_scale_s": None, "history_s": 0.0,
                  "relative_expansion_per_s": None, "approach_state": "unknown",
                  "ttc_invalid_reason": "insufficient_history"}
        track_id = detection["track_id"]
        if track_id is None:
            return result
        if not valid:
            self.histories.pop(track_id, None)
            result["quality"] = "unstable"
            result["ttc_invalid_reason"] = "unstable_motion"
            return result
        history = self.histories.setdefault(track_id, deque())
        # Do not interpret a changed class or reacquired stale track as continuous motion.
        if history and (history[-1][1] != detection["class_id"]
                        or timestamp <= history[-1][0]
                        or timestamp-history[-1][0] > self.cfg["reset_gap_s"]):
            history.clear()
        history.append((timestamp, detection["class_id"], *geometry["point"],
                        geometry["height"], geometry["clipped"]))
        while history and timestamp-history[0][0] > self.cfg["history_window_s"] + 1e-9:
            history.popleft()
        duration = timestamp-history[0][0]
        result["history_s"] = duration
        if len(history) < 3 or duration + 1e-9 < self.cfg["min_history_s"]:
            return result
        samples = np.asarray([row[:5] for row in history], float)
        times = samples[:,0]-timestamp
        matrix = np.column_stack([times, np.ones(len(times))])
        fitted, *_ = np.linalg.lstsq(matrix, samples[:,2:5], rcond=None)
        residual = float(np.max(np.sqrt(np.mean((matrix @ fitted-samples[:,2:5])**2, axis=0))))
        if residual > self.cfg["max_motion_residual"]:
            result["quality"] = "unstable"
            result["ttc_invalid_reason"] = "unstable_motion"
            return result
        vx, vy, dh = map(float, fitted[0])
        result["quality"] = "valid"
        result["velocity_norm_per_s"] = [vx, vy]
        # TTC uses relative expansion; do not subtract forward ego-motion.
        expansion = dh / geometry["height"]
        if any(row[5] for row in history):
            result["ttc_invalid_reason"] = "clipped_box"
        else:
            result["relative_expansion_per_s"] = float(expansion)
            threshold = self.cfg["min_expansion_rate"]
            result["approach_state"] = "approaching" if expansion >= threshold else ("receding" if expansion <= -threshold else "steady")
            if expansion >= threshold:
                result["ttc_scale_s"] = float(1 / expansion)
                result["ttc_invalid_reason"] = None
            else:
                result["ttc_invalid_reason"] = "not_expanding"
        # Nine samples (including now) over the configurable short horizon.
        if ((self.cfg["relative_entry_enabled"] or detection["class_name"] not in self.cfg["static_ground_classes"])
                and abs(vx) >= self.cfg["min_lateral_speed"] and geometry["corridor_overlap"] < self.cfg["overlap_threshold"]):
            for future in np.linspace(0, self.cfg["prediction_horizon_s"], 9)[1:]:
                projected = np.asarray(geometry["footprint"]) + [vx*future, vy*future, vx*future, vy*future]
                polygon = corridor_polygon if corridor_polygon is not None else self.cfg["corridor_polygon"]
                polygons = polygon if isinstance(polygon[0][0],(list,tuple,np.ndarray)) else [polygon]
                if max(overlap(projected,p) for p in polygons) >= self.cfg["overlap_threshold"]:
                    result["time_to_path_s"] = float(future)
                    if detection["class_name"] not in self.cfg["static_ground_classes"]:
                        result["time_to_corridor_s"] = float(future)
                    break
        return result
