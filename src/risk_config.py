"""Experimental risk defaults; all image coordinates are normalized, not metres."""
from copy import deepcopy
import math
import cv2
import numpy as np

DEFAULT_RISK = {
    "enabled": False,
    "corridor_polygon": [[0.42, 0.45], [0.58, 0.45], [0.98, 1.0], [0.02, 1.0]],
    "immediate_polygon": [[0.205, 0.75], [0.795, 0.75], [0.98, 1.0], [0.02, 1.0]],
    "footprint_height_ratio": 0.15,
    "footprint_max_height": 0.02,
    "exit_overlap_threshold": 0.10,
    "exit_margin": 0.02,
    "static_caution_y": 0.65,
    "static_danger_y": 0.88,
    "static_ground_classes": ["pole", "bollard", "tree_trunk", "potted_plant",
        "fire_hydrant", "utility_box", "parking_meter", "kiosk", "bench",
        "barricade", "trash_bin"],
    "sidewalk_roi_enabled": True,
    "roi_max_shift": 0.15,
    "roi_fit_residual": 0.04,
    "roi_confirm_s": 0.20,
    "roi_smooth_s": 0.30,
    "roi_max_shift_per_s": 0.12,
    "roi_hold_s": 0.50,
    "roi_return_s": 0.50,
    "clear_confirm_s": 0.50,
    "uncertainty_hold_s": 0.80,
    "id_bridge_s": 0.30,
    "id_bridge_iou": 0.50,
    "overlap_threshold": 0.20,
    "edge_margin_ratio": 0.08,
    "history_window_s": 0.60,
    "min_history_s": 0.20,
    "prediction_horizon_s": 0.80,
    "min_lateral_speed": 0.05,
    "lateral_near_y": 0.60,
    "reset_gap_s": 0.50,
    "ttc_alerts": False,
    "ttc_danger_s": 1.50,
    "ttc_caution_s": 3.00,
    "min_expansion_rate": 0.05,
    "max_motion_residual": 0.03,
    "camera_max_rotation_deg": 4.0,
    "camera_max_translation": 0.12,
    "camera_max_scale_change": 0.10,
    "release_hold_s": 0.50,
    "repeat_cooldown_s": 2.0,
    "event_match_iou": 0.30,
    "draw_roi": True,
    "review_overlay": False,
    "log_jsonl": True,
}
# New rules are enabled explicitly in inference.yaml, so saved older configs replay unchanged.
DEFAULT_RISK.update({
    "roi_ground_adapt_enabled":False,
    "side_proximity_enabled":False, "side_near_y":.50,
    "side_min_height":.18, "side_min_width":.12,
    "full_static_footprint":False, "relative_entry_enabled":False,
    "roi_recalibration_enabled":False, "roi_jitter_shift":.03, "roi_large_shift":.08,
    "roi_change_confirm_s":.30, "roi_small_confirm_s":.50, "roi_candidate_spread":.03,
    "surface_risk_enabled":False, "surface_confirm_s":.20, "surface_clear_s":.50,
    "surface_min_area":.003, "surface_band_fraction":.12, "surface_unstable_change":.18,
    "class_bridge_enabled":False, "visibility_advisory_s":1.0,
    "warning_grouping_enabled":False,
    "warning_group_iou":.65, "warning_group_containment":.85,
    "warning_group_max_area_ratio":1.60,
})
DEFAULT_TRACKING = {
    "enabled": True, "backend": "botsort",
    "track_high_thresh": 0.25, "track_low_thresh": 0.10,
    "new_track_thresh": 0.25, "track_buffer": 30,
    "match_thresh": 0.80, "fuse_score": True,
}

def _merge(value, defaults, name):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError(f"{name}: unknown keys or invalid mapping")
    return {**deepcopy(defaults), **deepcopy(value)}

def _number(value, name, low, high=None, strictly_positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < low
            or (high is not None and value > high)
            or (strictly_positive and value == 0)):
        raise ValueError(f"{name}: invalid numeric value")

def risk_config(value=None):
    cfg = _merge(value, DEFAULT_RISK, "risk")
    for key in ("enabled", "ttc_alerts", "draw_roi", "log_jsonl", "review_overlay", "sidewalk_roi_enabled"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"risk.{key} must be boolean")
    for key in ("corridor_polygon", "immediate_polygon"):
        try:
            p = np.asarray(cfg[key], dtype=np.float32)
        except (ValueError, TypeError) as error:
            raise ValueError(f"risk.{key}: invalid polygon") from error
        if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()
                or not cv2.isContourConvex(p) or abs(cv2.contourArea(p)) < 1e-6):
            raise ValueError(f"risk.{key}: must be a convex polygon in [0,1]")
    for key in ("roi_ground_adapt_enabled","side_proximity_enabled","full_static_footprint","relative_entry_enabled",
                "roi_recalibration_enabled","surface_risk_enabled","class_bridge_enabled",
                "warning_grouping_enabled"):
        if not isinstance(cfg[key],bool):
            raise ValueError(f"risk.{key} must be boolean")
    for key in ("side_near_y","side_min_height","side_min_width","roi_jitter_shift",
                "roi_large_shift","roi_candidate_spread","surface_min_area",
                "surface_band_fraction","surface_unstable_change","warning_group_iou",
                "warning_group_containment"):
        _number(cfg[key],f"risk.{key}",0,1,True)
    for key in ("roi_change_confirm_s","roi_small_confirm_s","surface_confirm_s",
                "surface_clear_s","visibility_advisory_s"):
        _number(cfg[key],f"risk.{key}",0,strictly_positive=True)
    _number(cfg["warning_group_max_area_ratio"],"risk.warning_group_max_area_ratio",1,
            strictly_positive=True)
    if cfg["roi_jitter_shift"] >= cfg["roi_large_shift"]:
        raise ValueError("ROI jitter threshold must be below large shift")
    unit = ("footprint_height_ratio", "overlap_threshold", "edge_margin_ratio",
            "min_lateral_speed", "lateral_near_y", "min_expansion_rate",
            "max_motion_residual", "camera_max_translation", "camera_max_scale_change",
            "event_match_iou", "footprint_max_height", "exit_overlap_threshold",
            "exit_margin", "static_caution_y", "static_danger_y", "roi_max_shift",
            "roi_fit_residual", "roi_max_shift_per_s", "id_bridge_iou")
    for key in unit:
        _number(cfg[key], f"risk.{key}", 0, 1, True)
    for key in ("history_window_s", "min_history_s", "prediction_horizon_s", "reset_gap_s",
                "ttc_danger_s", "ttc_caution_s", "release_hold_s", "repeat_cooldown_s",
                "camera_max_rotation_deg", "roi_confirm_s", "roi_smooth_s", "roi_hold_s",
                "roi_return_s", "clear_confirm_s", "uncertainty_hold_s", "id_bridge_s"):
        _number(cfg[key], f"risk.{key}", 0, strictly_positive=True)
    names = cfg["static_ground_classes"]
    if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("static_ground_classes must be a list of class names")
    if cfg["exit_overlap_threshold"] >= cfg["overlap_threshold"]:
        raise ValueError("exit overlap must be below entry overlap")
    if cfg["static_caution_y"] >= cfg["static_danger_y"]:
        raise ValueError("static caution position must be below danger position")
    if cfg["clear_confirm_s"] > cfg["uncertainty_hold_s"]:
        raise ValueError("clear confirmation must not exceed uncertainty hold")
    if cfg["min_history_s"] > cfg["history_window_s"]:
        raise ValueError("min_history_s must not exceed history_window_s")
    if cfg["ttc_danger_s"] > cfg["ttc_caution_s"]:
        raise ValueError("ttc_danger_s must not exceed ttc_caution_s")
    outer = np.asarray(cfg["corridor_polygon"], np.float32)
    if any(cv2.pointPolygonTest(outer, tuple(map(float, point)), False) < -1e-6
           for point in cfg["immediate_polygon"]):
        raise ValueError("immediate_polygon must be inside corridor_polygon")
    return cfg

def tracking_config(value=None):
    cfg = _merge(value, DEFAULT_TRACKING, "tracking")
    if cfg["backend"] not in ("botsort", "bytetrack"):
        raise ValueError("tracking.backend must be botsort or bytetrack")
    for key in ("enabled", "fuse_score"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"tracking.{key} must be boolean")
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        _number(cfg[key], f"tracking.{key}", 0, 1)
    if cfg["track_low_thresh"] > cfg["track_high_thresh"]:
        raise ValueError("track_low_thresh must not exceed track_high_thresh")
    if isinstance(cfg["track_buffer"], bool) or not isinstance(cfg["track_buffer"], int) or cfg["track_buffer"] < 1:
        raise ValueError("track_buffer must be a positive integer")
    return cfg
