"""Experimental obstacle risk assessment with preserved detections and optional IDs."""
import math
from copy import deepcopy
from src.risk_config import risk_config
from src.risk_geometry import geometry, sidewalk_context, surrounding_walkability
from src.risk_motion import CameraMotionGuard, MotionHistory
from src.tracking import DetectionTracker
from src.alert_policy import AlertPolicy
from src.path_roi import SidewalkGuidedROI
from src.risk_proximity import proximity
from src.surface_risk import SurfaceRisk
from src.warning_groups import group_warnings

class VideoClock:
    """Use source PTS. Nominal FPS fallback is explicitly invalid for motion."""
    def __init__(self, fps):
        self.fps, self.previous = fps, None

    def read(self, pts_ms, index):
        valid = (isinstance(pts_ms, (int,float)) and math.isfinite(pts_ms) and pts_ms >= 0
                 and (self.previous is None or pts_ms > self.previous))
        if valid:
            self.previous = pts_ms
            return pts_ms/1000, True, "pts"
        return index/self.fps, False, "nominal_fps_motion_disabled"

class RiskEngine:
    def __init__(self, config=None, tracking=None, tracker=None, camera_guard=None):
        self.config = risk_config(config)
        self.tracker = tracker if tracker is not None else DetectionTracker(tracking)
        self.camera_guard = camera_guard if camera_guard is not None else CameraMotionGuard(self.config)
        self.motion = MotionHistory(self.config)
        self.alerts = AlertPolicy(self.config)
        self.path_roi = SidewalkGuidedROI(self.config)
        self.surface = SurfaceRisk(self.config)
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.camera_guard.reset()
        self.motion.reset()
        self.alerts.reset()
        self.path_roi.reset()
        self.surface.reset()
        self.previous_time, self.previous_shape = None, None
        self.previous_tracker_status = self.tracker.status
        self.epoch = getattr(self, "epoch", -1) + 1

    def update(self, frame, detections, timestamp_s, timestamp_valid=True, class_map=None, label_ids=None):
        shape = frame.shape[:2]
        discontinuity = (self.previous_shape is not None and shape != self.previous_shape
                         or self.previous_time is not None and
                         (timestamp_s <= self.previous_time or timestamp_s-self.previous_time > self.config["reset_gap_s"]))
        if discontinuity:
            self.reset()
        self.previous_time, self.previous_shape = timestamp_s, shape
        # No motion quantities are trusted after timestamp loss.
        if not timestamp_valid:
            self.motion.reset()
        camera_stable = bool(self.camera_guard.update(frame))
        if not camera_stable:
            self.motion.reset()
        self.motion.prune(timestamp_s)
        tracked = self.tracker.attach(detections, frame)
        tracking_reset = self.tracker.status == "failed" and self.previous_tracker_status != "failed"
        if tracking_reset:
            self.motion.reset()
            self.alerts.reset()
            self.epoch += 1
        self.previous_tracker_status = self.tracker.status
        roi = self.path_roi.update(class_map, label_ids, shape, timestamp_s, timestamp_valid)
        results = []
        for detection in tracked:
            item = {**deepcopy(detection), "risk_level": "monitor", "assessment_quality": "limited",
                    "reasons": [], "observed": True, "geometry": None, "motion": None,
                    "sidewalk": {"status":"unavailable","walkable_fraction":None}}
            # Traffic-light detection/selection/display remain entirely in the existing modules.
            if detection["class_name"] == "traffic_light":
                item.update(risk_level=None, assessment_quality="not_applicable",
                            reasons=["traffic_light_not_assessed"])
                results.append(item)
                continue
            g = geometry(detection, shape, self.config, roi)
            if g is None:
                item.update(assessment_quality="unknown", reasons=["invalid_box"])
                results.append(item)
                continue
            m = self.motion.update(detection, g, timestamp_s, timestamp_valid and camera_stable,
                                   roi.get("corridor_polygons",roi["corridor_polygon"]))
            p = proximity(detection, g, self.config)
            item.update(geometry=g, motion=m, proximity=p, release_evidence=None)
            threshold = self.config["overlap_threshold"]
            entry_y = (max(self.config["lateral_near_y"],roi.get("path_top_y",0)+.10)
                       if self.config["roi_ground_adapt_enabled"] else self.config["lateral_near_y"])
            related = max(g["corridor_overlap"],g["immediate_overlap"]) >= threshold
            static = p["policy"] == "static_ground"
            if static:
                if related and p["band"] == "near":
                    item["risk_level"] = "danger"
                    item["reasons"].append("static_near_contact")
                elif related and p["band"] in ("middle","unknown"):
                    item["risk_level"] = "caution"
                    item["reasons"].append("static_path_candidate")
                elif related:
                    if g["close_candidate"]:
                        item["risk_level"] = "caution"
                        item["reasons"].append("large_static_candidate")
                    else:
                        item["reasons"].append("far_static_path_candidate")
            else:
                if g["immediate_overlap"] >= threshold:
                    item["risk_level"] = "danger"
                    item["reasons"].append("near_path_occupied")
                elif related:
                    item["risk_level"] = "caution"
                    item["reasons"].append("path_occupied")
            if g["side_proximity"]:
                if item["risk_level"] == "monitor":
                    item["risk_level"] = "caution"
                item["reasons"].append("side_close_candidate")
            if g["edge_contact"]:
                item["reasons"].append("edge_candidate")
            if m["time_to_corridor_s"] is not None and g["point"][1] >= entry_y:
                if item["risk_level"] == "monitor":
                    item["risk_level"] = "caution"
                item["reasons"].append("lateral_entry")
            future_related = (self.config["relative_entry_enabled"] and m.get("time_to_path_s") is not None
                              and g["point"][1]>=entry_y)
            if future_related and static:
                if item["risk_level"] == "monitor":
                    item["risk_level"] = "caution"
                item["reasons"].append("relative_path_entry")
            if self.config["ttc_alerts"] and m["ttc_scale_s"] is not None and (related or future_related):
                ttc = m["ttc_scale_s"]
                if ttc <= self.config["ttc_danger_s"]:
                    item["risk_level"] = "danger"
                    item["reasons"].append("short_ttc")
                elif ttc <= self.config["ttc_caution_s"]:
                    if item["risk_level"] == "monitor":
                        item["risk_level"] = "caution"
                    item["reasons"].append("approaching")
            surroundings = surrounding_walkability(g,class_map,label_ids,shape,self.config)
            item["surrounding_walkability"] = surroundings
            if (self.config["walkable_surroundings_filter_enabled"]
                    and item["risk_level"] == "danger"
                    and surroundings["all_non_walkable"]):
                item["risk_level"] = "caution"
                item["reasons"].append("nonwalkable_surroundings")
                item["release_evidence"] = "nonwalkable_surroundings"
            clear = (timestamp_valid and camera_stable and not g["clipped"] and not roi["changed"])
            if clear and item["risk_level"] == "monitor":
                if (max(g["corridor_overlap"],g["immediate_overlap"]) < self.config["exit_overlap_threshold"]
                        and g["horizontal_path_gap"] >= self.config["exit_margin"]
                        and not g["side_proximity"] and not future_related
                        and m["quality"] == "valid"):
                    item["release_evidence"] = "image_path_exit"
            elif clear and item["risk_level"] == "caution" and m["quality"] == "valid":
                item["release_evidence"] = "lower_proximity_or_urgency"
            if m["quality"] == "valid" and not g["clipped"]:
                item["assessment_quality"] = "valid"
            results.append(item)
        events = self.alerts.update(results, timestamp_s)
        surface, surface_events = self.surface.update(class_map,label_ids,shape,roi,timestamp_s,
                                                     timestamp_valid,camera_stable,results)
        events.extend(surface_events)
        warning_groups=group_warnings(results,self.config)
        return {"timestamp_s":timestamp_s, "timestamp_valid":timestamp_valid,
                "state_epoch":self.epoch, "state_reset":bool(discontinuity or tracking_reset),
                "camera_motion_stable":camera_stable, "tracker_status":self.tracker.status,
                "roi":roi, "surface":surface, "advisories":self.alerts.advisories,
                "warning_groups":warning_groups,
                "detections":results, "events":events}

    def add_sidewalk_context(self, prediction, class_map, label_ids, shape):
        # Same-frame context only. An absent/non-walkable mask never vetoes a warning.
        for item in prediction["detections"]:
            if item["geometry"] is not None:
                item["sidewalk"] = sidewalk_context(item["geometry"], class_map, label_ids, shape)
        return prediction
