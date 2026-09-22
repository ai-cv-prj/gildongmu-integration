"""Attach optional tracking IDs without filtering or replacing detector output."""
from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import warnings
import numpy as np
from src.risk_config import tracking_config

class DetectionTracker:
    def __init__(self, config=None):
        self.config = tracking_config(config)
        self.backend = None
        self.status = "disabled"
        if self.config["enabled"]:
            try:
                if importlib.util.find_spec("lap") is None:
                    raise ImportError("lap is required; install requirements.txt")
                from ultralytics.trackers.byte_tracker import BYTETracker
                from ultralytics.trackers.bot_sort import BOTSORT
                args = dict(self.config)
                args.update(gmc_method="sparseOptFlow", with_reid=False, model="auto",
                            proximity_thresh=.5, appearance_thresh=.8)
                cls = BOTSORT if args["backend"] == "botsort" else BYTETracker
                self.backend = cls(SimpleNamespace(**args))
                self.status = "active"
            except (ImportError, RuntimeError) as error:
                self.status = "unavailable"
                warnings.warn(f"Tracking unavailable; raw detections retained: {error}", RuntimeWarning)

    def reset(self):
        if self.backend is not None:
            self.backend.reset()

    def attach(self, detections, frame):
        enriched = [{**deepcopy(d), "detection_index": i, "track_id": None}
                    for i, d in enumerate(detections)]
        if self.backend is None:
            return enriched
        try:
            from ultralytics.engine.results import Boxes
            data = np.asarray([list(d["xyxy"]) + [d["confidence"], d["class_id"]]
                               for d in detections], dtype=np.float32).reshape(-1, 6)
            tracks = self.backend.update(Boxes(data, frame.shape[:2]), frame)
            assignments = {}
            for row in tracks:
                index = int(row[-1])
                if index < 0 or index >= len(enriched) or index in assignments:
                    raise ValueError("Invalid tracker detection index")
                assignments[index] = int(row[4])
            for index, track_id in assignments.items():
                enriched[index]["track_id"] = track_id
            self.status = "active"
        except Exception as error:
            # A tracking failure must not suppress a currently detected object.
            if self.status != "failed":
                warnings.warn(f"Tracking failed; raw detections retained: {error}", RuntimeWarning)
            self.status = "failed"
            try:
                self.backend.reset()
            finally:
                self.backend = None
        return enriched
