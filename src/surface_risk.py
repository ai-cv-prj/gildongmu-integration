"""Warnings for observed non-walkable path regions, independent of YOLO boxes.

Semantic labels alone indicate a path check, never metric distance or confirmed
collision. Missing/unstable evidence is not a reason to declare the scene safe.
"""
import cv2
import numpy as np


class SurfaceRisk:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.pending_since = None
        self.clear_since = None
        self.active = False
        self.previous = None
        self.last_event = -float("inf")
        self.last_time = None
        self.last_observed = None

    def update(self, class_map, label_ids, shape, roi, timestamp, valid, camera_stable, detections):
        result = {"enabled":self.cfg["surface_risk_enabled"], "status":"unavailable",
                  "alert_level":None, "regions":[], "reasons":[],
                  "path_nonwalkable_fraction":None, "near_nonwalkable_fraction":None,
                  "suppressed_duplicate":False}
        if not result["enabled"]:
            result["status"] = "disabled"
            return result, []
        if self.last_time is not None and (timestamp<=self.last_time or timestamp-self.last_time>self.cfg["reset_gap_s"]):
            self.reset()
        self.last_time = timestamp
        if class_map is None or not label_ids or class_map.shape != tuple(shape[:2]):
            self.pending_since = None
            self.clear_since = None
            if self.active and self.last_observed is not None and timestamp-self.last_observed<=self.cfg["uncertainty_hold_s"]:
                result.update(status="uncertain",alert_level="caution",reasons=["mask_unavailable"])
            else:
                self.active = False
            return result, []
        if not all(key in label_ids for key in ("walkable","crosswalk","non_walkable")):
            return result, []
        self.last_observed = timestamp
        h,w = shape[:2]
        sw,sh = 324,max(64,round(h*324/w))
        labels = cv2.resize(class_map.astype(np.uint8),(sw,sh),interpolation=cv2.INTER_NEAREST)
        walk = np.isin(labels,[label_ids["walkable"],label_ids["crosswalk"]])
        path,near = np.zeros((sh,sw),np.uint8),np.zeros((sh,sw),np.uint8)
        def pixels(poly):
            return np.rint(np.asarray(poly)*[sw-1,sh-1]).astype(np.int32)
        for poly in roi.get("corridor_polygons",[roi["corridor_polygon"]]):
            cv2.fillPoly(path,[pixels(poly)],1)
        cv2.fillPoly(near,[pixels(roi["immediate_polygon"])],1)
        # The fixed near region can extend outside a vertically adjusted path.
        path=np.maximum(path,near)
        blocked = ((labels==label_ids["non_walkable"]) & (path>0)).astype(np.uint8)
        path_area = max(1,int(path.sum()))
        fraction = float(blocked.sum()/path_area)
        near_fraction = float((blocked*(near>0)).sum()/max(1,near.sum()))
        change = (0.0 if self.previous is None else
                  float(((walk!=self.previous)&(path>0)).sum()/path_area))
        self.previous = walk.copy()
        n,components,stats,_ = cv2.connectedComponentsWithStats(blocked,8)
        candidates = []
        start = min(p[1] for p in roi["corridor_polygon"])
        bands = np.linspace(start,1,5)
        for component_id in range(1,n):
            x,y,bw,bh,area = map(int,stats[component_id])
            if area/path_area < self.cfg["surface_min_area"]:
                continue
            cmask = components==component_id
            band_fraction = max(float(cmask[round(a*sh):round(b*sh)].sum()/
                                max(1,path[round(a*sh):round(b*sh)].sum()))
                                for a,b in zip(bands,bands[1:]))
            if band_fraction < self.cfg["surface_band_fraction"]:
                continue
            contours,_ = cv2.findContours(cmask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours,key=cv2.contourArea)
            poly = cv2.approxPolyDP(contour,2,True).reshape(-1,2)/[sw-1,sh-1]
            if len(poly)<3:
                continue
            candidates.append({"polygon":poly.tolist(),
                "box_norm":[x/sw,y/sh,(x+bw)/sw,(y+bh)/sh],
                "area_fraction":float(area/path_area), "band_fraction":band_fraction})
        candidates.sort(key=lambda x:x["area_fraction"],reverse=True)
        result.update(path_nonwalkable_fraction=fraction,near_nonwalkable_fraction=near_fraction,
                      label_change_fraction=change,regions=candidates[:3],status="clear")
        unreliable = (not valid or not camera_stable or
                      change>=self.cfg["surface_unstable_change"] or fraction>=.80)
        if fraction>=.80:
            result["regions"]=[]
        observed = bool(candidates)
        if not valid:
            self.pending_since = None
            result["status"] = "uncertain"
            return result, []
        events = []
        if observed:
            self.clear_since = None
            if self.pending_since is None:
                self.pending_since = timestamp
            if timestamp-self.pending_since+1e-9>=self.cfg["surface_confirm_s"]:
                self.active = True
            result["status"] = "uncertain" if unreliable else ("active" if self.active else "confirming")
            result["reasons"] = ["path_observation_uncertain" if unreliable else "non_walkable_in_path"]
        else:
            self.pending_since = None
            if self.active:
                # Do not clear on a shaking camera or a single green frame.
                if unreliable:
                    self.clear_since = None
                    result["status"] = "uncertain"
                else:
                    if self.clear_since is None:
                        self.clear_since = timestamp
                    if timestamp-self.clear_since+1e-9>=self.cfg["surface_clear_s"]:
                        self.active = False
                        events.append({"source":"surface","type":"cleared","level":"monitor",
                                       "safety_confirmed":False,"reason":"image_path_visible"})
                    else:
                        result["status"] = "held"
        if self.active:
            result["alert_level"] = "caution"
            if not result["reasons"]:
                result["reasons"] = ["confirming_path_visibility"]
            # Keep semantic observations in the log/overlay but avoid a duplicate event.
            duplicate = False
            for region in candidates:
                x1,y1,x2,y2 = region["box_norm"]
                for item in detections:
                    if item.get("alert_level",item.get("risk_level")) not in ("caution","danger"):
                        continue
                    box = (item.get("geometry") or {}).get("box_norm")
                    if box is None:
                        continue
                    ix=max(0,min(x2,box[2])-max(x1,box[0]))
                    iy=max(0,min(y2,box[3])-max(y1,box[1]))
                    if ix*iy/max(1e-9,(x2-x1)*(y2-y1)) >= .5:
                        duplicate = True
            result["suppressed_duplicate"] = duplicate
            if not duplicate and timestamp-self.last_event>=self.cfg["repeat_cooldown_s"]:
                events.append({"source":"surface","type":"raised" if self.last_event==-float("inf") else "repeated",
                               "level":"caution","reasons":result["reasons"]})
                self.last_event = timestamp
        else:
            self.last_event = -float("inf")
        return result,events
