"""Bounded sidewalk guidance for a fixed-facing moving camera; no route intent inference."""
from copy import deepcopy
import math
import cv2
import numpy as np
from src.roi_recalibration import DirectionCalibration
from src.ground_extent import GroundExtent

class SidewalkGuidedROI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = np.asarray(cfg["corridor_polygon"], float)
        self.top_y = float(self.base[:,1].min())
        self.top_indices = np.flatnonzero(self.base[:,1] == self.top_y)
        self.base_center = float(self.base[self.top_indices,0].mean())
        self.reset()

    def reset(self):
        self.calibration = DirectionCalibration(self.cfg)
        self.ground_extent = GroundExtent(self.cfg)
        self.center = self.base_center
        self.previous_time = None
        self.valid_since = None
        self.last_valid = None
        self.held_center = self.center
        self.last_confidence = 0.0

    def _candidate(self, class_map, label_ids, shape):
        info = {"reason":"mask_unavailable", "confidence":0.0, "band_count":0, "boundary_bands":0}
        if class_map is None or not label_ids or class_map.shape != tuple(shape[:2]):
            return None, info
        if "walkable" not in label_ids:
            return None, info
        h,w = shape[:2]
        small = cv2.resize((class_map == label_ids["walkable"]).astype(np.uint8),
                           (270,max(60,round(h*270/w))),interpolation=cv2.INTER_NEAREST)
        mh,mw = small.shape
        small[:round((.25 if self.cfg["roi_recalibration_enabled"] else .45)*mh)] = 0
        small = cv2.morphologyEx(small,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        _,components,_,_ = cv2.connectedComponentsWithStats(small,connectivity=8)
        seed = components[round(.83*mh):round(.98*mh),round(.2*mw):round(.8*mw)]
        ids,counts = np.unique(seed[seed>0],return_counts=True)
        if not len(ids) or counts.max() < max(12,seed.size*.01):
            info["reason"]="no_near_connected_sidewalk"
            return None,info
        order=np.argsort(counts)[::-1]
        if len(order)>1 and counts[order[1]]>.8*counts[order[0]]:
            info["reason"]="ambiguous_components"
            return None,info
        component=components==ids[order[0]]
        anchors=[]
        bands = np.linspace(.30,.85,12) if self.cfg["roi_recalibration_enabled"] else np.linspace(.55,.9,8)
        for y in bands:
            band=component[max(0,round((y-.015)*mh)):min(mh,round((y+.015)*mh))]
            xs=np.flatnonzero(band.mean(axis=0)>.35)
            if len(xs)<.06*mw:
                continue
            info["band_count"]+=1
            # Unseen boundaries cannot supply direction: an all-walkable image is not a perfect fit.
            if xs[0]<=2 or xs[-1]>=mw-3:
                continue
            gaps=np.diff(xs)
            if (gaps>.12*mw).any():
                continue
            left,right=np.quantile(xs,[.1,.9])/mw
            anchors.append([float(y),float((left+right)/2)])
        info["boundary_bands"]=len(anchors)
        min_bands = 3 if self.cfg["roi_recalibration_enabled"] else 5
        if info["band_count"]<min_bands or len(anchors)<3 or np.ptp(np.asarray(anchors)[:,0])<.12:
            info["reason"]="insufficient_visible_boundaries"
            return None,info
        points=np.asarray(anchors)
        a,b=np.polyfit(points[:,0],points[:,1],1)
        residual=float(np.sqrt(np.mean((points[:,1]-(a*points[:,0]+b))**2)))
        info["fit_residual"]=residual
        if residual>self.cfg["roi_fit_residual"]:
            info["reason"]="unstable_centerline"
            return None,info
        target=float(np.clip(a*self.top_y+b,self.base_center-self.cfg["roi_max_shift"],
                             self.base_center+self.cfg["roi_max_shift"]))
        info.update(reason="sidewalk_centerline",confidence=float(min(1,len(anchors)/6)*(1-residual/self.cfg["roi_fit_residual"])),
                    candidate_center=target)
        return target,info

    def update(self, class_map, label_ids, shape, timestamp, timestamp_valid=True):
        if self.cfg["roi_recalibration_enabled"]:
            extent = (self.ground_extent.update(class_map,label_ids,shape,timestamp,timestamp_valid)
                      if self.cfg["roi_ground_adapt_enabled"] else None)
            if extent is not None:
                self.top_y=extent["top_y"]
            target,info = (self._candidate(class_map,label_ids,shape)
                           if self.cfg["sidewalk_roi_enabled"] and timestamp_valid else
                           (None,{"reason":"disabled_or_invalid_time","confidence":0.0}))
            result=self.calibration.update(target,info,timestamp,timestamp_valid)
            if extent is not None:
                polygons=[]
                for polygon in result["corridor_polygons"]:
                    points=np.asarray(polygon,float)
                    top=points[:,1]==points[:,1].min()
                    points[top,1]=extent["top_y"]
                    # Near ROI remains fixed and is also included in risk geometry.
                    polygons.append(points.tolist())
                result.update(corridor_polygon=polygons[0],corridor_polygons=polygons,
                              path_top_y=extent["top_y"],ground_extent=extent,
                              changed=result["changed"] or extent["changed"])
            return result
        if self.previous_time is not None and (timestamp<=self.previous_time or timestamp-self.previous_time>self.cfg["reset_gap_s"]):
            self.reset()
        dt=0 if self.previous_time is None else timestamp-self.previous_time
        self.previous_time=timestamp
        before=self.center
        target,info=(self._candidate(class_map,label_ids,shape) if self.cfg["sidewalk_roi_enabled"] and timestamp_valid
                     else (None,{"reason":"disabled_or_invalid_time","confidence":0.0,"band_count":0,"boundary_bands":0}))
        source="fixed"
        if target is not None:
            if self.valid_since is None:
                self.valid_since=timestamp
            if timestamp-self.valid_since+1e-9 >= self.cfg["roi_confirm_s"]:
                step=(target-self.center)*(1-math.exp(-dt/self.cfg["roi_smooth_s"]))
                limit=self.cfg["roi_max_shift_per_s"]*dt
                self.center+=float(np.clip(step,-limit,limit))
                self.last_valid=timestamp
                self.held_center=self.center
                self.last_confidence=info["confidence"]
                source="sidewalk"
            elif self.last_valid is not None:
                source="held"
        else:
            self.valid_since=None
        if source != "sidewalk" and self.last_valid is not None:
            age=timestamp-self.last_valid
            if age<=self.cfg["roi_hold_s"]:
                source="held"
            else:
                fraction=min(1,(age-self.cfg["roi_hold_s"])/self.cfg["roi_return_s"])
                self.center=self.held_center*(1-fraction)+self.base_center*fraction
                source="returning" if fraction<1 else "fixed"
        if not timestamp_valid:
            self.center=self.base_center
            self.last_valid=None
            self.valid_since=None
            source="fixed"
        polygon=self.base.copy()
        # Preserve convexity and image bounds; lower vertices never move.
        shift=self.center-self.base_center
        low=-float(self.base[self.top_indices,0].min())
        high=1-float(self.base[self.top_indices,0].max())
        polygon[self.top_indices,0]+=np.clip(shift,low,high)
        if not cv2.isContourConvex(polygon.astype(np.float32)):
            polygon=self.base.copy()
            self.center=self.base_center
            source="fixed"
            info["reason"]="nonconvex_candidate"
        return {**info,"source":source,"corridor_polygon":polygon.tolist(),
                "immediate_polygon":deepcopy(self.cfg["immediate_polygon"]),
                "center_x":self.center,"changed":abs(self.center-before)>1e-5,
                "direction_confidence":self.last_confidence if source=="held" else info["confidence"]}
