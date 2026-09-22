"""Estimate visible walkable-ground extent, not physical ground height or distance."""
from collections import deque
import cv2
import numpy as np


class GroundExtent:
    def __init__(self, cfg):
        self.cfg=cfg
        self.base_y=min(p[1] for p in cfg["corridor_polygon"])
        self.top_y=self.base_y
        self.previous_time=None
        self.samples=deque()

    def update(self, class_map, label_ids, shape, timestamp, valid=True):
        if self.previous_time is not None and (timestamp<=self.previous_time or
                timestamp-self.previous_time>self.cfg["reset_gap_s"]):
            self.__init__(self.cfg)
        self.previous_time=timestamp
        before=self.top_y
        reason="mask_unavailable"
        target=self.base_y
        if class_map is not None and label_ids and class_map.shape==tuple(shape[:2]) and valid:
            h,w=shape[:2];mh=max(64,round(h*270/w));mw=270
            mask=cv2.resize(np.isin(class_map,[label_ids.get("walkable",-1),
                label_ids.get("crosswalk",-1)]).astype(np.uint8),(mw,mh),interpolation=cv2.INTER_NEAREST)
            _,components,_,_=cv2.connectedComponentsWithStats(mask,8)
            seed=components[round(.80*mh):round(.98*mh),round(.2*mw):round(.8*mw)]
            ids,counts=np.unique(seed[seed>0],return_counts=True)
            if len(ids):
                component=components==ids[int(np.argmax(counts))]
                widths=component[:,round(.15*mw):round(.85*mw)].sum(axis=1)
                support=widths>=.08*mw
                length=max(3,round(.04*mh))
                runs=np.convolve(support.astype(int),np.ones(length,dtype=int),"valid")
                candidates=np.flatnonzero((runs>=length)&(np.arange(len(runs))>=round(.20*mh)))
                if len(candidates):
                    target=float(np.clip(candidates[0]/mh+.04,self.base_y,.65))
                    reason="connected_walkable_extent"
                else:
                    target=max(self.base_y,.45);reason="ground_extent_uncertain"
            else:
                target=max(self.base_y,.45);reason="ground_extent_uncertain"
        self.samples.append((timestamp,target))
        while self.samples and timestamp-self.samples[0][0]>.30:
            self.samples.popleft()
        # Use the available extent immediately on the first frame, then a short median.
        self.top_y=float(np.median([v for _,v in self.samples]))
        return {"top_y":self.top_y,"reason":reason,"changed":abs(before-self.top_y)>.01}
