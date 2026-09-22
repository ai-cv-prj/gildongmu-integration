"""Bounded warning memory. Visibility loss is not evidence of physical safety."""
LEVEL = {"monitor":0,"caution":1,"danger":2}

def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0

class AlertPolicy:
    def __init__(self,cfg):
        self.cfg=cfg
        self.reset()

    def reset(self):
        self.states={}
        self.counter=0
        self.advisories=[]

    def _spatial_match(self,state,item,timestamp,current_ids):
        same_class = state["class_id"]==item["class_id"]
        if ((not same_class and not self.cfg["class_bridge_enabled"]) or
                timestamp-state["last_seen"]>self.cfg["id_bridge_s"]):
            return False
        old_id,new_id=state["track_id"],item["track_id"]
        threshold=self.cfg["event_match_iou"]
        if old_id is not None and new_id is not None:
            if old_id in current_ids and old_id!=new_id:
                return False
            threshold=self.cfg["id_bridge_iou"]
        if not same_class:
            threshold=max(threshold,.65)
        a,b=state["box"],item["xyxy"]
        old_area=max(1e-6,(a[2]-a[0])*(a[3]-a[1]))
        ratio=(b[2]-b[0])*(b[3]-b[1])/old_area
        return .5<=ratio<=2 and iou(a,b)>=threshold

    def update(self,assessments,timestamp):
        events,used=[],set()
        self.advisories=[]
        current_ids={x["track_id"] for x in assessments if x["track_id"] is not None}
        for item in assessments:
            if item["risk_level"] is None:
                continue
            matches=[(key,s) for key,s in self.states.items() if key not in used
                     and item["track_id"] is not None and s["track_id"]==item["track_id"]
                     and s["class_id"]==item["class_id"]]
            bridged=False
            if not matches:
                candidates=[(key,s) for key,s in self.states.items() if key not in used
                            and self._spatial_match(s,item,timestamp,current_ids)]
                # Require a unique match in both directions; do not transfer alerts across a crowd.
                matches=[(key,s) for key,s in candidates
                         if sum(self._spatial_match(s,x,timestamp,current_ids) for x in assessments
                                if x["risk_level"] is not None)==1]
                bridged=len(matches)==1
            if len(matches)==1:
                key,state=matches[0]
            else:
                self.counter+=1
                key=self.counter
                state={"level":"monitor","last_alert":-float("inf"),"last_support":timestamp,
                       "clear_since":None,"clear_target":None}
                self.states[key]=state
            used.add(key)
            old=state["level"]
            raw=item["risk_level"]
            new=raw
            status="active" if raw!="monitor" else "observed"
            hold_reason=None
            release_reason=None
            if LEVEL[raw]>=LEVEL[old]:
                state["last_support"]=timestamp
                state["clear_since"]=None
                state["clear_target"]=None
            else:
                evidence=item.get("release_evidence")
                if evidence:
                    if state["clear_since"] is None or state["clear_target"]!=raw:
                        state["clear_since"]=timestamp
                    state["clear_target"]=raw
                else:
                    state["clear_since"]=None
                    state["clear_target"]=None
                confirmed=state["clear_since"] is not None and timestamp-state["clear_since"]+1e-9>=self.cfg["clear_confirm_s"]
                if confirmed:
                    release_reason=evidence
                    status="clear_confirmed" if raw=="monitor" else "active"
                elif timestamp-state["last_support"]<self.cfg["uncertainty_hold_s"]:
                    new=old
                    hold_reason="confirming_exit" if evidence else "limited_observation"
                    status="held"
                else:
                    release_reason="uncertainty_timeout"
                    status="uncertain"
            g = item.get("geometry") or {}
            near_now = bool(g.get("close_candidate") or g.get("side_proximity") or
                            g.get("immediate_overlap",0)>=self.cfg["overlap_threshold"])
            if near_now:
                state["last_near"] = timestamp
            # A partially visible box may shrink before the nearby object leaves view.
            near_recent = near_now or (new!="monitor" and
                timestamp-state.get("last_near",-float("inf"))<=self.cfg["visibility_advisory_s"])
            state.update(track_id=item["track_id"],class_id=item["class_id"],box=item["xyxy"],
                         last_seen=timestamp,level=new,
                         direction=g.get("side_direction","front"),
                         near_candidate=near_recent)
            state["visible_warning"] = new if new!="monitor" else None
            item.update(alert_level=new,event_id=key,alert_status=status,hold_reason=hold_reason,
                        release_reason=release_reason,event_identity_bridged=bridged)
            if LEVEL[new]>0 and (LEVEL[new]>LEVEL[old] or timestamp-state["last_alert"]>=self.cfg["repeat_cooldown_s"]):
                kind="escalated" if LEVEL[old]>0 and LEVEL[new]>LEVEL[old] else ("raised" if LEVEL[old]==0 else "repeated")
                events.append({"event_id":key,"type":kind,"level":new,
                               "detection_index":item["detection_index"],"reasons":item["reasons"]})
                state["last_alert"]=timestamp
            elif LEVEL[new]<LEVEL[old]:
                events.append({"event_id":key,"type":"cleared" if new=="monitor" else "deescalated",
                               "level":new,"reason":release_reason,
                               "safety_confirmed":False,
                               "image_path_exit_confirmed":release_reason=="image_path_exit"})
        for key,state in list(self.states.items()):
            age=timestamp-state["last_seen"]
            if (self.cfg["side_proximity_enabled"] and key not in used and
                    state.get("visible_warning") and state.get("near_candidate") and
                    age<=self.cfg["visibility_advisory_s"]):
                self.advisories.append({"event_id":key,"level":"caution",
                    "direction":state.get("direction","front"),"reason":"near_object_out_of_view",
                    "age_s":age,"observed":False,"safety_confirmed":False})
            if key not in used and timestamp-state["last_seen"]>=self.cfg["release_hold_s"]:
                if state["level"]!="monitor":
                    events.append({"event_id":key,"type":"lost","level":"unknown",
                                   "reason":"visibility_lost","safety_confirmed":False})
                state["level"]="monitor"
            if timestamp-state["last_seen"]>max(self.cfg["repeat_cooldown_s"],self.cfg["release_hold_s"]):
                del self.states[key]
        return events
