"""Reported video regressions and shared-profile warning integration."""
from copy import deepcopy
from pathlib import Path
import unittest
import cv2
import numpy as np
import yaml
from src.risk import RiskEngine
from src.risk_config import risk_config
from src.roi_recalibration import DirectionCalibration
from src.surface_risk import SurfaceRisk
from src.alert_policy import AlertPolicy
from src.risk_visualization import draw_risk
from src.warning_groups import group_warnings
from test_risk import detection,engine,FRAME

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/"configs/inference.yaml").read_text())["risk"]
LABELS={"non_walkable":0,"walkable":1,"crosswalk":2}


class SharedProfileTests(unittest.TestCase):
    def test_shipped_config_enables_only_shared_image_profile(self):
        cfg=risk_config(CFG)
        self.assertTrue(cfg["surface_risk_enabled"])
        self.assertTrue(cfg["side_proximity_enabled"])
        self.assertTrue(cfg["ttc_alerts"])
        self.assertLess(min(p[1] for p in cfg["corridor_polygon"]),.372)
        self.assertFalse(any("body" in k or "head_height" in k or "ground_height" in k for k in cfg))

    def test_g25u_person_is_warned_before_leaving_left_edge(self):
        e=engine(config=CFG)
        normalized=[(.371,.201,.476,.372),(.19,.178,.348,.44),(.036,.155,.234,.511)]
        for i,box in enumerate(normalized):
            d=detection([v*100 for v in box])
            item=e.update(FRAME,[d],i*.1)["detections"][0]
            self.assertIn(item["alert_level"],("caution","danger"))

    def test_g25u_close_tree_warns_even_with_clipped_top(self):
        for box in [(12.7,0,35.2,50.7),(0,.1,29.2,65.4)]:
            item=engine(None,config=CFG).update(FRAME,[detection(box,"tree_trunk",27)],0)["detections"][0]
            self.assertEqual(item["risk_level"],"caution")
            self.assertIsNone(item["motion"]["ttc_scale_s"])

    def test_side_proximity_requires_size_and_low_contact(self):
        for box,wanted in [((0,10,20,40),"monitor"),((0,45,5,55),"monitor"),
                           ((0,25,20,60),"caution")]:
            item=engine(None,config=CFG).update(FRAME,[detection(box)],0)["detections"][0]
            self.assertEqual(item["risk_level"],wanted)

    def test_reported_far_pole_does_not_return_to_danger(self):
        f=np.zeros((1920,1080,3),np.uint8)
        item=engine(config=CFG).update(f,[detection([483.43,435.45,557.05,1489.25],"pole",20)],0)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")

    def test_static_contact_uses_full_detected_width(self):
        item=engine(config=CFG).update(FRAME,[detection((5,0,35,65),"tree_trunk",27)],0)["detections"][0]
        self.assertEqual(item["geometry"]["footprint"][::2],[.05,.35])

    def test_surface_warning_exists_without_any_yolo_detection(self):
        e=engine(config=CFG)
        labels=np.ones((100,100),np.uint8)
        labels[42:65,35:52]=0
        for t in (0,.1,.2):
            result=e.update(FRAME,[],t,class_map=labels,label_ids=LABELS)
        self.assertEqual(result["detections"],[])
        self.assertEqual(result["surface"]["alert_level"],"caution")
        self.assertTrue(any(x.get("source")=="surface" for x in result["events"]))
        self.assertTrue(result["surface"]["regions"])

    def test_surrounding_walkability_controls_near_danger(self):
        for val,wanted in ((0,"caution"),(1,"danger")):
            result=engine(None,config=CFG).update(FRAME,[detection()],0,
                         class_map=np.full((100,100),val,np.uint8),label_ids=LABELS)
            self.assertEqual(result["detections"][0]["risk_level"],wanted)

    def test_obstacle_on_nonwalkable_surroundings_is_downgraded(self):
        labels=np.zeros((100,100),np.uint8)
        item=engine(None,config=CFG).update(
            FRAME,[detection((40,55,60,90),"car",3)],0,
            class_map=labels,label_ids=LABELS)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")
        self.assertIn("nonwalkable_surroundings",item["reasons"])

    def test_walkable_obstacle_context_keeps_danger(self):
        labels=np.zeros((100,100),np.uint8)
        labels[76:90,36:40]=1
        item=engine(None,config=CFG).update(
            FRAME,[detection((40,55,60,90),"car",3)],0,
            class_map=labels,label_ids=LABELS)["detections"][0]
        self.assertEqual(item["risk_level"],"danger")

    def test_clipped_context_uses_only_visible_regions(self):
        labels=np.zeros((100,100),np.uint8)
        item=engine(None,config=CFG).update(
            FRAME,[detection((0,55,25,100),"car",3)],0,
            class_map=labels,label_ids=LABELS)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")
        self.assertNotIn("left",item["surrounding_walkability"]["regions"])
        self.assertNotIn("bottom",item["surrounding_walkability"]["regions"])

    def test_missing_surface_mask_cannot_produce_danger(self):
        item=engine(None,config=CFG).update(
            FRAME,[detection((40,55,60,90),"car",3)],0)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")

    def test_approaching_obstacle_without_walkable_surroundings_is_downgraded(self):
        e=engine(config=CFG)
        labels=np.zeros((100,100),np.uint8)
        for t in (0,.1,.2):
            box_height=20/(1-t/.8)
            item=e.update(FRAME,[detection((40,90-box_height,60,90),"car",3)],t,
                          class_map=labels,label_ids=LABELS)["detections"][0]
        self.assertEqual(item["motion"]["approach_state"],"approaching")
        self.assertEqual(item["risk_level"],"caution")

    def test_non_vehicle_obstacle_uses_same_surroundings_rule(self):
        labels=np.zeros((100,100),np.uint8)
        item=engine(None,config=CFG).update(
            FRAME,[detection((40,55,60,90),"person",0)],0,
            class_map=labels,label_ids=LABELS)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")
        self.assertIn("nonwalkable_surroundings",item["reasons"])

    def test_nonwalkable_surroundings_remove_previous_danger_immediately(self):
        e=engine(config=CFG)
        walkable=np.ones((100,100),np.uint8)
        blocked=np.zeros((100,100),np.uint8)
        first=e.update(FRAME,[detection()],0,class_map=walkable,label_ids=LABELS)
        second=e.update(FRAME,[detection()],.1,class_map=blocked,label_ids=LABELS)
        self.assertEqual(first["detections"][0]["alert_level"],"danger")
        self.assertEqual(second["detections"][0]["risk_level"],"caution")
        self.assertEqual(second["detections"][0]["alert_level"],"caution")

    def test_no_id_and_raw_detection_preservation(self):
        raw=[detection((0,25,20,60)),detection((45,30,55,90),"traffic_light",25)]
        saved=deepcopy(raw)
        result=engine(None,config=CFG).update(FRAME,raw,0)
        self.assertEqual(raw,saved)
        self.assertEqual(len(result["detections"]),len(raw))
        self.assertIsNone(result["detections"][1]["risk_level"])
        self.assertTrue(all(d["track_id"] is None for d in result["detections"]))

    def test_warning_grouping_preserves_every_detection_and_highest_level(self):
        cfg=risk_config(CFG)
        items=[
            {"detection_index":0,"class_id":2,"class_name":"car","confidence":.70,
             "xyxy":[10,10,50,50],"risk_level":"caution","alert_level":"caution",
             "event_id":1,"track_id":10,"reasons":["path_occupied"]},
            {"detection_index":1,"class_id":2,"class_name":"car","confidence":.80,
             "xyxy":[11,11,51,51],"risk_level":"danger","alert_level":"danger",
             "event_id":2,"track_id":11,"reasons":["near_path_occupied"]},
            {"detection_index":2,"class_id":2,"class_name":"car","confidence":.90,
             "xyxy":[70,10,95,50],"risk_level":"caution","alert_level":"caution",
             "event_id":3,"track_id":12,"reasons":["path_occupied"]},
        ]
        before=[(x["detection_index"],x["risk_level"],x["alert_level"],x["xyxy"][:]) for x in items]
        groups=group_warnings(items,cfg)
        self.assertEqual(len(items),3)
        self.assertEqual(before,[(x["detection_index"],x["risk_level"],x["alert_level"],x["xyxy"][:])
                                 for x in items])
        self.assertEqual(sorted(g["size"] for g in groups),[1,2])
        merged=next(g for g in groups if g["size"]==2)
        self.assertEqual(merged["level"],"danger")
        self.assertEqual(merged["primary_detection_index"],1)
        self.assertEqual(sum(x.get("warning_primary",False) for x in items[:2]),1)

    def test_risk_engine_keeps_group_members_and_audit_events(self):
        raw=[detection((40,55,60,90)),detection((41,55,61,90))]
        saved=deepcopy(raw)
        result=engine(config=CFG).update(FRAME,raw,0)
        self.assertEqual(raw,saved)
        self.assertEqual(len(result["detections"]),2)
        self.assertEqual(len(result["warning_groups"]),1)
        self.assertEqual(result["warning_groups"][0]["level"],"caution")
        self.assertEqual(result["warning_groups"][0]["size"],2)
        self.assertEqual(len([e for e in result["events"] if e.get("source","object")=="object"]),2)

    def test_warning_grouping_never_combines_different_classes(self):
        cfg=risk_config(CFG)
        items=[
            {"detection_index":0,"class_id":0,"class_name":"person","confidence":.9,
             "xyxy":[10,10,50,50],"risk_level":"danger","alert_level":"danger",
             "event_id":1,"track_id":1,"reasons":[]},
            {"detection_index":1,"class_id":2,"class_name":"car","confidence":.9,
             "xyxy":[10,10,50,50],"risk_level":"danger","alert_level":"danger",
             "event_id":2,"track_id":2,"reasons":[]},
        ]
        self.assertEqual(len(group_warnings(items,cfg)),2)

    def test_warning_grouping_disabled_does_not_touch_detections(self):
        items=[{"detection_index":0,"class_id":2,"class_name":"car","confidence":.9,
                "xyxy":[10,10,50,50],"risk_level":"danger","alert_level":"danger",
                "event_id":1,"track_id":1,"reasons":[]}]
        saved=deepcopy(items)
        self.assertEqual(group_warnings(items,risk_config()),[])
        self.assertEqual(items,saved)

    def test_lost_close_object_has_advisory_without_ghost_box_and_expires(self):
        e=engine(config=CFG)
        e.update(FRAME,[detection((0,25,20,60))],0)
        result=e.update(FRAME,[],.1)
        self.assertEqual(result["detections"],[])
        self.assertEqual(result["advisories"][0]["direction"],"left")
        self.assertFalse(result["advisories"][0]["observed"])
        for t in (.4,.7,1.01):
            result=e.update(FRAME,[],t)
        self.assertEqual(result["advisories"],[])

    def test_partial_box_keeps_recent_close_memory_until_visibility_loss(self):
        e=engine(config=CFG)
        e.update(FRAME,[detection((0,25,20,60))],0)
        partial=e.update(FRAME,[detection((0,25,8,60))],.1)["detections"][0]
        self.assertFalse(partial["geometry"]["close_candidate"])
        self.assertEqual(partial["alert_level"],"caution")
        result=e.update(FRAME,[],.2)
        self.assertEqual(result["detections"],[])
        self.assertEqual(result["advisories"][0]["direction"],"left")

    def test_lost_large_static_candidate_is_remembered_before_surface_warning(self):
        e=engine(config=CFG)
        first=e.update(FRAME,[detection((12.7,0,35.2,50.7),"tree_trunk",27)],0)
        self.assertEqual(first["detections"][0]["risk_level"],"caution")
        result=e.update(FRAME,[],.1)
        self.assertTrue(result["advisories"])
        self.assertEqual(result["detections"],[])

    def test_class_switch_keeps_unique_near_warning_but_resets_motion(self):
        e=engine(config=CFG)
        first=e.update(FRAME,[detection((0,25,20,60),"tree_trunk",27)],0)["detections"][0]
        second=e.update(FRAME,[detection((0,25,20,60),"pole",20)],.1)["detections"][0]
        self.assertEqual(first["event_id"],second["event_id"])
        self.assertTrue(second["event_identity_bridged"])
        self.assertEqual(second["motion"]["history_s"],0)

    def test_review_overlay_with_surface_and_advisory_preserves_source(self):
        e=engine(config=CFG)
        mask=np.ones((100,100),np.uint8);mask[40:75,30:55]=0
        for t in (0,.1,.2):
            result=e.update(FRAME,[detection((0,25,20,60))],t,class_map=mask,label_ids=LABELS)
        result=e.update(FRAME,[],.3,class_map=mask,label_ids=LABELS)
        shown=draw_risk(FRAME,result,risk_config({**CFG,"review_overlay":True}))
        self.assertEqual(shown.shape,FRAME.shape)
        self.assertEqual(int(FRAME.sum()),0)

    def test_upward_view_places_roi_on_visible_ground_not_sky(self):
        e=engine(config=CFG)
        mask=np.zeros((100,100),np.uint8);mask[58:,:]=1
        r=e.update(FRAME,[],0,class_map=mask,label_ids=LABELS)
        self.assertGreater(r["roi"]["path_top_y"],.55)
        self.assertLessEqual(r["roi"]["path_top_y"],.65)
        self.assertEqual(r["roi"]["immediate_polygon"],CFG["immediate_polygon"])

    def test_no_sidewalk_shows_uncertainty_without_filling_sky(self):
        e=engine(config=CFG)
        for t in (0,.1,.2):
            r=e.update(FRAME,[],t,class_map=np.zeros((100,100),np.uint8),label_ids=LABELS)
        self.assertEqual(r["surface"]["alert_level"],"caution")
        self.assertEqual(r["surface"]["status"],"uncertain")
        self.assertEqual(r["surface"]["regions"],[])

    def test_new_state_resets_between_videos(self):
        e=engine(config=CFG)
        labels=np.zeros((100,100),np.uint8)
        for t in (0,.1,.2):
            e.update(FRAME,[detection((0,25,20,60))],t,class_map=labels,label_ids=LABELS)
        e.reset()
        result=e.update(FRAME,[],0,class_map=np.ones_like(labels),label_ids=LABELS)
        self.assertFalse(result["surface"]["alert_level"])
        self.assertEqual(result["advisories"],[])


class DirectionConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=risk_config(CFG)
        self.roi=DirectionCalibration(self.cfg)

    def call(self,target,t):
        return self.roi.update(target,{"confidence":.9 if target is not None else 0,
                                     "reason":"test"},t,True)

    def test_initial_snap_then_ignore_small_jitter(self):
        for t in (0,.1,.2):
            r=self.call(.60,t)
        self.assertAlmostEqual(r["center_x"],.60)
        for t in (.3,.4,.5):
            r=self.call(.61,t)
        self.assertAlmostEqual(r["center_x"],.60)

    def test_large_change_requires_consistency_and_keeps_both_paths(self):
        for t in (0,.1,.2):
            self.call(.60,t)
        r=self.call(.40,.3)
        self.assertEqual(len(r["corridor_polygons"]),2)
        self.assertAlmostEqual(r["center_x"],.60)
        for t in (.4,.5,.6):
            r=self.call(.40,t)
        self.assertAlmostEqual(r["center_x"],.40)
        self.assertEqual(len(r["corridor_polygons"]),1)

    def test_alternating_direction_never_confirms(self):
        for i in range(12):
            r=self.call(.4 if i%2 else .6,i*.05)
        self.assertFalse(r["calibrated"])
        self.assertAlmostEqual(r["center_x"],.5)

    def test_missing_mask_uses_previous_and_fixed_path_union(self):
        for t in (0,.1,.2):
            self.call(.60,t)
        for t in (.5,.8):
            r=self.call(None,t)
        self.assertEqual(r["source"],"fallback")
        self.assertEqual(len(r["corridor_polygons"]),2)
        self.assertAlmostEqual(r["center_x"],.60)


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=risk_config(CFG)
        self.surface=SurfaceRisk(self.cfg)
        self.roi={"corridor_polygon":self.cfg["corridor_polygon"],
                  "immediate_polygon":self.cfg["immediate_polygon"]}

    def call(self,mask,t,valid=True,stable=True):
        return self.surface.update(mask,LABELS,FRAME.shape,self.roi,t,valid,stable,[])

    def test_clear_crosswalk_and_external_wall_do_not_warn(self):
        for value in (1,2):
            mask=np.full((100,100),value,np.uint8);mask[:30,:]=0
            for t in (0,.1,.2):
                result,events=self.call(mask,t)
            self.assertIsNone(result["alert_level"])
            self.assertFalse(events)
            self.surface.reset()

    def test_one_frame_failure_does_not_create_or_clear_warning(self):
        clear=np.ones((100,100),np.uint8);blocked=clear.copy();blocked[45:70,35:55]=0
        self.call(blocked,0)
        r,_=self.call(clear,.1)
        self.assertIsNone(r["alert_level"])
        for t in (.2,.3,.4):
            r,_=self.call(blocked,t)
        self.assertEqual(r["alert_level"],"caution")
        r,_=self.call(clear,.5)
        self.assertEqual(r["alert_level"],"caution")
        for t in (.6,.8,1.01):
            r,events=self.call(clear,t)
        self.assertIsNone(r["alert_level"])
        self.assertFalse(events[0]["safety_confirmed"])

    def test_complete_mask_loss_is_uncertain_not_danger(self):
        for t in (0,.1,.2):
            r,_=self.call(np.zeros((100,100),np.uint8),t)
        self.assertEqual(r["status"],"uncertain")
        self.assertEqual(r["alert_level"],"caution")

    def test_invalid_time_does_not_accumulate_confirmation(self):
        mask=np.zeros((100,100),np.uint8)
        for t in (0,.1,.2,.3):
            r,events=self.call(mask,t,False)
        self.assertIsNone(r["alert_level"])
        self.assertFalse(events)

if __name__=="__main__":
    unittest.main()
