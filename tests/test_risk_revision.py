"""Regression coverage for reported pole/person failures and uncertain visibility."""
import unittest
from copy import deepcopy
from types import SimpleNamespace
import cv2
import numpy as np
from src.risk_config import risk_config
from src.risk import RiskEngine
from src.path_roi import SidewalkGuidedROI
from src.alert_policy import AlertPolicy
from test_risk import detection,FixedTracker,engine,FRAME

class RevisionTests(unittest.TestCase):
    def test_reported_pole_remains_candidate_not_danger(self):
        frame=np.zeros((1920,1080,3),np.uint8)
        box=[483.43176,435.45465,557.04510,1489.24927]
        item=engine(config={"ttc_alerts":True}).update(frame,[detection(box,"pole",20)],0)["detections"][0]
        self.assertEqual(item["risk_level"],"caution")
        self.assertEqual(item["proximity"]["band"],"middle")
        self.assertNotIn("near_path_occupied",item["reasons"])

    def test_reported_approaching_person_keeps_near_risk(self):
        frame=np.zeros((1920,1080,3),np.uint8)
        boxes=[[799.2517,997.7887,972.3651,1518.3188],
               [800,970,1000,1610],[810,850,1030,1720]]
        e=engine(config={"ttc_alerts":True})
        for i,box in enumerate(boxes):
            item=e.update(frame,[detection(box)],i*.1)["detections"][0]
            self.assertEqual(item["risk_level"],"danger")
            self.assertEqual(item["alert_level"],"danger")

    def test_far_static_object_and_immediately_near_static_object(self):
        for bottom,wanted in ((60,"monitor"),(77,"caution"),(92,"danger")):
            d=detection((49,10,51,bottom),"pole",20)
            item=engine(None).update(FRAME,[d],0)["detections"][0]
            self.assertEqual(item["risk_level"],wanted)
            self.assertIsNone(item["track_id"])

    def test_static_screen_motion_is_not_mobile_lateral_entry(self):
        e=engine()
        for t in (0,.1,.2):
            result=e.update(FRAME,[detection((2+50*t,35,12+50*t,65),"pole",20)],t)
        self.assertNotIn("lateral_entry",result["detections"][0]["reasons"])
        self.assertIsNone(result["detections"][0]["motion"]["time_to_corridor_s"])

    def test_capped_ground_strip_for_tall_objects(self):
        item=engine().update(FRAME,[detection((40,0,60,95),"pole",20)],0)["detections"][0]
        f=item["geometry"]["footprint"]
        self.assertLessEqual(f[3]-f[1],.020001)
        self.assertTrue(item["geometry"]["clipped"])
        self.assertTrue(item["proximity"]["contact_reliable"])

    def test_invalid_exit_config_rejected(self):
        for cfg in ({"exit_overlap_threshold":.3},{"static_caution_y":.9},
                    {"clear_confirm_s":2},{"static_ground_classes":"pole"}):
            with self.assertRaises(ValueError):risk_config(cfg)

class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg=risk_config()
        self.roi=SidewalkGuidedROI(self.cfg)
        self.labels={"walkable":1,"crosswalk":2,"non_walkable":0}
        self.shape=(480,270,3)

    def corridor_mask(self):
        a=np.zeros(self.shape[:2],np.uint8)
        cv2.fillPoly(a,[np.array([[155,216],[185,216],[235,479],[65,479]],np.int32)],1)
        return a

    def test_all_walkable_and_empty_have_no_direction_confidence(self):
        for value in (0,1):
            for t in (0,.1,.2,.3,.4):
                result=self.roi.update(np.full(self.shape[:2],value,np.uint8),self.labels,self.shape,t)
            self.assertEqual(result["source"],"fixed")
            self.assertEqual(result["center_x"],.5)
            self.assertEqual(result["direction_confidence"],0)

    def test_centerline_is_bounded_and_lower_roi_does_not_move(self):
        a=self.corridor_mask()
        before=.5
        for t in np.arange(0,1,.1):
            result=self.roi.update(a,self.labels,self.shape,float(t))
            self.assertLessEqual(abs(result["center_x"]-before),.012001)
            before=result["center_x"]
        self.assertEqual(result["source"],"sidewalk")
        self.assertGreater(result["center_x"],.5)
        self.assertLessEqual(result["center_x"],.65)
        self.assertEqual(result["corridor_polygon"][2:],[[.98,1.],[.02,1.]])
        self.assertEqual(result["immediate_polygon"],self.cfg["immediate_polygon"])

    def test_missing_mask_holds_then_returns_without_jump(self):
        for t in np.arange(0,1,.1):
            result=self.roi.update(self.corridor_mask(),self.labels,self.shape,float(t))
        center=result["center_x"]
        result=self.roi.update(None,None,self.shape,1.1)
        self.assertEqual(result["source"],"held")
        self.assertAlmostEqual(result["center_x"],center)
        for t in (1.3,1.5,1.7,1.9,2.1):
            result=self.roi.update(None,None,self.shape,t)
        self.assertEqual(result["source"],"fixed")
        self.assertAlmostEqual(result["center_x"],.5)

    def test_timestamp_reset_and_crosswalk_only_fallback(self):
        for t in (0,.1,.2,.3,.4):
            self.roi.update(self.corridor_mask(),self.labels,self.shape,t)
        result=self.roi.update(None,None,self.shape,.2,False)
        self.assertEqual(result["source"],"fixed")
        result=self.roi.update(self.corridor_mask()*2,self.labels,self.shape,.3)
        self.assertEqual(result["source"],"fixed")

class WarningMemoryTests(unittest.TestCase):
    def setUp(self):
        self.policy=AlertPolicy(risk_config())

    def item(self,level="danger",identity=1,box=(40,40,60,90),evidence=None):
        return {**detection(box),"track_id":identity,"detection_index":0,"risk_level":level,
                "reasons":[],"release_evidence":evidence}

    def update(self,t,**kw):
        item=self.item(**kw)
        events=self.policy.update([item],t)
        return item,events

    def test_exit_requires_sustained_evidence(self):
        self.update(0)
        for t in (.1,.3,.59):
            item,_=self.update(t,level="monitor",evidence="image_path_exit")
            self.assertEqual(item["alert_level"],"danger")
        item,events=self.update(.61,level="monitor",evidence="image_path_exit")
        self.assertEqual(item["alert_level"],"monitor")
        self.assertEqual(item["release_reason"],"image_path_exit")
        self.assertFalse(events[0]["safety_confirmed"])

    def test_unknown_evidence_expires_without_claiming_safe(self):
        self.update(0)
        item,_=self.update(.5,level="monitor")
        self.assertEqual(item["alert_level"],"danger")
        item,events=self.update(.81,level="monitor")
        self.assertEqual(item["alert_status"],"uncertain")
        self.assertEqual(events[0]["reason"],"uncertainty_timeout")
        self.assertFalse(events[0]["safety_confirmed"])

    def test_disappearance_emits_lost_no_ghost_or_safety_claim(self):
        self.update(0)
        self.assertEqual(self.policy.update([],.3),[])
        event=self.policy.update([],.51)[0]
        self.assertEqual(event["type"],"lost")
        self.assertEqual(event["level"],"unknown")
        self.assertFalse(event["safety_confirmed"])

    def test_confirmed_id_switch_carries_warning_only_for_unique_match(self):
        old,_=self.update(0)
        new,_=self.update(.1,identity=2,level="monitor",box=(41,40,61,90))
        self.assertEqual(new["event_id"],old["event_id"])
        self.assertEqual(new["alert_level"],"danger")
        self.assertTrue(new["event_identity_bridged"])

    def test_ambiguous_people_do_not_inherit_each_others_warning(self):
        old,_=self.update(0)
        a=self.item("monitor",2,(39,40,59,90))
        b=self.item("monitor",3,(41,40,61,90))
        self.policy.update([a,b],.1)
        self.assertNotEqual(a["event_id"],old["event_id"])
        self.assertNotEqual(b["event_id"],old["event_id"])
        self.assertEqual(a["alert_level"],"monitor")
        self.assertEqual(b["alert_level"],"monitor")

if __name__=="__main__":
    unittest.main()
