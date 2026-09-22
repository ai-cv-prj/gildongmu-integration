"""SESAC-78 이식: 대상 선택·보류·방향·이동 보정 및 표시의 회귀 검사."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.traffic_association import FrameContext, TemporalSelector, estimate_vanishing_point
from src.traffic_motion import estimate_camera_motion, motion_gray, transform_box
from src.visualization import draw_traffic
from test_traffic import fake_pipeline, FRAME, NEAR, FAR, CROSSWALK


class TrackingTests(unittest.TestCase):
    def test_target_kept_despite_confidence_and_order_change_and_reclassified(self):
        pipe = fake_pipeline([(NEAR, .5, 0), (FAR, .7, 0), (CROSSWALK, .9, 1)])
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[620, 280]):
            for _ in range(3):
                first = pipe.predict(FRAME)
        pipe.detector = fake_pipeline([(FAR, .99, 0), (NEAR, .4, 0)]).detector
        pipe._classify.return_value = ('red', .99)
        second = pipe.predict(FRAME)
        self.assertEqual(second['selected_detection_index'], 1)
        self.assertEqual(second['association']['status'], 'tracked')
        self.assertEqual(second['detections'][1]['track_id'], first['detections'][0]['track_id'])
        self.assertEqual(second['signal_state'], 'red')
        self.assertEqual(pipe._classify.call_count, 2)

    def start_challenge(self):
        pipe = fake_pipeline([(NEAR, .8, 0)])
        pipe.predict(FRAME)
        pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .7, 0), (CROSSWALK, .9, 1)]).detector
        return pipe

    def test_challenger_blocks_color_until_three_frames_and_changes_id(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            for fid in (2, 3):
                result = pipe.predict(FRAME, frame_id=fid, captured_at_ms=fid * 200)
                self.assertEqual(result['signal_state'], 'unknown')
                self.assertIsNone(result['selected_detection_index'])
                self.assertEqual(result['candidate_detection_index'], 1)
                self.assertEqual(result['association']['reason'], 'waiting_for_temporal_consistency')
                self.assertTrue(all(d['color_confidence'] is None for d in result['detections']))
            self.assertEqual(pipe._classify.call_count, 1)
            result = pipe.predict(FRAME, frame_id=4, captured_at_ms=800)
        self.assertEqual(result['association']['status'], 'matched')
        self.assertEqual(result['association']['selection_origin'], 'crosswalk_matched')
        self.assertEqual(result['selected_detection_index'], 1)
        self.assertEqual(result['detections'][1]['track_id'], 2)
        self.assertEqual(pipe._classify.call_count, 2)

    def test_conflict_then_missing_geometry_and_single_detection_stay_blocked(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            pipe.predict(FRAME)
        for items in ([(NEAR, .8, 0), (FAR, .7, 0)], [(NEAR, .8, 0)]):
            pipe.detector = fake_pipeline(items).detector
            result = pipe.predict(FRAME)
            self.assertIsNone(result['selected_detection_index'])
            self.assertEqual(result['signal_state'], 'unknown')
        self.assertEqual(pipe._classify.call_count, 1)

    def test_original_target_needs_three_confirmations_after_conflict(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            pipe.predict(FRAME)
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[620, 280]):
            for _ in range(2):
                result = pipe.predict(FRAME)
                self.assertEqual(result['signal_state'], 'unknown')
            result = pipe.predict(FRAME)
        self.assertEqual(result['association']['status'], 'matched')
        self.assertEqual(result['detections'][0]['track_id'], 1)
        self.assertEqual(result['association']['selection_origin'], 'crosswalk_matched')

    def test_flickering_candidates_restart_confirmation(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', side_effect=[
                [1020, 280], [620, 280], [1020, 280], [1020, 280]]):
            for _ in range(4):
                result = pipe.predict(FRAME)
                self.assertIsNone(result['selected_detection_index'])
        self.assertEqual(result['association']['stable_frames'], 2)
        self.assertEqual(pipe._classify.call_count, 1)

    def test_discontinuities_reset_and_do_not_silently_retain_target(self):
        for fid, timestamp in [(3, 200), (2, 1101), (2, -1)]:
            with self.subTest(fid=fid, timestamp=timestamp):
                pipe = fake_pipeline([(NEAR, .8, 0)])
                pipe.predict(FRAME, frame_id=1, captured_at_ms=0)
                pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .7, 0)]).detector
                result = pipe.predict(FRAME, frame_id=fid, captured_at_ms=timestamp)
                self.assertEqual(result['association']['tracking']['reason'], 'discontinuous_frames')
                self.assertIsNone(result['selected_detection_index'])

    def test_missing_target_is_not_restored_and_reset_clears_state(self):
        pipe = fake_pipeline([(NEAR, .8, 0)])
        pipe.predict(FRAME)
        pipe.detector = fake_pipeline([]).detector
        result = pipe.predict(FRAME)
        self.assertEqual(result['detections'], [])
        self.assertIsNone(pipe.selector.target_id)
        pipe.reset()
        self.assertEqual(pipe.frame_id, 0)
        self.assertIsNone(pipe.selector.previous_gray)

    def test_crosswalk_candidates_and_selected_index_remain_aligned(self):
        pipe = fake_pipeline([(NEAR, .8, 0), (FAR, .7, 0),
                              ([50, 300, 900, 680], .4, 1), (CROSSWALK, .9, 1)], stable=1)
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[620, 280]):
            result = pipe.predict(FRAME)
        self.assertEqual(len(result['crosswalks']), 2)
        self.assertEqual(result['detected_crosswalk_count'], 1)
        self.assertEqual(result['crosswalks'][0]['crosswalk_status'], 'below_confidence')
        self.assertEqual(result['crosswalks'][1]['crosswalk_status'], 'used')
        self.assertEqual(result['crosswalk_diagnostics']['selected_crosswalk_index'], 1)

    def test_position_rejection_keeps_bbox_and_reasons(self):
        pipe = fake_pipeline([(NEAR, .8, 0), (FAR, .7, 0), ([300, 100, 980, 300], .9, 1)])
        result = pipe.predict(FRAME)
        self.assertEqual(result['crosswalks'][0]['crosswalk_status'], 'position_rejected')
        self.assertIn('bottom_too_high', result['crosswalks'][0]['exclusion_reasons'])
        self.assertEqual(result['crosswalk_diagnostics']['detection_status'], 'position_rejected')

    def test_diagnostics_use_configured_crosswalk_threshold(self):
        pipe = fake_pipeline([(CROSSWALK, .4, 1)])
        pipe.config['crosswalk_min_confidence'] = .3
        result = pipe.predict(FRAME)
        self.assertEqual(result['crosswalks'][0]['crosswalk_status'], 'eligible')
        self.assertEqual(result['crosswalk_diagnostics']['connection_confidence'], .3)

    def test_overlay_colors_and_detection_vs_selection_labels(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            result = pipe.predict(FRAME)
        with patch('src.visualization.cv2.putText', wraps=cv2.putText) as text:
            rendered = draw_traffic(FRAME, result)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('UNSELECTED det' in label for label in labels))
        self.assertTrue(any('CANDIDATE det' in label for label in labels))
        self.assertTrue(any('COLOR HELD' in label for label in labels))
        self.assertTrue(any('waiting_for_temporal_consistency' in label for label in labels))
        self.assertEqual(rendered[100, 600].tolist(), [255, 140, 79])
        self.assertEqual(rendered[100, 1000].tolist(), [32, 176, 255])
        self.assertTrue(np.all(FRAME == 100))

    def test_same_provisional_target_requires_three_confirmations_and_keeps_id(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[620, 280]):
            for _ in range(2):
                result = pipe.predict(FRAME)
                self.assertIsNone(result['selected_detection_index'])
                self.assertEqual(result['signal_state'], 'unknown')
            pipe._classify.assert_called_once()
            result = pipe.predict(FRAME)
        self.assertEqual(result['selected_detection_index'], 0)
        self.assertEqual(result['detections'][0]['track_id'], 1)
        self.assertEqual(result['association']['selection_origin'], 'crosswalk_matched')

    def test_multiple_signals_without_crosswalk_hold_provisional_color(self):
        pipe = fake_pipeline([(NEAR, .8, 0)])
        first = pipe.predict(FRAME)
        self.assertEqual(first['association']['selection_origin'], 'single_signal')
        pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .9, 0)]).detector
        for _ in range(4):
            result = pipe.predict(FRAME)
            self.assertIsNone(result['selected_detection_index'])
            self.assertEqual(result['signal_state'], 'unknown')
        pipe._classify.assert_called_once()

    def test_confirmed_target_stays_locked_despite_direction_and_order_changes(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            for _ in range(3):
                confirmed = pipe.predict(FRAME)
        target_id = confirmed['detections'][1]['track_id']
        for point in ([620, 280], None, [820, 280]):
            with self.subTest(point=point), patch(
                'src.traffic_association.estimate_vanishing_point', return_value=point
            ):
                for i in range(15):
                    order = [FAR, NEAR] if i % 2 else [NEAR, FAR]
                    pipe.detector = fake_pipeline([(order[0], .99, 0),
                        (order[1], .6, 0), (CROSSWALK, .9, 1)]).detector
                    color = 'red' if i % 2 else 'green'
                    pipe._classify.return_value = (color, .99)
                    result = pipe.predict(FRAME)
                    index = order.index(FAR)
                    self.assertEqual(result['selected_detection_index'], index)
                    self.assertEqual(result['detections'][index]['track_id'], target_id)
                    self.assertEqual(result['signal_state'], color)
                    self.assertEqual(result['association']['reason'], 'previous_target_retained')

    def test_crosswalk_confirmation_survives_detection_order_change(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            self.assertIsNone(pipe.predict(FRAME)['selected_detection_index'])
            pipe.detector = fake_pipeline([(FAR, .9, 0), (NEAR, .8, 0), (CROSSWALK, .9, 1)]).detector
            self.assertIsNone(pipe.predict(FRAME)['selected_detection_index'])
            result = pipe.predict(FRAME)
        self.assertEqual(result['selected_detection_index'], 0)
        self.assertEqual(result['detections'][0]['track_id'], 2)

    def test_crosswalk_disappearance_restarts_provisional_confirmation(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            pipe.predict(FRAME)
            pipe.predict(FRAME)
            pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .9, 0)]).detector
            self.assertIsNone(pipe.predict(FRAME)['selected_detection_index'])
            pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .9, 0), (CROSSWALK, .9, 1)]).detector
            for _ in range(2):
                self.assertIsNone(pipe.predict(FRAME)['selected_detection_index'])
            self.assertEqual(pipe.predict(FRAME)['selected_detection_index'], 1)

    def test_lost_target_immediately_reselects_single_signal_with_new_id(self):
        pipe = fake_pipeline([(NEAR, .8, 0)])
        first = pipe.predict(FRAME)
        pipe.detector = fake_pipeline([(FAR, .9, 0)]).detector
        result = pipe.predict(FRAME)
        self.assertEqual(result['selected_detection_index'], 0)
        self.assertNotEqual(result['detections'][0]['track_id'], first['detections'][0]['track_id'])
        self.assertEqual(result['association']['tracking']['reason'], 'target_missing')

    def test_loss_clears_provisional_confirmation_without_retention(self):
        pipe = self.start_challenge()
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            before = pipe.predict(FRAME)
        pipe.detector = fake_pipeline([]).detector
        missing = pipe.predict(FRAME)
        self.assertEqual(missing['signal_state'], 'unknown')
        self.assertEqual(missing['detections'], [])
        self.assertIsNone(pipe.selector.target_id)
        pipe.detector = fake_pipeline([(NEAR, .8, 0)]).detector
        result = pipe.predict(FRAME)
        self.assertEqual(result['selected_detection_index'], 0)
        self.assertEqual(result['association']['selection_origin'], 'single_signal')
        self.assertNotIn(result['detections'][0]['track_id'],
                         [item['track_id'] for item in before['detections']])

    def test_provisional_confirmation_respects_configured_frame_count(self):
        pipe = fake_pipeline([(NEAR, .8, 0)], stable=2)
        pipe.predict(FRAME)
        pipe.detector = fake_pipeline([(NEAR, .8, 0), (FAR, .7, 0), (CROSSWALK, .9, 1)]).detector
        with patch('src.traffic_association.estimate_vanishing_point', return_value=[1020, 280]):
            self.assertIsNone(pipe.predict(FRAME)['selected_detection_index'])
            self.assertEqual(pipe.predict(FRAME)['selected_detection_index'], 1)


def painted_crossing(count=6, parallel=False):
    frame = np.full((640, 480, 3), 60, dtype=np.uint8)
    for y in [230, 280, 340, 415, 505, 605][:count]:
        corners = [[240 + side * (100 if parallel else .35 * (yy - 100)), yy]
                   for yy, side in [(y - 8, -1), (y - 8, 1), (y + 8, 1), (y + 8, -1)]]
        cv2.fillConvexPoly(frame, np.array(corners, dtype=np.int32), (240, 240, 240))
    return frame


def scene():
    image = np.zeros((640, 480, 3), np.uint8)
    for x, y in np.random.default_rng(38).integers([20, 20], [460, 620], (350, 2)):
        cv2.circle(image, (int(x), int(y)), 3, (180, 180, 180), -1)
    return image


def shifted(image, dx, dy=0):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (480, 640))


class GeometryMotionTests(unittest.TestCase):
    def test_paint_boundaries_estimate_expected_direction(self):
        point = estimate_vanishing_point(painted_crossing(), [0, 180, 480, 640], cv2)
        np.testing.assert_allclose(point, [240, 100], atol=6)

    def test_insufficient_parallel_clipped_and_noisy_stripes_stay_unknown(self):
        for frame, box in [*( (painted_crossing(n), [0, 180, 480, 640]) for n in (0, 1, 2)),
                           (painted_crossing(parallel=True), [0, 180, 480, 640]),
                           (painted_crossing(), [200, 180, 300, 640]),
                           (np.random.default_rng(27).integers(0, 170, (640, 480, 3), dtype=np.uint8), [0, 180, 480, 640])]:
            self.assertIsNone(estimate_vanishing_point(frame, box, cv2))

    def test_motion_compensates_global_shift_but_rejects_local_movement(self):
        image = scene()
        matrix, info = estimate_camera_motion(motion_gray(image, cv2),
            motion_gray(shifted(image, 40, 18), cv2), image.shape[:2], cv2)
        self.assertEqual(info['reason'], 'compensated')
        np.testing.assert_allclose(transform_box([100, 100, 120, 140], matrix), [140, 118, 160, 158], atol=1)
        image[240:] = 0
        matrix, _ = estimate_camera_motion(motion_gray(image, cv2),
            motion_gray(shifted(image, 40), cv2), image.shape[:2], cv2)
        self.assertIsNone(matrix)

    def test_switch_confirmation_survives_camera_shift(self):
        image, selector = scene(), TemporalSelector()
        selector.select(image, [{'xyxy': [100, 100, 120, 140], 'track_id': 1}], [], cv2, FrameContext(1, 0))
        with patch('src.traffic_association.estimate_vanishing_point',
                   side_effect=lambda frame, box, cv: [310 + box[0], 180]):
            for fid, dx in enumerate([40, 80, 120], 2):
                signals = [{'xyxy': [100 + dx, 100, 120 + dx, 140], 'track_id': 1},
                           {'xyxy': [300 + dx, 100, 320 + dx, 140], 'track_id': 2}]
                result = selector.select(shifted(image, dx), signals,
                    [{'xyxy': [dx, 200, 400 + dx, 600]}], cv2, FrameContext(fid, (fid - 1) * 200))
                self.assertEqual(result['tracking']['camera_motion']['reason'],
                                 'compensated' if fid > 2 else 'no_previous_candidate')
                self.assertEqual(result['signal_index'], 1 if fid == 4 else None)
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(result['selection_origin'], 'crosswalk_matched')


if __name__ == '__main__':
    unittest.main()
