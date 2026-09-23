"""신호등 선택·색상 분류·도보/장애물 통합의 회귀 테스트."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from src.traffic import DEFAULTS, TrafficSignalPipeline, validate_traffic_config
from src.traffic_association import TemporalSelector, associate
from src.visualization import draw_traffic
from src.pipeline import process_video, run_video_inference
from scripts.run_video_inference import main


FRAME = np.full((720, 1280, 3), 100, dtype=np.uint8)
NEAR = [600, 100, 640, 180]
FAR = [1000, 100, 1040, 180]
CROSSWALK = [300, 350, 980, 700]


def fake_pipeline(items, stable=3):
    pipeline = TrafficSignalPipeline.__new__(TrafficSignalPipeline)
    pipeline.config = {**DEFAULTS, "association_stable_frames": stable}
    pipeline.device = "cpu"
    pipeline.reset()
    boxes = SimpleNamespace(
        xyxy=torch.tensor([box for box, _, _ in items]),
        conf=torch.tensor([score for _, score, _ in items]),
        cls=torch.tensor([cid for _, _, cid in items]),
    )
    boxes.cpu = lambda: boxes
    result = SimpleNamespace(orig_shape=FRAME.shape[:2], boxes=boxes)
    pipeline.detector = SimpleNamespace(predict=Mock(return_value=[result]))
    pipeline._classify = Mock(return_value=("green", 0.95))
    return pipeline


class TrafficTests(unittest.TestCase):
    def test_single_signal_needs_no_crosswalk_or_temporal_wait(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0)])
        result = pipeline.predict(FRAME)
        self.assertEqual(result["signal_state"], "green")
        self.assertEqual(result["association"]["status"], "single_signal")
        self.assertEqual(result["association"]["reason"], "crosswalk_relation_unverified")
        self.assertEqual(len(result["detections"]), 1)
        pipeline._classify.assert_called_once()

    def test_multiple_signals_wait_then_classify_only_selected(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0), (FAR, 0.9, 0), (CROSSWALK, 0.9, 1)])
        with patch("src.traffic_association.estimate_vanishing_point", return_value=[640, 280]):
            for _ in range(2):
                result = pipeline.predict(FRAME)
                self.assertEqual(result["signal_state"], "unknown")
                self.assertEqual(len(result["detections"]), 2)
                self.assertEqual(result["detections"][0]["selection_status"], "candidate")
                pipeline._classify.assert_not_called()
            result = pipeline.predict(FRAME)
        self.assertEqual(result["association"]["status"], "matched")
        self.assertEqual(result["detections"][0]["xyxy"], NEAR)
        self.assertEqual(result["signal_state"], "green")
        pipeline._classify.assert_called_once()

    def test_multiple_without_crosswalk_does_not_guess(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0), (FAR, 0.9, 0)])
        result = pipeline.predict(FRAME)
        self.assertEqual(len(result["detections"]), 2)
        self.assertTrue(all(d["selection_status"] == "unselected" for d in result["detections"]))
        self.assertEqual(result["signal_state"], "unknown")
        pipeline._classify.assert_not_called()

    def test_no_signal_resets_temporal_history(self):
        pipeline = fake_pipeline([])
        pipeline.selector.streak = 5
        result = pipeline.predict(FRAME)
        self.assertEqual(result["association"]["reason"], "no_signal_detected")
        self.assertEqual(pipeline.selector.streak, 0)

    def test_unavailable_vanishing_point_does_not_guess(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0), (FAR, 0.9, 0), (CROSSWALK, 0.9, 1)])
        with patch("src.traffic_association.estimate_vanishing_point", return_value=None):
            result = pipeline.predict(FRAME)
        self.assertEqual(result["association"]["reason"], "vanishing_point_unavailable")
        pipeline._classify.assert_not_called()

    def test_ambiguous_signals_do_not_force_selection(self):
        signals = [{"xyxy": [590, 100, 630, 180]}, {"xyxy": [650, 100, 690, 180]}]
        with patch("src.traffic_association.estimate_vanishing_point", return_value=[640, 280]):
            result = associate(FRAME, signals, [{"xyxy": CROSSWALK}], cv2)
        self.assertEqual(result["reason"], "ambiguous_signals")
        self.assertIsNone(result["signal_index"])

    def test_size_breaks_close_direction_tie(self):
        signals = [{"xyxy": [600, 100, 630, 160]}, {"xyxy": [660, 100, 675, 130]}]
        with patch("src.traffic_association.estimate_vanishing_point", return_value=[640, 280]):
            result = associate(FRAME, signals, [{"xyxy": CROSSWALK}], cv2)
        self.assertEqual(result["signal_index"], 0)

    def test_box_identity_survives_detection_order_change(self):
        selector = TemporalSelector(2)
        decision = {"status": "candidate", "signal_index": 0, "crosswalk_index": 0}
        selector.update(dict(decision), [{"xyxy": NEAR, "track_id": 1}, {"xyxy": FAR, "track_id": 2}], [{"xyxy": CROSSWALK}])
        decision["signal_index"] = 1
        result = selector.update(decision, [{"xyxy": FAR, "track_id": 2}, {"xyxy": NEAR, "track_id": 1}], [{"xyxy": CROSSWALK}])
        self.assertEqual(result["status"], "matched")

    def test_thresholds_for_signals_and_crosswalks_are_independent(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0), (FAR, 0.6, 0), (CROSSWALK, 0.6, 1)])
        pipeline.config["conf"] = 0.8
        result = pipeline.predict(FRAME)
        self.assertEqual(result["detected_signal_count"], 1)
        self.assertEqual(len(result["crosswalks"]), 1)
        self.assertEqual(pipeline.detector.predict.call_args.kwargs["conf"], 0.1)

    def test_classifier_uses_checkpoint_order_and_confidence(self):
        pipeline = fake_pipeline([])
        del pipeline._classify
        pipeline.class_names = ["green", "red"]
        pipeline.classifier = Mock(return_value=torch.tensor([[0.0, 5.0]]))
        frame = np.zeros_like(FRAME)
        frame[:, :, 2] = 255
        color, confidence = pipeline._classify(frame, NEAR)
        self.assertEqual(color, "red")
        self.assertGreater(confidence, 0.99)
        batch = pipeline.classifier.call_args.args[0]
        self.assertEqual(tuple(batch.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(batch[0, 0, 112, 112].item(), (1 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(batch[0, 0, 0, 0].item(), -0.485 / 0.229, places=5)
        pipeline.classifier.return_value = torch.zeros((1, 2))
        self.assertEqual(pipeline._classify(frame, NEAR)[0], "unknown")

    def test_tiny_crop_skips_classifier(self):
        pipeline = fake_pipeline([])
        del pipeline._classify
        pipeline.classifier = Mock()
        self.assertEqual(pipeline._classify(FRAME, [0, 0, 1, 1]), ("unknown", None))
        pipeline.classifier.assert_not_called()

    def test_overlay_preserves_input_and_uses_signal_color(self):
        pipeline = fake_pipeline([(NEAR, 0.9, 0), (CROSSWALK, 0.9, 1)])
        result = pipeline.predict(FRAME)
        frame = FRAME.copy()
        rendered = draw_traffic(frame, result)
        np.testing.assert_array_equal(frame, FRAME)
        self.assertEqual(rendered[100, 600].tolist(), [0, 255, 0])
        self.assertEqual(rendered[350, 300].tolist(), [255, 123, 181])

    def test_invalid_settings(self):
        config = {"weights": "yolo.pt", "classifier_weights": "classifier.pt"}
        for key, value in (("conf", float("nan")), ("association_stable_frames", 0),
                           ("imgsz", True), ("classifier_weights", "")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_traffic_config({**config, key: value})

    def test_cli_accepts_new_modes_and_weight_paths(self):
        for mode in ("traffic", "all"):
            with patch("sys.argv", ["run", "--mode", mode, "--traffic-weights", "signal.pt",
                                    "--traffic-classifier-weights", "color.pt"]), patch(
                "src.pipeline.run_video_inference"
            ) as run:
                main()
            self.assertEqual(run.call_args.kwargs["mode"], mode)
            self.assertEqual(run.call_args.kwargs["traffic_weights"], Path("signal.pt"))

    def test_modes_load_only_requested_models_once(self):
        for mode in ("traffic", "both", "all"):
            with patch("src.pipeline.find_sample_videos", return_value=[Path("/tmp/a.mp4"), Path("/tmp/b.mp4")]), \
                 patch.object(Path, "is_file", return_value=True), \
                 patch.object(Path, "is_dir", return_value=True), \
                 patch.object(Path, "exists", return_value=False), \
                 patch("src.pipeline.SidewalkSegmenter") as sidewalk, \
                 patch("src.pipeline.ObstacleDetector") as obstacle, \
                 patch("src.pipeline.TrafficSignalPipeline") as traffic, \
                 patch("src.pipeline.process_video") as process, redirect_stdout(io.StringIO()):
                sidewalk.return_value.device = torch.device("cpu")
                obstacle.return_value.device = "cpu"
                traffic.return_value.device = "cpu"
                run_video_inference(mode=mode, device="cpu")
            self.assertEqual(traffic.call_count, int(mode != "both"))
            self.assertEqual(sidewalk.call_count, int(mode != "traffic"))
            self.assertEqual(obstacle.call_count, int(mode != "traffic"))
            self.assertEqual(process.call_count, 2)
            if mode == "all":
                self.assertEqual(traffic.call_args.kwargs["device"], "cpu")

    def test_video_resets_state_and_suppresses_generic_traffic_boxes(self):
        prediction = fake_pipeline([]).predict(FRAME)
        traffic = SimpleNamespace(reset=Mock(), predict=Mock(return_value=prediction))
        obstacle = SimpleNamespace(predict=Mock(return_value=[
            {"class_name": "traffic_light"}, {"class_name": "person"}
        ]))
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_COUNT: 1}.__getitem__
        with tempfile.TemporaryDirectory() as folder, \
             patch("src.pipeline.cv2.VideoCapture", return_value=capture), \
             patch("src.pipeline.draw_detections", side_effect=lambda frame, detections: frame) as draw, \
             redirect_stdout(io.StringIO()):
            for i in range(2):
                capture.read.side_effect = [(True, FRAME.copy()), (False, None)]
                self.assertEqual(process_video("mock.mp4", Path(folder) / f"{i}.mp4",
                                               detector=obstacle, traffic=traffic), 1)
        self.assertEqual(traffic.reset.call_count, 2)
        self.assertEqual(draw.call_args.args[1], [{"class_name": "person"}])
        np.testing.assert_array_equal(traffic.predict.call_args.args[0], FRAME)
        self.assertEqual(traffic.predict.call_args.kwargs, {"frame_id": 1, "captured_at_ms": 0.0})

    def test_traffic_failure_does_not_publish_partial_video(self):
        traffic = SimpleNamespace(reset=Mock(), predict=Mock(side_effect=RuntimeError("traffic failed")))
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_COUNT: 1}.__getitem__
        capture.read.return_value = (True, FRAME)
        with tempfile.TemporaryDirectory() as folder, patch(
            "src.pipeline.cv2.VideoCapture", return_value=capture
        ):
            with self.assertRaisesRegex(RuntimeError, "traffic failed"):
                process_video("mock.mp4", Path(folder) / "out.mp4", traffic=traffic)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
