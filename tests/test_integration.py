"""
file_path: tests/test_integration.py

도보·장애물 추론의 입출력 형식, 시각화, 영상 파이프라인을 검증한다.
실제 가중치나 GPU 없이 실행하며, 임시 MP4는 테스트 후 정리한다.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import torch

from src.pipeline import (
    DEFAULT_CONFIG,
    PROJECT_DIR,
    find_sample_videos,
    load_config,
    process_video,
    resolve_path,
    run_video_inference,
)
from src.sidewalk import SidewalkSegmenter, get_segmentation_label_ids
from src.obstacle import EXPECTED_CLASS_NAMES, ObstacleDetector, validate_yolo_config
from src.visualization import (
    LABEL_COLORS,
    OBJECT_COLORS,
    UNKNOWN_OBJECT_COLOR,
    draw_detections,
    overlay_segmentation,
)
from scripts.run_video_inference import main


LABEL_IDS = {"non_walkable": 0, "walkable": 1, "crosswalk": 2}


# 영상 읽기 결과 모사
def make_capture(frame_count=3, reported_frames=3):
    """실제 프레임 수와 영상 메타데이터를 독립적으로 설정한 캡처를 반환한다."""
    capture = Mock()
    capture.isOpened.return_value = True
    metadata = {
        cv2.CAP_PROP_FRAME_WIDTH: 32,
        cv2.CAP_PROP_FRAME_HEIGHT: 24,
        cv2.CAP_PROP_FPS: 12,
        cv2.CAP_PROP_FRAME_COUNT: reported_frames,
    }
    capture.get.side_effect = metadata.__getitem__
    frame = np.full((24, 32, 3), 100, dtype=np.uint8)
    capture.read.side_effect = [(True, frame)] * frame_count + [(False, None)]
    return capture


class IntegrationTests(unittest.TestCase):
    """Mask2Former·YOLO 단독 및 통합 추론의 회귀 테스트."""

    # 기본 설정 및 경로 확인
    def test_default_config_and_paths(self):
        """설정 경로는 프로젝트 루트를 기준으로 해석한다."""
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(resolve_path(config["mask2former"]["weights"]), PROJECT_DIR / "weights/mask2former")
        self.assertNotIn("model_dir", config)
        self.assertEqual(resolve_path(config["output_dir"]), PROJECT_DIR / "outputs/runs/manual")
        self.assertEqual(resolve_path("/tmp/example.mp4"), Path("/tmp/example.mp4"))
        self.assertEqual(config["overlay_alpha"], 0.55)
        self.assertEqual(config["mode"], "both")
        self.assertEqual(config["yolo"]["conf"], 0.25)
        self.assertEqual(config["yolo"]["imgsz"], 640)
        self.assertEqual(config["yolo"]["head"], "nms")

    # 모델별 가중치 설정 검증
    def test_invalid_mask2former_weights_config_rejected(self):
        """누락되거나 잘못된 Mask2Former 가중치 설정은 명확하게 거부한다."""
        config = load_config(DEFAULT_CONFIG)
        for section in (None, "weights/mask2former", {}, {"weights": ""}, {"weights": " "}, {"weights": None}, {"weights": 123}):
            with self.subTest(section=section), patch(
                "src.pipeline.yaml.safe_load", return_value={**config, "mask2former": section}
            ), self.assertRaisesRegex(ValueError, "mask2former"):
                load_config(DEFAULT_CONFIG)

    # 새 옵션과 이전 별칭 확인
    def test_mask2former_cli_option_and_legacy_alias(self):
        """새 옵션과 이전 옵션 모두 동일한 가중치 인자로 전달한다."""
        for option in ("--mask2former-weights", "--model-dir"):
            with self.subTest(option=option), patch("sys.argv", [
                "run_video_inference", option, "weights/custom-mask2former",
                "--yolo-weights", "weights/yolo/custom.pt",
            ]), patch("src.pipeline.run_video_inference") as run:
                main()
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs["mask2former_weights"], Path("weights/custom-mask2former"))
                self.assertEqual(run.call_args.kwargs["yolo_weights"], Path("weights/yolo/custom.pt"))
                self.assertNotIn("model_dir", run.call_args.kwargs)

    # 통일된 가중치 인자로 모델 로딩
    def test_segmenter_loads_weights_directory(self):
        """weights 인자를 전처리기와 모델의 로컬 폴더로 전달한다."""
        weights = PROJECT_DIR / "weights/mask2former"
        with patch.object(Path, "is_file", return_value=True), patch(
            "src.sidewalk.AutoImageProcessor.from_pretrained"
        ) as processor, patch("src.sidewalk.Mask2FormerForUniversalSegmentation.from_pretrained") as model:
            model.return_value.config.id2label = {value: key for key, value in LABEL_IDS.items()}
            segmenter = SidewalkSegmenter(weights=weights, device="cpu")
            processor.assert_called_once_with(weights, local_files_only=True)
            model.assert_called_once_with(weights, local_files_only=True)
            self.assertEqual(segmenter.label_ids, LABEL_IDS)

    # 모델에 저장된 클래스 번호 사용
    def test_label_ids_follow_model_config(self):
        """클래스 번호를 하드코딩하지 않고 모델 설정을 사용한다."""
        model = SimpleNamespace(config=SimpleNamespace(
            id2label={"0": "crosswalk", "1": "non_walkable", "2": "walkable"}
        ))
        self.assertEqual(get_segmentation_label_ids(model), {
            "crosswalk": 0, "non_walkable": 1, "walkable": 2,
        })

    # 예전 2클래스 가중치 차단
    def test_two_class_model_rejected(self):
        """횡단보도가 없는 모델 설정은 거부한다."""
        model = SimpleNamespace(config=SimpleNamespace(
            id2label={0: "non_walkable", 1: "walkable"}
        ))
        with self.assertRaisesRegex(ValueError, "3클래스"):
            get_segmentation_label_ids(model)

    # 전처리 및 출력 좌표 확인
    def test_predict_returns_original_size_class_map(self):
        """BGR→RGB 변환 후 원본 크기의 클래스 지도만 반환한다."""
        frame = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
        original = frame.copy()
        expected = torch.tensor([[0, 1, 2], [2, 1, 0]])
        segmenter = SidewalkSegmenter.__new__(SidewalkSegmenter)
        segmenter.device = torch.device("cpu")
        segmenter.model = Mock()
        segmenter.processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 2, 3)})
        segmenter.processor.post_process_semantic_segmentation.return_value = [expected]

        result = segmenter.predict(frame)

        np.testing.assert_array_equal(result, expected.numpy())
        self.assertTrue(np.issubdtype(result.dtype, np.integer))
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(
            segmenter.processor.call_args.kwargs["images"], frame[:, :, ::-1]
        )
        self.assertEqual(
            segmenter.processor.post_process_semantic_segmentation.call_args.kwargs,
            {"target_sizes": [(2, 3)]},
        )

    # 기존 색상 합성과 동일한지 확인
    def test_overlay_matches_original_algorithm(self):
        """기존의 0.45 원본·0.55 색상 합성과 일치하고 원본은 유지한다."""
        frame = np.random.default_rng(42).integers(0, 256, (10, 12, 3), dtype=np.uint8)
        original = frame.copy()
        class_map = np.tile(np.arange(12) % 3, (10, 1))
        expected = frame.copy()
        for name, color in LABEL_COLORS.items():
            mask = class_map == LABEL_IDS[name]
            overlay = frame.copy()
            overlay[mask] = color
            expected[mask] = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)[mask]

        result = overlay_segmentation(frame, class_map, LABEL_IDS)

        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result[class_map == 0], frame[class_map == 0])
        np.testing.assert_array_equal(frame, original)

    # 잘못된 마스크와 투명도 확인
    def test_overlay_rejects_invalid_inputs(self):
        """프레임과 다른 크기의 지도 및 범위 밖 투명도를 거부한다."""
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            overlay_segmentation(frame, np.zeros((3, 2)), LABEL_IDS)
        for alpha in (-0.1, 1.1, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                overlay_segmentation(frame, np.zeros((2, 3)), LABEL_IDS, alpha)

    # MP4 필터 및 정렬 확인
    def test_sample_videos_sorted_and_filtered(self):
        """확장자 대소문자와 관계없이 MP4만 이름순으로 조회한다."""
        folder = Mock()
        folder.iterdir.return_value = [Path("b.MP4"), Path("ignore.txt"), Path("a.mp4")]
        with patch("src.pipeline.resolve_path", return_value=folder), patch.object(
            Path, "is_file", return_value=True
        ):
            self.assertEqual(find_sample_videos("unused"), [Path("a.mp4"), Path("b.MP4")])

    # 비어 있는 샘플 폴더 확인
    def test_no_sample_videos_rejected(self):
        """영상이 없으면 모델을 불러오기 전에 안내한다."""
        folder = Mock()
        folder.iterdir.return_value = []
        with patch("src.pipeline.resolve_path", return_value=folder):
            with self.assertRaisesRegex(FileNotFoundError, "MP4"):
                find_sample_videos("unused")

    # 여러 영상에서 모델 한 번 로딩
    def test_multiple_videos_load_model_once(self):
        """모든 영상에 같은 모델 인스턴스를 전달한다."""
        videos = [Path("/tmp/a.mp4"), Path("/tmp/b.mp4")]
        with patch("src.pipeline.find_sample_videos", return_value=videos), patch.object(
            Path, "is_file", return_value=True
        ), patch.object(Path, "is_dir", return_value=True), patch.object(
            Path, "exists", return_value=False
        ), patch("src.pipeline.SidewalkSegmenter") as loader, patch(
            "src.pipeline.ObstacleDetector"
        ) as detector_loader, patch(
            "src.pipeline.process_video"
        ) as process, redirect_stdout(io.StringIO()):
            outputs = run_video_inference(device="cpu", mode="sidewalk")
        loader.assert_called_once_with(PROJECT_DIR / "weights/mask2former", device="cpu")
        detector_loader.assert_not_called()
        self.assertEqual(process.call_count, 2)
        self.assertEqual(outputs, [
            PROJECT_DIR / "outputs/runs/manual/result_a.mp4",
            PROJECT_DIR / "outputs/runs/manual/result_b.mp4",
        ])
        for call in process.call_args_list:
            self.assertIs(call.args[2], loader.return_value)

    # 입력 및 기존 결과 덮어쓰기 방지
    def test_existing_output_rejected_before_model_load(self):
        """기존 파일을 출력으로 지정하면 모델 로딩 전에 거부한다."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video, patch(
            "src.pipeline.SidewalkSegmenter"
        ) as loader:
            with self.assertRaises(FileExistsError):
                run_video_inference(video_path=video.name, output_path=video.name)
            loader.assert_not_called()

    # 상충하는 실행 옵션 확인
    def test_conflicting_options_rejected(self):
        """단일 영상과 폴더 또는 출력 파일과 폴더의 동시 지정을 거부한다."""
        with self.assertRaises(ValueError):
            run_video_inference(video_path="a.mp4", sample_dir="samples")
        with self.assertRaises(ValueError):
            run_video_inference(output_path="a.mp4", output_dir="outputs")
        with patch("src.pipeline.find_sample_videos", return_value=[Path("a.mp4"), Path("b.mp4")]):
            with self.assertRaisesRegex(ValueError, "한 개"):
                run_video_inference(output_path="a.mp4")

    # 영상 열기 실패 시 자원 해제
    def test_capture_released_on_open_failure(self):
        """입력 영상을 열지 못해도 캡처 자원을 해제한다."""
        with patch("src.pipeline.cv2.VideoCapture") as capture, patch.object(
            Path, "exists", return_value=False
        ):
            capture.return_value.isOpened.return_value = False
            with self.assertRaises(RuntimeError):
                process_video("missing.mp4", "/tmp/unused.mp4", Mock())
            capture.return_value.release.assert_called_once()

    # 실패 및 Ctrl+C 이후 재실행 확인
    def test_failed_inference_cleans_temporary_file_and_allows_retry(self):
        """추론 오류나 사용자 중단 후 미완성 파일 없이 같은 경로로 재시도한다."""
        class_map = np.ones((24, 32), dtype=np.int64)
        for error in (RuntimeError("추론 오류"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.NamedTemporaryFile(
                suffix=".mp4"
            ) as destination:
                output_path = Path(destination.name)
                destination.close()
                pattern = f".{output_path.stem}.*.partial.mp4"
                segmenter = SimpleNamespace(
                    label_ids=LABEL_IDS,
                    predict=Mock(side_effect=[class_map, error]),
                )
                try:
                    messages = io.StringIO()
                    with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture()), redirect_stdout(messages):
                        with self.assertRaises(type(error)):
                            process_video("mock.mp4", output_path, segmenter)
                    self.assertFalse(output_path.exists())
                    self.assertEqual(list(output_path.parent.glob(pattern)), [])
                    self.assertNotIn("결과 영상 저장:", messages.getvalue())

                    segmenter.predict = Mock(return_value=class_map)
                    with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture()), redirect_stdout(io.StringIO()):
                        self.assertEqual(process_video("mock.mp4", output_path, segmenter), 3)
                    self.assertTrue(output_path.is_file())
                    self.assertEqual(list(output_path.parent.glob(pattern)), [])
                finally:
                    output_path.unlink(missing_ok=True)

    # 영상 읽기 중단 및 메타데이터 불일치 확인
    def test_frame_count_mismatch_does_not_publish_result(self):
        """프레임 수가 다르거나 읽힌 프레임이 없으면 완료 파일을 남기지 않는다."""
        segmenter = SimpleNamespace(
            label_ids=LABEL_IDS, predict=Mock(return_value=np.ones((24, 32), dtype=np.int64))
        )
        for actual, reported in ((1, 3), (3, 1), (0, 3)):
            with self.subTest(actual=actual, reported=reported), tempfile.NamedTemporaryFile(
                suffix=".mp4"
            ) as destination:
                output_path = Path(destination.name)
                destination.close()
                messages = io.StringIO()
                try:
                    with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture(actual, reported)), redirect_stdout(messages):
                        with self.assertRaisesRegex(RuntimeError, "프레임"):
                            process_video("mock.mp4", output_path, segmenter)
                    self.assertFalse(output_path.exists())
                    self.assertEqual(list(output_path.parent.glob(f".{output_path.stem}.*.partial.mp4")), [])
                    self.assertNotIn("결과 영상 저장:", messages.getvalue())
                finally:
                    output_path.unlink(missing_ok=True)

    # 총 프레임 수를 알 수 없는 영상 확인
    def test_unknown_frame_count_warns_and_processes_video(self):
        """유효한 전체 프레임 수가 없으면 경고 후 읽을 수 있는 영상을 저장한다."""
        segmenter = SimpleNamespace(
            label_ids=LABEL_IDS, predict=Mock(return_value=np.ones((24, 32), dtype=np.int64))
        )
        for reported in (0, -1, float("nan"), float("inf")):
            with self.subTest(reported=reported), tempfile.NamedTemporaryFile(suffix=".mp4") as destination:
                output_path = Path(destination.name)
                destination.close()
                try:
                    with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture(2, reported)), redirect_stdout(io.StringIO()):
                        with self.assertWarnsRegex(RuntimeWarning, "누락 여부"):
                            self.assertEqual(process_video("mock.mp4", output_path, segmenter), 2)
                    self.assertTrue(output_path.is_file())
                finally:
                    output_path.unlink(missing_ok=True)

    # 실행 도중 생성된 최종 파일 보호
    def test_output_created_during_inference_is_not_overwritten(self):
        """다른 작업이 최종 경로에 파일을 생성해도 기존 내용을 보존한다."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as destination:
            output_path = Path(destination.name)
            destination.close()

            # 다른 작업의 결과 생성 모사
            def predict(frame):
                """추론 도중 최종 경로에 테스트용 기존 파일을 만든다."""
                output_path.write_bytes(b"existing result")
                return np.ones(frame.shape[:2], dtype=np.int64)

            segmenter = SimpleNamespace(label_ids=LABEL_IDS, predict=predict)
            try:
                with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture(1, 1)), redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileExistsError):
                        process_video("mock.mp4", output_path, segmenter)
                self.assertEqual(output_path.read_bytes(), b"existing result")
                self.assertEqual(list(output_path.parent.glob(f".{output_path.stem}.*.partial.mp4")), [])
            finally:
                output_path.unlink(missing_ok=True)

    # 영상 인코더 초기화 실패 확인
    def test_writer_open_failure_cleans_temporary_file(self):
        """인코더를 열지 못해도 이번 작업의 임시 파일을 정리한다."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as destination:
            output_path = Path(destination.name)
            destination.close()
            try:
                with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture()), patch(
                    "src.pipeline.cv2.VideoWriter"
                ) as writer:
                    writer.return_value.isOpened.return_value = False
                    with self.assertRaisesRegex(RuntimeError, "생성할 수 없습니다"):
                        process_video("mock.mp4", output_path, Mock())
                    writer.return_value.release.assert_called_once()
                self.assertFalse(output_path.exists())
                self.assertEqual(list(output_path.parent.glob(f".{output_path.stem}.*.partial.mp4")), [])
            finally:
                output_path.unlink(missing_ok=True)

    # 실제 MP4 입출력 확인
    def test_mp4_preserves_size_fps_and_frame_count(self):
        """작은 영상의 크기·FPS·프레임 수를 유지하며 추론 결과를 저장한다."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source, tempfile.NamedTemporaryFile(
            suffix=".mp4"
        ) as destination:
            output_path = Path(destination.name)
            destination.close()
            writer = cv2.VideoWriter(source.name, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (32, 24))
            try:
                if not writer.isOpened():
                    self.skipTest("MP4 인코더를 사용할 수 없습니다.")
                for _ in range(3):
                    writer.write(np.full((24, 32, 3), 100, dtype=np.uint8))
            finally:
                writer.release()
            segmenter = SimpleNamespace(
                label_ids=LABEL_IDS,
                predict=Mock(return_value=np.tile(np.arange(32) % 3, (24, 1))),
            )
            try:
                with redirect_stdout(io.StringIO()):
                    detector = SimpleNamespace(predict=Mock(return_value=[{
                        "xyxy": [2, 3, 20, 20], "class_id": 0,
                        "class_name": "person", "confidence": 0.9,
                    }]))
                    count = process_video(source.name, output_path, segmenter, detector=detector)
                self.assertEqual(count, 3)
                self.assertEqual(segmenter.predict.call_count, 3)
                self.assertEqual(detector.predict.call_count, 3)
                for seg_call, det_call in zip(segmenter.predict.call_args_list, detector.predict.call_args_list):
                    self.assertIs(seg_call.args[0], det_call.args[0])
                capture = cv2.VideoCapture(str(output_path))
                try:
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 32)
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 24)
                    self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12.0, places=2)
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                    self.assertEqual(decoded, 3)
                finally:
                    capture.release()
            finally:
                output_path.unlink(missing_ok=True)

    # YOLO 추론 옵션 검증
    def test_yolo_config_validation(self):
        """범위 밖 신뢰도·입력 크기·잘못된 헤드를 실행 전에 거부한다."""
        config = {"weights": "model.pt", "conf": 0.25, "imgsz": 640, "head": "nms"}
        validate_yolo_config(config)
        for key, value in (
            ("conf", -0.1), ("conf", 1.1), ("conf", float("nan")), ("conf", True),
            ("imgsz", 0), ("imgsz", 640.5), ("imgsz", True), ("weights", ""), ("head", "invalid"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_yolo_config({**config, key: value})

    # YOLO 가중치 및 클래스 검증
    def test_detector_loads_local_weights_and_checks_classes(self):
        """한 번 로딩한 탐지 모델의 32클래스 번호·이름을 확인한다."""
        factory = Mock()
        factory.return_value.task = "detect"
        factory.return_value.names = dict(enumerate(EXPECTED_CLASS_NAMES))
        with tempfile.NamedTemporaryFile(suffix=".pt") as weights, patch.dict(
            "sys.modules", {"ultralytics": SimpleNamespace(YOLO=factory)}
        ):
            detector = ObstacleDetector(weights.name, device="cpu")
            self.assertEqual(detector.conf, 0.25)
            self.assertEqual(detector.imgsz, 640)
            factory.assert_called_once_with(weights.name)
            factory.return_value.names = {0: "person"}
            with self.assertRaisesRegex(ValueError, "32클래스"):
                ObstacleDetector(weights.name, device="cpu")

    # 누락된 로컬 가중치 차단
    def test_missing_yolo_weights_rejected_before_import(self):
        """없는 가중치를 자동 다운로드하지 않고 로딩 전에 거부한다."""
        factory = Mock()
        with patch.object(Path, "is_file", return_value=False), patch.dict(
            "sys.modules", {"ultralytics": SimpleNamespace(YOLO=factory)}
        ):
            with self.assertRaises(FileNotFoundError):
                ObstacleDetector("missing.pt", device="cpu")
            factory.assert_not_called()

    # 원본 BGR 및 박스 좌표 보존
    def test_detector_preserves_coordinates_and_passes_approved_options(self):
        """원본 픽셀 좌표를 그대로 반환하고 승인된 NMS 옵션을 전달한다."""
        frame = np.full((60, 100, 3), [10, 20, 30], dtype=np.uint8)
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector.device, detector.conf, detector.imgsz = "cpu", 0.25, 640
        detector.class_names = dict(enumerate(EXPECTED_CLASS_NAMES))
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy = torch.tensor([[12.5, 20, 95, 58]])
        boxes.cls = torch.tensor([1.0])
        boxes.conf = torch.tensor([0.875])
        detector.model = Mock()
        detector.model.predict.return_value = [SimpleNamespace(orig_shape=(60, 100), boxes=boxes)]
        result = detector.predict(frame)
        self.assertEqual(result, [{
            "xyxy": [12.5, 20.0, 95.0, 58.0], "class_id": 1,
            "class_name": "bicycle", "confidence": 0.875,
        }])
        options = detector.model.predict.call_args.kwargs
        self.assertIs(options["source"], frame)
        self.assertEqual(options["imgsz"], 640)
        self.assertEqual(options["conf"], 0.25)
        self.assertIsNone(options["nms"])
        self.assertFalse(options["save"])
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])

    # 미탐지 및 좌표계 오류 구분
    def test_detector_empty_results_and_wrong_shape(self):
        """미탐지는 빈 목록으로 처리하고 다른 해상도의 결과는 거부한다."""
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector.device, detector.conf, detector.imgsz = "cpu", 0.25, 640
        detector.model = Mock()
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        for boxes in (None, []):
            detector.model.predict.return_value = [SimpleNamespace(orig_shape=(24, 32), boxes=boxes)]
            self.assertEqual(detector.predict(frame), [])
        detector.model.predict.return_value = [SimpleNamespace(orig_shape=(640, 640), boxes=None)]
        with self.assertRaisesRegex(ValueError, "높이·너비"):
            detector.predict(frame)

    # 마스크와 객체 박스 동시 표시
    def test_boxes_preserve_mask_and_original_image(self):
        """객체 박스 밖의 마스크와 입력 이미지를 유지한다."""
        frame = np.full((80, 100, 3), 100, dtype=np.uint8)
        mask = np.ones((80, 100), dtype=np.int64)
        mask[:, 50:] = 2
        colored = overlay_segmentation(frame, mask, LABEL_IDS)
        original = colored.copy()
        detection = {"xyxy": [20, 30, 60, 60], "class_name": "person", "confidence": 0.9}
        result = draw_detections(colored, [detection])
        np.testing.assert_array_equal(colored, original)
        np.testing.assert_array_equal(result[30, 20], OBJECT_COLORS["person"])
        np.testing.assert_array_equal(result[75, 10], original[75, 10])
        np.testing.assert_array_equal(result[75, 90], original[75, 90])
        np.testing.assert_array_equal(draw_detections(colored, []), colored)

    # 전체 클래스의 색상 정의 확인
    def test_object_palette_covers_all_classes_with_unique_colors(self):
        """팀의 32클래스에 중복 없는 유효한 BGR 색상이 정의되어 있다."""
        self.assertEqual(set(OBJECT_COLORS), set(EXPECTED_CLASS_NAMES))
        self.assertEqual(len(set(OBJECT_COLORS.values())), len(EXPECTED_CLASS_NAMES))
        for name, color in OBJECT_COLORS.items():
            with self.subTest(class_name=name):
                self.assertEqual(len(color), 3)
                self.assertTrue(all(isinstance(channel, int) and 0 <= channel <= 255 for channel in color))
                self.assertNotIn(color, LABEL_COLORS.values())

    # 객체 종류별 색상 및 입력 순서 독립성 확인
    def test_detection_colors_are_stable_and_labels_match_boxes(self):
        """탐지 순서와 관계없이 같은 클래스의 박스와 글자에 같은 색을 사용한다."""
        frame = np.full((280, 500, 3), 40, dtype=np.uint8)
        original = frame.copy()
        detections = [
            {"xyxy": [20, 60, 80, 120], "class_name": "person", "confidence": 0.9},
            {"xyxy": [180, 60, 240, 120], "class_name": "bicycle", "confidence": 0.8},
            {"xyxy": [340, 60, 400, 120], "class_name": "car", "confidence": 0.7},
            {"xyxy": [20, 180, 80, 240], "class_name": "person", "confidence": 0.6},
        ]
        with patch("src.visualization.cv2.putText", wraps=cv2.putText) as put_text:
            result = draw_detections(frame, detections)
        self.assertEqual(put_text.call_count, len(detections))
        for detection, call in zip(detections, put_text.call_args_list):
            color = OBJECT_COLORS[detection["class_name"]]
            x1, y1, _, _ = detection["xyxy"]
            np.testing.assert_array_equal(result[y1 + 10, x1], color)
            self.assertEqual(call.args[5], color)
            self.assertEqual(call.args[1], f"{detection['class_name']} {detection['confidence']:.2f}")
        np.testing.assert_array_equal(result, draw_detections(frame, list(reversed(detections))))
        np.testing.assert_array_equal(result, draw_detections(frame, detections))
        np.testing.assert_array_equal(frame, original)

    # 미등록 클래스의 표시 확인
    def test_unknown_detection_class_uses_fallback_color(self):
        """색상표에 없는 클래스는 오류 없이 회색으로 표시한다."""
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        detections = [{"xyxy": [20, 30, 60, 60], "class_name": "unknown", "confidence": 0.9}]
        result = draw_detections(frame, detections)
        np.testing.assert_array_equal(result[40, 20], UNKNOWN_OBJECT_COLOR)

    # 통합 및 장애물 단독 모델 재사용
    def test_models_load_once_and_cli_overrides_config(self):
        """모드에 필요한 모델만 로딩하고 명령어 옵션을 우선 적용한다."""
        for mode in ("both", "obstacle"):
            with self.subTest(mode=mode), patch("src.pipeline.find_sample_videos", return_value=[
                Path("/tmp/a.mp4"), Path("/tmp/b.mp4")
            ]), patch.object(Path, "is_file", return_value=True), patch.object(
                Path, "is_dir", return_value=True
            ), patch.object(Path, "exists", return_value=False), patch(
                "src.pipeline.SidewalkSegmenter"
            ) as seg, patch("src.pipeline.ObstacleDetector") as det, patch(
                "src.pipeline.process_video"
            ) as process, redirect_stdout(io.StringIO()):
                det.return_value.device = "cpu"
                run_video_inference(
                    mode=mode, device="cpu", mask2former_weights="/tmp/custom-mask2former",
                    yolo_weights="/tmp/custom.pt", conf=0.4, imgsz=320,
                )
                det.assert_called_once_with(Path("/tmp/custom.pt"), device="cpu", conf=0.4, imgsz=320, head="nms")
                self.assertEqual(seg.call_count, int(mode == "both"))
                if mode == "both":
                    seg.assert_called_once_with(Path("/tmp/custom-mask2former"), device="cpu")
                self.assertEqual(process.call_count, 2)
                for call in process.call_args_list:
                    self.assertIs(call.kwargs["detector"], det.return_value)
                    self.assertIs(call.args[2], seg.return_value if mode == "both" else None)

    # 장애물 단독 처리 및 오류 정리
    def test_obstacle_only_video_and_failure_cleanup(self):
        """도보 모델 없이 처리하고 YOLO 오류 발생 시 임시 파일을 정리한다."""
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.NamedTemporaryFile(suffix=".mp4") as destination:
                output_path = Path(destination.name)
                destination.close()
                detector = SimpleNamespace(predict=Mock(return_value=[]))
                if failure:
                    detector.predict.side_effect = RuntimeError("YOLO 오류")
                try:
                    with patch("src.pipeline.cv2.VideoCapture", return_value=make_capture()), redirect_stdout(io.StringIO()):
                        if failure:
                            with self.assertRaisesRegex(RuntimeError, "YOLO 오류"):
                                process_video("mock.mp4", output_path, detector=detector)
                            self.assertFalse(output_path.exists())
                        else:
                            self.assertEqual(process_video("mock.mp4", output_path, detector=detector), 3)
                            self.assertTrue(output_path.is_file())
                    self.assertEqual(list(output_path.parent.glob(f".{output_path.stem}.*.partial.mp4")), [])
                finally:
                    output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
