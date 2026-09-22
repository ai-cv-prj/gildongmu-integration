"""보행자 신호등 YOLO → 횡단보도 연결 → MobileNetV3 색상 분류.

선택 로직은 gildongmu-test-app의 SESAC-73 버전을 이식한다.
입출력 좌표는 원본 BGR 프레임 픽셀 기준이다.
"""
import math
from pathlib import Path

import cv2
import torch

from src.traffic_association import FrameContext, TemporalSelector, crosswalk_diagnostics


CLASS_NAMES = {0: "pedestrian_signal", 1: "crosswalk"}
DEFAULTS = {
    "conf": 0.25,
    "imgsz": 960,
    "crosswalk_min_confidence": 0.50,
    "classifier_min_confidence": 0.60,
    "association_stable_frames": 3,
}


def validate_traffic_config(config):
    """기존 2클래스 검출기와 색상 분류기를 별도로 지정한다."""
    if not isinstance(config, dict):
        raise ValueError("traffic 설정은 사전이어야 합니다.")
    for key in ("weights", "classifier_weights"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"traffic.{key}에 로컬 가중치 경로를 지정하세요.")
    for key in ("conf", "crosswalk_min_confidence", "classifier_min_confidence"):
        value = config.get(key, DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
            raise ValueError(f"traffic.{key}는 0~1이어야 합니다.")
    for key in ("imgsz", "association_stable_frames"):
        value = config.get(key, DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"traffic.{key}는 양의 정수여야 합니다.")


class TrafficSignalPipeline:
    """한 영상의 연속 프레임에서 대상을 고르고 선택된 신호만 분류한다."""

    def __init__(self, weights, classifier_weights, device="auto", **options):
        config = {**DEFAULTS, **options, "weights": str(weights),
                  "classifier_weights": str(classifier_weights)}
        validate_traffic_config(config)
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"알 수 없는 traffic 설정: {sorted(unknown)}")
        for path in (weights, classifier_weights):
            if not Path(path).is_file() or Path(path).suffix.lower() != ".pt":
                raise FileNotFoundError(f"로컬 신호등 .pt 가중치가 없습니다: {path}")
        device = str(device)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")
        self.device = device
        self.config = config

        from ultralytics import YOLO
        from torchvision import models

        self.detector = YOLO(str(weights))
        if self.detector.task != "detect" or self.detector.names != CLASS_NAMES:
            raise ValueError("신호등 YOLO는 class 0 pedestrian_signal, class 1 crosswalk가 필요합니다.")
        payload = torch.load(classifier_weights, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("model_name") != "mobilenet_v3_small":
            raise ValueError("분류기는 mobilenet_v3_small 체크포인트여야 합니다.")
        if payload.get("classifier_imgsz") != 224:
            raise ValueError("분류기 체크포인트 입력 크기는 224여야 합니다.")
        self.class_names = payload.get("class_names", [])
        if len(self.class_names) != 2 or set(self.class_names) != {"red", "green"}:
            raise ValueError("분류기 class_names에는 red, green이 각각 한 번 있어야 합니다.")
        classifier = models.mobilenet_v3_small(weights=None)
        classifier.classifier[-1] = torch.nn.Linear(
            classifier.classifier[-1].in_features, len(self.class_names)
        )
        classifier.load_state_dict(payload["model_state_dict"], strict=True)
        self.classifier = classifier.to(device).eval()
        self.reset()

    def reset(self):
        """영상이 바뀌면 이전 영상의 선택 이력을 비운다."""
        self.selector = TemporalSelector(self.config["association_stable_frames"])
        self.frame_id = 0

    def _classify(self, frame, box):
        """학습 때와 동일한 여백·비율 유지·검정 패딩·ImageNet 정규화."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * 0.10, (y2 - y1) * 0.10
        left, top = max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y))
        right, bottom = min(width, math.ceil(x2 + pad_x)), min(height, math.ceil(y2 + pad_y))
        if right - left < 6 or bottom - top < 6:
            return "unknown", None
        crop = frame[top:bottom, left:right]
        scale = min(224 / crop.shape[1], 224 / crop.shape[0])
        rw, rh = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        rgb = cv2.cvtColor(cv2.resize(crop, (rw, rh)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        canvas = torch.zeros((3, 224, 224), dtype=torch.float32)
        ox, oy = (224 - rw) // 2, (224 - rh) // 2
        canvas[:, oy:oy + rh, ox:ox + rw] = tensor
        mean = canvas.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = canvas.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.inference_mode():
            probabilities = self.classifier(((canvas - mean) / std)[None].to(self.device))
            score, index = probabilities.float().softmax(dim=1)[0].max(dim=0)
        confidence = float(score.item())
        color = self.class_names[int(index.item())]
        return (color if confidence >= self.config["classifier_min_confidence"] else "unknown",
                confidence)

    def predict(self, frame, *, frame_id=None, captured_at_ms=None):
        """모든 검출을 반환하고 현재 선택 대상만 분류한다.

        영상 처리기는 프레임 번호와 영상 시각을 전달한다. 직접 호출 시에는
        연속 번호와 200ms 간격을 사용하며, 영상이 바뀌면 reset()해야 한다.
        """
        frame_id = self.frame_id + 1 if frame_id is None else frame_id
        captured_at_ms = (frame_id - 1) * 200 if captured_at_ms is None else captured_at_ms
        context = FrameContext(frame_id, captured_at_ms)
        self.frame_id = frame_id
        detector_confidence = min(self.config["conf"], self.config["crosswalk_min_confidence"])
        result = self.detector.predict(
            source=frame, imgsz=self.config["imgsz"], conf=detector_confidence,
            device=self.device, quantize=32, verbose=False, save=False, save_txt=False,
            save_crop=False, stream=False,
        )[0]
        if tuple(result.orig_shape) != frame.shape[:2]:
            raise ValueError("신호등 YOLO 결과와 원본 프레임의 크기가 다릅니다.")
        signals, crosswalks, candidates = [], [], []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, score, cid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                cid = int(cid)
                item = {"xyxy": xyxy, "confidence": float(score),
                        "class_id": cid, "class_name": CLASS_NAMES[cid]}
                if cid == 0 and score >= self.config["conf"]:
                    signals.append(item)
                elif cid == 1 and score >= detector_confidence:
                    candidates.append(item)
                    if score >= self.config["crosswalk_min_confidence"]:
                        crosswalks.append(item)
        decision = self.selector.select(frame, signals, crosswalks, cv2, context)
        selected_index = decision.get("signal_index")
        candidate_index = decision.get("candidate_signal_index")
        visible, state = [], "unknown"
        for index, signal in enumerate(signals):
            color, score = "unknown", None
            selection = "unselected"
            if index == selected_index:
                color, score = self._classify(frame, signal["xyxy"])
                state, selection = color, "selected"
            elif index == candidate_index:
                selection = "candidate"
            visible.append({**signal, "signal_state": color, "color_confidence": score,
                            "selection_status": selection,
                            "track_id": decision.get("track_id") if selection == "selected" else None})
        height, width = frame.shape[:2]
        crossing_boxes, diagnostics = crosswalk_diagnostics(
            candidates, crosswalks, decision, width, height,
            self.config["crosswalk_min_confidence"],
        )
        diagnostics["detector_confidence"] = detector_confidence
        decision["color"] = state
        return {"detections": visible, "crosswalks": crossing_boxes, "association": decision,
                "signal_state": state, "detected_signal_count": len(signals),
                "selected_detection_index": selected_index,
                "candidate_detection_index": candidate_index,
                "detected_crosswalk_count": len(crosswalks),
                "crosswalk_candidate_count": len(candidates),
                "crosswalk_diagnostics": diagnostics}
