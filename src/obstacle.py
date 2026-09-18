"""
file_path: src/obstacle.py

파인튜닝한 YOLO를 불러와 원본 좌표계의 장애물 탐지 결과를 반환한다.
영상 읽기와 박스 표시는 다른 모듈에서 처리한다.
"""

from pathlib import Path

import torch


# 팀 장애물 모델의 클래스 순서
EXPECTED_CLASS_NAMES = (
    "person", "bicycle", "bus", "car", "handcart", "cat", "dog", "motorcycle",
    "kick_scooter", "stroller", "truck", "wheelchair", "bird", "barricade",
    "bench", "bollard", "chair", "fire_hydrant", "kiosk", "parking_meter",
    "pole", "potted_plant", "utility_box", "transit_stop", "table",
    "traffic_light", "traffic_sign", "tree_trunk", "movable_obstacle",
    "suitcase", "skateboard", "trash_bin",
)


# YOLO 설정 검증
def validate_yolo_config(config):
    """가중치 경로와 신뢰도·입력 크기·추론 헤드 설정을 확인한다."""
    required = {"weights", "conf", "imgsz", "head"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("yolo 설정에는 weights, conf, imgsz, head가 필요합니다.")
    if not isinstance(config["weights"], str) or not config["weights"].strip():
        raise ValueError("yolo.weights에는 가중치 경로를 지정하세요.")
    conf = config["conf"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise ValueError("yolo.conf는 0부터 1 사이의 숫자여야 합니다.")
    imgsz = config["imgsz"]
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("yolo.imgsz는 양의 정수여야 합니다.")
    if config["head"] != "nms":
        raise ValueError("현재 통합에서는 yolo.head: nms만 지원합니다.")


class ObstacleDetector:
    """한 번 로딩한 YOLO로 여러 프레임의 장애물을 탐지한다."""

    # 로컬 가중치 및 클래스 확인
    def __init__(self, weights, device="auto", conf=0.25, imgsz=640, head="nms"):
        """32클래스 장애물 가중치를 불러오고 추론 옵션을 저장한다."""
        validate_yolo_config({"weights": str(weights), "conf": conf, "imgsz": imgsz, "head": head})
        weights = Path(weights).expanduser().resolve()
        if weights.suffix.lower() != ".pt" or not weights.is_file():
            raise FileNotFoundError(f"로컬 YOLO .pt 가중치가 없습니다: {weights}")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")

        # 도보 단독 실행에는 Ultralytics 로딩 생략
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(str(weights))
        if self.model.task != "detect":
            raise ValueError("객체 탐지용 YOLO 가중치가 필요합니다.")
        self.class_names = self.model.names
        if self.class_names != dict(enumerate(EXPECTED_CLASS_NAMES)):
            raise ValueError("YOLO 가중치의 클래스 번호·이름이 팀의 32클래스 정의와 다릅니다.")

    # 원본 프레임의 장애물 탐지
    def predict(self, frame):
        """BGR 프레임에서 박스 좌표·클래스·신뢰도 목록을 반환한다."""
        result = self.model.predict(
            source=frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            nms=None, verbose=False, save=False, save_txt=False, save_crop=False,
        )[0]
        if tuple(result.orig_shape) != frame.shape[:2]:
            raise ValueError("YOLO 결과와 원본 프레임의 높이·너비가 다릅니다.")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        # Ultralytics의 xyxy는 원본 픽셀 좌표, 추가 배율 변환 불필요
        return [
            {
                "xyxy": xyxy,
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence,
            }
            for xyxy, class_id, confidence in zip(
                boxes.xyxy.cpu().tolist(), boxes.cls.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
            )
        ]
