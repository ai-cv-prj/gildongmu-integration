"""
file_path: src/sidewalk.py

파인튜닝한 Mask2Former를 불러와 프레임별 클래스 지도를 반환한다.
영상 읽기와 색상 표시는 다른 모듈에서 처리한다.
"""

from pathlib import Path

import cv2
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# 저장된 클래스 번호 조회
def get_segmentation_label_ids(model):
    """모델 설정에서 보행불가·보행가능·횡단보도 클래스 번호를 조회한다."""
    label_ids = {
        name: int(label_id) for label_id, name in model.config.id2label.items()
    }
    required = {"non_walkable", "walkable", "crosswalk"}
    if set(label_ids) != required or len(model.config.id2label) != 3:
        raise ValueError("non_walkable/walkable/crosswalk의 3클래스 가중치가 필요합니다.")
    if set(label_ids.values()) != {0, 1, 2}:
        raise ValueError("클래스 번호는 0, 1, 2여야 합니다.")
    return label_ids


class SidewalkSegmenter:
    """한 번 로딩한 도보 모델로 여러 프레임을 추론한다."""

    # 모델 및 전처리기 로딩
    def __init__(self, weights, device="auto"):
        """로컬 모델 폴더의 가중치와 전처리 설정을 불러온다."""
        weights = Path(weights).expanduser().resolve()
        for filename in ("config.json", "preprocessor_config.json"):
            if not (weights / filename).is_file():
                raise FileNotFoundError(
                    f"모델 파일이 없습니다: {weights / filename}\n"
                    "저장된 model 폴더 전체를 준비하거나 --mask2former-weights로 지정하세요."
                )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. --device cpu로 실행하세요.")

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            weights, local_files_only=True
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            weights, local_files_only=True
        )
        self.label_ids = get_segmentation_label_ids(self.model)
        self.model = self.model.to(self.device).eval()

    # 원본 좌표계의 클래스 지도 추론
    def predict(self, frame):
        """BGR 프레임을 받아 원본 높이·너비의 정수 클래스 지도를 반환한다."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb_frame, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            class_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[frame.shape[:2]]
            )[0]
        return class_map.cpu().numpy()
