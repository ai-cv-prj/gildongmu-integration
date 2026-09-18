"""
file_path: src/pipeline.py

같은 원본 프레임의 도보·장애물 추론 결과를 MP4로 저장한다.
모델은 한 번만 로딩하며, 단독 실행과 통합 실행을 지원한다.
"""

import math
import os
import tempfile
import warnings
from pathlib import Path

import cv2
import yaml

from src.sidewalk import SidewalkSegmenter
from src.obstacle import ObstacleDetector, validate_yolo_config
from src.visualization import draw_detections, overlay_segmentation


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "inference.yaml"


# 프로젝트 기준 경로 해석
def resolve_path(value):
    """상대 경로는 통합 레포 루트, 절대 경로는 지정 위치로 해석한다."""
    path = Path(value).expanduser()
    return (PROJECT_DIR / path).resolve()


# YAML 설정 읽기
def load_config(config_path):
    """모델별 가중치 경로와 공통 추론 설정을 확인한다."""
    with resolve_path(config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required = {"mask2former", "sample_dir", "output_dir", "device", "overlay_alpha"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"설정에 필요한 항목: {', '.join(sorted(required))}")
    mask2former_config = config["mask2former"]
    if not isinstance(mask2former_config, dict):
        raise ValueError("mask2former 설정에는 weights 항목이 필요합니다.")
    mask2former_weights = mask2former_config.get("weights")
    if not isinstance(mask2former_weights, str) or not mask2former_weights.strip():
        raise ValueError("mask2former.weights에는 비어 있지 않은 폴더 경로를 지정하세요.")
    for key in ("sample_dir", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key}에는 비어 있지 않은 경로를 지정하세요.")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
    alpha = config["overlay_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("overlay_alpha는 0부터 1 사이의 숫자여야 합니다.")
    return config


# 샘플 MP4 조회
def find_sample_videos(sample_dir):
    """지정 폴더 바로 아래의 모든 MP4를 파일명 순으로 조회한다."""
    sample_dir = resolve_path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"샘플 폴더가 없습니다: {sample_dir}")
    videos = sorted(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise FileNotFoundError(f"샘플 MP4 영상이 없습니다: {sample_dir}")
    return videos


# 영상 한 개 처리
def process_video(video_path, output_path, segmenter=None, alpha=0.55, detector=None):
    """임시 MP4로 처리한 뒤 프레임 수 확인에 성공하면 최종 파일을 저장한다."""
    if segmenter is None and detector is None:
        raise ValueError("도보 또는 장애물 모델이 하나 이상 필요합니다.")
    video_path, output_path = Path(video_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"결과 영상이 이미 있습니다: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"출력 폴더가 없습니다: {output_path.parent}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("결과 영상 확장자는 .mp4여야 합니다.")

    capture = cv2.VideoCapture(str(video_path))
    writer = None
    temporary_path = None
    processed_frames = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = (
            int(reported_frames)
            if math.isfinite(reported_frames) and reported_frames >= 1 else None
        )
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"영상 크기 또는 FPS가 올바르지 않습니다: {video_path}")
        if total_frames is None:
            warnings.warn(
                f"전체 프레임 수를 알 수 없어 누락 여부를 검증할 수 없습니다: {video_path}",
                RuntimeWarning,
                stacklevel=2,
            )

        # 같은 출력 폴더에 이번 작업 전용 임시 파일 생성
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".partial.mp4",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        writer = cv2.VideoWriter(
            str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"결과 영상을 생성할 수 없습니다: {output_path}")

        while True:
            success, frame = capture.read()
            if not success:
                break
            # 두 모델 모두 색칠 전의 같은 원본 프레임 사용
            class_map = segmenter.predict(frame) if segmenter is not None else None
            detections = detector.predict(frame) if detector is not None else []
            result = (
                overlay_segmentation(frame, class_map, segmenter.label_ids, alpha)
                if segmenter is not None else frame
            )
            if detector is not None:
                result = draw_detections(result, detections)
            writer.write(result)
            processed_frames += 1
            print(
                f"\r영상 처리: {processed_frames}/{total_frames or '?'}",
                end="", flush=True,
            )
        if processed_frames == 0:
            raise RuntimeError(f"읽을 수 있는 프레임이 없습니다: {video_path}")
        if total_frames is not None and processed_frames != total_frames:
            raise RuntimeError(
                f"영상 프레임 수 불일치: 예상={total_frames}, 처리={processed_frames}\n"
                f"읽기 오류 또는 영상 메타데이터 오류를 확인하세요: {video_path}\n"
                "최종 결과 파일은 저장하지 않습니다."
            )

        # 인코딩 종료 후 최종 이름 등록, 기존 파일 덮어쓰기 금지
        writer.release()
        writer = None
        os.link(temporary_path, output_path)
    finally:
        try:
            capture.release()
            if writer is not None:
                writer.release()
        finally:
            # 성공·오류·Ctrl+C 모두 이번 작업의 임시 파일만 정리
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    print(f"\n결과 영상 저장: {output_path}")
    return processed_frames


# 설정 및 명령어 옵션으로 추론 실행
def run_video_inference(
    config_path=DEFAULT_CONFIG,
    *,
    mask2former_weights=None,
    video_path=None,
    sample_dir=None,
    output_path=None,
    output_dir=None,
    device=None,
    mode=None,
    yolo_weights=None,
    conf=None,
    imgsz=None,
):
    """명령어 옵션을 설정에 우선 적용하고 모든 대상 영상을 처리한다."""
    if video_path is not None and sample_dir is not None:
        raise ValueError("--video-path와 --sample-dir은 동시에 지정할 수 없습니다.")
    if output_path is not None and output_dir is not None:
        raise ValueError("--output-path와 --output-dir은 동시에 지정할 수 없습니다.")
    config = load_config(config_path)
    mode = mode if mode is not None else config.get("mode", "both")
    if mode not in ("both", "sidewalk", "obstacle"):
        raise ValueError("mode는 both, sidewalk, obstacle 중 하나여야 합니다.")
    device = device if device is not None else config["device"]
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device는 auto, cpu, cuda 중 하나여야 합니다.")
    yolo_config = {}
    if mode in ("both", "obstacle"):
        if not isinstance(config.get("yolo", {}), dict):
            raise ValueError("yolo 설정은 weights, conf, imgsz, head 항목으로 작성하세요.")
        yolo_config = dict(config.get("yolo", {}))
        for key, value in (("weights", yolo_weights), ("conf", conf), ("imgsz", imgsz)):
            if value is not None:
                yolo_config[key] = str(value) if key == "weights" else value
        validate_yolo_config(yolo_config)
    mask2former_weights = resolve_path(
        mask2former_weights if mask2former_weights is not None else config["mask2former"]["weights"]
    )
    videos = (
        [resolve_path(video_path)] if video_path is not None
        else find_sample_videos(sample_dir if sample_dir is not None else config["sample_dir"])
    )
    if output_path is not None and len(videos) != 1:
        raise ValueError("--output-path는 영상 한 개 처리 시에만 지정할 수 있습니다.")
    destination = resolve_path(output_dir if output_dir is not None else config["output_dir"])
    outputs = [
        resolve_path(output_path) if output_path is not None
        else destination / f"result_{video.stem}.mp4"
        for video in videos
    ]

    # 전체 입출력 사전 검증
    if len(set(outputs)) != len(outputs):
        raise ValueError("결과 파일명이 겹칩니다. 입력 영상 이름을 구분해 주세요.")
    for video, output in zip(videos, outputs):
        if not video.is_file():
            raise FileNotFoundError(f"입력 영상이 없습니다: {video}")
        if output.suffix.lower() != ".mp4":
            raise ValueError(f"결과 영상 확장자는 .mp4여야 합니다: {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"출력 폴더가 없습니다: {output.parent}")
        if output.exists():
            raise FileExistsError(f"결과 영상이 이미 있습니다: {output}")

    # 사용할 모델만 로딩, 모든 영상에서 재사용
    detector = None
    if mode in ("both", "obstacle"):
        yolo_weights = resolve_path(yolo_config["weights"])
        detector = ObstacleDetector(
            yolo_weights, device=device,
            conf=yolo_config["conf"], imgsz=yolo_config["imgsz"], head=yolo_config["head"],
        )
        device = detector.device
        print(
            f"YOLO: {yolo_weights}\n"
            f"YOLO 설정: conf={detector.conf}, imgsz={detector.imgsz}, head=nms"
        )
    segmenter = (
        SidewalkSegmenter(mask2former_weights, device=device)
        if mode in ("both", "sidewalk") else None
    )
    if segmenter is not None:
        device = segmenter.device
        print(f"Mask2Former: {mask2former_weights}")
    print(f"추론 모드: {mode} | 장치: {device}")
    for index, (video, output) in enumerate(zip(videos, outputs), start=1):
        print(f"입력 영상 [{index}/{len(videos)}]: {video}")
        process_video(video, output, segmenter, config["overlay_alpha"], detector=detector)
    return outputs
