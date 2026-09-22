"""
file_path: scripts/run_video_inference.py

명령어 옵션을 받아 도보·장애물·신호등 영상 추론을 시작한다.
실제 처리 로직은 src.pipeline에서 실행한다.

[실행]
python -m scripts.run_video_inference --sample-dir data/samples/sample1

[결과물]
outputs/videos/result_*.mp4로 저장됨
"""

import argparse
from pathlib import Path


# 영상 추론 시작
def main():
    """실행 옵션을 처리하고 영상 추론 파이프라인을 호출한다."""
    parser = argparse.ArgumentParser(description="도보·장애물·보행자 신호등 통합 영상 추론")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument(
        "--mask2former-weights", "--model-dir", dest="mask2former_weights", type=Path,
        help="Mask2Former 가중치·모델·전처리 설정 폴더 (--model-dir도 사용 가능)",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video-path", type=Path, help="입력 영상 한 개")
    inputs.add_argument("--sample-dir", type=Path, help="모든 MP4를 처리할 샘플 폴더")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-path", type=Path, help="영상 한 개의 결과 MP4 경로")
    outputs.add_argument("--output-dir", type=Path, help="결과를 저장할 기존 폴더")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--mode", choices=("both", "sidewalk", "obstacle", "traffic", "all"),
                        help="both: 도보+장애물, all: 도보+장애물+신호등, 나머지: 단독 추론")
    parser.add_argument("--traffic-weights", type=Path, help="2클래스 신호등+횡단보도 YOLO 가중치")
    parser.add_argument("--traffic-classifier-weights", type=Path, help="MobileNetV3-Small 색상 분류 가중치")
    parser.add_argument("--yolo-weights", type=Path, help="YOLO .pt 가중치 경로")
    parser.add_argument("--conf", type=float, help="YOLO 신뢰도 기준")
    parser.add_argument("--imgsz", type=int, help="YOLO 입력 크기")
    parser.add_argument("--risk", action=argparse.BooleanOptionalAction, default=None,
                        help="experimental obstacle risk overlay and JSONL (--no-risk disables)")
    args = parser.parse_args()

    # 도움말 확인에는 추론 패키지 로딩 생략
    from src.pipeline import run_video_inference

    run_video_inference(
        config_path=args.config,
        mask2former_weights=args.mask2former_weights,
        video_path=args.video_path,
        sample_dir=args.sample_dir,
        output_path=args.output_path,
        output_dir=args.output_dir,
        device=args.device,
        mode=args.mode,
        yolo_weights=args.yolo_weights,
        conf=args.conf,
        imgsz=args.imgsz,
        traffic_weights=args.traffic_weights,
        traffic_classifier_weights=args.traffic_classifier_weights,
        risk=args.risk,
    )


if __name__ == "__main__":
    main()
