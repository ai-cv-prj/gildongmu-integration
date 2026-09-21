<!-- file_path: README.md
Mask2Former·YOLO 통합 추론의 준비, 실행 및 검증 방법 안내.
-->

# gildongmu-integration

보행 영역을 찾는 **Mask2Former**, 장애물 **YOLO**, 보행자 신호등 **YOLO + MobileNetV3-Small**의
결과를 한 영상에 표시하는 프로젝트입니다. 같은 원본 프레임을 각각 분석한 뒤 결과를 겹쳐 그립니다.

| 탐지 결과 | 영상 표시 |
| --- | --- |
| 보행가능 영역 | 반투명 초록색 |
| 횡단보도 | 반투명 핑크색 |
| 보행불가 영역 | 색칠하지 않음 |
| YOLO 객체 | 클래스별 고정 색상의 박스 + 같은 색의 영문 이름·신뢰도 |
| 선택된 보행자 신호등 | 빨강·초록·회색(unknown) 박스 + 색상 분류 신뢰도 |
| 신호등 YOLO의 횡단보도 | 핑크 박스, 연결에 사용한 횡단보도는 굵은 청록 박스 |

현재는 **저장된 영상 파일을 분석하는 기능**입니다. 파인튜닝, 실시간 카메라 입력, BEV,
거리 추정, 음성·진동 알림은 포함하지 않습니다.
2026-09-21부터 실험용 장애물 위험 판단(ROI·추적·측방 진입)과 JSONL 기록을 지원합니다.
초기값과 수식, 실행 방법은 [위험 판단 MVP](docs/risk_mvp.md)를 참고하세요.
현재 설정에서 활성화되어 있으며 `--no-risk`로 기존 탐지 표시만 사용할 수 있습니다.
신호등 색상과 횡단보도 연결은 추정 결과이며, 사용자의 실제 횡단 의도나 횡단 안전성을 보장하지 않습니다.
YOLO는 초록·핑크 영역에 한정하지 않고 전체 화면에서 객체를 탐지합니다.
색칠된 영역이 실제로 안전하다는 뜻은 아니며, 탐지 결과만으로 횡단·이동 여부를 판단하지 않습니다.

## 1. 실행 준비

### 가상환경과 패키지

아래 명령어는 Linux/WSL 기준입니다. 먼저 터미널에서 `gildongmu-integration` 폴더로 이동하세요.
Python 3.12가 설치되어 있어야 합니다.

```bash
# .venv가 없을 때만 생성
python3.12 -m venv .venv

# 터미널을 새로 열었다면 활성화
source .venv/bin/activate

# 버전 확인 및 패키지 설치
python --version
python -m pip install -r requirements.txt
```

이미 환경을 준비했다면 가상환경 활성화만 하면 됩니다.
OpenCV는 `opencv-python`만 사용합니다. 기존 `opencv-python-headless`가 있다면 먼저 제거하고 설치하세요.
두 패키지를 함께 설치하면 같은 `cv2` 모듈을 공유해 문제가 생길 수 있습니다.

GPU를 사용할 수 있는지 확인하려면:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`True`면 PyTorch에서 CUDA를 사용할 수 있습니다. `False`면 기본 설정에서 CPU로 실행됩니다.
GPU를 쓰려면 NVIDIA 드라이버와 CUDA 지원 PyTorch 설치 구성을 확인해야 합니다.

### 가중치와 영상 배치

아래는 파일을 배치하는 예시입니다. 가중치·영상·결과 폴더는 Git에서 제외되어 있으므로
레포를 clone한 것만으로 준비되지 않습니다. 필요한 파일과 입출력 폴더는 구글 드라이브 '모델' 폴더에서 다운로드 받으세요.

```text
gildongmu-integration/
├── weights/
│   ├── mask2former/
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   └── model.safetensors
│   ├── yolo/
│   │   └── finetune_v2_exp02_stage2_best.pt
│   └── traffic/
│       ├── best_YOLO.pt
│       └── best_MobileNet.pt
├── data/
│   └── samples/
│       └── sample1/
│           └── OBS_260914_G24P_001.mp4
└── outputs/
    └── videos/
```

- **Mask2Former:** 학습 결과의 `model` 폴더 내용 전체를 넣습니다. 가중치 파일만 넣으면 안 됩니다.
  분할 저장된 모델이라면 모든 가중치 조각과 index 파일도 필요합니다.
  `non_walkable`, `walkable`, `crosswalk`의 3클래스 모델을 사용합니다.
- **YOLO:** 팀에서 파인튜닝한 `.pt` 파일을 넣습니다. 클래스 번호·이름이 팀의 32클래스 정의와 일치해야 합니다.
  신뢰할 수 있는 가중치만 사용하세요.
- **신호등 YOLO:** class 0 `pedestrian_signal`, class 1 `crosswalk`인 2클래스 검출기입니다.
  장애물 YOLO와는 별도 모델입니다.
- **색상 분류기:** 입력 224의 `mobilenet_v3_small` 체크포인트입니다. 체크포인트에 저장된
  `class_names` 순서를 사용하므로 `[green, red]`를 임의로 `[red, green]`으로 바꾸지 않습니다.
- **영상:** 폴더 일괄 처리는 MP4만 지원합니다. 하위 폴더까지 자동 탐색하지 않습니다.
- **출력:** `outputs/runs/manual/`가 미리 있어야 합니다. 추론 코드는 입출력 폴더를 자동 생성하지 않습니다.

## 2. 영상 실행하기

모든 실행 명령어는 **통합 레포 루트에서, 가상환경을 활성화한 상태**로 실행합니다.
아래 예시의 영상명은 본인이 준비한 파일명에 맞추세요.

### sample1의 모든 영상 처리

```bash
python -m scripts.run_video_inference --sample-dir data/samples/sample1
```

현재 설정은 `mode: both`이므로 두 모델을 모두 사용합니다.
지정 폴더 바로 아래의 MP4를 파일명 순서대로 처리합니다.
`sample2`를 쓰려면 `--sample-dir data/samples/sample2`로 바꾸면 됩니다.

주의: 옵션을 생략하면 YAML의 `sample_dir: data/samples`를 사용합니다.
영상이 `sample1` 안에 있다면 위처럼 하위 폴더를 지정하거나 YAML의 `sample_dir`를 바꿔야 합니다.

### 기존 결과가 있거나 영상 하나만 처리할 때

기존 결과는 덮어쓰거나 자동으로 건너뛰지 않습니다. 같은 이름이 있으면 실행을 중단합니다.
재실행할 때는 다음처럼 **다른 결과 파일명**을 지정하세요.

```bash
python -m scripts.run_video_inference \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_OBS_260914_G24P_001_combined.mp4
```

이 이름도 이미 존재하면 다른 이름을 지정해야 합니다.

### 도보·장애물을 따로 확인할 때

비교 결과가 서로 겹치지 않도록 출력 이름을 구분합니다.

```bash
# 도보 마스크만 표시
python -m scripts.run_video_inference --mode sidewalk \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_OBS_260914_G24P_001_sidewalk.mp4

# 장애물 박스만 표시
python -m scripts.run_video_inference --mode obstacle \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_OBS_260914_G24P_001_obstacle.mp4
```

단독 모드에서는 해당 모델만 로딩합니다. YAML은 아래의 전체 설정 구조를 유지하세요.

### 신호등 단독 또는 전체 통합 실행

```bash
# 신호등 검출 + 횡단보도 연결 + 색상 분류
python -m scripts.run_video_inference --mode traffic \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_signal.mp4

# 도보 + 장애물 + 신호등을 한 영상에 표시
python -m scripts.run_video_inference --mode all \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_all.mp4
```

신호등 가중치를 다른 곳에 두었다면 `--traffic-weights /경로/YOLO.pt`와
`--traffic-classifier-weights /경로/MobileNet.pt`로 지정합니다.
기존 `both`는 도보+장애물만 실행하고, 신호등까지 사용하려면 `all`을 선택합니다.
`traffic` 모드에는 도보·장애물 가중치가 필요하지 않습니다.

신호등 선택 순서:

1. 검출된 보행자 신호등이 하나면 바로 색상을 분류합니다. 횡단보도와의 관계는
   확인되지 않았으므로 `crosswalk_relation_unverified`를 함께 표시합니다.
2. 두 개 이상이면 화면 아래쪽·중앙에 있는 횡단보도를 선택하고, 그 영역의 영상 선분에서
   소실점을 추정합니다. 박스 중앙을 소실점으로 간주하지 않습니다.
3. 소실점과의 정규화된 **수평 거리**를 우선 비교하고, 박스 크기는 보조 점수로 사용합니다.
   방향이 비슷하고 면적 차이가 충분할 때 큰 후보를 선택합니다.
4. 같은 신호등·횡단보도 조합이 기본 3프레임 연속 유지되면 선택된 신호등만 분류합니다.
   대기 중 후보는 회색 `unknown`으로 표시합니다.
5. 횡단보도나 소실점이 없거나 후보를 구분하기 어려우면 색상을 확정하지 않습니다.
   화면 상단에 후보 수와 `unknown` 사유를 표시합니다.

횡단보도 선택은 **신호등 전용 YOLO의 crosswalk 박스**를 사용합니다.
Mask2Former의 핑크 마스크는 화면에 함께 표시하지만 현재 신호등 연결의 입력은 아닙니다.
신호등 모듈이 켜져 있을 때 장애물 모델의 일반 `traffic_light` 박스는 표시에서 제외하여
선택된 보행자 신호등 표시와 혼동되지 않게 합니다. 다른 장애물 박스는 그대로 표시합니다.
영상마다 선택 이력은 초기화하며 모델 가중치는 한 번만 로딩합니다.

2026-09-19 로컬 연동에는 기존 테스트 앱의
`backend/models/traffic/best_YOLO_v1.pt`를 `weights/traffic/best_YOLO.pt`로,
`backend/models/traffic/classifier/best_MobileNet.pt`를 `weights/traffic/best_MobileNet.pt`로
복사했습니다. 두 파일 모두 원본과 SHA-256이 일치합니다. 가중치는 Git 제외 대상이므로
다른 PC에서는 별도로 배치해야 합니다. 추가 파인튜닝 가중치로 자동 교체하지 않습니다.

## 3. 결과 확인하기

기본 저장 위치는 `outputs/runs/manual/result_원본파일명.mp4`입니다.
예를 들어 `test.mp4`의 결과는 `result_test.mp4`로 저장됩니다.
`--output-path`를 지정하면 지정한 이름을 그대로 사용합니다.

- 원본 영상 크기와 저장 FPS를 유지하며 MP4로 다시 인코딩합니다. 원본 오디오는 포함하지 않습니다.
- 진행 중에는 콘솔에 `영상 처리: 처리한 프레임 수/전체 프레임 수`가 표시됩니다.
- 완료되면 `결과 영상 저장: ...` 메시지가 나옵니다.
- 위험 기능을 켜면 MP4와 같은 이름의 `.risk.jsonl`에 객체 좌표·선택적 ID·위험 등급·판단 사유·이벤트를 저장합니다.
- `--no-risk`에서는 영상만 저장합니다. 성능 평가표나 새 모델 가중치는 생성하지 않습니다.
- 저장 FPS를 유지한다는 뜻이지, 그 속도로 실시간 추론한다는 뜻은 아닙니다. 별도 속도 측정이 필요합니다.

## 4. 설정 바꾸기

기본 설정 파일은 [configs/inference.yaml](configs/inference.yaml)입니다.

```yaml
sample_dir: data/samples
output_dir: outputs/runs/manual
device: auto
overlay_alpha: 0.55
mode: both

mask2former:
  weights: weights/mask2former

yolo:
  weights: weights/yolo/finetune_v2_exp02_stage2_best.pt
  conf: 0.25
  imgsz: 640
  head: nms

traffic:
  weights: weights/traffic/best_YOLO.pt
  classifier_weights: weights/traffic/best_MobileNet.pt
  conf: 0.25
  imgsz: 960
  crosswalk_min_confidence: 0.50
  classifier_min_confidence: 0.60
  association_stable_frames: 3
```

| 설정 | 의미 |
| --- | --- |
| `sample_dir` | 처리할 MP4가 직접 들어 있는 폴더 |
| `output_dir` | 결과를 저장할 기존 폴더 |
| `device` | `auto`: CUDA 가능 시 GPU, 아니면 CPU / `cuda`: GPU 지정 / `cpu`: CPU 지정 |
| `mode` | `both`: 도보+장애물 / `all`: 전체 / `sidewalk`, `obstacle`, `traffic`: 각 단독 |
| `mask2former.weights` | 가중치와 모델·전처리 설정이 있는 **폴더** |
| `yolo.weights` | YOLO 가중치 **파일** |
| `overlay_alpha` | 마스크 색상 비율. `0.55`는 원본 45% + 색상 55%. 클수록 진하게 표시 |
| `yolo.conf` | 객체 신뢰도 기준. 높이면 더 엄격하게 걸러지지만 놓치는 객체가 늘 수 있음 |
| `yolo.imgsz` | YOLO 내부 전처리 크기 기준. 결과 영상의 크기를 바꾸는 값은 아님 |
| `yolo.head` | 현재는 `nms`만 지원. 겹치는 탐지 박스를 정리하는 후처리 사용 |
| `traffic.weights` / `traffic.classifier_weights` | 신호등 YOLO / MobileNet 가중치 파일 |
| `traffic.conf` / `traffic.imgsz` | 신호등 검출 기준 / YOLO 입력 크기 |
| `traffic.crosswalk_min_confidence` | 연결에 사용할 횡단보도 검출 기준 |
| `traffic.classifier_min_confidence` | 이 값보다 낮으면 색상을 `unknown` 처리 |
| `traffic.association_stable_frames` | 여러 신호등 중 같은 대상이 유지되어야 하는 프레임 수. 1이면 첫 후보부터 사용 |

`overlay_alpha`는 0~1 범위이며 탐지 성능이나 YOLO 박스에는 영향을 주지 않습니다.
Mask2Former의 전처리는 저장된 `preprocessor_config.json`을 사용합니다.
YOLO 신뢰도 0.25는 시작 설정이며, 실제 영상의 오탐·미탐을 확인해 조정해야 합니다.

### 명령어 옵션으로 바꾸기

**명령어 옵션이 YAML보다 우선**합니다. 상대 경로는 모두 통합 레포 루트 기준이며 절대 경로도 가능합니다.
가중치를 생략하면 YAML에 지정된 것을 사용하고, 최신 가중치를 자동 선택하지는 않습니다.

```bash
python -m scripts.run_video_inference \
  --sample-dir data/samples/sample1 \
  --mask2former-weights weights/mask2former \
  --yolo-weights weights/yolo/finetune_v2_exp02_stage2_best.pt \
  --conf 0.25 --imgsz 640 --device cuda
```

| 옵션 | 용도 |
| --- | --- |
| `--config` | 다른 YAML 설정 파일 선택 |
| `--sample-dir` / `--video-path` | 폴더 전체 / 영상 한 개 선택. 둘 중 하나만 지정 |
| `--output-dir` / `--output-path` | 출력 폴더 / 결과 파일명 지정. 둘 중 하나만 지정 |
| `--mask2former-weights` / `--yolo-weights` | 모델별 가중치 경로 지정 |
| `--mode` | `both`, `sidewalk`, `obstacle`, `traffic`, `all` 선택 |
| `--traffic-weights` / `--traffic-classifier-weights` | 신호등 YOLO / MobileNet 경로 지정 |
| `--device` | `auto`, `cpu`, `cuda` 선택 |
| `--conf` / `--imgsz` | YOLO 신뢰도 / 전처리 크기 지정 |

`--output-path`는 처리할 영상이 한 개일 때만 사용할 수 있고 확장자는 `.mp4`여야 합니다.
`--conf`, `--imgsz`는 장애물 YOLO 옵션입니다. 신호등 임계값·입력 크기는 YAML의 `traffic` 항목에서 설정합니다.
기존 `--model-dir`는 `--mask2former-weights`의 별칭으로 사용할 수 있습니다.
이전 YAML의 최상위 `model_dir`는 `mask2former` 아래의 `weights`로 옮겨야 합니다.
전체 옵션은 `python -m scripts.run_video_inference --help`로 확인합니다.

## 5. 자주 발생하는 문제

| 메시지·상황 | 확인할 것 |
| --- | --- |
| 결과 영상이 이미 있음 | 다른 `--output-path`나 기존 출력 폴더를 지정. 자동 덮어쓰기·건너뛰기 없음 |
| 샘플 MP4가 없음 | `data/samples`가 아니라 실제 영상이 들어 있는 `data/samples/sample1`을 지정했는지 확인 |
| 출력 폴더가 없음 | `outputs/runs/manual` 등 지정한 폴더를 먼저 준비. 자동 생성하지 않음 |
| 모델 파일이 없음 | Mask2Former 설정 파일까지 모두 준비했는지, YOLO 파일명이 설정과 같은지 확인 |
| 3클래스·32클래스 오류 | 현재 코드에 맞는 팀 가중치인지 확인. 임의의 기본 모델은 사용할 수 없음 |
| CUDA를 사용할 수 없음 | GPU 환경 확인. CPU로 실행하려면 명령어에 `--device cpu` 추가 |
| `No module named ...` | 통합 레포 루트인지, 가상환경이 활성화됐는지, requirements 설치가 됐는지 확인 |
| 영상 프레임 수 불일치 | 입력 파일의 읽기 오류·메타데이터 확인. 이 경우 최종 결과는 저장하지 않음 |

### 실행 도중 중단하면?

- 처리 중에는 출력 폴더의 고유한 `.partial.mp4` 임시 파일에 저장하고, 완료 후 최종 파일명으로 등록합니다.
- 추론 오류나 `Ctrl+C` 발생 시 이번 작업의 임시 파일을 정리합니다. 영상 중간부터 이어서 처리하지는 않습니다.
- 여러 영상 중 앞서 완료한 결과는 유지됩니다. 실패한 영상은 `--video-path`로 개별 재실행하세요.
- 전체 프레임 수를 알 수 없으면 누락 여부를 검증할 수 없다는 경고 후 읽을 수 있는 프레임을 처리합니다.
- 강제 종료나 전원 종료 시에는 임시 파일이 남을 수 있습니다. 실행 중인 작업의 임시 파일은 건드리지 마세요.
- 최종 등록에는 같은 파일시스템의 하드 링크를 사용하며, 실행 도중 만들어진 다른 결과도 덮어쓰지 않습니다.

## 6. 코드 구성과 테스트

| 파일 | 역할 |
| --- | --- |
| [scripts/run_video_inference.py](scripts/run_video_inference.py) | 명령어 옵션을 받아 실행 시작 |
| [configs/inference.yaml](configs/inference.yaml) | 가중치·입출력 경로와 추론 설정 |
| [src/pipeline.py](src/pipeline.py) | 영상 읽기 → 모드별 모델 추론 → 결과 합성 → MP4 저장 |
| [src/sidewalk.py](src/sidewalk.py) | Mask2Former 로딩과 보행 영역 추론 |
| [src/obstacle.py](src/obstacle.py) | YOLO 로딩과 장애물 추론 |
| [src/traffic.py](src/traffic.py) | 신호등 YOLO, MobileNet 로딩·선택 대상 색상 분류 |
| [src/traffic_association.py](src/traffic_association.py) | 횡단보도·소실점·크기 비교 및 연속 프레임 선택 안정화 |
| [src/visualization.py](src/visualization.py) | 반투명 마스크와 클래스별 색상의 객체 박스·글자 표시 |
| [tests/test_integration.py](tests/test_integration.py) | 설정·옵션·좌표·클래스별 색상·영상 저장 동작 검증. 일반 추론 실행에는 사용하지 않음 |
| [tests/test_traffic.py](tests/test_traffic.py) | 단일·복수 신호등 선택, 색상 전처리, unknown, 모드 호환성 검증 |

각 모델은 한 번만 로딩하고 모든 영상에서 재사용합니다.
프레임마다 활성화한 모델을 순서대로 실행하며, 모두 색칠 전의 같은 원본을 입력받습니다.
원본 도보·장애물 레포의 코드나 학습 데이터는 실행에 필요하지 않습니다.

개발 시 결과를 연결하는 기준:

- 공통 입력: OpenCV BGR 이미지 `(높이, 너비, 3)`.
- `SidewalkSegmenter.predict(frame)`: 원본 크기의 정수 클래스 지도 반환. 번호는 `label_ids`로 조회.
- `ObstacleDetector.predict(frame)`: `xyxy`, `class_id`, `class_name`, `confidence`를 담은 목록 반환. 탐지 객체가 없으면 빈 목록.
- `TrafficSignalPipeline.predict(frame)`: 최대 1개의 `detections`, `crosswalks`, `association`,
  `signal_state`, `detected_signal_count`를 반환합니다. 색상은 `red`, `green`, `unknown`입니다.
  `reset()`은 다음 영상의 첫 프레임 전에 호출합니다.
- `xyxy`는 원본 픽셀 기준 `[왼쪽, 위, 오른쪽, 아래]`. 다시 크기 비율을 곱하지 않습니다.

자동 테스트 실행:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

실제 가중치나 GPU 없이 검사하며, 테스트용 임시 영상은 정리합니다.
이 테스트는 코드 연결과 입출력 동작을 확인하는 것으로, **실제 탐지 정확도나 실시간 속도를 보장하지 않습니다.**
실제 가중치를 사용하는 통합 추론은 별도로 실행하고, 같은 입력에 대한 단독·통합 결과를 비교해야 합니다.

2026-09-19 연동 검증: 기존 32개 + 신호등 17개, 총 49개 자동 테스트 통과.
기존 학습 환경의 Python 3.14 / PyTorch 2.11.0+cu128과 별도 임시 설치한
Transformers 5.17.0으로 CPU 검증했습니다. 위 Python 3.12 기준 고정 의존성 전체를
새 가상환경에 설치한 검증은 수행하지 않았습니다.
실제 폰 테스트의 81~83번 이미지에서 신호등 추론과 3프레임 MP4 저장을 확인했습니다.
81번은 신호등 미검출, 82·83번은 1개 검출 및 green 분류였습니다.
따라서 이 샘플로 복수 신호등 연결의 실제 정확도를 확인한 것은 아닙니다.
도보·장애물 가중치가 현재 통합 폴더에 없어 세 파트 전체의 실제 추론은 아직 미검증입니다.

## 7. 진행 상황과 남은 문제 (2026-09-19)

### 완료한 작업

| 항목 | 현재 상태 |
| --- | --- |
| 신호등 파이프라인 연동 | 신호등 전용 YOLO → 횡단보도 연결 → MobileNetV3-Small 색상 분류 연결 |
| 대상 선택 | 1개 검출 시 즉시 분류, 2개 이상이면 횡단보도 소실점의 수평 거리와 신호등 크기로 비교 |
| 선택 안정화 | 기본 3프레임 연속 같은 신호등·횡단보도 조합일 때 확정, 판단이 모호하면 unknown |
| 실행 모드 | traffic 단독 및 all 통합 추가, 기존 both·sidewalk·obstacle 유지 |
| 영상 표시 | 선택된 신호등의 색상 박스, 횡단보도 박스, 소실점 및 판단 사유 표시 |
| 구조 확인 | scripts 실행 진입점, src 추론·표시, configs 설정, tests 검증 구조 유지 |
| 입출력 규칙 | 원본 BGR·원본 픽셀 좌표, 모델 1회 로딩, 영상별 선택 이력 초기화, 결과 덮어쓰기 금지 유지 |
| 검증 | 자동 테스트 49개 통과, 실제 신호등 가중치로 3프레임 MP4 저장 및 재생 프레임 수 확인 |

### 미해결 문제: 일반 신호등 박스가 숨겨지는 동작

현재 all 모드에서는 신호등 파이프라인이 실행되면 장애물 모델의 traffic_light 결과를
표시 목록에서 모두 제외합니다. 같은 신호등에 두 모델의 박스가 겹치는 것을 막으려는 처리지만,
신호등 전용 모델이 검출·선택에 실패한 경우에도 적용됩니다.

| 장애물 모델 결과 | 신호등 전용 모델 결과 | 현재 all 화면 |
| --- | --- | --- |
| traffic_light 검출 | 대상 선택 및 색상 분류 성공 | 선택된 보행자 신호등의 색상 박스만 표시 |
| traffic_light 검출 | 신호등 미검출 | 신호등 박스가 표시되지 않음 |
| traffic_light 검출 | 복수 후보가 모호해 대상 없음 | 일반 신호등 박스도 숨겨짐, unknown 사유 표시 |

이는 검출 모델의 출력을 바꾸는 것이 아니라 src/pipeline.py에서 표시 전에 걸러내는 동작입니다.
차량용 신호등도 traffic_light이면 숨겨지며, 사람·자동차 등 다른 클래스에는 영향을 주지 않습니다.
신호등 모듈이 꺼진 기존 both·obstacle 모드는 영향을 받지 않습니다.
위 표는 코드의 조건별 동작을 설명한 것으로, 같은 실제 프레임에서 장애물 모델만 검출에
성공했다고 확인한 결과는 아닙니다. 이 표시 정책은 현재 유지한 상태이며 수정하지 않았습니다.

### 후속 확인 사항

1. 일반 traffic_light 박스를 유지할지, 표시 여부를 설정으로 분리할지 결정합니다.
   함께 표시한다면 일반 객체 박스와 선택된 보행자 신호등의 색상 결과를 구분해야 합니다.
2. 도보·장애물 가중치를 준비하고 같은 영상으로 각 단독 모드와 all 모드를 비교합니다.
3. 보행자 신호등이 여러 개 검출되는 실제 영상에서 횡단보도 연결과 선택 정확도를 확인합니다.
   지금까지 확인한 81~83번 프레임은 0개 또는 1개 검출이라 이 검증을 대신할 수 없습니다.
4. 프로젝트 기준인 Python 3.12와 requirements.txt의 고정 의존성으로 실행을 검증합니다.
   현재 자동 테스트와 신호등 추론은 앞 절에 기록한 별도 환경에서 수행했습니다.

요약: 가상환경 활성화 → 가중치·영상 준비 → 샘플 폴더 선택 → 실행 → outputs/runs/manual 결과 확인.

## 8. 장애물 위험 판단 MVP (2026-09-21)

[초기 ROI 좌표·임계값·수식·검증 결과](docs/risk_mvp.md)를 참고하세요.
전체 화면 탐지를 유지하며 신호등 탐지·선택·색상 분류·표시 부분은 변경하지 않았습니다.

2026-09-21 전체 화면·ROI 자르기 실측과 8개 영상 검토는
[ROI 비교 및 위험 검토 기록](docs/roi_and_risk_review_20260921.md)을 참고하세요.
당시 별도 검토 설정에서 TTC 위험 반영과 큰 ROI·등급 표시를 켰습니다. 현재 3차 설정은 TTC 반영을 기본으로 켭니다.
거리(m)·접근 속도(m/s)는 아직 추정하지 않습니다. 자동 테스트 72개가 통과했습니다.

### 위험 결과 피드백 반영

하단 ROI 확장, 조건부 보도 방향, 정적 장애물 근접 구간, 경고 해제 확인을 반영했습니다.
[실제 구현·초기값·후속 검증 사항](docs/risk_revision_implementation_20260921.md)을 참고하세요.
자동 테스트 87개가 통과했습니다. 화면 이탈/관측 소실은 실제 신체 주변의 안전 확인을 뜻하지 않습니다.
새 결과와 좌우 비교 영상은 실행한 로컬 환경의 `outputs/experiments/` 아래에 생성합니다.

### 공통 ROI와 보도 기반 경고 (3차)

개인별 신체 치수와 지면 높이 변화는 제외하고, 공통 진행 ROI·측면 근접·보도 기반 경고를 추가했습니다.
[구현 기준과 초기값](docs/risk_shared_profile_20260921.md)을 참고하세요. 자동 테스트 116개가 통과했습니다.
강하게 겹친 같은 클래스의 경고는 한 알림 단위로 표시하되, 모든 탐지·개별 위험 등급·감사 이벤트는 유지합니다.
기존 MP4 8개와 sample2의 MP4 2개를 처리하며 MOV는 제외합니다.
결과 영상과 회차별 안내는 `outputs/` 아래에 생성하며, 용량 때문에 저장소에 포함하지 않습니다.
저장소에서 확인할 수 있는 근거는 위 docs 문서들이고, 영상은 직접 실행해 재현합니다.
