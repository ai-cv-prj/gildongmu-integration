<!-- file_path: README.md
Mask2Former·YOLO 통합 추론의 준비, 실행 및 검증 방법 안내.
-->

# gildongmu-integration

보행 영역을 찾는 **Mask2Former**와 장애물을 찾는 **YOLO**의 결과를 한 영상에 표시하는 프로젝트입니다.
두 모델의 가중치를 합치는 것이 아니라, 같은 원본 프레임을 각각 분석한 뒤 결과를 겹쳐 그립니다.

| 탐지 결과 | 영상 표시 |
| --- | --- |
| 보행가능 영역 | 반투명 초록색 |
| 횡단보도 | 반투명 핑크색 |
| 보행불가 영역 | 색칠하지 않음 |
| YOLO 객체 | 클래스별 고정 색상의 박스 + 같은 색의 영문 이름·신뢰도 |

현재는 **저장된 영상 파일을 분석하는 기능**입니다. 파인튜닝, 실시간 카메라 입력, BEV,
거리 추정, 위험 판단, 음성·진동 알림은 포함하지 않습니다.
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
│   └── yolo/
│       └── finetune_v2_exp02_stage2_best.pt
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
- **영상:** 폴더 일괄 처리는 MP4만 지원합니다. 하위 폴더까지 자동 탐색하지 않습니다.
- **출력:** `outputs/videos/`가 미리 있어야 합니다. 추론 코드는 입출력 폴더를 자동 생성하지 않습니다.

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
  --output-path outputs/videos/result_OBS_260914_G24P_001_combined.mp4
```

이 이름도 이미 존재하면 다른 이름을 지정해야 합니다.

### 도보·장애물을 따로 확인할 때

비교 결과가 서로 겹치지 않도록 출력 이름을 구분합니다.

```bash
# 도보 마스크만 표시
python -m scripts.run_video_inference --mode sidewalk \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/videos/result_OBS_260914_G24P_001_sidewalk.mp4

# 장애물 박스만 표시
python -m scripts.run_video_inference --mode obstacle \
  --video-path data/samples/sample1/OBS_260914_G24P_001.mp4 \
  --output-path outputs/videos/result_OBS_260914_G24P_001_obstacle.mp4
```

단독 모드에서는 해당 모델만 로딩합니다. YAML은 아래의 전체 설정 구조를 유지하세요.

## 3. 결과 확인하기

기본 저장 위치는 `outputs/videos/result_원본파일명.mp4`입니다.
예를 들어 `test.mp4`의 결과는 `result_test.mp4`로 저장됩니다.
`--output-path`를 지정하면 지정한 이름을 그대로 사용합니다.

- 원본 영상 크기와 저장 FPS를 유지하며 MP4로 다시 인코딩합니다. 원본 오디오는 포함하지 않습니다.
- 진행 중에는 콘솔에 `영상 처리: 처리한 프레임 수/전체 프레임 수`가 표시됩니다.
- 완료되면 `결과 영상 저장: ...` 메시지가 나옵니다.
- 영상 파일만 저장합니다. 객체 좌표 JSON, 성능 평가표, 새 모델 가중치는 저장하지 않습니다.
- 저장 FPS를 유지한다는 뜻이지, 그 속도로 실시간 추론한다는 뜻은 아닙니다. 별도 속도 측정이 필요합니다.

## 4. 설정 바꾸기

기본 설정 파일은 [configs/inference.yaml](configs/inference.yaml)입니다.

```yaml
sample_dir: data/samples
output_dir: outputs/videos
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
```

| 설정 | 의미 |
| --- | --- |
| `sample_dir` | 처리할 MP4가 직접 들어 있는 폴더 |
| `output_dir` | 결과를 저장할 기존 폴더 |
| `device` | `auto`: CUDA 가능 시 GPU, 아니면 CPU / `cuda`: GPU 지정 / `cpu`: CPU 지정 |
| `mode` | `both`: 통합 / `sidewalk`: 도보만 / `obstacle`: 장애물만 |
| `mask2former.weights` | 가중치와 모델·전처리 설정이 있는 **폴더** |
| `yolo.weights` | YOLO 가중치 **파일** |
| `overlay_alpha` | 마스크 색상 비율. `0.55`는 원본 45% + 색상 55%. 클수록 진하게 표시 |
| `yolo.conf` | 객체 신뢰도 기준. 높이면 더 엄격하게 걸러지지만 놓치는 객체가 늘 수 있음 |
| `yolo.imgsz` | YOLO 내부 전처리 크기 기준. 결과 영상의 크기를 바꾸는 값은 아님 |
| `yolo.head` | 현재는 `nms`만 지원. 겹치는 탐지 박스를 정리하는 후처리 사용 |

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
| `--mode` | `both`, `sidewalk`, `obstacle` 선택 |
| `--device` | `auto`, `cpu`, `cuda` 선택 |
| `--conf` / `--imgsz` | YOLO 신뢰도 / 전처리 크기 지정 |

`--output-path`는 처리할 영상이 한 개일 때만 사용할 수 있고 확장자는 `.mp4`여야 합니다.
기존 `--model-dir`는 `--mask2former-weights`의 별칭으로 사용할 수 있습니다.
이전 YAML의 최상위 `model_dir`는 `mask2former` 아래의 `weights`로 옮겨야 합니다.
전체 옵션은 `python -m scripts.run_video_inference --help`로 확인합니다.

## 5. 자주 발생하는 문제

| 메시지·상황 | 확인할 것 |
| --- | --- |
| 결과 영상이 이미 있음 | 다른 `--output-path`나 기존 출력 폴더를 지정. 자동 덮어쓰기·건너뛰기 없음 |
| 샘플 MP4가 없음 | `data/samples`가 아니라 실제 영상이 들어 있는 `data/samples/sample1`을 지정했는지 확인 |
| 출력 폴더가 없음 | `outputs/videos` 등 지정한 폴더를 먼저 준비. 자동 생성하지 않음 |
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
| [src/pipeline.py](src/pipeline.py) | 영상 읽기 → 두 모델 추론 → 결과 합성 → MP4 저장 |
| [src/sidewalk.py](src/sidewalk.py) | Mask2Former 로딩과 보행 영역 추론 |
| [src/obstacle.py](src/obstacle.py) | YOLO 로딩과 장애물 추론 |
| [src/visualization.py](src/visualization.py) | 반투명 마스크와 클래스별 색상의 객체 박스·글자 표시 |
| [tests/test_integration.py](tests/test_integration.py) | 설정·옵션·좌표·클래스별 색상·영상 저장 동작 검증. 일반 추론 실행에는 사용하지 않음 |

각 모델은 한 번만 로딩하고 모든 영상에서 재사용합니다.
프레임마다 두 모델을 순서대로 실행하며, 둘 다 색칠 전의 같은 원본을 입력받습니다.
원본 도보·장애물 레포의 코드나 학습 데이터는 실행에 필요하지 않습니다.

개발 시 결과를 연결하는 기준:

- 공통 입력: OpenCV BGR 이미지 `(높이, 너비, 3)`.
- `SidewalkSegmenter.predict(frame)`: 원본 크기의 정수 클래스 지도 반환. 번호는 `label_ids`로 조회.
- `ObstacleDetector.predict(frame)`: `xyxy`, `class_id`, `class_name`, `confidence`를 담은 목록 반환. 탐지 객체가 없으면 빈 목록.
- `xyxy`는 원본 픽셀 기준 `[왼쪽, 위, 오른쪽, 아래]`. 다시 크기 비율을 곱하지 않습니다.

자동 테스트 실행:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

실제 가중치나 GPU 없이 검사하며, 테스트용 임시 영상은 정리합니다.
이 테스트는 코드 연결과 입출력 동작을 확인하는 것으로, **실제 탐지 정확도나 실시간 속도를 보장하지 않습니다.**
실제 가중치를 사용하는 통합 추론은 별도로 실행하고, 같은 입력에 대한 단독·통합 결과를 비교해야 합니다.

요약: 가상환경 활성화 → 가중치·영상 준비 → 샘플 폴더 선택 → 실행 → `outputs/videos` 결과 확인.
