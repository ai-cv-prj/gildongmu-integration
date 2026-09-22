# 보행 장애물 위험 판단 구현안

작성: 2026-09-21  
대상: gildongmu-integration 통합 레포 루트  
상태: 아래는 최초 설계안이다. 2026-09-21에 ROI·원본 탐지 보존 추적·측방 진입·로그 중심의 MVP를 구현했다. 실제 파일 구성, 초기값, 구현 범위와 남은 항목은 [risk_mvp.md](risk_mvp.md)를 기준으로 확인한다.

## 1. 결정과 적용 범위

전체 화면 YOLO 탐지를 보존하고, 즉시 위험 판단과 시간 이력 기반 판단을 결합한다.
고정 ROI는 탐지 범위를 자르는 용도가 아니라 진행 경로의 위험도를 판단하는 기준이다.
Mask2Former 보도/횡단보도 마스크는 보조 문맥이며, 마스크 밖이라는 이유로 장애물을 제외하지 않는다.

현재 저장된 RGB MP4에는 ARCore의 depth, pose, 카메라 보정 정보가 없다.
따라서 현재 레포의 첫 구현은 ROI + tracking으로 진행하고, depth는 선택 입력으로 설계한다.
ARCore가 TTC보다 부적합하거나 정확도가 낮아서 제외하는 것은 아니다.
ARCore는 거리 관측을 제공하고 TTC는 접근 추세로부터 시간 여유를 추정하므로 함께 사용할 수 있다.
지원 Android 기기에서 RGB/depth/pose를 수집할 수 있다면 ARCore 검증을 병행할 가치가 있다.

카메라 시야에 아직 들어오지 않은 뒤쪽·옆쪽 물체는 이 단안 RGB 시스템으로 관측할 수 없다.
시야 경계로 진입하거나 다른 객체 뒤에서 나타나는 경우는 최초 검출부터 처리한다.
관측 범위 밖을 감지해야 한다면 광각/추가 카메라 등 입력 하드웨어 확장이 필요하다.
현재 32개 클래스에 없는 물체와 YOLO 미검출을 tracking만으로 복원할 수는 없다.

## 2. 실제 환경 확인과 추적 검증

obs_env에 lap==0.5.12만 설치했다.
설치 명령은 python -m pip install --no-deps --only-binary=:all: lap==0.5.12 이다.
Ultralytics 8.4.152, NumPy 2.5.3, Torch 2.14.0 등 기존 핵심 패키지 버전은 유지됐다.
requirements.txt에는 lap 고정 버전 한 줄을 추가했다.

검증 자료: outputs/experiments/2026-09-21_risk-detection-evaluation/02_tracking-backend-check/
- verify_tracking.py: 합성 입력과 실제 영상 검증 코드
- report.json: 버전, 검증 결과, 실행 시간
- detections.jsonl: 프레임별 원본 탐지와 선택적 ID
- tracking_comparison.mp4: 두 추적기 비교 영상
- frame_0060.jpg: 비교 영상의 한 프레임

실제 가중치 finetune_v2_exp02_stage2_best.pt와 OBS_260913_G24P_001.mp4 첫 120프레임을 사용했다.
영상은 1080x1920, 약 30.05 FPS이며 YOLO 입력은 imgsz=640, conf=0.25, CPU였다.
각 프레임의 YOLO forward는 한 번만 수행하고 결과를 두 추적기에 공유했다.
준비용 YOLO 워밍업 한 회는 실행 시간 집계에서 제외했다.

| 항목 | ByteTrack | BoT-SORT |
| --- | ---: | ---: |
| 원본 탐지 관측 수 | 1,392 | 1,392 |
| ID 연결 관측 수 | 998 | 1,183 |
| ID 없는 관측 수 | 394 | 209 |
| ID 없는 탐지가 있었던 프레임 | 112 | 94 |
| 생성된 고유 ID 수 | 73 | 30 |
| 추적 update 평균 | 0.944 ms | 8.392 ms |
| 추적 update p95 | 1.260 ms | 9.569 ms |

관측 수는 여러 프레임에서 센 박스의 합계이며 실제 객체 수가 아니다.
두 결합 결과 모두 원본 박스 1,392건의 좌표와 관측 수를 보존했다.
비교 영상은 저장 후 120프레임이 디코딩되는 것을 확인했다.
BoT-SORT는 sparseOptFlow, with_reid=False이고 ByteTrack은 카메라 보정이 없다.
더 적은 ID 수나 더 많은 ID 연결이 정확한 연결이라는 뜻은 아니다. 정답 ID 평가는 하지 않았다.
Mask2Former, 위험 판단, 스마트폰 실행 성능은 이번 검증 범위에 포함하지 않았다.
검증 영상의 색상은 ID 유무 확인용이며 제품의 클래스 색상 정책과 별개다.

두 추적기 모두 합성 입력에서 다음을 통과했다.
1. 첫 프레임의 객체는 즉시 ID 출력.
2. 두 번째 프레임에 추가된 객체는 원본 탐지에는 남고 ID는 없음.
3. 다음 프레임에 매칭되면 ID 확정.
4. 미검출 객체는 lost 목록에 남지만 현재 출력에는 없음.
5. 짧은 미검출 후 같은 ID로 재연결.
6. 탐지 0건 입력도 상태 갱신하며 출력 0건.
7. reset 후 이전 영상의 상태가 사라짐.

## 3. 파일별 변경안

| 파일 | 역할 |
| --- | --- |
| src/obstacle.py | 기존 predict(frame) 목록과 원본 좌표·confidence 보존. 위험 기능 때문에 track()으로 교체하지 않음 |
| src/frame_context.py (신규) | stream_id, session_id, frame_index, 촬영 timestamp, 크기, 선택적 카메라/자세/depth 정보 |
| src/tracking.py (신규) | 탐지 목록을 tracker 입력으로 변환, detection 인덱스로 ID 부착, 영상별 reset |
| src/risk_geometry.py (신규) | 정규화 ROI, 발밑 접점/하단 띠, 경계 접촉, 경로 점유, 보도 문맥 |
| src/risk_motion.py (신규) | ID별 시간 이력, 측방 진입, 상대 확대율, TTC 근사와 유효성 검사 |
| src/risk.py (신규) | 즉시 후보와 움직임 근거를 규칙으로 결합, 등급·이유·판단 품질 반환 |
| src/alert_policy.py (신규) | 경고 이벤트 유지·해제, 중복 억제, 위험 상향 시 재경고 |
| src/depth.py (후속) | 유효 depth를 선택 입력으로 공급. 현재 단계는 None 허용 |
| src/pipeline.py | 모델 결과와 상태 모듈 연결, 소스 시간 전달, 영상 시작 reset, JSONL 저장 |
| src/visualization.py | 클래스 색상 유지, ID 텍스트, 위험 배지와 ROI·진입 방향 표시 |
| configs/inference.yaml | tracking/risk의 활성화, 임계값, 출력 옵션 |
| configs/trackers/*.yaml (신규) | ByteTrack/BoT-SORT를 명시적으로 선택 |
| tests/test_tracking.py, tests/test_risk.py (신규) | 원본 탐지 보존, 갑작스러운 진입, 상태 격리, 경고 정책 검증 |

현재 설치 버전의 기본 tracker는 tracktrack.yaml이다. tracker 이름을 생략하지 않는다.
첫 위험 PoC에서는 기존 conf=0.25를 유지한다. 따라서 ByteTrack의 0.1~0.25 저신뢰 재연결 경로는 활용하지 못한다.
후속 실험에서 탐지 입력 conf를 낮춘다면 경고 기준과 추적 재연결 기준을 분리하고 미탐/오경고를 재평가한다.
보행 카메라의 우선 비교 후보는 BoT-SORT이며 ByteTrack을 가벼운 기준안으로 유지한다.

## 4. 데이터 계약

원본 탐지 필드:
- detection_index: 이 프레임에서의 입력 순서. 프레임 간 ID로 사용하지 않음
- xyxy, class_id, class_name, confidence: 기존 YOLO 결과 그대로

부가 추적/움직임 필드:
- track_id: 없을 수 있음
- observed: 현재 실제 검출 여부
- history_seconds, last_seen_timestamp
- edge_contact: left/right/top/bottom 중 해당 경계
- motion_quality: valid / insufficient / unstable
- image_velocity_norm_per_s: 화면 크기로 정규화한 상대 이동. m/s가 아님
- time_to_corridor_s: 화면상 경로 진입 예상 시간. 실제 충돌 시간과 구분
- ttc_scale_s: 신뢰할 수 있을 때만 계산, 나머지는 null

위험 출력:
- risk_level: monitor / caution / danger
- assessment_quality: valid / limited / unknown
- reasons: near_path_occupied, lateral_entry, looming, new_edge_candidate 등
- depth_z_m: 선택적 카메라 광축 방향 거리. 없으면 null
- confidence는 객체 탐지 신뢰도로 보존하며 충돌 확률로 해석하지 않음

판단 품질과 위험 등급을 분리한다.
이력이 부족하다고 근거리 경로 점유 경고를 취소하지 않고, 미검출을 안전 판정으로 변환하지 않는다.
시간 이력 키는 (stream_id, session_id, track_id)로 구성한다.

## 5. 프레임 처리 순서

다음은 제안 인터페이스를 보여 주는 의사코드다.

```python
# 영상마다
tracker.reset()
motion.reset()
alerts.reset()
risk.reset()

for context, frame in source:
    raw = detector.predict(frame)              # 전체 화면 YOLO 1회

    # ID와 segmentation을 기다리지 않는 첫 경고 경로
    immediate = risk.evaluate_immediate(raw, context)
    alert_events = alerts.raise_immediate(immediate, context)

    tracked = tracker.attach(raw, frame, context)
    motion_features = motion.update(tracked, context)
    sidewalk = sidewalk_provider.get_for(context)
    depth = depth_provider.get_for(context)    # 미지원/누락이면 None

    assessed = risk.evaluate(
        raw, tracked, motion_features, sidewalk, depth, context
    )
    alert_events += alerts.reconcile(assessed, context)
    render(frame, raw, tracked, assessed)
    write_jsonl(context, raw, tracked, assessed, alert_events)
```

같은 프레임에서 즉시 경고와 종합 경고를 중복 발행하지 않도록 이벤트 키를 공유한다.
위험 판단에는 현재 pipeline의 traffic_light 표시 필터 이전 원본 목록을 전달한다.
신호등 색상과 장애물 충돌 위험은 각각의 의미로 처리한다.

기존 offline MP4 검증에서는 같은 프레임의 segmentation 결과를 종합 판단에 사용할 수 있다.
현재 pipeline은 Mask2Former를 먼저 기다린 뒤 YOLO를 실행하므로 실시간 즉시 경고에는 실행 순서 변경이 필요하다.
실시간 단계에서는 YOLO 결과 직후 즉시 판단을 발행하고 segmentation은 별도 주기로 최신 결과를 공급한다.
서로 다른 시점의 마스크는 나이와 정렬 가능성을 검사하고, 오래됐거나 정렬이 불가능하면 unknown으로 취급한다.
큐는 길게 쌓지 않는다. 단, 모델 주기 분리만으로 GPU 연산 경쟁이나 지연이 사라지는 것은 아니므로 측정한다.

## 6. 시야 가장자리와 갑작스러운 등장

탐지는 항상 전체 화면에서 수행한다. 중앙 ROI 바깥 객체도 추적 후보에 포함한다.
경계 감시 영역은 좌/우뿐 아니라 상/하 경계도 포함한다.
중앙 ROI는 예상 진행 경로, 더 가까운 하단 ROI는 즉시 경로 점유 판단용이다.
ROI는 카메라 장착 위치·각도·보행 방향이 고정된 조건에서 보정해야 한다.
카메라가 보행 방향과 다르게 향하는 경우에는 경로 ROI의 판단 품질을 낮춘다.

| 상황 | 첫 관측에서의 처리 | 이력이 생긴 뒤 |
| --- | --- | --- |
| 왼쪽/오른쪽에서 일부만 보이는 자전거 | 원본 박스 유지, 경계 후보. 가까운 경로 점유면 ID 없이 danger | 안쪽 진입 추세와 경로 교차를 계산 |
| 다른 객체 뒤에서 갑자기 나타난 사람 | 경계 접촉 여부와 무관하게 모든 신규 탐지를 평가 | 접근·횡단·통과 추세로 재평가 |
| 전방 볼라드로 사용자가 접근 | 정지 물체여도 경로 점유로 경고 | 상대 확대율과 유효 TTC 보완 |
| 옆 차도를 따라 평행 이동하는 자동차 | 경계에 있다는 이유만으로 danger를 주지 않음 | 경로 진입이 없으면 우선순위 낮게 유지 |
| 정면을 빠르게 가로지르는 킥보드 | 가까운 경로와 겹치면 즉시 경고 | 박스 확대가 없어도 측방 경로 진입 판단 |
| 상단에서 내려오는/몸 높이에 걸리는 객체 | 지면 접점 규칙이 부적절하면 bbox의 신체 통과 영역 점유로 후보 평가 | 영상만으로 높이·충돌 확정이 어려우면 limited |
| 카메라 급회전/큰 흔들림 | 현재 박스의 즉시 판단은 유지, 경로 방향 신뢰도 낮춤 | 운동/확대율 계산 중지 또는 이력 재시작 |
| 가림 또는 일시 미검출 | 새 박스를 지어내지 않음. 기존 경고의 짧은 유지 여부는 별도 정책 | 재등장 시 연결 품질 검사 후 이력 복구 |
| 아직 카메라 밖에 있는 객체 | 관측 불가 | 시야에 처음 들어온 프레임부터 처리 |

하단 중앙점 하나로 판단하지 않고 하단 띠의 폭과 경로 점유도 함께 사용한다.
경계에 닿아 박스가 잘렸을 가능성이 있으면 높이/면적 변화가 인위적으로 커질 수 있으므로 TTC를 계산하지 않는다.
보도 마스크가 가림 때문에 비었거나 객체가 아직 차도에 있어도 진입 후보는 유지한다.
crosswalk 마스크가 있다는 사실을 실제 횡단 의도나 안전 상태로 해석하지 않는다.
신규/경계 접촉만으로 위험을 확정하지 않으며, 근거리 경로 점유 같은 공간 근거를 함께 요구한다.

## 7. 추적, 방향, TTC의 계산과 품질

tracking.py는 현재 목록을 [x1,y1,x2,y2,confidence,class_id] 배열로 변환하고 Boxes로 감싼다.
tracker.update의 마지막 열 detection index로 ID를 원본 목록의 복사본에 부착한다.
트래커가 반환한 보정 박스로 원본 YOLO 박스를 덮어쓰지 않는다.
탐지가 0건인 프레임에도 빈 결과로 update하여 lost 상태를 정상 갱신한다.
현재 관측이 없는 lost 트랙의 내부 예측은 재연결에 사용하고 현재 검출 박스로 표시하지 않는다.

프레임 시각은 촬영/source timestamp를 사용한다. 추론 실행 시간이 시간 간격의 기준이 아니다.
MP4는 검증된 PTS를 우선하며 고정 FPS 파일에서만 frame_index/fps를 fallback으로 허용한다.
변동 FPS 파일의 타임스탬프를 복원하지 못하면 운동 기반 수치의 품질을 unknown으로 처리한다.
비정상 시간 역전, 큰 프레임 간격, 해상도/회전 변경에는 상태를 초기화하거나 해당 운동 계산을 무효화한다.
라이브에서 timestamp를 추가하는 것만으로 기존 칼만 필터의 고정 dt가 바뀌지는 않는다.
추적 입력 주기를 관리하고, 프레임 드롭이 빈번하면 dt를 지원하는 추적기 수정/교체를 별도로 평가한다.

측방 진입:
- 최근 유효한 짧은 구간에서 바닥 접점 또는 보수적인 박스 폭의 이동을 추정한다.
- 현재 위치에서 짧은 예측 구간 동안 휩쓰는 영역이 진행 경로와 겹치는지 계산한다.
- 불확실성 여유 폭을 포함하되 길게 외삽하지 않는다.
- 이것은 화면상 경로 진입 단서이며 3D 충돌 증명이나 실제 이동 속도가 아니다.

확대 기반 TTC:
- 객체 높이 h와 시간 미분 dh/dt로 TTC ≈ h / (dh/dt)를 근사한다.
- dh/dt가 양수이고 충분히 크며 안정적인 경우에만 사용한다.
- 고정 물리 크기/자세와 대략 일정한 상대 접근 운동 가정이 필요하다.
- 이력 부족, ID 변경 의심, 큰 클래스 변화, 가림, 경계 잘림, 급회전, 나쁜 시간 간격이면 null.
- 측방 횡단은 TTC가 없더라도 위험할 수 있으므로 TTC를 필수 조건으로 사용하지 않는다.
- 카메라 전진 때문에 정지 장애물이 커지는 것은 유효한 상대 접근 단서다.
  전역 움직임을 모두 빼서 정지 장애물의 접근 위험까지 제거하면 안 된다.
- BoT-SORT GMC는 ID 매칭 보조이며 정확한 카메라 자세/객체 속도를 제공하는 센서가 아니다.

## 8. 규칙과 알림

첫 버전은 설명 가능한 우선순위 규칙을 사용한다.
1. 현재 가까운 진행 경로의 점유가 충분하면 ID·TTC 없이 danger 후보.
2. 외부에서 경로로 진입하는 추세와 근접 단서가 결합되면 caution/danger.
3. 경로상 객체의 유효한 확대 TTC가 짧아지면 위험 수준 상향.
4. depth가 유효하면 거리 근거를 추가하지만 누락 시 기존 경로로 즉시 fallback.
5. 나머지는 monitor. 정보가 부족하면 assessment_quality에 기록.

처음 등장한 후보의 최소 관측 시간을 일괄 경고 조건으로 넣지 않는다.
약한 후보의 반복성 확인과 danger의 즉시 상승을 분리한다.
위험 하향에는 짧은 유지 시간을 적용해 깜빡임을 줄인다.
같은 객체의 반복 경고는 억제하되 danger 상승·재접근에는 다시 알린다.
ID 없는 후보에는 짧은 공간/시간 이벤트 연결을 사용하고, 근접한 서로 다른 객체의 경고를 합치지 않도록 범위를 제한한다.
lost 트랙 유지 기간과 경고 유지 기간은 별개 설정이다.

## 9. 설정 초안

다음 숫자는 보정과 회귀 실험을 시작하기 위한 예시이며 검증된 안전 임계값이 아니다.
ROI 좌표는 촬영 장착 조건을 정한 뒤 설정한다.

```yaml
tracking:
  enabled: true
  tracker: configs/trackers/botsort.yaml
  preserve_untracked: true
  reset_gap_s: 0.5

risk:
  enabled: true
  detection_scope: full_frame
  use_sidewalk_context: true
  require_sidewalk_overlap: false
  camera_profile: calibrated_forward_mount
  corridor_polygon: null       # 해당 장착 조건의 정규화 좌표 필수
  immediate_polygon: null     # 해당 장착 조건의 정규화 좌표 필수
  edge_margin_ratio: 0.08
  motion:
    history_window_s: 0.6
    min_history_s: 0.2         # 운동 계산에만 적용, 즉시 경고에는 미적용
    prediction_horizon_s: 0.8
  ttc:
    enabled: false            # 우선 로그로 검증 후 경고 근거로 활성화
  depth:
    provider: none            # 향후 recorded_arcore / live_arcore
  alerts:
    release_hold_s: 0.3
    repeat_cooldown_s: 2.0
    repeat_on_escalation: true
  log_jsonl: true
```

risk 비활성화 시 기존 모드와 결과 동작을 유지한다.
tracker 실패 시 원본 탐지와 즉시 판단은 유지하고 degraded 이유를 기록한다.
운동/depth/segmentation의 모듈 장애로 원본 탐지가 사라지지 않도록 한다.

## 10. ARCore 연결 계약

Android 수집부에서 RGB, depth, timestamp, intrinsics, pose, 좌표변환/회전 정보를 전달한다.
일반 RGB MP4를 입력으로 과거 depth가 자동 생성되는 구조는 아니다.
녹화 기반 검증을 원하면 RGB와 동기화된 depth·메타데이터 파일을 함께 수집한다.
depth 지도가 RGB와 같은 크기·크롭이라고 가정하지 않고 좌표변환을 적용한다.
ARCore의 값은 광축 방향 Z 거리이며 픽셀 광선을 따른 유클리드 거리와 구분한다.
무효/누락/오래된 depth, 지원하지 않는 기기, 추적 실패는 None으로 처리한다.
객체 bbox 전체의 최솟값 하나로 거리를 정하지 않는다.
박스 내부의 유효 픽셀 분포·일관성·배경 혼합을 확인하고 신뢰할 수 있는 경우만 거리 근거로 사용한다.
경계에서 일부만 보이는 새 객체의 depth가 없거나 부정확해도 즉시 탐지 경로는 실행한다.
객체 마스크 추가는 bbox 기반 거리 혼합 문제가 실제로 확인됐을 때 판단한다.

공식 근거:
- https://developers.google.com/ar/develop/depth
- https://developers.google.com/ar/develop/java/depth/developer-guide

## 11. 구현 순서와 합격 기준

1차: 원본 탐지 보존 adapter + ROI 즉시 판단 + JSONL + 표시 + 영상별 reset.
2차: 경계/가림 후 등장 처리 + 측방 진입 + 알림 안정화.
3차: TTC를 로그 전용으로 평가하고 유효성/경로 조건을 검증한 뒤 경고에 반영.
4차: 목표 Android 기기를 확보하면 ARCore 입력을 추가하고 같은 위험 평가기로 비교.

필수 자동 검증:
- 기존 predict와 원본 detection 개수/좌표/클래스/confidence 보존
- 새 객체의 첫 프레임에 track_id가 없어도 즉시 경고 가능
- 좌/우 경계 및 프레임 중앙의 가림 뒤 등장, 경로 밖 평행 이동의 구분
- 정지 장애물 접근, 확대 없는 횡단, 잘린 박스의 TTC 무효화
- segmentation 공백/오래된 마스크/depth 누락에도 원본 탐지와 즉시 판단 유지
- 빈 detection, ID 스위치, 영상 전환, timestamp 역전/긴 간격
- 중복 이벤트 억제와 위험 상향 재경고
- risk를 끈 기존 모드와 신호등 표시 동작의 회귀 검사

실영상 평가는 장면별 정답 위험 구간과 대상 ID를 별도로 작성한다.
위험 이벤트 재현율, 분당 오경고, 최초 검출→경고 지연, 위험 상황 시작→경고 지연을 구분한다.
ID 정확도는 IDF1/ID switch 등으로 확인하며 고유 ID 수를 정확도로 대체하지 않는다.
CPU/GPU 모델 시간 외에 수집→경고 p50/p95 지연, frame age, 열·배터리를 목표 기기에서 측정한다.
현 샘플 4초 구간의 실행 성공만으로 갑작스러운 진입에 대한 정확도나 안전성을 판정하지 않는다.
