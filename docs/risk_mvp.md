# 장애물 위험 판단 MVP: 실제 구현과 초기 가정

> 2026-09-21 후속 수정으로 ROI·접점·경고 해제·보도 방향 설정이 변경됐다. 아래는 초기 MVP 기록이다. 현재 구현은 [피드백 반영 구현](risk_revision_implementation_20260921.md)을 참고한다.

2026-09-21. 초기 구현은 저장된 MP4를 대상으로 한다. 위험 등급은 실험용 규칙의 결과이며 실제 충돌 확률이나 측정 거리가 아니다.

## 구현한 내용

- YOLO는 원본 전체 화면을 한 번 탐지한다. ROI로 입력을 자르지 않는다.
- BoT-SORT(기본) 또는 ByteTrack으로 선택적 ID를 붙인다. 원본 박스·클래스·confidence를 유지한다.
- ID가 없거나 추적기가 실패해도 현재 검출 객체의 경로 점유를 평가한다.
- 가까운 경로 점유, 일반 경로 점유, 짧은 측방 진입 예측을 판단한다.
- Mask2Former는 동일 프레임의 객체 접점 주변 보도/횡단보도 비율을 문맥으로 기록한다.
  초기 MVP에서는 이 비율로 위험 등급을 올리거나 내리지 않는다.
- 카메라 급변, 시간 오류, 이력 부족, 클래스 변화, 가림·경계 잘림에 대한 운동/TTC 제한을 둔다.
- TTC는 유효할 때 로그에 계산하되 기본 설정에서 경고에는 사용하지 않는다.
- 상대 확대/축소율과 approaching/receding/steady/unknown 상태를 기록한다. 미터 거리·속도는 아니다.
- review_overlay=true로 ROI·등급·TTC를 크게 표시하는 임시 검토 화면을 사용할 수 있다.
- 경고 이벤트의 중복 억제, 상향 재경고, 하향 유지 시간을 적용한다.
- 결과 MP4와 같은 이름의 .risk.jsonl에 객체별 판단과 사유를 저장한다.
- 신호등 탐지·연결·색상 분류·표시 코드와 traffic 설정은 수정하지 않았다.
  일반 traffic_light는 추가 위험 평가 대상에서 제외한다.
- ARCore, 음성·진동 출력, 실시간 카메라·비동기 추론은 아직 구현하지 않았다.
  현재 이벤트는 영상 표시와 JSONL 기록에 사용된다.

## ROI를 잘라 탐지하지 않는 이유

중앙 진행 경로 바깥에서 들어오는 객체를 일찍 관찰하기 위해 전체 프레임을 사용한다.
이는 위험 후보를 잘라 버리지 않기 위한 선택이다. 실제 위험 재현율이 얼마나 개선되는지는 라벨링된 영상으로 평가해야 한다.
같은 imgsz에서는 넓은 시야의 작은 객체가 축소되므로 전체 화면 탐지가 모든 객체의 정확도를 보장하지는 않는다.
현재 imgsz=640와 conf=0.25는 기존 설정 그대로다.

고정 ROI는 카메라가 전방을 향한다는 가정을 둔다.
휴대폰을 옆으로 향하거나 촬영 자세·장착 높이가 바뀌면 좌표를 다시 보정해야 한다.
카메라 시야 밖 물체를 감지하거나 진행 방향 자체를 추정하는 기능은 없다.
바닥에 놓인 물체의 bbox 하단을 지면 접점 근사로 사용하므로 매달린 물체·머리 높이 장애물에는 별도 검증이 필요하다.

## 좌표와 초기값

좌측 상단=(0,0), 우측 하단=(1,1)이다. 픽셀 좌표는 (x/W, y/H)로 정규화한다.
아래 값은 기존 데이터에서 학습하거나 최적화한 값이 아니라 개발자가 정한 시작값이다.
모두 configs/inference.yaml의 risk 항목에서 조정할 수 있다.

| 항목 | 초기값 | 의미 |
| --- | --- | --- |
| corridor_polygon | (0.42,0.45), (0.58,0.45), (0.90,1), (0.10,1) | 위쪽 좁고 아래쪽 넓은 예상 진행 경로 |
| immediate_polygon | (0.30,0.75), (0.70,0.75), (0.90,1), (0.10,1) | 가까운 경로의 화면상 근사 |
| footprint_height_ratio | 0.15 | bbox 하단 15% 높이의 띠 |
| overlap_threshold | 0.20 | 띠 면적 중 ROI와 겹치는 비율 20% |
| edge_margin_ratio | 0.08 | 네 경계에서 화면 폭/높이의 8% 이내 후보 |
| history_window_s | 0.60초 | 운동 회귀에 사용하는 최근 이력 |
| min_history_s | 0.20초 | 운동 계산 최소 기간. 최소 3개 관측도 필요 |
| prediction_horizon_s | 0.80초 | 화면상 측방 진입 예측 길이 |
| min_lateral_speed | 0.05 /초 | 초당 화면 폭 5% 이상의 상대 수평 이동 |
| lateral_near_y | 0.60 | 하단 접점이 화면 높이 60% 아래인 경우 진입 주의 |
| reset_gap_s | 0.50초 | 긴 프레임 간격 또는 재관측 간격의 초기화 기준 |
| min_expansion_rate | 0.05 /초 | TTC를 계산할 최소 상대 확대율 |
| max_motion_residual | 0.03 | 위치·높이 선형 회귀의 최대 정규화 RMSE |
| camera_max_rotation_deg | 프레임 간 4도 | 전역 회전 추정이 이 값보다 크면 운동 판단 제한 |
| camera_max_translation | 0.12 | 정규화 평행이동 벡터 크기 |
| camera_max_scale_change | 0.10 | 프레임 간 전역 스케일 변화 10% |
| ttc_alerts | false | TTC는 기록만 하고 경고에 사용하지 않음 |
| ttc_danger_s / ttc_caution_s | 1.5 / 3.0초 | ttc_alerts를 켤 경우에만 적용할 실험값 |
| release_hold_s | 0.30초 | 경고 하향·관측 소실 후 해제 유지 시간 |
| repeat_cooldown_s | 2.0초 | 같은 경고 재발행 간격. 상향 경고는 즉시 발행 |
| event_match_iou | 0.30 | ID 없는 관측과 이벤트의 짧은 공간 연결 기준 |

경계 8% 표시는 후보 표시용이다. 경계에 있다는 사실만으로 caution/danger가 되지 않는다.
TTC의 bbox 잘림 검사는 별도로 원본 이미지 가장자리 1픽셀 이내 접촉을 사용한다.

카메라 품질 검사의 내부 시작값:
- grayscale 가로 320픽셀로 축소, 특징점 최대 120개
- goodFeaturesToTrack: qualityLevel=0.01, minDistance=8
- 최소 대응점 8개, affine RANSAC 오차 3픽셀, inlier 비율 최소 0.5
이는 정확한 카메라 자세 추정이나 IMU 보정이 아니다. 관측이 부족하면 운동값을 제한한다.

tracking 기본값은 Ultralytics의 두 추적기 기본 임계값을 사용한다.
track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
track_buffer=30, match_thresh=0.8, fuse_score=true.
BoT-SORT는 sparseOptFlow, with_reid=false를 사용하며 별도 ReID 모델은 실행하지 않는다.
현재 conf=0.25이므로 0.1~0.25 검출을 활용하는 저신뢰 재연결 단계는 사용되지 않는다.
track_buffer는 내부 ID 재연결 기간이며 사라진 박스 표시 기간이 아니다.

## 수식과 등급 결정

bbox B=(x1,y1,x2,y2), 높이 h=y2-y1일 때:
- 하단 띠 F=(x1, y2-0.15h, x2, y2)
- 경로 겹침률 r(R)=area(F∩R)/area(F)
- 즉시 영역의 r≥0.20이면 danger
- 그 외 진행 경로의 r≥0.20이면 caution
- 둘 다 아니면 기본 monitor

작거나 폭이 넓은 객체를 하단 중앙점 하나로만 판단하지 않기 위해 면적 비율을 사용한다.
20%는 물리적 충돌 기준이 아니며 객체 종류·촬영 조건에 따라 조정해야 한다.

측방 이동:
- 최근 이력의 접점 p=((x1+x2)/2,y2)에 대해 p(t)≈a+v·t를 최소제곱으로 맞춘다.
- 미래 띠 F(t+τ)=F(t)+v·τ가 진행 경로와 겹치는지 확인한다.
- 0.8초를 8구간으로 나눠 0.1초 간격으로 검사한다. horizon 변경 시 간격도 바뀐다.
- 현재 경로 밖이고 |vx|≥0.05/초, y2≥0.60이며 예상 겹침률≥0.20이면 lateral_entry 사유의 caution.
- 현재 이미 가까운 경로를 점유하면 이력을 기다리지 않고 danger.
- 이 시간은 화면상 경로 진입 예상 시간이며 실제 충돌 시간이나 m/s가 아니다.

확대 TTC:
- 현재 정규화 bbox 높이 h와 회귀 기울기 dh/dt를 사용한다.
- 상대 확대율 q=(dh/dt)/h, TTC≈1/q=h/(dh/dt).
- q≥0.05/초, 시간·카메라·이력이 유효하고 이력 중 잘린 박스가 없어야 계산한다.
- 박스 자세가 일정하고 대략 일정한 상대 접근이라는 가정이 있다.
- 측방 횡단에 TTC를 필수 조건으로 걸지 않는다.
- TTC를 경고에 켜더라도 진행 경로 겹침 조건을 함께 요구한다.

assessment_quality의 valid는 위 계산 조건이 충족됐다는 의미이며 위험 판정의 정답 보장이 아니다.
alert_level은 해제 유지 시간을 반영한 표시/알림 등급이고 risk_level은 현재 관측으로 계산한 등급이다.

## 코드와 실행

- src/tracking.py: 원본 탐지 보존 adapter. 실패 시 해당 영상의 추적을 중단하고 ID 없는 탐지를 유지.
- src/risk_config.py: 초기값과 검증.
- src/risk_geometry.py: ROI 겹침·경계·보도 문맥.
- src/risk_motion.py: 전역 움직임 검사·이력·측방 진입·TTC.
- src/risk.py: 시간·상태 관리와 등급 규칙.
- src/alert_policy.py: 중복 억제·상향·해제.
- src/risk_visualization.py: 별도 위험 표시. 기존 클래스 색상/신호등 함수 보존.
- src/risk_log.py: 기존 파일을 덮어쓰지 않는 로그 저장.
- src/pipeline.py: YOLO→위험 평가→Mask2Former 문맥→기존 신호등→표시·저장.

현재 설정은 risk.enabled=true이다. both, obstacle, all에서 장애물 위험 기능을 사용한다.
traffic, sidewalk 단독 모드에는 위험 모듈을 실행하지 않는다.
기존과 같은 표시가 필요하면 --no-risk 또는 risk.enabled=false를 사용한다.

```bash
# 통합 레포 루트에서, 기존 환경 사용
obs_env/bin/python -m scripts.run_video_inference \
  --video-path data/samples/sample1/OBS_260913_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_OBS_260913_G24P_001_risk.mp4 \
  --mode both --risk

# 기존 탐지 표시만
obs_env/bin/python -m scripts.run_video_inference \
  --video-path data/samples/sample1/OBS_260913_G24P_001.mp4 \
  --output-path outputs/runs/manual/result_OBS_260913_G24P_001_no_risk.mp4 \
  --mode both --no-risk
```

출력은 MP4와 .risk.jsonl이다. JSONL 첫 프레임에는 적용된 risk/tracking 설정도 기록한다.
파일이 이미 있으면 다른 출력 이름을 사용한다. 추론 오류 시 이번 실행의 임시 파일을 정리한다.
시간은 소스 PTS를 사용한다. PTS가 유효하지 않으면 명목 FPS로 표시 시각만 계산하고 운동 판단을 끈다.
시간 역전·긴 공백·해상도 변경은 상태를 초기화한다. state_epoch로 로그에서 구분할 수 있다.
실시간 프레임 드롭에 맞춰 기존 칼만 필터의 dt를 수정하는 기능은 아직 없다.

## 검증과 한계

- 기존 통합/신호등 49개와 신규 위험/추적 23개, 총 72개 테스트가 통과했다.
- ID 없는 첫 검출, 측방 진입, 평행 이동, 정지 경로 점유, TTC 제한, 누락 마스크, 시간 오류,
  영상별 reset, 빈 탐지, 추적 오류, 로그 정리, 위험을 켠 상태의 신호등 호출 보존을 검사했다.
- 실제 YOLO+Mask2Former+BoT-SORT를 CUDA에서 원본 샘플 앞 30프레임으로 실행했다.
- 출력 영상 30프레임 디코딩, JSONL 30행, 유효 PTS, tracker active를 확인했다.
- 원본 탐지 관측 343건에 보도 문맥이 연결됐다. monitor 247건, caution 96건이었다.
  카메라 안정성 검사는 29프레임에서 통과했고 첫 프레임은 이전 영상이 없어 제한됐다.
- 이 숫자는 위험 판정 정확도가 아니다. 실제 정답 위험 구간과 비교하는 평가가 별도로 필요하다.
- 기본 ROI가 가까운 장애물·갑작스러운 진입에 맞는지는 장착 조건별로 검증해야 한다.
- bbox 하단 접점이 부적절한 공중·머리 높이 장애물, 카메라가 다른 방향을 향하는 경우,
  시야 밖 객체, 검출되지 않은 객체는 현재 구현의 남은 한계다.

실행 산출물: outputs/experiments/2026-09-21_risk-detection-evaluation/01_mvp-smoke-test/result_risk.mp4,
result_risk.risk.jsonl, summary.json.

## 8개 영상 검토 실행 (2026-09-21)

기본 설정의 ttc_alerts=false와 달리, 이번 검토 영상의 별도 설정은
ttc_alerts=true, review_overlay=true, overlay_alpha=0.30이다.
유효한 TTC가 1.5초 이하이고 진행 ROI를 점유하면 danger로 올린다.
화면의 size %/s는 박스 높이의 상대 변화율이며 물리적 접근 속도가 아니다.
전체 화면/ROI 자르기 비교 수치와 해석, 속도·거리의 구현 한계는
[ROI 비교 및 8개 영상 검토](roi_and_risk_review_20260921.md)에 기록했다.
