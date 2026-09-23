"""gildongmu-test-app SESAC-78의 횡단보도 연결·대상 선택 로직.

원본: b3e4707. 신호등 객체 추적은 별도 BoT-SORT 모듈에서 수행한다.
2026-09-22 테스트 앱의 임시 선택·횡단보도 확인·확정 대상 유지 정책을 반영.
픽셀 좌표와 영상 프레임 문맥에 맞춰 이식.
"""
import math
from dataclasses import dataclass

from src.traffic_geometry import estimate_stripe_direction
from src.traffic_motion import estimate_camera_motion, motion_gray, transform_box


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    captured_at_ms: float
    confidence: float = 0.25


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


SIGNAL_DUPLICATE_IOU = 0.60


def suppress_duplicate_signals(signals):
    """같은 위치에 겹친 신호등 검출은 높은 신뢰도 하나만 남긴다.

    트래킹·대상 개수 판단 전에 적용한다. 살아남은 박스의 원래 순서를 유지하며,
    떨어진 다른 신호등과 횡단보도는 제거하지 않는다.
    """
    kept = []
    for index in sorted(range(len(signals)), key=lambda i: signals[i]["confidence"], reverse=True):
        if all(box_iou(signals[index]["xyxy"], signals[other]["xyxy"]) < SIGNAL_DUPLICATE_IOU
               for other in kept):
            kept.append(index)
    return [signals[index] for index in sorted(kept)]


def estimate_vanishing_point(frame, box, cv2):
    return estimate_stripe_direction(frame, box, cv2)


def crosswalk_position(box, width, height):
    bottom = box[3] / height
    offset = abs(center(box)[0] - width / 2) / width
    reasons = []
    if bottom < 0.55:
        reasons.append("bottom_too_high")
    if offset > 0.30:
        reasons.append("off_center")
    return bottom, offset, reasons


def choose_near_crosswalk(crosswalks, width, height):
    """가까운 쪽 경계가 화면 아래에 있고 중심에 가까운 횡단보도를 우선한다."""
    candidates = []
    for index, item in enumerate(crosswalks):
        bottom, offset, reasons = crosswalk_position(item["xyxy"], width, height)
        if not reasons:
            candidates.append((bottom - 0.5 * offset, index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def crosswalk_diagnostics(candidates, crosswalks, association, width, height, connection_confidence=0.50):
    """연결 조건은 유지하면서 검출 후보와 조건 탈락 사유를 별도로 제공한다."""
    boxes = []
    qualified_index = 0
    used_index = association.get("crosswalk_index")
    eligible_count = 0
    selected_crosswalk_index = None
    for item in candidates:
        bottom, offset, reasons = crosswalk_position(item["xyxy"], width, height)
        qualified = item["confidence"] >= connection_confidence
        used = qualified and qualified_index == used_index
        if qualified:
            qualified_index += 1
        if not qualified:
            reasons.insert(0, "below_confidence")
        eligible_count += not reasons
        status = ("below_confidence" if not qualified else "position_rejected" if reasons
                  else "used" if used else "eligible")
        if used:
            selected_crosswalk_index = len(boxes)
        boxes.append({
            "class_id": item["class_id"], "class_name": "crosswalk",
            "confidence": item["confidence"],
            "xyxy": list(item["xyxy"]),
            "crosswalk_status": status, "exclusion_reasons": reasons,
            "bottom_ratio": bottom, "center_offset_ratio": offset,
            "connection_reason": association["reason"] if used else None,
        })
    detection_status = ("not_detected" if not candidates else "below_confidence" if not crosswalks
                        else "position_rejected" if not eligible_count else "eligible")
    connection_status = association["reason"] or association["status"]
    if connection_status == "no_unambiguous_near_crosswalk":
        connection_status = "ambiguous_crosswalks" if eligible_count else "not_attempted"
    return boxes, {
        "detection_status": detection_status, "connection_status": connection_status,
        "candidate_count": len(candidates), "qualified_count": len(crosswalks),
        "eligible_count": eligible_count, "selected_crosswalk_index": selected_crosswalk_index,
        "connection_confidence": connection_confidence,
    }


def associate(frame, signals, crosswalks, cv2, *, require_geometry=False):
    """현재 프레임에서 선택한 신호등 인덱스 또는 선택하지 못한 사유를 반환한다."""
    height, width = frame.shape[:2]
    decision = {"status": "unknown", "reason": None, "crosswalk_index": None,
                "signal_index": None, "vanishing_point": None, "candidates": []}
    if not signals:
        decision["reason"] = "no_signal_detected"
        return decision
    if len(signals) == 1 and not require_geometry:
        decision["status"] = "single_signal"
        decision["reason"] = "crosswalk_relation_unverified"
        decision["signal_index"] = 0
        return decision
    crosswalk_index = choose_near_crosswalk(crosswalks, width, height)
    if crosswalk_index is None:
        decision["reason"] = "no_unambiguous_near_crosswalk"
        return decision
    decision["crosswalk_index"] = crosswalk_index
    crosswalk = crosswalks[crosswalk_index]
    vp = estimate_vanishing_point(frame, crosswalk["xyxy"], cv2)
    if vp is None:
        decision["reason"] = "vanishing_point_unavailable"
        return decision
    decision["vanishing_point"] = vp
    ranked = []
    for index, signal in enumerate(signals):
        box = signal["xyxy"]
        sx, sy = center(box)
        # 보행자 신호등은 횡단보도 끝의 옆쪽에 있을 수 있으므로,
        # 횡단보도보다 위에 있고 진행 방향에서 크게 벗어나지 않는지 확인한다.
        horizontal = abs(sx - vp[0]) / width
        if sy >= crosswalk["xyxy"][1] or horizontal > 0.22:
            continue
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        size_bonus = min(0.08, 0.08 * math.sqrt(area / (width * height)) / 0.04)
        score = 1 - horizontal / 0.22 + size_bonus
        ranked.append((score, index, horizontal, area))
        decision["candidates"].append({"signal_index": index, "score": round(score, 4),
                                       "horizontal_distance": round(horizontal, 4),
                                       "size_bonus": round(size_bonus, 4)})
    ranked.sort(reverse=True)
    if not ranked:
        decision["reason"] = "no_signal_in_crossing_direction"
    elif len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.12:
        first, second = ranked[:2]
        # 방향이 비슷한 후보는 면적 차이가 충분할 때 크기로 구분할 수 있다.
        # 방향 차이가 뚜렷하면 크기만으로 선택을 뒤집지 않는다.
        if abs(first[2] - second[2]) <= 0.04 and min(first[3], second[3]) > 0 \
                and max(first[3], second[3]) / min(first[3], second[3]) >= 2:
            decision["status"] = "candidate"
            decision["signal_index"] = first[1] if first[3] > second[3] else second[1]
        else:
            decision["reason"] = "ambiguous_signals"
    else:
        decision["status"] = "candidate"
        decision["signal_index"] = ranked[0][1]
    return decision


class TemporalSelector:
    """BoT-SORT ID로 대상을 유지하고 횡단보도 연결의 연속성을 확인한다."""

    TRACK_MAX_GAP_MS = 1000

    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.target_origin = None
        self.target_id = None
        self.target_requires_crosswalk = False
        self.previous_context = None
        self.previous_shape = None
        self.previous_gray = None
        self._clear_pending()

    def _clear_pending(self):
        self.last_track_id = None
        self.last_crosswalk = None
        self.streak = 0

    def _clear_target(self):
        self.target_origin = None
        self.target_id = None
        self.target_requires_crosswalk = False

    def _acquire_target(self, decision, signals, origin):
        self.target_origin = origin
        self.target_id = signals[decision["signal_index"]]["track_id"]
        self.target_requires_crosswalk = False
        decision.update(selection_origin=self.target_origin, track_id=self.target_id)

    def _match_target(self, signals):
        matches = [i for i, signal in enumerate(signals)
                   if signal.get("track_id") == self.target_id]
        if len(matches) == 1:
            return matches[0], {"reason": "matched", "tracker": "botsort"}
        return None, {"reason": "ambiguous_match" if matches else "target_missing",
                      "tracker": "botsort"}

    def select(self, frame, signals, crosswalks, cv2, context):
        """횡단보도로 확정한 대상은 유지하고, 임시 대상은 복수 검출 때 연결을 확인한다.

        신호등 하나는 바로 선택하고, 여러 개는 횡단보도 연결을 연속 확인한다.
        미검출 대상을 시간 기준으로 보관하거나 과거 박스·색상을 출력하지 않는다.
        """
        previous = self.previous_context
        continuous = previous is None or (
            context.frame_id == previous.frame_id + 1
            and 0 <= context.captured_at_ms - previous.captured_at_ms <= self.TRACK_MAX_GAP_MS
            and frame.shape[:2] == self.previous_shape
        )
        tracking = {"reason": "no_previous_target" if continuous else "discontinuous_frames"}
        tracking["tracker"] = "botsort"
        if not continuous:
            self._clear_target()
            self._clear_pending()
            self.previous_gray = None
        gray = motion_gray(frame, cv2)
        motion, motion_diagnostic = estimate_camera_motion(
            self.previous_gray, gray, frame.shape[:2], cv2,
        ) if self.last_crosswalk is not None else (
            None, {"reason": "no_previous_candidate"}
        )
        self.previous_gray = gray
        self.last_crosswalk = transform_box(self.last_crosswalk, motion)
        self.previous_context = context
        self.previous_shape = frame.shape[:2]

        if self.target_id is not None:
            index, tracking = self._match_target(signals)
            tracking.update(previous_frame_id=previous.frame_id, previous_track_id=self.target_id)
            tracking["camera_motion"] = motion_diagnostic
            if index is not None:
                if self.target_origin == "single_signal":
                    self.target_requires_crosswalk |= len(signals) > 1
                    if self.target_requires_crosswalk:
                        # 복수 검출 이후에는 하나만 남아도 시작한 연결 확인을 끝낸다.
                        # 확인 중에는 임시 대상의 색상을 안내하지 않는다.
                        decision = self.update(
                            associate(frame, signals, crosswalks, cv2, require_geometry=True),
                            signals, crosswalks,
                        )
                        decision["tracking"] = tracking
                        if decision["signal_index"] is not None:
                            if decision["signal_index"] == index:
                                self.target_origin = "crosswalk_matched"
                                self.target_requires_crosswalk = False
                                decision.update(selection_origin=self.target_origin,
                                                track_id=self.target_id)
                            else:
                                self._acquire_target(decision, signals, "crosswalk_matched")
                            self._clear_pending()
                        return decision
                self._clear_pending()
                return {"status": "tracked", "reason": "previous_target_retained",
                        "signal_index": index,
                        "crosswalk_index": None,
                        "selection_origin": self.target_origin, "track_id": self.target_id,
                        "tracking": tracking}
            self._clear_pending()
            self._clear_target()

        decision = associate(frame, signals, crosswalks, cv2)
        tracking["camera_motion"] = motion_diagnostic
        single = decision["status"] == "single_signal"
        decision = self.update(decision, signals, crosswalks)
        decision["tracking"] = tracking
        if decision["signal_index"] is not None:
            self._acquire_target(decision, signals, "single_signal" if single else "crosswalk_matched")
        return decision

    def update(self, decision, signals, crosswalks):
        index = decision["signal_index"]
        if index is not None and signals[index].get("track_id") is None:
            self._clear_pending()
            decision.update(status="unknown", reason="waiting_for_tracking",
                            candidate_signal_index=index, signal_index=None)
            return decision
        if decision["status"] == "single_signal":
            self._clear_pending()
            return decision
        if decision["status"] != "candidate" or index is None:
            self._clear_pending()
            return decision
        crosswalk_box = crosswalks[decision["crosswalk_index"]]["xyxy"]
        consistent = (
            self.last_track_id is not None and self.last_crosswalk is not None
            and signals[index]["track_id"] == self.last_track_id
            and box_iou(crosswalk_box, self.last_crosswalk) >= 0.3
        )
        self.streak = self.streak + 1 if consistent else 1
        self.last_track_id = signals[index].get("track_id")
        self.last_crosswalk = crosswalk_box
        decision["stable_frames"] = self.streak
        if self.streak < self.required_frames:
            decision["status"] = "unknown"
            decision["reason"] = "waiting_for_temporal_consistency"
            decision["candidate_signal_index"] = index
            decision["signal_index"] = None
        else:
            decision["status"] = "matched"
        return decision
