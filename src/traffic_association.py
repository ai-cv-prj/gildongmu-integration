"""gildongmu-test-app SESAC-73의 횡단보도 연결·현재 대상 추적·변경 로직.

원본: f5481e8 (판단 로직 ab2033b). 픽셀 좌표와 영상 프레임 문맥에 맞춰 이식.
"""
import math
from dataclasses import dataclass

from src.traffic_geometry import estimate_stripe_direction
from src.traffic_motion import estimate_camera_motion, motion_gray, transform_box


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    captured_at_ms: float


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
    # 초기 추적 기준. 검출 신뢰도 대신 연속 프레임의 위치와 크기를 비교한다.
    TRACK_MIN_IOU = 0.20
    TRACK_MAX_CENTER_DISTANCE = 0.50  # 이전 박스 대각선 길이에 대한 비율
    TRACK_MAX_SIZE_RATIO = 2.0
    TRACK_SCORE_MARGIN = 0.15
    TRACK_MAX_GAP_MS = 1000

    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.target_box = None
        self.target_origin = None
        self.target_id = None
        self.next_target_id = 1
        self.previous_context = None
        self.previous_shape = None
        self.previous_gray = None
        self.target_needs_revalidation = False
        self._clear_pending()

    def _clear_pending(self):
        self.last_box = None
        self.last_crosswalk = None
        self.streak = 0

    def _clear_target(self):
        self.target_box = None
        self.target_origin = None
        self.target_id = None
        self.target_needs_revalidation = False

    def _acquire_target(self, decision, signals, origin):
        self.target_box = list(signals[decision["signal_index"]]["xyxy"])
        self.target_origin = origin
        self.target_id = self.next_target_id
        self.next_target_id += 1
        self.target_needs_revalidation = False
        decision.update(selection_origin=self.target_origin, track_id=self.target_id)

    def _reconsider_target(self, decision, index, signals, crosswalks, tracking):
        """다른 연결 후보는 연속 확인하며, 연결 근거가 충돌하면 색상 출력을 보류한다.

        새 후보를 확인하는 동안 현재 검출된 기존 대상은 내부 상태에 유지한다.
        충돌이 발생한 뒤에는 방향 추정 실패나 단일 검출만으로 기존 색상을
        복구하지 않는다. 같은 연결을 설정된 횟수만큼 연속 확인해야 한다(기본 3회).
        """
        tracking["revalidation_status"] = decision["status"]
        tracking["revalidation_reason"] = decision["reason"]
        if decision["status"] == "candidate":
            challenger = decision["signal_index"]
            if challenger == index and not self.target_needs_revalidation:
                self._clear_pending()
                # 최초 선택 근거를 유지한다. 연결이 한 번 관측됐다는 이유만으로
                # 단일 신호등 선택을 횡단보도 연결 확인 완료로 바꾸지 않는다.
                return None
            self.target_needs_revalidation = True
            decision = self.update(decision, signals, crosswalks)
            switching = challenger != index
            tracking["target_change"] = {
                "state": "confirmed" if decision["signal_index"] is not None else "pending",
                "previous_track_id": self.target_id,
                "candidate_signal_index": challenger,
                "stable_frames": decision["stable_frames"],
                "switching": switching,
            }
            if decision["signal_index"] is None:
                decision["reason"] = ("waiting_for_target_switch" if switching
                                      else "waiting_for_target_revalidation")
            elif switching:
                self._acquire_target(decision, signals, "crosswalk_matched")
                decision["reason"] = "target_switched"
                self._clear_pending()
            else:
                self.target_needs_revalidation = False
                self.target_origin = "crosswalk_matched"
                decision.update(status="tracked", reason="target_revalidated",
                                track_id=self.target_id, selection_origin=self.target_origin)
                self._clear_pending()
        else:
            self._clear_pending()
            # 방향 추정 실패만으로 다른 대상이 맞다고 판단하지 않는다.
            # 다만 방향 근거가 명확히 충돌하면 색상 출력을 보류한다.
            conflict = decision["reason"] in {"ambiguous_signals", "no_signal_in_crossing_direction"}
            if not self.target_needs_revalidation and not conflict:
                return None
            self.target_needs_revalidation = True
            tracking["target_change"] = {"state": "blocked", "previous_track_id": self.target_id}
        decision["tracking"] = tracking
        return decision

    def _match_target(self, signals):
        box = self.target_box
        width, height = box[2] - box[0], box[3] - box[1]
        if min(width, height) <= 0:
            return None, {"reason": "target_missing"}
        ranked = []
        for index, signal in enumerate(signals):
            other = signal["xyxy"]
            ow, oh = other[2] - other[0], other[3] - other[1]
            if min(ow, oh) <= 0:
                continue
            size_ratio = max(width / ow, ow / width, height / oh, oh / height,
                             width * height / (ow * oh), ow * oh / (width * height))
            iou = box_iou(box, other)
            distance = math.dist(center(box), center(other)) / math.hypot(width, height)
            if (iou >= self.TRACK_MIN_IOU and distance <= self.TRACK_MAX_CENTER_DISTANCE
                    and size_ratio <= self.TRACK_MAX_SIZE_RATIO):
                ranked.append((iou - 0.25 * distance, index, iou, distance))
        ranked.sort(reverse=True)
        if not ranked:
            return None, {"reason": "target_missing"}
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.TRACK_SCORE_MARGIN:
            return None, {"reason": "ambiguous_match"}
        _, index, iou, distance = ranked[0]
        return index, {"reason": "matched", "iou": round(iou, 4),
                       "center_distance": round(distance, 4)}

    def select(self, frame, signals, crosswalks, cv2, context):
        """현재 보이는 대상을 추적하고 횡단보도 방향으로 다른 후보도 다시 확인한다.

        과거 이력만으로 박스나 색상을 출력하지 않는다. 대상이 사라지거나
        매칭이 모호하면 추적을 종료하고 프레임별 최초 선택 절차로 돌아간다.
        """
        previous = self.previous_context
        continuous = previous is None or (
            context.frame_id == previous.frame_id + 1
            and 0 <= context.captured_at_ms - previous.captured_at_ms <= self.TRACK_MAX_GAP_MS
            and frame.shape[:2] == self.previous_shape
        )
        tracking = {"reason": "no_previous_target" if continuous else "discontinuous_frames"}
        if not continuous:
            self._clear_target()
            self._clear_pending()
            self.previous_gray = None
        gray = motion_gray(frame, cv2)
        motion, motion_diagnostic = estimate_camera_motion(
            self.previous_gray, gray, frame.shape[:2], cv2,
        ) if self.target_box is not None or self.last_box is not None else (
            None, {"reason": "no_previous_candidate"}
        )
        self.previous_gray = gray
        self.target_box = transform_box(self.target_box, motion)
        self.last_box = transform_box(self.last_box, motion)
        self.last_crosswalk = transform_box(self.last_crosswalk, motion)
        self.previous_context = context
        self.previous_shape = frame.shape[:2]

        if self.target_box is not None:
            index, tracking = self._match_target(signals)
            tracking.update(previous_frame_id=previous.frame_id, previous_track_id=self.target_id)
            tracking["camera_motion"] = motion_diagnostic
            if index is not None:
                self.target_box = list(signals[index]["xyxy"])
                geometry = None
                if len(signals) > 1 or self.target_needs_revalidation:
                    geometry = associate(frame, signals, crosswalks, cv2, require_geometry=True)
                    reconsidered = self._reconsider_target(geometry, index, signals, crosswalks, tracking)
                    if reconsidered is not None:
                        return reconsidered
                self._clear_pending()
                return {"status": "tracked", "reason": "previous_target_retained",
                        "signal_index": index,
                        "crosswalk_index": geometry.get("crosswalk_index") if geometry else None,
                        "selection_origin": self.target_origin, "track_id": self.target_id,
                        "tracking": tracking}
            self._clear_target()
            self._clear_pending()

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
        if decision["status"] == "single_signal":
            self._clear_pending()
            return decision
        if decision["status"] != "candidate" or index is None:
            self._clear_pending()
            return decision
        box = signals[index]["xyxy"]
        crosswalk_box = crosswalks[decision["crosswalk_index"]]["xyxy"]
        consistent = (
            self.last_box is not None and self.last_crosswalk is not None
            and box_iou(box, self.last_box) >= 0.3
            and box_iou(crosswalk_box, self.last_crosswalk) >= 0.3
        )
        self.streak = self.streak + 1 if consistent else 1
        self.last_box = box
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
