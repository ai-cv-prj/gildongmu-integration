"""Conservative image-space association of a near crosswalk and pedestrian signal."""

import math


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
    """Intersect two differently sloped, near-vertical crosswalk edge lines.

    A detector rectangle does not encode a vanishing point. Return None unless
    image edges provide a geometrically consistent estimate.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 80 or y2 - y1 < 80:
        return None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    segments = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=30,
        minLineLength=max(25, int((y2 - y1) * 0.12)), maxLineGap=15,
    )
    if segments is None:
        return None
    lines = []
    for (ax, ay, bx, by) in segments[:, 0]:
        dx, dy = bx - ax, by - ay
        if abs(dy) < 25 or abs(dx) > abs(dy) * 0.9:
            continue
        if ay > by:
            ax, ay, bx, by = bx, by, ax, ay
            dx, dy = -dx, -dy
        slope = dx / dy
        lines.append((x1 + ax, y1 + ay, x1 + bx, y1 + by, slope))
    candidates = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if abs(a[4] - b[4]) < 0.12:
                continue
            # x = slope*y + intercept
            intercept_a = a[0] - a[4] * a[1]
            intercept_b = b[0] - b[4] * b[1]
            py = (intercept_b - intercept_a) / (a[4] - b[4])
            px = a[4] * py + intercept_a
            if (y1 - 0.5 * height <= py <= y1 + 0.4 * (y2 - y1)
                    and x1 - 0.25 * width <= px <= x2 + 0.25 * width):
                candidates.append((px, py))
    if len(candidates) < 3:
        return None
    xs = sorted(point[0] for point in candidates)
    ys = sorted(point[1] for point in candidates)
    mid = len(candidates) // 2
    px, py = xs[mid], ys[mid]
    inliers = sum(math.hypot(x - px, y - py) < 0.08 * width for x, y in candidates)
    if inliers < 3 or inliers / len(candidates) < 0.6:
        return None
    return [float(px), float(py)]


def choose_near_crosswalk(crosswalks, width, height):
    """Prefer the crossing whose near edge is low and centered in the image."""
    candidates = []
    for index, item in enumerate(crosswalks):
        box = item["xyxy"]
        bottom = box[3] / height
        offset = abs(center(box)[0] - width / 2) / width
        if bottom >= 0.55 and offset <= 0.30:
            candidates.append((bottom - 0.5 * offset, index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def associate(frame, signals, crosswalks, cv2):
    """Return a frame decision with a signal index or an explicit unknown reason."""
    height, width = frame.shape[:2]
    decision = {"status": "unknown", "reason": None, "crosswalk_index": None,
                "signal_index": None, "vanishing_point": None, "candidates": []}
    if not signals:
        decision["reason"] = "no_signal_detected"
        return decision
    if len(signals) == 1:
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
        # Pedestrian signal heads may sit beside the far end, but should be
        # above the crossing and reasonably close to its forward direction.
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
        # Size may resolve a geometric tie, but cannot overturn a clear
        # directional difference. Require a substantial area difference.
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
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.last_box = None
        self.last_crosswalk = None
        self.streak = 0

    def update(self, decision, signals, crosswalks):
        index = decision["signal_index"]
        if decision["status"] == "single_signal":
            self.last_box = None
            self.last_crosswalk = None
            self.streak = 0
            return decision
        if decision["status"] != "candidate" or index is None:
            self.last_box = None
            self.last_crosswalk = None
            self.streak = 0
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
