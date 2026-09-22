"""
file_path: src/visualization.py

클래스 지도의 반투명 색상과 클래스별 고정 색상의 객체 박스를 표시한다.
보행가능은 초록색, 횡단보도는 핑크색, 보행불가는 원본을 유지한다.
"""

import cv2


# OpenCV BGR 색상
LABEL_COLORS = {
    "walkable": (0, 255, 0),
    "crosswalk": (180, 105, 255),
}


# YOLO 클래스별 고정 BGR 색상, 위험도와 무관
OBJECT_COLORS = {
    "person": (0, 165, 255),
    "bicycle": (255, 144, 30),
    "bus": (0, 215, 255),
    "car": (0, 69, 255),
    "handcart": (147, 20, 255),
    "cat": (250, 206, 135),
    "dog": (71, 99, 255),
    "motorcycle": (226, 43, 138),
    "kick_scooter": (208, 224, 64),
    "stroller": (238, 104, 123),
    "truck": (60, 20, 220),
    "wheelchair": (113, 179, 60),
    "bird": (255, 191, 0),
    "barricade": (0, 140, 255),
    "bench": (32, 165, 218),
    "bollard": (214, 112, 218),
    "chair": (193, 182, 255),
    "fire_hydrant": (114, 128, 250),
    "kiosk": (237, 149, 100),
    "parking_meter": (238, 130, 238),
    "pole": (170, 178, 32),
    "potted_plant": (50, 205, 154),
    "utility_box": (170, 205, 102),
    "transit_stop": (0, 255, 255),
    "table": (96, 164, 244),
    "traffic_light": (0, 255, 127),
    "traffic_sign": (255, 0, 255),
    "tree_trunk": (30, 105, 210),
    "movable_obstacle": (128, 128, 240),
    "suitcase": (45, 82, 160),
    "skateboard": (255, 112, 132),
    "trash_bin": (196, 196, 0),
}
UNKNOWN_OBJECT_COLOR = (200, 200, 200)  # 미등록 클래스 표시용 회색


# 클래스별 반투명 색상 표시
def overlay_segmentation(frame, class_map, label_ids, alpha=0.55):
    """원본을 변경하지 않고 클래스별 색상을 합성한 프레임을 반환한다."""
    if class_map.shape != frame.shape[:2]:
        raise ValueError("클래스 지도와 원본 프레임의 높이·너비가 다릅니다.")
    if not 0 <= alpha <= 1:
        raise ValueError("overlay_alpha는 0부터 1 사이여야 합니다.")
    result = frame.copy()
    for label_name, color in LABEL_COLORS.items():
        mask = class_map == label_ids[label_name]
        overlay = frame.copy()
        overlay[mask] = color
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        result[mask] = blended[mask]
    return result


def draw_traffic(frame, prediction):
    """전체 검출·확인 중 후보·안내 대상과 횡단보도 실패 사유를 구분한다."""
    result = frame.copy()
    height, width = frame.shape[:2]
    association = prediction["association"]
    colors = {"red": (0, 0, 255), "green": (0, 255, 0), "unknown": (160, 160, 160)}
    blue, yellow, purple, cyan = (255, 140, 79), (32, 176, 255), (255, 123, 181), (201, 201, 34)
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels, occupied = [], []

    def point(x, y):
        return (max(0, min(width - 1, round(x))), max(0, min(height - 1, round(y))))

    def fit_text(text, scale=0.5):
        text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
        return min(scale, scale * max(1, width - 16) / max(1, text_width))

    def add_box(box, label, color, thickness):
        x1, y1, x2, y2 = box
        cv2.rectangle(result, point(x1, y1), point(x2, y2), color, thickness)
        scale = fit_text(label)
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        tw, th = tw + 8, th + baseline + 6
        lx = max(0, min(round(x1), width - tw))
        preferred = round(y1) - th - 3
        candidates = [preferred, round(y2) + 3]
        candidates += [y + h + 2 for _, y, _, h in occupied]
        ly = next((y for y in candidates if y >= 30 and y + th < height - 50
                   and all(lx + tw <= x or lx >= x + w or y + th <= oy or y >= oy + h
                           for x, oy, w, h in occupied)),
                  max(0, min(preferred, height - th)))
        occupied.append((lx, ly, tw, th))
        labels.append((label, color, lx, ly, tw, th, scale, baseline))

    for crossing in prediction["crosswalks"]:
        status = crossing.get("crosswalk_status", "eligible")
        color = cyan if status == "used" else purple
        detail = {"used": "LINK USED", "eligible": "ELIGIBLE",
                  "below_confidence": "LOW CONF", "position_rejected": "POSITION REJECTED"}[status]
        reasons = crossing.get("exclusion_reasons", [])
        reason_text = ",".join(reason for reason in reasons if reason != "below_confidence")
        label = f"CROSSWALK {crossing['confidence'] * 100:.1f}% {detail}"
        if reason_text:
            label += f" ({reason_text})"
        add_box(crossing["xyxy"], label, color, 3 if status == "used" else 2)

    vp = association.get("vanishing_point")
    vp_visible = vp is not None and 0 <= vp[0] < width and 0 <= vp[1] < height
    if vp_visible:
        cv2.drawMarker(result, point(*vp), cyan, cv2.MARKER_CROSS, 16, 2)
    for signal in prediction["detections"]:
        selection = signal["selection_status"]
        if selection == "selected":
            color = colors[signal["signal_state"]]
            label = f"TARGET {signal['signal_state'].upper()}"
            if signal.get("color_confidence") is not None:
                label += f" color {signal['color_confidence'] * 100:.1f}%"
            if signal.get("track_id") is not None:
                label += f" #{signal['track_id']}"
        else:
            color = yellow if selection == "candidate" else blue
            label = f"{'CANDIDATE' if selection == 'candidate' else 'UNSELECTED'} det {signal['confidence'] * 100:.1f}%"
        add_box(signal["xyxy"], label, color, 4 if selection == "selected" else 2)
        if vp_visible and selection in {"selected", "candidate"}:
            x1, y1, x2, y2 = signal["xyxy"]
            cv2.line(result, point((x1 + x2) / 2, (y1 + y2) / 2), point(*vp), color, 1)

    # 라벨은 모든 박스 뒤에 그려 다른 테두리에 가려지지 않게 한다.
    for label, color, x, y, w, h, scale, baseline in labels:
        cv2.rectangle(result, point(x, y), point(x + w, y + h), color, -1)
        cv2.putText(result, label, point(x + 4, y + h - baseline - 3), font,
                    scale, (15, 15, 15), 1, cv2.LINE_AA)

    state = prediction["signal_state"]
    count = prediction["detected_signal_count"]
    selected = prediction.get("selected_detection_index")
    candidate = prediction.get("candidate_detection_index")
    detail = ("NO DETECTION" if not count else f"TARGET {state.upper()}" if selected is not None
              else "CONFIRMING / COLOR HELD" if candidate is not None else "NO TARGET / COLOR HELD")
    text = f"SIGNALS {count} | {detail}"
    cv2.rectangle(result, (0, 0), (width - 1, 28), (24, 24, 24), -1)
    cv2.putText(result, text, (8, 20), font, fit_text(text), colors[state], 1, cv2.LINE_AA)
    info = prediction.get("crosswalk_diagnostics", {})
    reason = info.get("connection_status") or association.get("reason") or association["status"]
    lines = [f"CROSSWALK {len(prediction['crosswalks'])} | {info.get('detection_status', 'unknown')}",
             f"LINK: {reason}"]
    cv2.rectangle(result, (0, max(0, height - 50)), (width - 1, height - 1), (24, 24, 24), -1)
    for i, line in enumerate(lines):
        cv2.putText(result, line, (8, max(12, height - 31 + i * 22)), font,
                    fit_text(line), (240, 240, 240), 1, cv2.LINE_AA)
    return result


# 클래스별 색상의 객체 박스 및 이름 표시
def draw_detections(frame, detections):
    """입력 이미지를 유지하며 클래스별 색상으로 박스·이름·신뢰도를 표시한다."""
    result = frame.copy()
    height, width = frame.shape[:2]
    for detection in detections:
        color = OBJECT_COLORS.get(detection["class_name"], UNKNOWN_OBJECT_COLOR)
        x1, y1, x2, y2 = (int(round(value)) for value in detection["xyxy"])
        x1, x2 = (max(0, min(width - 1, value)) for value in (x1, x2))
        y1, y2 = (max(0, min(height - 1, value)) for value in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(
            result, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )
    return result
