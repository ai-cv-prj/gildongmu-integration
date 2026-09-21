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
    """대상 신호의 색상, 횡단보도 박스, 소실점과 선택 실패 사유를 표시한다."""
    result = frame.copy()
    height, width = frame.shape[:2]
    association = prediction["association"]

    def point(x, y):
        return (max(0, min(width - 1, round(x))), max(0, min(height - 1, round(y))))

    for index, crossing in enumerate(prediction["crosswalks"]):
        x1, y1, x2, y2 = crossing["xyxy"]
        selected = index == association.get("crosswalk_index")
        color = (255, 255, 0) if selected else LABEL_COLORS["crosswalk"]
        cv2.rectangle(result, point(x1, y1), point(x2, y2), color, 3 if selected else 1)
        label = "selected crosswalk" if selected else "crosswalk"
        cv2.putText(result, label, point(x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    state = prediction["signal_state"]
    colors = {"red": (0, 0, 255), "green": (0, 255, 0), "unknown": (160, 160, 160)}
    color = colors[state]
    vp = association.get("vanishing_point")
    if vp is not None and 0 <= vp[0] < width and 0 <= vp[1] < height:
        cv2.drawMarker(result, point(*vp), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
    for signal in prediction["detections"]:
        x1, y1, x2, y2 = signal["xyxy"]
        cv2.rectangle(result, point(x1, y1), point(x2, y2), color, 3)
        score = signal["color_confidence"]
        label = f"signal: {state}" + (f" {score:.2f}" if score is not None else "")
        cv2.putText(result, label, point(x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if vp is not None and 0 <= vp[0] < width and 0 <= vp[1] < height:
            cv2.line(result, point((x1 + x2) / 2, (y1 + y2) / 2), point(*vp), color, 1)
    reason = association.get("reason") or association["status"]
    text = f"Signal: {state} | candidates: {prediction['detected_signal_count']} | {reason}"
    cv2.putText(result, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
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
