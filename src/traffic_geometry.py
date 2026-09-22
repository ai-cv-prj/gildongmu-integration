"""반복되는 횡단보도 도색 줄무늬의 경계로 진행 방향을 추정한다.

검출 박스는 탐색 범위만 제한한다. 각 경계선을 추정하려면 잘리지 않은 줄무늬
끝점이 최소 3개 필요하다. 근거가 부족하거나 여러 방향이 비슷하게 나타나면
방향을 미확정으로 남긴다. 영상 특징만으로 사용자가 건너려는 횡단보도를
확정할 수는 없다.
"""
import numpy as np


def stripe_candidates(frame, box, cv2):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 80 or y2 - y1 < 80:
        return None, {}
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, -8,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(100, roi.shape[0] * roi.shape[1] * 0.001):
            continue
        rectangle = cv2.minAreaRect(contour)
        rw, rh = rectangle[1]
        if (min(rw, rh) < 5 or max(rw, rh) / min(rw, rh) < 2.5
                or area / max(1, rw * rh) < 0.55):
            continue
        corners = cv2.boxPoints(rectangle)
        edges = np.roll(corners, -1, axis=0) - corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        if abs(edge[1]) > abs(edge[0]) * 0.7:
            continue
        # 실제 윤곽선 끝점을 사용해 화면에서 잘린 줄무늬를 바깥으로 연장하지 않는다.
        points = contour.reshape(-1, 2)
        axis = edge / np.linalg.norm(edge)
        if axis[0] < 0:
            axis = -axis
        projection = points @ axis
        low, high = projection.min(), projection.max()
        band = max(2, (high - low) * 0.025)
        left = points[projection <= low + band].mean(axis=0) + [x1, y1]
        right = points[projection >= high - band].mean(axis=0) + [x1, y1]
        bars.append({
            'left': left.tolist(), 'right': right.tolist(), 'area': area,
            'clipped_left': bool(points[:, 0].min() <= 2),
            'clipped_right': bool(points[:, 0].max() >= roi.shape[1] - 3),
        })
    # 무늬가 많은 노면에서도 경계선 조합을 비교하는 연산량이 과도해지지 않도록 제한한다.
    bars = sorted(bars, key=lambda bar: bar['area'], reverse=True)[:48]
    return None, {'mask': mask, 'bars': bars, 'roi': [x1, y1, x2, y2]}


def fit_rails(bars, width, roi_height):
    rails = []
    for side in ('left', 'right'):
        points = np.array([bar[side] for bar in bars if not bar['clipped_' + side]],
                          dtype=float).reshape(-1, 2)
        while len(points) >= 3:
            best = None
            for i, a in enumerate(points):
                for b in points[i + 1:]:
                    if abs(b[1] - a[1]) < max(40, 0.1 * roi_height):
                        continue
                    slope = (b[0] - a[0]) / (b[1] - a[1])
                    intercept = a[0] - slope * a[1]
                    if abs(slope) > 2:
                        continue
                    mask = abs(points[:, 0] - slope * points[:, 1] - intercept) <= max(3, 0.012 * width)
                    count = int(mask.sum())
                    span = float(np.ptp(points[mask, 1]))
                    if count < 3 or span < max(40, 0.1 * roi_height):
                        continue
                    score = count + span / roi_height
                    if best is None or score > best[0]:
                        best = score, mask
            if best is None:
                break
            support = points[best[1]]
            slope, intercept = np.polyfit(support[:, 1], support[:, 0], 1)
            error = float(np.sqrt(np.mean((support[:, 0] - slope * support[:, 1] - intercept) ** 2)))
            rails.append({'slope': float(slope), 'intercept': float(intercept),
                          'points': support.tolist(), 'error': error, 'score': best[0]})
            points = points[~best[1]]
    return rails


def direction(bars, width, height, roi):
    rails = fit_rails(bars, width, roi[3] - roi[1])
    pairs = []
    for i, a in enumerate(rails):
        for j, b in enumerate(rails[i + 1:], i + 1):
            delta = a['slope'] - b['slope']
            if abs(delta) < 0.12:
                continue
            y = (b['intercept'] - a['intercept']) / delta
            x = a['slope'] * y + a['intercept']
            farthest_support = min(p[1] for rail in (a, b) for p in rail['points'])
            if not (roi[1] - 0.5 * height <= y <= min(roi[1] + 0.4 * (roi[3] - roi[1]), farthest_support - 15)
                    and roi[0] - 0.25 * width <= x <= roi[2] + 0.25 * width):
                continue
            # 거의 평행하거나 잡음이 많아 교점이 불안정한 경계선 조합은 제외한다.
            uncertainty = (a['error'] + b['error'] + 2) / abs(delta)
            if uncertainty > 0.10 * height:
                continue
            pairs.append((min(a['score'], b['score']), [float(x), float(y)], i, j))
    pairs.sort(reverse=True)
    if not pairs:
        return None, rails, pairs
    best = pairs[0]
    if any(pair[0] >= best[0] * 0.8 and abs(pair[1][0] - best[1][0]) > 0.08 * width
           for pair in pairs[1:]):
        return None, rails, pairs
    return best[1], rails, pairs


def estimate_stripe_direction(frame, box, cv2):
    _, diagnostic = stripe_candidates(frame, box, cv2)
    if not diagnostic:
        return None
    height, width = frame.shape[:2]
    point, _, _ = direction(diagnostic['bars'], width, height, diagnostic['roi'])
    return point
