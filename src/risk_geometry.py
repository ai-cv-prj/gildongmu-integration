"""Image-space path occupancy. Sidewalk labels supply context, never a hard gate."""
import cv2
import numpy as np

def rectangle(box):
    x1, y1, x2, y2 = box
    return np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)

def overlap(box, polygon):
    area = (box[2]-box[0]) * (box[3]-box[1])
    if area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(rectangle(box), np.asarray(polygon, np.float32))
    return float(np.clip(intersection / area, 0, 1))

def geometry(detection, shape, cfg, roi=None):
    height, width = shape[:2]
    box = np.asarray(detection["xyxy"], dtype=float) / [width, height, width, height]
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    box = np.clip(box, 0, 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    x1, y1, x2, y2 = map(float, box)
    strip_height = min((y2-y1)*cfg["footprint_height_ratio"], cfg["footprint_max_height"])
    left,right = x1,x2
    if detection["class_name"] in cfg["static_ground_classes"] and not cfg["full_static_footprint"]:
        center=(x1+x2)/2
        radius=max(.002,(x2-x1)*.25)
        left,right=max(0,center-radius),min(1,center+radius)
    strip = [left, y2-strip_height, right, y2]
    corridor = roi["corridor_polygon"] if roi else cfg["corridor_polygon"]
    immediate = roi["immediate_polygon"] if roi else cfg["immediate_polygon"]
    corridors = roi.get("corridor_polygons",[corridor]) if roi else [corridor]
    def gap(polygon):
        intersections=[]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            if abs(a[1]-b[1])<1e-9:
                if abs(y2-a[1])<1e-6: intersections.extend([a[0],b[0]])
            elif min(a[1],b[1])<=y2<=max(a[1],b[1]):
                intersections.append(a[0]+(y2-a[1])*(b[0]-a[0])/(b[1]-a[1]))
        return max(0,min(intersections)-right,left-max(intersections)) if intersections else 1.0
    top_y = roi.get("path_top_y",min(p[1] for p in corridor)) if roi else min(p[1] for p in corridor)
    close_y = max(cfg["side_near_y"],top_y+.20) if cfg["roi_ground_adapt_enabled"] else cfg["side_near_y"]
    margin = cfg["edge_margin_ratio"]
    edges = [name for name, yes in (
        ("left", x1 <= margin), ("right", x2 >= 1-margin),
        ("top", y1 <= margin), ("bottom", y2 >= 1-margin)) if yes]
    return {
        "box_norm": list(map(float, box)), "footprint": strip,
        "point": [(x1+x2)/2, y2], "height": y2-y1, "width": x2-x1,
        "close_candidate": bool(cfg["side_proximity_enabled"] and y2>=close_y
            and (y2-y1)>=cfg["side_min_height"] and (x2-x1)>=cfg["side_min_width"]),
        "side_proximity": bool(cfg["side_proximity_enabled"] and y2>=close_y
            and max(overlap(strip,poly) for poly in corridors)<cfg["overlap_threshold"]
            and (y2-y1)>=cfg["side_min_height"] and (x2-x1)>=cfg["side_min_width"]),
        "side_direction": "left" if (x1+x2)/2<.5 else "right",
        "corridor_overlap": max(overlap(strip, poly) for poly in corridors),
        "immediate_overlap": overlap(strip, immediate),
        "edge_contact": edges,
        "horizontal_path_gap": min([gap(poly) for poly in corridors]+[gap(immediate)]),
        "bottom_clipped": y2 >= 1-1/height,
        # One original-image pixel at the frame edge makes scale censored.
        "clipped": x1 <= 1/width or y1 <= 1/height or x2 >= 1-1/width or y2 >= 1-1/height,
    }

def sidewalk_context(item, class_map, label_ids, shape):
    if class_map is None or label_ids is None or class_map.shape != tuple(shape[:2]):
        return {"status": "unavailable", "walkable_fraction": None}
    height, width = shape[:2]
    x1, _, x2, y2 = item["box_norm"]
    # Sample the contact neighbourhood, including ground just below the box.
    left, right = max(0, int(x1*width)), min(width, max(int(x1*width)+1, int(np.ceil(x2*width))))
    top, bottom = max(0, int((y2-.02)*height)), min(height, int(np.ceil((y2+.02)*height)))
    patch = class_map[top:bottom, left:right]
    ids = [label_ids[name] for name in ("walkable", "crosswalk") if name in label_ids]
    if not patch.size or not ids:
        return {"status": "unavailable", "walkable_fraction": None}
    fraction = float(np.isin(patch, ids).mean())
    return {"status": "available", "walkable_fraction": fraction}


# 장애물 bbox 주변의 보행가능영역 검사
def surrounding_walkability(item, class_map, label_ids, shape, cfg):
    """
    차량 bbox의 보이는 왼쪽·오른쪽·아래쪽 영역에서 보행가능 비율을 계산한다.
    화면 밖으로 잘린 영역은 검사 대상에서 제외한다.
    """
    unavailable = {"status":"unavailable", "regions":{}, "all_non_walkable":True}
    if class_map is None or label_ids is None or class_map.shape != tuple(shape[:2]):
        return unavailable
    walkable_ids = [label_ids[name] for name in ("walkable", "crosswalk") if name in label_ids]
    if not walkable_ids:
        return unavailable

    height, width = shape[:2]
    x1, y1, x2, y2 = item["box_norm"]
    left, top = int(np.floor(x1*width)), int(np.floor(y1*height))
    right, bottom = int(np.ceil(x2*width)), int(np.ceil(y2*height))
    box_width, box_height = max(1,right-left), max(1,bottom-top)
    side_width = max(cfg["surrounding_min_region_pixels"],
                     min(round(box_width*cfg["surrounding_side_width_ratio"]),
                         round(width*cfg["surrounding_max_side_width_ratio"])))
    side_top = bottom-max(cfg["surrounding_min_region_pixels"],
                          round(box_height*cfg["surrounding_side_height_ratio"]))
    bottom_height = max(cfg["surrounding_min_region_pixels"],
                        min(round(box_height*cfg["surrounding_bottom_height_ratio"]),
                            round(height*cfg["surrounding_max_bottom_height_ratio"])))
    candidates = {
        "left": (left-side_width, side_top, left, bottom),
        "right": (right, side_top, right+side_width, bottom),
        "bottom": (left, bottom, right, bottom+bottom_height),
    }
    regions = {}
    for name, (rx1,ry1,rx2,ry2) in candidates.items():
        rx1,rx2=max(0,rx1),min(width,rx2)
        ry1,ry2=max(0,ry1),min(height,ry2)
        if rx2<=rx1 or ry2<=ry1:
            continue
        patch=class_map[ry1:ry2,rx1:rx2]
        if patch.size < cfg["surrounding_min_region_pixels"]:
            continue
        regions[name]={"walkable_fraction":float(np.isin(patch,walkable_ids).mean()),
                       "pixel_count":int(patch.size)}
    fractions=[region["walkable_fraction"] for region in regions.values()]
    return {"status":"available" if regions else "clipped",
            "regions":regions,
            "all_non_walkable":bool(not fractions or
                all(value < cfg["surrounding_walkable_threshold"] for value in fractions))}
