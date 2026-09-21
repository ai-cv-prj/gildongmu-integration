"""Group only highly overlapping warning presentation; preserve every detection."""
import math

LEVEL = {"monitor":0, "caution":1, "danger":2}


def _overlap(a, b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-intersection
    return ((intersection/union if union>0 else 0.0),
            (intersection/min(area_a,area_b) if min(area_a,area_b)>0 else 0.0),
            (max(area_a,area_b)/min(area_a,area_b) if min(area_a,area_b)>0 else float("inf")))


def _same_warning(a, b, cfg):
    if a["class_id"] != b["class_id"]:
        return False
    boxes=(a.get("xyxy"),b.get("xyxy"))
    if any(box is None or len(box)!=4 or not all(math.isfinite(v) for v in box)
           for box in boxes):
        return False
    iou,containment,area_ratio=_overlap(*boxes)
    if area_ratio > cfg["warning_group_max_area_ratio"]:
        return False
    return (iou >= cfg["warning_group_iou"] or
            containment >= cfg["warning_group_containment"])


def group_warnings(detections, cfg):
    """Return notification groups without removing or lowering any detection."""
    if not cfg["warning_grouping_enabled"]:
        return []
    warnings=[item for item in detections
              if LEVEL.get(item.get("alert_level",item.get("risk_level")),0)>0]
    # Highest urgency and confidence become the visible representative.
    warnings.sort(key=lambda item:(LEVEL[item.get("alert_level",item["risk_level"])],
                  item.get("confidence",0),-item["detection_index"]),reverse=True)
    clusters=[]
    for item in warnings:
        # Complete-link matching prevents a chain of adjacent objects becoming one group.
        cluster=next((members for members in clusters
                      if all(_same_warning(item,member,cfg) for member in members)),None)
        if cluster is None:
            clusters.append([item])
        else:
            cluster.append(item)
    output=[]
    for group_id,members in enumerate(clusters,1):
        primary=max(members,key=lambda item:(LEVEL[item.get("alert_level",item["risk_level"])],
                    item.get("confidence",0),-item["detection_index"]))
        level=max((item.get("alert_level",item["risk_level"]) for item in members),
                  key=LEVEL.get)
        reasons=[]
        for item in members:
            for reason in item.get("reasons",[]):
                if reason not in reasons:
                    reasons.append(reason)
        for item in members:
            item.update(warning_group_id=group_id,
                        warning_group_size=len(members),
                        warning_primary=item is primary)
        output.append({"group_id":group_id,"level":level,
            "class_id":primary["class_id"],"class_name":primary["class_name"],
            "primary_detection_index":primary["detection_index"],
            "member_detection_indices":[item["detection_index"] for item in members],
            "member_event_ids":[item.get("event_id") for item in members],
            "track_ids":[item.get("track_id") for item in members],
            "reasons":reasons,"size":len(members)})
    return output
