"""Additional risk overlay; existing class colors and traffic drawing are untouched."""
import cv2
import numpy as np

COLORS = {"monitor": (180,180,180), "caution": (0,200,255), "danger": (0,0,255)}

def draw_risk(frame, prediction, config):
    config = dict(config)
    for key in ("corridor_polygon", "immediate_polygon"):
        if prediction.get("roi"):
            config[key] = prediction["roi"][key]
    if config.get("review_overlay", False):
        return draw_review(frame, prediction, config)
    result = draw_scene_regions(frame, prediction)
    h, w = frame.shape[:2]
    if config["draw_roi"]:
        for key, color in (("corridor_polygon",(255,190,0)), ("immediate_polygon",(0,120,255))):
            points = np.rint(np.asarray(config[key])*[w-1,h-1]).astype(np.int32)
            cv2.polylines(result, [points], True, color, 2, cv2.LINE_AA)
    counts = {"monitor":0,"caution":0,"danger":0}
    for item in prediction["detections"]:
        level = item.get("alert_level", item["risk_level"])
        if level is None:
            continue
        if level == "monitor" or item.get("warning_primary", True):
            counts[level] += 1
        x1,y1,x2,y2 = item["xyxy"]
        if not np.isfinite([x1,y1,x2,y2]).all():
            continue
        if level != "monitor" and not item.get("warning_primary", True):
            continue
        identity = f'#{item["track_id"]}' if item["track_id"] is not None else "no-ID"
        grouped = f' x{item.get("warning_group_size",1)}' if item.get("warning_group_size",1)>1 else ""
        text = f"{identity} {level.upper()}{grouped}"
        if item["motion"] and item["motion"]["time_to_corridor_s"] is not None:
            text += " entering"
        px, py = max(0,min(w-1,round(x1))), max(15,min(h-40,round(y2)+16))
        cv2.putText(result,text,(px,py),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(result,text,(px,py),cv2.FONT_HERSHEY_SIMPLEX,.5,COLORS[level],1,cv2.LINE_AA)
    status = f'Risk: danger={counts["danger"]} caution={counts["caution"]} | tracker={prediction["tracker_status"]} | EXPERIMENTAL ROI'
    cv2.putText(result,status,(8,max(16,h-12)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(result,status,(8,max(16,h-12)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    draw_scene_status(result,prediction,(8,max(20,h-34)),.48)
    return result


def draw_review(frame, prediction, config):
    """Large, temporary inspection overlay. Red/amber outlines encode risk, not class."""
    result = draw_scene_regions(frame, prediction)
    h, w = frame.shape[:2]
    scale = max(.45, w/1080)
    thick = max(3, round(7*scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    def text(message, position, size, color, weight=2):
        cv2.putText(result, message, position, font, size, (0,0,0), weight+3, cv2.LINE_AA)
        cv2.putText(result, message, position, font, size, color, weight, cv2.LINE_AA)

    if config["draw_roi"]:
        for key, color, alpha, label in (
            ("corridor_polygon",(255,220,20),.09,"PATH ROI"),
            ("immediate_polygon",(220,40,250),.14,"NEAR ROI")):
            points = np.rint(np.asarray(config[key])*[w-1,h-1]).astype(np.int32)
            overlay = result.copy()
            cv2.fillPoly(overlay,[points],color)
            result = cv2.addWeighted(overlay,alpha,result,1-alpha,0)
            cv2.polylines(result,[points],True,(0,0,0),thick+4,cv2.LINE_AA)
            cv2.polylines(result,[points],True,color,thick,cv2.LINE_AA)
            left, top = points[0]
            text(label,(max(8,int(left)),max(30,int(top)-14)),.9*scale,color,max(1,round(2*scale)))
    counts = {"monitor":0,"caution":0,"danger":0}
    occupied = []
    cards = []
    levels = {"monitor":0,"caution":1,"danger":2}
    items = sorted(prediction["detections"],key=lambda x:levels.get(x.get("alert_level",x["risk_level"]),-1),reverse=True)
    for item in items:
        level = item.get("alert_level",item["risk_level"])
        if level is None:
            continue
        if level == "monitor" or item.get("warning_primary", True):
            counts[level] += 1
        box = item["xyxy"]
        if not np.isfinite(box).all():
            continue
        x1,y1,x2,y2 = map(int,np.rint(box))
        x1,x2 = max(0,min(w-1,x1)),max(0,min(w-1,x2))
        y1,y2 = max(0,min(h-1,y1)),max(0,min(h-1,y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = COLORS[level] if level != "monitor" else (200,200,200)
        stroke = thick if level != "monitor" else max(1,thick//3)
        if level != "monitor":
            cv2.rectangle(result,(x1,y1),(x2,y2),(0,0,0),stroke+4)
        cv2.rectangle(result,(x1,y1),(x2,y2),color,stroke)
        if level != "monitor" and not item.get("warning_primary", True):
            continue
        identity = f'#{item["track_id"]}' if item["track_id"] is not None else "NEW"
        display_level = "UNKNOWN" if item.get("alert_status") == "uncertain" and level == "monitor" else level.upper()
        grouped = f' x{item.get("warning_group_size",1)}' if item.get("warning_group_size",1)>1 else ""
        first = f'{display_level} | {item["class_name"]} {identity}{grouped}'
        motion = item.get("motion") or {}
        detail = ""
        if level != "monitor":
            ttc = motion.get("ttc_scale_s")
            expansion = motion.get("relative_expansion_per_s")
            prox=(item.get("proximity") or {}).get("band","unknown").upper()
            reasons=item.get("reasons",[])
            prox = ("SIDE CLOSE" if "side_close_candidate" in reasons else
                    "CLOSE CANDIDATE" if "large_static_candidate" in reasons else
                    "NEAR CONTACT" if "static_near_contact" in reasons else
                    "NEAR PATH" if "near_path_occupied" in reasons else "PATH")
            detail = prox + " | " + (f"TTC~{ttc:.1f}s" if ttc is not None else "TTC --")
            if expansion is not None:
                detail += f" | size {expansion*100:+.0f}%/s"
            if item.get("hold_reason"):
                detail += " | HOLD"
            if motion.get("time_to_corridor_s") is not None:
                detail += f' | entry~{motion["time_to_corridor_s"]:.1f}s'
        size = (.88 if level != "monitor" else .58)*scale
        weight = max(1,round((2 if level != "monitor" else 1)*scale))
        lines = [first] + ([detail] if detail else [])
        # Fit every line to the image width; do not truncate the risk word.
        while max(cv2.getTextSize(line,font,size,weight)[0][0] for line in lines) > w-20 and size > .3:
            size *= .9
        line_height = max(18,round(34*scale))
        label_height = line_height*len(lines)+10
        label_width = min(w-12,max(cv2.getTextSize(line,font,size,weight)[0][0] for line in lines)+14)
        left = min(x1,w-label_width-4)
        top = max(4,y1-label_height-8)
        # Avoid stacking labels over already annotated high-priority objects.
        for candidate in (top,y1+8,y2+8):
            candidate=max(4,min(h-label_height-155,candidate))
            rect=(left,candidate,left+label_width,candidate+label_height)
            if not any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in occupied):
                top=candidate
                break
        occupied.append((left,top,left+label_width,top+label_height))
        cards.append((left,top,label_width,label_height,color,lines,line_height,size,weight))
    # Draw labels after all boxes, with the highest-risk cards last.
    for left,top,label_width,label_height,color,lines,line_height,size,weight in reversed(cards):
        cv2.rectangle(result,(left,top),(left+label_width,top+label_height),(15,15,15),-1)
        cv2.rectangle(result,(left,top),(left+label_width,top+label_height),color,max(1,round(2*scale)))
        for i,line in enumerate(lines):
            cv2.putText(result,line,(left+7,top+(i+1)*line_height),font,size,color,weight,cv2.LINE_AA)
    panel_height=max(110,round(202*scale))
    cv2.rectangle(result,(0,h-panel_height),(w-1,h-1),(16,16,16),-1)
    scene_warnings=int(bool((prediction.get("surface") or {}).get("alert_level"))) + int(bool(prediction.get("advisories")))
    title=f'DANGER {counts["danger"]}   CAUTION {counts["caution"]+scene_warnings}   MONITOR/UNK {counts["monitor"]}'
    text(title,(16,h-panel_height+round(34*scale)),.95*scale,(255,255,255),max(1,round(2*scale)))
    mode="ON (experimental)" if config["ttc_alerts"] else "LOG ONLY"
    roi=prediction.get("roi") or {}
    text(f'ROI: {roi.get("source","fixed")} | direction confidence: {roi.get("direction_confidence",0):.2f}',
         (16,h-panel_height+round(66*scale)),.66*scale,(255,255,255),max(1,round(scale)))
    text("TTC risk: "+mode+" | metric distance: unavailable",
         (16,h-panel_height+round(98*scale)),.66*scale,(255,255,255),max(1,round(scale)))
    text("CYAN: path   MAGENTA: near   AMBER: path check / side watch",
         (16,h-panel_height+round(130*scale)),.60*scale,(255,255,255),max(1,round(scale)))
    surface=prediction.get("surface") or {}
    state=("CAUTION" if surface.get("alert_level") else surface.get("status","off")).upper()
    if surface.get("status")=="uncertain":
        state="CAUTION / UNCERTAIN" if surface.get("alert_level") else "UNCERTAIN"
    fraction=surface.get("path_nonwalkable_fraction")
    line=f'PATH CHECK: {state}' + (f' | non-walkable {fraction:.0%}' if fraction is not None else "")
    text(line,(16,h-panel_height+round(160*scale)),.65*scale,(0,210,255),max(1,round(scale)))
    draw_scene_status(result,prediction,(16,h-panel_height+round(190*scale)),.62*scale)
    return result


def draw_scene_regions(frame, prediction):
    """Draw observed semantic regions; never invent a detection box for lost tracks."""
    result=frame.copy()
    h,w=result.shape[:2]
    surface=prediction.get("surface") or {}
    if surface.get("alert_level"):
        for region in surface.get("regions",[]):
            points=np.rint(np.asarray(region["polygon"])*[w-1,h-1]).astype(np.int32)
            tint=result.copy()
            cv2.fillPoly(tint,[points],(0,190,255))
            result=cv2.addWeighted(tint,.20,result,.80,0)
            cv2.polylines(result,[points],True,(0,210,255),max(2,round(w/270)),cv2.LINE_AA)
    return result


def draw_scene_status(result, prediction, position, scale):
    messages=[]
    surface=prediction.get("surface") or {}
    if surface.get("alert_level"):
        messages.append("PATH: check surroundings" if surface.get("status")=="uncertain" else "PATH: non-walkable area")
    directions=sorted({a["direction"].upper() for a in prediction.get("advisories",[])})
    if directions:
        messages.append("/".join(directions)+": nearby object out of view")
    if messages:
        text=" | ".join(messages)
        while cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]>result.shape[1]-24 and scale>.3:
            scale*=.9
        cv2.putText(result,text,position,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(result,text,position,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,215,255),1,cv2.LINE_AA)
