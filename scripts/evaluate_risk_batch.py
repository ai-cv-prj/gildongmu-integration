"""Evaluate every frame of MP4 samples, preserving baselines and recording provenance."""
import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import process_video,find_sample_videos
from src.sidewalk import SidewalkSegmenter
from src.obstacle import ObstacleDetector


def ffmpeg_binary():
    executable=shutil.which("ffmpeg")
    if executable:
        return executable
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def frame_count(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unreadable video: {path}")
    count=0
    while cap.read()[0]:
        count+=1
    info={"frames":count,"width":int(cap.get(3)),"height":int(cap.get(4)),"fps":cap.get(5)}
    cap.release()
    return info


def h264(path,expected):
    """Replace only this run's generated video after validating the encoded copy."""
    temporary=path.with_name("."+path.stem+".h264.mp4")
    subprocess.run([ffmpeg_binary(),"-nostdin","-v","error","-n","-i",str(path),"-map","0:v:0",
                    "-c:v","libx264","-preset","veryfast","-crf","22","-pix_fmt","yuv420p",
                    "-movflags","+faststart","-an",str(temporary)],check=True)
    decoded=frame_count(temporary)
    if decoded["frames"]!=expected:
        raise RuntimeError(f"Encoded frame mismatch: {path}: {decoded}")
    temporary.replace(path)
    return decoded


def compare_logs(newpath,oldpath=None):
    counts,oldcounts,reasons,events,surfaces,rois=Counter(),Counter(),Counter(),Counter(),Counter(),Counter()
    surface_frames=advisory_frames=object_warning_frames=any_warning_frames=diff=rows=0
    old_file=oldpath.open() if oldpath and oldpath.exists() else None
    try:
        with newpath.open() as nf:
            for nl,ol in zip_longest(nf,old_file or []):
                if nl is None or old_file is not None and ol is None:
                    raise RuntimeError("Unequal JSONL lengths")
                row=json.loads(nl);prev=json.loads(ol) if ol else None
                if row["frame_index"]!=rows:
                    raise RuntimeError("Nonsequential frame indices")
                rows+=1
                nd=row["detections"]
                counts.update(d.get("alert_level",d["risk_level"]) or "not_applicable" for d in nd)
                reasons.update(reason for d in nd for reason in d.get("reasons",[]))
                surfaces[row.get("surface",{}).get("status","absent")]+=1
                rois[row["roi"]["source"]]+=1
                events.update(e.get("source","object")+":"+e["type"] for e in row["events"])
                surf=bool(row.get("surface",{}).get("alert_level"))
                advisory=bool(row.get("advisories"))
                obj=any(d.get("alert_level") in ("danger","caution") for d in nd)
                surface_frames+=surf;advisory_frames+=advisory;object_warning_frames+=obj
                any_warning_frames+=surf or advisory or obj
                if prev:
                    if prev["frame_index"]!=row["frame_index"]:
                        raise RuntimeError("Baseline frame index mismatch")
                    od=prev["detections"]
                    same=len(nd)==len(od) and all(
                        a["class_id"]==b["class_id"] and
                        np.allclose(a["xyxy"],b["xyxy"],atol=.001,rtol=0) and
                        abs(a["confidence"]-b["confidence"])<1e-5 for a,b in zip(nd,od))
                    diff+=not same
                    oldcounts.update(d.get("alert_level",d["risk_level"]) or "not_applicable" for d in od)
    finally:
        if old_file:old_file.close()
    return {"jsonl_frames":rows,"risk_observations":dict(counts),
            "baseline_risk_observations":dict(oldcounts),
            "differing_detection_frames":diff if oldpath else None,
            "surface_warning_frames":surface_frames,"advisory_frames":advisory_frames,
            "object_warning_frames":object_warning_frames,"any_warning_frames":any_warning_frames,
            "reason_observations":dict(reasons),"events":dict(events),
            "surface_status_frames":dict(surfaces),"roi_source_frames":dict(rois)}


def paired_video(new,old,out,entry,capture_times):
    a,b=cv2.VideoCapture(str(old)),cv2.VideoCapture(str(new))
    if not a.isOpened() or not b.isOpened():
        raise RuntimeError("Cannot read comparison sources")
    encoder=subprocess.Popen([ffmpeg_binary(),"-nostdin","-v","error","-n","-f","rawvideo",
        "-pix_fmt","bgr24","-s","1080x1008","-r",str(entry["fps"]),"-i","pipe:0",
        "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
        "-movflags","+faststart","-an",str(out)],stdin=subprocess.PIPE)
    count=0
    targets={round(t*entry["fps"]) for t in capture_times}
    try:
        while True:
            oka,fa=a.read();okb,fb=b.read()
            if oka!=okb:raise RuntimeError("Comparison source lengths differ")
            if not oka:break
            pair=np.zeros((1008,1080,3),np.uint8)
            pair[48:,:540]=cv2.resize(fa,(540,960));pair[48:,540:]=cv2.resize(fb,(540,960))
            for x,label in ((12,"ATTEMPT 2"),(552,"ATTEMPT 3")):
                cv2.putText(pair,label,(x,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(pair,f"{count/entry['fps']:.2f}s",(925,32),
                        cv2.FONT_HERSHEY_SIMPLEX,.65,(80,240,255),2,cv2.LINE_AA)
            encoder.stdin.write(pair.tobytes())
            if count in targets:
                cv2.imwrite(str(out.with_name(out.stem+f"_f{count}.jpg")),pair)
            count+=1
    finally:
        a.release();b.release();encoder.stdin.close()
        rc=encoder.wait()
    if rc or count!=entry["expected_frames"]:
        raise RuntimeError(f"Comparison encoding failed: {rc}, {count}")
    verified=frame_count(out)["frames"]
    if verified!=count:raise RuntimeError("Comparison decoding mismatch")
    return verified


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--baseline-dir",required=True,type=Path)
    args=ap.parse_args()
    out=args.output_dir.resolve();base=args.baseline_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/"manifest.json"
    if manifest_path.exists():raise FileExistsError(manifest_path)
    cfg=yaml.safe_load((ROOT/"configs/inference.yaml").read_text())
    cfg.update(device="cuda",mode="both",overlay_alpha=.30)
    cfg["risk"].update(enabled=True,review_overlay=True,log_jsonl=True)
    analysis=out/"analysis";analysis.mkdir(exist_ok=True)
    (out/"comparison").mkdir(exist_ok=True)
    config_path=out/"effective_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    source_paths=list((ROOT/"src").glob("*.py"))+[ROOT/"configs/inference.yaml",Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for p in source_paths:
        target=analysis/"source"/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    entries=[]
    for sample in ("sample1","sample2"):
        (out/sample).mkdir(exist_ok=True)
        for p in find_sample_videos(ROOT/"data/samples"/sample):
            cap=cv2.VideoCapture(str(p))
            entry={"input":str(p),"sample":sample,"expected_frames":int(cap.get(7)),
                   "width":int(cap.get(3)),"height":int(cap.get(4)),"fps":cap.get(5),
                   "output":str(out/sample/("result_"+p.stem+".mp4")),"status":"pending"}
            cap.release()
            prior=base/("result_"+p.stem+".mp4")
            if sample=="sample1" and not prior.exists():raise FileNotFoundError(prior)
            entry["baseline"]=str(prior) if prior.exists() else None
            entries.append(entry)
    if len(entries)!=10:raise RuntimeError(f"Expected 8 + 2 MP4 videos, found {len(entries)}")
    excluded=[str(p) for p in (ROOT/"data/samples/sample2").iterdir() if p.suffix.lower()==".mov"]
    manifest={"status":"running","full_frame_detection":True,"all_source_frames":True,
              "config":str(config_path),"source_hashes":hashes,"excluded_inputs":excluded,
              "total_expected_frames":sum(e["expected_frames"] for e in entries),"videos":entries}
    def save():manifest_path.write_text(json.dumps(manifest,indent=2))
    save();started=time.perf_counter()
    try:
        det=ObstacleDetector(ROOT/cfg["yolo"]["weights"],device="cuda",
             conf=cfg["yolo"]["conf"],imgsz=cfg["yolo"]["imgsz"],head=cfg["yolo"]["head"])
        seg=SidewalkSegmenter(ROOT/cfg["mask2former"]["weights"],device="cuda")
        for i,e in enumerate(entries):
            e["status"]="inferring";save()
            print(f"INFER {i+1}/{len(entries)} {Path(e['input']).name}",flush=True)
            with (analysis/("inference_"+Path(e["input"]).stem+".log")).open("w",buffering=1) as log:
                with contextlib.redirect_stdout(log):
                    count=process_video(e["input"],e["output"],seg,cfg["overlay_alpha"],detector=det,
                                        risk_config=cfg["risk"],tracking_config=cfg["tracking"])
            if count!=e["expected_frames"]:raise RuntimeError("Source frame count mismatch")
            e["processed_frames"]=count;e["status"]="inferred";save()
        # Both models finish before CPU encoding and paired review.
        del det,seg
        manifest["status"]="encoding_and_verifying";save()
        for i,e in enumerate(entries):
            p=Path(e["output"]);e["status"]="verifying";save()
            print(f"VERIFY {i+1}/{len(entries)} {p.name}",flush=True)
            decoded=h264(p,e["expected_frames"])
            if (decoded["width"],decoded["height"])!=(e["width"],e["height"]):
                raise RuntimeError("Output resolution mismatch")
            e.update(decoded_frames=decoded["frames"],codec="h264",bytes=p.stat().st_size)
            prior=Path(e["baseline"]) if e["baseline"] else None
            e.update(compare_logs(p.with_suffix(".risk.jsonl"),prior.with_suffix(".risk.jsonl") if prior else None))
            if e["jsonl_frames"]!=e["expected_frames"]:raise RuntimeError("Log frame count mismatch")
            if prior:
                name=p.stem
                times=([9] if name.endswith(("G24P_003","G24P_004")) else
                       [12,13.5,30.5,36.5] if name.endswith("G25U_002") else
                       [14,15.3] if name.endswith("G25U_003") else [3])
                pair=out/"comparison"/("compare_"+name.removeprefix("result_")+".mp4")
                e["comparison"]=str(pair)
                e["comparison_decoded_frames"]=paired_video(p,prior,pair,e,times)
            else:
                cap=cv2.VideoCapture(str(p));cap.set(cv2.CAP_PROP_POS_FRAMES,min(90,e["expected_frames"]-1))
                ok,frame=cap.read();cap.release()
                if ok:cv2.imwrite(str(p.with_suffix(".jpg")),cv2.resize(frame,(540,960)))
            e["status"]="complete";save()
        protected=json.loads((analysis/"protected_hashes.json").read_text())
        manifest["protected_files_unchanged"]=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
        manifest["source_unchanged_during_run"]=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not manifest["protected_files_unchanged"] or not manifest["source_unchanged_during_run"]:
            raise RuntimeError("Source changed during evaluation")
        manifest.update(status="complete",elapsed_s=time.perf_counter()-started,
                        total_verified_frames=sum(e["decoded_frames"] for e in entries))
    except BaseException as error:
        manifest.update(status="failed",error=repr(error));raise
    finally:
        save()
    print(json.dumps({"status":manifest["status"],"frames":manifest["total_verified_frames"],
                      "elapsed_s":manifest["elapsed_s"]}),flush=True)


if __name__=="__main__":
    main()
