#!/usr/bin/env python3
"""
Save only the TRUSTWORTHY labels of a whole folder (e.g. PetFace cat/ or dog/), in the
same CSV format as autolabel_folders.py. Uncertain images are left out and listed
separately with the reason, so nothing is lost and you can look at them later.

For every image:
  - reviewed by you (in --reviewed)      -> your label is saved (status "reviewed")
  - otherwise the model labels it twice: normally and on a mirrored copy. It is saved only if
      a face is found, with score >= --low-conf, and only one face
      the model's species = the folder's species
      the mirrored copy finds the face too and both answers agree (< --max-err of the face size)
      at least one point is visible (confidence >= --kp-conf)
      the keypoint layout passes the shape check (--template; upside-down heads are allowed)
    The saved points are the average of the normal and mirrored answer (more accurate).
    One visible eye is named by its side of the nose. Hidden points are saved as missing (v=0).

Output (--out folder)
  good.csv       the trustworthy labels (columns as autolabel_folders.py + est_error)
  left_out.csv   images not saved: file, identity, reason, details
  summary.json   settings, counts per reason, share kept

Resumable: run the same command again and it continues where it stopped.

Usage
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  M=$D/finetune_v3/runs/finetune/weights/best.pt
  python save_good_labels.py --root $D/cat --species cat --model $M --reviewed $D/reviewed_sets/cat \
      --template $D/quality/template/template.json --out $D/final_labels/cat --kp-conf 0.8 --device 0 --half
  (test first with --limit 2000)
"""

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel_folders import COLS, KP, list_images, load, pad_image, species_of  # noqa: E402

GOOD_COLS = COLS + ["est_error"]
LEFT_COLS = ["file", "identity", "reason", "details", "score", "est_error"]
REASONS = {
    "no_face": "no face found",
    "low_conf": "face found, but the model was unsure (low score)",
    "multi_face": "more than one face found",
    "species_mismatch": "model says the other species than the folder",
    "mirror_no_face": "no face found in the mirrored copy",
    "doubtful": "normal and mirrored answer disagree",
    "no_keypoints": "no point recognised",
    "unusual_layout": "keypoint layout fails the shape check",
    "reviewed_no_face": "you reviewed it and removed the face",
}


def faces_of(r, px, py, w, h, mirrored=False, cw=None):
    """All faces of one result in original-image coordinates: [(score, cls, box, [[x, y, c] x3])]."""
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    kps_all = r.keypoints.data.cpu().tolist() if r.keypoints is not None else [None] * len(r.boxes)
    for b, s, c, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(),
                          r.boxes.cls.cpu().int().tolist(), kps_all):
        if mirrored:
            b = [cw - b[2], b[1], cw - b[0], b[3]]
            k = [[cw - p[0], p[1], p[2] if len(p) > 2 else 1.0] for p in k] if k else None
        b = [min(max(b[0] - px, 0), w), min(max(b[1] - py, 0), h), min(max(b[2] - px, 0), w), min(max(b[3] - py, 0), h)]
        pts = [[p[0] - px, p[1] - py, p[2] if len(p) > 2 else 1.0] for p in k[:3]] if k else [[0, 0, 0]] * 3
        eyes = sorted(pts[:2], key=lambda p: p[0])          # eye_left / eye_right as seen in the image
        out.append((s, c, b, [eyes[0], eyes[1], pts[2]]))
    out.sort(key=lambda f: -f[0])
    return out


def name_single_eye(pts, cx):
    """pts [[x, y, v] x3] (eye_left, eye_right, nose): one visible eye is named by its side."""
    el, er, n = pts
    if bool(el[2]) != bool(er[2]):
        e = el if el[2] else er
        ref = n[0] if n[2] else cx
        none = [0.0, 0.0, 0]
        return [e, none, n] if e[0] <= ref else [none, e, n]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="image folder, e.g. $D/cat")
    ap.add_argument("--species", required=True, choices=["cat", "dog"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--reviewed", type=Path, nargs="*", default=[],
                    help="saved sets from save_reviewed.py: your labels replace the model's")
    ap.add_argument("--template", type=Path, default=None, help="template.json for the shape check")
    ap.add_argument("--pad", type=float, default=0.5, help="same as the website's --pad")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="face detection threshold")
    ap.add_argument("--low-conf", type=float, default=0.5, help="faces with a lower score are left out")
    ap.add_argument("--kp-conf", type=float, default=0.8, help="point shown if confidence >= this (website value)")
    ap.add_argument("--max-err", type=float, default=0.05,
                    help="largest allowed half-disagreement normal vs mirrored, share of the face size")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true", help="FP16 on the GPU (faster)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (test run)")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"--root not found: {args.root}  (is $D set?)")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- your reviewed labels
    reviewed = {}
    for s in args.reviewed:
        for jf in (s / "json").rglob("*.json"):
            reviewed[jf.relative_to(s / "json").as_posix()[:-5]] = json.loads(jf.read_text())
    print(f"reviewed labels: {len(reviewed)}")

    qc = T = None
    if args.template:
        import quality_check as qc
        T = json.loads(args.template.read_text())
        qc.CROPS = qc.FACE_CROPS = args.pad > 0

    print(f"listing images in {args.root} ...")
    files = list_images(args.root, args.limit)
    good_p, left_p = args.out / "good.csv", args.out / "left_out.csv"
    done = set()
    for p in (good_p, left_p):
        if p.exists():
            with p.open(newline="") as fh:
                done |= {r["file"] for r in csv.DictReader(fh)}
    todo = [f for f in files if f not in done]
    print(f"{len(files)} images, {len(done)} already done, {len(todo)} to do")

    gfh, lfh = good_p.open("a", newline=""), left_p.open("a", newline="")
    gw, lw = csv.writer(gfh), csv.writer(lfh)
    if gfh.tell() == 0:
        gw.writerow(GOOD_COLS)
    if lfh.tell() == 0:
        lw.writerow(LEFT_COLS)

    from ultralytics import YOLO
    model = YOLO(args.model)
    fp16 = {"half": True} if args.half else {}           # only pass it when asked (newer versions warn)
    counts = Counter()
    t0 = time.time()
    pool = ThreadPoolExecutor(args.workers)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    def left(rel, reason, details="", score="", est=""):
        lw.writerow([rel, Path(rel).parent.as_posix(), reason, details, score, est])
        counts[reason] += 1

    def good(rel, w, h, status, score, msp, n, box, pts, est):
        row = {"file": rel, "identity": Path(rel).parent.as_posix(), "species": args.species, "status": status,
               "flag": "ok", "score": score, "model_species": msp, "n_faces": n, "width": w, "height": h,
               "x1": f"{box[0]:.1f}", "y1": f"{box[1]:.1f}", "x2": f"{box[2]:.1f}", "y2": f"{box[3]:.1f}",
               "est_error": est}
        for k, (x, y, v) in zip(KP, pts):
            row.update({f"{k}_x": f"{x:.1f}" if v else "0.0", f"{k}_y": f"{y:.1f}" if v else "0.0", f"{k}_v": v})
        gw.writerow([row[c] for c in GOOD_COLS])
        counts["saved_" + status] += 1

    next_imgs = pool.map(lambda r: load(args.root, r), batches[0]) if batches else iter(())
    for bi, batch in enumerate(batches):
        imgs = list(next_imgs)
        if bi + 1 < len(batches):
            next_imgs = pool.map(lambda r: load(args.root, r), batches[bi + 1])   # prefetch
        # reviewed images: your label, no model needed
        model_idx = []
        for i, (rel, a) in enumerate(zip(batch, imgs)):
            d = reviewed.get(rel)
            if d is None:
                model_idx.append(i)
                continue
            objs = [o for o in d.get("objects", []) if not o.get("excluded")]
            h, w = a.shape[:2]
            if not objs:
                left(rel, "reviewed_no_face", REASONS["reviewed_no_face"])
                continue
            o = objs[0]
            k = o.get("keypoints") or {}
            pts = [[*(k.get(n) or [0, 0, 0])[:2], 2 if (k.get(n) or [0, 0, 0])[2] else 0] for n in KP]
            box = o["bbox_xyxy"]
            good(rel, w, h, "reviewed", "", "", len(objs), box, name_single_eye(pts, (box[0] + box[2]) / 2), "")
        if model_idx:
            sub = [imgs[i] for i in model_idx]
            padded = [pad_image(a, args.pad) for a in sub]
            canv = [c for c, _, _ in padded]
            # numpy images must be BGR for ultralytics (our arrays are RGB)
            res = model.predict([c[..., ::-1].copy() for c in canv], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False, **fp16)
            resm = model.predict([c[:, ::-1, ::-1].copy() for c in canv], imgsz=args.imgsz, conf=args.conf,
                                 device=args.device, verbose=False, **fp16)
            for i, (canvas, px, py), r, rm in zip(model_idx, padded, res, resm):
                rel, a = batch[i], imgs[i]
                h, w = a.shape[:2]
                fs = faces_of(r, px, py, w, h)
                if not fs:
                    left(rel, "no_face", REASONS["no_face"])
                    continue
                score, cls, box, pts = fs[0]
                sc = f"{score:.3f}"
                msp = species_of(model.names[cls])
                if score < args.low_conf:
                    left(rel, "low_conf", REASONS["low_conf"], sc)
                    continue
                if len(fs) > 1 and fs[1][0] >= args.low_conf:
                    left(rel, "multi_face", REASONS["multi_face"], sc)
                    continue
                if msp in ("cat", "dog") and msp != args.species:
                    left(rel, "species_mismatch", f"model says {msp}", sc)
                    continue
                fm = faces_of(rm, px, py, w, h, mirrored=True, cw=canvas.shape[1])
                if not fm:
                    left(rel, "mirror_no_face", REASONS["mirror_no_face"], sc)
                    continue
                _, _, mbox, mpts = fm[0]
                size = max(box[2] - box[0], box[3] - box[1], 1.0)
                # average of both answers; a point is shown if its mean confidence >= kp_conf
                avg = []
                for p, q in zip(pts, mpts):
                    c = (p[2] + q[2]) / 2
                    avg.append([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 2 if c >= args.kp_conf else 0])
                both = [j for j in range(3) if pts[j][2] >= args.kp_conf and mpts[j][2] >= args.kp_conf]
                est = max((math.hypot(pts[j][0] - mpts[j][0], pts[j][1] - mpts[j][1]) / size / 2 for j in both),
                          default=0.0)
                es = f"{est:.4f}"
                if est > args.max_err:
                    left(rel, "doubtful", f"answers {100 * est:.1f}% apart", sc, es)
                    continue
                abox = [(x + y) / 2 for x, y in zip(box, mbox)]
                avg = name_single_eye(avg, (abox[0] + abox[2]) / 2)
                if not any(p[2] for p in avg):
                    left(rel, "no_keypoints", REASONS["no_keypoints"], sc, es)
                    continue
                if qc:
                    f = dict(image=rel, path="", W=w, H=h, species=args.species, box=abox,
                             kps={k: p for k, p in zip(KP, avg)}, origin="", idx=0, pre_flags=[])
                    qc.analyze(f)
                    qc.score_face(f, T)
                    if f["flagged"]:
                        left(rel, "unusual_layout", "; ".join(f["reasons"][:2]), sc, es)
                        continue
                good(rel, w, h, "auto", sc, msp, len(fs), abox, avg, es)
        gfh.flush()
        lfh.flush()
        n_done = min((bi + 1) * args.batch, len(todo))
        if bi % 20 == 0 or n_done == len(todo):
            rate = n_done / (time.time() - t0)
            eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
            kept = counts["saved_auto"] + counts["saved_reviewed"]
            print(f"  {n_done}/{len(todo)}  {rate:.0f} img/s  ~{eta:.0f} min left  kept {kept} "
                  f"({100 * kept / max(n_done, 1):.1f}%)")
    gfh.close()
    lfh.close()

    # ---- summary over the whole output (also earlier runs)
    tot = Counter()
    with good_p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            tot["saved_" + r["status"]] += 1
    with left_p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            tot[r["reason"]] += 1
    n_all = sum(tot.values())
    kept = tot["saved_auto"] + tot["saved_reviewed"]
    summary = {
        "date": dt.datetime.now().isoformat(timespec="seconds"), "root": str(args.root), "species": args.species,
        "model": str(args.model), "images": n_all, "kept": kept, "kept_pct": round(100 * kept / max(n_all, 1), 2),
        "counts": dict(tot), "settings": {k: str(v) for k, v in vars(args).items() if k != "reviewed"},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{args.species}: {n_all} images -> kept {kept} ({summary['kept_pct']}%): "
          f"{tot['saved_auto']} auto + {tot['saved_reviewed']} reviewed")
    print("left out:")
    for k, v in sorted(((k, v) for k, v in tot.items() if not k.startswith("saved_")), key=lambda kv: -kv[1]):
        print(f"  {v:8d}  {REASONS.get(k, k)}")
    print(f"-> {good_p}\n-> {left_p}")


if __name__ == "__main__":
    main()
