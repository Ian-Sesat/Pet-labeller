#!/usr/bin/env python3
"""
Find the best keypoint threshold (--kp-conf) for the website.

The model gives every point (left eye, right eye, nose) a confidence. The website
shows a point only when its confidence is >= --kp-conf (default 0.5). Too low: points
you marked missing are shown and you have to delete them. Too high: visible points are
hidden and you have to add them. This script tries many thresholds on your validation
faces and counts the points you would have to fix (move > 5% of the face size, add or
delete) for each one. The model runs only once, so this is quick.

Usage
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  python tune_kp_conf.py --work $D/finetune_v3 --model $D/finetune_v3/runs/finetune/weights/best.pt --device 0

--work is the finetune_labeler.py folder: it uses the same validation faces (data/ and
eval_reviewed-edited/), with the same number of cats and dogs as the evaluation.
Then start the website with  --kp-conf <best value>  and next_batch.py with the same.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_labeler import KP, SPECIES_NAMES, balanced_images, iou, name_single_eye, read_gt  # noqa: E402
from PIL import Image  # noqa: E402


def collect(model, root: Path, imgs, conf, device, imgsz):
    """Run the model once. Returns matched faces: species, face size, true points, raw predicted points."""
    faces, missed = [], 0
    for p in imgs:
        with Image.open(p) as im:
            W, H = im.size
        gt_faces = read_gt(root / "labels" / "val" / f"{p.stem}.txt", W, H)
        r = model.predict(str(p), conf=conf, verbose=False, device=device, imgsz=imgsz)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            kps = r.keypoints.data.cpu().tolist() if r.keypoints is not None else [None] * len(r.boxes)
            preds = [{"box": b, "kps": k} for b, k in zip(r.boxes.xyxy.cpu().tolist(), kps)]
        used = set()
        for g in gt_faces:
            best, bi = 0.0, None
            for i, pr in enumerate(preds):
                if i not in used and iou(g["box"], pr["box"]) > best:
                    best, bi = iou(g["box"], pr["box"]), i
            if bi is None or best < 0.3 or not preds[bi]["kps"]:
                missed += 1
                continue
            used.add(bi)
            pr = preds[bi]
            gt = name_single_eye({k: [gx, gy, int(gv > 0)] for k, (gx, gy, gv) in zip(KP, g["kps"])},
                                 (g["box"][0] + g["box"][2]) / 2)
            faces.append({"sp": SPECIES_NAMES.get(g["cls"], "other"),
                          "size": max(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]),
                          "gt": gt, "pred": pr["kps"], "cx": (pr["box"][0] + pr["box"][2]) / 2})
    return faces, missed


def score(faces, t):
    """Points to fix at threshold t: moved (> 5%), added (hidden), deleted (shown but marked missing)."""
    moved = added = deleted = gt_vis = gt_miss = 0
    for f in faces:
        sh = name_single_eye({k: [q[0], q[1], int((q[2] if len(q) > 2 else 1) >= t)]
                              for k, q in zip(KP, f["pred"])}, f["cx"])
        for k in KP:
            g, s = f["gt"][k], sh[k]
            gt_vis += bool(g[2])
            gt_miss += not g[2]
            if g[2] and s[2]:
                moved += math.hypot(s[0] - g[0], s[1] - g[1]) / f["size"] > 0.05
            elif g[2]:
                added += 1
            elif s[2]:
                deleted += 1
    n = max(3 * len(faces), 1)
    return {"fix": 100 * (moved + added + deleted) / n, "moved": 100 * moved / n, "added": 100 * added / n,
            "deleted": 100 * deleted / n, "hides": 100 * added / max(gt_vis, 1),
            "shows": 100 * deleted / gt_miss if gt_miss else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="finetune_labeler.py --work folder")
    ap.add_argument("--model", type=Path, required=True, help="the model you label with")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="face (box) confidence, as in the website")
    ap.add_argument("--seed", type=int, default=0, help="same as finetune_labeler.py --seed")
    ap.add_argument("--current", type=float, default=0.5, help="the threshold you use now")
    args = ap.parse_args()

    tests = []
    if (args.work / "eval_reviewed-edited" / "images" / "val").is_dir():
        tests.append(("faces you corrected by hand", args.work / "eval_reviewed-edited"))
    tests.append(("all reviewed val faces", args.work / "data"))
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    thresholds = [round(0.30 + 0.05 * i, 2) for i in range(14)]          # 0.30 ... 0.95

    best_overall = None
    for title, root in tests:
        imgs, counts, balanced = balanced_images(root, args.seed + 7, True)
        if not imgs:
            continue
        faces, missed = collect(model, root, imgs, args.conf, args.device, args.imgsz)
        by_sp = {sp: [f for f in faces if f["sp"] == sp] for sp in ("cat", "dog")}
        print(f"\n{title}: {len(faces)} faces ({', '.join(f'{len(v)} {k}s' for k, v in by_sp.items() if v)})"
              f"{f', {missed} not found' if missed else ''}")
        print(f"  {'threshold':>9s} {'to fix':>8s} {'cats':>7s} {'dogs':>7s}   {'moved':>6s} {'add':>6s} "
              f"{'delete':>7s}   {'hides visible':>13s} {'shows missing':>13s}")
        rows = []
        for t in thresholds:
            a = score(faces, t)
            c = score(by_sp["cat"], t) if by_sp["cat"] else None
            d = score(by_sp["dog"], t) if by_sp["dog"] else None
            rows.append((t, a, c, d))
        best_t = min(rows, key=lambda r: (round(r[1]["fix"], 2), abs(r[0] - args.current)))[0]
        for t, a, c, d in rows:
            tag = "  <- best" if t == best_t else ("  <- now" if abs(t - args.current) < 1e-9 else "")
            if t == best_t and abs(t - args.current) < 1e-9:
                tag = "  <- best (= now)"
            fmt = lambda v: f"{v:6.1f}%" if v is not None and not math.isnan(v) else "     – "
            print(f"  {t:9.2f} {fmt(a['fix']):>8s} {fmt(c['fix'] if c else None):>7s} {fmt(d['fix'] if d else None):>7s}   "
                  f"{fmt(a['moved']):>6s} {fmt(a['added']):>6s} {fmt(a['deleted']):>7s}   "
                  f"{fmt(a['hides']):>13s} {fmt(a['shows']):>13s}{tag}")
        now = next(r for r in rows if abs(r[0] - args.current) < 1e-9)[1]["fix"]
        best = next(r for r in rows if r[0] == best_t)
        print(f"  best: --kp-conf {best_t}  ->  {best[1]['fix']:.1f}% points to fix (now {now:.1f}% at {args.current})"
              + (f"; cats {best[2]['fix']:.1f}%, dogs {best[3]['fix']:.1f}%" if best[2] and best[3] else ""))
        if best_overall is None:
            best_overall = (best_t, title, best[1]["fix"], now)

    if best_overall:
        t, title, b, now = best_overall
        print(f"\nRecommendation (decided on {title}): --kp-conf {t}  ({now:.1f}% -> {b:.1f}% points to fix)")
        if abs(t - args.current) < 1e-9:
            print("Your current threshold is already the best one; keep it.")
        else:
            print(f"Website:   python app.py ... --kp-conf {t}")
            print(f"Batches:   python next_batch.py ... --kp-conf {t}")
        print("moved = > 5% off; add = model hid a visible point; delete = model showed a point you marked missing")


if __name__ == "__main__":
    main()
