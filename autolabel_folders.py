#!/usr/bin/env python3
"""
Batch auto-labeling of large image folders (e.g. PetFace cat/ and dog/) with
the trained YOLO-pose face model: one face box + eye_left, eye_right, nose
per image. Species comes from --species (the folder), not from the model.

Commands
  run     label every image, write one CSV per root (resumable: rerun to continue)
  review  copy uncertain + random images into a folder you open in the labeler website
  merge   write your corrections from the website back into the CSV

Examples
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  M=$D/yolo_pfd1/runs/pet_faces/weights/best.pt

  python autolabel_folders.py run --root $D/cat --species cat --model $M --out $D/petface_labels/cat.csv
  python autolabel_folders.py run --root $D/dog --species dog --model $M --out $D/petface_labels/dog.csv
      (test first with --limit 2000)

  python autolabel_folders.py review --csv $D/petface_labels/cat.csv --root $D/cat \
         --review-dir $D/petface_review/cat --random 300
  python app.py --images $D/petface_review/cat/images --labels $D/petface_review/cat/labels \
         --model $M                    # correct, press Enter per image
  python autolabel_folders.py merge --csv $D/petface_labels/cat.csv --review-dir $D/petface_review/cat

CSV columns
  file (relative to root), identity (folder name), species, status (auto / reviewed),
  flag (ok / no_face / low_conf / multi_face / species_mismatch / missing_keypoint),
  score, model_species, n_faces, width, height, x1 y1 x2 y2 (pixels),
  eye_left_x/_y/_v, eye_right_x/_y/_v, nose_x/_y/_v  (v: 2 visible, 1 unsure, 0 missing)
"""

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KP = ["eye_left", "eye_right", "nose"]
COLS = (["file", "identity", "species", "status", "flag", "score", "model_species", "n_faces",
         "width", "height", "x1", "y1", "x2", "y2"]
        + [f"{k}_{a}" for k in KP for a in ("x", "y", "v")])


# --------------------------------------------------------------------------
def list_images(root: Path, limit=None):
    files = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            files.append(p.relative_to(root).as_posix())
            if limit and len(files) >= limit:
                break
    return files


def load(root: Path, rel: str):
    with Image.open(root / rel) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def pad_image(a: np.ndarray, pad: float):
    """Put the image in the middle of a gray canvas so the face looks smaller,
    like the full photos the model was trained on."""
    h, w = a.shape[:2]
    py, px = int(round(h * pad)), int(round(w * pad))
    canvas = np.full((h + 2 * py, w + 2 * px, 3), 114, dtype=np.uint8)
    canvas[py:py + h, px:px + w] = a
    return canvas, px, py


def kp_canonical(points, kp_names, kp_conf):
    nose, eyes = (0.0, 0.0, 0), []
    for p, name in zip(points, kp_names):
        x, y = float(p[0]), float(p[1])
        c = float(p[2]) if len(p) > 2 else 1.0
        v = 2 if c >= kp_conf else 1
        pt = (x, y, v)
        if name.startswith("nose"):
            nose = pt
        elif name.startswith("eye"):
            eyes.append(pt)
    eyes = sorted(eyes, key=lambda e: e[0]) + [(0.0, 0.0, 0)] * (2 - len(eyes))
    return [eyes[0], eyes[1], nose]


def species_of(name: str):
    n = name.lower()
    return "cat" if "cat" in n else "dog" if "dog" in n else "unknown"


# --------------------------------------------------------------------------
def cmd_run(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    kp_names = args.kpt_names.split(",")
    root = args.root
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"listing images in {root} ...")
    files = list_images(root, args.limit)
    done = set()
    if out.exists():
        with out.open(newline="") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    todo = [f for f in files if f not in done]
    print(f"{len(files)} images, {len(done)} already labeled, {len(todo)} to do")
    if not todo:
        return

    new_file = not out.exists()
    fh = out.open("a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(COLS)

    counts = {}
    t0 = time.time()
    pool = ThreadPoolExecutor(args.workers)
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    next_imgs = pool.map(lambda r: load(root, r), batches[0])
    for bi, batch in enumerate(batches):
        imgs = list(next_imgs)
        if bi + 1 < len(batches):
            next_imgs = pool.map(lambda r: load(root, r), batches[bi + 1])   # prefetch
        padded = [pad_image(a, args.pad) for a in imgs]
        # numpy images must be BGR for ultralytics; load() returns RGB (fixed 2026-11-24)
        results = model.predict([p[0][..., ::-1].copy() for p in padded], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False, half=args.half)
        for rel, a, (_, px, py), r in zip(batch, imgs, padded, results):
            h, w = a.shape[:2]
            identity = Path(rel).parent.as_posix()
            row = {"file": rel, "identity": identity, "species": args.species, "status": "auto",
                   "width": w, "height": h}
            n = 0 if r.boxes is None else len(r.boxes)
            row["n_faces"] = n
            if n == 0:
                row.update(flag="no_face", score="", model_species="")
                row.update({c: "" for c in COLS if c not in row})
            else:
                i = int(r.boxes.conf.argmax())
                score = float(r.boxes.conf[i])
                x1, y1, x2, y2 = (r.boxes.xyxy[i].cpu().tolist())
                x1, x2 = min(max(x1 - px, 0), w), min(max(x2 - px, 0), w)
                y1, y2 = min(max(y1 - py, 0), h), min(max(y2 - py, 0), h)
                msp = species_of(model.names[int(r.boxes.cls[i])])
                kps = [(0.0, 0.0, 0)] * 3
                if r.keypoints is not None:
                    raw = r.keypoints.data[i].cpu().tolist()
                    kps = kp_canonical(raw, kp_names, args.kp_conf)
                    kps = [(min(max(x - px, 0), w), min(max(y - py, 0), h), v) if v else (0.0, 0.0, 0)
                           for x, y, v in kps]
                flags = []
                if score < args.low_conf:
                    flags.append("low_conf")
                if n > 1:
                    flags.append("multi_face")
                if args.species in ("cat", "dog") and msp in ("cat", "dog") and msp != args.species:
                    flags.append("species_mismatch")
                if any(v < 2 for *_, v in kps):
                    flags.append("missing_keypoint")
                row.update(flag="|".join(flags) or "ok", score=f"{score:.3f}", model_species=msp,
                           x1=f"{x1:.1f}", y1=f"{y1:.1f}", x2=f"{x2:.1f}", y2=f"{y2:.1f}")
                for k, (x, y, v) in zip(KP, kps):
                    row.update({f"{k}_x": f"{x:.1f}", f"{k}_y": f"{y:.1f}", f"{k}_v": v})
            for f in row["flag"].split("|"):
                counts[f] = counts.get(f, 0) + 1
            writer.writerow([row[c] for c in COLS])
        fh.flush()
        n_done = min((bi + 1) * args.batch, len(todo))
        if bi % 20 == 0 or n_done == len(todo):
            rate = n_done / (time.time() - t0)
            eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
            print(f"  {n_done}/{len(todo)}  {rate:.0f} img/s  ~{eta:.0f} min left  {counts}")
    fh.close()
    print(f"done -> {out}\nflags: {counts}")


# --------------------------------------------------------------------------
def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def cmd_review(args):
    rows = read_csv(args.csv)
    rng = random.Random(args.seed)
    if args.files:        # exact list, e.g. suspicious_files.txt from quality_check.py
        by_file = {r["file"]: r for r in rows}
        wanted = [l.strip() for l in args.files.read_text().splitlines() if l.strip()]
        flagged = [by_file[f] for f in wanted if f in by_file and by_file[f]["status"] != "reviewed"]
        pick = flagged
    else:
        flagged = [r for r in rows if r["flag"] != "ok" and r["status"] != "reviewed"]
        rng.shuffle(flagged)
        flagged = flagged[:args.max_flagged]
        ok = [r for r in rows if r["flag"] == "ok" and r["status"] != "reviewed"]
        pick = flagged + rng.sample(ok, min(args.random, len(ok)))
    img_dir, lbl_dir = args.review_dir / "images", args.review_dir / "labels"
    for r in pick:
        link = img_dir / r["file"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to((args.root / r["file"]).resolve())
        objects = []
        if r["x1"]:
            sp = r["species"]
            cid = {"cat": 0, "dog": 1}.get(sp)
            objects.append({
                "class_id": cid, "class_name": ["cat_face", "dog_face"][cid] if cid is not None else None,
                "species": sp, "score": float(r["score"]),
                "bbox_xyxy": [float(r[c]) for c in ("x1", "y1", "x2", "y2")],
                "keypoints": {k: [float(r[f"{k}_x"] or 0), float(r[f"{k}_y"] or 0), int(r[f"{k}_v"] or 0)]
                              for k in KP},
                "source": "autolabel", "edited": False})
        lp = lbl_dir / f"{r['file']}.json"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps({"image": r["file"], "status": "draft", "width": int(r["width"]),
                                  "height": int(r["height"]), "objects": objects,
                                  "model": "autolabel", "flag": r["flag"]}, indent=1))
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if args.files and (args.files.parent / "suspicious_quality.json").exists():
        q = json.loads((args.files.parent / "suspicious_quality.json").read_text())
        q = {r["file"]: q[r["file"]] for r in pick if r["file"] in q}     # keep quality order
        (lbl_dir / "_quality.json").write_text(json.dumps(q, indent=1))
        print("reasons copied -> the website's 'Needs check' filter shows them")
    print(f"review set: {len(flagged)} flagged + {len(pick) - len(flagged)} random -> {args.review_dir}")
    print(f"open it with:  python app.py --images {img_dir} --labels {lbl_dir} --model <best.pt>")


def cmd_merge(args):
    rows = read_csv(args.csv)
    by_file = {r["file"]: r for r in rows}
    lbl_dir = args.review_dir / "labels"
    n = 0
    for lp in lbl_dir.rglob("*.json"):
        lab = json.loads(lp.read_text())
        if lab.get("status") != "reviewed" or lab["image"] not in by_file:
            continue
        r = by_file[lab["image"]]
        objs = [o for o in lab.get("objects", []) if not o.get("excluded")]
        r["status"] = "reviewed"
        r["n_faces"] = len(objs)
        if not objs:
            r.update(flag="no_face", score="", x1="", y1="", x2="", y2="")
            for k in KP:
                r.update({f"{k}_x": "", f"{k}_y": "", f"{k}_v": ""})
        else:
            o = objs[0]
            r["flag"] = "ok"
            r["x1"], r["y1"], r["x2"], r["y2"] = (f"{v:.1f}" for v in o["bbox_xyxy"])
            for k in KP:
                x, y, v = (o.get("keypoints") or {}).get(k) or [0, 0, 0]
                r.update({f"{k}_x": f"{x:.1f}", f"{k}_y": f"{y:.1f}", f"{k}_v": int(v)})
        n += 1
    tmp = args.csv.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.csv)
    print(f"merged {n} reviewed images into {args.csv}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--species", required=True, choices=["cat", "dog", "unknown"])
    r.add_argument("--model", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--kpt-names", default="eye_left,eye_right,nose")
    r.add_argument("--pad", type=float, default=0.5,
                   help="gray border per side, as fraction of image size (0 = none)")
    r.add_argument("--imgsz", type=int, default=448)
    r.add_argument("--conf", type=float, default=0.05)
    r.add_argument("--low-conf", type=float, default=0.5, help="below this score -> flag low_conf")
    r.add_argument("--kp-conf", type=float, default=0.5)
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--workers", type=int, default=8, help="threads reading images")
    r.add_argument("--device", default=None)
    r.add_argument("--half", action="store_true", help="FP16 on GPU (faster)")
    r.add_argument("--limit", type=int, default=None, help="only the first N images (test run)")

    v = sub.add_parser("review")
    v.add_argument("--csv", type=Path, required=True)
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--review-dir", type=Path, required=True)
    v.add_argument("--random", type=int, default=300, help="random 'ok' images to spot-check")
    v.add_argument("--max-flagged", type=int, default=2000)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--files", type=Path, default=None,
                   help="review exactly these files (one per line), e.g. quality_check suspicious_files.txt")

    m = sub.add_parser("merge")
    m.add_argument("--csv", type=Path, required=True)
    m.add_argument("--review-dir", type=Path, required=True)

    args = ap.parse_args()
    {"run": cmd_run, "review": cmd_review, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
