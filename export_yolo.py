#!/usr/bin/env python3
"""
Export the master pet-face dataset (build_pet_face_dataset.py) to a clean
YOLO-pose training set for face detection: cat_face / dog_face + eye_left,
eye_right, nose.

Rules
  - only faces with a known species and not excluded
  - an image is skipped if ANY of its faces has unknown species (an unlabeled
    face in an image would teach the model that a face is background)
  - images that still need review are skipped (use --include-review to keep)
  - split by group: same animal (PetFace identity) or same picture (perceptual
    hash / sha1) always lands in ONE split -> no leakage between train and val
  - --max-per-origin caps a source, e.g. PetFace=10000, to keep the mix balanced
  - images are symlinked, labels written as YOLO-pose txt + data.yaml + manifest.csv

Usage
  python export_yolo.py --master $D/pet_faces --out $D/yolo_merged
  python export_yolo.py --master $D/pet_faces --out $D/yolo_merged \
         --val 0.1 --test 0.1 --max-per-origin PetFace=10000
  then:  python train_face_model.py --data $D/yolo_merged/data.yaml --work $D/yolo_merged
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

KP = ["eye_left", "eye_right", "nose"]
CLASS_NAMES = ["cat_face", "dog_face"]


def split_of(group: str, val: float, test: float) -> str:
    h = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def yolo_line(o, W, H):
    x1, y1, x2, y2 = o["bbox_xyxy"]
    x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    vals = [f"{(x1 + x2) / 2 / W:.6f}", f"{(y1 + y2) / 2 / H:.6f}",
            f"{(x2 - x1) / W:.6f}", f"{(y2 - y1) / H:.6f}"]
    kps = o.get("keypoints") or {}
    for k in KP:
        x, y, v = kps.get(k) or [0, 0, 0]
        v = int(v)
        if v:
            vals += [f"{min(max(x / W, 0), 1):.6f}", f"{min(max(y / H, 0), 1):.6f}", str(v)]
        else:
            vals += ["0.000000", "0.000000", "0"]
    return f"{o['class_id']} " + " ".join(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path, required=True, help="folder with annotations.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--include-review", action="store_true", help="also export images that need review")
    ap.add_argument("--max-per-origin", nargs="*", default=[],
                    help="caps like PetFace=10000 'Tsinghua Dogs=5000'")
    ap.add_argument("--require-keypoints", action="store_true",
                    help="skip faces where an eye or the nose is missing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="delete --out/images and --out/labels first")
    args = ap.parse_args()

    caps = {}
    for c in args.max_per_origin:
        name, _, n = c.rpartition("=")
        caps[name] = int(n)

    records = [json.loads(l) for l in (args.master / "annotations.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(records)                       # so caps take a random subset

    if args.clean:
        for d in ("images", "labels"):
            shutil.rmtree(args.out / d, ignore_errors=True)

    skipped = Counter()
    per_origin = Counter()
    table = defaultdict(Counter)
    manifest = []
    for rec in records:
        origin = rec.get("origin_dataset", "unknown")
        if rec.get("needs_review") and not args.include_review:
            skipped["needs_review"] += 1
            continue
        objs = [o for o in rec["objects"] if not o.get("excluded")]
        if not objs:
            skipped["no_faces_left"] += 1
            continue
        if any(o.get("class_id") is None for o in objs):
            skipped["unknown_species_face"] += 1
            continue
        if args.require_keypoints and any(
                any(((o.get("keypoints") or {}).get(k) or [0, 0, 0])[2] == 0 for k in KP) for o in objs):
            skipped["missing_keypoint"] += 1
            continue
        if origin in caps and per_origin[origin] >= caps[origin]:
            skipped[f"cap_{origin}"] += 1
            continue
        W, H = rec["width"], rec["height"]
        lines = [l for l in (yolo_line(o, W, H) for o in objs) if l]
        if not lines:
            skipped["bad_boxes"] += 1
            continue

        group = rec.get("identity") or rec.get("phash") or rec["sha1"]
        split = split_of(group, args.val, args.test)
        src = (args.master / rec["file"]).resolve()
        ext = Path(rec["file"]).suffix.lower()
        img_dst = args.out / "images" / split / f"{rec['image_id']}{ext}"
        lbl_dst = args.out / "labels" / split / f"{rec['image_id']}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        if not img_dst.exists():
            img_dst.symlink_to(src)
        lbl_dst.write_text("\n".join(lines) + "\n")

        per_origin[origin] += 1
        for o in objs:
            table[(split, origin)][o["species"]] += 1
        manifest.append([rec["image_id"], split, origin, rec.get("identity") or "",
                         len(lines), "|".join(o["species"] for o in objs), str(src)])

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [3, 3]\n"
        "flip_idx: [1, 0, 2]   # eye_left <-> eye_right\n"
        "names:\n  0: cat_face\n  1: dog_face\n"
        "# keypoints: 0 eye_left, 1 eye_right (as seen in the image), 2 nose\n")
    with (args.out / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split", "origin_dataset", "identity", "faces", "species", "source_file"])
        w.writerows(manifest)

    print(f"Exported {len(manifest)} images to {args.out}")
    print("Skipped:", dict(skipped) or "none")
    print("\nFaces per split and origin:")
    totals = defaultdict(Counter)
    for (split, origin), c in sorted(table.items()):
        totals[split].update(c)
        print(f"  {split:5s} {origin:30s} " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    for split in ("train", "val", "test"):
        if split in totals:
            print(f"  {split:5s} {'TOTAL':30s} " + ", ".join(f"{k}: {v}" for k, v in sorted(totals[split].items())))
    print(f"\nData config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
