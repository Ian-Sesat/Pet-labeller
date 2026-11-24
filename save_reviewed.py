#!/usr/bin/env python3
"""
Save the images you REVIEWED in the labeler website, with their labels, into a
separate training-set folder that stays usable on its own (e.g. to train the
next labeler model).

Only labels with status "reviewed" are saved. Rerun it any time: new reviews
are added, labels you changed since the last run are updated, the rest is left
alone. Images are COPIED, so the set keeps working even if the source folder
changes.

Usage
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  python save_reviewed.py --labels $D/new_labels --images $D/cat \
         --out $D/reviewed_sets/cat --species cat
  python save_reviewed.py ... --dry-run      # only show what would be saved
  python save_reviewed.py ... --sync         # also drop images no longer reviewed

Output
  <out>/images/<same path as in --images>     copied images
  <out>/labels/<same path>.txt                YOLO-pose: class xc yc w h + eye_left, eye_right, nose (x y v)
  <out>/json/<same path>.json                 full labeler label (all info, reopenable in the website)
  <out>/manifest.csv                          one row per image: identity, species, faces, sha1, dates ...
  <out>/data.yaml                             ready for ultralytics (train = val = images for now)
  <out>/info.json                             species, sources, history of every save run
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

KP = ["eye_left", "eye_right", "nose"]
SPECIES = {"cat": 0, "dog": 1}
CLASS_NAMES = ["cat_face", "dog_face"]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def yolo_lines(objects, W, H):
    lines = []
    for o in objects:
        cid = o.get("class_id")
        if cid is None:
            continue
        x1, y1, x2, y2 = o["bbox_xyxy"]
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        vals = [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
        parts = [f"{v:.6f}" for v in vals]
        for k in KP:
            x, y, v = ((o.get("keypoints") or {}).get(k) or [0, 0, 0])
            v = int(v)
            if v:
                parts += [f"{min(max(x / W, 0), 1):.6f}", f"{min(max(y / H, 0), 1):.6f}", str(v)]
            else:
                parts += ["0.000000", "0.000000", "0"]
        lines.append(f"{cid} " + " ".join(parts))
    return lines


STATUS = Counter()        # status of every label file seen (for the summary)


def reviewed_labels(labels_dir: Path):
    """Yield (relative image name, label dict) for every reviewed label file."""
    for jf in sorted(labels_dir.rglob("*.json")):
        rel = jf.relative_to(labels_dir)
        if any(p.startswith("_") for p in rel.parts):      # _export, _quality.json ...
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:  # noqa: BLE001
            print(f"  unreadable label file skipped: {jf}")
            continue
        STATUS[d.get("status", "none")] += 1
        if d.get("status") == "reviewed":
            yield d.get("image") or rel.as_posix()[:-5], d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="the website's labels folder")
    ap.add_argument("--images", type=Path, required=True, help="the website's images folder")
    ap.add_argument("--out", type=Path, required=True, help="training-set folder to save into")
    ap.add_argument("--species", choices=list(SPECIES), default=None,
                    help="all saved faces get this species (the folder you labeled, e.g. cat)")
    ap.add_argument("--sync", action="store_true",
                    help="remove saved images that are no longer reviewed in --labels")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    args = ap.parse_args()

    for name, p in (("--labels", args.labels), ("--images", args.images)):
        if not p.is_dir():
            raise SystemExit(f"{name} folder not found: {p}\n"
                             "(if the path starts with /new_labels or /cat, $D is not set in this terminal)")
    print(f"labels: {args.labels}\nimages: {args.images}")
    out = args.out
    now = dt.datetime.now().isoformat(timespec="seconds")
    counts = Counter()
    seen = set()
    species_seen = Counter()

    for name, d in reviewed_labels(args.labels):
        src = args.images / name
        if not src.is_file():
            counts["skipped_image_missing"] += 1
            print(f"  image not found, skipped: {src}")
            continue
        objects = [o for o in d.get("objects", []) if not o.get("excluded")]
        if args.species:
            for o in objects:
                if o.get("species") not in (None, "unknown", args.species):
                    counts["species_overridden"] += 1
                cid = SPECIES[args.species]
                o.update(species=args.species, class_id=cid, class_name=CLASS_NAMES[cid])
        if any(o.get("class_id") is None for o in objects):
            counts["skipped_unknown_species"] += 1
            print(f"  face without species, skipped (use --species or fix it in the website): {name}")
            continue
        seen.add(name)
        species_seen.update(o["species"] for o in objects)

        dst_img = out / "images" / name
        dst_json = out / "json" / f"{name}.json"
        dst_txt = (out / "labels" / name).with_suffix(".txt")
        if dst_json.exists():
            old = json.loads(dst_json.read_text())
            if old.get("updated") == d.get("updated") and dst_img.exists():
                counts["unchanged"] += 1
                continue
            counts["updated"] += 1
        else:
            counts["new"] += 1
        if args.dry_run:
            continue

        W, H = d.get("width"), d.get("height")
        if not W or not H:
            from PIL import Image, ImageOps
            with Image.open(src) as im:
                W, H = ImageOps.exif_transpose(im).size
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_img)
        lines = yolo_lines(objects, W, H)
        dst_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
        saved = dict(d, objects=objects, width=W, height=H, saved_at=now,
                     sha1=sha1_file(dst_img), source_image=str(src.resolve()),
                     source_label_dir=str(args.labels.resolve()))
        dst_json.write_text(json.dumps(saved, indent=1))
        if not lines:
            counts["saved_without_face"] += 1

    if args.sync:
        for jf in sorted((out / "json").rglob("*.json")) if (out / "json").is_dir() else []:
            name = jf.relative_to(out / "json").as_posix()[:-5]
            if name in seen:
                continue
            counts["removed_no_longer_reviewed"] += 1
            if not args.dry_run:
                for p in (jf, out / "images" / name, (out / "labels" / name).with_suffix(".txt")):
                    p.unlink(missing_ok=True)

    print("label files found by status:", dict(STATUS) or "none")
    if not STATUS:
        raise SystemExit("no label files (*.json) in --labels: is this the folder the website saved to?")
    if not STATUS.get("reviewed"):
        raise SystemExit("no REVIEWED labels yet: in the website press Enter (Reviewed & next) on checked images")
    if args.dry_run:
        print("DRY RUN, nothing written:", dict(counts))
        return

    # manifest + data.yaml + info.json, rebuilt from everything in the set
    rows, faces, kp_vis = [], Counter(), Counter()
    for jf in sorted((out / "json").rglob("*.json")) if (out / "json").is_dir() else []:
        s = json.loads(jf.read_text())
        name = jf.relative_to(out / "json").as_posix()[:-5]
        objs = s.get("objects", [])
        faces.update(o["species"] for o in objs)
        for o in objs:
            for k in KP:
                kp_vis[f"{k}_{['missing', 'unsure', 'visible'][int(((o.get('keypoints') or {}).get(k) or [0, 0, 0])[2])]}"] += 1
        rows.append({
            "file": name, "identity": name.split("/")[0] if "/" in name else "",
            "species": "|".join(sorted({o["species"] for o in objs})) or "none",
            "faces": len(objs), "width": s.get("width"), "height": s.get("height"),
            "sha1": s.get("sha1", ""), "reviewed_at": s.get("updated", ""), "saved_at": s.get("saved_at", ""),
            "model": s.get("model", ""), "edited_by_hand": any(o.get("edited") for o in objs),
            "source_image": s.get("source_image", ""),
        })
    if rows:
        with (out / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        "kpt_shape: [3, 3]\nflip_idx: [1, 0, 2]\n"
        "names:\n  0: cat_face\n  1: dog_face\n"
        "# keypoints: 0 eye_left, 1 eye_right (as seen in the image), 2 nose\n"
        "# reviewed set: make a proper train/val split (per identity) before training\n")
    info_path = out / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"created": now, "runs": []}
    identities = {r["identity"] for r in rows if r["identity"]}
    sha_dups = sum(c - 1 for c in Counter(r["sha1"] for r in rows if r["sha1"]).values() if c > 1)
    info.update({
        "species": sorted(set(species_seen) | set(faces)) or ([args.species] if args.species else []),
        "images": len(rows), "faces": dict(faces), "identities": len(identities),
        "keypoints": dict(kp_vis), "duplicate_images": sha_dups,
        "source_images": str(args.images.resolve()), "source_labels": str(args.labels.resolve()),
        "label_format": "labeler JSON (json/) + YOLO-pose txt (labels/), keypoints eye_left, eye_right, nose",
    })
    info["runs"].append({"date": now, **dict(counts), "images_total": len(rows)})
    info_path.write_text(json.dumps(info, indent=1))

    print(f"saved set: {out}")
    print("this run:", dict(counts))
    print(f"total in set: {len(rows)} images, faces {dict(faces)}, {len(identities)} identities"
          + (f", {sha_dups} duplicate images" if sha_dups else ""))


if __name__ == "__main__":
    main()
