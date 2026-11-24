#!/usr/bin/env python3
"""
Check a YOLO-pose pet-face dataset (e.g. the one made by train_face_model.py
--prepare-only, or the labeler's _export folder) before training.

1. Automatic checks on EVERY label line
   - every image has a label file and every label file has an image
   - 14 numbers per line: class xc yc w h + 3 x (x y v)
   - class is 0 (cat_face) or 1 (dog_face)
   - box and keypoints inside the image (0..1), box not empty
   - eye_left is left of eye_right (keypoint order is consistent)
   - keypoints lie inside their face box
   - nose below the eyes (upright faces; a few exceptions are normal)
   - class matches the species in the file name (pfd1 naming)
   - optional --hash: identical images in different splits (data leakage)
2. Picture grids to look at with your own eyes (in <dataset>/check/):
   - random_<split>_<class>.jpg   random samples with box + keypoints drawn
   - suspicious_<problem>.jpg     samples of every problem found
3. report.txt with all numbers and the list of problem files

Usage
  pip install pillow
  python check_yolo_dataset.py /media/.../yolo_pfd1
  python check_yolo_dataset.py /media/.../yolo_pfd1 --samples 32 --hash
"""

import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "cat_face", 1: "dog_face"}
CLASS_COL = {0: (255, 159, 67), 1: (77, 171, 247)}
KP_NAMES = ["eye_left", "eye_right", "nose"]
KP_COL = [(81, 207, 102), (255, 107, 107), (255, 212, 59)]


def species_from_filename(stem):
    """Species implied by pet-face-dataset-1 file names (None = can't tell)."""
    if re.fullmatch(r"\d+_\d+", stem):
        return 0
    if re.search(r"_n\d+$", stem):
        return 1
    m = re.fullmatch(r"([A-Za-z]+)_\d+", stem)
    if m:
        w = m.group(1).lower()
        if w.endswith("cat"):
            return 0
        if w.endswith("dog"):
            return 1
    return None


def parse_label(path):
    """Return (rows, problems). rows = [(cls, box, kps)]."""
    rows, problems = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 14:
            problems.append(f"line {n}: {len(parts)} numbers instead of 14")
            continue
        try:
            v = [float(x) for x in parts]
        except ValueError:
            problems.append(f"line {n}: not numeric")
            continue
        kps = [v[5:8], v[8:11], v[11:14]]
        rows.append((int(v[0]), v[1:5], kps))
    return rows, problems


def check_face(cls, box, kps, stem):
    flags = []
    xc, yc, w, h = box
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    if cls not in CLASS_NAMES:
        flags.append("bad_class")
    if w <= 0 or h <= 0:
        flags.append("empty_box")
    if min(x1, y1) < -1e-4 or max(x2, y2) > 1 + 1e-4:
        flags.append("box_outside_image")
    for x, y, v in kps:
        if v not in (0, 1, 2):
            flags.append("bad_visibility")
        if v and not (0 <= x <= 1 and 0 <= y <= 1):
            flags.append("keypoint_outside_image")
    (lx, ly, lv), (rx, ry, rv), (nx, ny, nv) = kps
    if lv and rv and lx > rx:
        flags.append("eyes_swapped")
    mx, my = 0.15 * w, 0.15 * h
    for x, y, v in kps:
        if v and not (x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my):
            flags.append("keypoint_outside_box")
            break
    if lv and rv and nv and ny < min(ly, ry):
        flags.append("nose_above_eyes")
    if lv and rv and abs(lx - rx) < 0.05 * w:
        flags.append("eyes_almost_identical")
    if sum(1 for *_, v in kps if v == 0) > 0:
        flags.append("missing_keypoint")
    expected = species_from_filename(stem)
    if expected is not None and cls != expected:
        flags.append("class_differs_from_filename")
    return flags


def draw_tile(img_path, rows, size=256, highlight=None):
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 250))
    r = max(3, round(max(W, H) / 120))
    for i, (cls, (xc, yc, w, h), kps) in enumerate(rows):
        col = CLASS_COL.get(cls, (200, 200, 200))
        if highlight is not None and i != highlight:
            col = tuple(c // 2 for c in col)
        d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                    outline=col, width=lw)
        for (x, y, v), kc in zip(kps, KP_COL):
            if v:
                d.ellipse([x * W - r, y * H - r, x * W + r, y * H + r],
                          fill=kc if v == 2 else None, outline=kc, width=max(1, lw // 2))
    im.thumbnail((size, size))
    tile = Image.new("RGB", (size, size + 16), (25, 25, 25))
    tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return tile


def make_grid(items, out_path, title, cols=6, size=256):
    """items: [(img_path, rows, caption, highlight)]"""
    if not items:
        return
    font = ImageFont.load_default()
    rows_n = (len(items) + cols - 1) // cols
    grid = Image.new("RGB", (cols * size, rows_n * (size + 16) + 24), (15, 15, 15))
    d = ImageDraw.Draw(grid)
    d.text((6, 6), title + "   box: orange = cat, blue = dog | points: green = left eye, "
           "red = right eye, yellow = nose (hollow = occluded)", fill=(230, 230, 230), font=font)
    for k, (p, rows, cap, hl) in enumerate(items):
        x, y = (k % cols) * size, 24 + (k // cols) * (size + 16)
        grid.paste(draw_tile(p, rows, size, hl), (x, y))
        d.text((x + 4, y + size + 2), cap[:42], fill=(200, 200, 200), font=font)
    grid.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder containing images/ and labels/")
    ap.add_argument("--samples", type=int, default=24, help="random images per split and class")
    ap.add_argument("--hash", action="store_true", help="also look for identical images across splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = args.root
    out = root / "check"
    out.mkdir(exist_ok=True)

    report = []
    say = lambda *a: (print(*a), report.append(" ".join(str(x) for x in a)))  # noqa: E731

    images = [p for p in (root / "images").rglob("*") if p.suffix.lower() in IMG_EXT]
    labels = {p.relative_to(root / "labels").with_suffix(""): p for p in (root / "labels").rglob("*.txt")}
    say(f"Dataset: {root}")
    say(f"images: {len(images)}   label files: {len(labels)}")

    per_split = defaultdict(Counter)
    flags_total = Counter()
    suspicious = defaultdict(list)
    samples = defaultdict(list)
    file_problems = []
    seen_keys = set()
    hashes = defaultdict(list)
    box_sizes = defaultdict(list)

    for p in sorted(images):
        rel = p.relative_to(root / "images")
        split = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        key = rel.with_suffix("")
        seen_keys.add(key)
        lp = labels.get(key)
        if lp is None:
            file_problems.append(f"no label file: {rel}")
            per_split[split]["image_without_label"] += 1
            continue
        rows, probs = parse_label(lp)
        for pr in probs:
            file_problems.append(f"{lp.relative_to(root)}: {pr}")
        if not rows:
            per_split[split]["background_images"] += 1
        per_split[split]["images"] += 1
        for i, (cls, box, kps) in enumerate(rows):
            per_split[split][CLASS_NAMES.get(cls, f"class_{cls}")] += 1
            box_sizes[cls].append(box[2] * box[3])
            fl = check_face(cls, box, kps, p.stem)
            flags_total.update(fl)
            for f in fl:
                suspicious[f].append((p, rows, f"{p.name} [{i + 1}]", i))
            samples[(split, cls)].append((p, rows, p.name, None))
        if args.hash:
            hashes[hashlib.sha1(p.read_bytes()).hexdigest()].append(split + "/" + p.name)

    for key, lp in labels.items():
        if key not in seen_keys:
            file_problems.append(f"label without image: {lp.relative_to(root)}")

    say("\nFaces per split:")
    for split, c in sorted(per_split.items()):
        say(f"  {split:6s} " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    for cls, s in sorted(box_sizes.items()):
        s = sorted(s)
        say(f"  {CLASS_NAMES.get(cls, cls)} box area (fraction of image): median {s[len(s)//2]:.3f}, "
            f"min {s[0]:.4f}, max {s[-1]:.3f}")

    n_faces = sum(len(v) for v in samples.values())
    say("\nChecks (faces affected):")
    critical = ["bad_class", "empty_box", "box_outside_image", "keypoint_outside_image",
                "bad_visibility", "eyes_swapped", "class_differs_from_filename"]
    for f in critical + ["keypoint_outside_box", "nose_above_eyes", "eyes_almost_identical",
                         "missing_keypoint"]:
        n = flags_total.get(f, 0)
        mark = "OK " if n == 0 else ("ERR" if f in critical else "see")
        say(f"  [{mark}] {f:28s} {n:6d}  ({100 * n / max(n_faces, 1):.2f}%)")
    say(f"  file-level problems: {len(file_problems)}")
    for line in file_problems[:20]:
        say(f"     {line}")

    if args.hash:
        leaks = [v for v in hashes.values() if len({x.split('/')[0] for x in v}) > 1]
        say(f"\nIdentical images in more than one split: {len(leaks)}")
        for v in leaks[:10]:
            say(f"     {v}")

    say(f"\nPicture grids in {out}:")
    for (split, cls), items in sorted(samples.items()):
        pick = rng.sample(items, min(args.samples, len(items)))
        path = out / f"random_{split}_{CLASS_NAMES.get(cls, cls)}.jpg"
        make_grid(pick, path, f"{split} / {CLASS_NAMES.get(cls, cls)} - random {len(pick)}")
        say(f"  {path.name}")
    for f, items in sorted(suspicious.items()):
        if f == "missing_keypoint" and len(items) > 5000:
            continue
        pick = rng.sample(items, min(args.samples, len(items)))
        path = out / f"suspicious_{f}.jpg"
        make_grid(pick, path, f"{f} - {len(items)} faces, showing {len(pick)} (bright = affected face)")
        say(f"  {path.name}")
    with (out / "suspicious_files.txt").open("w") as fh:
        for f, items in sorted(suspicious.items()):
            for p, _, cap, _ in items:
                fh.write(f"{f}\t{p.relative_to(root)}\t{cap}\n")
    (out / "report.txt").write_text("\n".join(report) + "\n")
    say(f"\nReport: {out / 'report.txt'}   all problem files: {out / 'suspicious_files.txt'}")


if __name__ == "__main__":
    main()
