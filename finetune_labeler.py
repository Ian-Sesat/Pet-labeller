#!/usr/bin/env python3
"""
Fine-tune the labeler model (best.pt) on your REVIEWED cats and dogs, then
check that the new model is really better than the old one.

What it does
  1. prepare
     - reads the saved sets from save_reviewed.py (reviewed_sets/cat, reviewed_sets/dog)
     - leaves out images without a face (reviewed with the box deleted)
     - splits train / val BY ANIMAL (identity folder), never by image; the split
       is stored in split.csv and reused next time, so val stays val
     - puts every crop on the same gray border the website uses (--pad 0.5)
       and moves boxes + keypoints accordingly
     - adds a random sample of dataset-1 training images ("replay"), so the
       model does not forget what it learned there
  2. train   fine-tune best.pt gently (low learning rate, few epochs)
  3. eval    old vs new model on the same held-out images:
       reviewed val (crops, like in the website) and dataset-1 val (forgetting check)
       -> faces missed, box overlap (IoU), keypoint error, species correct

Usage
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  M=$D/yolo_pfd1/runs/pet_faces/weights/best.pt
  python finetune_labeler.py --sets $D/reviewed_sets/cat $D/reviewed_sets/dog \
      --base-model $M --replay $D/yolo_pfd1 --work $D/finetune_v2
  python finetune_labeler.py ... --prepare-only     # build the dataset, look at it first
  python finetune_labeler.py ... --eval-only        # compare again (e.g. other weights: --new-model)
  python finetune_labeler.py ... --eval-only --eval-on reviewed pfd1-val pfd1-test pfd1-train

--replay is the folder train_face_model.py prepared (images/train, labels/train,
images/val, labels/val with eye_left, eye_right, nose). Leave it out to train on
the reviewed images only (not recommended: the model may forget).
New model: <work>/runs/finetune/weights/best.pt  - results: <work>/eval.md
"""

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

KP = ["eye_left", "eye_right", "nose"]
SPECIES = {"cat": 0, "dog": 1}
GRAY = (114, 114, 114)                      # same border colour as the website
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# 1. prepare
# --------------------------------------------------------------------------
def pad_image(img, pad):
    """Exactly what app.py does before running the model on a crop."""
    px, py = int(round(img.width * pad)), int(round(img.height * pad))
    canvas = Image.new("RGB", (img.width + 2 * px, img.height + 2 * py), GRAY)
    canvas.paste(img, (px, py))
    return canvas, px, py


KP_CONF = 0.5          # a predicted point counts as shown when its confidence is >= this (as in the website)


def name_single_eye(pts, cx):
    """pts {name: [x, y, visible]}: if exactly one eye is visible, name it by its side of the
    nose (or of the face centre cx). You may label a single eye left OR right; the model
    needs one consistent rule, so training and evaluation use this one."""
    seen = [k for k in ("eye_left", "eye_right") if pts[k][2]]
    if len(seen) == 1:
        e = pts[seen[0]]
        ref = pts["nose"][0] if pts["nose"][2] else cx
        none = [0.0, 0.0, 0]
        pts["eye_left"], pts["eye_right"] = (e, none) if e[0] <= ref else (none, e)
    return pts


def yolo_line(o, px, py, CW, CH, W, H):
    """One face -> YOLO-pose line on the padded canvas (box clipped to the original image)."""
    x1, y1, x2, y2 = o["bbox_xyxy"]
    x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    x1, x2, y1, y2 = x1 + px, x2 + px, y1 + py, y2 + py
    parts = [f"{(x1 + x2) / 2 / CW:.6f}", f"{(y1 + y2) / 2 / CH:.6f}",
             f"{(x2 - x1) / CW:.6f}", f"{(y2 - y1) / CH:.6f}"]
    pts = {}
    for k in KP:
        x, y, v = ((o.get("keypoints") or {}).get(k) or [0, 0, 0])
        v = int(v)
        pts[k] = [x, y, v] if v and 0 <= x <= W and 0 <= y <= H else [0.0, 0.0, 0]
    name_single_eye(pts, (x1 + x2) / 2 - px)     # 1, 2 or 3 points; missing ones stay v=0 (ignored in training)
    for k in KP:
        x, y, v = pts[k]
        parts += [f"{(x + px) / CW:.6f}", f"{(y + py) / CH:.6f}", str(v)] if v else ["0.000000", "0.000000", "0"]
    return f"{SPECIES[o['species']]} " + " ".join(parts)


def load_set(set_dir: Path):
    """Reviewed set from save_reviewed.py -> list of (name, identity, label dict)."""
    items = []
    for jf in sorted((set_dir / "json").rglob("*.json")):
        d = json.loads(jf.read_text())
        name = jf.relative_to(set_dir / "json").as_posix()[:-5]
        identity = name.split("/")[0] if "/" in name else name
        items.append((name, identity, d))
    return items


def make_split(sets, work: Path, val_frac, seed):
    """Stable split by animal: identities already in split.csv keep their split."""
    path = work / "split.csv"
    known = {}
    if path.exists():
        with path.open() as fh:
            for r in csv.DictReader(fh):
                known[(r["set"], r["identity"])] = r["split"]
    rng = random.Random(seed)
    for set_name, items in sets.items():
        ids = sorted({i for _, i, _ in items})
        new = [i for i in ids if (set_name, i) not in known]
        rng.shuffle(new)
        have_val = sum(1 for i in ids if known.get((set_name, i)) == "val")
        want_val = max(1, round(len(ids) * val_frac))
        for i in new:
            if have_val < want_val:
                known[(set_name, i)] = "val"
                have_val += 1
            else:
                known[(set_name, i)] = "train"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "identity", "split"])
        for (s, i), sp in sorted(known.items()):
            w.writerow([s, i, sp])
    return known


def prepare(args):
    work = args.work
    data = work / "data"
    if data.exists():
        shutil.rmtree(data)                     # rebuilt every time from the sets
    for sp in ("train", "val"):
        (data / "images" / sp).mkdir(parents=True, exist_ok=True)
        (data / "labels" / sp).mkdir(parents=True, exist_ok=True)

    sets = {}
    for s in args.sets:
        if not (s / "json").is_dir():
            raise SystemExit(f"{s} is not a saved set from save_reviewed.py (no json/ folder)")
        sets[s.name] = load_set(s)
    split = make_split(sets, work, args.val_frac, args.seed)

    counts = Counter()
    rows = []
    for set_name, items in sets.items():
        set_dir = next(s for s in args.sets if s.name == set_name)
        for name, identity, d in items:
            objs = [o for o in d.get("objects", []) if o.get("species") in SPECIES and not o.get("excluded")]
            if not objs:
                counts[f"{set_name}_skipped_no_face"] += 1
                continue
            sp = split[(set_name, identity)]
            with Image.open(set_dir / "images" / name) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
            W, H = img.size
            if args.pad > 0:
                canvas, px, py = pad_image(img, args.pad)
            else:
                canvas, px, py = img, 0, 0
            lines = [l for l in (yolo_line(o, px, py, canvas.width, canvas.height, W, H) for o in objs) if l]
            if not lines:
                counts[f"{set_name}_skipped_bad_box"] += 1
                continue
            flat = f"{set_name}__{name.replace('/', '__')}"
            stem = Path(flat).with_suffix("").name
            canvas.save(data / "images" / sp / f"{stem}.png")
            (data / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            counts[f"{set_name}_{sp}"] += 1
            rows.append((stem, set_name, identity, sp))

    # replay: dataset-1 images, unpadded full photos, train only
    if args.replay:
        rimg, rlbl = args.replay / "images" / "train", args.replay / "labels" / "train"
        if not rimg.is_dir():
            raise SystemExit(f"--replay: {rimg} not found (use the --work folder of train_face_model.py)")
        cand = sorted(p for p in rimg.iterdir() if p.suffix.lower() in IMG_EXT and (rlbl / f"{p.stem}.txt").is_file())
        pick = random.Random(args.seed).sample(cand, min(args.replay_n, len(cand)))
        for p in pick:
            dst = data / "images" / "train" / f"pfd1__{p.name}"
            dst.symlink_to(p.resolve())
            shutil.copy(rlbl / f"{p.stem}.txt", data / "labels" / "train" / f"pfd1__{p.stem}.txt")
        counts["replay_train"] = len(pick)
        # forgetting check: a fixed sample of dataset-1 val
        vimg, vlbl = args.replay / "images" / "val", args.replay / "labels" / "val"
        if vimg.is_dir():
            fv = work / "data_pfd1val"
            if fv.exists():
                shutil.rmtree(fv)
            (fv / "images" / "val").mkdir(parents=True)
            (fv / "labels" / "val").mkdir(parents=True)
            vc = sorted(p for p in vimg.iterdir() if p.suffix.lower() in IMG_EXT and (vlbl / f"{p.stem}.txt").is_file())
            for p in sample_balanced(vc, vlbl, args.replay_val_n, args.seed):
                (fv / "images" / "val" / p.name).symlink_to(p.resolve())
                shutil.copy(vlbl / f"{p.stem}.txt", fv / "labels" / "val" / f"{p.stem}.txt")
            write_yaml(fv)
            (fv / ".balanced").write_text("cats and dogs sampled equally\n")
            counts["pfd1_val_for_forgetting_check"] = len(list((fv / "images" / "val").iterdir()))

    write_yaml(data)
    with (work / "prepared_images.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "set", "identity", "split"])
        w.writerows(rows)
    ids = Counter((s, sp) for (s, _), sp in split.items())
    print("prepared:", dict(sorted(counts.items())))
    print("animals per split:", {f"{s}_{sp}": n for (s, sp), n in sorted(ids.items())})
    print(f"dataset: {data / 'data.yaml'}")
    if not any(k.endswith("_val") for k in counts):
        raise SystemExit("no validation images - review more animals first")
    return counts


def write_yaml(root: Path):
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [3, 3]\nflip_idx: [1, 0, 2]\n"
        "names:\n  0: cat_face\n  1: dog_face\n"
        "# keypoints: 0 eye_left, 1 eye_right (as seen in the image), 2 nose\n")


# --------------------------------------------------------------------------
# 3. eval: old vs new on the same images
# --------------------------------------------------------------------------
def read_gt(lbl: Path, W, H):
    faces = []
    for line in lbl.read_text().splitlines():
        v = [float(x) for x in line.split()]
        if len(v) < 5:
            continue
        xc, yc, w, h = v[1:5]
        kps = [(v[i] * W, v[i + 1] * H, int(v[i + 2])) for i in range(5, len(v) - 2, 3)]
        faces.append({"cls": int(v[0]), "box": [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                      "kps": kps})
    return faces


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


SPECIES_NAMES = {0: "cat", 1: "dog"}


def balanced_images(root: Path, seed, balance=True):
    """Val images of a test set; with balance, the same number of cats and dogs (the larger
    species is randomly reduced, fixed seed). Species = class of the first face in the label."""
    by_sp = defaultdict(list)
    for p in sorted((root / "images" / "val").iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lbl = root / "labels" / "val" / f"{p.stem}.txt"
        first = lbl.read_text().split() if lbl.exists() else []
        by_sp[SPECIES_NAMES.get(int(float(first[0])), "other") if first else "other"].append(p)
    counts = {k: len(v) for k, v in by_sp.items()}
    if not balance or not by_sp.get("cat") or not by_sp.get("dog"):
        return [p for v in by_sp.values() for p in v], counts, False
    n = min(len(by_sp["cat"]), len(by_sp["dog"]))
    rng = random.Random(seed)
    return sorted(rng.sample(by_sp["cat"], n) + rng.sample(by_sp["dog"], n)), counts, True


def metrics(faces, extra):
    """faces: [{sp, found, iou, cls_ok, err{kp: e}}] -> summary numbers."""
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")
    n = len(faces)
    found = [f for f in faces if f["found"]]
    errs = defaultdict(list)
    for f in found:
        for k, e in f["err"].items():
            errs[k].append(e)
    all_err = [e for v in errs.values() for e in v]
    st = Counter(s for f in found for s, _ in f.get("pts", {}).values())
    bad_shown = sum(e > 0.05 for e in all_err)
    gt_visible = st["shown"] + st["hidden"]
    gt_missing = st["extra"] + st["both_missing"]
    out = {
        "faces": n,
        # the work per point: move it (> 5% off), add it (hidden) or delete it (extra)
        "pts_to_fix_pct": 100 * (bad_shown + st["hidden"] + st["extra"]) / max(3 * len(found), 1),
        "pts_hidden_pct": 100 * st["hidden"] / max(gt_visible, 1),
        "pts_extra_pct": 100 * st["extra"] / gt_missing if gt_missing else float("nan"),
        "pts_missing_in_labels": gt_missing,
        "missed_pct": 100 * (n - len(found)) / max(n, 1),
        "box_iou_mean": sum(f["iou"] for f in found) / max(len(found), 1),
        "species_correct_pct": 100 * sum(f["cls_ok"] for f in found) / max(len(found), 1),
        "kp_error_median": med(all_err),
        "kp_within_5pct": 100 * sum(e <= 0.05 for e in all_err) / max(len(all_err), 1),
        "kp_within_10pct": 100 * sum(e <= 0.10 for e in all_err) / max(len(all_err), 1),
        "extra_detections": extra,
    }
    for k in KP:
        out[f"kp_error_median_{k}"] = med(errs[k])
        hid = sum(1 for f in found if f.get("pts", {}).get(k, ("",))[0] == "hidden")
        ext = sum(1 for f in found if f.get("pts", {}).get(k, ("",))[0] == "extra")
        # share of faces where this point needs a fix (moved, added or deleted)
        out[f"kp_bad_{k}"] = 100 * (sum(e > 0.05 for e in errs[k]) + hid + ext) / max(len(found), 1)
    return out


def evaluate(model_path, root: Path, conf, device, imgsz, imgs):
    """Per ground-truth face: matched prediction (best IoU >= 0.3), its IoU, species,
    keypoint error relative to the face size (max of box width/height).
    Returns {"all": ..., "cat": ..., "dog": ...}."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    faces, extra = [], Counter()
    for p in imgs:
        with Image.open(p) as im:
            W, H = im.size
        gt = read_gt(root / "labels" / "val" / f"{p.stem}.txt", W, H)
        r = model.predict(str(p), conf=conf, verbose=False, device=device, imgsz=imgsz)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            kps = r.keypoints.data.cpu().tolist() if r.keypoints is not None else [None] * len(r.boxes)
            for b, c, k in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().int().tolist(), kps):
                preds.append({"box": b, "cls": c, "kps": k})
        used = set()
        img_sp = SPECIES_NAMES.get(gt[0]["cls"], "other") if gt else "other"
        for g in gt:
            sp = SPECIES_NAMES.get(g["cls"], "other")
            best, bi = 0.0, None
            for i, pr in enumerate(preds):
                if i not in used:
                    v = iou(g["box"], pr["box"])
                    if v > best:
                        best, bi = v, i
            if bi is None or best < 0.3:
                faces.append({"sp": sp, "found": False, "iou": 0, "cls_ok": False, "err": {}})
                continue
            used.add(bi)
            pr = preds[bi]
            size = max(g["box"][2] - g["box"][0], g["box"][3] - g["box"][1])
            cx = (pr["box"][0] + pr["box"][2]) / 2
            gt = name_single_eye({k: [gx, gy, int(gv > 0)] for k, (gx, gy, gv) in zip(KP, g["kps"])},
                                 (g["box"][0] + g["box"][2]) / 2)
            q = pr["kps"] or [[0, 0, 0]] * 3
            sh = name_single_eye({k: [p[0], p[1], int((p[2] if len(p) > 2 else 1) >= KP_CONF)]
                                  for k, p in zip(KP, q)}, cx)
            pts = {}
            for k in KP:
                if gt[k][2] and sh[k][2]:
                    pts[k] = ("shown", math.hypot(sh[k][0] - gt[k][0], sh[k][1] - gt[k][1]) / size)
                elif gt[k][2]:
                    pts[k] = ("hidden", None)          # you would have to add it
                elif sh[k][2]:
                    pts[k] = ("extra", None)           # you would have to delete it
                else:
                    pts[k] = ("both_missing", None)
            err = {k: e for k, (st, e) in pts.items() if st == "shown"}
            faces.append({"sp": sp, "found": True, "iou": best, "cls_ok": pr["cls"] == g["cls"],
                          "err": err, "pts": pts})
        extra[img_sp] += len(preds) - len(used)
    res = {"all": metrics(faces, sum(extra.values()))}
    for sp in ("cat", "dog"):
        fs = [f for f in faces if f["sp"] == sp]
        if fs:
            res[sp] = metrics(fs, extra[sp])
    res["all"]["images"] = len(imgs)
    return res


ROWS = [  # key, label, better is higher?, format
    ("pts_to_fix_pct", "points you would have to fix (%)", False, "{:.1f}"),
    ("pts_hidden_pct", "  visible points the model hides (%)", False, "{:.1f}"),
    ("pts_extra_pct", "  points shown that you marked missing (%)", False, "{:.1f}"),
    ("missed_pct", "faces missed (%)", False, "{:.1f}"),
    ("box_iou_mean", "box overlap with the label (IoU)", True, "{:.3f}"),
    ("species_correct_pct", "species correct (%)", True, "{:.1f}"),
    ("kp_error_median", "keypoint error, median (% of face size)", False, "pct"),
    ("kp_error_median_eye_left", "  left eye", False, "pct"),
    ("kp_error_median_eye_right", "  right eye", False, "pct"),
    ("kp_error_median_nose", "  nose", False, "pct"),
    ("kp_within_5pct", "keypoints within 5% of face size (%)", True, "{:.1f}"),
    ("kp_within_10pct", "keypoints within 10% of face size (%)", True, "{:.1f}"),
    ("extra_detections", "extra boxes (no matching face)", False, "{:.0f}"),
]


def fmt(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{100 * v:.1f}%" if f == "pct" else f.format(v)


EVAL_SETS = {
    "pfd1-val": ("val", "Dataset-1 val (forgetting check; the old model was selected on it)"),
    "pfd1-test": ("test", "Dataset-1 test (never seen by either model: the fairest check)"),
    "pfd1-train": ("train", "Dataset-1 train sample (old model trained on these; only the old-vs-new difference counts)"),
}


def sample_balanced(cands, lbl_dir: Path, n, seed):
    """n images, half cats and half dogs where possible (class of the first face in the label)."""
    by = defaultdict(list)
    for p in cands:
        first = (lbl_dir / f"{p.stem}.txt").read_text().split()
        by[int(float(first[0])) if first else -1].append(p)
    rng = random.Random(seed)
    half = n // 2
    pick = []
    for c in (0, 1):
        pick += rng.sample(by[c], min(half, len(by[c])))
    rest = [p for v in by.values() for p in v if p not in set(pick)]
    if len(pick) < n:                                # one species too small: fill up
        pick += rng.sample(rest, min(n - len(pick), len(rest)))
    return pick


def build_eval_set(args, key):
    """Fixed random sample of a dataset-1 split -> <work>/eval_<key>. For train, images
    mixed into the fine-tune (replay) are excluded, so the new model never trained on them."""
    split = EVAL_SETS[key][0]
    out = args.work / f"eval_{key}"
    if key == "pfd1-val" and (args.work / "data_pfd1val" / ".balanced").exists():
        return args.work / "data_pfd1val"               # made during prepare
    if (out / ".balanced").exists() and any((out / "images" / "val").iterdir()):
        return out
    if out.exists():
        shutil.rmtree(out)                           # older, unbalanced sample: rebuild
    if not args.replay:
        return None
    img, lbl = args.replay / "images" / split, args.replay / "labels" / split
    if not img.is_dir():
        print(f"  {key}: {img} not found, skipped")
        return None
    exclude = set()
    if split == "train":
        exclude = {p.name[len("pfd1__"):] for p in (args.work / "data" / "images" / "train").glob("pfd1__*")}
    cand = sorted(p for p in img.iterdir() if p.suffix.lower() in IMG_EXT
                  and p.name not in exclude and (lbl / f"{p.stem}.txt").is_file())
    if not cand:
        return None
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for p in sample_balanced(cand, lbl, args.eval_n, args.seed + 1):
        (out / "images" / "val" / p.name).symlink_to(p.resolve())
        shutil.copy(lbl / f"{p.stem}.txt", out / "labels" / "val" / f"{p.stem}.txt")
    write_yaml(out)
    (out / ".balanced").write_text("cats and dogs sampled equally\n")
    return out


def build_edited_set(args):
    """Reviewed val images where you CORRECTED the model by hand. Unchanged images are the
    old model's own answers, so only corrected ones compare old and new fairly."""
    out = args.work / "eval_reviewed-edited"
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "val").mkdir(parents=True)
    (out / "labels" / "val").mkdir(parents=True)
    val_imgs = args.work / "data" / "images" / "val"
    n = 0
    for s_dir in args.sets:
        for name, _, d in load_set(s_dir):
            if not any(o.get("edited") for o in d.get("objects", [])):
                continue
            stem = Path(f"{s_dir.name}__{name.replace('/', '__')}").with_suffix("").name
            src = val_imgs / f"{stem}.png"
            if src.exists():
                (out / "images" / "val" / src.name).symlink_to(src.resolve())
                shutil.copy(args.work / "data" / "labels" / "val" / f"{stem}.txt", out / "labels" / "val")
                n += 1
    write_yaml(out)
    return out, n


def mark(o, n, higher, f):
    if o is None or n is None or math.isnan(o) or math.isnan(n) or fmt(o, f) == fmt(n, f):
        return " "
    return "+" if (n > o) == higher else "-"


def bad_pct(r):
    """Points you would have to fix: moved (> 5% off), added (hidden) or deleted (extra)."""
    return r.get("pts_to_fix_pct", 100 - r["kp_within_5pct"])


def struggle_lines(r, label):
    """Which animal / which point is hardest for the NEW model on one test set."""
    sps = [sp for sp in ("cat", "dog") if sp in r["new"]]
    if len(sps) < 2:
        return []
    b = {sp: bad_pct(r["new"][sp]) for sp in sps}
    worse, better = sorted(sps, key=lambda sp: -b[sp])
    lines = [f"{label}: points to fix (move / add / delete) cats {b['cat']:.1f}%, dogs {b['dog']:.1f}%; "
             f"faces missed cats {r['new']['cat']['missed_pct']:.1f}%, dogs {r['new']['dog']['missed_pct']:.1f}%"]
    if b[worse] - b[better] >= 1.0:
        lines.append(f"  -> the model struggles more with {worse.upper()}S")
    else:
        lines.append("  -> cats and dogs about equally hard")
    for sp in sps:
        kb = {k: r["new"][sp][f"kp_bad_{k}"] for k in KP}
        hardest = max(kb, key=kb.get)
        lines.append(f"  hardest point for {sp}s: {hardest.replace('_', ' ')} ({kb[hardest]:.1f}% of faces need a fix; "
                     + ", ".join(f"{k.replace('_', ' ')} {v:.1f}%" for k, v in kb.items()) + ")")
    return lines


def compare(args):
    new = args.new_model or (args.work / "runs" / "finetune" / "weights" / "best.pt")
    if not Path(new).exists():
        raise SystemExit(f"new model not found: {new}")
    tests = []
    ed_title = "Reviewed val, only faces you corrected by hand (the fair test on your crops)"
    rv_title = "Reviewed val (your crops, padded like the website)"
    if "reviewed" in args.eval_on:
        root, n_ed = build_edited_set(args)
        if n_ed:
            tests.append((ed_title, root))
        else:
            print("  no hand-corrected images in reviewed val: the fair test is skipped")
        tests.append((rv_title, args.work / "data"))
    for key in [k for k in args.eval_on if k in EVAL_SETS]:
        root = build_eval_set(args, key)
        if root is not None:
            tests.append((EVAL_SETS[key][1], root))
        else:
            print(f"  {key}: needs --replay (the train_face_model.py --work folder), skipped")
    results = {}
    md = [f"# Old vs new labeler model\n\nold: `{args.base_model}`  \nnew: `{new}`  \n"
          f"confidence {args.conf}; a face counts as found if a box overlaps it with IoU >= 0.3. "
          f"{'Every test set has the same number of cats and dogs. ' if not args.no_balance else ''}"
          "`+` = new is better, `-` = new is worse.\n"]
    for title, root in tests:
        imgs, counts, balanced = balanced_images(root, args.seed + 7, not args.no_balance)
        if not imgs:
            continue
        old_r = evaluate(args.base_model, root, args.conf, args.device, args.imgsz, imgs)
        new_r = evaluate(new, root, args.conf, args.device, args.imgsz, imgs)
        results[title] = {"old": old_r["all"], "new": new_r["all"], "by_species": {
            sp: {"old": old_r[sp], "new": new_r[sp]} for sp in ("cat", "dog") if sp in old_r}}
        sps = [sp for sp in ("cat", "dog") if sp in old_r]
        info = (f"{len(imgs)} images: " + ", ".join(f"{old_r[sp]['faces']} {sp}s" for sp in sps)
                + (f" (balanced from {counts.get('cat', 0)} cats, {counts.get('dog', 0)} dogs)" if balanced
                   and counts.get("cat") != counts.get("dog") else ""))
        cols = ["all"] + sps
        print(f"\n{title}\n  {info}")
        print(f"  {'':40s}" + "".join(f"{c:>19s}" for c in cols))
        print(f"  {'':40s}" + "".join(f"{'old':>9s}{'new':>10s}" for _ in cols))
        md.append(f"\n## {title}\n{info}\n\n| | " + " | ".join(f"{c} old | {c} new" for c in cols)
                  + " |\n|---|" + "---:|---:|" * len(cols))
        for key, label, higher, f in ROWS:
            line, mdl = f"  {label:40s}", f"| {label} |"
            for c in cols:
                o, n = (old_r[c].get(key), new_r[c].get(key))
                m = mark(o, n, higher, f)
                line += f"{fmt(o, f):>9s}{fmt(n, f):>9s}{m}"
                mdl += f" {fmt(o, f)} | {fmt(n, f)} {m.strip()} |"
            print(line)
            md.append(mdl)
    (args.work / "eval.json").write_text(json.dumps(results, indent=1))

    # which animal / point is hardest (new model)
    summary = []
    for title in (ed_title, rv_title, EVAL_SETS["pfd1-test"][1]):
        if title in results:
            r = {"new": {sp: results[title]["by_species"][sp]["new"] for sp in results[title]["by_species"]}}
            summary += struggle_lines(r, title.split(" (")[0])
    if summary:
        print("\nWhere the new model struggles")
        print("\n".join("  " + l for l in summary))
        md.append("\n## Where the new model struggles\n" + "\n".join(f"- {l.strip()}" for l in summary))

    mine = {rv_title, ed_title}
    forgot_on = [t for t in results if t not in mine and (
        results[t]["new"]["kp_error_median"] > results[t]["old"]["kp_error_median"] * 1.15
        or results[t]["new"]["missed_pct"] > results[t]["old"]["missed_pct"] + 2)]
    decide = ed_title if (ed_title in results and results[ed_title]["old"]["faces"] >= 20) else rv_title
    if decide in results:
        rv = results[decide]
        # what costs review time: points you have to move (> 5% of the face size) and missed faces
        bad_old, bad_new = bad_pct(rv["old"]), bad_pct(rv["new"])
        good = bad_new <= bad_old * 0.9 and rv["new"]["missed_pct"] <= rv["old"]["missed_pct"] + 0.5
        worse_sp = [sp for sp, v in rv["by_species"].items()
                    if bad_pct(v["new"]) > bad_pct(v["old"]) * 1.1 + 0.5
                    or v["new"]["missed_pct"] > v["old"]["missed_pct"] + 1]
        per_sp = "; ".join(f"{sp}s {bad_pct(v['old']):.1f}% -> {bad_pct(v['new']):.1f}%" for sp, v in rv["by_species"].items())
        txt = (f"Decided on: {decide.split(' (')[0]} ({rv['old']['faces']} faces) | points to fix "
               f"old {bad_old:.1f}% -> new {bad_new:.1f}% ({per_sp})")
        print("\n" + txt)
        md.append(f"\n{txt}. Points to fix = moved (> 5% of face size), added (model hid a visible point) or deleted (model showed a point you marked missing). The new model must cut them by at least 10% overall, without missing more faces "
                  "and without getting clearly worse on cats or on dogs.")
        if good and worse_sp:
            verdict = (f"Keep the old model for now: better overall, but worse on {' and '.join(s + 's' for s in worse_sp)}. "
                       f"Review more {worse_sp[0]}s, then fine-tune again.")
        elif good:
            verdict = ("Use the new model: better on your crops, for cats and dogs" + (
                f", but it got clearly worse on: {'; '.join(t.split(' (')[0] for t in forgot_on)} (forgetting)."
                if forgot_on else ", and no forgetting on dataset-1."))
        else:
            verdict = "Keep the old model: the new one is not better on your crops."
    else:
        verdict = ("Clearly worse on: " + "; ".join(t.split(" (")[0] for t in forgot_on)) if forgot_on \
            else "No clear difference on dataset-1."
    md.append(f"\n**Verdict:** {verdict}\n\nThe reviewed val set is small: differences of a few percent can be chance.\n")
    (args.work / "eval.md").write_text("\n".join(md))
    print(f"\nVerdict: {verdict}\nresults -> {args.work / 'eval.md'}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", type=Path, nargs="+", required=True, help="saved sets from save_reviewed.py")
    ap.add_argument("--base-model", type=Path, required=True, help="the current best.pt to start from")
    ap.add_argument("--work", type=Path, required=True, help="output folder")
    ap.add_argument("--replay", type=Path, default=None, help="train_face_model.py --work folder (dataset-1)")
    ap.add_argument("--replay-n", type=int, default=3000, help="dataset-1 train images mixed in")
    ap.add_argument("--replay-val-n", type=int, default=500, help="dataset-1 val images for the forgetting check")
    ap.add_argument("--pad", type=float, default=0.5, help="gray border, same as the website's --pad")
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of ANIMALS held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.002, help="learning rate (first training used 0.01)")
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (e.g. 10 = backbone)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence used in the evaluation")
    ap.add_argument("--eval-on", nargs="+", default=["reviewed", "pfd1-val", "pfd1-test"],
                    choices=["reviewed", *EVAL_SETS],
                    help="test sets for old vs new (pfd1-train = images the old model trained on, "
                         "excluding those mixed into the fine-tune)")
    ap.add_argument("--eval-n", type=int, default=500, help="images per dataset-1 test set")
    ap.add_argument("--no-balance", action="store_true",
                    help="evaluate on all test images instead of the same number of cats and dogs")
    ap.add_argument("--new-model", type=Path, default=None, help="--eval-only: weights to compare (default: this run)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.base_model.is_file():
        raise SystemExit(f"--base-model not found: {args.base_model}  (is $M set?)")

    if args.eval_only:
        compare(args)
        return
    prepare(args)
    if args.prepare_only:
        return

    from ultralytics import YOLO
    model = YOLO(str(args.base_model))
    model.train(data=str(args.work / "data" / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, lr0=args.lr0, lrf=0.1, optimizer="SGD",
                warmup_epochs=1, freeze=args.freeze or None, close_mosaic=max(1, args.epochs // 4),
                fliplr=0.5, project=str((args.work / "runs").resolve()), name="finetune", exist_ok=True)
    compare(args)
    print(f"\nnew model: {args.work / 'runs' / 'finetune' / 'weights' / 'best.pt'}")
    print("to use it in the website: M=<that path> and restart app.py")


if __name__ == "__main__":
    main()
