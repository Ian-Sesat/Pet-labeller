#!/usr/bin/env python3
"""
Label quality checker for pet-face labels
(cat / dog face box + keypoints eye_left, eye_right, nose).

Idea: learn from a trusted labeled set (dataset-1 train) what normal labels
look like (the TEMPLATE), then score any other labeled set against it and
rank the images whose labels look wrong.

Commands
  build-template   learn the template from a labeled set   -> template.json
  check            score a labeled set against the template -> report.html + CSVs

Input formats (--format)
  pfd1     pet face detection dataset-1 folder (images/<subset>, labels/<subset>);
           species from file names; choose subsets with --subsets
  yolo     YOLO-pose folder (images/..., labels/...) with classes 0 cat_face,
           1 dog_face and keypoints eye_left, eye_right, nose
           (train_face_model.py --work folder, or the labeler's _export)
  labeler  labeler output: --path = labels folder, --images = images folder
  master   folder with annotations.jsonl from build_pet_face_dataset.py
  csv      CSV from autolabel_folders.py run (--path = the CSV, --images = the cat/ or dog/
           folder); species comes from the folder; images without a face count as wrong

Examples
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  python quality_check.py build-template --format pfd1 \
      --path $D/pet_face_detection_dataset_1 --subsets train --out $D/quality/template
  python quality_check.py check --format pfd1 --path $D/pet_face_detection_dataset_1 \
      --subsets val test --template $D/quality/template/template.json \
      --out $D/quality/pfd1_valtest --selftest
  python quality_check.py check --format labeler --path $D/new_labels --images $D/new_images \
      --template $D/quality/template/template.json --out $D/quality/new_labels
  python quality_check.py check --format csv --path $D/petface_labels/cat.csv --images $D/cat \
      --template $D/quality/template/template.json --out $D/quality/cat

Per face it computes box shape, keypoint positions relative to the box, eye
distance, eye tilt and nose position relative to the eyes. The template keeps
robust statistics (median, MAD, percentiles) and a joint shape model
(Mahalanobis distance) per species. A face is flagged when it breaks a hard
rule or its shape distance is above the template's 99th percentile.
Needs: numpy, pillow  (+ build_pet_face_dataset.py in the same folder for --format pfd1)
"""

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

KP = ["eye_left", "eye_right", "nose"]
SHORT = {"eye_left": "el", "eye_right": "er", "nose": "n"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIES = ["cat", "dog"]
FACE_CROPS = False        # set for --format csv: PetFace images are face crops, a full-frame box is normal

# feature name -> readable label
FEATURES = {
    "aspect": "box width / height",
    "log_area": "box size (log10 share of image)",
    "u_el": "left eye x in box", "v_el": "left eye y in box",
    "u_er": "right eye x in box", "v_er": "right eye y in box",
    "u_n": "nose x in box", "v_n": "nose y in box",
    "eye_dist": "eye distance / box width",
    "eye_tilt": "eye line tilt (degrees)",
    "nose_drop": "nose below eyes / eye distance",
    "nose_offset": "nose sideways / eye distance",
}
JOINT = ["aspect", "u_el", "v_el", "u_er", "v_er", "u_n", "v_n"]   # joint shape model
KP_JOINT = ["eye_tilt", "nose_drop", "nose_offset"]                # box-free model (--crops)
CROPS = False      # --crops: face crops, the box is cut by the image edge -> judge keypoints only
EXPLAIN = [f for f in FEATURES if f != "log_area"]                 # used for reasons
FULL = list(FEATURES)                                              # needed for template use

ERROR_FLAGS = {
    "box_degenerate": "box has no size",
    "eyes_coincide": "both eyes at the same spot",
    "eyes_too_far_apart": "eyes further apart than the box is wide",
    "eyes_swapped": "left and right eye swapped",
    "duplicate_box": "second box on the same face",
    "no_face_label": "no face box (the model found no face)",
    "eye_left_outside_box": "left eye outside the box",
    "eye_right_outside_box": "right eye outside the box",
    "nose_outside_box": "nose outside the box",
    "eye_left_outside_image": "left eye outside the image",
    "eye_right_outside_image": "right eye outside the image",
    "nose_outside_image": "nose outside the image",
}
INFO_FLAGS = {          # shown as information only: no score, never makes an image suspicious
    "nose_above_eyes": "nose above the eyes: head tilted or upside down (judged upright)",
}
WARN_FLAGS = {
    "box_outside_image": "box goes past the image edge",
    "box_covers_whole_image": "box covers almost the whole image",
    "box_tiny": "box smaller than 12 px",
    "no_keypoints": "no keypoints",
    "species_unknown": "species unknown",
    "image_in_template": "same image is in the template set (leakage)",
    "duplicate_image": "same image appears twice in this set",
    "blurry": "image is very blurry",
    "too_dark_or_bright": "image very dark or very bright",
    "image_tiny": "image smaller than 64 px",
    "low_confidence": "model was unsure about this face (low score)",
    "multiple_faces": "model found more than one face",
    "species_mismatch": "model thinks this is the other species than the folder says",
    "keypoint_unsure": "a keypoint was placed with low confidence",
}


# --------------------------------------------------------------------------
# Readers -> list of faces
#   face = {image, path, W, H, species, box [x1,y1,x2,y2] px (unclipped),
#           kps {eye_left/eye_right/nose: [x, y, v]} px or None, origin, idx, pre_flags}
# --------------------------------------------------------------------------
def image_size(path):
    with Image.open(path) as im:
        return im.size


def pts_from_norm(kps, W, H):
    return [[x * W, y * H, int(round(v))] if v > 0 else [0.0, 0.0, 0] for x, y, v in kps]


def sort_eyes(eyes, nose, box):
    missing = [0.0, 0.0, 0]
    vis = [e for e in eyes if e and e[2] > 0]
    if len(vis) == 2:
        return sorted(vis, key=lambda e: e[0])
    if len(vis) == 1:
        ref = nose[0] if nose and nose[2] > 0 else (box[0] + box[2]) / 2
        return [vis[0], missing] if vis[0][0] <= ref else [missing, vis[0]]
    return [missing, missing]


def canonical_face(kps_px, box, order):
    """order 'pfd1' = nose, eye, eye (eyes get sorted); 'canonical' = eye_left, eye_right, nose."""
    flags = []
    if not kps_px or len(kps_px) < 3:
        return None, flags
    if order == "pfd1":
        nose = kps_px[0]
        el, er = sort_eyes(kps_px[1:3], nose, box)
    else:
        el, er, nose = kps_px[:3]
        if el[2] > 0 and er[2] > 0 and el[0] > er[0]:
            flags.append("eyes_swapped")
    return {"eye_left": el, "eye_right": er, "nose": nose}, flags


def read_pfd1(root: Path, subsets, xml_dir):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from build_pet_face_dataset import load_voc_species, parse_yolo_pose, pfd1_flip_copies, pfd1_origin
    except ImportError:
        sys.exit("--format pfd1 needs build_pet_face_dataset.py in the same folder")
    xml = load_voc_species(xml_dir) if xml_dir else None
    flips = pfd1_flip_copies(root)
    faces, n_img = [], 0
    for subset in subsets:
        d = root / "images" / subset
        if not d.is_dir():
            print(f"missing {d}, skipped")
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            parsed = None
            for cand in (root / "labels" / subset / f"{img.stem}.txt", root / "labels" / subset / img.name):
                if cand.is_file():
                    parsed = parse_yolo_pose(cand)
                    if parsed is not None:
                        break
            if parsed is None:
                continue
            n_img += 1
            origin, sp = pfd1_origin(img.stem, subset, xml)
            W, H = image_size(img)
            for i, (_, b, kps) in enumerate(parsed):
                xc, yc, w, h = b
                box = [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
                kp, fl = canonical_face(pts_from_norm(kps, W, H), box, "pfd1")
                copy_of = flips.get((subset, img.stem))
                if copy_of:
                    fl = fl + ["flip_copy"]
                faces.append(dict(image=f"{subset}/{img.name}", path=str(img), W=W, H=H,
                                  species=sp or "unknown", box=box, kps=kp, origin=origin,
                                  idx=i, pre_flags=fl,
                                  copy_of="/".join(copy_of) if copy_of else None))
    return faces, n_img


def read_yolo(root: Path):
    faces, n_img = [], 0
    img_root = root / "images"
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or not img.is_file():
            continue
        rel = img.relative_to(img_root)
        lbl = root / "labels" / rel.with_suffix(".txt")
        if not lbl.is_file():
            continue
        n_img += 1
        W, H = image_size(img)
        for i, line in enumerate(l for l in lbl.read_text().splitlines() if l.strip()):
            v = [float(x) for x in line.split()]
            cls, (xc, yc, w, h) = int(v[0]), v[1:5]
            kps = [tuple(v[j:j + 3]) for j in range(5, len(v) - 2, 3)]
            box = [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
            kp, fl = canonical_face(pts_from_norm(kps, W, H), box, "canonical")
            faces.append(dict(image=rel.as_posix(), path=str(img), W=W, H=H,
                              species=SPECIES[cls] if cls in (0, 1) else "unknown", box=box,
                              kps=kp, origin=rel.parts[0] if len(rel.parts) > 1 else "",
                              idx=i, pre_flags=fl))
    return faces, n_img


def read_labeler(labels: Path, images: Path):
    if images is None:
        sys.exit("--format labeler needs --images (the labeler's images folder)")
    faces, n_img = [], 0
    for jf in sorted(labels.rglob("*.json")):
        if "_export" in jf.parts or jf.name.startswith("_"):
            continue
        d = json.loads(jf.read_text())
        if d.get("status", "none") == "none":
            continue
        name = d.get("image") or jf.relative_to(labels).as_posix()[:-5]
        path = images / name
        if not path.is_file():
            continue
        n_img += 1
        W, H = d.get("width"), d.get("height")
        if not W or not H:
            W, H = image_size(path)
        for i, o in enumerate(d.get("objects", [])):
            if o.get("excluded"):
                continue
            k = o.get("keypoints")
            kp, fl = canonical_face([k[n] for n in KP], o["bbox_xyxy"], "canonical") if k else (None, [])
            faces.append(dict(image=name, path=str(path), W=W, H=H,
                              species=o.get("species") or "unknown", box=list(o["bbox_xyxy"]),
                              kps=kp, origin=d.get("status", ""), idx=i, pre_flags=fl))
    return faces, n_img


def read_master(root: Path):
    faces, n_img = [], 0
    for line in (root / "annotations.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_img += 1
        for i, o in enumerate(r["objects"]):
            if o.get("excluded"):
                continue
            k = o.get("keypoints")
            kp, fl = canonical_face([k[n] for n in KP], o["bbox_xyxy"], "canonical") if k else (None, [])
            faces.append(dict(image=r["image_id"], path=str(root / r["file"]), W=r["width"],
                              H=r["height"], species=o.get("species") or "unknown",
                              box=list(o["bbox_xyxy"]), kps=kp, origin=r.get("origin_dataset", ""),
                              idx=i, pre_flags=fl))
    return faces, n_img


def read_autolabel_csv(csv_path: Path, root: Path):
    """CSV written by autolabel_folders.py: one row per image, species from the folder."""
    if root is None:
        sys.exit("--format csv needs --images (the cat/ or dog/ folder the CSV belongs to)")
    flag_map = {"low_conf": "low_confidence", "multi_face": "multiple_faces",
                "species_mismatch": "species_mismatch"}
    faces, n_img = [], 0
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            n_img += 1
            W, H = int(float(r["width"])), int(float(r["height"]))
            sp = r["species"] if r["species"] in SPECIES else "unknown"
            pre = [flag_map[x] for x in (r.get("flag") or "").split("|") if x in flag_map]
            base = dict(image=r["file"], path=str(root / r["file"]), W=W, H=H, species=sp,
                        origin=sp, idx=0)
            if not r.get("x1"):
                faces.append(dict(base, box=[0, 0, 0, 0], kps=None, pre_flags=["no_face_label"]))
                continue
            kps = {k: [float(r[f"{k}_x"] or 0), float(r[f"{k}_y"] or 0), int(float(r[f"{k}_v"] or 0))]
                   for k in KP}
            if any(v[2] == 1 for v in kps.values()):
                pre.append("keypoint_unsure")
            box = [float(r[c]) for c in ("x1", "y1", "x2", "y2")]
            kp, fl = canonical_face([kps[k] for k in KP], box, "canonical")
            faces.append(dict(base, box=box, kps=kp, pre_flags=pre + fl))
    return faces, n_img


def read_faces(args):
    p = Path(args.path)
    if args.format == "pfd1":
        faces, n = read_pfd1(p, args.subsets, args.species_xml_dir)
    elif args.format == "yolo":
        faces, n = read_yolo(p)
    elif args.format == "labeler":
        faces, n = read_labeler(p, Path(args.images) if args.images else None)
    elif args.format == "csv":
        faces, n = read_autolabel_csv(p, Path(args.images) if args.images else None)
    else:
        faces, n = read_master(p)
    if getattr(args, "species", None):
        changed = sum(f["species"] != args.species for f in faces)
        for f in faces:
            f["species"] = args.species
        print(f"--species {args.species}: species set on all faces ({changed} changed)")
    print(f"read {n} labeled images, {len(faces)} faces ({args.format}: {p})")
    return faces, n


# --------------------------------------------------------------------------
# Features + rules
# --------------------------------------------------------------------------
def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def analyze(f):
    """Fill f['feat'] (features) and f['flags'] (rule names)."""
    flags, feat = list(f["pre_flags"]), {}
    if f["species"] not in SPECIES:
        flags.append("species_unknown")
    if "no_face_label" in flags:
        f["cbox"] = f["box"]
        f["feat"], f["flags"] = feat, flags
        return
    x1, y1, x2, y2 = f["box"]
    W, H = f["W"], f["H"]
    if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
        flags.append("box_outside_image")
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    bw, bh = cx2 - cx1, cy2 - cy1
    f["cbox"] = [cx1, cy1, cx2, cy2]
    if bw < 2 or bh < 2:
        flags.append("box_degenerate")
        f["feat"], f["flags"] = feat, flags
        return
    feat["aspect"] = bw / bh
    feat["log_area"] = math.log10(bw * bh / (W * H))
    if bw * bh / (W * H) > 0.95 and not FACE_CROPS:
        flags.append("box_covers_whole_image")
    if min(bw, bh) < 12:
        flags.append("box_tiny")
    k = f["kps"]
    if not k:
        flags.append("no_keypoints")
        f["feat"], f["flags"] = feat, flags
        return
    vis = {n: k[n] for n in KP if k.get(n) and k[n][2] > 0}
    # a face may have 1, 2 or 3 points: missing points are not a problem
    eyes_seen = [n for n in ("eye_left", "eye_right") if n in vis]
    if len(eyes_seen) == 1:
        # a single eye may be labeled left OR right: judge it by where it sits
        e = vis.pop(eyes_seen[0])
        ref = vis["nose"][0] if "nose" in vis else (cx1 + cx2) / 2
        vis["eye_left" if e[0] <= ref else "eye_right"] = e
    for n, p in vis.items():
        if not (0 <= p[0] <= W and 0 <= p[1] <= H):
            flags.append(f"{n}_outside_image")
        if (p[0] < cx1 - 0.05 * bw or p[0] > cx2 + 0.05 * bw
                or p[1] < cy1 - 0.05 * bh or p[1] > cy2 + 0.05 * bh):
            flags.append(f"{n}_outside_box")
        feat["u_" + SHORT[n]] = (p[0] - cx1) / bw
        feat["v_" + SHORT[n]] = (p[1] - cy1) / bh
    if "eye_left" in vis and "eye_right" in vis:
        el, er = vis["eye_left"], vis["eye_right"]
        dx, dy = er[0] - el[0], er[1] - el[1]
        d = math.hypot(dx, dy)
        feat["eye_dist"] = d / bw
        if feat["eye_dist"] < 0.03:
            flags.append("eyes_coincide")
        elif feat["eye_dist"] > 1.0:
            flags.append("eyes_too_far_apart")
        if d > 0:
            feat["eye_tilt"] = math.degrees(math.atan2(dy, dx))
            if "nose" in vis:
                nx, ny = vis["nose"][0] - (el[0] + er[0]) / 2, vis["nose"][1] - (el[1] + er[1]) / 2
                ux, uy = dx / d, dy / d                # along the eye line
                feat["nose_offset"] = (nx * ux + ny * uy) / d
                feat["nose_drop"] = (nx * -uy + ny * ux) / d   # perpendicular, + = below
                if feat["nose_drop"] < 0 and "eyes_swapped" not in flags:
                    # head tilted / upside down: allowed. Judge the upright equivalent
                    # (flipped top-to-bottom) so a correct label is not called unusual.
                    flags.append("nose_above_eyes")
                    feat["nose_drop"] = -feat["nose_drop"]
                    feat["eye_tilt"] = -feat["eye_tilt"]
                    for kk in ("v_el", "v_er", "v_n"):
                        if kk in feat:
                            feat[kk] = 1 - feat[kk]
    f["feat"], f["flags"] = feat, flags


def analyze_all(faces):
    by_img = defaultdict(list)
    for f in faces:
        analyze(f)
        by_img[f["image"]].append(f)
    for fs in by_img.values():                 # duplicate boxes on one face
        for i in range(len(fs)):
            for j in range(i):
                if iou(fs[i]["box"], fs[j]["box"]) > 0.7:
                    fs[i]["flags"].append("duplicate_box")
                    break
    return by_img


def n_errors(f):
    return sum(fl in ERROR_FLAGS for fl in f["flags"])


# --------------------------------------------------------------------------
# Image-level checks (hash, blur, brightness)
# --------------------------------------------------------------------------
def sha1_file(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def image_quality(path):
    """(laplacian variance on a 512 px gray copy, mean brightness, min side)."""
    with Image.open(path) as im:
        side = min(im.size)
        g = im.convert("L")
        g.thumbnail((512, 512))
        a = np.asarray(g, dtype=np.float32)
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var()), float(a.mean()), side


def same_as_template_source(args, template):
    """True when the checked set overlaps the set the template was built from
    (then 'image is in the template set' is expected and not leakage)."""
    src = (template or {}).get("source", {})
    if not src or src.get("format") != args.format:
        return False
    try:
        same_path = Path(src["path"]).resolve() == Path(args.path).resolve()
    except OSError:
        return False
    if not same_path:
        return False
    if args.format == "pfd1":
        return bool(set(src.get("subsets") or []) & set(args.subsets))
    return True


def image_checks(by_img, args, template=None):
    """Returns {image: {'sha1', 'blur', 'bright', 'side', 'flags'}}."""
    info = {}
    paths = {img: fs[0]["path"] for img, fs in by_img.items()}
    for n, (img, path) in enumerate(paths.items(), 1):
        if n % 2000 == 0:
            print(f"  image checks {n}/{len(paths)}")
        r = {"flags": []}
        try:
            if not args.no_hash:
                r["sha1"] = sha1_file(path)
            if args.image_quality:
                r["blur"], r["bright"], r["side"] = image_quality(path)
        except Exception as e:  # noqa: BLE001
            r["flags"].append("unreadable")
            print(f"  cannot read {path}: {e}")
        info[img] = r
    if not args.no_hash:
        seen = {}
        for img, r in info.items():
            h = r.get("sha1")
            if h and h in seen:
                r["flags"].append("duplicate_image")
            elif h:
                seen[h] = img
        if template and template.get("sha1") and same_as_template_source(args, template):
            print("  checking the template's own images: leakage check skipped (duplicates still checked)")
        elif template and template.get("sha1"):
            tset = set(template["sha1"])
            for r in info.values():
                if r.get("sha1") in tset:
                    r["flags"].append("image_in_template")
    if args.image_quality:
        blur_thr = (template or {}).get("image", {}).get("blur_p1") or 30.0
        for r in info.values():
            if "blur" not in r:
                continue
            if r["blur"] < blur_thr:
                r["flags"].append("blurry")
            if r["bright"] < 25 or r["bright"] > 235:
                r["flags"].append("too_dark_or_bright")
            if r["side"] < 64:
                r["flags"].append("image_tiny")
    return info


# --------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------
def robust(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    std = X.std(axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1e-6))
    return med, scale


def build_template(args):
    faces, n_img = read_faces(args)
    by_img = analyze_all(faces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T = {"version": 1, "created": dt.datetime.now().isoformat(timespec="seconds"),
         "source": {"format": args.format, "path": str(args.path), "subsets": args.subsets,
                    "images": n_img, "faces": len(faces)},
         "features": FEATURES, "joint": JOINT, "species": {}}
    issues = []                                    # template faces that look wrong

    for sp in SPECIES:
        fs = [f for f in faces if f["species"] == sp and "flip_copy" not in f["flags"]]
        usable = [f for f in fs if not n_errors(f) and all(k in f["feat"] for k in FULL)]
        if len(usable) < 30:
            print(f"{sp}: only {len(usable)} usable faces, skipped")
            continue
        X = np.array([[f["feat"][k] for k in FULL] for f in usable])
        ex = [FULL.index(k) for k in EXPLAIN]
        med, scale = robust(X)
        z = np.abs((X - med) / scale)[:, ex]
        keep = z.max(axis=1) <= 5
        for f, zz, kk in zip(usable, z, keep):
            if not kk:
                worst = EXPLAIN[int(zz.argmax())]
                issues.append((f, f"outlier: {FEATURES[worst]} = {f['feat'][worst]:.2f} "
                                  f"(z {zz.max():.1f})"))
        Xk = X[keep]
        med, scale = robust(Xk)
        z2 = np.abs((Xk - med) / scale)[:, ex]
        core = Xk[z2.max(axis=1) <= 4]
        ji = [FULL.index(k) for k in JOINT]
        J = core[:, ji]
        mu = J.mean(axis=0)
        C = np.cov(J, rowvar=False) + np.eye(len(JOINT)) * 1e-6
        Cinv = np.linalg.pinv(C)
        D = Xk[:, ji] - mu
        d2 = np.einsum("ij,jk,ik->i", D, Cinv, D)
        ki = [FULL.index(k) for k in KP_JOINT]
        mu_k = core[:, ki].mean(axis=0)
        Ck_inv = np.linalg.pinv(np.cov(core[:, ki], rowvar=False) + np.eye(len(KP_JOINT)) * 1e-6)
        Dk = Xk[:, ki] - mu_k
        d2k = np.einsum("ij,jk,ik->i", Dk, Ck_inv, Dk)
        pct = {k: {str(q): float(np.percentile(Xk[:, i], q)) for q in (1, 5, 25, 50, 75, 95, 99)}
               for i, k in enumerate(FULL)}
        hist = {}
        for i, k in enumerate(FULL):
            lo, hi = np.percentile(Xk[:, i], [0.5, 99.5])
            pad = (hi - lo) * 0.15 or 0.01
            edges = np.linspace(lo - pad, hi + pad, 41)
            cnt, _ = np.histogram(Xk[:, i], bins=edges)
            hist[k] = {"edges": edges.round(5).tolist(), "density": (cnt / cnt.sum()).round(5).tolist()}
        vis = Counter()
        for f in fs:
            if f["kps"]:
                for n in KP:
                    vis[f"{n}_{['missing', 'unsure', 'visible'][min(f['kps'][n][2], 2)]}"] += 1
        T["species"][sp] = {
            "faces": len(fs), "usable": len(usable), "used": int(keep.sum()),
            "removed_outliers": int((~keep).sum()),
            "median": dict(zip(FULL, med.round(5).tolist())),
            "scale": dict(zip(FULL, scale.round(6).tolist())),
            "percentiles": pct, "hist": hist,
            "joint_mean": mu.round(6).tolist(), "joint_cov_inv": Cinv.round(6).tolist(),
            "d2_thresholds": {str(q): float(np.percentile(d2, q)) for q in (95, 99, 99.9)},
            "kp_joint_mean": mu_k.round(6).tolist(), "kp_joint_cov_inv": Ck_inv.round(6).tolist(),
            "kp_d2_thresholds": {str(q): float(np.percentile(d2k, q)) for q in (95, 99, 99.9)},
            "visibility": dict(vis),
        }
        print(f"{sp}: {len(fs)} faces, {len(usable)} usable, {int(keep.sum())} in template, "
              f"{int((~keep).sum())} outliers removed")

    if not T["species"]:
        sys.exit("no species with enough faces - is the species known from the file names?")
    for f in faces:
        if "flip_copy" in f["flags"]:
            continue
        if n_errors(f) or f["species"] not in SPECIES:
            issues.append((f, "; ".join(ERROR_FLAGS.get(x, WARN_FLAGS.get(x, INFO_FLAGS.get(x, x))) for x in f["flags"])))

    if not args.no_hash or args.image_quality:
        info = image_checks(by_img, args)
        if not args.no_hash:
            T["sha1"] = sorted({r["sha1"] for r in info.values() if r.get("sha1")})
        if args.image_quality:
            blurs = [r["blur"] for r in info.values() if "blur" in r]
            T["image"] = {"blur_p1": float(np.percentile(blurs, 1)),
                          "blur_median": float(np.median(blurs))}

    (out / "template.json").write_text(json.dumps(T))
    with (out / "template_issues.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "face_index", "species", "origin", "path", "issue"])
        for f, why in issues:
            w.writerow([f["image"], f["idx"], f["species"], f["origin"], f["path"], why])
    print(f"template -> {out / 'template.json'}")
    print(f"{len(issues)} suspicious faces in the template set itself -> {out / 'template_issues.csv'}")
    write_template_card(T, out / "template_card.html")
    print(f"template card -> {out / 'template_card.html'}")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def shape_distance(feat, t):
    keys, pre = (KP_JOINT, "kp_") if CROPS else (JOINT, "")
    if not all(k in feat for k in keys):
        return None
    d = np.array([feat[k] for k in keys]) - np.array(t[pre + "joint_mean"])
    d2 = float(d @ np.array(t[pre + "joint_cov_inv"]) @ d)
    return d2 / t[pre + "d2_thresholds"]["99"]     # >= 1 means beyond the 99th percentile


def score_face(f, T):
    """Adds f['shape'] (x of the 99% limit), f['ref'], f['reasons'], f['score']."""
    cands = [f["species"]] if f["species"] in T["species"] else list(T["species"])
    best_sp, best = cands[0], None
    for sp in cands:
        s = shape_distance(f["feat"], T["species"][sp])
        if s is not None and (best is None or s < best):
            best_sp, best = sp, s
    t = T["species"][best_sp]
    reasons = [ERROR_FLAGS[x] for x in f["flags"] if x in ERROR_FLAGS]
    if best is not None and best >= 1:
        what = "keypoint layout" if CROPS else "face shape"
        reasons.append(f"unusual {what} ({best:.1f}x the normal limit)")
    explain = KP_JOINT if CROPS else EXPLAIN
    zs = {k: (f["feat"][k] - t["median"][k]) / t["scale"][k] for k in explain if k in f["feat"]}
    for k, z in sorted(zs.items(), key=lambda kv: -abs(kv[1])):
        if abs(z) > 3.5:
            p = t["percentiles"][k]
            reasons.append(f"{FEATURES[k]} {f['feat'][k]:.2f} (normal {p['1']:.2f} to {p['99']:.2f})")
    reasons += [WARN_FLAGS[x] for x in f["flags"] if x in WARN_FLAGS]
    reasons += [INFO_FLAGS[x] for x in f["flags"] if x in INFO_FLAGS]
    if "flip_copy" in f["flags"]:
        reasons.insert(0, f"upside-down copy of {f.get('copy_of')} (augmentation)")
    if f["species"] not in SPECIES:
        reasons.append(f"compared with the {best_sp} template")
    errs = n_errors(f)
    warns = sum(x in WARN_FLAGS for x in f["flags"])
    f["shape"], f["ref"], f["z"], f["reasons"] = best, best_sp, zs, reasons
    f["score"] = round((best or 0) + 2 * errs + 0.5 * warns, 3)
    f["flagged"] = bool(errs) or (best is not None and best >= 1)


def check(args):
    T = json.loads(Path(args.template).read_text())
    if CROPS and any("kp_joint_mean" not in t for t in T["species"].values()):
        sys.exit("this template is older than --crops: rebuild it once with build-template")
    faces, n_img = read_faces(args)
    by_img = analyze_all(faces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = image_checks(by_img, args, T) if (not args.no_hash or args.image_quality) else {}

    for f in faces:
        score_face(f, T)
    images = []
    for img, fs in by_img.items():
        iflags = info.get(img, {}).get("flags", [])
        reasons = []
        for f in fs:
            prefix = f"face {f['idx'] + 1}: " if len(fs) > 1 else ""
            reasons += [prefix + r for r in f["reasons"]]
        reasons += [WARN_FLAGS.get(x, x) for x in iflags]
        score = max(f["score"] for f in fs) + 0.5 * len(iflags)
        images.append({"image": img, "path": fs[0]["path"], "faces": fs, "score": round(score, 3),
                       "flagged": any(f["flagged"] for f in fs), "reasons": reasons,
                       "origin": fs[0]["origin"], "iflags": iflags, "info": info.get(img, {})})
    for r in images:
        r["category"] = categorize(r)
    images.sort(key=lambda r: (CAT_ORDER.get(r["category"], 99), -r["score"]))
    names = {c[0]: c[1] for c in CATEGORIES}

    with (out / "suspicious_images.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "category", "image", "score", "flagged", "faces", "origin", "path", "reasons"])
        for i, r in enumerate(images, 1):
            w.writerow([i, names.get(r["category"], "ok"), r["image"], r["score"], r["flagged"],
                        len(r["faces"]), r["origin"], r["path"], " | ".join(r["reasons"])])
    with (out / "faces.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "face_index", "species", "origin", "score", "flagged", "shape_vs_limit",
                    "flags"] + FULL + [f"z_{k}" for k in EXPLAIN])
        for f in faces:
            w.writerow([f["image"], f["idx"], f["species"], f["origin"], f["score"], f["flagged"],
                        "" if f["shape"] is None else round(f["shape"], 3), "|".join(f["flags"])]
                       + [round(f["feat"][k], 4) if k in f["feat"] else "" for k in FULL]
                       + [round(f["z"][k], 2) if k in f["z"] else "" for k in EXPLAIN])

    st = selftest(faces, T, args.selftest_n) if args.selftest else None
    if args.format == "csv":
        lst = out / "suspicious_files.txt"
        lst.write_text("".join(r["image"] + "\n" for r in images if r["category"]))
        names_ = {c[0]: c[1] for c in CATEGORIES}
        (out / "suspicious_quality.json").write_text(json.dumps(
            {r["image"]: {"score": r["score"], "category": names_[r["category"]], "reasons": r["reasons"]}
             for r in images if r["category"]}, indent=1))
        print(f"suspicious file list (for autolabel_folders.py review --files) -> {lst}")
    if args.format == "labeler":
        q = {r["image"]: {"score": r["score"], "category": names[r["category"]], "reasons": r["reasons"]}
             for r in images if r["category"]}
        (Path(args.path) / "_quality.json").write_text(json.dumps(q, indent=1))
        print(f"labeler 'Needs check' list -> {Path(args.path) / '_quality.json'}")

    write_report(T, faces, images, n_img, args, out, st)
    cats = Counter(r["category"] for r in images if r["category"])
    print(f"\n{n_img} images, {len(faces)} faces, {sum(cats.values())} suspicious images:")
    for key, name, _ in CATEGORIES:
        print(f"  {name:26s} {cats.get(key, 0)}")
    print(f"report -> {out / 'report.html'}")
    print(f"ranked list -> {out / 'suspicious_images.csv'}")


# --------------------------------------------------------------------------
# Categories: every suspicious image lands in exactly one, worst first
# --------------------------------------------------------------------------
CATEGORIES = [
    ("wrong", "Wrong labels",
     "A hard rule is broken: a keypoint outside its box or the image, the nose above the eyes, both eyes "
     "on one spot, eyes swapped, or two boxes on one face. These are almost always labeling mistakes."),
    ("shape", "Unusual face shape",
     "Box and keypoints together are beyond the 99% limit of the template. Usually a misplaced point or a "
     "box that is too big, too small or shifted; sometimes a correct but extreme pose (side view, head tilt)."),
    ("borderline", "Borderline values",
     "The overall shape is inside the limit, but at least one measurement is far from normal "
     "(more than 3.5x the usual spread). Worth a quick look."),
    ("data", "Image and data problems",
     "The face labels look fine, but the image or annotation has a problem: missing keypoints, unknown "
     "species, box past the image edge, duplicate or leaked images, blurry, dark or tiny images."),
    ("copy", "Upside-down copies (augmentation)",
     "Vertically flipped copies of another image in the dataset (<name>_1, box mirrored top to bottom). "
     "Not real data; skipped by train_face_model.py and build_pet_face_dataset.py. Their errors "
     "(nose above the eyes) come from the flip, not from labeling."),
]
CAT_ORDER = {c[0]: i for i, c in enumerate(CATEGORIES)}


def categorize(r):
    fs = r["faces"]
    if any("flip_copy" in f["flags"] for f in fs):
        return "copy"
    if any(n_errors(f) for f in fs):
        return "wrong"
    if any(f["shape"] is not None and f["shape"] >= 1 for f in fs):
        return "shape"
    if any(abs(z) > 3.5 for f in fs for z in f["z"].values()):
        return "borderline"
    if r["iflags"] or any(x in WARN_FLAGS for f in fs for x in f["flags"]):
        return "data"
    return None


# --------------------------------------------------------------------------
# Self-test: break clean labels on purpose, see if the checker notices
# --------------------------------------------------------------------------
def corruptions():
    def nose_side(f, bw, bh):
        f["kps"]["nose"][0] += 0.3 * bw

    def nose_up(f, bw, bh):
        e = f["kps"]
        e["nose"][1] = (e["eye_left"][1] + e["eye_right"][1]) / 2 - 0.05 * bh

    def eye_up(f, bw, bh):
        f["kps"]["eye_left"][1] -= 0.25 * bh

    def eye_on_other(f, bw, bh):
        e = f["kps"]
        e["eye_left"] = [e["eye_right"][0] - 0.02 * bw, e["eye_right"][1], 2]

    def box_scale(k):
        def fn(f, bw, bh):
            x1, y1, x2, y2 = f["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            f["box"] = [cx - bw * k / 2, cy - bh * k / 2, cx + bw * k / 2, cy + bh * k / 2]
        return fn

    def box_shift(f, bw, bh):
        x1, y1, x2, y2 = f["box"]
        f["box"] = [x1 + 0.3 * bw, y1, x2 + 0.3 * bw, y2]

    return {
        "nose moved sideways 30%": nose_side,
        "nose moved onto the eye line": nose_up,
        "left eye moved up 25%": eye_up,
        "left eye placed on right eye": eye_on_other,
        "box 1.6x too big": box_scale(1.6),
        "box 0.7x too small": box_scale(0.7),
        "box shifted 30% sideways": box_shift,
    }


def selftest(faces, T, n):
    clean = [f for f in faces if f["species"] in T["species"] and not n_errors(f)
             and "flip_copy" not in f["flags"] and all(k in f["feat"] for k in FULL)]
    if len(clean) < 20:
        print("self-test skipped: not enough clean faces")
        return None
    rng = random.Random(0)
    # upper bound: rule-clean faces can still hold real shape errors, which count here too
    false_alarm = sum(f["flagged"] for f in clean) / len(clean)
    base = [f for f in clean if not f["flagged"]]       # break only faces that look fine now
    sample = rng.sample(base, min(n, len(base)))
    rows = []
    for name, fn in corruptions().items():
        hits = 0
        for f in sample:
            g = json.loads(json.dumps({k: f[k] for k in ("image", "path", "W", "H", "species", "box",
                                                         "kps", "origin", "idx", "pre_flags")}))
            x1, y1, x2, y2 = f["cbox"]
            fn(g, x2 - x1, y2 - y1)
            analyze(g)
            score_face(g, T)
            hits += g["flagged"]
        rows.append((name, hits / len(sample)))
    print(f"self-test on {len(sample)} faces; flagged among rule-clean faces: {100 * false_alarm:.1f}% "
          f"(upper bound for false alarms, includes real shape errors)")
    for name, r in rows:
        print(f"  caught {100 * r:5.1f}%  {name}")
    return {"n": len(sample), "false_alarm": false_alarm, "rows": rows}


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
CSS = """
:root{--bg:#f7f7f5;--card:#fff;--text:#1d1f22;--muted:#6b7078;--line:#e3e3df;--tpl:#c9ccd1;--cur:#0c8599;
--bad:#c92a2a;--warn:#e67700;--ok:#2b8a3e}
@media (prefers-color-scheme:dark){:root{--bg:#141619;--card:#1c1f23;--text:#e7e9ec;--muted:#9aa0a6;
--line:#2e333a;--tpl:#4a5058;--cur:#3bc9db;--bad:#ff8787;--warn:#ffa94d;--ok:#69db7c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.big{font-size:26px;font-weight:650}table{border-collapse:collapse;width:100%;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
td,th{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.4px}
td.num{text-align:right;font-variant-numeric:tabular-nums}.shift-hi{color:var(--bad);font-weight:600}
.shift-mid{color:var(--warn)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.item img{width:100%;display:block;background:#000}.item .b{padding:8px 10px}
.item ul{margin:4px 0 0;padding-left:18px}.item li{margin:1px 0}.score{font-weight:650}
.err{color:var(--bad)}a{color:var(--cur)}h2 .muted{font-weight:400;font-size:14px}code{font-size:12px;word-break:break-all;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:5px;margin-right:14px}
.sw{width:14px;height:10px;display:inline-block;border-radius:2px}
.tw{overflow-x:auto;border-radius:10px;margin-bottom:6px}.tw table{min-width:640px}
"""


def svg_hist(h, values=None, w=220, ht=54):
    edges, dens = h["edges"], h["density"]
    lo, hi = edges[0], edges[-1]
    cur = None
    if values is not None and len(values):
        cnt, _ = np.histogram(np.clip(values, lo, hi), bins=edges)
        cur = cnt / max(cnt.sum(), 1)
    top = max(max(dens), max(cur) if cur is not None else 0) or 1
    bw = w / len(dens)
    bars = "".join(
        f'<rect x="{i * bw:.1f}" y="{ht - d / top * ht:.1f}" width="{bw:.1f}" height="{d / top * ht:.1f}" '
        f'fill="var(--tpl)"/>' for i, d in enumerate(dens))
    line = ""
    if cur is not None:
        pts = " ".join(f"{(i + .5) * bw:.1f},{ht - c / top * ht:.1f}" for i, c in enumerate(cur))
        line = f'<polyline points="{pts}" fill="none" stroke="var(--cur)" stroke-width="2"/>'
    return f'<svg width="{w}" height="{ht}" viewBox="0 0 {w} {ht}">{bars}{line}</svg>'


def feature_table(T, faces=None):
    rows = []
    for sp, t in T["species"].items():
        vals = {k: np.array([f["feat"][k] for f in (faces or []) if f["species"] == sp and k in f["feat"]])
                for k in FULL}
        n_cur = sum(1 for f in (faces or []) if f["species"] == sp)
        rows.append(f'<h3>{sp.capitalize()} faces <span class="muted">· template {t["used"]} faces'
                    f'{f" · checked {n_cur} faces" if faces is not None else ""}</span></h3>')
        head = "<tr><th>feature</th><th>template median</th><th>normal range (1–99%)</th>"
        if faces is not None:
            head += "<th>checked median</th><th>shift</th>"
        head += "<th>distribution</th></tr>"
        body = ""
        for k in FULL:
            p = t["percentiles"][k]
            cells = (f"<td>{FEATURES[k]}</td><td class='num'>{t['median'][k]:.3f}</td>"
                     f"<td class='num'>{p['1']:.3f} – {p['99']:.3f}</td>")
            v = vals[k] if faces is not None else None
            if faces is not None:
                if len(v):
                    sh = (float(np.median(v)) - t["median"][k]) / t["scale"][k]
                    cls = "shift-hi" if abs(sh) > 1 else "shift-mid" if abs(sh) > 0.5 else ""
                    cells += f"<td class='num'>{np.median(v):.3f}</td><td class='num {cls}'>{sh:+.2f}</td>"
                else:
                    cells += "<td class='num'>–</td><td class='num'>–</td>"
            cells += f"<td>{svg_hist(t['hist'][k], v)}</td>"
            body += f"<tr>{cells}</tr>"
        rows.append(f"<table>{head}{body}</table>")
    return "".join(rows)


def page(title, body):
    body = body.replace("<table>", "<div class='tw'><table>").replace("</table>", "</table></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def write_template_card(T, path):
    s = T["source"]
    body = (f"<h1>Label template</h1><p class='muted'>Built {T['created']} from {s['format']} "
            f"<code>{html.escape(s['path'])}</code> {', '.join(s.get('subsets') or [])} · "
            f"{s['images']} images, {s['faces']} faces</p><div class='cards'>")
    for sp, t in T["species"].items():
        body += (f"<div class='card'><div class='muted'>{sp} faces in template</div>"
                 f"<div class='big'>{t['used']}</div><div class='muted'>{t['removed_outliers']} outliers removed, "
                 f"{t['faces'] - t['usable']} with rule errors or missing points</div></div>")
    body += ("</div><h2>What a normal label looks like</h2><p class='muted'>Keypoint positions are relative "
             "to the face box (0 = left/top edge, 1 = right/bottom edge). Left/right as seen in the image.</p>")
    body += feature_table(T)
    Path(path).write_text(page("Label template", body))


def draw_thumb(r, path, size=420):
    with Image.open(r["path"]) as im:
        f0 = r["faces"][0]
        if im.size != (f0["W"], f0["H"]):
            t = ImageOps.exif_transpose(im)
            if t.size == (f0["W"], f0["H"]):
                im = t
        im = im.convert("RGB")
        k = min(size / im.width, size / im.height, 1.0)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))))
    d = ImageDraw.Draw(im)
    for f in r["faces"]:
        sx, sy = im.width / f["W"], im.height / f["H"]
        x1, y1, x2, y2 = f["box"]
        col = (230, 73, 128) if f["flagged"] else ((255, 159, 67) if f["species"] == "cat" else
                                                    (77, 171, 247) if f["species"] == "dog" else (173, 181, 189))
        d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=col, width=3 if f["flagged"] else 2)
        d.text((x1 * sx + 3, max(y1 * sy - 12, 0)), f"{f['idx'] + 1} {f['species']}", fill=col)
        if f["kps"]:
            for n, c in (("eye_left", (81, 207, 102)), ("eye_right", (255, 107, 107)), ("nose", (255, 212, 59))):
                x, y, v = f["kps"][n]
                if v:
                    rr = 4
                    d.ellipse([x * sx - rr, y * sy - rr, x * sx + rr, y * sy + rr],
                              fill=c if v == 2 else None, outline=(0, 0, 0) if v == 2 else c, width=1 if v == 2 else 2)
    im.save(path, quality=85)


def write_report(T, faces, images, n_img, args, out, st):
    thumbs = out / "thumbs"
    thumbs.mkdir(exist_ok=True)
    flagged = [r for r in images if r["flagged"]]
    real = [f for f in faces if "flip_copy" not in f["flags"]]          # copies have their own group
    rule_counts = Counter(x for f in real for x in f["flags"])
    img_flag_counts = Counter(x for r in images for x in r["iflags"])
    sp_counts = Counter(f["species"] for f in faces)
    origin_rows = defaultdict(lambda: [0, 0])
    for r in images:
        origin_rows[r["origin"]][0] += 1
        origin_rows[r["origin"]][1] += r["flagged"]

    body = (f"<h1>Label quality report</h1><p class='muted'>{dt.datetime.now():%Y-%m-%d %H:%M} · "
            f"{args.format} <code>{html.escape(str(args.path))}</code> "
            f"{' '.join(args.subsets) if args.format == 'pfd1' else ''} · template "
            f"<code>{html.escape(str(args.template))}</code></p>")
    body += "<div class='cards'>"
    body += f"<div class='card'><div class='muted'>images checked</div><div class='big'>{n_img}</div></div>"
    body += f"<div class='card'><div class='muted'>faces</div><div class='big'>{len(faces)}</div>" \
            f"<div class='muted'>{', '.join(f'{k} {v}' for k, v in sp_counts.most_common())}</div></div>"
    susp = [r for r in images if r["category"]]
    body += f"<div class='card'><div class='muted'>suspicious images</div><div class='big err'>{len(susp)}</div>" \
            f"<div class='muted'>{100 * len(susp) / max(n_img, 1):.1f}% · all listed below</div></div>"
    nerr = sum(1 for f in real if n_errors(f))
    body += f"<div class='card'><div class='muted'>faces breaking a hard rule</div><div class='big'>{nerr}</div></div>"
    body += "</div>"

    if st:
        body += ("<h2>Self-test: does the checker catch broken labels?</h2><p class='muted'>Clean faces from "
                 f"this set were broken on purpose ({st['n']} faces per test). Flagged among faces without "
                 f"rule errors: <b>{100 * st['false_alarm']:.1f}%</b>. That is the upper bound for false "
                 f"alarms (by design about 1%); the rest are real shape problems.</p>"
                 "<table><tr><th>deliberate error</th>"
                 "<th>caught</th></tr>")
        for name, rate in st["rows"]:
            body += f"<tr><td>{name}</td><td class='num'>{100 * rate:.0f}%</td></tr>"
        body += "</table>"

    body += "<h2>Rule problems</h2>"
    if rule_counts or img_flag_counts:
        body += "<table><tr><th>problem</th><th>type</th><th>count</th></tr>"
        for k, v in (rule_counts + img_flag_counts).most_common():
            kind = "error" if k in ERROR_FLAGS else "info" if k in INFO_FLAGS else "warning"
            body += (f"<tr><td>{ERROR_FLAGS.get(k) or WARN_FLAGS.get(k) or INFO_FLAGS.get(k, k)}</td>"
                     f"<td class='{'err' if kind == 'error' else 'muted'}'>{kind}</td><td class='num'>{v}</td></tr>")
        body += "</table>"
    else:
        body += "<p class='muted'>None.</p>"

    if len(origin_rows) > 1:
        body += "<h2>By source</h2><table><tr><th>source</th><th>images</th><th>flagged</th><th>%</th></tr>"
        for o, (n, fl) in sorted(origin_rows.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
            body += (f"<tr><td>{html.escape(str(o) or '–')}</td><td class='num'>{n}</td>"
                     f"<td class='num'>{fl}</td><td class='num'>{100 * fl / max(n, 1):.1f}</td></tr>")
        body += "</table>"

    body += ("<h2>Checked labels vs template</h2><p class='legend muted'>"
             "<span><i class='sw' style='background:var(--tpl)'></i>template</span>"
             "<span><i class='sw' style='background:var(--cur)'></i>checked set</span>"
             "Shift = difference of medians in units of the template spread; above 0.5 means this set is "
             "labeled systematically differently.</p>")
    if CROPS:
        body += ("<p class='card'><b>Crop mode.</b> These images are face crops, so the box is cut by the "
                 "image edge and the box rows below are shown for information only. Faces are judged on "
                 "the keypoint layout: eye line tilt, nose below the eyes and nose sideways, plus the hard "
                 "rules.</p>")
    body += feature_table(T, faces)

    groups = {key: [r for r in susp if r["category"] == key] for key, _, _ in CATEGORIES}
    body += ("<h2 id='list'>All suspicious images</h2><p class='muted'>Every suspicious image, in four groups "
             "from wrong labels to abnormal images, worst first inside each group. Pink box = flagged face. "
             "Points: green left eye, red right eye, yellow nose (hollow = unsure). Same order as "
             "<code>suspicious_images.csv</code>.</p><table><tr><th>group</th><th>images</th><th>meaning</th></tr>")
    for key, name, desc in CATEGORIES:
        body += (f"<tr><td><a href='#{key}'>{name}</a></td><td class='num'>{len(groups[key])}</td>"
                 f"<td class='muted'>{desc}</td></tr>")
    body += "</table>"
    limit = args.top or None
    rank, total = 0, len(susp)
    for key, name, desc in CATEGORIES:
        rows = groups[key]
        shown = rows[:limit]
        body += (f"<h2 id='{key}'>{name} <span class='muted'>· {len(rows)} images"
                 f"{f', first {len(shown)} shown' if len(shown) < len(rows) else ''}</span></h2>"
                 f"<p class='muted'>{desc} <a href='#list'>back to overview</a></p>")
        if not rows:
            body += "<p class='muted'>None.</p>"
            continue
        body += "<div class='grid'>"
        for r in shown:
            rank += 1
            if rank % 500 == 0:
                print(f"  thumbnails {rank}/{total}")
            tname = f"{rank:05d}.jpg"
            try:
                draw_thumb(r, thumbs / tname)
                img_tag = f"<img src='thumbs/{tname}' loading='lazy' alt=''>"
            except Exception:  # noqa: BLE001
                img_tag = "<div class='b muted'>image not readable</div>"
            reasons = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"]) or "<li class='muted'>–</li>"
            body += (f"<div class='item'>{img_tag}<div class='b'><span class='score'>#{rank} · score "
                     f"{r['score']:.2f}</span><ul>{reasons}</ul><code>{html.escape(r['image'])}</code>"
                     f"<br><code>{html.escape(str(r['origin']))}</code></div></div>")
        body += "</div>"
    (out / "report.html").write_text(page("Label quality report", body))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--format", choices=["pfd1", "yolo", "labeler", "master", "csv"], required=True)
        p.add_argument("--path", required=True, help="dataset folder (labels folder for --format labeler)")
        p.add_argument("--images", default=None, help="images folder (--format labeler / csv)")
        p.add_argument("--subsets", nargs="+", default=["train"], help="pfd1 subsets, e.g. train / val test")
        p.add_argument("--species-xml-dir", default=None,
                       help="pfd1: Kaggle 'Dog and Cat Detection' annotations/ to know Cats_Test species")
        p.add_argument("--out", required=True, help="output folder")
        p.add_argument("--no-hash", action="store_true", help="skip duplicate / leakage check (faster)")
        p.add_argument("--image-quality", action="store_true", help="also check blur and brightness (slower)")
        p.add_argument("--species", choices=["cat", "dog"], default=None,
                       help="all images show this species (e.g. the cat/ folder): overrides the labels")

    b = sub.add_parser("build-template", help="learn the template from a trusted labeled set")
    common(b)
    c = sub.add_parser("check", help="score a labeled set against the template")
    common(c)
    c.add_argument("--template", required=True, help="template.json from build-template")
    c.add_argument("--top", type=int, default=0,
                   help="max images shown per group in the report (default 0 = all)")
    c.add_argument("--selftest", action="store_true", help="break clean labels on purpose and measure detection")
    c.add_argument("--selftest-n", type=int, default=1000)
    c.add_argument("--crops", action="store_true",
                   help="face crops: judge the keypoint layout only, not the box (box is cut by the image)")
    args = ap.parse_args()
    global FACE_CROPS, CROPS
    CROPS = getattr(args, "crops", False)
    FACE_CROPS = args.format == "csv" or CROPS
    if args.cmd == "build-template":
        build_template(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
