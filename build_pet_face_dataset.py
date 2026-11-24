#!/usr/bin/env python3
"""
Master pet-face dataset builder  (cat_face / dog_face + 3 facial keypoints)

ONE master annotation file holds all information about every image and face,
independent of where it came from. Training formats (YOLO-pose, COCO, face
crops for re-identification) are exported from it later, so new datasets can
be added at any time without redoing existing labels.

Output folder
  pet_faces/
    images/<source>/<image_id>.<ext>   copied images (symlinks with --link)
    annotations.jsonl                  MASTER FILE: one JSON object per image
    boxes.csv                          same data flattened, one row per face
    sources.json                       registry of added datasets (url, license, counts)
    README.md                          field descriptions (written automatically)
    review/needs_review.csv            faces to check by hand (fill column "decision")
    review/crops/                      small face crops for quick checking
    build_log.txt

Commands
  add           add a source dataset (images already in the dataset are skipped)
  apply-review  write your decisions from review/needs_review.csv back
  stats         print an overview

Examples
  pip install pillow                      # required
  pip install torch transformers          # only for --clip (cat/dog check)
  pip install imagehash                   # optional: perceptual hash for near-duplicates

  python build_pet_face_dataset.py add --source pfd1 \
      --src-root /media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941/pet_face_detection_dataset_1 \
      --out      /media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941/pet_faces \
      --species-xml-dir /path/to/kaggle_dog_and_cat_detection/annotations   # optional

Species comes from the file name. Cats_Test* files (used for cats AND dogs)
are resolved with the original Kaggle "Dog and Cat Detection" XML files
(--species-xml-dir). Anything still unknown goes to review. --clip is an
optional extra check, not needed.
  python build_pet_face_dataset.py stats        --out .../pet_faces
  python build_pet_face_dataset.py apply-review --out .../pet_faces

Adding another dataset later
  1. write an adapter function like adapter_pfd1() that yields images with
     boxes (normalized YOLO xc, yc, w, h) and, if available, nose/eye points
  2. register it in ADAPTERS and describe it in SOURCES
  3. run `add --source <name>`; exact duplicates of existing images are skipped
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SCHEMA_VERSION = 1
CLASS_NAMES = ["cat_face", "dog_face"]           # class_id = index
SPECIES_CLASS = {"cat": 0, "dog": 1}
KEYPOINT_NAMES = ["eye_left", "eye_right", "nose"]  # left/right AS SEEN IN THE IMAGE
FLIP_IDX = [1, 0, 2]                              # for horizontal-flip augmentation
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------
# Source registry: everything you need to cite / license-check later
# --------------------------------------------------------------------------
SOURCES = {
    "pfd1": {
        "name": "pet face detection dataset-1",
        "author": "Chi Chon, Ng",
        "url": "https://zenodo.org/records/7604865",
        "doi": "10.5281/zenodo.7604865",
        "license": "CC BY 4.0 (annotations). Images keep the terms of their origin datasets.",
        "origin_datasets": [
            "Tsinghua Dogs (Zou et al., 2020)",
            "AnimalWeb (Khan et al., CVPR 2020)",
            "Kaggle Cat Dataset (Crawford & Nain, 2018)",
            "Kaggle Dog and Cat Detection (Larxel, 2020; Oxford-IIIT Pet subset)",
            "Kashtanka (kashtanka_test subset)",
        ],
        "original_label_format": "YOLO-pose: class xc yc w h, then 3 keypoints (x y v): nose, eye, eye",
        "original_classes": {"0": "pet face (cat and dog not separated)"},
    },
}


# --------------------------------------------------------------------------
# Adapters: one per source dataset. Each yields a standard dict per image:
#   path, rel_path, subset, label_path (None = no label file found),
#   origin_dataset, objects=[{bbox_yolo, nose, eyes, species_hint, original_class}]
# Coordinates are normalized (0..1). nose = (x, y, v); eyes = [(x, y, v), (x, y, v)]
# --------------------------------------------------------------------------
def parse_yolo_pose(path: Path):
    """Return [(class, [xc,yc,w,h], [(x,y,v), ...]), ...]; [] for an empty file;
    None if the file is not a readable YOLO label file."""
    try:
        text = path.read_bytes().decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    result = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) < 5:
            return None
        rest = nums[5:]
        if len(rest) % 3 == 0:
            kps = [tuple(rest[i:i + 3]) for i in range(0, len(rest), 3)]
        elif len(rest) % 2 == 0:
            kps = [(rest[i], rest[i + 1], 2.0) for i in range(0, len(rest), 2)]
        else:
            return None
        result.append((int(nums[0]), nums[1:5], kps))
    return result


def load_voc_species(xml_dir):
    """{file stem: 'cat'|'dog'} from Pascal-VOC XML files (Kaggle Dog and Cat
    Detection: annotations/Cats_TestN.xml with <object><name>cat|dog</name>)."""
    import xml.etree.ElementTree as ET
    result, mixed = {}, 0
    for x in Path(xml_dir).glob("*.xml"):
        try:
            names = {o.findtext("name", "").strip().lower()
                     for o in ET.parse(x).getroot().iter("object")}
        except ET.ParseError:
            continue
        names &= {"cat", "dog"}
        if len(names) == 1:
            result[x.stem] = names.pop()
        elif len(names) > 1:
            mixed += 1
    print(f"species from XML: {len(result)} files ({mixed} with both cat and dog ignored)")
    return result


def pfd1_origin(stem: str, subset: str, xml_species=None):
    """Guess origin dataset and species from the file name (pfd1 naming).
    Copies with an extra suffix (teddy_n161908_1, 00000282_026_1) use the base name."""
    origin, species = _pfd1_origin(stem, subset, xml_species)
    m = re.fullmatch(r"(.+)_\d+", stem)
    if origin == "unknown" and m:
        origin, species = _pfd1_origin(m.group(1), subset, xml_species)
    return origin, species


def _pfd1_origin(stem: str, subset: str, xml_species=None):
    if subset == "kashtanka_test":
        return "Kashtanka", None
    if re.fullmatch(r"\d+_\d+", stem):                 # 00000801_010
        return "Kaggle Cat Dataset", "cat"
    if re.search(r"_n\d+$", stem):                     # Norfolk_terrier_n111050
        return "Tsinghua Dogs", "dog"
    if stem.startswith("Cats_Test"):                   # used for cats AND dogs
        return "Kaggle Dog and Cat Detection", (xml_species or {}).get(stem)
    m = re.fullmatch(r"([A-Za-z]+)_\d+", stem)         # feralcat_28, bolognesedog_3
    if m:
        word = m.group(1).lower()
        if word.endswith("cat"):
            return "AnimalWeb", "cat"
        if word.endswith("dog"):
            return "AnimalWeb", "dog"
        return "AnimalWeb", None
    return "unknown", None


def _first_box(root: Path, subset: str, stem: str):
    try:
        line = (root / "labels" / subset / f"{stem}.txt").read_text().split("\n")[0].split()
        return [float(v) for v in line[1:5]]
    except (OSError, ValueError, IndexError):
        return None


def pfd1_flip_copies(root: Path):
    """pfd1 contains upside-down copies of images, named <original>_1, whose box is the
    original's box mirrored vertically (y -> 1 - y). Returns {(subset, stem): (subset, original stem)}.
    Kashtanka is skipped: there '_1' means another photo of the same animal."""
    where = {}
    for p in (root / "images").glob("*/*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.parent.name != "kashtanka_test":
            where[p.stem] = p.parent.name
    copies = {}
    for stem, sub in where.items():
        if not stem.endswith("_1") or stem[:-2] not in where:
            continue
        base, bsub = stem[:-2], where[stem[:-2]]
        a, b = _first_box(root, sub, stem), _first_box(root, bsub, base)
        if (a and b and abs(a[0] - b[0]) < 1e-3 and abs(a[1] - (1 - b[1])) < 1e-3
                and abs(a[2] - b[2]) < 1e-3 and abs(a[3] - b[3]) < 1e-3):
            copies[(sub, stem)] = (bsub, base)
    return copies


def adapter_pfd1(src_root: Path, opts):
    xml_species = load_voc_species(opts.species_xml_dir) if opts.species_xml_dir else None
    img_root, lbl_root = src_root / "images", src_root / "labels"
    if not img_root.is_dir():
        sys.exit(f"{img_root} not found - is --src-root the dataset folder?")
    flips = {} if getattr(opts, "keep_flip_copies", False) else pfd1_flip_copies(src_root)
    if flips:
        print(f"skipping {len(flips)} upside-down copies (<name>_1 files, augmentation baked into pfd1)")
    for img in sorted(img_root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXT:
            continue
        subset = img.parent.name
        if (subset, img.stem) in flips:
            continue
        label_path, parsed = None, None
        # normal case: x.txt; broken case in pfd1: label saved as x.jpeg
        for cand in (lbl_root / subset / f"{img.stem}.txt", lbl_root / subset / img.name):
            if cand.is_file():
                parsed = parse_yolo_pose(cand)
                if parsed is not None:
                    label_path = cand
                    break
        origin, species = pfd1_origin(img.stem, subset, xml_species)
        objects = []
        for cls, box, kps in parsed or []:
            objects.append({
                "bbox_yolo": box,
                "nose": kps[0] if len(kps) >= 1 else None,
                "eyes": list(kps[1:3]) if len(kps) >= 3 else [],
                "species_hint": species,
                "original_class": cls,
            })
        yield {
            "path": img,
            "rel_path": img.relative_to(src_root).as_posix(),
            "subset": subset,
            "label_path": label_path.relative_to(src_root).as_posix() if label_path else None,
            "origin_dataset": origin,
            "objects": objects,
        }


ADAPTERS = {"pfd1": adapter_pfd1}


# --------------------------------------------------------------------------
# Standardizing one face
# --------------------------------------------------------------------------
def _kp_to_px(kp, W, H, flags):
    if kp is None:
        return None
    x, y, v = kp
    v = int(round(v))
    if v <= 0:
        return [0.0, 0.0, 0]
    px, py = x * W, y * H
    if not (0 <= px <= W and 0 <= py <= H):
        flags.add("keypoint_outside_image")
        px, py = min(max(px, 0), W), min(max(py, 0), H)
    return [round(px, 1), round(py, 1), v]


def build_object(o, W, H, source):
    flags = set()
    xc, yc, w, h = o["bbox_yolo"]
    raw = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
    x1, y1 = max(raw[0], 0.0), max(raw[1], 0.0)
    x2, y2 = min(raw[2], float(W)), min(raw[3], float(H))
    if any(abs(a - b) > 0.5 for a, b in zip(raw, (x1, y1, x2, y2))):
        flags.add("box_clipped")
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None                                    # degenerate box

    keypoints, eye_ratio = None, None
    nose = _kp_to_px(o.get("nose"), W, H, flags)
    eyes = [_kp_to_px(e, W, H, flags) for e in (o.get("eyes") or [])]
    if nose is not None or eyes:
        missing = [0.0, 0.0, 0]
        left = right = missing
        vis = [e for e in eyes if e and e[2] > 0]
        if len(vis) == 2:
            left, right = sorted(vis, key=lambda e: e[0])
        elif len(vis) == 1:
            ref_x = nose[0] if nose and nose[2] > 0 else (x1 + x2) / 2
            if vis[0][0] <= ref_x:
                left = vis[0]
            else:
                right = vis[0]
        keypoints = {"eye_left": left, "eye_right": right, "nose": nose or missing}

        if left[2] > 0 and right[2] > 0:
            dist = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5
            eye_ratio = round(dist / (x2 - x1), 3)     # ~0.3-0.5 frontal, ~0 profile
            if eye_ratio < 0.05:
                flags.add("eyes_almost_identical")
            if nose[2] > 0 and nose[1] < min(left[1], right[1]):
                flags.add("nose_above_eyes")
        else:
            flags.add("eye_not_visible")

    return {
        "class_id": None,
        "class_name": None,
        "species": None,
        "species_source": None,
        "species_rule": o.get("species_hint"),
        "clip_cat_prob": None,
        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "bbox_yolo": [round((x1 + x2) / 2 / W, 6), round((y1 + y2) / 2 / H, 6),
                      round((x2 - x1) / W, 6), round((y2 - y1) / H, 6)],
        "keypoints": keypoints,
        "eye_distance_ratio": eye_ratio,
        "box_source": source,
        "keypoint_source": source if keypoints else None,
        "original_class": o.get("original_class"),
        "flags": sorted(flags),
        "excluded": False,
        "review_reasons": [],
    }


def decide_species(o, accept):
    """Combine the filename rule and CLIP into a final species + review reasons."""
    if o["species_source"] == "manual":
        return
    rule, p, reasons = o["species_rule"], o["clip_cat_prob"], []
    if p is None:
        species, src = (rule, "filename") if rule else ("unknown", None)
        if not rule:
            reasons.append("species_unknown")
    else:
        clip_sp = "cat" if p >= 0.5 else "dog"
        conf = max(p, 1 - p)
        if rule == clip_sp:
            species, src = rule, "filename+clip"
        elif rule:
            species, src = rule, "filename"
            reasons.append(f"filename_{rule}_but_clip_{clip_sp}")
        elif conf >= accept:
            species, src = clip_sp, "clip"
        else:
            species, src = "unknown", None
            reasons.append("clip_unsure")
    o["species"], o["species_source"] = species, src
    o["class_id"] = SPECIES_CLASS.get(species)
    o["class_name"] = CLASS_NAMES[o["class_id"]] if o["class_id"] is not None else None
    o["review_reasons"] = reasons


def update_review_state(rec):
    reasons = list(rec.get("record_review_reasons", []))
    for i, o in enumerate(rec["objects"]):
        if not o["excluded"]:
            reasons += [f"face{i}:{r}" for r in o["review_reasons"]]
    rec["needs_review"] = bool(reasons)


# --------------------------------------------------------------------------
# CLIP cat/dog check on the face crop
# --------------------------------------------------------------------------
def face_crop(img: Image.Image, box, margin=0.2):
    x1, y1, x2, y2 = box
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    return img.crop((max(0, x1 - mw), max(0, y1 - mh),
                     min(img.width, x2 + mw), min(img.height, y2 + mh)))


def run_clip(records, out: Path, model_name, batch_size=64):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    def as_tensor(x, key):
        if torch.is_tensor(x):
            return x
        for attr in (key, "pooler_output"):
            if getattr(x, attr, None) is not None:
                return getattr(x, attr)
        raise TypeError(f"unexpected CLIP output: {type(x)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CLIP {model_name} on {dev}")
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    prompts = {
        "cat": ["a photo of a cat's face", "a close-up photo of a cat", "a photo of a kitten's face"],
        "dog": ["a photo of a dog's face", "a close-up photo of a dog", "a photo of a puppy's face"],
    }
    with torch.no_grad():
        t = proc(text=prompts["cat"] + prompts["dog"], return_tensors="pt", padding=True).to(dev)
        tf = as_tensor(model.get_text_features(**t), "text_embeds")
        tf = tf / tf.norm(dim=-1, keepdim=True)
        n = len(prompts["cat"])
        cls = torch.stack([tf[:n].mean(0), tf[n:].mean(0)])
        cls = cls / cls.norm(dim=-1, keepdim=True)

    jobs = [(r, o) for r in records for o in r["objects"]]
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        crops = []
        for r, o in chunk:
            with Image.open(out / r["file"]) as im:
                crops.append(face_crop(im.convert("RGB"), o["bbox_xyxy"]))
        with torch.no_grad():
            px = proc(images=crops, return_tensors="pt").to(dev)
            f = as_tensor(model.get_image_features(**px), "image_embeds")
            f = f / f.norm(dim=-1, keepdim=True)
            p_cat = (100 * f @ cls.T).softmax(-1)[:, 0].tolist()
        for (r, o), p in zip(chunk, p_cat):
            o["clip_cat_prob"] = round(p, 4)
        done = min(start + batch_size, len(jobs))
        if done % (batch_size * 20) < batch_size or done == len(jobs):
            print(f"  CLIP {done}/{len(jobs)}")


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def load_db(out: Path):
    db = {}
    path = out / "annotations.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                db[r["image_id"]] = r
    return db


def save_db(out: Path, db):
    tmp = out / "annotations.jsonl.tmp"
    with tmp.open("w") as f:
        for r in db.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out / "annotations.jsonl")


def write_boxes_csv(out: Path, db):
    cols = ["image_id", "file", "width", "height", "source", "source_subset", "origin_dataset",
            "face_index", "class_id", "class_name", "species", "species_source", "species_rule",
            "clip_cat_prob", "x1", "y1", "x2", "y2",
            "eye_left_x", "eye_left_y", "eye_left_v", "eye_right_x", "eye_right_y", "eye_right_v",
            "nose_x", "nose_y", "nose_v", "eye_distance_ratio", "flags", "excluded", "needs_review"]
    with (out / "boxes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in db.values():
            for i, o in enumerate(r["objects"]):
                kp = o["keypoints"] or {}
                kp_vals = [v for name in KEYPOINT_NAMES for v in (kp.get(name) or ["", "", ""])]
                w.writerow([r["image_id"], r["file"], r["width"], r["height"], r["source"],
                            r["source_subset"], r["origin_dataset"], i, o["class_id"],
                            o["class_name"], o["species"], o["species_source"], o["species_rule"],
                            o["clip_cat_prob"], *o["bbox_xyxy"], *kp_vals,
                            o["eye_distance_ratio"], "|".join(o["flags"]), o["excluded"],
                            r["needs_review"]])


def write_review(out: Path, db):
    rdir = out / "review"
    cdir = rdir / "crops"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in db.values():
        if not r["needs_review"]:
            continue
        for i, o in enumerate(r["objects"]):
            if o["excluded"] or not (o["review_reasons"] or r.get("record_review_reasons")):
                continue
            crop_rel = f"review/crops/{r['image_id']}__face{i}.jpg"
            if not (out / crop_rel).exists():
                with Image.open(out / r["file"]) as im:
                    c = face_crop(im.convert("RGB"), o["bbox_xyxy"])
                    c.thumbnail((224, 224))
                    c.save(out / crop_rel, quality=90)
            rows.append([r["image_id"], i, crop_rel, r["file"], r["origin_dataset"],
                         o["species"], o["species_rule"], o["clip_cat_prob"],
                         "|".join(r.get("record_review_reasons", []) + o["review_reasons"]), ""])
    with (rdir / "needs_review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "face_index", "crop", "file", "origin_dataset", "species_now",
                    "species_from_filename", "clip_cat_prob", "reasons",
                    "decision (cat / dog / drop / ok)"])
        w.writerows(rows)
    return len(rows)


README = """# Pet face master dataset

Master file: `annotations.jsonl` (one JSON object per image). Everything else
(`boxes.csv`, YOLO / COCO exports, face crops) is generated from it.

Classes: {classes}
Keypoints (pixel coordinates, `[x, y, v]`, v: 0 = missing, 1 = occluded, 2 = visible):
{keypoints} - left/right **as seen in the image**. Horizontal flip index: {flip}

## Image record
| field | meaning |
|---|---|
| image_id | unique id: `<source>_<subset>_<original file stem>` |
| file | path of the image, relative to this folder |
| width, height | image size in pixels (as stored, before EXIF rotation) |
| mode, format | PIL color mode (RGB, RGBA, L ...) and file format |
| exif_orientation | EXIF orientation tag (1 = normal; other values are flagged for review) |
| sha1 | exact-duplicate check across all sources |
| phash | perceptual hash for near-duplicates (if `imagehash` is installed) |
| source | dataset key in `sources.json` (license, url, citation) |
| source_subset | folder in the source dataset (e.g. train / val / test) |
| source_path, source_label_path | original files, relative to the source root |
| origin_dataset | the dataset the image originally came from (e.g. Tsinghua Dogs) |
| split | empty here; train/val/test is assigned at export time |
| objects | list of faces (below) |
| record_review_reasons, needs_review | why this image should be checked |
| added, schema_version | bookkeeping |

## Face object
| field | meaning |
|---|---|
| class_id, class_name | 0 cat_face, 1 dog_face, null if species unknown |
| species | cat / dog / unknown |
| species_source | filename, clip, filename+clip or manual |
| species_rule | species guessed from the file name (null = no rule) |
| clip_cat_prob | CLIP probability of "cat" on the face crop (null = not run) |
| bbox_xyxy | box in pixels, clipped to the image |
| bbox_yolo | same box as normalized xc, yc, w, h |
| keypoints | dict eye_left / eye_right / nose -> [x, y, v] in pixels |
| eye_distance_ratio | eye distance / box width; ~0.3-0.5 frontal, near 0 side view |
| box_source, keypoint_source | which source produced the annotation |
| original_class | class id in the source dataset |
| flags | box_clipped, keypoint_outside_image, eyes_almost_identical, nose_above_eyes, eye_not_visible |
| excluded | true = removed during review (kept for traceability, skipped at export) |
| review_reasons | why this face needs checking |
"""


def write_meta(out: Path, db, source=None, src_root=None, counts=None):
    reg_path = out / "sources.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    if source:
        entry = dict(SOURCES[source])
        entry.update({"added": dt.date.today().isoformat(), "src_root": str(src_root),
                      "import_counts": dict(counts or {})})
        reg[source] = entry
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    (out / "README.md").write_text(README.format(
        classes=", ".join(f"{i} = {n}" for i, n in enumerate(CLASS_NAMES)),
        keypoints=", ".join(KEYPOINT_NAMES), flip=FLIP_IDX))
    write_boxes_csv(out, db)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_add(args):
    out, src_root = args.out, args.src_root
    out.mkdir(parents=True, exist_ok=True)
    if args.source not in ADAPTERS:
        sys.exit(f"unknown source {args.source}; known: {list(ADAPTERS)}")
    try:
        import imagehash
    except ImportError:
        imagehash = None

    db = load_db(out)
    by_sha = {r["sha1"]: r["image_id"] for r in db.values()}
    counts = Counter()
    new = []
    log = (out / "build_log.txt").open("a")
    log.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} add {args.source} "
              f"from {src_root}\n")

    for n, item in enumerate(ADAPTERS[args.source](src_root, args), 1):
        if n % 2000 == 0:
            print(f"  scanned {n} images ...")
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", item["path"].stem)
        image_id = f"{args.source}_{item['subset']}_{stem}"
        if image_id in db:
            counts["already_in_dataset"] += 1
            continue
        if item["label_path"] is None:
            counts["skipped_no_label"] += 1
            log.write(f"no label file: {item['rel_path']}\n")
            continue
        data = item["path"].read_bytes()
        sha1 = hashlib.sha1(data).hexdigest()
        if sha1 in by_sha:
            counts["skipped_exact_duplicate"] += 1
            log.write(f"duplicate: {item['rel_path']} == {by_sha[sha1]}\n")
            continue
        try:
            im = Image.open(io.BytesIO(data))
            W, H, mode, fmt = im.width, im.height, im.mode, im.format
            try:
                orientation = int(im.getexif().get(0x0112, 1))
            except Exception:  # noqa: BLE001
                orientation = 1
            phash = str(imagehash.phash(im.convert("RGB"))) if imagehash else None
            Image.open(io.BytesIO(data)).verify()
        except Exception as e:  # noqa: BLE001
            counts["skipped_corrupt_image"] += 1
            log.write(f"corrupt image: {item['rel_path']}: {e}\n")
            continue

        objects = []
        for o in item["objects"]:
            obj = build_object(o, W, H, args.source)
            if obj is None:
                counts["dropped_degenerate_box"] += 1
                log.write(f"degenerate box dropped: {item['rel_path']}\n")
            else:
                objects.append(obj)
        if not objects:
            counts["skipped_no_valid_face"] += 1
            log.write(f"no valid face: {item['rel_path']}\n")
            continue

        ext = item["path"].suffix.lower()
        rel_file = f"images/{args.source}/{image_id}{ext}"
        dest = out / rel_file
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            if args.link:
                dest.symlink_to(item["path"].resolve())
            else:
                dest.write_bytes(data)

        rec = {
            "image_id": image_id, "file": rel_file, "width": W, "height": H,
            "mode": mode, "format": fmt, "exif_orientation": orientation,
            "sha1": sha1, "phash": phash,
            "source": args.source, "source_subset": item["subset"],
            "source_path": item["rel_path"], "source_label_path": item["label_path"],
            "origin_dataset": item["origin_dataset"], "split": None,
            "objects": objects,
            "record_review_reasons": ["exif_rotated"] if orientation != 1 else [],
            "needs_review": False,
            "added": dt.date.today().isoformat(), "schema_version": SCHEMA_VERSION,
        }
        db[image_id] = rec
        by_sha[sha1] = image_id
        new.append(rec)
        counts["added_images"] += 1
        counts["added_faces"] += len(objects)

    print(f"Scanned. New images: {len(new)}")
    if args.clip and new:
        run_clip(new, out, args.clip_model, args.batch_size)
    for rec in new:
        for o in rec["objects"]:
            decide_species(o, args.clip_accept)
        update_review_state(rec)

    save_db(out, db)
    write_meta(out, db, args.source, src_root, counts)
    n_review = write_review(out, db)
    log.write(f"counts: {dict(counts)}\n")
    log.close()
    print("Import summary:", dict(counts))
    print(f"Faces to review: {n_review}  ->  {out / 'review' / 'needs_review.csv'}")
    print_stats(db)


def cmd_apply_review(args):
    out = args.out
    db = load_db(out)
    path = out / "review" / "needs_review.csv"
    changed = Counter()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            decision = (row.get("decision (cat / dog / drop / ok)") or "").strip().lower()
            if not decision:
                continue
            rec = db.get(row["image_id"])
            if rec is None:
                continue
            o = rec["objects"][int(row["face_index"])]
            if decision in SPECIES_CLASS:
                o["species"], o["species_source"] = decision, "manual"
                o["class_id"] = SPECIES_CLASS[decision]
                o["class_name"] = CLASS_NAMES[o["class_id"]]
            elif decision == "drop":
                o["excluded"] = True
            elif decision != "ok":
                print(f"unknown decision '{decision}' for {row['image_id']}, ignored")
                continue
            o["review_reasons"] = []
            rec["record_review_reasons"] = []
            update_review_state(rec)
            changed[decision] += 1
    save_db(out, db)
    write_meta(out, db)
    n_review = write_review(out, db)
    print("Applied:", dict(changed), f"| still to review: {n_review}")


def print_stats(db):
    species = defaultdict(Counter)
    flags, subsets = Counter(), Counter()
    n_img = n_face = n_review = 0
    for r in db.values():
        n_img += 1
        n_review += r["needs_review"]
        subsets[(r["source"], r["source_subset"])] += 1
        for o in r["objects"]:
            if o["excluded"]:
                continue
            n_face += 1
            species[r["origin_dataset"]][o["species"]] += 1
            flags.update(o["flags"])
    print(f"\nDataset: {n_img} images, {n_face} faces, {n_review} images need review")
    total = Counter()
    print("Faces by origin dataset:")
    for origin, c in sorted(species.items()):
        total.update(c)
        print(f"  {origin:32s} " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())))
    print("Total: " + ", ".join(f"{k}: {v}" for k, v in sorted(total.items())))
    print("Images per source subset: " +
          ", ".join(f"{s}/{sub}: {v}" for (s, sub), v in sorted(subsets.items())))
    print("Flags: " + (", ".join(f"{k}: {v}" for k, v in flags.most_common()) or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add a source dataset")
    a.add_argument("--source", required=True, choices=list(ADAPTERS))
    a.add_argument("--src-root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--link", action="store_true", help="symlink images instead of copying")
    a.add_argument("--keep-flip-copies", action="store_true",
                   help="pfd1: keep the upside-down <name>_1 copies (skipped by default)")
    a.add_argument("--species-xml-dir", type=Path, default=None,
                   help="folder of Pascal-VOC XMLs giving cat/dog per file name "
                        "(Kaggle 'Dog and Cat Detection' annotations/)")
    a.add_argument("--clip", action="store_true",
                   help="optional extra cat/dog check with CLIP (not needed)")
    a.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    a.add_argument("--clip-accept", type=float, default=0.95,
                   help="min CLIP confidence to accept a species when the filename gives none")
    a.add_argument("--batch-size", type=int, default=64)

    r = sub.add_parser("apply-review", help="apply decisions from review/needs_review.csv")
    r.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("stats", help="print an overview")
    s.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "apply-review":
        cmd_apply_review(args)
    else:
        print_stats(load_db(args.out))


if __name__ == "__main__":
    main()
