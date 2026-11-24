#!/usr/bin/env python3
"""
Train a YOLO-pose model that finds cat_face / dog_face plus 3 keypoints
(eye_left, eye_right, nose) on "pet face detection dataset-1". The trained
model (best.pt) powers the auto-labeler website (app.py).

What it does
  1. converts pfd1 into a clean YOLO-pose dataset in --work:
       - class from the file name: 0 = cat_face, 1 = dog_face
         (Cats_Test* via --species-xml-dir; unknown species and kashtanka_test skipped)
       - keypoints reordered to eye_left, eye_right, nose (left/right as seen
         in the image), so horizontal flips can be used (flip_idx [1, 0, 2])
       - boxes / keypoints clipped to the image, images that are exact
         duplicates of a train image removed from val/test
       - images are symlinked, not copied
  2. trains an ultralytics pose model on it

Usage
  pip install ultralytics
  python train_face_model.py --src-root $D/pet_face_detection_dataset_1 --work $D/yolo_pfd1
  python train_face_model.py ... --prepare-only          # only build the dataset
  python train_face_model.py ... --model yolo11m-pose.pt --epochs 150 --batch 32

  # train on the merged clean dataset from export_yolo.py instead:
  python train_face_model.py --data $D/yolo_merged/data.yaml --work $D/yolo_merged

Needs build_pet_face_dataset.py in the same folder (reuses its pfd1 parsing).
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
try:
    from build_pet_face_dataset import IMG_EXT, load_voc_species, parse_yolo_pose, pfd1_flip_copies, pfd1_origin
except ImportError:
    sys.exit("build_pet_face_dataset.py must be in the same folder as this script")

SPLITS = {"train": "train", "val": "val", "test": "test"}   # pfd1 subset -> split
CLASS = {"cat": 0, "dog": 1}


def clip01(v):
    return min(max(v, 0.0), 1.0)


def canonical_keypoints(kps, box):
    """pfd1 order is nose, eye, eye -> return [eye_left, eye_right, nose] (x, y, v)."""
    missing = (0.0, 0.0, 0)
    if len(kps) < 3:
        return None
    norm = [(clip01(x), clip01(y), int(round(v))) if v > 0 else missing for x, y, v in kps[:3]]
    nose, eyes = norm[0], [e for e in norm[1:3] if e[2] > 0]
    left = right = missing
    if len(eyes) == 2:
        left, right = sorted(eyes, key=lambda e: e[0])
    elif len(eyes) == 1:
        ref = nose[0] if nose[2] > 0 else box[0]
        left, right = (eyes[0], missing) if eyes[0][0] <= ref else (missing, eyes[0])
    return [left, right, nose]


def convert_line(cls_id, box, kps):
    xc, yc, w, h = box
    x1, y1 = clip01(xc - w / 2), clip01(yc - h / 2)
    x2, y2 = clip01(xc + w / 2), clip01(yc + h / 2)
    if x2 - x1 <= 0.002 or y2 - y1 <= 0.002:
        return None
    box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    kp = canonical_keypoints(kps, box)
    if kp is None:
        return None
    vals = [f"{v:.6f}" for v in box]
    for x, y, v in kp:
        vals += [f"{x:.6f}", f"{y:.6f}", str(v)]
    return f"{cls_id} " + " ".join(vals)


def prepare(src_root: Path, work: Path, xml_dir, keep_flip_copies=False):
    xml_species = load_voc_species(xml_dir) if xml_dir else None
    flips = {} if keep_flip_copies else pfd1_flip_copies(src_root)
    if flips:
        print(f"skipping {len(flips)} upside-down copies (<name>_1, augmentation baked into pfd1); "
              "YOLO can flip on the fly if wanted (flipud)")
    counts = Counter()
    train_hashes = set()
    for subset, split in SPLITS.items():          # train first, so val/test can be de-duplicated
        img_dir = src_root / "images" / subset
        if not img_dir.is_dir():
            print(f"missing {img_dir}, skipped")
            continue
        out_img = work / "images" / split
        out_lbl = work / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for old in list(out_img.iterdir()) + list(out_lbl.glob("*.txt")):   # rebuild cleanly
            if old.is_symlink() or old.suffix == ".txt":
                old.unlink()
        for cache in (work / "labels").glob("*.cache"):
            cache.unlink()
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            if (subset, img.stem) in flips:
                counts[f"{split}_skipped_flip_copy"] += 1
                continue
            _, species = pfd1_origin(img.stem, subset, xml_species)
            if species not in CLASS:
                counts[f"{split}_skipped_unknown_species"] += 1
                continue
            parsed = None
            for cand in (src_root / "labels" / subset / f"{img.stem}.txt",
                         src_root / "labels" / subset / img.name):
                if cand.is_file():
                    parsed = parse_yolo_pose(cand)
                    if parsed is not None:
                        break
            if not parsed:
                counts[f"{split}_skipped_no_label"] += 1
                continue
            digest = hashlib.sha1(img.read_bytes()).hexdigest()
            if split == "train":
                train_hashes.add(digest)
            elif digest in train_hashes:
                counts[f"{split}_skipped_duplicate_of_train"] += 1
                continue
            lines = [convert_line(CLASS[species], box, kps) for _, box, kps in parsed]
            lines = [l for l in lines if l]
            if not lines:
                counts[f"{split}_skipped_bad_box"] += 1
                continue
            link = out_img / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
            (out_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
            counts[f"{split}_{species}"] += 1

    yaml = work / "data.yaml"
    yaml.write_text(
        f"path: {work.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [3, 3]\n"
        "flip_idx: [1, 0, 2]   # eye_left <-> eye_right\n"
        "names:\n  0: cat_face\n  1: dog_face\n"
        "# keypoints: 0 eye_left, 1 eye_right (as seen in the image), 2 nose\n"
    )
    print("Prepared:", dict(sorted(counts.items())))
    print(f"Dataset config: {yaml}")
    return yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", type=Path, default=None, help="pet_face_detection_dataset_1 folder")
    ap.add_argument("--data", type=Path, default=None,
                    help="existing data.yaml (e.g. from export_yolo.py): skip preparing pfd1")
    ap.add_argument("--work", type=Path, required=True, help="output folder for dataset + runs")
    ap.add_argument("--species-xml-dir", type=Path, default=None,
                    help="Kaggle 'Dog and Cat Detection' annotations/ (labels Cats_Test* images)")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="pretrained pose model to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--keep-flip-copies", action="store_true",
                    help="keep pfd1's upside-down <name>_1 copies (skipped by default)")
    args = ap.parse_args()

    if args.data:
        yaml = args.data
    elif args.src_root:
        yaml = prepare(args.src_root, args.work, args.species_xml_dir, args.keep_flip_copies)
    else:
        ap.error("give --src-root (prepare dataset-1) or --data (existing data.yaml)")
    if args.prepare_only:
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=str((args.work / "runs").resolve()),
                name="pet_faces", fliplr=0.5, exist_ok=True)
    best = args.work / "runs" / "pet_faces" / "weights" / "best.pt"
    print(f"\nBest model: {best}")
    print(f"Start the labeler with:  python app.py --images <folder> --labels <folder> --model {best}")


if __name__ == "__main__":
    main()
