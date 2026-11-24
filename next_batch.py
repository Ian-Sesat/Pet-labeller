#!/usr/bin/env python3
"""
Pick the next images to review by hand, so your time goes where it helps most.

Without knowing the right answer, it estimates how wrong each auto-label is:
  - mirror test: the model labels the image AND a mirrored copy; where the two
    answers disagree, the model is unsure (strongest signal)
  - rotation test: the same with the image turned by +/- 15 degrees; catches
    faces where the model is wrong in the same way on the normal and mirrored image
  - old vs new model disagree (with --old-model)
  - no face found, low confidence, unusual keypoint layout (with --template)
"estimated error" = how far the answer is from the average of the two answers
(half their disagreement), largest of the 3 points, as a share of the face size
(same unit as the fine-tune evaluation: above 5% = a point you would move).
The refreshed drafts use the AVERAGE of the normal and mirrored answer, which is
usually more accurate than either one.

The batch (default 500) has three groups:
  high    (60%) highest estimated error: most useful corrections, review first
  low     (20%) lowest estimated error: quick confirms; if these need fixing,
                the error estimate is not trustworthy
  random  (20%) random images: show the TRUE quality of the auto-labels (taken
                from validation animals in split.csv, so they also grow the fair test)

The drafts of the batch images are refreshed with the new model (old answer kept
in the label as previous_auto). Reviewed labels are never touched.
The website shows the batch under "Next batch".

Usage
  D=/media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941
  python next_batch.py --labels $D/dog_labels --images $D/dog --species dog \
      --model $D/finetune_v2/runs/finetune/weights/best.pt \
      --old-model $D/yolo_pfd1/runs/pet_faces/weights/best.pt \
      --template $D/quality/template/template.json \
      --split $D/finetune_v2/split.csv --set-name dog --n 500 --device 0
"""

import argparse
import csv
import datetime as dt
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

KP = ["eye_left", "eye_right", "nose"]
SPECIES = ["cat", "dog"]
GRAY = (114, 114, 114)
KP_CONF = 0.5      # keypoints below this confidence count as not recognised (not shown)
MISS = 0.08        # estimated error when a face is found normally but not in a mirrored / turned copy


def pad_image(img, pad):
    px, py = int(round(img.width * pad)), int(round(img.height * pad))
    canvas = Image.new("RGB", (img.width + 2 * px, img.height + 2 * py), GRAY)
    canvas.paste(img, (px, py))
    return canvas, px, py


def load(path, pad):
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    canvas, px, py = pad_image(img, pad) if pad > 0 else (img, 0, 0)
    return img.size, canvas, px, py


def top_face(r, px, py, W, H, mirrored=False, CW=None):
    """Best face of one ultralytics result -> {box, conf, cls, kps{name: [x, y, c]}} in image coords."""
    if r.boxes is None or len(r.boxes) == 0:
        return None
    i = int(r.boxes.conf.argmax())
    b = r.boxes.xyxy[i].tolist()
    conf, cls = float(r.boxes.conf[i]), int(r.boxes.cls[i])
    pts = r.keypoints.data[i].tolist() if r.keypoints is not None else []
    if mirrored:                                   # map back from the mirrored canvas
        b = [CW - b[2], b[1], CW - b[0], b[3]]
        pts = [[CW - p[0], p[1], *(p[2:] or [1.0])] for p in pts]
    b = [min(max(b[0] - px, 0), W), min(max(b[1] - py, 0), H), min(max(b[2] - px, 0), W), min(max(b[3] - py, 0), H)]
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return None
    pts = [[p[0] - px, p[1] - py, p[2] if len(p) > 2 else 1.0] for p in pts[:3]]
    if len(pts) < 3:
        return {"box": b, "conf": conf, "cls": cls, "kps": None}
    eyes = sorted(pts[:2], key=lambda p: p[0])     # eye_left / eye_right as seen in the image
    return {"box": b, "conf": conf, "cls": cls, "kps": {"eye_left": eyes[0], "eye_right": eyes[1], "nose": pts[2]}}


def rotate_face(f, angle, cx, cy):
    """Map a face found on an image turned by `angle` degrees (PIL, counter-clockwise)
    back onto the unturned image."""
    if f is None or not f["kps"]:
        return f
    a = math.radians(angle)                       # undo: turn the points back
    ca, sa = math.cos(a), math.sin(a)

    def back(x, y):
        dx, dy = x - cx, y - cy
        return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca
    kps = {k: [*back(p[0], p[1]), p[2]] for k, p in f["kps"].items()}
    xs, ys = zip(*(back(x, y) for x in (f["box"][0], f["box"][2]) for y in (f["box"][1], f["box"][3])))
    return {"box": [min(xs), min(ys), max(xs), max(ys)], "conf": f["conf"], "cls": f["cls"],
            "kps": {"eye_left": min(kps["eye_left"], kps["eye_right"]),
                    "eye_right": max(kps["eye_left"], kps["eye_right"]), "nose": kps["nose"]}}


def kp_dist(a, b):
    """Largest keypoint distance between two faces, as a share of a's face size."""
    if not a or not b or not a["kps"] or not b["kps"]:
        return None
    size = max(a["box"][2] - a["box"][0], a["box"][3] - a["box"][1])
    # compare only points both answers actually recognise (a face may have 1, 2 or 3)
    ks = [k for k in KP if a["kps"][k][2] >= KP_CONF and b["kps"][k][2] >= KP_CONF] or KP
    return max(math.hypot(a["kps"][k][0] - b["kps"][k][0], a["kps"][k][1] - b["kps"][k][1]) for k in ks) / size


def predict_all(model_path, items, args, mirror):
    """items: [(name, path)] -> {name: (face, mirrored_face, (W, H), [turned faces])}"""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    out = {}
    pool = ThreadPoolExecutor(args.workers)
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        loaded = list(pool.map(lambda it: load(it[1], args.pad), chunk))
        canv = [c for _, c, _, _ in loaded]
        res = model.predict(canv, imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)
        resm = model.predict([c.transpose(Image.FLIP_LEFT_RIGHT) for c in canv], imgsz=args.imgsz,
                             conf=args.conf, verbose=False, device=args.device) if mirror else [None] * len(canv)
        turned = []
        for ang in (args.rotate, -args.rotate) if mirror and args.rotate else ():
            rr = model.predict([c.rotate(ang, resample=Image.BILINEAR, fillcolor=GRAY) for c in canv],
                               imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)
            turned.append((ang, rr))
        for i, ((name, _), (wh, c, px, py), r, rm) in enumerate(zip(chunk, loaded, res, resm)):
            f = top_face(r, px, py, *wh)
            fm = top_face(rm, px, py, *wh, mirrored=True, CW=c.width) if rm is not None else None
            rot = []
            for ang, rr in turned:                  # turn back on the canvas, then remove the border
                ft = top_face(rr[i], 0, 0, c.width, c.height)
                ft = rotate_face(ft, ang, c.width / 2, c.height / 2)
                if ft and ft["kps"]:
                    ft["kps"] = {k: [p[0] - px, p[1] - py, p[2]] for k, p in ft["kps"].items()}
                    ft["box"] = [ft["box"][0] - px, ft["box"][1] - py, ft["box"][2] - px, ft["box"][3] - py]
                rot.append(ft)
            out[name] = (f, fm, wh, rot)
        done = start + len(chunk)
        if done % (args.batch * 20) < args.batch or done == len(items):
            print(f"  {Path(model_path).name}: {done}/{len(items)}")
    return out


def average_faces(a, b):
    """Test-time augmentation: mean of the normal and the mirrored answer."""
    if not b or not a["kps"] or not b["kps"]:
        return a
    kps = {k: [(a["kps"][k][0] + b["kps"][k][0]) / 2, (a["kps"][k][1] + b["kps"][k][1]) / 2,
               min(a["kps"][k][2], b["kps"][k][2])] for k in KP}
    box = [(x + y) / 2 for x, y in zip(a["box"], b["box"])]
    return {"box": box, "conf": min(a["conf"], b["conf"]), "cls": a["cls"], "kps": kps}


def to_object(f, species, model_name):
    cid = SPECIES.index(species)
    kps = None
    if f["kps"]:                                   # not recognised -> missing (not shown)
        kps = {k: ([round(f["kps"][k][0], 1), round(f["kps"][k][1], 1), 2] if f["kps"][k][2] >= KP_CONF
                   else [0.0, 0.0, 0]) for k in KP}
        seen = [k for k in ("eye_left", "eye_right") if kps[k][2]]
        if len(seen) == 1:                         # one eye: name it by its side of the nose / face
            e = kps[seen[0]]
            ref = kps["nose"][0] if kps["nose"][2] else (f["box"][0] + f["box"][2]) / 2
            kps["eye_left"], kps["eye_right"] = (e, [0.0, 0.0, 0]) if e[0] <= ref else ([0.0, 0.0, 0], e)
    return {"class_id": cid, "class_name": f"{species}_face", "species": species, "score": round(f["conf"], 3),
            "bbox_xyxy": [round(v, 1) for v in f["box"]], "keypoints": kps, "source": model_name, "edited": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="the website's labels folder")
    ap.add_argument("--images", type=Path, required=True, help="the website's images folder")
    ap.add_argument("--species", choices=SPECIES, required=True)
    ap.add_argument("--model", type=Path, required=True, help="the model you label with now (new)")
    ap.add_argument("--old-model", type=Path, default=None, help="previous model: disagreement = extra signal")
    ap.add_argument("--template", type=Path, default=None, help="template.json: unusual layout = extra signal")
    ap.add_argument("--split", type=Path, default=None, help="split.csv from finetune_labeler.py")
    ap.add_argument("--set-name", default=None, help="name of this set in split.csv (cat / dog)")
    ap.add_argument("--n", type=int, default=500, help="batch size")
    ap.add_argument("--share-high", type=float, default=0.6)
    ap.add_argument("--share-low", type=float, default=0.2)
    ap.add_argument("--pool", type=int, default=20000, help="unreviewed images scored (0 = all)")
    ap.add_argument("--pad", type=float, default=0.5, help="same as the website's --pad")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp-conf", type=float, default=0.5,
                    help="keypoints below this confidence are not shown (same as the website's --kp-conf)")
    ap.add_argument("--rotate", type=float, default=15, help="rotation test angle in degrees (0 = off)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None, help="default: different every run")
    ap.add_argument("--no-update-drafts", action="store_true", help="do not refresh the batch drafts")
    args = ap.parse_args()
    global KP_CONF
    KP_CONF = args.kp_conf
    for nm, p in (("--labels", args.labels), ("--images", args.images)):
        if not p.is_dir():
            raise SystemExit(f"{nm} folder not found: {p}  (is $D set?)")
    rng = random.Random(args.seed)

    # ---- candidates: images not reviewed yet
    print("reading label status (large folders take a few minutes) ...")
    status = {}
    for jf in args.labels.rglob("*.json"):
        rel = jf.relative_to(args.labels)
        if any(p.startswith("_") for p in rel.parts):
            continue
        try:
            status[rel.as_posix()[:-5]] = json.loads(jf.read_text()).get("status", "none")
        except Exception:  # noqa: BLE001
            continue
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    cands = []
    for p in args.images.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            name = p.relative_to(args.images).as_posix()
            if status.get(name, "none") != "reviewed":
                cands.append(name)
    old_batch = args.labels / "_batch.json"
    if old_batch.exists():                         # do not pick unfinished batch images again
        prev = json.loads(old_batch.read_text())
        left = [n for n in prev if status.get(n) != "reviewed"]
        print(f"previous batch: {len(prev) - len(left)} of {len(prev)} reviewed; the rest go back into the pool")
    n_rev = sum(1 for s in status.values() if s == "reviewed")
    print(f"{len(cands)} unreviewed images, {n_rev} reviewed")
    if not cands:
        raise SystemExit("nothing left to review")
    val_ids = set()
    if args.split and args.split.exists():
        with args.split.open() as fh:
            val_ids = {x["identity"] for x in csv.DictReader(fh)
                       if x["split"] == "val" and (not args.set_name or x["set"] == args.set_name)}
    ident = lambda n: n.split("/")[0] if "/" in n else n
    pool = cands if not args.pool or args.pool >= len(cands) else rng.sample(cands, args.pool)
    if val_ids:                                    # the random group comes from validation animals:
        in_pool = set(pool)                        # make sure all their unreviewed images are scored
        val_c = [n for n in cands if ident(n) in val_ids and n not in in_pool]
        pool += rng.sample(val_c, min(len(val_c), 5000))
        print(f"validation animals: {len(val_ids)}, their unreviewed images: "
              f"{sum(ident(n) in val_ids for n in pool)}")
    items = [(n, args.images / n) for n in pool]

    # ---- predictions
    new = predict_all(args.model, items, args, mirror=True)
    old = predict_all(args.old_model, items, args, mirror=False) if args.old_model else {}
    qc = T = None
    if args.template:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import quality_check as qc
        T = json.loads(args.template.read_text())
        qc.CROPS = qc.FACE_CROPS = args.pad > 0

    rows = []
    for name in pool:
        f, fm, (W, H), rot = new[name]
        r = {"name": name, "identity": name.split("/")[0] if "/" in name else name, "face": f,
             "conf": f["conf"] if f else 0.0}
        if f is None:
            r.update(est=1.0, why="no face found")
        else:
            parts = []
            miss = []                              # face not found in a copy: a hint, not a certain error
            dm = kp_dist(f, fm) if fm else None
            if dm is None:
                miss.append("mirrored")
            parts.append(("mirror test", dm / 2 if dm is not None else MISS))
            if rot:
                rd = [kp_dist(f, t) for t in rot]
                if any(d is None for d in rd):
                    miss.append("turned")
                parts.append(("rotation test", max(d / 2 if d is not None else MISS for d in rd)))
            if old:
                o = old[name][0]
                do = kp_dist(f, o) if o else None
                if do is None:
                    miss.append("old model")
                parts.append(("old vs new model", do / 2 if do is not None else MISS))
            r["face"] = average_faces(f, fm)
            parts = [(w, v) for w, v in parts if v is not None]
            est = max((v for _, v in parts), default=1.0)
            why = ", ".join(f"{w} {100 * v:.1f}%" for w, v in parts if v != MISS)
            if miss:
                why += (", " if why else "") + "no face in the " + " / ".join(miss) + " copy"
            if f["conf"] < 0.5:
                why += f", low confidence {f['conf']:.2f}"
            if qc and f["kps"]:
                fc = dict(image=name, path="", W=W, H=H, species=args.species, box=f["box"],
                          kps={k: [f["kps"][k][0], f["kps"][k][1], 2 if f["kps"][k][2] >= KP_CONF else 0]
                               for k in KP}, origin="", idx=0, pre_flags=[])
                qc.analyze(fc)
                qc.score_face(fc, T)
                if fc["flagged"]:
                    est = max(est, 0.05)           # at least "worth a look"
                    why += ", unusual layout: " + "; ".join(fc["reasons"][:2])
            r.update(est=est, why=why)
        rows.append(r)

    # ---- the batch
    n_high = round(args.n * args.share_high)
    n_low = round(args.n * args.share_low)
    n_rand = args.n - n_high - n_low
    rand_pool = [r for r in rows if r["identity"] in val_ids]
    random_pick = rng.sample(rand_pool, min(n_rand, len(rand_pool)))
    n_val_rand = len(random_pick)
    if len(random_pick) < n_rand:                  # too few validation images left: top up
        rest_r = [r for r in rows if r["identity"] not in val_ids]
        random_pick += rng.sample(rest_r, min(n_rand - len(random_pick), len(rest_r)))
    taken = {r["name"] for r in random_pick}
    # high / low from training animals, so the random part stays a fair, unbiased test
    rest = [r for r in rows if r["name"] not in taken and (r["identity"] not in val_ids or not val_ids)]
    rest.sort(key=lambda r: (-r["est"], r["conf"]))  # no face first, then largest disagreement
    high = rest[:n_high]
    taken |= {r["name"] for r in high}
    low = [r for r in reversed(rest) if r["name"] not in taken and r["face"] is not None][:n_low]
    batch = [("high", r) for r in high] + [("low", r) for r in low] + [("random", r) for r in random_pick]

    # ---- refresh the drafts of the batch images with the new model
    updated = kept_edited = 0
    model_name = f"yolo:{args.model.name}"
    if not args.no_update_drafts:
        for _, r in batch:
            lp = args.labels / f"{r['name']}.json"
            d = json.loads(lp.read_text()) if lp.exists() else {"image": r["name"], "status": "none"}
            if d.get("status") == "reviewed" or any(o.get("edited") for o in d.get("objects", [])):
                kept_edited += d.get("status") != "reviewed"
                continue                           # your work: never overwritten
            W, H = new[r["name"]][2]
            prev_objs = d.get("objects", [])
            d.update(status="draft", width=W, height=H, model=model_name,
                     objects=[to_object(r["face"], args.species, model_name)] if r["face"] else [],
                     updated=dt.datetime.now().isoformat(timespec="seconds"))
            if prev_objs:
                d["previous_auto"] = prev_objs
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(json.dumps(d, indent=1))
            updated += 1

    out = {}
    for rank, (group, r) in enumerate(batch, 1):
        out[r["name"]] = {"rank": rank, "group": group, "est_error": round(r["est"], 4), "reason": r["why"]}
    (args.labels / "_batch.json").write_text(json.dumps(out, indent=1))
    with (args.labels / "_batch.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "group", "image", "estimated_error_pct", "reason"])
        for name, b in out.items():
            w.writerow([b["rank"], b["group"], name, round(100 * b["est_error"], 1), b["reason"]])

    # ---- how good are the auto-labels overall? (estimate on the whole scored pool)
    ests = sorted(r["est"] for r in rows)
    over5 = sum(e > 0.05 for e in ests) / len(ests)
    noface = sum(r["face"] is None for r in rows) / len(rows)
    print(f"\nscored {len(rows)} unreviewed images with {args.model.name}"
          f"{' + ' + args.old_model.name if args.old_model else ''}")
    print(f"  estimated error, median: {100 * ests[len(ests) // 2]:.1f}% of face size")
    print(f"  images that probably need a correction (> 5%): {100 * over5:.1f}%  |  no face found: {100 * noface:.1f}%")
    n_noface = sum(r["face"] is None for r in high)
    print(f"batch of {len(batch)}: high {len(high)} ({n_noface} with no face found, the rest est. error "
          f"{100 * high[-1]['est']:.1f}% to {100 * max([r['est'] for r in high if r['face'] is not None] or [0]):.1f}%), "
          f"low {len(low)}, random {len(random_pick)}"
          + (f" ({n_val_rand} from validation animals"
             + (f", {len(random_pick) - n_val_rand} from other animals: not enough validation images left)"
                if len(random_pick) > n_val_rand else ")") if val_ids else ""))
    print(f"drafts refreshed with the new model: {updated}"
          + (f" (kept {kept_edited} drafts you had edited by hand)" if kept_edited else ""))
    print(f"-> {args.labels / '_batch.json'}  (website filter 'Next batch')  and  {args.labels / '_batch.csv'}")


if __name__ == "__main__":
    main()
