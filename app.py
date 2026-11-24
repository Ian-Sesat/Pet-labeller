#!/usr/bin/env python3
"""
Pet Face Labeler - a local website that automatically finds cat and dog faces
plus their keypoints (eye_left, eye_right, nose) and lets you correct them.

Backends
  yolo  YOLO-pose model (e.g. best.pt from train_face_model.py): faces + keypoints
  owl   OWLv2 zero-shot detector: faces only, no training needed; you place
        keypoints by hand (keys 1 / 2 / 3)
  none  manual labeling only

Install
  pip install fastapi uvicorn python-multipart pillow
  pip install ultralytics              # for --backend yolo
  pip install torch transformers       # for --backend owl

Run
  python app.py --images /path/to/new_images --labels /path/to/labels \
                --backend yolo --model /path/to/best.pt
  then open http://localhost:8000
  (server on another machine:  ssh -L 8000:localhost:8000 user@server)

Labels: one JSON per image in --labels (same relative path + ".json"), status
none / draft (auto or edited, not checked) / reviewed. "Export" writes a
YOLO-pose dataset (images symlinked) + annotations.jsonl into --labels/_export.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps

CLASS_NAMES = ["cat_face", "dog_face"]
SPECIES = ["cat", "dog"]
KP_NAMES = ["eye_left", "eye_right", "nose"]       # left/right as seen in the image
FLIP_IDX = [1, 0, 2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATIC = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------
# Detection backends
# --------------------------------------------------------------------------
def species_from_class_name(name: str):
    n = name.lower()
    if "cat" in n:
        return "cat"
    if "dog" in n:
        return "dog"
    return None


def make_object(species, score, box, keypoints, source):
    cid = SPECIES.index(species) if species in SPECIES else None
    return {
        "class_id": cid,
        "class_name": CLASS_NAMES[cid] if cid is not None else None,
        "species": species if species in SPECIES else "unknown",
        "score": round(float(score), 3) if score is not None else None,
        "bbox_xyxy": [round(float(v), 1) for v in box],
        "keypoints": keypoints,
        "source": source,
        "edited": False,
    }


def canonical_keypoints(points, names, kp_conf):
    """points in model order [(x, y[, conf]), ...] -> {eye_left, eye_right, nose}.
    Any two 'eye*' points are re-sorted by x (left/right as seen in the image)."""
    missing = [0.0, 0.0, 0]
    nose, eyes = missing, []
    for p, name in zip(points, names):
        x, y = float(p[0]), float(p[1])
        c = float(p[2]) if len(p) > 2 else 1.0
        v = 2 if c >= kp_conf and not (x <= 0 and y <= 0) else 0   # not recognised -> not shown
        pt = [round(x, 1), round(y, 1), 2] if v else missing
        if name.startswith("nose"):
            nose = pt
        elif name.startswith("eye") and v:
            eyes.append(pt)
    if len(eyes) == 2:
        left, right = sorted(eyes, key=lambda e: e[0])
    elif len(eyes) == 1:                         # one eye: name it by its side of the nose / face
        ref = nose[0] if nose[2] else None
        left, right = (eyes[0], missing) if ref is None or eyes[0][0] <= ref else (missing, eyes[0])
    else:
        left = right = missing
    return {"eye_left": left, "eye_right": right, "nose": nose}


class YoloBackend:
    default_conf = 0.25

    def __init__(self, weights, kpt_names, kp_conf, device):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.kpt_names = kpt_names
        self.kp_conf = kp_conf
        self.name = f"yolo:{Path(weights).name}"
        names = self.model.names
        self.classes = [names[i] for i in sorted(names)]

    def predict(self, img: Image.Image, conf: float):
        # agnostic_nms: one face can't get both a cat box and a dog box
        r = self.model.predict(img, conf=conf, verbose=False, device=self.device, agnostic_nms=True)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().tolist()
        cls = r.boxes.cls.cpu().int().tolist()
        scores = r.boxes.conf.cpu().tolist()
        kps = r.keypoints.data.cpu().tolist() if r.keypoints is not None else [None] * len(boxes)
        out = []
        for b, c, s, k in zip(boxes, cls, scores, kps):
            sp = species_from_class_name(self.model.names[c])
            kp = canonical_keypoints(k, self.kpt_names, self.kp_conf) if k else None
            out.append(make_object(sp, s, b, kp, self.name))
        return out


def nms(boxes, scores, iou_thr=0.5):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        ok = True
        for j in keep:
            a, b = boxes[i], boxes[j]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


class OwlBackend:
    default_conf = 0.15

    def __init__(self, model_name, device):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.queries = ["a photo of a cat's face", "a photo of a dog's face"]
        self.name = f"owlv2:{model_name.split('/')[-1]}"
        self.classes = CLASS_NAMES

    def predict(self, img: Image.Image, conf: float):
        torch = self.torch
        inputs = self.proc(text=[self.queries], images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        side = max(img.size)                     # OWLv2 pads the image to a square
        res = self.proc.post_process_object_detection(
            outputs=outputs, threshold=conf, target_sizes=torch.tensor([[side, side]]))[0]
        boxes = [[max(0, b[0]), max(0, b[1]), min(img.width, b[2]), min(img.height, b[3])]
                 for b in res["boxes"].cpu().tolist()]
        scores = res["scores"].cpu().tolist()
        labels = res["labels"].cpu().tolist()
        return [make_object(SPECIES[labels[i]], scores[i], boxes[i], None, self.name)
                for i in nms(boxes, scores)]


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class State:
    images_dir: Path
    labels_dir: Path
    backend = None
    pad = 0.0
    single_face = False
    species = None              # "cat" / "dog": species known from the folder
    template = None             # template.json from quality_check.py (live shape check)
    crops = False               # face crops: judge the keypoint layout, not the box
    qc = None                   # quality_check module
    feat_cache: dict = {}       # image -> (updated, [face results]) for statistics
    lock = threading.Lock()
    image_list: list = []
    status: dict = {}
    quality: dict = {}
    batch: dict = {}            # from next_batch.py: {image: {rank, group, est_error, reason}}          # from quality_check.py: {image: {score, reasons}}
    progress = {"running": False, "done": 0, "total": 0, "errors": 0, "stop": False}


S = State()
app = FastAPI(title="Pet Face Labeler")


def load_quality():
    qp = S.labels_dir / "_quality.json"
    S.quality = json.loads(qp.read_text()) if qp.exists() else {}
    for i, v in enumerate(S.quality.values()):      # file order = wrong labels first, then by score
        v.setdefault("rank", i)


def load_batch():
    bp = S.labels_dir / "_batch.json"
    S.batch = json.loads(bp.read_text()) if bp.exists() else {}


def save_quality():
    (S.labels_dir / "_quality.json").write_text(json.dumps(S.quality, indent=1))


def scan_images():
    load_quality()
    load_batch()
    labels_res = S.labels_dir.resolve()
    names = []
    for p in sorted(S.images_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        if labels_res == p.resolve().parent or labels_res in p.resolve().parents:
            continue
        names.append(p.relative_to(S.images_dir).as_posix())
    S.image_list = names
    for n in names:
        if n not in S.status:
            lp = label_path(n)
            st = "none"
            if lp.exists():
                try:
                    st = json.loads(lp.read_text()).get("status", "draft")
                except Exception:  # noqa: BLE001
                    st = "draft"
            S.status[n] = st


def image_path(name: str) -> Path:
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() not in IMG_EXT:
        raise HTTPException(404, "bad image name")
    p = S.images_dir / rel
    if not p.is_file():
        raise HTTPException(404, "image not found")
    return p


def label_path(name: str) -> Path:
    return S.labels_dir / f"{name}.json"


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def read_label(name: str):
    lp = label_path(name)
    if lp.exists():
        return json.loads(lp.read_text())
    return {"image": name, "status": "none", "objects": []}


def write_label(name: str, data: dict):
    data["image"] = name
    data["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    lp = label_path(name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = lp.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(lp)
    S.status[name] = data.get("status", "draft")


def pad_image(img: Image.Image, pad: float):
    """Put the image in the middle of a gray border so a face crop looks like a face
    in a full photo (what the model was trained on). Returns canvas and offsets."""
    px, py = int(round(img.width * pad)), int(round(img.height * pad))
    canvas = Image.new("RGB", (img.width + 2 * px, img.height + 2 * py), (114, 114, 114))
    canvas.paste(img, (px, py))
    return canvas, px, py


def unpad_object(o, px, py, W, H):
    """Move a prediction from the padded canvas back into the image; None if it lies outside."""
    x1, y1, x2, y2 = o["bbox_xyxy"]
    x1, x2 = min(max(x1 - px, 0), W), min(max(x2 - px, 0), W)
    y1, y2 = min(max(y1 - py, 0), H), min(max(y2 - py, 0), H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    o["bbox_xyxy"] = [round(v, 1) for v in (x1, y1, x2, y2)]
    if o.get("keypoints"):
        for k, (x, y, v) in o["keypoints"].items():
            x, y = x - px, y - py
            inside = 0 <= x <= W and 0 <= y <= H
            o["keypoints"][k] = [round(x, 1), round(y, 1), v] if v and inside else [0.0, 0.0, 0]
    return o


def set_species(o, sp):
    cid = SPECIES.index(sp)
    o.update(species=sp, class_id=cid, class_name=CLASS_NAMES[cid])


def run_model(name: str, conf: float):
    if S.backend is None:
        raise HTTPException(400, "no detection backend loaded")
    img = load_rgb(image_path(name))
    src, px, py = pad_image(img, S.pad) if S.pad > 0 else (img, 0, 0)
    with S.lock:
        objects = S.backend.predict(src, conf)
    if S.pad > 0:
        objects = [o for o in (unpad_object(o, px, py, img.width, img.height) for o in objects) if o]
    if S.single_face and objects:
        objects = [max(objects, key=lambda o: o["score"] or 0)]
    if S.species:
        for o in objects:
            model_sp = o["species"]
            set_species(o, S.species)
            if model_sp in SPECIES and model_sp != S.species:
                o["model_species"] = model_sp          # kept for information only
    return {"width": img.width, "height": img.height, "objects": objects, "model": S.backend.name}


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/api/config")
def config():
    b = S.backend
    return {
        "backend": b.name if b else None,
        "default_conf": b.default_conf if b else 0.25,
        "has_keypoints": isinstance(b, YoloBackend),
        "species": S.species, "single_face": S.single_face, "pad": S.pad,
        "template": S.template is not None, "crops": S.crops,
        "classes": CLASS_NAMES,
        "keypoints": KP_NAMES,
        "images_dir": str(S.images_dir),
        "folder_key": hashlib.sha1(str(S.images_dir.resolve()).encode()).hexdigest()[:8],
        "labels_dir": str(S.labels_dir),
    }


@app.get("/api/images")
def images():
    return [{"name": n, "status": S.status.get(n, "none"),
             **({"q": S.quality[n]["score"], "qr": S.quality[n]["rank"]} if n in S.quality else {}),
             **({"b": S.batch[n]["rank"], "bg": S.batch[n]["group"], "be": S.batch[n]["est_error"]}
                if n in S.batch else {})}
            for n in S.image_list]


@app.post("/api/rescan")
def rescan():
    scan_images()
    return {"count": len(S.image_list)}


# never let the browser reuse an image: cat/ and dog/ folders share names like 00000/01.png
NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/api/image")
def get_image(name: str):
    p = image_path(name)
    with Image.open(p) as im:
        orientation = im.getexif().get(0x0112, 1)
        if orientation in (None, 1):
            return FileResponse(p, headers=NO_CACHE)
        fixed = ImageOps.exif_transpose(im).convert("RGB")
    buf = io.BytesIO()
    fixed.save(buf, "JPEG", quality=92)
    return Response(buf.getvalue(), media_type="image/jpeg", headers=NO_CACHE)


@app.get("/api/labels")
def get_labels(name: str):
    image_path(name)
    data = read_label(name)
    if name in S.quality:
        data["_quality"] = S.quality[name]
    if name in S.batch:
        data["_batch"] = S.batch[name]
    return data


@app.put("/api/labels")
def put_labels(name: str, data: dict = Body(...)):
    image_path(name)
    data.pop("_quality", None)
    data.pop("_batch", None)
    if data.get("status") == "reviewed" and name in S.quality:
        del S.quality[name]                    # fixed / confirmed -> off the "Needs check" list
        save_quality()
    if data.get("status") not in ("none", "draft", "reviewed"):
        data["status"] = "draft"
    if data["status"] == "none" and data.get("objects"):
        data["status"] = "draft"
    write_label(name, data)
    return {"ok": True, "status": data["status"]}


# --------------------------------------------------------------------------
# Live shape check + statistics (needs --template)
# --------------------------------------------------------------------------
def check_objects(objects, W, H, name=""):
    """Compare each face with the template. Returns per face: flagged, shape (x of the
    99% limit), reasons, features, ghosts (typical keypoint positions to draw)."""
    qc, T = S.qc, S.template
    results = []
    for i, o in enumerate(objects):
        k = o.get("keypoints")
        kp, fl = qc.canonical_face([k[n] for n in KP_NAMES], o["bbox_xyxy"], "canonical") if k else (None, [])
        sp = o.get("species") if o.get("species") in SPECIES else "unknown"
        f = dict(image=name, path="", W=W, H=H, species=sp, box=list(o["bbox_xyxy"]),
                 kps=kp, origin="", idx=i, pre_flags=fl)
        qc.analyze(f)
        if not f["feat"] and not f["flags"]:
            results.append({"flagged": False, "shape": None, "reasons": [], "features": {}, "ghosts": []})
            continue
        qc.score_face(f, T)
        t = T["species"][f["ref"]]
        results.append({"flagged": f["flagged"], "shape": f["shape"], "ref": f["ref"],
                        "reasons": f["reasons"], "features": {a: round(b, 4) for a, b in f["feat"].items()},
                        "ghosts": ghosts(f, t)})
    return results


def ghosts(f, t):
    """Where the keypoints usually are, given this face: r = about 2 robust spreads."""
    med, sc, out = t["median"], t["scale"], []
    x1, y1, x2, y2 = f.get("cbox") or f["box"]
    bw, bh = x2 - x1, y2 - y1
    if not S.crops and bw > 2 and bh > 2:          # from the box (full photos only)
        for n, s in (("eye_left", "el"), ("eye_right", "er"), ("nose", "n")):
            out.append({"kp": n, "from": "box", "x": x1 + med["u_" + s] * bw, "y": y1 + med["v_" + s] * bh,
                        "r": max(2 * (sc["u_" + s] * bw + sc["v_" + s] * bh) / 2, 3)})
    k = f["kps"] or {}
    el, er = k.get("eye_left"), k.get("eye_right")
    if el and er and el[2] and er[2]:               # nose from the two eyes
        dx, dy = er[0] - el[0], er[1] - el[1]
        d = math.hypot(dx, dy)
        if d > 1:
            ux, uy = dx / d, dy / d
            mx, my = (el[0] + er[0]) / 2, (el[1] + er[1]) / 2
            a, b = med["nose_offset"] * d, med["nose_drop"] * d
            out.append({"kp": "nose", "from": "eyes", "x": mx + a * ux - b * uy, "y": my + a * uy + b * ux,
                        "r": max(2 * (sc["nose_offset"] + sc["nose_drop"]) / 2 * d, 3)})
    return out


@app.post("/api/check")
def api_check(data: dict = Body(...)):
    if S.template is None:
        return {"enabled": False, "faces": []}
    W, H = data.get("width"), data.get("height")
    if not W or not H:
        return {"enabled": True, "faces": []}
    return {"enabled": True, "faces": check_objects(data.get("objects", []), W, H, data.get("image", ""))}


@app.get("/api/stats")
def api_stats():
    """Statistics of the REVIEWED labels in this folder, compared with the template."""
    if S.template is None:
        return {"enabled": False}
    qc, T = S.qc, S.template
    feats = {"cat": [], "dog": []}
    flagged, n_rev, n_faces = [], 0, Counter()
    for name in S.image_list:
        if S.status.get(name) != "reviewed":
            continue
        n_rev += 1
        lp = label_path(name)
        try:
            mtime = lp.stat().st_mtime
        except OSError:
            continue
        cached = S.feat_cache.get(name)
        if not cached or cached[0] != mtime:
            d = json.loads(lp.read_text())
            res = check_objects(d.get("objects", []), d.get("width"), d.get("height"), name) \
                if d.get("width") and d.get("height") else []
            sps = [o.get("species") for o in d.get("objects", [])]
            cached = (mtime, res, sps)
            S.feat_cache[name] = cached
        _, res, sps = cached
        for r, sp in zip(res, sps):
            if sp in feats:
                n_faces[sp] += 1
                feats[sp].append(r["features"])
            if r["flagged"]:
                flagged.append({"name": name, "shape": r["shape"], "reasons": r["reasons"][:3]})
    keys = qc.KP_JOINT if S.crops else [k for k in qc.EXPLAIN]
    species_out = {}
    for sp, rows in feats.items():
        if not rows or sp not in T["species"]:
            continue
        t = T["species"][sp]
        table = []
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if not vals:
                continue
            vals.sort()
            mine = vals[len(vals) // 2]
            table.append({"feature": k, "label": qc.FEATURES[k], "template": t["median"][k],
                          "range": [t["percentiles"][k]["1"], t["percentiles"][k]["99"]],
                          "yours": mine, "n": len(vals), "shift": (mine - t["median"][k]) / t["scale"][k],
                          "svg": qc.svg_hist(t["hist"][k], vals, w=150, ht=34)})
        species_out[sp] = {"faces": n_faces[sp], "table": table}
    flagged.sort(key=lambda r: -(r["shape"] or 0))
    return {"enabled": True, "reviewed": n_rev, "total": len(S.image_list), "faces": dict(n_faces),
            "flagged": flagged[:50], "n_flagged": len(flagged), "species": species_out,
            "crops": S.crops, "template_source": T.get("source", {})}


@app.post("/api/predict")
def predict(name: str, conf: float = 0.25):
    return run_model(name, conf)


def autolabel_worker(conf: float, overwrite_drafts: bool):
    P = S.progress
    targets = [n for n in S.image_list
               if S.status.get(n, "none") == "none"
               or (overwrite_drafts and S.status.get(n) == "draft")]
    P.update(running=True, done=0, total=len(targets), errors=0, stop=False, skipped_edited=0)
    for n in targets:
        if P["stop"]:
            break
        try:
            if S.status.get(n) == "draft":
                old = read_label(n)
                if any(o.get("edited") for o in old.get("objects", [])):
                    P["skipped_edited"] = P.get("skipped_edited", 0) + 1
                    P["done"] += 1
                    continue                    # your corrections, not yet marked reviewed
            r = run_model(n, conf)
            write_label(n, {"status": "draft", "width": r["width"], "height": r["height"],
                            "objects": r["objects"], "model": r["model"]})
        except Exception:  # noqa: BLE001
            P["errors"] += 1
            traceback.print_exc()
        P["done"] += 1
    P["running"] = False


@app.get("/api/autolabel_count")
def autolabel_count():
    c = {"none": 0, "draft": 0, "reviewed": 0}
    for n in S.image_list:
        c[S.status.get(n, "none")] = c.get(S.status.get(n, "none"), 0) + 1
    return c


@app.post("/api/autolabel_all")
def autolabel_all(conf: float = 0.25, overwrite_drafts: bool = False):
    if S.backend is None:
        raise HTTPException(400, "no detection backend loaded")
    if S.progress["running"]:
        raise HTTPException(409, "already running")
    threading.Thread(target=autolabel_worker, args=(conf, overwrite_drafts), daemon=True).start()
    return {"started": True}


@app.get("/api/progress")
def progress():
    return S.progress


@app.post("/api/stop")
def stop():
    S.progress["stop"] = True
    return {"ok": True}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    dest = S.images_dir / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in IMG_EXT:
            continue
        target = dest / name
        i = 1
        while target.exists():
            target = dest / f"{Path(name).stem}_{i}{Path(name).suffix}"
            i += 1
        target.write_bytes(await f.read())
        saved += 1
    scan_images()
    return {"saved": saved}


@app.post("/api/export")
def export(data: dict = Body(default={})):
    only_reviewed = bool(data.get("only_reviewed", True))
    out = S.labels_dir / "_export"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"images": 0, "faces": 0, "skipped_unknown_species": 0}
    with (out / "annotations.jsonl").open("w") as jl:
        for name in S.image_list:
            st = S.status.get(name, "none")
            if st == "none" or (only_reviewed and st != "reviewed"):
                continue
            lab = read_label(name)
            W, H = lab.get("width"), lab.get("height")
            if not W or not H:
                with Image.open(image_path(name)) as im:
                    W, H = ImageOps.exif_transpose(im).size
            flat = name.replace("/", "__")
            link = out / "images" / flat
            if not link.exists():
                link.symlink_to(image_path(name).resolve())
            lines = []
            for o in lab.get("objects", []):
                if o.get("class_id") is None:
                    counts["skipped_unknown_species"] += 1
                    continue
                x1, y1, x2, y2 = o["bbox_xyxy"]
                x1, x2 = sorted((max(0, x1), min(W, x2)))
                y1, y2 = sorted((max(0, y1), min(H, y2)))
                vals = [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
                kp_vals = []
                for k in KP_NAMES:
                    x, y, v = (o.get("keypoints") or {}).get(k) or [0, 0, 0]
                    kp_vals += ([min(max(x / W, 0), 1), min(max(y / H, 0), 1), int(v)]
                                if v else [0, 0, 0])
                lines.append(f"{o['class_id']} " + " ".join(f"{v:.6f}" for v in vals) + " "
                             + " ".join(f"{v:.6f}" if i % 3 < 2 else str(v)
                                        for i, v in enumerate(kp_vals)))
                counts["faces"] += 1
            (out / "labels" / f"{Path(flat).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            jl.write(json.dumps({"image": name, "file": str(image_path(name).resolve()),
                                 "width": W, "height": H, "status": st,
                                 "objects": lab.get("objects", [])}) + "\n")
            counts["images"] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        f"kpt_shape: [3, 3]\nflip_idx: {FLIP_IDX}\n"
        "names:\n  0: cat_face\n  1: dog_face\n"
        "# keypoints: 0 eye_left, 1 eye_right (as seen in the image), 2 nose\n")
    return {"folder": str(out), **counts}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder with images (searched recursively)")
    ap.add_argument("--labels", type=Path, required=True, help="folder for label JSON files")
    ap.add_argument("--backend", choices=["yolo", "owl", "none"], default="yolo")
    ap.add_argument("--model", default=None, help="YOLO-pose weights (.pt) for --backend yolo")
    ap.add_argument("--owl-model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--kpt-names", default="eye_left,eye_right,nose",
                    help="keypoint order of the YOLO model (any 'eye*' points are sorted left/right)")
    ap.add_argument("--kp-conf", type=float, default=0.5,
                    help="keypoints below this confidence are not shown (missing)")
    ap.add_argument("--device", default=None, help="e.g. 0, cuda, cpu")
    ap.add_argument("--species", choices=["cat", "dog"], default=None,
                    help="all images show this species (e.g. the cat/ folder): every face gets it")
    ap.add_argument("--single-face", action="store_true",
                    help="keep only the best box per image (face crops with one animal)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="gray border per side as a fraction of the image before predicting; "
                         "0.5 for face crops (default 0 = full photos)")
    ap.add_argument("--template", default=None,
                    help="template.json from quality_check.py: live shape check + statistics while labeling")
    ap.add_argument("--crops", default=None, action=argparse.BooleanOptionalAction,
                    help="judge only the keypoint layout (face crops); default: on when --pad > 0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    S.images_dir = args.images
    S.labels_dir = args.labels
    if not S.images_dir.is_dir():
        raise SystemExit(f"images folder not found: {S.images_dir}")
    S.labels_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "yolo":
        if not args.model:
            raise SystemExit("--backend yolo needs --model path/to/best.pt (or use --backend owl)")
        S.backend = YoloBackend(args.model, args.kpt_names.split(","), args.kp_conf, args.device)
    elif args.backend == "owl":
        S.backend = OwlBackend(args.owl_model, args.device)
    S.pad, S.single_face, S.species = args.pad, args.single_face, args.species
    if args.template:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import quality_check as qc
        S.qc = qc
        S.template = json.loads(Path(args.template).read_text())
        S.crops = args.crops if args.crops is not None else S.pad > 0
        qc.CROPS = qc.FACE_CROPS = S.crops
        if S.crops and any("kp_joint_mean" not in t for t in S.template["species"].values()):
            raise SystemExit("this template is older than crop mode: rebuild it with quality_check.py build-template")
        print(f"template: {args.template} ({', '.join(S.template['species'])})"
              f"{' | crop mode: keypoint layout only' if S.crops else ''}")
    print(f"backend: {S.backend.name if S.backend else 'none'}"
          f"{f' | species: {S.species}' if S.species else ''}"
          f"{' | one face per image' if S.single_face else ''}{f' | pad {S.pad}' if S.pad else ''}")
    print("scanning images (a large folder takes a while) ...")

    scan_images()
    print(f"{len(S.image_list)} images in {S.images_dir}")
    print(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
