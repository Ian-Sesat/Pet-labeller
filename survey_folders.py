#!/usr/bin/env python3
"""
Survey image folders before labeling: structure, counts, file types, existing
annotation files, image sizes, example names, broken images.

Usage
  python survey_folders.py /media/.../cat /media/.../dog --out survey.txt
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", default="survey.txt")
    ap.add_argument("--sample", type=int, default=300, help="images opened per root for size check")
    args = ap.parse_args()
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731
    rng = random.Random(0)

    for root in args.roots:
        say("=" * 72)
        say(f"ROOT: {root}")
        say("=" * 72)
        if not root.exists():
            say("  does not exist")
            continue
        per_dir = defaultdict(Counter)
        examples = defaultdict(list)
        images, other_ext, other_examples = [], Counter(), defaultdict(list)
        total_bytes = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower() or "(none)"
            rel_dir = p.parent.relative_to(root)
            per_dir[rel_dir][ext] += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
            if ext in IMG_EXT:
                images.append(p)
                if len(examples[rel_dir]) < 3:
                    examples[rel_dir].append(p.name)
            else:
                other_ext[ext] += 1
                if len(other_examples[ext]) < 3:
                    other_examples[ext].append(str(p.relative_to(root)))

        say(f"images: {len(images)}   total size: {total_bytes / 1e9:.2f} GB   folders: {len(per_dir)}")
        say("image types: " + ", ".join(f"{k}: {v}" for k, v in
                                         Counter(p.suffix.lower() for p in images).most_common()))
        say("other files: " + (", ".join(f"{k}: {v}" for k, v in other_ext.most_common()) or "none"))
        for ext, ex in other_examples.items():
            say(f"   {ext} e.g. {ex}")

        say("\nFolders (images per folder, first 60):")
        for d in sorted(per_dir, key=str)[:60]:
            n_img = sum(v for k, v in per_dir[d].items() if k in IMG_EXT)
            rest = {k: v for k, v in per_dir[d].items() if k not in IMG_EXT}
            say(f"  {str(d) if str(d) != '.' else '(root)'}: {n_img} images"
                + (f"  + {rest}" if rest else "") + (f"   e.g. {examples[d]}" if examples[d] else ""))
        if len(per_dir) > 60:
            say(f"  ... {len(per_dir) - 60} more folders")

        # small text files: print a few lines (annotations / readmes)
        txts = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower()
                in {".txt", ".csv", ".json", ".xml", ".md", ".yaml", ".yml"}]
        for p in txts[:4]:
            say(f"\n--- {p.relative_to(root)} (first lines)")
            try:
                for l in p.read_text(errors="replace").splitlines()[:6]:
                    say("    " + l[:160])
            except OSError as e:
                say(f"    unreadable: {e}")

        if images:
            from PIL import Image
            sample = rng.sample(images, min(args.sample, len(images)))
            sizes, modes, bad = [], Counter(), []
            for p in sample:
                try:
                    with Image.open(p) as im:
                        sizes.append(im.size)
                        modes[im.mode] += 1
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{p.relative_to(root)}: {type(e).__name__}")
            if sizes:
                ws, hs = sorted(s[0] for s in sizes), sorted(s[1] for s in sizes)
                say(f"\nImage size (sample of {len(sample)}): width min {ws[0]} median {ws[len(ws)//2]} "
                    f"max {ws[-1]} | height min {hs[0]} median {hs[len(hs)//2]} max {hs[-1]}")
                say("most common: " + ", ".join(f"{w}x{h} ({c})" for (w, h), c in Counter(sizes).most_common(5)))
                say("color modes: " + ", ".join(f"{k}: {v}" for k, v in modes.items()))
            say(f"broken in sample: {len(bad)}" + (f"  e.g. {bad[:3]}" if bad else ""))
        say()

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
