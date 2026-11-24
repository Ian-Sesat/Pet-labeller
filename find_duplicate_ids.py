#!/usr/bin/env python3
"""
Find identities (animal folders) that share identical photos - usually the same
animal registered under several IDs. Does NOT change your data; writes CSVs.

Folder layout: <root>/<identity>/<photo>   (PetFace cat/ and dog/)

  python find_duplicate_ids.py --root $D/cat --out $D/quality/cat_ids

Output
  hashes.csv          sha1 of every photo (cached: a rerun only hashes new files)
  duplicate_photos.csv    every photo that exists more than once, with its copies
  id_groups.csv       groups of identities linked by shared photos + suggestion
  id_merge_map.json   {identity: identity it should be merged into} for "merge" groups

Suggestion per group of identities
  merge   every pair in the group shares >= --min-shared photos (default 2), or the shared
          photos are at least half of the smaller identity: same animal, merge the IDs
  review  linked by a single shared photo only: could be one mis-filed photo; look at it
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def load_hashes(root: Path, cache: Path, workers: int):
    known = {}
    if cache.exists():
        with cache.open(newline="") as f:
            known = {r["file"]: r["sha1"] for r in csv.DictReader(f)}
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.suffix.lower() in IMG_EXT and p.is_file())
    todo = [f for f in files if f not in known]
    print(f"{len(files)} photos, {len(files) - len(todo)} already hashed, hashing {len(todo)}")
    with ThreadPoolExecutor(workers) as ex:
        for i, (f, h) in enumerate(zip(todo, ex.map(lambda f: sha1(root / f), todo, chunksize=256)), 1):
            known[f] = h
            if i % 50000 == 0:
                print(f"  {i}/{len(todo)}")
    hashes = {f: known[f] for f in files}
    with cache.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "sha1"])
        w.writerows(sorted(hashes.items()))
    return hashes


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder with one subfolder per animal")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-shared", type=int, default=2,
                    help="shared photos needed per pair to suggest 'merge' (default 2)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hashes = load_hashes(args.root, args.out / "hashes.csv", args.workers)
    by_hash = defaultdict(list)
    for f, h in hashes.items():
        by_hash[h].append(f)
    ident = lambda f: f.split("/")[0]                                  # noqa: E731
    id_size = Counter(ident(f) for f in hashes)

    dup_groups = [sorted(fs) for fs in by_hash.values() if len(fs) > 1]
    pair_shared = Counter()                                            # (id_a, id_b) -> shared photos
    within = across = 0
    with (args.out / "duplicate_photos.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["copies", "identities", "files"])
        for fs in sorted(dup_groups, key=lambda g: (-len(g), g)):
            ids = sorted({ident(f) for f in fs})
            w.writerow([len(fs), len(ids), " ".join(fs)])
            if len(ids) == 1:
                within += 1
            else:
                across += 1
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pair_shared[(ids[i], ids[j])] += 1

    uf = UnionFind()
    for a, b in pair_shared:
        uf.union(a, b)
    groups = defaultdict(set)
    for a, b in pair_shared:
        groups[uf.find(a)].update((a, b))

    merge_map, rows, sugg = {}, [], Counter()
    for root_id, ids in groups.items():
        ids = sorted(ids)
        pairs = {k: v for k, v in pair_shared.items() if k[0] in ids}
        strong = all(v >= args.min_shared or v * 2 >= min(id_size[k[0]], id_size[k[1]])
                     for k, v in pairs.items())
        suggestion = "merge" if strong else "review"
        sugg[suggestion] += 1
        keep = max(ids, key=lambda i: (id_size[i], -int(i) if i.isdigit() else 0))   # largest ID stays
        if suggestion == "merge":
            for i in ids:
                if i != keep:
                    merge_map[i] = keep
        rows.append([len(ids), suggestion, keep, " ".join(ids),
                     " ".join(f"{i}:{id_size[i]}" for i in ids),
                     " ".join(f"{a}-{b}:{v}" for (a, b), v in sorted(pairs.items())),
                     sum(pairs.values())])

    with (args.out / "id_groups.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_identities", "suggestion", "keep_id", "identities", "photos_per_identity",
                    "shared_photos_per_pair", "shared_photos_total"])
        w.writerows(sorted(rows, key=lambda r: (-r[0], -r[6])))
    (args.out / "id_merge_map.json").write_text(json.dumps(merge_map, indent=1))

    n_ids = len(id_size)
    involved = sum(len(g) for g in groups.values())
    print(f"\n{n_ids} identities, {len(hashes)} photos")
    print(f"duplicated photos: {len(dup_groups)}  ({within} within one identity, {across} across identities)")
    print(f"identities sharing photos with another identity: {involved} ({100 * involved / max(n_ids, 1):.1f}%)"
          f" in {len(groups)} groups")
    print(f"  suggest merge:  {sugg['merge']} groups -> {len(merge_map)} IDs would be merged away")
    print(f"  suggest review: {sugg['review']} groups (only one shared photo)")
    sizes = Counter(len(g) for g in groups.values())
    print("group sizes: " + ", ".join(f"{k} IDs: {v}" for k, v in sorted(sizes.items())))
    print(f"\n-> {args.out / 'id_groups.csv'}\n-> {args.out / 'duplicate_photos.csv'}"
          f"\n-> {args.out / 'id_merge_map.json'}")


if __name__ == "__main__":
    main()
