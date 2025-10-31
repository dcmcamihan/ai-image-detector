#!/usr/bin/env python3
"""
Compute simple average-hash (aHash) for all images in a directory tree and report exact duplicates.
Outputs a CSV mapping hash -> file list, and a duplicates CSV listing groups with >1 items.
Usage:
  python scripts/check_duplicate_hashes.py --root data/standardized_jpg --out results/dupe_hashes.csv --dupes results/duplicates.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import csv


def ahash(image: Image.Image, hash_size: int = 8) -> str:
    img = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    avg = pixels.mean()
    bits = pixels > avg
    # pack bits into hex
    bit_string = ''.join('1' if b else '0' for b in bits.flatten())
    return f"{int(bit_string, 2):0{hash_size*hash_size//4}x}"


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str, help="CSV file of all hashes")
    ap.add_argument("--dupes", required=True, type=str, help="CSV file of duplicate groups")
    args = ap.parse_args()

    root = Path(args.root)
    all_rows = []
    by_hash: dict[str, list[str]] = {}

    for img_path in iter_images(root):
        try:
            with Image.open(img_path) as im:
                h = ahash(im)
        except Exception:
            continue
        all_rows.append((h, str(img_path)))
        by_hash.setdefault(h, []).append(str(img_path))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "path"])
        w.writerows(all_rows)

    dupes = [(h, paths) for h, paths in by_hash.items() if len(paths) > 1]
    with open(args.dupes, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "paths"])
        for h, paths in dupes:
            w.writerow([h, "|".join(paths)])

    print(f"Wrote {len(all_rows)} hashes to {args.out}; {len(dupes)} duplicate groups to {args.dupes}")


if __name__ == "__main__":
    main()
