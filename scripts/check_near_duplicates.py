#!/usr/bin/env python3
"""
Approximate near-duplicate finder using average-hash (aHash) with Hamming distance.
Finds groups of images whose aHash distance <= threshold (default 5).
Outputs CSV of groups for manual review.

Usage:
  python scripts/check_near_duplicates.py --root data/standardized_jpg --threshold 5 --out results/near_duplicates.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import csv


def ahash(image: Image.Image, hash_size: int = 8) -> int:
    img = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    root = Path(args.root)
    items: list[tuple[int, str]] = []
    for img_path in iter_images(root):
        try:
            with Image.open(img_path) as im:
                h = ahash(im)
        except Exception:
            continue
        items.append((h, str(img_path)))

    # Naive O(N^2) grouping; suitable for subsets or per-generator runs.
    # For very large sets, pre-bucket by top bits.
    visited = set()
    groups: list[list[str]] = []
    n = len(items)
    for i in range(n):
        if i in visited:
            continue
        hi, pi = items[i]
        group = [pi]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            hj, pj = items[j]
            if hamming(hi, hj) <= args.threshold:
                group.append(pj)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "paths"])
        for gid, g in enumerate(groups):
            w.writerow([gid, "|".join(g)])

    print(f"Found {len(groups)} near-duplicate groups (threshold={args.threshold}). Written to {args.out}")


if __name__ == "__main__":
    main()
