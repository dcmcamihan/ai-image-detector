#!/usr/bin/env python3
"""
Quality probe: compute simple blur score (variance of Laplacian) and record image resolution.
Saves a CSV for OOD/quality tracking alongside evaluation datasets.

Usage:
  python scripts/quality_probe.py --root data/standardized_jpg --out results/quality_probe.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import csv


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


def blur_variance_laplacian(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    rows = []
    for path in iter_images(Path(args.root)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        blur = blur_variance_laplacian(img)
        rows.append((str(path), w, h, blur))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "width", "height", "blur_var_laplacian"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} quality rows to {args.out}")


if __name__ == "__main__":
    main()
