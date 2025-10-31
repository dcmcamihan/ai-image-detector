#!/usr/bin/env python3
"""
Duplicate cleanup utility.
Reads duplicates CSVs produced by check_duplicate_hashes.py and/or near-duplicates CSVs,
then (optionally) moves or deletes files. Default is dry-run listing.

Usage examples:
  # dry run on exact dupes
  python scripts/cleanup_duplicates.py --csv results/adm_duplicates.csv

  # move second+ items of each exact-duplicate group to quarantine folder
  python scripts/cleanup_duplicates.py --csv results/adm_duplicates.csv --action move --quarantine data/quarantine/adm

  # delete second+ items (use with care)
  python scripts/cleanup_duplicates.py --csv results/adm_duplicates.csv --action delete

  # near dups file format: group_id, paths (pipe-delimited). Use --near to parse that format
  python scripts/cleanup_duplicates.py --csv results/adm_near_duplicates.csv --near --action move --quarantine data/quarantine/adm_near
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import shutil


def parse_exact(csv_path: Path) -> list[list[str]]:
    groups: list[list[str]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        # expects columns: hash, paths (pipe-delimited) OR multiple rows per hash
        if "paths" in r.fieldnames:
            for row in r:
                paths = row["paths"].split("|")
                if len(paths) > 1:
                    groups.append(paths)
        else:
            # group rows by hash
            by_hash: dict[str, list[str]] = {}
            for row in r:
                by_hash.setdefault(row["hash"], []).append(row["path"])
            for paths in by_hash.values():
                if len(paths) > 1:
                    groups.append(paths)
    return groups


def parse_near(csv_path: Path) -> list[list[str]]:
    groups: list[list[str]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            paths = row["paths"].split("|")
            if len(paths) > 1:
                groups.append(paths)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to duplicates or near-duplicates CSV")
    ap.add_argument("--near", action="store_true", help="Parse CSV as near-duplicate format")
    ap.add_argument("--action", choices=["dryrun", "move", "delete"], default="dryrun")
    ap.add_argument("--quarantine", default="data/quarantine", help="Folder to move files into (for action=move)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    groups = parse_near(csv_path) if args.near else parse_exact(csv_path)
    print(f"Loaded {len(groups)} groups from {csv_path}")

    if args.action == "dryrun":
        kept, acted = 0, 0
        for g in groups:
            if not g:
                continue
            kept += 1
            acted += max(0, len(g) - 1)
        print(f"Dry-run summary: keep 1st of {kept} groups, mark {acted} files for action.")
        return

    quarantine_root = Path(args.quarantine)
    if args.action == "move":
        quarantine_root.mkdir(parents=True, exist_ok=True)

    acted = 0
    for g in groups:
        # Keep the first path; act on the rest
        for path in g[1:]:
            p = Path(path)
            if not p.exists():
                continue
            if args.action == "delete":
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Failed to delete {p}: {e}")
                acted += 1
            elif args.action == "move":
                rel = p.name
                dest = quarantine_root / rel
                # ensure unique dest
                i = 1
                while dest.exists():
                    dest = quarantine_root / f"{p.stem}_{i}{p.suffix}"
                    i += 1
                try:
                    shutil.move(str(p), str(dest))
                except Exception as e:
                    print(f"Failed to move {p} -> {dest}: {e}")
                acted += 1
    print(f"Action '{args.action}' completed for {acted} files.")


if __name__ == "__main__":
    main()
