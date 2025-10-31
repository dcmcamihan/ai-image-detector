from pathlib import Path
from collections import Counter
import argparse


def main():
    ap = argparse.ArgumentParser(description="Count file extensions per label folder (ai/nature) under root")
    ap.add_argument("--root", type=str, default="data/hard_cases_web", help="Dataset root to scan (expects train/val/ai|nature)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    counts = {"ai": Counter(), "nature": Counter()}
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for split in ("train", "val"):
        for cls in ("ai", "nature"):
            d = root / split / cls
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file():
                    counts[cls][f.suffix.lower() if f.suffix else ""] += 1
    print(counts)


if __name__ == "__main__":
    main()