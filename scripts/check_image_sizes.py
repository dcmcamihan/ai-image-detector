from pathlib import Path
from PIL import Image
import argparse


def collect_files(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    ai = []
    nature = []
    # Support either a root with train/val subfolders or a flat root with ai/nature
    candidates = []
    if (root / "train").exists() or (root / "val").exists():
        candidates.extend([(root / "train" / "ai"), (root / "train" / "nature"),
                           (root / "val" / "ai"), (root / "val" / "nature")])
    else:
        candidates.extend([root / "ai", root / "nature"])

    for d in candidates:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                (ai if d.name == "ai" else nature).append(f)
    return ai, nature


def main():
    ap = argparse.ArgumentParser(description="Check image sizes and dimensions under a dataset root")
    ap.add_argument("--root", type=str, default="data/hard_cases_web", help="Dataset root (with train/val or direct ai/nature)")
    args = ap.parse_args()

    data_root = Path(args.root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Root not found: {data_root}")

    ai_files, nature_files = collect_files(data_root)

    print(f"Found {len(ai_files)} AI images")
    print(f"Found {len(nature_files)} Nature images")

    if ai_files:
        ai_avg_size = sum(f.stat().st_size for f in ai_files) / len(ai_files)
        print(f"AI avg size: {ai_avg_size / 1024:.2f} KB")
    else:
        print("No AI images found")

    if nature_files:
        nature_avg_size = sum(f.stat().st_size for f in nature_files) / len(nature_files)
        print(f"Nature avg size: {nature_avg_size / 1024:.2f} KB")
    else:
        print("No Nature images found")

    # Check dimensions of a few images
    def print_image_dimensions(files, label, num_samples=5):
        print(f"\nChecking dimensions of {label} images:")
        for f in files[:num_samples]:
            try:
                with Image.open(f) as img:
                    print(f"{f.name}: {img.size} (width x height)")
            except Exception as e:
                print(f"Could not open {f}: {e}")

    print_image_dimensions(ai_files, "AI")
    print_image_dimensions(nature_files, "Nature")


if __name__ == "__main__":
    main()