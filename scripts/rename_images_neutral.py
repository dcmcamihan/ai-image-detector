import argparse
from pathlib import Path
from tqdm import tqdm


def rename_images_neutrally(root_dir: str, force_jpg: bool = True):
    """
    Recursively find class folders named 'ai' or 'nature' under root_dir and
    rename images to neutral numeric filenames within each class folder:
    000001.jpg, 000002.jpg, ...

    Does not change folder structure. If force_jpg is True, renames extensions
    to .jpg as well (does not re-encode; run standardization beforehand).
    """
    root = Path(root_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    class_dirs = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name in {"ai", "nature"}:
            class_dirs.append(p)

    for class_dir in class_dirs:
        files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in exts])
        for i, f in enumerate(tqdm(files, desc=f"{class_dir}", leave=False)):
            new_name = f"{i:06d}.jpg" if force_jpg else f"{i:06d}{f.suffix.lower()}"
            target = class_dir / new_name
            if f == target:
                continue
            # Avoid overwriting by staging to a temp name if needed
            if target.exists():
                temp = class_dir / (new_name + ".tmp")
                f.rename(temp)
                temp.rename(target)
            else:
                f.rename(target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename images neutrally under ai/nature folders")
    ap.add_argument("--root", type=str, required=False, default="data/general")
    ap.add_argument("--no_force_jpg", action="store_true", help="Do not force .jpg extension")
    args = ap.parse_args()

    rename_images_neutrally(args.root, force_jpg=(not args.no_force_jpg))
    print("Renamed all images neutrally (no class info in filenames).")