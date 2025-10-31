import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def standardize_images(
    src_root: Path,
    dst_root: Path,
    *,
    target_long_side: int = 512,
    jpeg_quality: int = 95,
    square_size: int | None = None,
) -> None:
    """
    Create a standardized copy of an image dataset:
    - Converts every image to RGB
    - If square_size is set: force resize to (square_size, square_size)
    - Else: resize so the longer side == target_long_side (keeps aspect)
    - Writes to dst_root mirroring src_root structure

    This does NOT modify the source tree.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    for dirpath, _, filenames in os.walk(src_root):
        rel = Path(dirpath).relative_to(src_root)
        out_dir = dst_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in tqdm(filenames, desc=str(rel) if str(rel) else ".", leave=False):
            if Path(fname).suffix.lower() not in exts:
                continue
            in_path = Path(dirpath) / fname
            base = Path(fname).stem
            out_path = out_dir / f"{base}.jpg"

            try:
                with Image.open(in_path) as img:
                    img = img.convert("RGB")
                    if square_size is not None:
                        img = img.resize((square_size, square_size), Image.BICUBIC)
                    else:
                        w, h = img.size
                        if w >= h:
                            new_w = target_long_side
                            new_h = max(1, int(h * (target_long_side / float(w))))
                        else:
                            new_h = target_long_side
                            new_w = max(1, int(w * (target_long_side / float(h))))
                        img = img.resize((new_w, new_h), Image.BICUBIC)
                    img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
            except Exception as e:
                print(f"Skip {in_path}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Standardize dataset images to JPEG with resizing.")
    ap.add_argument("--src", type=str, required=False,
                    default=str(Path(__file__).resolve().parents[1] / "data" / "hard_cases_web"))
    ap.add_argument("--dst", type=str, required=False,
                    default=str(Path(__file__).resolve().parents[1] / "data" / "hard_cases_web"))
    ap.add_argument("--target_long_side", type=int, default=512, help="Long-side size if square_size is not set")
    ap.add_argument("--square_size", type=int, default=None, help="If set, resize to (S,S)")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    print(f"Source:   {src}")
    print(f"Standard: {dst}")
    standardize_images(src, dst, target_long_side=args.target_long_side,
                       jpeg_quality=args.jpeg_quality, square_size=args.square_size)
    print("Done. You can point training/eval to the standardized directory.")


