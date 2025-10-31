from pathlib import Path
from collections import Counter, defaultdict
import argparse


def main():
    ap = argparse.ArgumentParser(description="Summarize dataset splits and sample filename prefixes")
    ap.add_argument("--root", type=str, default="data/standardized_jpg", help="Either a super-root containing generator dirs, or a single dataset dir with train/val")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    if not ROOT.exists():
        raise FileNotFoundError(f"Root not found: {ROOT}")

    summary = defaultdict(lambda: defaultdict(Counter))

    def summarize_one(name: str, base: Path):
        for split in ("train", "val"):
            d = base / split
            if not d.exists():
                continue
            for cls in d.iterdir():
                if not cls.is_dir():
                    continue
                files = [f for f in cls.iterdir() if f.is_file() and not f.name.startswith(".")]
                summary[name][split][cls.name] = len(files)
                prefixes = [f.name.split("_")[0] for f in files if "_" in f.name]
                summary[name][f"{split}_{cls.name}_prefixes"] = Counter(prefixes).most_common(5)

    # If ROOT has train/val directly, treat as a single dataset named after the folder
    if (ROOT / "train").exists() or (ROOT / "val").exists():
        summarize_one(ROOT.name, ROOT)
    else:
        # Else assume ROOT contains multiple generator/dataset folders
        for gen in ROOT.iterdir():
            if gen.is_dir():
                summarize_one(gen.name, gen)

    for gen, data in summary.items():
        print("GENERATOR:", gen)
        for k, v in data.items():
            if isinstance(v, Counter) or isinstance(v, list):
                print("  ", k, v)
            else:
                print("  ", k, v)
        print()


if __name__ == "__main__":
    main()