from imagededup.methods import PHash
from pathlib import Path
import pickle

def cache_encodings(path, phash, cache_name):
    cache_path = Path("results") / f"{cache_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached encodings from {cache_path}")
        return pickle.load(open(cache_path, "rb"))
    print(f"Encoding {cache_name} images...")
    encodings = phash.encode_images(image_dir=str(path), recursive=True)
    print(f"Encoded {len(encodings)} {cache_name} images.")
    pickle.dump(encodings, open(cache_path, "wb"))
    return encodings

def find_cross_split_duplicates(train_encodings, val_encodings, test_encodings, phash, max_distance_threshold):
    def compare_encodings(base_encodings, target_encodings):
        duplicates = {}
        for base_key, base_hash in base_encodings.items():
            for target_key, target_hash in target_encodings.items():
                distance = phash.hamming_distance(base_hash, target_hash)
                if distance <= max_distance_threshold:
                    duplicates.setdefault(base_key, []).append(target_key)
        return duplicates

    print("Finding duplicates between training and validation sets...")
    train_val_duplicates = compare_encodings(train_encodings, val_encodings)

    print("Finding duplicates between training and test sets...")
    train_test_duplicates = compare_encodings(train_encodings, test_encodings)

    print("Finding duplicates between validation and test sets...")
    val_test_duplicates = compare_encodings(val_encodings, test_encodings)

    return train_val_duplicates, train_test_duplicates, val_test_duplicates

def write_duplicates(f, header, duplicates, base_dir, target_dir):
    f.write(f"\n===== {header} =====\n")
    for key, dup_list in duplicates.items():
        if dup_list:
            f.write(f"{base_dir}/{key} → {[str(target_dir / d) for d in dup_list]}\n")

def main():
    train_dir = Path("data/standardized_jpg/sdv5/train")
    val_dir = Path("data/standardized_jpg/sdv5/val")
    test_dir = Path("data/test_subset")  # fixed path
    output_file = "results/cross_split_duplicates.txt"

    phash = PHash()

    # Cache encodings for each split
    train_encodings = cache_encodings(train_dir, phash, "train")
    val_encodings = cache_encodings(val_dir, phash, "val")
    test_encodings = cache_encodings(test_dir, phash, "test")

    # Find cross-split duplicates
    max_distance_threshold = 8
    train_val_duplicates, train_test_duplicates, val_test_duplicates = find_cross_split_duplicates(
        train_encodings, val_encodings, test_encodings, phash, max_distance_threshold
    )

    # Write results to file
    output_file = Path(output_file)
    with open(output_file, "w") as f:
        write_duplicates(f, "TRAIN ↔ VAL", train_val_duplicates, train_dir, val_dir)
        write_duplicates(f, "TRAIN ↔ TEST", train_test_duplicates, train_dir, test_dir)
        write_duplicates(f, "VAL ↔ TEST", val_test_duplicates, val_dir, test_dir)

    print(f"Cross-split duplicate detection complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()