import hashlib
from pathlib import Path

def file_hash(p: Path, blocksize=65536):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            h.update(block)
    return h.hexdigest()

def find_duplicates(train_dir, test_dir):
    train_hashes = {}
    test_hashes = {}

    # Compute hashes for training images
    print("Computing hashes for training images...")
    for f in train_dir.rglob("*"):
        if f.is_file():
            train_hashes[file_hash(f)] = f.relative_to(train_dir)

    # Compute hashes for test images
    print("Computing hashes for test images...")
    for f in test_dir.rglob("*"):
        if f.is_file():
            test_hashes[file_hash(f)] = f.relative_to(test_dir)

    # Find duplicates
    duplicates = []
    for h, test_file in test_hashes.items():
        if h in train_hashes:
            duplicates.append((train_hashes[h], test_file))

    return duplicates

if __name__ == "__main__":
    train_dir = Path("data/standardized_jpg/sdv5/train")
    test_dir = Path("data/test_subset")

    duplicates = find_duplicates(train_dir, test_dir)

    if duplicates:
        print(f"Found {len(duplicates)} duplicates between training and test sets:")
        for train_file, test_file in duplicates:
            print(f"Train: {train_file} ↔ Test: {test_file}")
    else:
        print("No duplicates found between training and test sets.")