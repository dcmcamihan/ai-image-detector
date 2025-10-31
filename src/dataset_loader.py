"""
PyTorch Dataset & DataLoader for AI Image Detector.
Merges multiple datasets under a root folder (e.g., data/standardized_jpg/*).
Uses ImageFolder + preprocessing transforms from preprocess_images.py.
Supports train & validation splits.
"""

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch
import numpy as np
import random

class LabelAwareDataset(Dataset):
    def __init__(self, base_ds: ImageFolder, conditional):
        self.base_ds = base_ds
        self.conditional = conditional

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        path = None
        if hasattr(self.base_ds, "samples"):
            path = self.base_ds.samples[idx][0]
        img = self.conditional(img, int(label), path)
        return img, label


class ConditionalQuality:
    def __init__(self,
                 p_sharp_ai: float = 0.10,
                 p_blur_real: float = 0.10,
                 sharp_boost_generators: tuple[str, ...] = ("sdv5", "wukong")):
        self.p_sharp_ai = p_sharp_ai
        self.p_blur_real = p_blur_real
        self.boost = set(sharp_boost_generators)
        self.blur = T.GaussianBlur(3)
        self.sharp = T.RandomAdjustSharpness(sharpness_factor=2)

    def _top_folder(self, path: str | None) -> str:
        if not path:
            return "unknown"
        try:
            part = str(path).split("/standardized_jpg/")[1].split("/")[0]
            return part
        except Exception:
            return "unknown"

    def _is_real(self, top: str) -> bool:
        return top in {"general", "hard_cases_web"}

    def __call__(self, img, label: int, path: str | None):
        top = self._top_folder(path)
        is_real = self._is_real(top)

        if is_real:
            if random.random() < self.p_blur_real:
                img = self.blur(img)
        else:
            p = self.p_sharp_ai
            if top in self.boost:
                p = max(p, 0.20)
            if random.random() < p:
                img = self.sharp(img)
        return img


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augmented: bool = False,
    return_class_weights: bool = False,
    hard_cases_dir: str | Path | None = None,
    enable_quality_balance: bool = False,
    p_sharp_ai: float = 0.10,
    p_blur_real: float = 0.15,
    sharp_boost_generators: tuple[str, ...] = ("sdv5", "wukong"),
    upsample_hard_cases_factor: float | None = None,
):
    """
    Load datasets from the specified data_root and return train/val DataLoaders.

    Args:
        data_root: Root folder containing datasets or a specific dataset folder.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoader.
        image_size: Input image size for transforms.

    Returns:
        train_loader, val_loader, class_names
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

    if augmented:
        from .preprocess_images import get_transforms_augmented as _get_tfms
    else:
        from .preprocess_images import get_transforms as _get_tfms
    train_tfms, val_tfms = _get_tfms(image_size=image_size)

    train_datasets = []
    val_datasets = []

    # Check if data_root contains 'train' and 'val' folders directly
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if train_dir.exists() and val_dir.exists():
        train_base = ImageFolder(train_dir, transform=train_tfms)
        val_base = ImageFolder(val_dir, transform=val_tfms)
        if enable_quality_balance:
            cq = ConditionalQuality(p_sharp_ai=p_sharp_ai,
                                    p_blur_real=p_blur_real,
                                    sharp_boost_generators=sharp_boost_generators)
            train_base = LabelAwareDataset(train_base, cq)
        train_datasets.append(train_base)
        val_datasets.append(val_base)
    else:
        # Otherwise, iterate through subfolders (e.g., adm, biggan, etc.)
        for dataset_name in data_root.iterdir():
            if not dataset_name.is_dir():
                continue

            train_dir = dataset_name / "train"
            val_dir = dataset_name / "val"

            if train_dir.exists():
                base = ImageFolder(train_dir, transform=train_tfms)
                if enable_quality_balance:
                    cq = ConditionalQuality(p_sharp_ai=p_sharp_ai,
                                            p_blur_real=p_blur_real,
                                            sharp_boost_generators=sharp_boost_generators)
                    base = LabelAwareDataset(base, cq)
                train_datasets.append(base)
            if val_dir.exists():
                val_datasets.append(ImageFolder(val_dir, transform=val_tfms))

    # Optionally add a compact hard-cases subset
    if hard_cases_dir:
        hc_root = Path(hard_cases_dir)
        hc_train = hc_root / "train"
        hc_val = hc_root / "val"
        if hc_train.exists():
            base = ImageFolder(hc_train, transform=train_tfms)
            if enable_quality_balance:
                cq = ConditionalQuality(p_sharp_ai=p_sharp_ai,
                                        p_blur_real=p_blur_real,
                                        sharp_boost_generators=sharp_boost_generators)
                base = LabelAwareDataset(base, cq)
            train_datasets.append(base)
        if hc_val.exists():
            val_datasets.append(ImageFolder(hc_val, transform=val_tfms))

    # Ensure we found at least one dataset
    if not train_datasets or not val_datasets:
        raise FileNotFoundError(f"No train/val folders found under '{data_root}'.")

    # Merge all datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Build class-balanced sampling weights for the training set
    # ConcatDataset does not expose targets directly; aggregate from sub-datasets
    def _concat_targets(dsets):
        all_targets = []
        for ds in dsets:
            base = ds.base_ds if isinstance(ds, LabelAwareDataset) else ds
            if hasattr(base, "targets"):
                all_targets.extend(base.targets)
            elif hasattr(base, "samples"):
                # fallback: samples is List[(path, target)]
                all_targets.extend([y for _, y in base.samples])
        return np.array(all_targets, dtype=np.int64)

    def _concat_paths(dsets):
        all_paths = []
        for ds in dsets:
            base = ds.base_ds if isinstance(ds, LabelAwareDataset) else ds
            if hasattr(base, "samples"):
                all_paths.extend([x for x, _ in base.samples])
        return np.array(all_paths, dtype=object)

    train_targets = _concat_targets(train_datasets)
    train_paths = _concat_paths(train_datasets)
    class_counts = np.bincount(train_targets)
    # Avoid division by zero in case of missing class
    class_counts[class_counts == 0] = 1
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]

    if upsample_hard_cases_factor and upsample_hard_cases_factor > 1.0:
        mask = np.array(["/hard_cases_web/" in str(p) for p in train_paths])
        sample_weights[mask] = sample_weights[mask] * float(upsample_hard_cases_factor)
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # do not set shuffle when using sampler
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit from pinned memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Assume all datasets share the same classes
    _first = train_datasets[0]
    base0 = _first.base_ds if isinstance(_first, LabelAwareDataset) else _first
    class_names = base0.classes

    if return_class_weights:
        import torch
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return train_loader, val_loader, class_names, cw_tensor
    return train_loader, val_loader, class_names


# Demo / quick test
if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders(batch_size=8)

    print(f"Classes: {class_names}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Peek at one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")