"""
Preprocesses raw AI images:
- Checks for corrupted images
- Resizes to IMAGE_SIZE (224x224)
- Normalizes pixel values (0-1)
- Saves processed images to data/processed/ with the same folder structure
"""

import os
import random
from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError, ImageFilter
import torch
from tqdm import tqdm
from torchvision import transforms as T

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SIZE = 224

def jpeg_compression(img: Image.Image) -> Image.Image:
    """Random JPEG compression simulation (30% chance)."""
    if random.random() < 0.3:
        buffer = BytesIO()
        quality = random.randint(30, 90)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return img

def get_transforms_basic(image_size: int = IMAGE_SIZE):
    """
    Return torchvision transforms for training and validation.

    Args:
        image_size: target square size for resizing/cropping

    Returns:
        (train_transforms, val_transforms)
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tfms = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(15),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tfms, val_tfms


def get_transforms_augmented(image_size: int = IMAGE_SIZE):
    """
    Stronger augmentation pipeline for robustness experiments.
    Includes random JPEG compression simulation, jitter, rotation, grayscale, blur.
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tfms = T.Compose([
        T.Lambda(jpeg_compression),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(3)], p=0.3),
        # Additional realistic blur range
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.2),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
        T.RandomApply([T.RandomAffine(5, translate=(0.05, 0.05))], p=0.2),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.15)], p=0.1),
        T.RandomApply([
            T.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.4))))
        ], p=0.1),
        # RandomToneCurve-like gamma adjustment
        T.RandomApply([T.Lambda(lambda img: img.point(lambda p: int(255 * ((p/255.0) ** random.uniform(0.8, 1.2)))))], p=0.2),
        # Broaden color jitter to include saturation/hue
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)], p=0.3),
        # Mild unsharp mask occasionally to mimic editing artifacts
        T.RandomApply([T.Lambda(lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150)))], p=0.1),
        # Tiny extra blur branch
        T.RandomApply([T.Lambda(lambda img: img if random.random() > 0.5 else img.filter(ImageFilter.GaussianBlur(radius=0.5)))], p=0.1),
        T.RandomRotation(15),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        # Mild Gaussian noise on tensor
        T.RandomApply([T.Lambda(lambda t: (t + torch.randn_like(t) * 0.02).clamp(0.0, 1.0))], p=0.2),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tfms, val_tfms


def get_transforms(image_size: int = IMAGE_SIZE):
    """
    Backward compatibility shim. Returns the basic transforms.
    """
    return get_transforms_basic(image_size)

def preprocess_image(input_path, output_path):
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return True
    except UnidentifiedImageError:
        print(f"Corrupted image skipped: {input_path}")
        return False

def main():
    # Update paths to target only the 'general' folder
    general_dir = PROJECT_ROOT / "data" / "standardized_jpg" / "general"
    processed_dir = PROJECT_ROOT / "data" / "standardized_jpg" / "general"

    print(f"Starting preprocessing for 'general' folder...\nSource dir: {general_dir}\nProcessed dir: {processed_dir}")
    
    total_images = 0
    processed_count = 0
    corrupted_count = 0

    # Walk through the 'general' folder
    for split in ["train", "val"]:
        split_path = general_dir / split
        if not split_path.exists():
            continue

        for class_name in os.listdir(split_path):
            class_path = split_path / class_name
            if not class_path.is_dir():
                continue

            for img_file in tqdm(list(class_path.iterdir()), desc=f"{split}/{class_name}"):
                total_images += 1
                output_file = processed_dir / split / class_name / img_file.name
                success = preprocess_image(img_file, output_file)
                if success:
                    processed_count += 1
                else:
                    corrupted_count += 1

    print(f"\nPreprocessing complete for 'general' folder!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Corrupted/skipped: {corrupted_count}")

if __name__ == "__main__":
    main()