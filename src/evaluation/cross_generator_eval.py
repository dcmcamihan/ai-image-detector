import os
from pathlib import Path
import json

import torch
import pandas as pd
import torch.nn.functional as F

from src.models.model_factory import create_model
from src.dataset_loader import get_dataloaders
from .metrics_utils import compute_metrics
from .confusion_matrix_plot import plot_confusion_matrix
from src.preprocess_images import get_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def cross_generator_evaluation(model_path: str, model_name: str = "resnet18", tta: int = 0, temperature: float = 1.0, threshold: float | None = None, sweep: int = 0):
    """
    Evaluate generalization by testing on each generator held out.
    Assumes processed data is under data/processed/<generator>/val/...
    """
    data_root = Path("data/standardized_jpg")
    print(f"Using standardized dataset at: {data_root}")
    generators = [d for d in os.listdir(data_root) if (data_root / d).is_dir()]

    device = (
        torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
    )

    results = []
    sweep_store = {}

    for test_gen in generators:
        print(f"\nTesting generalization on held-out generator: {test_gen}")

        # Load checkpoint first and infer classifier head type (dropout vs linear)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        use_sequential_head = any(k.startswith("fc.1.") for k in state_dict.keys())
        inferred_dropout = 0.5 if use_sequential_head else 0.0

        # Build model with inferred head shape
        model = create_model(model_name, num_classes=2, pretrained=False, dropout=inferred_dropout)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Load only the test generator's val split (build loader directly)
        batch_size = 32
        _, val_tfms = get_transforms()
        val_dir = data_root / test_gen / "val"
        val_dataset = ImageFolder(str(val_dir), transform=val_tfms)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        all_preds, all_labels = [], []
        all_scores = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                # Base logits + TTA variants
                logits_list = []
                logits_list.append(model(images))
                if tta and tta > 0:
                    # hflip
                    logits_list.append(model(torch.flip(images, dims=[3])))
                    if tta >= 4:
                        # vflip
                        logits_list.append(model(torch.flip(images, dims=[2])))
                        # hvflip
                        logits_list.append(model(torch.flip(images, dims=[2, 3])))
                # Average logits across TTA variants
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
                # Temperature scaling
                if temperature and temperature > 0:
                    logits = logits / float(temperature)
                probs = F.softmax(logits, dim=1)
                if threshold is not None:
                    preds = (probs[:, 1] >= float(threshold)).long()
                else:
                    preds = torch.argmax(probs, dim=1)
                all_scores.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds, average="binary", y_score=all_scores)
        metrics["test_generator"] = test_gen
        results.append(metrics)

        # Save per-generator confusion matrix
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(all_labels, all_preds, ["ai", "nature"], results_dir / f"confusion_matrix_{model_name}_{test_gen}.png", save_raw=True)

    df = pd.DataFrame(results)
    results_path = Path("results") / f"cross_generator_results_{model_name}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Cross-generator results saved to {results_path}")

    # Optional sweep: compute best thresholds for accuracy and F1, per generator and global
    if sweep:
        try:
            # Re-evaluate per generator with a scan of thresholds
            thresholds = [i / 100 for i in range(10, 91)]
            all_global_scores = []
            all_global_labels = []
            per_gen_thresholds = {}
            # Re-run minimal loops to collect scores/labels per generator
            for test_gen in [d for d in os.listdir(data_root) if (data_root / d).is_dir()]:
                _, val_tfms = get_transforms()
                val_dir = data_root / test_gen / "val"
                val_dataset = ImageFolder(str(val_dir), transform=val_tfms)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                # Load model once per sweep pass
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint["model_state_dict"]
                use_sequential_head = any(k.startswith("fc.1.") for k in state_dict.keys())
                inferred_dropout = 0.5 if use_sequential_head else 0.0
                model = create_model(model_name, num_classes=2, pretrained=False, dropout=inferred_dropout)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                scores, labels_lst = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        logits_list = [model(images)]
                        if tta and tta > 0:
                            images_flipped = torch.flip(images, dims=[3])
                            logits_list.append(model(images_flipped))
                        logits = torch.stack(logits_list, dim=0).mean(dim=0)
                        if temperature and temperature > 0:
                            logits = logits / float(temperature)
                        probs = F.softmax(logits, dim=1)
                        scores.extend(probs[:, 1].cpu().numpy())
                        labels_lst.extend(labels.numpy().tolist())

                # Accumulate for global sweep
                all_global_scores.extend(scores)
                all_global_labels.extend(labels_lst)

                # Per-generator best thresholds
                best_acc = (-1.0, None)
                best_f1 = (-1.0, None)
                import numpy as np
                arr_scores = np.array(scores)
                arr_labels = np.array(labels_lst)
                for thr in thresholds:
                    pred = (arr_scores >= thr).astype(int)
                    from sklearn.metrics import accuracy_score, f1_score
                    acc = accuracy_score(arr_labels, pred)
                    f1 = f1_score(arr_labels, pred)
                    if acc > best_acc[0]:
                        best_acc = (acc, thr)
                    if f1 > best_f1[0]:
                        best_f1 = (f1, thr)
                per_gen_thresholds[test_gen] = {
                    "best_acc": {"acc": best_acc[0], "threshold": best_acc[1]},
                    "best_f1": {"f1": best_f1[0], "threshold": best_f1[1]},
                }

            # Global best thresholds
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score
            g_scores = np.array(all_global_scores)
            g_labels = np.array(all_global_labels)
            g_best_acc = (-1.0, None)
            g_best_f1 = (-1.0, None)
            for thr in thresholds:
                pred = (g_scores >= thr).astype(int)
                acc = accuracy_score(g_labels, pred)
                f1 = f1_score(g_labels, pred)
                if acc > g_best_acc[0]:
                    g_best_acc = (acc, thr)
                if f1 > g_best_f1[0]:
                    g_best_f1 = (f1, thr)
            sweep_store = {
                "global": {
                    "best_acc": {"acc": g_best_acc[0], "threshold": g_best_acc[1]},
                    "best_f1": {"f1": g_best_f1[0], "threshold": g_best_f1[1]},
                },
                "per_generator": per_gen_thresholds,
            }
            thr_path = Path("results") / f"thresholds_{model_name}.json"
            with open(thr_path, "w") as f:
                json.dump(sweep_store, f, indent=2)
            print(f"Threshold sweep saved to {thr_path}")
        except Exception as e:
            print(f"Warning: threshold sweep failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--tta", type=int, default=0, help="Enable simple TTA if >0 (original + hflip)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for logit scaling before softmax")
    parser.add_argument("--threshold", type=float, default=None, help="Global decision threshold for positive class (probs[:,1])")
    parser.add_argument("--sweep", type=int, default=0, help="If >0, compute best thresholds (acc/F1) globally and per-generator")
    args = parser.parse_args()
    cross_generator_evaluation(
        args.model_path,
        model_name=args.model_name,
        tta=args.tta,
        temperature=args.temperature,
        threshold=args.threshold,
        sweep=args.sweep,
    )


