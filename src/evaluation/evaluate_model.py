import argparse
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from src.models.model_factory import create_model
from src.dataset_loader import get_dataloaders
from .metrics_utils import compute_metrics
from .confusion_matrix_plot import plot_confusion_matrix


def evaluate(
    model_path: str,
    model_name: str = "resnet18",
    *,
    data_root: str | None = None,
    log_path: str | None = None,
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Use provided dataset root if given, otherwise default to 'general'
    data_root = Path(data_root or "data/standardized_jpg/general")
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {data_root}")

    _, val_loader, class_names = get_dataloaders(data_root=data_root, batch_size=32)

    # Load checkpoint first to infer head structure (dropout vs linear)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    # If keys like 'fc.1.weight' exist, the head is Sequential(Dropout, Linear)
    use_sequential_head = any(k.startswith("fc.1.") for k in state_dict.keys())
    inferred_dropout = 0.5 if use_sequential_head else 0.0

    model = create_model(
        model_name,
        num_classes=len(class_names),
        pretrained=False,
        dropout=inferred_dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            # assume class 1 is the positive class for ROC-AUC
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, average="binary", y_score=all_scores)
    print(f"Evaluation Metrics ({model_name}): {metrics}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"metrics_summary_{model_name}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    plot_confusion_matrix(all_labels, all_preds, class_names, results_dir / f"confusion_matrix_{model_name}.png", save_raw=True)

    if log_path:
        with open(log_path, "a") as f:
            f.write(f"{model_name}: {metrics}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/saved/last_model.pth")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root for evaluation")
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    evaluate(
        args.model_path,
        model_name=args.model_name,
        data_root=args.data_root,
        log_path=args.log_path,
    )