import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str, *, save_raw: bool = False):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    if save_raw:
        # Save raw counts matrix alongside normalized
        raw_path = Path(str(save_path).replace(".png", "_raw.png"))
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Raw Counts)")
        plt.tight_layout()
        plt.savefig(raw_path, dpi=300)
        plt.close()
        print(f"Raw confusion matrix saved to {raw_path}")


