# ...existing code...
"""
Enhanced report generator for AI Image Detector.

Produces:
 - consolidated metrics CSV
 - per-model, per-generator summary CSVs
 - attractive PNG charts (metrics bar, heatmaps, per-generator bar/heatmap,
   class distributions, ROC/PR if scores available, calibration, confusion)
 - sample mosaics (per-generator and global)
 - optional UMAP of model embeddings (requires --embeddings and a model_path)
 - HTML report combining everything (results/report_assets/report.html)

Usage (from repo root, venv activated):
    python scripts/reporting/generate_full_report.py \
        --model swin_tiny_patch4_window7_224 \
        --model-path models/regularized_training/model_swin_t_regularized.pth \
        [--results-dir results] [--data-root data/standardized_jpg] \
        [--output-dir results/report_assets] [--embeddings]

Notes:
 - Renames 'nature' -> 'real' in exported tables and plots.
 - Will gracefully skip plots if required inputs are missing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# Optional heavy imports only used when embeddings requested
try:
    import torch
    import timm
    from torchvision import transforms
except Exception:
    torch = None
    timm = None

# Aesthetic palette (consistent)
PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8d99ae"]
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette(PALETTE))
plt.rcParams.update({"figure.dpi": 140, "axes.titlesize": 13, "axes.labelsize": 11})


def find_metrics_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("metrics_summary_*.csv"))


def load_metrics(metrics_files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in metrics_files:
        try:
            df = pd.read_csv(f)
            row = df.iloc[0].to_dict()
        except Exception:
            row = {}
            for line in f.read_text().splitlines():
                if "," in line:
                    k, v = line.split(",", 1)
                    try:
                        row[k.strip()] = float(v.strip())
                    except Exception:
                        row[k.strip()] = v.strip()
        model_name = f.stem.replace("metrics_summary_", "")
        row["model"] = model_name
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model").sort_index()


def find_cross_generator_results(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("cross_generator_results_*.csv"))


# def load_cross_generator_df(files: list[Path]) -> pd.DataFrame:
#     if not files:
#         return pd.DataFrame()
#     dfs = []
#     for f in files:
#         try:
#             tmp = pd.read_csv(f)
#             tmp["model"] = f.stem.replace("cross_generator_results_", "")
#             dfs.append(tmp)
#         except Exception:
#             continue
#     if not dfs:
#         return pd.DataFrame()
#     df = pd.concat(dfs, ignore_index=True)
#     # normalize label name
#     df = df.rename(columns=lambda c: c.strip())
#     df = df.replace({"nature": "real"})
#     return df


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_png(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metrics_comparison(metrics_df: pd.DataFrame, out: Path):
    if metrics_df.empty:
        return
    metrics = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in metrics_df.columns]
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df[metrics].plot.bar(ax=ax, rot=45, width=0.75, color=PALETTE[: len(metrics)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model metrics comparison")
    ax.legend(title="Metric")
    save_png(fig, out / "metrics_comparison_bar.png")


def plot_metrics_heatmap(metrics_df: pd.DataFrame, out: Path):
    if metrics_df.empty:
        return
    heat_metrics = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in metrics_df.columns]
    if not heat_metrics:
        return
    # heatmap: models x metrics
    data = metrics_df[heat_metrics].fillna(0)
    fig, ax = plt.subplots(figsize=(8, max(3, len(data) * 0.6)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "score"})
    ax.set_title("Metrics heatmap (models x metrics)")
    save_png(fig, out / "metrics_heatmap_models_metrics.png")


# def per_generator_summary_tables(cross_df: pd.DataFrame, out: Path):
#     if cross_df.empty:
#         return pd.DataFrame()
#     # group by generator and model
#     metrics_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in cross_df.columns]
#     grouped = cross_df.groupby(["generator", "model"])[metrics_cols].mean().reset_index()
#     grouped.to_csv(out / "per_generator_model_metrics.csv", index=False)
#     # pivot to generator x model (accuracy) for a heatmap
#     if "accuracy" in metrics_cols:
#         pivot = grouped.pivot(index="generator", columns="model", values="accuracy").fillna(0)
#         fig, ax = plt.subplots(figsize=(10, max(4, pivot.shape[0] * 0.4)))
#         sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu", ax=ax)
#         ax.set_title("Per-generator accuracy (models)")
#         save_png(fig, out / "heatmap_per_generator_accuracy.png")
#     return grouped


def plot_per_generator_bars(cross_df: pd.DataFrame, out: Path):
    if cross_df.empty:
        return
    # average accuracy per generator (across models)
    if "accuracy" not in cross_df.columns:
        return
    agg = cross_df.groupby("generator")["accuracy"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, len(agg) * 0.25)))
    sns.barplot(x=agg.values, y=agg.index, palette=PALETTE, ax=ax)
    ax.set_xlabel("Mean accuracy (across models)")
    ax.set_title("Generator difficulty (mean accuracy across models)")
    save_png(fig, out / "generator_mean_accuracy_barh.png")


def plot_class_distribution(data_root: Path, out: Path):
    counts = []
    for gen in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if gen.name.startswith("."):
            continue
        for split in ("train", "val"):
            p = gen / split
            if not p.exists():
                continue
            ai = sum(1 for _ in (p / "ai").glob("*.jpg")) if (p / "ai").exists() else 0
            nat = sum(1 for _ in (p / "nature").glob("*.jpg")) if (p / "nature").exists() else 0
            counts.append({"generator": gen.name, "split": split, "ai": ai, "real": nat})
    df = pd.DataFrame(counts)
    if df.empty:
        return
    # stacked bar for train split
    train = df[df["split"] == "train"].set_index("generator")[["ai", "real"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    train.sort_index().plot(kind="bar", stacked=True, color=[PALETTE[1], PALETTE[2]], ax=ax)
    ax.set_ylabel("Image count")
    ax.set_title("Train images per generator (ai vs real)")
    save_png(fig, out / "train_counts_by_generator_stacked.png")
    # pie chart global distribution
    totals = train.sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(totals.values, labels=totals.index.str.upper(), colors=[PALETTE[1], PALETTE[2]], autopct="%1.1f%%")
    ax.set_title("Overall train split composition (AI vs Real)")
    save_png(fig, out / "train_global_pie.png")


def create_sample_mosaics_global(data_root: Path, out: Path, n_per_class: int = 9):
    g = data_root / "general" / "val"
    if not g.exists():
        return
    def sample(subdir, n):
        imgs = [p for p in (subdir).glob("*.jpg") if not p.name.startswith(".")]
        if not imgs:
            return []
        np.random.shuffle(imgs)
        return imgs[:n]
    ai = sample(g / "ai", n_per_class)
    nat = sample(g / "nature", n_per_class)
    if ai:
        grid = make_grid(ai, thumb=200)
        Image.fromarray(grid).save(out / "sample_global_ai_grid.png")
    if nat:
        grid = make_grid(nat, thumb=200)
        Image.fromarray(grid).save(out / "sample_global_real_grid.png")


def make_grid(img_paths, grid_w=3, thumb=224):
    imgs = [Image.open(p).convert("RGB").resize((thumb, thumb)) for p in img_paths]
    rows = []
    for i in range(0, len(imgs), grid_w):
        row = np.hstack([np.asarray(im) for im in imgs[i : i + grid_w]])
        rows.append(row)
    return np.vstack(rows)


def create_sample_mosaics_per_generator(data_root: Path, out: Path, n=6):
    for gen in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if gen.name.startswith("."):
            continue
        v = gen / "val"
        if not v.exists():
            continue
        ai = [p for p in (v / "ai").glob("*.jpg") if not p.name.startswith(".")]
        nat = [p for p in (v / "nature").glob("*.jpg") if not p.name.startswith(".")]
        if not ai or not nat:
            continue
        ai = np.random.choice(ai, min(n, len(ai)), replace=False)
        nat = np.random.choice(nat, min(n, len(nat)), replace=False)
        try:
            Image.fromarray(make_grid(ai, thumb=160, grid_w=3)).save(out / f"{gen.name}_ai_grid.png")
            Image.fromarray(make_grid(nat, thumb=160, grid_w=3)).save(out / f"{gen.name}_real_grid.png")
        except Exception:
            continue


def plot_confusion_if_available(results_dir: Path, out: Path):
    # prefer raw pngs (_raw), otherwise csv textual matrices
    pngs = list(results_dir.glob("*_raw.png")) + list(results_dir.glob("*confusion_matrix*.png"))
    if pngs:
        for p in pngs:
            (out / p.name).write_bytes(p.read_bytes())
        return
    csvs = list(results_dir.glob("*confusion_matrix*.csv")) + list(results_dir.glob("*confusion_matrix*.txt"))
    for c in csvs:
        try:
            cm = pd.read_csv(c, index_col=0).values
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_title("Confusion matrix")
            save_png(fig, out / "confusion_matrix_from_csv.png")
            return
        except Exception:
            continue


def plot_calibration_and_thresholds(results_dir: Path, out: Path):
    # temperature file (single value) and thresholds json
    temp_files = list(results_dir.glob("temperature_*.txt"))
    thr_files = list(results_dir.glob("thresholds_*.json"))
    if not temp_files and not thr_files:
        return
    # calibration: if temp exists show value
    if temp_files:
        try:
            t = float(temp_files[0].read_text().strip())
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh([0], [t], color=PALETTE[3])
            ax.set_xlim(0, max(0.5, t * 1.2))
            ax.set_yticks([])
            ax.set_title(f"Temperature scaling value ({temp_files[0].name}): {t:.3f}")
            save_png(fig, out / "temperature_value.png")
        except Exception:
            pass
    if thr_files:
        try:
            j = json.loads(thr_files[0].read_text())
            # try to find ai threshold
            th = j.get("global", {}).get("best_acc", {}).get("threshold", None) or j.get("threshold", None)
            if th is not None:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh([0], [th], color=PALETTE[4])
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_title(f"Chosen threshold (from {thr_files[0].name}): {th:.3f}")
                save_png(fig, out / "chosen_threshold.png")
        except Exception:
            pass


def plot_roc_pr_if_predictions(results_dir: Path, out: Path):
    # Look for per-model prediction files named predictions_{model}.csv with columns: y_true,y_score
    preds = list(results_dir.glob("predictions_*.csv")) + list(results_dir.glob("preds_*.csv"))
    if not preds:
        return
    for p in preds:
        try:
            df = pd.read_csv(p)
            if not {"y_true", "y_score"}.issubset(df.columns):
                continue
            y_true = df["y_true"].values
            y_score = df["y_score"].values
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            # ROC
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color=PALETTE[1], label=f"AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="#bbbbbb")
            ax.set_title(f"ROC ({p.stem})")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            save_png(fig, out / f"{p.stem}_roc.png")
            # PR
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(recall, precision, color=PALETTE[2], label=f"AP={ap:.3f}")
            ax.set_title(f"Precision-Recall ({p.stem})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            save_png(fig, out / f"{p.stem}_pr.png")
        except Exception:
            continue


def compute_embeddings_and_umap(data_root: Path, model_path: Path, out: Path, sample_per_gen=200):
    if torch is None or timm is None:
        print("[report] Skipping embeddings: torch/timm not available in environment.")
        return
    # load model (backbone only), extract penultimate features
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        # create model by name inference from filename when possible
        model_name = "swin_tiny_patch4_window7_224"
        model = timm.create_model(model_name, pretrained=False, num_classes=0)  # return features
        ckpt = torch.load(model_path, map_location=device)
        # try to load state_dict safely
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print("[report] Embedding extraction failed:", e)
        return

    # sample images across generators (val) up to sample_per_gen each
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    imgs = []
    meta = []
    for gen in sorted(p for p in data_root.iterdir() if p.is_dir()):
        v = gen / "val"
        if not v.exists():
            continue
        files = [p for p in v.glob("**/*.jpg") if not p.name.startswith(".")]
        if not files:
            continue
        sel = files[: min(sample_per_gen, len(files))]
        for p in sel:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(transform(im))
                meta.append({"path": str(p), "generator": gen.name, "label": "real" if "nature" in str(p) else "ai"})
            except Exception:
                continue
    if not imgs:
        return
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model.forward_features(batch) if hasattr(model, "forward_features") else model(batch)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
    feats_np = feats.cpu().numpy()
    # reduce with PCA then TSNE (or UMAP if available)
    try:
        pca = PCA(n_components=min(50, feats_np.shape[1]))
        z = pca.fit_transform(feats_np)
        tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        z2 = tsne.fit_transform(z)
        df = pd.DataFrame(z2, columns=("x", "y"))
        df = pd.concat([df, pd.DataFrame(meta)], axis=1)
        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x="x", y="y", hue="generator", style="label", palette="tab10", ax=ax, s=40)
        ax.set_title("t-SNE of model embeddings (val subset)")
        save_png(fig, out / "embeddings_tsne.png")
        df.to_csv(out / "embeddings_meta.csv", index=False)
    except Exception as e:
        print("[report] Embedding reduction failed:", e)


def build_html_report(out: Path, assets: list[Path], tables: dict[str, pd.DataFrame], html_path: Path):
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>Model Report</title>")
    parts.append(
        "<style>body{font-family:Inter,system-ui, -apple-system, 'Segoe UI', Roboto, Arial; padding:28px; background:#fafafa; color:#222} .card{background:#fff;padding:16px;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:18px} h1{color:#264653} table{border-collapse:collapse;width:100%} th,td{padding:8px;border-bottom:1px solid #eee;text-align:left} .center{display:flex;gap:16px;flex-wrap:wrap}</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>AI Image Detector — Detailed Report</h1>")
    parts.append("<div class='card'><h2>Summary Metrics</h2>")
    if "metrics" in tables and not tables["metrics"].empty:
        parts.append(tables["metrics"].to_html(index=True, float_format="{:.4f}".format))
    parts.append("</div>")

    parts.append("<div class='card'><h2>Generator Summary</h2>")
    if "generator_summary" in tables:
        parts.append(tables["generator_summary"].to_html(index=False))
    parts.append("</div>")

    parts.append("<div class='card'><h2>Assets</h2><div class='center'>")
    for asset in assets:
        parts.append(f"<div style='width:360px'><h4>{asset.name}</h4><img src='{asset.name}' style='width:100%;border-radius:6px' /></div>")
    parts.append("</div></div>")

    parts.append("</body></html>")
    html = "\n".join(parts)
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML report to {html_path}")


def main(
    results_dir: Path = Path("results"),
    data_root: Path = Path("data/standardized_jpg"),
    output_dir: Path = Path("results/report_assets"),
    model: Optional[str] = None,
    model_path: Optional[Path] = None,
    embeddings: bool = False,
):
    safe_mkdir(output_dir)

    metrics_files = find_metrics_files(results_dir)
    metrics_df = load_metrics(metrics_files)
    if model and model in metrics_df.index:
        metrics_df = metrics_df.loc[[model]]
    consolidated_csv = output_dir / "consolidated_metrics.csv"
    metrics_df.to_csv(consolidated_csv)
    print(f"Wrote consolidated metrics to {consolidated_csv}")

    # # cross-generator aggregation
    # cross_files = find_cross_generator_results(results_dir)
    # cross_df = load_cross_generator_df(cross_files)
    # if not cross_df.empty:
    #     cross_df.to_csv(output_dir / "cross_generator_combined.csv", index=False)
    #     print("Wrote cross-generator combined CSV.")

    # Charts and tables
    plot_metrics_comparison(metrics_df, output_dir)
    plot_metrics_heatmap(metrics_df, output_dir)
    # grouped = per_generator_summary_tables(cross_df, output_dir)
    # plot_per_generator_bars(cross_df, output_dir)

    plot_class_distribution(data_root, output_dir)
    create_sample_mosaics_global(data_root, output_dir)
    create_sample_mosaics_per_generator(data_root, output_dir)
    plot_confusion_if_available(results_dir, output_dir)
    plot_calibration_and_thresholds(results_dir, output_dir)
    plot_roc_pr_if_predictions(results_dir, output_dir)

    # generator summary table
    gen_rows = []
    for gen in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if gen.name.startswith("."):
            continue
        train_ai = sum(1 for _ in (gen / "train" / "ai").glob("*.jpg")) if (gen / "train" / "ai").exists() else 0
        train_nat = sum(1 for _ in (gen / "train" / "nature").glob("*.jpg")) if (gen / "train" / "nature").exists() else 0
        val_ai = sum(1 for _ in (gen / "val" / "ai").glob("*.jpg")) if (gen / "val" / "ai").exists() else 0
        val_nat = sum(1 for _ in (gen / "val" / "nature").glob("*.jpg")) if (gen / "val" / "nature").exists() else 0
        gen_rows.append({"generator": gen.name, "train_ai": train_ai, "train_real": train_nat, "val_ai": val_ai, "val_real": val_nat})
    gen_df = pd.DataFrame(gen_rows).sort_values("generator")
    gen_df.to_csv(output_dir / "generator_summary.csv", index=False)
    print(f"Wrote generator_summary.csv with {len(gen_df)} rows")

    # optional embeddings
    if embeddings and model_path:
        compute_embeddings_and_umap(data_root, model_path, output_dir)

    # copy PNG/JPG assets to list for HTML
    assets = [p for p in sorted(output_dir.iterdir()) if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    build_html_report(output_dir, assets, {"metrics": metrics_df, "generator_summary": gen_df}, output_dir / "report.html")

    # model checksum metadata
    try:
        if model_path and model_path.exists():
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            checksum = h.hexdigest()
            info = {"model": model, "model_path": str(model_path), "checksum": checksum, "checksum_short": checksum[:12]}
            (output_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"Wrote model_info.json with checksum {checksum[:12]}")
    except Exception:
        pass

    print("Report generation finished. Assets in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-root", type=str, default="data/standardized_jpg")
    parser.add_argument("--output-dir", type=str, default="results/report_assets")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Path to model .pth for checksum / embeddings")
    parser.add_argument("--embeddings", action="store_true", help="Compute embeddings & t-SNE (requires torch+timm)")
    args = parser.parse_args()
    main(Path(args.results_dir), Path(args.data_root), Path(args.output_dir), args.model, Path(args.model_path) if args.model_path else None, args.embeddings)
# ...existing code...