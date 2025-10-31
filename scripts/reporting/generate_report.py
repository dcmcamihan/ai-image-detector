#!/usr/bin/env python3
"""
Generate documentation-ready CSVs and visualizations from the app database.
Outputs go to reports/ with timestamped subfolder by default.

Usage:
  python scripts/reporting/generate_report.py \
    --db web_app_data/app.db \
    --outdir reports \
    --limit 0

If --limit > 0, only the most recent N rows are used (useful for quick runs).
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Optional plotting imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def read_db(db_path: Path, limit: int = 0) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    if limit and limit > 0:
        sql = (
            "SELECT * FROM predictions ORDER BY datetime(created_at) DESC LIMIT ?"
        )
        df = pd.read_sql_query(sql, con, params=(int(limit),))
    else:
        df = pd.read_sql_query("SELECT * FROM predictions", con)
    con.close()
    # Parse JSON columns
    def _parse_json_safe(s):
        try:
            return json.loads(s) if isinstance(s, str) and s else None
        except Exception:
            return None
    if 'model_version' in df.columns:
        df['model_version_parsed'] = df['model_version'].apply(_parse_json_safe)
        df['model_checksum'] = df['model_version_parsed'].apply(
            lambda d: (d or {}).get('checksum') if isinstance(d, dict) else None
        )
    if 'generator_json' in df.columns:
        df['generator_json_parsed'] = df['generator_json'].apply(_parse_json_safe)
    # Normalize created_at to date
    if 'created_at' in df.columns:
        df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['created_date'] = df['created_at_dt'].dt.date
    return df


def ensure_outdir(base_outdir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base_outdir / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    return out


def export_csvs(df: pd.DataFrame, out: Path):
    df.to_csv(out / "all_predictions.csv", index=False)
    if 'prediction' in df.columns:
        df[df['prediction'] == 'ai'].to_csv(out / "ai_predictions.csv", index=False)
        df[df['prediction'] == 'nature'].to_csv(out / "real_predictions.csv", index=False)
    # Flatten generator_json into wide columns
    if 'generator_json_parsed' in df.columns:
        gens = []
        for _, r in df.iterrows():
            gj = r.get('generator_json_parsed')
            if isinstance(gj, dict):
                row = {'id': r.get('id')}
                for k, v in gj.items():
                    row[k] = v
                gens.append(row)
        if gens:
            gdf = pd.DataFrame(gens).fillna(0.0)
            gdf.to_csv(out / "generator_likelihoods_wide.csv", index=False)
            # Also a long format for charts
            glong = gdf.melt(id_vars=['id'], var_name='generator', value_name='prob')
            glong.to_csv(out / "generator_likelihoods_long.csv", index=False)


def setup_style():
    if HAS_MPL:
        plt.rcParams.update({
            'figure.dpi': 140,
            'savefig.dpi': 220,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.facecolor': '#FCFCFD',
            'figure.facecolor': '#FFFFFF',
        })
        sns.set_theme(style="whitegrid", context="talk")


# Consistent palette
PALETTE = {
    'ai': '#e74c3c',          # Red-ish
    'nature': '#2ecc71',      # Green
    'accent1': '#3366CC',
    'accent2': '#8E44AD',
    'accent3': '#F39C12',
    'heatmap': 'YlGnBu',
}

LABEL_PALETTE = {'ai': PALETTE['ai'], 'nature': PALETTE['nature']}

def _beautify_axes(ax):
    for spine in ['top', 'right']:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.15)


def plot_distribution(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'prediction' not in df.columns:
        return
    counts = df['prediction'].value_counts().rename_axis('label').reset_index(name='count')
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    sns.barplot(data=counts, x='label', y='count', hue='label', palette=LABEL_PALETTE, legend=False, ax=ax)
    ax.set_title('Prediction Distribution')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    for i, r in counts.iterrows():
        ax.text(i, r['count'], str(int(r['count'])), ha='center', va='bottom', fontsize=9)
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'distribution_bar.png')
    plt.close(fig)

    # Donut (pie) chart
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    colors = [LABEL_PALETTE.get(lbl, PALETTE['accent3']) for lbl in counts['label']]
    wedges, texts, autotexts = ax.pie(
        counts['count'], labels=counts['label'], autopct='%1.1f%%', colors=colors,
        pctdistance=0.75, textprops={'color': '#2c3e50'}
    )
    # center circle
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title('Prediction Distribution')
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'distribution_pie.png')
    plt.close(fig)

    # Plotly interactive bar
    if HAS_PLOTLY:
        fig_px = px.bar(counts, x='label', y='count', color='label', color_discrete_map=LABEL_PALETTE,
                        title='Prediction Distribution')
        fig_px.update_layout(template='plotly_white')
        fig_px.write_html(str(out / 'figs' / 'distribution_bar.html'))


def plot_confidence(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'confidence' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sns.histplot(data=df, x='confidence', hue='prediction', bins=30, kde=True,
                 palette=LABEL_PALETTE, ax=ax, alpha=0.45)
    ax.set_title('Confidence Histogram by Prediction')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'confidence_hist.png')
    plt.close(fig)

    # Violin + box plot of confidence by label
    try:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        sns.violinplot(data=df, x='prediction', y='confidence', palette=LABEL_PALETTE, inner=None, ax=ax)
        sns.boxplot(data=df, x='prediction', y='confidence', width=0.2, showcaps=True,
                    boxprops={'facecolor':'#ffffff80', 'zorder': 3},
                    whiskerprops={'linewidth':1}, medianprops={'color':'#2c3e50'}, ax=ax)
        ax.set_title('Confidence by Prediction (Violin + Box)')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Confidence')
        _beautify_axes(ax)
        fig.tight_layout()
        fig.savefig(out / 'figs' / 'confidence_violin.png')
        plt.close(fig)
    except Exception:
        pass


def plot_daily_confidence(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'created_date' not in df.columns or 'confidence' not in df.columns:
        return
    daily = df.groupby('created_date')['confidence'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    sns.lineplot(data=daily, x='created_date', y='confidence', marker='o', color=PALETTE['accent1'], ax=ax)
    ax.set_title('Average Confidence Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg Confidence')
    _beautify_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'avg_conf_over_time.png')
    plt.close(fig)


def plot_inference_time(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'inference_time_ms' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    sns.histplot(df['inference_time_ms'], bins=30, color=PALETTE['accent2'], ax=ax)
    ax.set_title('Inference Time (ms)')
    ax.set_xlabel('ms')
    ax.set_ylabel('Count')
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'inference_time_hist.png')
    plt.close(fig)


def plot_blur_vs_conf(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'blur_score' not in df.columns or 'confidence' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.scatterplot(data=df, x='blur_score', y='confidence', hue='prediction',
                    palette=LABEL_PALETTE, ax=ax, alpha=0.8, s=26)
    ax.set_title('Blur Score vs Confidence')
    ax.set_xlabel('Blur Score')
    ax.set_ylabel('Confidence')
    ax.legend(title='Prediction')
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'blur_vs_confidence.png')
    plt.close(fig)


def plot_model_versions(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'model_checksum' not in df.columns:
        return
    cnt = df['model_checksum'].fillna('unknown').value_counts().rename_axis('checksum').reset_index(name='count')
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.barplot(data=cnt, x='checksum', y='count', color=PALETTE['accent1'], ax=ax)
    ax.set_title('Predictions by Model Checksum')
    ax.set_xlabel('Checksum')
    ax.set_ylabel('Count')
    for i, r in cnt.iterrows():
        ax.text(i, r['count'], str(int(r['count'])), ha='center', va='bottom', fontsize=8, rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'preds_by_model_checksum.png')
    plt.close(fig)


def plot_generator_heatmap(df: pd.DataFrame, out: Path):
    if not HAS_MPL or 'generator_json_parsed' not in df.columns:
        return
    # Build long format probabilities
    rows = []
    for _, r in df.iterrows():
        gj = r.get('generator_json_parsed')
        if isinstance(gj, dict):
            for k, v in gj.items():
                rows.append({'id': r.get('id'), 'generator': k, 'prob': float(v), 'prediction': r.get('prediction')})
    if not rows:
        return
    glong = pd.DataFrame(rows)
    # Heatmap: average generator prob per predicted label
    piv = glong.pivot_table(index='generator', columns='prediction', values='prob', aggfunc='mean', fill_value=0.0)
    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.45 * len(piv))))
    sns.heatmap(piv, annot=True, fmt='.2f', cmap=PALETTE['heatmap'], linewidths=0.4, linecolor='#FFFFFF',
                cbar_kws={'shrink': 0.8, 'label': 'Avg prob'}, ax=ax)
    ax.set_title('Avg Generator Likelihoods by Prediction')
    _beautify_axes(ax)
    fig.tight_layout()
    fig.savefig(out / 'figs' / 'generator_likelihoods_heatmap.png')
    plt.close(fig)
    # Plotly interactive heatmap
    if HAS_PLOTLY:
        fig_px = px.imshow(piv, color_continuous_scale='YlGnBu', origin='upper', aspect='auto')
        fig_px.update_layout(title='Avg Generator Likelihoods by Prediction', template='plotly_white')
        fig_px.write_html(str(out / 'figs' / 'generator_likelihoods_heatmap.html'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('web_app_data/app.db'))
    ap.add_argument('--outdir', type=Path, default=Path('reports'))
    ap.add_argument('--limit', type=int, default=0, help='Only use most recent N rows if > 0')
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = ensure_outdir(args.outdir)

    df = read_db(args.db, limit=args.limit)

    # Save a metadata summary
    meta = {
        'rows': int(len(df)),
        'time_range': {
            'min': (df['created_at'].min() if 'created_at' in df.columns and len(df) else None),
            'max': (df['created_at'].max() if 'created_at' in df.columns and len(df) else None),
        },
        'generated_at': datetime.now().isoformat(),
        'has_plotly': HAS_PLOTLY,
        'has_matplotlib': HAS_MPL,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))

    export_csvs(df, out)

    setup_style()
    plot_distribution(df, out)
    plot_confidence(df, out)
    plot_daily_confidence(df, out)
    plot_inference_time(df, out)
    plot_blur_vs_conf(df, out)
    plot_model_versions(df, out)
    plot_generator_heatmap(df, out)

    print(f"Reports written to: {out}")


if __name__ == '__main__':
    main()
