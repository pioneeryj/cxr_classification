"""Compare calibration metrics across all resnet50 experiment variants."""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "outputs_testinglosses"
METRICS = ['brier_score', 'ece', 'aurc']
METRIC_LABELS = {'brier_score': 'Brier Score', 'ece': 'ECE', 'aurc': 'AURC'}


def load_calibration_csv(csv_path):
    """Load a calibration CSV (unified or single-metric format)."""
    df = pd.read_csv(csv_path)
    return df


def find_and_load_all():
    """Find all resnet50 calibration CSVs, return structured data.

    Returns:
        List of dicts: [{
            'experiment': str,   # e.g. 'resnet50_8ep (uncalibrated)'
            'label_name': str,
            'brier_score': float,
            'ece': float,
            'aurc': float,
        }, ...]
    """
    records = []
    resnet_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "resnet50*/")))

    for exp_dir in resnet_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))

        # Find all calibration CSVs recursively
        csv_files = glob.glob(os.path.join(exp_dir, "**", "calibration_*.csv"), recursive=True)

        if not csv_files:
            continue

        # Group by calibration type (uncalibrated, global_t, label_wise_t)
        # Detect unified format vs split format
        unified_csvs = [f for f in csv_files if os.path.basename(f) in
                        ['calibration_uncalibrated.csv', 'calibration_global_t.csv', 'calibration_label_wise_t.csv']]
        split_csvs = [f for f in csv_files if f not in unified_csvs]

        # Handle unified CSVs (have all 3 metrics in one file)
        for csv_path in unified_csvs:
            fname = os.path.basename(csv_path)
            cal_type = fname.replace('calibration_', '').replace('.csv', '')
            label = f"{exp_name} ({cal_type})"

            df = load_calibration_csv(csv_path)
            for _, row in df.iterrows():
                rec = {
                    'experiment': label,
                    'label_name': row['label_name'],
                }
                for m in METRICS:
                    rec[m] = row.get(m, None)
                records.append(rec)

        # Handle split CSVs (one metric per file)
        if split_csvs:
            # Group by calibration type
            type_metrics = {}  # {cal_type: {metric: df}}
            for csv_path in split_csvs:
                fname = os.path.basename(csv_path)
                # e.g. calibration_uncalibrated_ece.csv -> cal_type=uncalibrated, metric=ece
                parts = fname.replace('calibration_', '').replace('.csv', '')
                for m in METRICS:
                    if parts.endswith(f'_{m}'):
                        cal_type = parts[:-(len(m) + 1)]
                        if cal_type not in type_metrics:
                            type_metrics[cal_type] = {}
                        type_metrics[cal_type][m] = load_calibration_csv(csv_path)
                        break

            for cal_type, metric_dfs in type_metrics.items():
                label = f"{exp_name} ({cal_type})"

                # Merge all metrics by label_name
                merged = None
                for m, df in metric_dfs.items():
                    df_sub = df[['label_name', m]].copy()
                    if merged is None:
                        merged = df_sub
                    else:
                        merged = merged.merge(df_sub, on='label_name', how='outer')

                if merged is not None:
                    for _, row in merged.iterrows():
                        rec = {
                            'experiment': label,
                            'label_name': row['label_name'],
                        }
                        for m in METRICS:
                            rec[m] = row.get(m, None)
                        records.append(rec)

    return pd.DataFrame(records)


def plot_overall_comparison(df):
    """Bar chart comparing Overall calibration metrics across all experiments."""
    overall = df[df['label_name'] == 'Overall'].copy()

    if overall.empty:
        print("No 'Overall' rows found.")
        return

    experiments = overall['experiment'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Calibration Metrics Comparison (Overall)\nResNet50 Variants',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    for ax, metric in zip(axes, METRICS):
        values = overall[metric].values.astype(float)
        bars = ax.bar(range(len(experiments)), values, color=colors)

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(BASE_DIR, 'calibration_overall_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_label_comparison(df):
    """Grouped bar chart comparing per-label calibration metrics across experiments."""
    per_label = df[df['label_name'] != 'Overall'].copy()

    if per_label.empty:
        print("No per-label rows found.")
        return

    experiments = per_label['experiment'].unique().tolist()
    labels = per_label['label_name'].unique().tolist()

    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Per-Label Calibration Metrics Comparison\nResNet50 Variants',
                 fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, METRICS):
        x = np.arange(len(labels))
        width = 0.8 / len(experiments)

        for i, exp in enumerate(experiments):
            exp_data = per_label[per_label['experiment'] == exp]
            values = []
            for lbl in labels:
                row = exp_data[exp_data['label_name'] == lbl]
                if not row.empty and pd.notna(row[metric].values[0]):
                    values.append(float(row[metric].values[0]))
                else:
                    values.append(0.0)

            offset = (i - len(experiments) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=exp, color=colors[i])

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Single legend at bottom
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = os.path.join(BASE_DIR, 'calibration_per_label_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = find_and_load_all()
    print(f"Loaded {len(df)} rows from {df['experiment'].nunique()} experiments")
    print(f"Experiments: {df['experiment'].unique().tolist()}")
    plot_overall_comparison(df)
    plot_per_label_comparison(df)
    print("Done.")
