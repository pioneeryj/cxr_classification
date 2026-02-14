"""Visualize test results and calibration metrics across epochs."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "outputs_testinglosses/BioViL_posweight=True_resampling=False_lora_r=16"
EPOCHS = list(range(8))  # 0~7


def load_epoch_data():
    """Load test_results and calibration CSVs for all epochs."""
    test_results = []
    calibration_results = []

    for ep in EPOCHS:
        ep_dir = os.path.join(BASE_DIR, f"BioViL_{ep}ep.pt", "biovil_post_hoc")

        # Test results
        tr_path = os.path.join(ep_dir, "test_results_uncalibrated.csv")
        tr = pd.read_csv(tr_path, index_col=0)
        tr['epoch'] = ep
        test_results.append(tr)

        # Calibration results
        cal_path = os.path.join(ep_dir, "calibration_uncalibrated.csv")
        cal = pd.read_csv(cal_path)
        cal['epoch'] = ep
        calibration_results.append(cal)

    return pd.concat(test_results), pd.concat(calibration_results)


def plot_test_results(df):
    """Plot classification metrics (AUC, AP, Accuracy, F1) across epochs."""
    metrics = ['AUC', 'AP', 'Accuracy', 'F1']
    diseases = df.index.unique()
    # Remove epoch from diseases if present
    diseases = [d for d in diseases if d != 'epoch']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classification Metrics across Epochs\n(BioViL, pos_weight=True, resampling=False, LoRA r=16)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, len(diseases)))

    for ax, metric in zip(axes.flatten(), metrics):
        for i, disease in enumerate(diseases):
            disease_data = df.loc[disease]
            epochs = disease_data['epoch'].values
            values = disease_data[metric].values
            ax.plot(epochs, values, marker='o', label=disease, color=colors[i],
                    linewidth=1.5, markersize=4)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(EPOCHS)
        ax.grid(True, alpha=0.3)

    # Single legend below the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(BASE_DIR, 'classification_metrics_by_epoch.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_calibration_results(df):
    """Plot calibration metrics (Brier Score, ECE, AURC) across epochs."""
    metrics = ['brier_score', 'ece', 'aurc']
    metric_labels = {'brier_score': 'Brier Score', 'ece': 'ECE', 'aurc': 'AURC'}

    # Get label names (exclude Overall)
    all_labels = df[df['label_name'] != 'Overall']['label_name'].unique()
    # Include Overall as a special entry
    entries = list(all_labels) + ['Overall']

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Calibration Metrics across Epochs\n(BioViL, pos_weight=True, resampling=False, LoRA r=16)',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, len(entries)))

    for ax, metric in zip(axes, metrics):
        for i, label_name in enumerate(entries):
            subset = df[df['label_name'] == label_name].sort_values('epoch')
            linewidth = 2.5 if label_name == 'Overall' else 1.5
            markersize = 7 if label_name == 'Overall' else 4
            linestyle = '--' if label_name == 'Overall' else '-'
            ax.plot(subset['epoch'], subset[metric], marker='o', label=label_name,
                    color=colors[i], linewidth=linewidth, markersize=markersize,
                    linestyle=linestyle)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xticks(EPOCHS)
        ax.grid(True, alpha=0.3)

    # Single legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_path = os.path.join(BASE_DIR, 'calibration_metrics_by_epoch.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_df, cal_df = load_epoch_data()
    plot_test_results(test_df)
    plot_calibration_results(cal_df)
    print("Done.")
