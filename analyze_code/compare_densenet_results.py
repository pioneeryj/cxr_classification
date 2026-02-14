"""Compare model results: BCE vs Uncalibrated vs Global-T vs Label-T.
Generates per-model visualizations (png) and overall summary (txt).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# Config
# ============================================================
OUTPUT_BASE = "/home/yoonji/mrg/cxr_classification/outputs"
PAPER_BASE = "/home/yoonji/mrg/paper_comparison/classification_calibration"

MODELS = {
    "densenet": {
        "display": "DenseNet121",
        "bce_dir": os.path.join(OUTPUT_BASE, "densenet121_bce"),
        "exp_dir": os.path.join(OUTPUT_BASE, "densenet121_experiment"),
    },
    "resnet": {
        "display": "ResNet50",
        "bce_dir": os.path.join(OUTPUT_BASE, "resnet50_calib_distill"),
        "exp_dir": os.path.join(OUTPUT_BASE, "resnet50_experiment"),
    },
    "biovil": {
        "display": "BioViL",
        "bce_dir": os.path.join(OUTPUT_BASE, "biovil_bce"),
        "exp_dir": os.path.join(OUTPUT_BASE, "biovil_experiment"),
    },
}

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices"
]

COND_NAMES = ["BCE", "Uncalibrated", "Global-T", "Label-T"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})


def get_cal_value(cal_dict, cond, label, col):
    df = cal_dict[cond]
    row = df[df["label_name"] == label]
    return row[col].values[0]


def process_model(model_key, model_info):
    display_name = model_info["display"]
    bce_dir = model_info["bce_dir"]
    exp_dir = model_info["exp_dir"]
    out_dir = os.path.join(PAPER_BASE, model_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {display_name}")
    print(f"  BCE dir: {bce_dir}")
    print(f"  EXP dir: {exp_dir}")
    print(f"  Output:  {out_dir}")
    print(f"{'='*60}")

    # --- Load data ---
    cls = {
        "BCE": pd.read_csv(os.path.join(bce_dir, "test_results_uncalibrated.csv"), index_col=0),
        "Uncalibrated": pd.read_csv(os.path.join(exp_dir, "test_results_uncalibrated.csv"), index_col=0),
        "Global-T": pd.read_csv(os.path.join(exp_dir, "test_results_global_t.csv"), index_col=0),
        "Label-T": pd.read_csv(os.path.join(exp_dir, "test_results_label_wise_t.csv"), index_col=0),
    }

    ece = {
        "BCE": pd.read_csv(os.path.join(bce_dir, "calibration_uncalibrated_ece.csv")),
        "Uncalibrated": pd.read_csv(os.path.join(exp_dir, "calibration_uncalibrated_ece.csv")),
        "Global-T": pd.read_csv(os.path.join(exp_dir, "calibration_global_t_ece.csv")),
        "Label-T": pd.read_csv(os.path.join(exp_dir, "calibration_label_wise_t_ece.csv")),
    }

    brier = {
        "BCE": pd.read_csv(os.path.join(bce_dir, "calibration_uncalibrated_brier_score.csv")),
        "Uncalibrated": pd.read_csv(os.path.join(exp_dir, "calibration_uncalibrated_brier_score.csv")),
        "Global-T": pd.read_csv(os.path.join(exp_dir, "calibration_global_t_brier_score.csv")),
        "Label-T": pd.read_csv(os.path.join(exp_dir, "calibration_label_wise_t_brier_score.csv")),
    }

    aurc = {
        "BCE": pd.read_csv(os.path.join(bce_dir, "calibration_uncalibrated_aurc.csv")),
        "Uncalibrated": pd.read_csv(os.path.join(exp_dir, "calibration_uncalibrated_aurc.csv")),
        "Global-T": pd.read_csv(os.path.join(exp_dir, "calibration_global_t_aurc.csv")),
        "Label-T": pd.read_csv(os.path.join(exp_dir, "calibration_label_wise_t_aurc.csv")),
    }

    cal_map = {"ECE": (ece, "ece"), "Brier": (brier, "brier_score"), "AURC": (aurc, "aurc")}

    # --- 1. Classification: AUC, AP ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, metric in zip(axes, ["AUC", "AP"]):
        x = np.arange(len(LABELS))
        width = 0.2
        for i, cond in enumerate(COND_NAMES):
            vals = [cls[cond].loc[l, metric] for l in LABELS]
            ax.bar(x + i * width, vals, width, label=cond, color=COLORS[i], alpha=0.85)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per Label")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"{display_name}: Classification Metrics (AUC, AP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classification_auc_ap.png"))
    plt.close()
    print(f"  Saved: classification_auc_ap.png")

    # --- 2. Classification: Accuracy, F1 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, metric in zip(axes, ["Accuracy", "F1"]):
        x = np.arange(len(LABELS))
        width = 0.2
        for i, cond in enumerate(COND_NAMES):
            vals = [cls[cond].loc[l, metric] for l in LABELS]
            ax.bar(x + i * width, vals, width, label=cond, color=COLORS[i], alpha=0.85)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per Label")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"{display_name}: Classification Metrics (Accuracy, F1)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classification_acc_f1.png"))
    plt.close()
    print(f"  Saved: classification_acc_f1.png")

    # --- 3. Calibration: ECE, Brier, AURC ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    cal_info = [
        (ece, "ece", "ECE (↓)"),
        (brier, "brier_score", "Brier Score (↓)"),
        (aurc, "aurc", "AURC (↓)"),
    ]
    for ax, (cal_dict, col, title) in zip(axes, cal_info):
        x = np.arange(len(LABELS))
        width = 0.2
        for i, cond in enumerate(COND_NAMES):
            vals = [get_cal_value(cal_dict, cond, l, col) for l in LABELS]
            ax.bar(x + i * width, vals, width, label=cond, color=COLORS[i], alpha=0.85)
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Label")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"{display_name}: Calibration Metrics Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_metrics.png"))
    plt.close()
    print(f"  Saved: calibration_metrics.png")

    # --- 4. Heatmap ---
    all_metric_names = ["AUC", "AP", "Accuracy", "F1", "ECE", "Brier", "AURC"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    for ax, cond in zip(axes.flat, COND_NAMES):
        data = np.zeros((len(LABELS), len(all_metric_names)))
        for j, metric in enumerate(all_metric_names):
            for i, label in enumerate(LABELS):
                if metric in ["AUC", "AP", "Accuracy", "F1"]:
                    data[i, j] = cls[cond].loc[label, metric]
                else:
                    cd, col = cal_map[metric]
                    data[i, j] = get_cal_value(cd, cond, label, col)
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(len(all_metric_names)))
        ax.set_yticks(np.arange(len(LABELS)))
        ax.set_xticklabels(all_metric_names, fontsize=9)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_title(cond, fontsize=12, fontweight='bold')
        for i in range(len(LABELS)):
            for j in range(len(all_metric_names)):
                val = data[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=6, color=color)
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle(f"{display_name}: All Metrics Heatmap per Condition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_all.png"))
    plt.close()
    print(f"  Saved: heatmap_all.png")

    # --- 5. Overall bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cls_overall = ["AUC", "AP", "Accuracy", "F1"]
    x = np.arange(len(cls_overall))
    width = 0.18
    for i, cond in enumerate(COND_NAMES):
        vals = [cls[cond][m].mean() for m in cls_overall]
        axes[0].bar(x + i * width, vals, width, label=cond, color=COLORS[i], alpha=0.85)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Classification Metrics")
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels(cls_overall)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    cal_overall = ["ECE", "Brier", "AURC"]
    x = np.arange(len(cal_overall))
    for i, cond in enumerate(COND_NAMES):
        vals = [get_cal_value(cal_map[m][0], cond, "Overall", cal_map[m][1]) for m in cal_overall]
        axes[1].bar(x + i * width, vals, width, label=cond, color=COLORS[i], alpha=0.85)
    axes[1].set_ylabel("Score (lower is better)")
    axes[1].set_title("Overall Calibration Metrics")
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels(cal_overall)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle(f"{display_name}: Overall Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_comparison.png"))
    plt.close()
    print(f"  Saved: overall_comparison.png")

    # --- 6. Summary text ---
    lines = []
    lines.append("=" * 80)
    lines.append(f"{display_name}: 4-Condition Comparison Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Conditions:")
    lines.append(f"  BCE          = {os.path.basename(bce_dir)} (BCE loss, uncalibrated test)")
    lines.append(f"  Uncalibrated = {os.path.basename(exp_dir)} (uncalibrated test)")
    lines.append(f"  Global-T     = {os.path.basename(exp_dir)} (global temperature scaling)")
    lines.append(f"  Label-T      = {os.path.basename(exp_dir)} (label-wise temperature scaling)")
    lines.append("")

    header = f"{'Metric':<12}" + "".join(f"{c:>15}" for c in COND_NAMES) + f"{'Best':>15}"

    lines.append("-" * 80)
    lines.append("Classification Metrics (higher is better)")
    lines.append("-" * 80)
    lines.append(header)
    lines.append("-" * 80)
    for metric in ["AUC", "AP", "Accuracy", "F1"]:
        vals = {c: cls[c][metric].mean() for c in COND_NAMES}
        best = max(vals, key=vals.get)
        row = f"{metric:<12}" + "".join(f"{vals[c]:>15.6f}" for c in COND_NAMES) + f"{best:>15}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 80)
    lines.append("Calibration Metrics (lower is better)")
    lines.append("-" * 80)
    lines.append(header)
    lines.append("-" * 80)
    for m_name in ["ECE", "Brier", "AURC"]:
        cd, col = cal_map[m_name]
        vals = {c: get_cal_value(cd, c, "Overall", col) for c in COND_NAMES}
        best = min(vals, key=vals.get)
        row = f"{m_name:<12}" + "".join(f"{vals[c]:>15.6f}" for c in COND_NAMES) + f"{best:>10}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 80)
    lines.append("Per-Label Detail")
    lines.append("=" * 80)

    for metric in ["AUC", "AP", "ECE", "Brier", "AURC"]:
        lines.append("")
        lines.append(f"--- {metric} ---")
        header_l = f"{'Label':<30}" + "".join(f"{c:>15}" for c in COND_NAMES) + f"{'Best':>15}"
        lines.append(header_l)
        lines.append("-" * 100)
        is_lower = metric in ["ECE", "Brier", "AURC"]
        for label in LABELS:
            vals = {}
            for cond in COND_NAMES:
                if metric in ["AUC", "AP", "Accuracy", "F1"]:
                    vals[cond] = cls[cond].loc[label, metric]
                else:
                    cd, col = cal_map[metric]
                    vals[cond] = get_cal_value(cd, cond, label, col)
            best = min(vals, key=vals.get) if is_lower else max(vals, key=vals.get)
            row = f"{label:<30}" + "".join(f"{vals[c]:>15.6f}" for c in COND_NAMES) + f"{best:>15}"
            lines.append(row)

    lines.append("")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)
    txt_path = os.path.join(out_dir, "overall_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(f"  Saved: overall_summary.txt")

    return summary_text


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    for model_key, model_info in MODELS.items():
        summary = process_model(model_key, model_info)
        print(summary)
        print()
