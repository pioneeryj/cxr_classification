"""
Experiment Results Visualization Script

Visualizes classification and calibration results for a single experiment,
comparing uncalibrated vs post-hoc calibrated (global T and label-wise T) performance.

Usage:
    python visualize_experiment_results.py --experiment_dir outputs/resnet50_experiment
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_experiment_data(experiment_dir: str) -> dict:
    """Load all result files from an experiment directory."""
    exp_path = Path(experiment_dir)
    data = {}

    # Load classification results
    calibration_types = ['uncalibrated', 'global_t', 'label_wise_t']

    for cal_type in calibration_types:
        # Classification metrics
        clf_file = exp_path / f'test_results_{cal_type}.csv'
        if clf_file.exists():
            data[f'classification_{cal_type}'] = pd.read_csv(clf_file, index_col=0)

        # Calibration metrics
        for metric in ['ece', 'brier_score', 'aurc']:
            cal_file = exp_path / f'calibration_{cal_type}_{metric}.csv'
            if cal_file.exists():
                data[f'{metric}_{cal_type}'] = pd.read_csv(cal_file)

    return data


def create_calibration_comparison_figure(data: dict, save_dir: Path):
    """
    Figure 1: Calibration metrics comparison (ECE, Brier Score, AURC)
    Shows before/after calibration for each metric.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['ece', 'brier_score', 'aurc']
    metric_names = ['ECE (Expected Calibration Error)', 'Brier Score', 'AURC']

    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        # Get data for each calibration type
        cal_types = ['uncalibrated', 'global_t', 'label_wise_t']
        cal_labels = ['Uncalibrated', 'Global T', 'Label-wise T']

        # Extract per-label data (exclude Overall)
        plot_data = []
        for cal_type in cal_types:
            key = f'{metric}_{cal_type}'
            if key in data:
                df = data[key]
                df_labels = df[df['label_index'] != -1].copy()
                plot_data.append(df_labels[metric].values)

        if not plot_data:
            continue

        # Get label names
        label_names = data[f'{metric}_uncalibrated'][
            data[f'{metric}_uncalibrated']['label_index'] != -1
        ]['label_name'].values

        x = np.arange(len(label_names))
        width = 0.25

        colors = ['#e74c3c', '#3498db', '#2ecc71']

        for i, (values, label, color) in enumerate(zip(plot_data, cal_labels, colors)):
            ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)

        ax.set_xlabel('Disease Label', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name}\nby Calibration Method', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'calibration_comparison.png'}")


def create_calibration_improvement_heatmap(data: dict, save_dir: Path):
    """
    Figure 2: Heatmap showing calibration improvement (reduction in ECE/Brier)
    for each label with different calibration methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    metrics = ['ece', 'brier_score']
    metric_names = ['ECE Reduction (%)', 'Brier Score Reduction (%)']

    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        uncal_key = f'{metric}_uncalibrated'
        global_key = f'{metric}_global_t'
        labelwise_key = f'{metric}_label_wise_t'

        if not all(k in data for k in [uncal_key, global_key, labelwise_key]):
            continue

        df_uncal = data[uncal_key][data[uncal_key]['label_index'] != -1]
        df_global = data[global_key][data[global_key]['label_index'] != -1]
        df_labelwise = data[labelwise_key][data[labelwise_key]['label_index'] != -1]

        # Calculate improvement (reduction) as percentage
        global_improvement = ((df_uncal[metric].values - df_global[metric].values)
                             / df_uncal[metric].values * 100)
        labelwise_improvement = ((df_uncal[metric].values - df_labelwise[metric].values)
                                / df_uncal[metric].values * 100)

        improvement_df = pd.DataFrame({
            'Global T': global_improvement,
            'Label-wise T': labelwise_improvement
        }, index=df_uncal['label_name'].values)

        sns.heatmap(improvement_df, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': 'Improvement (%)'})
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Disease Label', fontsize=11)

    plt.suptitle('Calibration Improvement by Method\n(Positive = Better Calibration)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_improvement_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'calibration_improvement_heatmap.png'}")


def create_classification_vs_calibration_scatter(data: dict, save_dir: Path):
    """
    Figure 3: Scatter plot showing relationship between classification performance (AUC)
    and calibration quality (ECE) for each label.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cal_types = ['uncalibrated', 'global_t', 'label_wise_t']
    cal_titles = ['Uncalibrated', 'Global T Calibration', 'Label-wise T Calibration']

    for ax, cal_type, title in zip(axes, cal_types, cal_titles):
        clf_key = f'classification_{cal_type}'
        ece_key = f'ece_{cal_type}'

        if clf_key not in data or ece_key not in data:
            continue

        clf_df = data[clf_key]
        ece_df = data[ece_key][data[ece_key]['label_index'] != -1]

        auc_values = clf_df['AUC'].values
        ece_values = ece_df['ece'].values
        labels = clf_df.index.values

        # Scatter plot with colors based on AUC
        scatter = ax.scatter(auc_values, ece_values, c=auc_values, cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (auc_values[i], ece_values[i]),
                       fontsize=8, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')

        ax.set_xlabel('AUC (Classification Performance)', fontsize=11)
        ax.set_ylabel('ECE (Calibration Error)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add correlation coefficient
        corr = np.corrcoef(auc_values, ece_values)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.colorbar(scatter, ax=ax, label='AUC')

    plt.suptitle('Classification Performance vs Calibration Error',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'classification_vs_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'classification_vs_calibration.png'}")


def create_labelwise_calibration_comparison(data: dict, save_dir: Path):
    """
    Figure 4: Detailed comparison of label-wise calibration effect on each class.
    Shows which classes benefit most from label-wise vs global calibration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: ECE comparison
    ax1 = axes[0, 0]
    ece_uncal = data['ece_uncalibrated'][data['ece_uncalibrated']['label_index'] != -1]
    ece_global = data['ece_global_t'][data['ece_global_t']['label_index'] != -1]
    ece_labelwise = data['ece_label_wise_t'][data['ece_label_wise_t']['label_index'] != -1]

    labels = ece_uncal['label_name'].values
    x = np.arange(len(labels))

    ax1.plot(x, ece_uncal['ece'].values, 'o-', label='Uncalibrated',
            color='#e74c3c', linewidth=2, markersize=8)
    ax1.plot(x, ece_global['ece'].values, 's-', label='Global T',
            color='#3498db', linewidth=2, markersize=8)
    ax1.plot(x, ece_labelwise['ece'].values, '^-', label='Label-wise T',
            color='#2ecc71', linewidth=2, markersize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('ECE', fontsize=11)
    ax1.set_title('ECE by Label and Calibration Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Label-wise vs Global improvement comparison
    ax2 = axes[0, 1]

    global_vs_uncal = (ece_uncal['ece'].values - ece_global['ece'].values) / ece_uncal['ece'].values * 100
    labelwise_vs_uncal = (ece_uncal['ece'].values - ece_labelwise['ece'].values) / ece_uncal['ece'].values * 100
    labelwise_vs_global = (ece_global['ece'].values - ece_labelwise['ece'].values) / ece_global['ece'].values * 100

    width = 0.35
    ax2.bar(x - width/2, global_vs_uncal, width, label='Global T vs Uncalibrated', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, labelwise_vs_uncal, width, label='Label-wise T vs Uncalibrated', color='#2ecc71', alpha=0.8)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('ECE Improvement (%)', fontsize=11)
    ax2.set_title('ECE Improvement vs Uncalibrated', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Which classes benefit more from label-wise?
    ax3 = axes[1, 0]

    labelwise_benefit = labelwise_vs_global
    colors = ['#2ecc71' if b > 0 else '#e74c3c' for b in labelwise_benefit]

    bars = ax3.barh(labels, labelwise_benefit, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Additional ECE Improvement (%)', fontsize=11)
    ax3.set_title('Label-wise T Improvement over Global T\n(Positive = Label-wise is better)',
                 fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, labelwise_benefit):
        width = bar.get_width()
        ax3.text(width + 0.5 if width > 0 else width - 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=9)

    # Subplot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate summary statistics
    overall_uncal = data['ece_uncalibrated'][data['ece_uncalibrated']['label_index'] == -1]['ece'].values[0]
    overall_global = data['ece_global_t'][data['ece_global_t']['label_index'] == -1]['ece'].values[0]
    overall_labelwise = data['ece_label_wise_t'][data['ece_label_wise_t']['label_index'] == -1]['ece'].values[0]

    brier_uncal = data['brier_score_uncalibrated'][data['brier_score_uncalibrated']['label_index'] == -1]['brier_score'].values[0]
    brier_global = data['brier_score_global_t'][data['brier_score_global_t']['label_index'] == -1]['brier_score'].values[0]
    brier_labelwise = data['brier_score_label_wise_t'][data['brier_score_label_wise_t']['label_index'] == -1]['brier_score'].values[0]

    summary_text = f"""
    ═══════════════════════════════════════════════════════════
                        CALIBRATION SUMMARY
    ═══════════════════════════════════════════════════════════

    OVERALL ECE (Expected Calibration Error):
    ─────────────────────────────────────────
      Uncalibrated:     {overall_uncal:.6f}
      Global T:         {overall_global:.6f}  ({((overall_uncal-overall_global)/overall_uncal*100):+.1f}%)
      Label-wise T:     {overall_labelwise:.6f}  ({((overall_uncal-overall_labelwise)/overall_uncal*100):+.1f}%)

    OVERALL BRIER SCORE:
    ─────────────────────────────────────────
      Uncalibrated:     {brier_uncal:.6f}
      Global T:         {brier_global:.6f}  ({((brier_uncal-brier_global)/brier_uncal*100):+.1f}%)
      Label-wise T:     {brier_labelwise:.6f}  ({((brier_uncal-brier_labelwise)/brier_uncal*100):+.1f}%)

    KEY INSIGHTS:
    ─────────────────────────────────────────
      • Labels benefiting most from Label-wise T:
        {', '.join(labels[np.argsort(labelwise_benefit)[-3:][::-1]])}

      • Labels where Global T works better:
        {', '.join(labels[np.argsort(labelwise_benefit)[:3]]) if np.min(labelwise_benefit) < 0 else 'None'}

    ═══════════════════════════════════════════════════════════
    """

    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                                              edgecolor='#dee2e6', linewidth=2))

    plt.tight_layout()
    plt.savefig(save_dir / 'labelwise_calibration_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'labelwise_calibration_analysis.png'}")


def create_classification_metrics_overview(data: dict, save_dir: Path):
    """
    Figure 5: Overview of classification metrics (AUC, AP, F1) across labels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    clf_data = data['classification_uncalibrated']  # Classification metrics are same across calibration

    labels = clf_data.index.values
    x = np.arange(len(labels))

    # Subplot 1: AUC by label
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlGn(clf_data['AUC'].values)
    bars = ax1.bar(x, clf_data['AUC'].values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (0.8)')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.7)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('AUC', fontsize=11)
    ax1.set_title('AUC by Disease Label', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, clf_data['AUC'].values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Subplot 2: AP by label
    ax2 = axes[0, 1]
    bars = ax2.bar(x, clf_data['AP'].values, color='#3498db', alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Average Precision (AP)', fontsize=11)
    ax2.set_title('Average Precision by Disease Label', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, clf_data['AP'].values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Subplot 3: F1 Score by label
    ax3 = axes[1, 0]
    bars = ax3.bar(x, clf_data['F1'].values, color='#e74c3c', alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('F1 Score', fontsize=11)
    ax3.set_title('F1 Score by Disease Label', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, clf_data['F1'].values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Subplot 4: Radar chart for overall metrics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create a summary table instead
    summary_df = clf_data[['AUC', 'AP', 'Accuracy', 'F1']].copy()
    summary_df = summary_df.round(4)

    # Add mean row
    mean_row = pd.DataFrame({
        'AUC': [clf_data['AUC'].mean()],
        'AP': [clf_data['AP'].mean()],
        'Accuracy': [clf_data['Accuracy'].mean()],
        'F1': [clf_data['F1'].mean()]
    }, index=['Mean'])
    summary_df = pd.concat([summary_df, mean_row])

    # Create table
    table = ax4.table(cellText=summary_df.values.round(4),
                     colLabels=summary_df.columns,
                     rowLabels=summary_df.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white')
        if col == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f8f9fa')

    ax4.set_title('Classification Metrics Summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Classification Performance Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'classification_metrics_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'classification_metrics_overview.png'}")


def create_class_imbalance_vs_performance(data: dict, save_dir: Path):
    """
    Figure 6: Analyze relationship between class support and various metrics.
    Shows how class imbalance affects classification and calibration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    clf_data = data['classification_uncalibrated']
    ece_data = data['ece_uncalibrated'][data['ece_uncalibrated']['label_index'] != -1]

    labels = clf_data.index.values

    # For this analysis, we'll use AP as a proxy for difficulty (lower AP = harder/more imbalanced)
    # and analyze its relationship with calibration

    # Subplot 1: AUC vs AP (proxy for class difficulty)
    ax1 = axes[0]
    scatter = ax1.scatter(clf_data['AP'].values, clf_data['AUC'].values,
                         c=clf_data['F1'].values, cmap='viridis',
                         s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
    for i, label in enumerate(labels):
        ax1.annotate(label, (clf_data['AP'].values[i], clf_data['AUC'].values[i]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    ax1.set_xlabel('Average Precision', fontsize=11)
    ax1.set_ylabel('AUC', fontsize=11)
    ax1.set_title('AUC vs Average Precision\n(Color = F1 Score)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='F1 Score')

    # Subplot 2: F1 vs ECE
    ax2 = axes[1]
    scatter = ax2.scatter(clf_data['F1'].values, ece_data['ece'].values,
                         c=clf_data['AUC'].values, cmap='RdYlGn',
                         s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
    for i, label in enumerate(labels):
        ax2.annotate(label, (clf_data['F1'].values[i], ece_data['ece'].values[i]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    ax2.set_xlabel('F1 Score', fontsize=11)
    ax2.set_ylabel('ECE (Calibration Error)', fontsize=11)
    ax2.set_title('F1 Score vs Calibration Error\n(Color = AUC)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='AUC')

    # Subplot 3: Calibration improvement vs F1 (do hard classes benefit more from calibration?)
    ax3 = axes[2]

    ece_global = data['ece_global_t'][data['ece_global_t']['label_index'] != -1]
    ece_improvement = (ece_data['ece'].values - ece_global['ece'].values) / ece_data['ece'].values * 100

    scatter = ax3.scatter(clf_data['F1'].values, ece_improvement,
                         c=clf_data['AUC'].values, cmap='RdYlGn',
                         s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Fit and plot trend line
    z = np.polyfit(clf_data['F1'].values, ece_improvement, 1)
    p = np.poly1d(z)
    x_line = np.linspace(clf_data['F1'].min(), clf_data['F1'].max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')

    for i, label in enumerate(labels):
        ax3.annotate(label, (clf_data['F1'].values[i], ece_improvement[i]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('F1 Score', fontsize=11)
    ax3.set_ylabel('ECE Improvement with Global T (%)', fontsize=11)
    ax3.set_title('Calibration Improvement vs F1 Score\n(Do harder classes benefit more?)',
                 fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='AUC')
    ax3.legend(loc='upper right')

    plt.suptitle('Class Difficulty Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_difficulty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'class_difficulty_analysis.png'}")


def create_comprehensive_summary(data: dict, save_dir: Path):
    """
    Figure 7: Comprehensive summary dashboard.
    """
    fig = plt.figure(figsize=(20, 14))

    # Create grid spec
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Top row: Overall metrics comparison
    ax1 = fig.add_subplot(gs[0, 0:2])

    metrics_list = ['ece', 'brier_score', 'aurc']
    cal_types = ['uncalibrated', 'global_t', 'label_wise_t']
    cal_labels = ['Uncalibrated', 'Global T', 'Label-wise T']

    overall_values = []
    for cal_type in cal_types:
        row = []
        for metric in metrics_list:
            key = f'{metric}_{cal_type}'
            if key in data:
                val = data[key][data[key]['label_index'] == -1][metric].values[0]
                row.append(val)
        overall_values.append(row)

    x = np.arange(len(metrics_list))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, (values, label, color) in enumerate(zip(overall_values, cal_labels, colors)):
        ax1.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['ECE', 'Brier Score', 'AURC'], fontsize=11)
    ax1.set_ylabel('Value (lower is better)', fontsize=11)
    ax1.set_title('Overall Calibration Metrics', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Top row: Classification metrics radar
    ax2 = fig.add_subplot(gs[0, 2:4])

    clf_data = data['classification_uncalibrated']
    mean_metrics = {
        'AUC': clf_data['AUC'].mean(),
        'AP': clf_data['AP'].mean(),
        'Accuracy': clf_data['Accuracy'].mean(),
        'F1': clf_data['F1'].mean()
    }

    categories = list(mean_metrics.keys())
    values = list(mean_metrics.values())

    ax2.bar(categories, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
           alpha=0.8, edgecolor='black', linewidth=0.5)

    for i, (cat, val) in enumerate(zip(categories, values)):
        ax2.text(i, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel('Mean Value', fontsize=11)
    ax2.set_title('Mean Classification Metrics Across All Labels', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    # Middle row: Top and bottom performers
    ax3 = fig.add_subplot(gs[1, 0:2])

    # Sort by AUC
    sorted_idx = np.argsort(clf_data['AUC'].values)
    sorted_labels = clf_data.index.values[sorted_idx]
    sorted_auc = clf_data['AUC'].values[sorted_idx]

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sorted_labels)))
    ax3.barh(sorted_labels, sorted_auc, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good threshold (0.8)')
    ax3.set_xlabel('AUC', fontsize=11)
    ax3.set_title('Labels Ranked by AUC', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.set_xlim(0.5, 1.0)

    # Middle row: ECE comparison
    ax4 = fig.add_subplot(gs[1, 2:4])

    ece_uncal = data['ece_uncalibrated'][data['ece_uncalibrated']['label_index'] != -1]
    ece_global = data['ece_global_t'][data['ece_global_t']['label_index'] != -1]
    ece_labelwise = data['ece_label_wise_t'][data['ece_label_wise_t']['label_index'] != -1]

    # Sort by uncalibrated ECE
    sorted_idx = np.argsort(ece_uncal['ece'].values)[::-1]
    sorted_labels = ece_uncal['label_name'].values[sorted_idx]

    y = np.arange(len(sorted_labels))
    height = 0.25

    ax4.barh(y - height, ece_uncal['ece'].values[sorted_idx], height,
            label='Uncalibrated', color='#e74c3c', alpha=0.8)
    ax4.barh(y, ece_global['ece'].values[sorted_idx], height,
            label='Global T', color='#3498db', alpha=0.8)
    ax4.barh(y + height, ece_labelwise['ece'].values[sorted_idx], height,
            label='Label-wise T', color='#2ecc71', alpha=0.8)

    ax4.set_yticks(y)
    ax4.set_yticklabels(sorted_labels)
    ax4.set_xlabel('ECE', fontsize=11)
    ax4.set_title('ECE by Label (Sorted by Uncalibrated ECE)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')

    # Bottom row: Insights
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Calculate insights
    best_auc = clf_data['AUC'].idxmax()
    worst_auc = clf_data['AUC'].idxmin()
    best_calibrated = ece_uncal.loc[ece_uncal['ece'].idxmin(), 'label_name']
    worst_calibrated = ece_uncal.loc[ece_uncal['ece'].idxmax(), 'label_name']

    global_improvement = ((ece_uncal['ece'].values - ece_global['ece'].values)
                         / ece_uncal['ece'].values * 100).mean()
    labelwise_improvement = ((ece_uncal['ece'].values - ece_labelwise['ece'].values)
                            / ece_uncal['ece'].values * 100).mean()

    insights_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                         KEY INSIGHTS & FINDINGS                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                       ║
    ║  CLASSIFICATION PERFORMANCE                           CALIBRATION ANALYSIS                            ║
    ║  ─────────────────────────────                        ────────────────────                            ║
    ║  • Best AUC: {best_auc:<25}       • Best calibrated (lowest ECE): {best_calibrated:<18}      ║
    ║  • Worst AUC: {worst_auc:<24}       • Worst calibrated (highest ECE): {worst_calibrated:<16}      ║
    ║  • Mean AUC: {clf_data['AUC'].mean():.4f}                            • Mean ECE (Uncalibrated): {ece_uncal['ece'].mean():.6f}           ║
    ║  • Mean F1: {clf_data['F1'].mean():.4f}                             • Mean ECE (Global T): {ece_global['ece'].mean():.6f}              ║
    ║                                                       • Mean ECE (Label-wise T): {ece_labelwise['ece'].mean():.6f}           ║
    ║                                                                                                       ║
    ║  CALIBRATION IMPROVEMENT                                                                              ║
    ║  ───────────────────────                                                                              ║
    ║  • Global T vs Uncalibrated: {global_improvement:+.2f}% average ECE reduction                                           ║
    ║  • Label-wise T vs Uncalibrated: {labelwise_improvement:+.2f}% average ECE reduction                                       ║
    ║                                                                                                       ║
    ║  OBSERVATIONS                                                                                         ║
    ║  ────────────                                                                                         ║
    ║  • Temperature scaling improves calibration for most labels                                           ║
    ║  • Label-wise calibration provides marginal additional benefit over global calibration                ║
    ║  • Labels with lower F1/AP scores (harder classes) show varied calibration improvement                ║
    ║                                                                                                       ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax5.text(0.5, 0.5, insights_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                                              edgecolor='#dee2e6', linewidth=2))

    plt.suptitle('Experiment Results Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'comprehensive_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'comprehensive_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory (e.g., outputs/resnet50_experiment)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: experiment_dir/visualizations)')

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {experiment_dir}")
    print(f"Saving visualizations to: {output_dir}")
    print("=" * 60)

    # Load data
    data = load_experiment_data(experiment_dir)

    if not data:
        raise ValueError("No data files found in the experiment directory")

    print(f"Loaded {len(data)} data files")
    print("=" * 60)

    # Create visualizations
    print("\nGenerating visualizations...")

    create_calibration_comparison_figure(data, output_dir)
    create_calibration_improvement_heatmap(data, output_dir)
    create_classification_vs_calibration_scatter(data, output_dir)
    create_labelwise_calibration_comparison(data, output_dir)
    create_classification_metrics_overview(data, output_dir)
    create_class_imbalance_vs_performance(data, output_dir)
    create_comprehensive_summary(data, output_dir)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Check the output directory: {output_dir}")


if __name__ == '__main__':
    main()
