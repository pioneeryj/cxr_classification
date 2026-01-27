"""
Multi-Model Comparison Visualization Script

Compares classification and calibration performance across multiple models.
Generates three comparison figures:
1. Classification performance comparison
2. Global T calibration performance comparison
3. Label-wise T calibration performance comparison

Usage:
    python compare_models.py --output_dir outputs
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


def load_all_models_data(output_dir: str) -> dict:
    """Load data from all experiment directories."""
    output_path = Path(output_dir)

    # Find all experiment directories
    experiment_dirs = [d for d in output_path.iterdir()
                       if d.is_dir() and d.name.endswith('_experiment')]

    all_data = {}

    for exp_dir in experiment_dirs:
        model_name = exp_dir.name.replace('_experiment', '')
        model_data = {}

        # Load classification results
        for cal_type in ['uncalibrated', 'global_t', 'label_wise_t']:
            clf_file = exp_dir / f'test_results_{cal_type}.csv'
            if clf_file.exists():
                model_data[f'classification_{cal_type}'] = pd.read_csv(clf_file, index_col=0)

            # Load calibration metrics
            for metric in ['ece', 'brier_score', 'aurc']:
                cal_file = exp_dir / f'calibration_{cal_type}_{metric}.csv'
                if cal_file.exists():
                    model_data[f'{metric}_{cal_type}'] = pd.read_csv(cal_file)

        if model_data:
            all_data[model_name] = model_data
            print(f"Loaded data for: {model_name}")

    return all_data


def create_classification_comparison(all_data: dict, save_dir: Path):
    """
    Figure 1: Classification performance comparison across all models.
    Shows AUC, AP, Accuracy, F1 for each model.
    """
    models = list(all_data.keys())
    n_models = len(models)

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))

    # Define metrics and colors
    metrics = ['AUC', 'AP', 'Accuracy', 'F1']
    model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Get label names from first model
    first_model = list(all_data.values())[0]
    if 'classification_uncalibrated' in first_model:
        labels = first_model['classification_uncalibrated'].index.tolist()
    else:
        print("No classification data found")
        return

    # Create 2x2 grid for metrics
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(2, 2, idx + 1)

        x = np.arange(len(labels))
        width = 0.8 / n_models

        for i, (model_name, model_data) in enumerate(all_data.items()):
            if 'classification_uncalibrated' not in model_data:
                continue

            clf_df = model_data['classification_uncalibrated']
            values = clf_df[metric].values

            bars = ax.bar(x + i * width - (n_models - 1) * width / 2,
                         values, width,
                         label=model_name,
                         color=model_colors[i],
                         alpha=0.85,
                         edgecolor='black',
                         linewidth=0.5)

        ax.set_xlabel('Disease Label', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add reference lines for AUC
        if metric == 'AUC':
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
            ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.7)')
            ax.set_ylim(0.5, 1.0)

    # Add summary table at bottom
    fig.subplots_adjust(bottom=0.25)

    # Create summary data
    summary_data = []
    for model_name, model_data in all_data.items():
        if 'classification_uncalibrated' not in model_data:
            continue
        clf_df = model_data['classification_uncalibrated']
        summary_data.append({
            'Model': model_name,
            'Mean AUC': f"{clf_df['AUC'].mean():.4f}",
            'Mean AP': f"{clf_df['AP'].mean():.4f}",
            'Mean Acc': f"{clf_df['Accuracy'].mean():.4f}",
            'Mean F1': f"{clf_df['F1'].mean():.4f}",
        })

    summary_df = pd.DataFrame(summary_data)

    # Add table
    table_ax = fig.add_axes([0.15, 0.02, 0.7, 0.12])
    table_ax.axis('off')

    table = table_ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(summary_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style table header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('gray')

    plt.suptitle('Classification Performance Comparison\n(Uncalibrated)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(save_dir / 'comparison_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'comparison_classification.png'}")


def create_calibration_comparison(all_data: dict, save_dir: Path, cal_type: str):
    """
    Create calibration performance comparison figure.

    Args:
        all_data: Dictionary of model data
        save_dir: Directory to save figure
        cal_type: 'global_t' or 'label_wise_t'
    """
    models = list(all_data.keys())
    n_models = len(models)

    cal_title = 'Global T' if cal_type == 'global_t' else 'Label-wise T'

    # Create figure
    fig = plt.figure(figsize=(24, 18))

    # Calibration metrics
    cal_metrics = ['ece', 'brier_score', 'aurc']
    cal_metric_names = ['ECE (Expected Calibration Error)', 'Brier Score', 'AURC']

    model_colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Get label names
    first_model = list(all_data.values())[0]
    metric_key = f'ece_{cal_type}'
    if metric_key in first_model:
        labels_df = first_model[metric_key]
        labels = labels_df[labels_df['label_index'] != -1]['label_name'].tolist()
    else:
        print(f"No {cal_type} calibration data found")
        return

    # Row 1: Per-label calibration metrics comparison (3 subplots)
    for idx, (metric, metric_name) in enumerate(zip(cal_metrics, cal_metric_names)):
        ax = fig.add_subplot(3, 3, idx + 1)

        x = np.arange(len(labels))
        width = 0.8 / n_models

        for i, (model_name, model_data) in enumerate(all_data.items()):
            key = f'{metric}_{cal_type}'
            if key not in model_data:
                continue

            df = model_data[key]
            df_labels = df[df['label_index'] != -1]
            values = df_labels[metric].values

            ax.bar(x + i * width - (n_models - 1) * width / 2,
                  values, width,
                  label=model_name,
                  color=model_colors[i],
                  alpha=0.85,
                  edgecolor='black',
                  linewidth=0.5)

        ax.set_xlabel('Disease Label', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Label', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Row 2: Overall metrics comparison and improvement from uncalibrated
    # Subplot 4: Overall calibration metrics
    ax4 = fig.add_subplot(3, 3, 4)

    x = np.arange(len(cal_metrics))
    width = 0.8 / n_models

    for i, (model_name, model_data) in enumerate(all_data.items()):
        overall_values = []
        for metric in cal_metrics:
            key = f'{metric}_{cal_type}'
            if key in model_data:
                df = model_data[key]
                overall_val = df[df['label_index'] == -1][metric].values[0]
                overall_values.append(overall_val)
            else:
                overall_values.append(0)

        ax4.bar(x + i * width - (n_models - 1) * width / 2,
               overall_values, width,
               label=model_name,
               color=model_colors[i],
               alpha=0.85,
               edgecolor='black',
               linewidth=0.5)

    ax4.set_xlabel('Metric', fontsize=11)
    ax4.set_ylabel('Value (lower is better)', fontsize=11)
    ax4.set_title('Overall Calibration Metrics', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['ECE', 'Brier Score', 'AURC'], fontsize=10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Subplot 5: ECE improvement from uncalibrated
    ax5 = fig.add_subplot(3, 3, 5)

    improvement_data = []
    for model_name, model_data in all_data.items():
        if 'ece_uncalibrated' in model_data and f'ece_{cal_type}' in model_data:
            uncal_df = model_data['ece_uncalibrated']
            cal_df = model_data[f'ece_{cal_type}']

            uncal_overall = uncal_df[uncal_df['label_index'] == -1]['ece'].values[0]
            cal_overall = cal_df[cal_df['label_index'] == -1]['ece'].values[0]

            improvement = (uncal_overall - cal_overall) / uncal_overall * 100
            improvement_data.append({'Model': model_name, 'ECE Improvement (%)': improvement})

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        bars = ax5.bar(imp_df['Model'], imp_df['ECE Improvement (%)'],
                      color=model_colors[:len(imp_df)], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('ECE Improvement (%)', fontsize=11)
        ax5.set_title(f'ECE Improvement vs Uncalibrated\n({cal_title})', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, val in zip(bars, imp_df['ECE Improvement (%)']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Subplot 6: Brier Score improvement
    ax6 = fig.add_subplot(3, 3, 6)

    improvement_data = []
    for model_name, model_data in all_data.items():
        if 'brier_score_uncalibrated' in model_data and f'brier_score_{cal_type}' in model_data:
            uncal_df = model_data['brier_score_uncalibrated']
            cal_df = model_data[f'brier_score_{cal_type}']

            uncal_overall = uncal_df[uncal_df['label_index'] == -1]['brier_score'].values[0]
            cal_overall = cal_df[cal_df['label_index'] == -1]['brier_score'].values[0]

            improvement = (uncal_overall - cal_overall) / uncal_overall * 100
            improvement_data.append({'Model': model_name, 'Brier Improvement (%)': improvement})

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        bars = ax6.bar(imp_df['Model'], imp_df['Brier Improvement (%)'],
                      color=model_colors[:len(imp_df)], alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_ylabel('Brier Score Improvement (%)', fontsize=11)
        ax6.set_title(f'Brier Score Improvement vs Uncalibrated\n({cal_title})', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, imp_df['Brier Improvement (%)']):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    # Row 3: Heatmaps for per-label comparison
    # Subplot 7: ECE heatmap
    ax7 = fig.add_subplot(3, 3, 7)

    ece_matrix = []
    for model_name, model_data in all_data.items():
        key = f'ece_{cal_type}'
        if key in model_data:
            df = model_data[key]
            df_labels = df[df['label_index'] != -1]
            ece_matrix.append(df_labels['ece'].values)

    if ece_matrix:
        ece_matrix = np.array(ece_matrix)
        sns.heatmap(ece_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=list(all_data.keys()),
                   ax=ax7, cbar_kws={'label': 'ECE'})
        ax7.set_title('ECE by Model and Label', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Disease Label', fontsize=10)
        ax7.set_ylabel('Model', fontsize=10)
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Subplot 8: Brier Score heatmap
    ax8 = fig.add_subplot(3, 3, 8)

    brier_matrix = []
    for model_name, model_data in all_data.items():
        key = f'brier_score_{cal_type}'
        if key in model_data:
            df = model_data[key]
            df_labels = df[df['label_index'] != -1]
            brier_matrix.append(df_labels['brier_score'].values)

    if brier_matrix:
        brier_matrix = np.array(brier_matrix)
        sns.heatmap(brier_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=list(all_data.keys()),
                   ax=ax8, cbar_kws={'label': 'Brier Score'})
        ax8.set_title('Brier Score by Model and Label', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Disease Label', fontsize=10)
        ax8.set_ylabel('Model', fontsize=10)
        plt.setp(ax8.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Subplot 9: Summary table
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    summary_data = []
    for model_name, model_data in all_data.items():
        row = {'Model': model_name}

        for metric in cal_metrics:
            key = f'{metric}_{cal_type}'
            if key in model_data:
                df = model_data[key]
                overall_val = df[df['label_index'] == -1][metric].values[0]
                row[metric.upper()] = f'{overall_val:.6f}'
            else:
                row[metric.upper()] = 'N/A'

        # Add improvement from uncalibrated
        if 'ece_uncalibrated' in model_data and f'ece_{cal_type}' in model_data:
            uncal = model_data['ece_uncalibrated'][model_data['ece_uncalibrated']['label_index'] == -1]['ece'].values[0]
            cal = model_data[f'ece_{cal_type}'][model_data[f'ece_{cal_type}']['label_index'] == -1]['ece'].values[0]
            row['ECE Impr.'] = f'{(uncal-cal)/uncal*100:+.1f}%'

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    table = ax9.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(summary_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('gray')

    ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'Calibration Performance Comparison\n({cal_title} Temperature Scaling)',
                 fontsize=18, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f'comparison_calibration_{cal_type}.png'
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple models')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory containing experiment outputs')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save comparison figures (default: output_dir)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    save_dir = Path(args.save_dir) if args.save_dir else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {output_dir}")
    print(f"Saving figures to: {save_dir}")
    print("=" * 60)

    # Load all model data
    all_data = load_all_models_data(output_dir)

    if not all_data:
        raise ValueError("No experiment data found")

    print(f"\nFound {len(all_data)} models: {list(all_data.keys())}")
    print("=" * 60)

    # Generate comparison figures
    print("\nGenerating comparison figures...")

    # 1. Classification comparison
    create_classification_comparison(all_data, save_dir)

    # 2. Global T calibration comparison
    create_calibration_comparison(all_data, save_dir, 'global_t')

    # 3. Label-wise T calibration comparison
    create_calibration_comparison(all_data, save_dir, 'label_wise_t')

    print("\n" + "=" * 60)
    print("All comparison figures generated successfully!")
    print(f"Check the output directory: {save_dir}")


if __name__ == '__main__':
    main()
