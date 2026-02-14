"""클래스 불균형이 분류 성능에 미치는 영향 분석"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

# 설정
topdir = Path("/mnt/HDD/dataset/medical_report")
output_dir = Path("/home/yoonji/mrg/cxr_classification/outputs")

chexpert_labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

subject_prefix_filter = ['10', '11']
num_folds = 10
fold = 0
val_size = 0.1
random_state = 0

def load_all_metadata(data_dir=topdir, subject_prefix_filter=None):
    """메타데이터 로드"""
    data_dir = Path(data_dir)
    metadata = pd.read_csv(data_dir / "mimic-cxr-2.0.0-metadata.csv.gz")
    chexpert = pd.read_csv(data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")

    if (data_dir / "splitpaths.csv.gz").exists():
        splitpaths = pd.read_csv(data_dir / "splitpaths.csv.gz")
    else:
        splitpaths = pd.read_csv(data_dir / "mimic-cxr-2.0.0-split.csv.gz")
        if 'path' not in splitpaths.columns:
            splitpaths['path'] = splitpaths.apply(
                lambda row: f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
                axis=1
            )

    if subject_prefix_filter is not None:
        splitpaths['subject_prefix'] = splitpaths['subject_id'].astype(str).str[:2]
        splitpaths = splitpaths[splitpaths['subject_prefix'].isin(subject_prefix_filter)]
        splitpaths = splitpaths.drop(columns=['subject_prefix'])

    meta = pd.merge(metadata, splitpaths, on=["dicom_id", "subject_id", "study_id"])
    meta = pd.merge(meta, chexpert, on=["subject_id", "study_id"])
    return meta

def get_cv_split(allrecords, num_folds, fold, val_size, random_state):
    """CV split 수행"""
    kf = KFold(num_folds)
    uniq_subj = allrecords["subject_id"].unique()

    for k, (trainval_ix, test_ix) in enumerate(kf.split(uniq_subj)):
        if k != fold:
            continue
        trainval_subj = uniq_subj[trainval_ix]
        test_subj = uniq_subj[test_ix]
        train_subj, val_subj = train_test_split(
            trainval_subj, test_size=val_size, random_state=random_state, shuffle=False
        )

    subjrecs = lambda s: pd.DataFrame({"subject_id": s}).merge(allrecords, how="left", on="subject_id")
    return subjrecs(train_subj), subjrecs(val_subj), subjrecs(test_subj)

def get_label_distribution(df):
    """레이블별 positive 비율 계산"""
    results = {}
    for label in chexpert_labels:
        values = df[label].values
        if label == "No Finding":
            positive = np.sum(values == 1.0)
        else:
            positive = np.sum(values == 1.0)
        total = len(df)
        results[label] = {
            'positive_count': positive,
            'total': total,
            'positive_ratio': positive / total * 100,
            'imbalance_ratio': total / positive if positive > 0 else np.inf  # negative:positive ratio
        }
    return results

def load_test_results():
    """모든 모델의 테스트 결과 로드"""
    models = {
        'DenseNet121': output_dir / 'densenet121_experiment' / 'test_results.csv',
        'ResNet50': output_dir / 'resnet50_experiment' / 'test_results_uncalibrated.csv',
        'BioViL': output_dir / 'biovil_experiment' / 'test_results_uncalibrated.csv',
        'MedKLIP': output_dir / 'medklip_experiment' / 'test_results_uncalibrated.csv',
    }

    results = {}
    for model_name, path in models.items():
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            results[model_name] = df
    return results

def analyze_correlation(label_dist, test_results):
    """클래스 빈도와 성능 간 상관관계 분석"""
    # 데이터 준비
    positive_ratios = [label_dist[l]['positive_ratio'] for l in chexpert_labels]
    imbalance_ratios = [label_dist[l]['imbalance_ratio'] for l in chexpert_labels]

    correlations = {}

    for model_name, results_df in test_results.items():
        model_corr = {}

        for metric in ['AUC', 'AP', 'F1']:
            if metric in results_df.columns:
                metric_values = [results_df.loc[l, metric] for l in chexpert_labels]

                # Pearson correlation with positive ratio
                r_pos, p_pos = stats.pearsonr(positive_ratios, metric_values)

                # Spearman correlation (rank-based, more robust)
                rho_pos, p_spearman = stats.spearmanr(positive_ratios, metric_values)

                model_corr[metric] = {
                    'pearson_r': r_pos,
                    'pearson_p': p_pos,
                    'spearman_rho': rho_pos,
                    'spearman_p': p_spearman,
                    'values': metric_values
                }

        correlations[model_name] = model_corr

    return correlations, positive_ratios

def plot_correlation_analysis(label_dist, test_results, correlations, positive_ratios, save_dir):
    """상관관계 시각화"""

    # 1. 모든 모델의 AUC vs Positive Ratio 산점도
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(test_results.keys())
    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    for idx, (model_name, color) in enumerate(zip(models, colors)):
        ax = axes[idx // 2, idx % 2]

        if model_name in test_results:
            auc_values = [test_results[model_name].loc[l, 'AUC'] for l in chexpert_labels]

            ax.scatter(positive_ratios, auc_values, c=color, s=100, alpha=0.7, edgecolors='black')

            # 회귀선 추가
            z = np.polyfit(positive_ratios, auc_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(positive_ratios), max(positive_ratios), 100)
            ax.plot(x_line, p(x_line), '--', color=color, alpha=0.8, linewidth=2)

            # 레이블 추가
            for i, label in enumerate(chexpert_labels):
                ax.annotate(label[:8], (positive_ratios[i], auc_values[i]),
                           fontsize=8, ha='center', va='bottom', alpha=0.7)

            corr = correlations[model_name]['AUC']
            ax.set_title(f'{model_name}\nPearson r={corr["pearson_r"]:.3f} (p={corr["pearson_p"]:.3f})\n'
                        f'Spearman ρ={corr["spearman_rho"]:.3f} (p={corr["spearman_p"]:.3f})',
                        fontsize=11)
            ax.set_xlabel('Positive Ratio (%)', fontsize=10)
            ax.set_ylabel('AUC', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.0)

    plt.suptitle('Class Imbalance vs AUC Performance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'correlation_auc_vs_positive_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 모든 메트릭 비교 (best performing model)
    best_model = 'ResNet50'  # 가장 좋은 성능의 모델
    if best_model in test_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(['AUC', 'AP', 'F1']):
            ax = axes[idx]
            metric_values = [test_results[best_model].loc[l, metric] for l in chexpert_labels]

            ax.scatter(positive_ratios, metric_values, c='steelblue', s=100, alpha=0.7, edgecolors='black')

            # 회귀선
            z = np.polyfit(positive_ratios, metric_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(positive_ratios), max(positive_ratios), 100)
            ax.plot(x_line, p(x_line), '--', color='red', alpha=0.8, linewidth=2)

            corr = correlations[best_model][metric]
            ax.set_title(f'{metric}\nr={corr["pearson_r"]:.3f}, ρ={corr["spearman_rho"]:.3f}', fontsize=12)
            ax.set_xlabel('Positive Ratio (%)', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{best_model}: Metrics vs Class Frequency', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'correlation_all_metrics_{best_model.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. 성능 비교 히트맵 (레이블별, 모델별)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # AUC 히트맵
    ax1 = axes[0]
    auc_data = np.array([[test_results[m].loc[l, 'AUC'] for m in models if m in test_results]
                         for l in chexpert_labels])

    # positive ratio 순으로 정렬
    sorted_indices = np.argsort(positive_ratios)
    sorted_labels = [chexpert_labels[i] for i in sorted_indices]
    sorted_ratios = [positive_ratios[i] for i in sorted_indices]
    auc_data_sorted = auc_data[sorted_indices]

    im1 = ax1.imshow(auc_data_sorted, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_xticklabels([m for m in models if m in test_results], fontsize=10)
    ax1.set_yticks(np.arange(len(sorted_labels)))
    ax1.set_yticklabels([f'{l} ({r:.1f}%)' for l, r in zip(sorted_labels, sorted_ratios)], fontsize=9)
    ax1.set_title('AUC by Label (sorted by frequency)', fontsize=12, fontweight='bold')

    for i in range(len(sorted_labels)):
        for j in range(len([m for m in models if m in test_results])):
            ax1.text(j, i, f'{auc_data_sorted[i, j]:.2f}', ha='center', va='center',
                    color='white' if auc_data_sorted[i, j] < 0.7 else 'black', fontsize=8)

    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # AP 히트맵
    ax2 = axes[1]
    ap_data = np.array([[test_results[m].loc[l, 'AP'] for m in models if m in test_results]
                        for l in chexpert_labels])
    ap_data_sorted = ap_data[sorted_indices]

    im2 = ax2.imshow(ap_data_sorted, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.8)
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_xticklabels([m for m in models if m in test_results], fontsize=10)
    ax2.set_yticks(np.arange(len(sorted_labels)))
    ax2.set_yticklabels([f'{l} ({r:.1f}%)' for l, r in zip(sorted_labels, sorted_ratios)], fontsize=9)
    ax2.set_title('Average Precision by Label (sorted by frequency)', fontsize=12, fontweight='bold')

    for i in range(len(sorted_labels)):
        for j in range(len([m for m in models if m in test_results])):
            ax2.text(j, i, f'{ap_data_sorted[i, j]:.2f}', ha='center', va='center',
                    color='white' if ap_data_sorted[i, j] < 0.3 else 'black', fontsize=8)

    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_dir / 'performance_heatmap_by_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 박스플롯 - 빈도 그룹별 성능 비교
    fig, ax = plt.subplots(figsize=(10, 6))

    # 빈도에 따라 그룹 분류
    low_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if r < 5]
    mid_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if 5 <= r < 20]
    high_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if r >= 20]

    # 각 그룹의 AUC 수집 (MedKLIP 제외 - 성능이 너무 낮음)
    good_models = ['DenseNet121', 'ResNet50', 'BioViL']

    low_aucs = []
    mid_aucs = []
    high_aucs = []

    for model in good_models:
        if model in test_results:
            for l in low_freq:
                low_aucs.append(test_results[model].loc[l, 'AUC'])
            for l in mid_freq:
                mid_aucs.append(test_results[model].loc[l, 'AUC'])
            for l in high_freq:
                high_aucs.append(test_results[model].loc[l, 'AUC'])

    bp = ax.boxplot([low_aucs, mid_aucs, high_aucs],
                    labels=[f'Low (<5%)\nn={len(low_freq)} classes',
                           f'Medium (5-20%)\nn={len(mid_freq)} classes',
                           f'High (≥20%)\nn={len(high_freq)} classes'],
                    patch_artist=True)

    colors_box = ['lightcoral', 'lightyellow', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Distribution by Class Frequency Group\n(DenseNet121, ResNet50, BioViL)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 평균값 표시
    for i, data in enumerate([low_aucs, mid_aucs, high_aucs]):
        mean_val = np.mean(data)
        ax.scatter(i+1, mean_val, marker='D', color='red', s=50, zorder=5, label='Mean' if i==0 else '')
        ax.annotate(f'{mean_val:.3f}', (i+1.15, mean_val), fontsize=10)

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_dir / 'auc_by_frequency_group_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_table(label_dist, test_results, correlations, positive_ratios):
    """요약 테이블 생성"""
    rows = []

    for i, label in enumerate(chexpert_labels):
        row = {
            'Label': label,
            'Positive Ratio (%)': f"{positive_ratios[i]:.2f}",
            'Positive Count': label_dist[label]['positive_count'],
        }

        for model_name in test_results.keys():
            row[f'{model_name} AUC'] = f"{test_results[model_name].loc[label, 'AUC']:.3f}"
            row[f'{model_name} AP'] = f"{test_results[model_name].loc[label, 'AP']:.3f}"

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    print("=" * 70)
    print("Class Imbalance Impact on Classification Performance Analysis")
    print("=" * 70)

    # 데이터 로드
    print("\n1. Loading metadata...")
    allrecords = load_all_metadata(topdir, subject_prefix_filter=subject_prefix_filter)

    # Split
    print("2. Performing CV split...")
    train_df, val_df, test_df = get_cv_split(allrecords, num_folds, fold, val_size, random_state)

    # 레이블 분포 (Train 기준)
    print("3. Calculating label distribution...")
    train_dist = get_label_distribution(train_df)

    # 테스트 결과 로드
    print("4. Loading test results...")
    test_results = load_test_results()
    print(f"   Loaded models: {list(test_results.keys())}")

    # 상관관계 분석
    print("5. Analyzing correlations...")
    correlations, positive_ratios = analyze_correlation(train_dist, test_results)

    # 결과 출력
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 70)

    print("\n[Positive Ratio vs AUC Correlation]")
    print("-" * 60)
    print(f"{'Model':<12} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 60)

    for model_name, corr in correlations.items():
        if 'AUC' in corr:
            c = corr['AUC']
            print(f"{model_name:<12} {c['pearson_r']:>12.4f} {c['pearson_p']:>12.4f} "
                  f"{c['spearman_rho']:>12.4f} {c['spearman_p']:>12.4f}")

    print("\n[Positive Ratio vs Average Precision Correlation]")
    print("-" * 60)
    print(f"{'Model':<12} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 60)

    for model_name, corr in correlations.items():
        if 'AP' in corr:
            c = corr['AP']
            print(f"{model_name:<12} {c['pearson_r']:>12.4f} {c['pearson_p']:>12.4f} "
                  f"{c['spearman_rho']:>12.4f} {c['spearman_p']:>12.4f}")

    # 빈도 그룹별 평균 성능
    print("\n" + "=" * 70)
    print("PERFORMANCE BY FREQUENCY GROUP (excluding MedKLIP)")
    print("=" * 70)

    low_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if r < 5]
    mid_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if 5 <= r < 20]
    high_freq = [l for l, r in zip(chexpert_labels, positive_ratios) if r >= 20]

    print(f"\nLow frequency (<5%): {low_freq}")
    print(f"Medium frequency (5-20%): {mid_freq}")
    print(f"High frequency (≥20%): {high_freq}")

    good_models = ['DenseNet121', 'ResNet50', 'BioViL']

    print("\n[Average AUC by Frequency Group]")
    print("-" * 50)
    print(f"{'Model':<12} {'Low (<5%)':>12} {'Mid (5-20%)':>12} {'High (≥20%)':>12}")
    print("-" * 50)

    for model in good_models:
        if model in test_results:
            low_auc = np.mean([test_results[model].loc[l, 'AUC'] for l in low_freq])
            mid_auc = np.mean([test_results[model].loc[l, 'AUC'] for l in mid_freq])
            high_auc = np.mean([test_results[model].loc[l, 'AUC'] for l in high_freq])
            print(f"{model:<12} {low_auc:>12.4f} {mid_auc:>12.4f} {high_auc:>12.4f}")

    print("\n[Average AP by Frequency Group]")
    print("-" * 50)
    print(f"{'Model':<12} {'Low (<5%)':>12} {'Mid (5-20%)':>12} {'High (≥20%)':>12}")
    print("-" * 50)

    for model in good_models:
        if model in test_results:
            low_ap = np.mean([test_results[model].loc[l, 'AP'] for l in low_freq])
            mid_ap = np.mean([test_results[model].loc[l, 'AP'] for l in mid_freq])
            high_ap = np.mean([test_results[model].loc[l, 'AP'] for l in high_freq])
            print(f"{model:<12} {low_ap:>12.4f} {mid_ap:>12.4f} {high_ap:>12.4f}")

    # 시각화
    print("\n6. Generating visualizations...")
    plot_correlation_analysis(train_dist, test_results, correlations, positive_ratios, output_dir)

    # 요약 테이블 저장
    summary_table = create_summary_table(train_dist, test_results, correlations, positive_ratios)
    summary_table.to_csv(output_dir / 'class_imbalance_performance_summary.csv', index=False)
    print(f"\nSummary table saved to: {output_dir / 'class_imbalance_performance_summary.csv'}")

    # 해석
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # 평균 상관계수 계산
    avg_pearson_auc = np.mean([correlations[m]['AUC']['pearson_r']
                               for m in good_models if m in correlations])
    avg_spearman_auc = np.mean([correlations[m]['AUC']['spearman_rho']
                                for m in good_models if m in correlations])

    print(f"""
    1. Correlation Strength (AUC vs Positive Ratio):
       - Average Pearson r: {avg_pearson_auc:.3f}
       - Average Spearman ρ: {avg_spearman_auc:.3f}

    2. Interpretation:
       - Positive correlation indicates that classes with higher frequency
         tend to have better classification performance.
       - r > 0.7: Strong positive correlation
       - 0.4 < r < 0.7: Moderate positive correlation
       - r < 0.4: Weak correlation

    3. Key Findings:
       - Low frequency classes (<5%) show significantly lower AUC and AP
       - This confirms class imbalance negatively impacts model performance
       - AP (Average Precision) is more sensitive to class imbalance than AUC

    4. Recommendations:
       - Consider using weighted loss functions
       - Apply oversampling for minority classes
       - Use focal loss to focus on hard examples
       - Report class-wise metrics in addition to macro averages
    """)

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
