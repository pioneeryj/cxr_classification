"""데이터셋 레이블 분포 분석 스크립트"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

# 설정
topdir = Path("/mnt/HDD/dataset/medical_report")
chexpert_labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# config에서 사용하는 설정
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
        print(f"Filtered to subject prefixes {subject_prefix_filter}: {len(splitpaths)} records")

    meta = pd.merge(metadata, splitpaths, on=["dicom_id", "subject_id", "study_id"])
    meta = pd.merge(meta, chexpert, on=["subject_id", "study_id"])

    return meta

def get_cv_split(allrecords, num_folds, fold, val_size, random_state):
    """CV split 수행 (subject level)"""
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
    trainrecords = subjrecs(train_subj)
    valrecords = subjrecs(val_subj)
    testrecords = subjrecs(test_subj)

    return trainrecords, valrecords, testrecords

def analyze_label_distribution(df, split_name, label_method="zeros_uncertain_nomask"):
    """레이블 분포 분석 (zeros_uncertain_nomask 기준)"""
    results = {}

    for label in chexpert_labels:
        values = df[label].values

        if label == "No Finding":
            # No Finding은 missing_neg 방식
            positive = np.sum(values == 1.0)
            negative = np.sum((values == 0.0) | np.isnan(values))
        else:
            # zeros_uncertain_nomask 방식
            positive = np.sum(values == 1.0)
            original_negative = np.sum(values == 0.0)
            original_uncertain = np.sum(values == -1.0)
            original_missing = np.sum(np.isnan(values))
            negative = original_negative + original_uncertain + original_missing

        total = len(df)
        results[label] = {
            'positive': positive,
            'negative': negative,
            'total': total,
            'positive_ratio': positive / total * 100,
            'negative_ratio': negative / total * 100,
        }

    return results

def create_distribution_table(train_dist, val_dist, test_dist):
    """분포 표 생성"""
    rows = []
    for label in chexpert_labels:
        row = {
            'Label': label,
            'Train Pos': train_dist[label]['positive'],
            'Train Neg': train_dist[label]['negative'],
            'Train Total': train_dist[label]['total'],
            'Train Pos%': f"{train_dist[label]['positive_ratio']:.2f}%",
            'Val Pos': val_dist[label]['positive'],
            'Val Neg': val_dist[label]['negative'],
            'Val Total': val_dist[label]['total'],
            'Val Pos%': f"{val_dist[label]['positive_ratio']:.2f}%",
            'Test Pos': test_dist[label]['positive'],
            'Test Neg': test_dist[label]['negative'],
            'Test Total': test_dist[label]['total'],
            'Test Pos%': f"{test_dist[label]['positive_ratio']:.2f}%",
        }
        rows.append(row)

    return pd.DataFrame(rows)

def plot_distribution(train_dist, val_dist, test_dist, save_path):
    """분포 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    labels = chexpert_labels
    x = np.arange(len(labels))
    width = 0.25

    # 1. Positive count comparison
    ax1 = axes[0, 0]
    train_pos = [train_dist[l]['positive'] for l in labels]
    val_pos = [val_dist[l]['positive'] for l in labels]
    test_pos = [test_dist[l]['positive'] for l in labels]

    ax1.bar(x - width, train_pos, width, label='Train', color='steelblue', alpha=0.8)
    ax1.bar(x, val_pos, width, label='Val', color='darkorange', alpha=0.8)
    ax1.bar(x + width, test_pos, width, label='Test', color='forestgreen', alpha=0.8)

    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Positive Label Count by Split', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Positive ratio comparison
    ax2 = axes[0, 1]
    train_ratio = [train_dist[l]['positive_ratio'] for l in labels]
    val_ratio = [val_dist[l]['positive_ratio'] for l in labels]
    test_ratio = [test_dist[l]['positive_ratio'] for l in labels]

    ax2.bar(x - width, train_ratio, width, label='Train', color='steelblue', alpha=0.8)
    ax2.bar(x, val_ratio, width, label='Val', color='darkorange', alpha=0.8)
    ax2.bar(x + width, test_ratio, width, label='Test', color='forestgreen', alpha=0.8)

    ax2.set_ylabel('Positive Ratio (%)', fontsize=12)
    ax2.set_title('Positive Label Ratio by Split', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Stacked bar for Train
    ax3 = axes[1, 0]
    train_pos_arr = np.array([train_dist[l]['positive'] for l in labels])
    train_neg_arr = np.array([train_dist[l]['negative'] for l in labels])

    ax3.barh(labels, train_pos_arr, label='Positive', color='coral', alpha=0.8)
    ax3.barh(labels, train_neg_arr, left=train_pos_arr, label='Negative', color='skyblue', alpha=0.8)

    ax3.set_xlabel('Count', fontsize=12)
    ax3.set_title('Train Set: Positive vs Negative Distribution', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(axis='x', alpha=0.3)

    # 4. Dataset size comparison
    ax4 = axes[1, 1]
    train_total = train_dist[labels[0]]['total']
    val_total = val_dist[labels[0]]['total']
    test_total = test_dist[labels[0]]['total']

    splits = ['Train', 'Val', 'Test']
    totals = [train_total, val_total, test_total]
    colors = ['steelblue', 'darkorange', 'forestgreen']

    bars = ax4.bar(splits, totals, color=colors, alpha=0.8, edgecolor='black')

    for bar, total in zip(bars, totals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{total:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Dataset Size by Split', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    total_all = sum(totals)
    for i, (split, total) in enumerate(zip(splits, totals)):
        pct = total / total_all * 100
        ax4.text(i, total / 2, f'{pct:.1f}%', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {save_path}")

def plot_heatmap(train_dist, val_dist, test_dist, save_path):
    """히트맵으로 positive ratio 시각화"""
    fig, ax = plt.subplots(figsize=(10, 10))

    data = np.array([
        [train_dist[l]['positive_ratio'] for l in chexpert_labels],
        [val_dist[l]['positive_ratio'] for l in chexpert_labels],
        [test_dist[l]['positive_ratio'] for l in chexpert_labels]
    ])

    im = ax.imshow(data.T, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Train', 'Val', 'Test'], fontsize=12)
    ax.set_yticks(np.arange(len(chexpert_labels)))
    ax.set_yticklabels(chexpert_labels, fontsize=11)

    for i in range(len(chexpert_labels)):
        for j in range(3):
            text = ax.text(j, i, f'{data[j, i]:.1f}%',
                          ha="center", va="center", color="black" if data[j, i] < 30 else "white",
                          fontsize=10)

    ax.set_title('Positive Label Ratio Heatmap (%)', fontsize=14, fontweight='bold', pad=20)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Positive Ratio (%)', rotation=-90, va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def main():
    print("=" * 60)
    print("CXR Classification Dataset Label Distribution Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Data directory: {topdir}")
    print(f"  - Subject prefix filter: {subject_prefix_filter}")
    print(f"  - Split type: CV (fold {fold} of {num_folds})")
    print(f"  - Val size: {val_size}")
    print(f"  - Label method: zeros_uncertain_nomask")
    print()

    # 데이터 로드
    print("Loading metadata...")
    allrecords = load_all_metadata(topdir, subject_prefix_filter=subject_prefix_filter)
    print(f"Total records after filtering: {len(allrecords)}")

    # CV split
    print("\nPerforming CV split...")
    train_df, val_df, test_df = get_cv_split(allrecords, num_folds, fold, val_size, random_state)

    print(f"\nDataset sizes:")
    print(f"  - Train: {len(train_df):,} samples")
    print(f"  - Val:   {len(val_df):,} samples")
    print(f"  - Test:  {len(test_df):,} samples")
    print(f"  - Total: {len(train_df) + len(val_df) + len(test_df):,} samples")

    # 레이블 분포 분석
    print("\nAnalyzing label distributions...")
    train_dist = analyze_label_distribution(train_df, 'Train')
    val_dist = analyze_label_distribution(val_df, 'Val')
    test_dist = analyze_label_distribution(test_df, 'Test')

    # 표 생성 및 출력
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION TABLE")
    print("=" * 60)

    dist_table = create_distribution_table(train_dist, val_dist, test_dist)
    print(dist_table.to_string(index=False))

    # CSV 저장
    output_dir = Path("/home/yoonji/mrg/cxr_classification/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "label_distribution_table.csv"
    dist_table.to_csv(csv_path, index=False)
    print(f"\nTable saved to: {csv_path}")

    # 시각화
    print("\nGenerating visualizations...")
    plot_distribution(train_dist, val_dist, test_dist, output_dir / "label_distribution_barplot.png")
    plot_heatmap(train_dist, val_dist, test_dist, output_dir / "label_distribution_heatmap.png")

    # 요약 통계
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print("\nPositive Ratio Summary (%):")
    summary_data = {
        'Label': chexpert_labels,
        'Train': [f"{train_dist[l]['positive_ratio']:.2f}" for l in chexpert_labels],
        'Val': [f"{val_dist[l]['positive_ratio']:.2f}" for l in chexpert_labels],
        'Test': [f"{test_dist[l]['positive_ratio']:.2f}" for l in chexpert_labels],
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\nAverage Positive Ratio across all labels:")
    train_avg = np.mean([train_dist[l]['positive_ratio'] for l in chexpert_labels])
    val_avg = np.mean([val_dist[l]['positive_ratio'] for l in chexpert_labels])
    test_avg = np.mean([test_dist[l]['positive_ratio'] for l in chexpert_labels])
    print(f"  - Train: {train_avg:.2f}%")
    print(f"  - Val:   {val_avg:.2f}%")
    print(f"  - Test:  {test_avg:.2f}%")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
