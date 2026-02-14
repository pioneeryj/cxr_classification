"""
Label Frequency Analysis for MIMIC-CXR Test Set

Computes log frequency for each label and groups them into:
- Head group (high frequency)
- Medium group (medium frequency)
- Tail group (low frequency)

Based on quantile-based grouping (33.3% each).

Usage:
    python analyze_label_frequency.py [--config CONFIG_FILE] [--output_dir OUTPUT_DIR]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config
from data_transforms import get_val_transform
import mimic_cxr_jpg


CHEXPERT_LABELS = mimic_cxr_jpg.chexpert_labels


def load_test_labels_from_config(config: dict) -> tuple:
    """
    Load test set labels using the same data loading logic as training/testing.
    This ensures we use the exact same dataset (with subject_prefix_filter etc.)
    """
    data_config = config['data']
    model_name = config['model']['name']
    transform = get_val_transform(model_name)

    if data_config['split_type'] == 'official':
        _, _, test_ds = mimic_cxr_jpg.official_split(
            datadir=data_config['datadir'],
            dicom_id_file=data_config.get('dicom_id_file'),
            image_subdir=data_config['image_subdir'],
            train_transform=transform,
            test_transform=transform,
            label_method=data_config['label_method'],
            subject_prefix_filter=data_config.get('subject_prefix_filter'),
        )
    else:
        _, _, test_ds = mimic_cxr_jpg.cv(
            num_folds=data_config['num_folds'],
            fold=data_config['fold'],
            datadir=data_config['datadir'],
            dicom_id_file=data_config.get('dicom_id_file'),
            image_subdir=data_config['image_subdir'],
            val_size=data_config['val_size'],
            random_state=data_config['random_state'],
            stratify=data_config['stratify'],
            train_transform=transform,
            test_transform=transform,
            label_method=data_config['label_method'],
            subject_prefix_filter=data_config.get('subject_prefix_filter'),
        )

    # Extract labels from dataset
    labels_list = []
    for i in range(len(test_ds)):
        try:
            _, labels, mask = test_ds[i]
            labels_list.append(labels.numpy())
        except:
            continue

    labels_array = np.stack(labels_list, axis=0)
    return labels_array, len(test_ds)


def load_test_labels_raw(data_dir: Path) -> pd.DataFrame:
    """Load MIMIC-CXR test set labels (raw, without config filtering)."""
    chexpert = pd.read_csv(data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split = pd.read_csv(data_dir / "mimic-cxr-2.0.0-split.csv.gz")

    data = pd.merge(split, chexpert, on=["subject_id", "study_id"])
    test_data = data[data["split"] == "test"]

    return test_data


def compute_label_frequency_from_array(labels_array: np.ndarray, label_names: list) -> pd.DataFrame:
    """
    Compute frequency statistics from numpy array of labels.
    labels_array: shape [n_samples, n_labels], values 0 or 1
    """
    n_samples = labels_array.shape[0]

    stats = []
    for idx, label in enumerate(label_names):
        # Count positive samples (label == 1)
        positive_count = int((labels_array[:, idx] == 1).sum())
        frequency = positive_count / n_samples
        log_frequency = np.log10(positive_count) if positive_count > 0 else 0

        stats.append({
            'label': label,
            'positive_count': positive_count,
            'total_samples': n_samples,
            'frequency': frequency,
            'log_frequency': log_frequency,
        })

    return pd.DataFrame(stats)


def compute_label_frequency(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Compute frequency statistics for each label from DataFrame.
    Positive label = 1.0 (treating -1 and NaN as negative)
    """
    n_samples = len(df)

    stats = []
    for label in labels:
        # Count positive samples (1.0 only)
        positive_count = (df[label] == 1.0).sum()
        frequency = positive_count / n_samples
        log_frequency = np.log10(positive_count) if positive_count > 0 else 0

        stats.append({
            'label': label,
            'positive_count': int(positive_count),
            'total_samples': n_samples,
            'frequency': frequency,
            'log_frequency': log_frequency,
        })

    return pd.DataFrame(stats)


def assign_groups_by_quantile(freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign labels to Head/Medium/Tail groups based on log_frequency quantiles.
    - Head: top 33.3% (high frequency)
    - Medium: middle 33.3%
    - Tail: bottom 33.3% (low frequency)
    """
    # Sort by log_frequency descending
    freq_df = freq_df.sort_values('log_frequency', ascending=False).reset_index(drop=True)

    n_labels = len(freq_df)

    # Calculate quantile thresholds
    q_high = freq_df['log_frequency'].quantile(0.667)  # Top 33.3%
    q_low = freq_df['log_frequency'].quantile(0.333)   # Bottom 33.3%

    # Assign groups
    def assign_group(log_freq):
        if log_freq >= q_high:
            return 'Head'
        elif log_freq >= q_low:
            return 'Medium'
        else:
            return 'Tail'

    freq_df['group'] = freq_df['log_frequency'].apply(assign_group)

    # Add rank
    freq_df['rank'] = range(1, n_labels + 1)

    return freq_df, q_high, q_low


def main():
    parser = argparse.ArgumentParser(description="Analyze label frequency in MIMIC-CXR test set")
    parser.add_argument("--config", type=str, default="configs/resnet50_config.yaml",
                        help="Path to model config file (to use same data filtering)")
    parser.add_argument("--data_dir", type=str, default="/mnt/HDD/dataset/medical_report",
                        help="Directory containing MIMIC-CXR metadata files (used if no config)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for results")
    parser.add_argument("--labels_npy", type=str, default=None,
                        help="Path to saved labels.npy file (fastest option)")
    parser.add_argument("--use_raw", action="store_true",
                        help="Use raw test set without config filtering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIMIC-CXR Test Set Label Frequency Analysis")
    print("=" * 80)

    if args.labels_npy:
        # Load from saved numpy file (fastest)
        print(f"\nLoading labels from: {args.labels_npy}")
        labels_array = np.load(args.labels_npy)
        print(f"Test set size: {labels_array.shape[0]} samples")

        # Compute frequency statistics
        print("\nComputing label frequencies...")
        freq_df = compute_label_frequency_from_array(labels_array, CHEXPERT_LABELS)

    elif args.use_raw:
        # Load raw test set without filtering
        print("\nLoading raw test set labels (no filtering)...")
        data_dir = Path(args.data_dir)
        test_data = load_test_labels_raw(data_dir)
        print(f"Test set size: {len(test_data)} samples")

        # Compute frequency statistics
        print("\nComputing label frequencies...")
        freq_df = compute_label_frequency(test_data, CHEXPERT_LABELS)
    else:
        # Load using config (same as training/testing)
        print(f"\nLoading test set using config: {args.config}")
        config = load_config(args.config)
        print(f"  subject_prefix_filter: {config['data'].get('subject_prefix_filter', 'None')}")
        print(f"  split_type: {config['data']['split_type']}")

        labels_array, n_samples = load_test_labels_from_config(config)
        print(f"Test set size: {n_samples} samples (loaded {labels_array.shape[0]})")

        # Compute frequency statistics
        print("\nComputing label frequencies...")
        freq_df = compute_label_frequency_from_array(labels_array, CHEXPERT_LABELS)

    # Assign groups
    freq_df, q_high, q_low = assign_groups_by_quantile(freq_df)

    # Print results
    print("\n" + "=" * 80)
    print("Label Frequency Statistics (sorted by frequency)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Label':<30} {'Count':>10} {'Freq':>10} {'LogFreq':>10} {'Group':<8}")
    print("-" * 80)

    for _, row in freq_df.iterrows():
        print(f"{row['rank']:<6} {row['label']:<30} {row['positive_count']:>10} "
              f"{row['frequency']:>10.4f} {row['log_frequency']:>10.4f} {row['group']:<8}")

    print("-" * 80)
    print(f"\nQuantile thresholds (log_frequency):")
    print(f"  Head >= {q_high:.4f}")
    print(f"  Medium >= {q_low:.4f}")
    print(f"  Tail < {q_low:.4f}")

    # Create group summary
    print("\n" + "=" * 80)
    print("Group Summary")
    print("=" * 80)

    group_summary = []
    for group_name in ['Head', 'Medium', 'Tail']:
        group_data = freq_df[freq_df['group'] == group_name]
        labels_in_group = group_data['label'].tolist()

        avg_frequency = group_data['frequency'].mean()
        avg_log_frequency = group_data['log_frequency'].mean()
        avg_count = group_data['positive_count'].mean()
        total_count = group_data['positive_count'].sum()

        group_summary.append({
            'group': group_name,
            'n_labels': len(labels_in_group),
            'labels': ', '.join(labels_in_group),
            'avg_frequency': avg_frequency,
            'avg_log_frequency': avg_log_frequency,
            'avg_positive_count': avg_count,
            'total_positive_count': total_count,
        })

        print(f"\n{group_name} Group ({len(labels_in_group)} labels):")
        print(f"  Labels: {', '.join(labels_in_group)}")
        print(f"  Avg Frequency: {avg_frequency:.4f}")
        print(f"  Avg Log Frequency: {avg_log_frequency:.4f}")
        print(f"  Avg Positive Count: {avg_count:.1f}")
        print(f"  Total Positive Count: {total_count}")

    # Save detailed results
    freq_df.to_csv(output_dir / 'label_frequency_detailed.csv', index=False)
    print(f"\nSaved detailed results to: {output_dir / 'label_frequency_detailed.csv'}")

    # Save group summary
    group_summary_df = pd.DataFrame(group_summary)
    group_summary_df.to_csv(output_dir / 'label_frequency_groups.csv', index=False)
    print(f"Saved group summary to: {output_dir / 'label_frequency_groups.csv'}")

    # Save simple group assignment (just label and group)
    simple_groups = freq_df[['label', 'group', 'positive_count', 'frequency', 'log_frequency', 'rank']]
    simple_groups.to_csv(output_dir / 'label_group_assignment.csv', index=False)
    print(f"Saved group assignment to: {output_dir / 'label_group_assignment.csv'}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
