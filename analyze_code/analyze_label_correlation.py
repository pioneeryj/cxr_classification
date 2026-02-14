"""
MIMIC-CXR Label Correlation Analysis

Computes pairwise label co-occurrence statistics from CV train split,
and saves significant pairs (|phi| >= threshold) as JSON for use as
auxiliary information in LLM-based medical report generation prompts.

Usage:
    python analyze_label_correlation.py [--num_folds 10] [--fold 0] [--phi_threshold 0.15]
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import argparse


CHEXPERT_LABELS = [
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


def load_labels_cv(data_dir: Path, num_folds: int = 10, fold: int = 0,
                   val_size: float = 0.1, random_state: int = 0) -> pd.DataFrame:
    """Load MIMIC-CXR labels using CV split (train portion only).

    Uses the same KFold logic as mimic_cxr_jpg.cv() to ensure consistency
    with the classification training pipeline.
    """
    chexpert = pd.read_csv(data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split_df = pd.read_csv(data_dir / "mimic-cxr-2.0.0-split.csv.gz")

    data = pd.merge(split_df, chexpert, on=["subject_id", "study_id"])

    # KFold at subject level (same as mimic_cxr_jpg.cv)
    uniq_subj = data["subject_id"].unique()
    kf = KFold(num_folds)
    for k, (trainval_ix, _) in enumerate(kf.split(uniq_subj)):
        if k != fold:
            continue
        trainval_subj = uniq_subj[trainval_ix]
        train_subj, _ = train_test_split(
            trainval_subj, test_size=val_size,
            random_state=random_state, shuffle=False,
        )

    return data[data["subject_id"].isin(train_subj)]


def compute_label_correlations(data: pd.DataFrame, phi_threshold: float = 0.15) -> list:
    """Compute pairwise label correlations and return significant pairs.

    For each pair of 14 CheXpert labels, computes:
    - phi coefficient (association strength)
    - P(B|A), P(A|B) (conditional co-occurrence rates)

    Returns only pairs with |phi| >= phi_threshold, sorted by |phi| descending.
    """
    # Binarize: 1.0 -> 1, everything else -> 0
    binary = pd.DataFrame({
        label: (data[label] == 1.0).astype(int) for label in CHEXPERT_LABELS
    })
    n_samples = len(binary)

    significant_pairs = []

    for i in range(len(CHEXPERT_LABELS)):
        for j in range(i + 1, len(CHEXPERT_LABELS)):
            la, lb = CHEXPERT_LABELS[i], CHEXPERT_LABELS[j]
            a, b = binary[la].values, binary[lb].values

            n11 = int(((a == 1) & (b == 1)).sum())
            n00 = int(((a == 0) & (b == 0)).sum())
            n10 = int(((a == 1) & (b == 0)).sum())
            n01 = int(((a == 0) & (b == 1)).sum())

            # Phi coefficient
            denom = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
            phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0

            if abs(phi) < phi_threshold:
                continue

            count_a = n11 + n10
            count_b = n11 + n01

            significant_pairs.append({
                "label_a": la,
                "label_b": lb,
                "correlation": "positive" if phi > 0 else "negative",
                "phi": round(phi, 4),
                "p_b_given_a": round(n11 / count_a, 4) if count_a > 0 else 0.0,
                "p_a_given_b": round(n11 / count_b, 4) if count_b > 0 else 0.0,
                "cooccurrence_rate": round(n11 / n_samples, 4),
            })

    significant_pairs.sort(key=lambda x: abs(x["phi"]), reverse=True)
    return significant_pairs


def main():
    parser = argparse.ArgumentParser(description="Compute label correlations for LLM prompt")
    parser.add_argument("--data_dir", type=str, default="/mnt/HDD/dataset/medical_report")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--phi_threshold", type=float, default=0.15,
                        help="Minimum |phi| to include a pair (default: 0.15)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CV train set (fold {args.fold}/{args.num_folds})...")
    data = load_labels_cv(data_dir, num_folds=args.num_folds, fold=args.fold)
    print(f"Train set: {len(data)} samples, {data['subject_id'].nunique()} subjects")

    print(f"Computing pairwise correlations (|phi| >= {args.phi_threshold})...")
    pairs = compute_label_correlations(data, phi_threshold=args.phi_threshold)

    result = {
        "description": "CheXpert label pairwise correlations from MIMIC-CXR training data. "
                       "Only pairs with significant association (|phi| >= threshold) are included.",
        "num_folds": args.num_folds,
        "fold": args.fold,
        "n_train_samples": len(data),
        "phi_threshold": args.phi_threshold,
        "num_pairs": len(pairs),
        "pairs": pairs,
    }

    out_path = output_dir / f"label_correlations_fold{args.fold}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(pairs)} significant pairs to {out_path}")
    print(f"\nSignificant pairs (|phi| >= {args.phi_threshold}):")
    for p in pairs:
        print(f"  {p['label_a']:30s} <-> {p['label_b']:30s}  "
              f"phi={p['phi']:+.4f} ({p['correlation']})  "
              f"P(B|A)={p['p_b_given_a']:.4f}  P(A|B)={p['p_a_given_b']:.4f}")


if __name__ == "__main__":
    main()
