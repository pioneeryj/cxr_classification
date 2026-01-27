"""Calibration metrics for multi-label CXR classification models."""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


def brier_score(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Brier score for binary classification.

    Brier score = mean((p - y)^2) where p is predicted probability and y is true label.
    Lower is better (perfect score = 0).

    Args:
        probs: Predicted probabilities [N, L] where N = num_samples, L = num_labels
        labels: True binary labels [N, L]
        mask: Valid label mask [N, L], 1 = valid, 0 = ignore

    Returns:
        Dictionary with per-label Brier scores and overall (pooled) score
    """
    assert probs.shape == labels.shape == mask.shape

    results = {}
    n_labels = probs.shape[1]

    # Compute per-label Brier score
    for label_idx in range(n_labels):
        label_mask = mask[:, label_idx] == 1

        if label_mask.sum() == 0:
            continue  # Skip labels with no valid samples

        label_probs = probs[label_mask, label_idx]
        label_true = labels[label_mask, label_idx]

        # Brier score: mean squared error
        brier = np.mean((label_probs - label_true) ** 2)
        results[f'label_{label_idx}'] = float(brier)

    # Compute overall (pooled) Brier score
    valid_mask = mask == 1
    if valid_mask.sum() > 0:
        valid_probs = probs[valid_mask]
        valid_labels = labels[valid_mask]
        overall_brier = np.mean((valid_probs - valid_labels) ** 2)
        results['overall'] = float(overall_brier)

    return results


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) using fixed binning.

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower is better (perfect calibration = 0).

    Args:
        probs: Predicted probabilities [N, L]
        labels: True binary labels [N, L]
        mask: Valid label mask [N, L]
        n_bins: Number of bins for calibration (default: 15)

    Returns:
        Dictionary with per-label ECE and overall (pooled) ECE
    """
    assert probs.shape == labels.shape == mask.shape

    results = {}
    n_labels = probs.shape[1]

    # Bin edges [0, 1/n_bins, 2/n_bins, ..., 1]
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Compute per-label ECE
    for label_idx in range(n_labels):
        label_mask = mask[:, label_idx] == 1

        if label_mask.sum() == 0:
            continue

        label_probs = probs[label_mask, label_idx]
        label_true = labels[label_mask, label_idx]

        ece = 0.0
        total_samples = len(label_probs)

        for bin_idx in range(n_bins):
            # Find samples in this bin
            bin_lower = bin_edges[bin_idx]
            bin_upper = bin_edges[bin_idx + 1]

            if bin_idx == n_bins - 1:  # Last bin includes right edge
                in_bin = (label_probs >= bin_lower) & (label_probs <= bin_upper)
            else:
                in_bin = (label_probs >= bin_lower) & (label_probs < bin_upper)

            if in_bin.sum() == 0:
                continue

            # Compute bin statistics
            bin_probs = label_probs[in_bin]
            bin_labels = label_true[in_bin]

            avg_confidence = np.mean(bin_probs)
            avg_accuracy = np.mean(bin_labels)
            bin_weight = len(bin_probs) / total_samples

            # Weighted absolute difference
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)

        results[f'label_{label_idx}'] = float(ece)

    # Compute overall (pooled) ECE
    valid_mask = mask == 1
    if valid_mask.sum() > 0:
        valid_probs = probs[valid_mask]
        valid_labels = labels[valid_mask]

        overall_ece = 0.0
        total_samples = len(valid_probs)

        for bin_idx in range(n_bins):
            bin_lower = bin_edges[bin_idx]
            bin_upper = bin_edges[bin_idx + 1]

            if bin_idx == n_bins - 1:
                in_bin = (valid_probs >= bin_lower) & (valid_probs <= bin_upper)
            else:
                in_bin = (valid_probs >= bin_lower) & (valid_probs < bin_upper)

            if in_bin.sum() == 0:
                continue

            bin_probs = valid_probs[in_bin]
            bin_labels = valid_labels[in_bin]

            avg_confidence = np.mean(bin_probs)
            avg_accuracy = np.mean(bin_labels)
            bin_weight = len(bin_probs) / total_samples

            overall_ece += bin_weight * np.abs(avg_confidence - avg_accuracy)

        results['overall'] = float(overall_ece)

    return results


def aurc(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Area Under Risk-Coverage Curve (AURC).

    AURC measures how well the model's confidence correlates with correctness.
    Lower is better (perfect AURC = 0).

    Confidence is defined as max(p, 1-p) (distance from decision boundary).
    Risk is classification error: (p >= 0.5) != y

    Args:
        probs: Predicted probabilities [N, L]
        labels: True binary labels [N, L]
        mask: Valid label mask [N, L]

    Returns:
        Dictionary with per-label AURC and overall (pooled) AURC
    """
    assert probs.shape == labels.shape == mask.shape

    results = {}
    n_labels = probs.shape[1]

    # Compute per-label AURC
    for label_idx in range(n_labels):
        label_mask = mask[:, label_idx] == 1

        if label_mask.sum() == 0:
            continue

        label_probs = probs[label_mask, label_idx]
        label_true = labels[label_mask, label_idx]

        # Compute confidence: max(p, 1-p)
        confidence = np.maximum(label_probs, 1 - label_probs)

        # Compute risk (classification error)
        predictions = (label_probs >= 0.5).astype(float)
        risk = (predictions != label_true).astype(float)

        # Sort by descending confidence
        sorted_indices = np.argsort(-confidence)
        sorted_risk = risk[sorted_indices]

        # Compute cumulative risk (coverage curve)
        n_samples = len(sorted_risk)
        cumulative_risk = np.cumsum(sorted_risk) / np.arange(1, n_samples + 1)

        # AURC = area under the risk-coverage curve
        # Trapezoidal rule for integration
        coverage = np.arange(1, n_samples + 1) / n_samples
        aurc_value = np.trapz(cumulative_risk, coverage)

        results[f'label_{label_idx}'] = float(aurc_value)

    # Compute overall (pooled) AURC
    valid_mask = mask == 1
    if valid_mask.sum() > 0:
        valid_probs = probs[valid_mask]
        valid_labels = labels[valid_mask]

        confidence = np.maximum(valid_probs, 1 - valid_probs)
        predictions = (valid_probs >= 0.5).astype(float)
        risk = (predictions != valid_labels).astype(float)

        sorted_indices = np.argsort(-confidence)
        sorted_risk = risk[sorted_indices]

        n_samples = len(sorted_risk)
        cumulative_risk = np.cumsum(sorted_risk) / np.arange(1, n_samples + 1)
        coverage = np.arange(1, n_samples + 1) / n_samples

        overall_aurc = np.trapz(cumulative_risk, coverage)
        results['overall'] = float(overall_aurc)

    return results


class CalibrationEvaluator:
    """
    Wrapper class for accumulating predictions and computing calibration metrics.

    Usage:
        evaluator = CalibrationEvaluator()

        # Accumulate during test loop
        for batch in test_loader:
            logits = model(batch['images'])
            evaluator.add_batch(logits, batch['labels'], batch['mask'])

        # Compute all metrics at the end
        results = evaluator.compute_metrics()
    """

    def __init__(self):
        """Initialize evaluator with empty buffers."""
        self.all_logits = []
        self.all_labels = []
        self.all_masks = []

    def add_batch(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        Add a batch of predictions to the accumulator.

        Args:
            logits: Model output logits [B, L]
            labels: True binary labels [B, L]
            mask: Valid label mask [B, L]
        """
        # Convert to numpy and move to CPU if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        self.all_logits.append(logits)
        self.all_labels.append(labels)
        self.all_masks.append(mask)

    def compute_metrics(self, n_bins: int = 15) -> Dict[str, Dict[str, float]]:
        """
        Compute all calibration metrics on accumulated data.

        Args:
            n_bins: Number of bins for ECE (default: 15)

        Returns:
            Dictionary with keys 'brier_score', 'ece', 'aurc', each containing
            per-label and overall metrics
        """
        if len(self.all_logits) == 0:
            raise ValueError("No data accumulated. Call add_batch() first.")

        # Concatenate all batches
        logits = np.concatenate(self.all_logits, axis=0)
        labels = np.concatenate(self.all_labels, axis=0)
        masks = np.concatenate(self.all_masks, axis=0)

        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))

        # Compute all metrics
        results = {
            'brier_score': brier_score(probs, labels, masks),
            'ece': expected_calibration_error(probs, labels, masks, n_bins=n_bins),
            'aurc': aurc(probs, labels, masks),
        }

        return results

    def reset(self):
        """Clear all accumulated data."""
        self.all_logits = []
        self.all_labels = []
        self.all_masks = []

    def print_results(self, results: Dict[str, Dict[str, float]], label_names: Optional[list] = None):
        """
        Pretty print calibration results.

        Args:
            results: Output from compute_metrics()
            label_names: Optional list of label names for display
        """
        print("\n" + "="*80)
        print("Calibration Metrics")
        print("="*80)

        for metric_name, metric_results in results.items():
            print(f"\n{metric_name.upper()}:")
            print("-"*80)

            # Print per-label results
            for key, value in sorted(metric_results.items()):
                if key == 'overall':
                    continue

                # Extract label index
                label_idx = int(key.split('_')[1])

                if label_names and label_idx < len(label_names):
                    label_str = f"{label_names[label_idx]:<30}"
                else:
                    label_str = f"Label {label_idx:<24}"

                print(f"  {label_str}: {value:.6f}")

            # Print overall result
            if 'overall' in metric_results:
                print("-"*80)
                print(f"  {'Overall (pooled)':<30}: {metric_results['overall']:.6f}")

        print("="*80)


# Example usage
if __name__ == '__main__':
    # Simulate test data
    np.random.seed(42)

    n_samples = 1000
    n_labels = 14

    # Simulate logits, labels, and masks
    logits = np.random.randn(n_samples, n_labels)
    labels = np.random.randint(0, 2, (n_samples, n_labels))
    masks = np.random.randint(0, 2, (n_samples, n_labels))  # Some invalid labels

    # Apply sigmoid
    probs = 1 / (1 + np.exp(-logits))

    # Compute metrics directly
    print("Direct computation:")
    brier = brier_score(probs, labels, masks)
    print(f"Brier Score (overall): {brier.get('overall', 'N/A')}")

    ece = expected_calibration_error(probs, labels, masks, n_bins=15)
    print(f"ECE (overall): {ece.get('overall', 'N/A')}")

    aurc_result = aurc(probs, labels, masks)
    print(f"AURC (overall): {aurc_result.get('overall', 'N/A')}")

    # Using wrapper
    print("\n\nUsing CalibrationEvaluator:")
    evaluator = CalibrationEvaluator()

    # Simulate batches
    batch_size = 100
    for i in range(0, n_samples, batch_size):
        batch_logits = torch.tensor(logits[i:i+batch_size])
        batch_labels = torch.tensor(labels[i:i+batch_size])
        batch_masks = torch.tensor(masks[i:i+batch_size])

        evaluator.add_batch(batch_logits, batch_labels, batch_masks)

    # Compute all metrics
    results = evaluator.compute_metrics(n_bins=15)

    # Print results
    label_names = [f"Disease_{i}" for i in range(n_labels)]
    evaluator.print_results(results, label_names)
