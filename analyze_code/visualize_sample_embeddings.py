"""
Test Sample Embedding Visualization

Extracts feature embeddings from test samples and visualizes them using t-SNE:
1. Per-label t-SNE visualization with clustering metrics
2. Decision boundary visualization
3. Multi-label embedding analysis

Usage:
    python visualize_sample_embeddings.py --config configs/resnet50_config.yaml \
        --checkpoint /path/to/checkpoint.pt --output_dir ./outputs/sample_embeddings
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from model_factory import get_model
from utils import load_config, set_seed
from data_transforms import get_val_transform
import mimic_cxr_jpg


def safe_collate_fn(batch):
    """Custom collate function that creates independent tensor copies."""
    images, labels, masks = zip(*batch)
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])
    return images_batch, labels_batch, masks_batch


class FeatureExtractor(nn.Module):
    """Wrapper to extract features before the classifier."""

    def __init__(self, model, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name.lower()
        self.features = None

        # Register hook to capture features
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture features before classifier."""
        if 'resnet' in self.model_name:
            # ResNet: hook before fc layer (after avgpool)
            def hook(module, input, output):
                self.features = output.view(output.size(0), -1)
            self.model.avgpool.register_forward_hook(hook)

        elif 'densenet' in self.model_name:
            # DenseNet: hook after features and relu, before classifier
            def hook(module, input, output):
                out = torch.nn.functional.relu(output, inplace=True)
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                self.features = out.view(out.size(0), -1)
            self.model.features.register_forward_hook(hook)

        elif self.model_name == 'biovil':
            # BioViL: capture img_embedding from encoder output
            original_forward = self.model.forward

            def new_forward(x, x_prev=None):
                out = self.model.encoder(current_image=x, previous_image=x_prev)
                if self.model.use_projected_128:
                    self.features = out.projected_global_embedding
                else:
                    self.features = out.img_embedding
                logits = self.model.classifier(self.features)
                return logits

            self.model.forward = new_forward

        elif self.model_name == 'medklip':
            # MedKLIP: capture features after res_l2
            original_forward = self.model.forward

            def new_forward(x):
                x = self.model.res_features(x)
                x = self.model.global_pool(x)
                x = x.squeeze(-1).squeeze(-1)
                x = self.model.res_l1(x)
                x = torch.nn.functional.relu(x)
                x = self.model.res_l2(x)
                self.features = x
                logits = self.model.classifier(x)
                return logits

            self.model.forward = new_forward

    def forward(self, x):
        _ = self.model(x)
        return self.features


def extract_embeddings(model, dataloader, device, max_samples=None):
    """Extract feature embeddings from all samples."""
    model.eval()
    all_features = []
    all_labels = []
    all_masks = []

    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images, labels, masks = batch
            images = images.to(device).float()

            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())

            n_samples += images.size(0)
            if max_samples and n_samples >= max_samples:
                break

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
        masks = masks[:max_samples]

    return features, labels, masks


def compute_tsne(features, perplexity=30, n_iter=1000, random_state=42, n_components=2):
    """Compute t-SNE embedding."""
    print(f"Computing t-SNE {n_components}D (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, learning_rate='auto', init='pca')
    embeddings = tsne.fit_transform(features)
    return embeddings


def visualize_3d_single_label(embeddings_3d, labels, masks, label_idx, label_name,
                              output_path):
    """
    Visualize 3D t-SNE for a single label with positive/negative samples colored.
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Get valid samples for this label
    valid_mask = masks[:, label_idx] == 1
    valid_embeddings = embeddings_3d[valid_mask]
    valid_labels = labels[valid_mask, label_idx]

    if len(valid_embeddings) == 0:
        print(f"  No valid samples for {label_name}")
        return

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    colors = ['#3498db', '#e74c3c']  # Blue for negative, Red for positive
    for class_val, color, label_text in [(0, colors[0], 'Negative'), (1, colors[1], 'Positive')]:
        mask = valid_labels == class_val
        ax.scatter(valid_embeddings[mask, 0], valid_embeddings[mask, 1], valid_embeddings[mask, 2],
                  c=color, label=f'{label_text} (n={mask.sum()})',
                  alpha=0.5, s=10, edgecolors='none')

    n_pos = int((valid_labels == 1).sum())
    n_neg = int((valid_labels == 0).sum())

    ax.set_title(f'{label_name}\n(+:{n_pos}, -:{n_neg})', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.set_zlabel('t-SNE 3', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_all_labels_grid(embeddings_3d, labels, masks, output_path):
    """Create a grid visualization of all labels in 3D."""
    from mpl_toolkits.mplot3d import Axes3D

    n_labels = len(mimic_cxr_jpg.chexpert_labels)
    n_cols = 4
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))

    colors = ['#3498db', '#e74c3c']

    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        valid_mask = masks[:, idx] == 1
        valid_embeddings = embeddings_3d[valid_mask]
        valid_labels = labels[valid_mask, idx]

        if len(valid_embeddings) == 0:
            ax.set_title(f'{label_name}\n(No samples)', fontsize=10)
            continue

        # Plot
        for class_val, color in [(0, colors[0]), (1, colors[1])]:
            mask = valid_labels == class_val
            ax.scatter(valid_embeddings[mask, 0], valid_embeddings[mask, 1], valid_embeddings[mask, 2],
                      c=color, alpha=0.4, s=3, edgecolors='none')

        n_pos = (valid_labels == 1).sum()
        n_neg = (valid_labels == 0).sum()
        ax.set_title(f'{label_name}\n+:{n_pos}, -:{n_neg}', fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.suptitle('3D t-SNE Visualization by Label (Blue: Negative, Red: Positive)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_multilabel_overlay(embeddings_3d, labels, masks, output_path):
    """
    3D visualization with color encoding for multi-label patterns.
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Count positive labels per sample
    n_positive_labels = (labels * masks).sum(axis=1)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                        c=n_positive_labels, cmap='YlOrRd',
                        alpha=0.6, s=10, edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Number of Positive Labels', fontsize=12)

    ax.set_title('3D t-SNE: Multi-label Distribution\n(Color = Number of Positive Labels)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.set_zlabel('t-SNE 3', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_3d_no_finding_vs_abnormal(embeddings_3d, labels, masks, output_path):
    """
    3D visualization of No Finding vs any abnormality.
    """
    from mpl_toolkits.mplot3d import Axes3D

    no_finding_idx = mimic_cxr_jpg.chexpert_labels.index('No Finding')

    # Get No Finding status
    no_finding = labels[:, no_finding_idx] == 1

    # Get any abnormality
    abnormal_indices = [i for i in range(len(mimic_cxr_jpg.chexpert_labels))
                       if i != no_finding_idx]
    has_abnormality = (labels[:, abnormal_indices] * masks[:, abnormal_indices]).sum(axis=1) > 0

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'No Finding': '#2ecc71', 'Abnormal': '#e74c3c', 'Mixed/Unknown': '#95a5a6'}

    # Categorize samples
    categories = []
    for i in range(len(labels)):
        if no_finding[i] and not has_abnormality[i]:
            categories.append('No Finding')
        elif has_abnormality[i] and not no_finding[i]:
            categories.append('Abnormal')
        else:
            categories.append('Mixed/Unknown')
    categories = np.array(categories)

    for cat, color in colors.items():
        mask = categories == cat
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                  c=color, label=f'{cat} (n={mask.sum()})',
                  alpha=0.5, s=10, edgecolors='none')

    ax.set_title('3D t-SNE: No Finding vs Abnormal', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.set_zlabel('t-SNE 3', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_all_labels_single_plot(embeddings_2d, labels, masks, output_path):
    """
    Visualize all labels in a single 2D plot with different colors per label.
    Each sample is colored by its positive labels.
    """
    n_labels = len(mimic_cxr_jpg.chexpert_labels)

    # Use distinct colormap
    cmap = plt.cm.get_cmap('tab20', n_labels)
    label_colors = [cmap(i) for i in range(n_labels)]

    fig, ax = plt.subplots(figsize=(16, 14))

    # First, plot all samples as gray background
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
              c='lightgray', alpha=0.3, s=5, edgecolors='none', label='_nolegend_')

    # Then overlay each label's positive samples
    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        valid_mask = masks[:, idx] == 1
        positive_mask = (labels[:, idx] == 1) & valid_mask

        if positive_mask.sum() > 0:
            ax.scatter(embeddings_2d[positive_mask, 0], embeddings_2d[positive_mask, 1],
                      c=[label_colors[idx]], alpha=0.6, s=15, edgecolors='none',
                      label=f'{label_name} ({positive_mask.sum()})')

    ax.set_title('t-SNE: All Labels (Positive Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_all_labels_single_plot_3d(embeddings_3d, labels, masks, output_path):
    """
    Visualize all labels in a single 3D plot with different colors per label.
    """
    from mpl_toolkits.mplot3d import Axes3D

    n_labels = len(mimic_cxr_jpg.chexpert_labels)
    cmap = plt.cm.get_cmap('tab20', n_labels)
    label_colors = [cmap(i) for i in range(n_labels)]

    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')

    # First, plot all samples as gray background
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
              c='lightgray', alpha=0.2, s=3, edgecolors='none')

    # Then overlay each label's positive samples
    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        valid_mask = masks[:, idx] == 1
        positive_mask = (labels[:, idx] == 1) & valid_mask

        if positive_mask.sum() > 0:
            ax.scatter(embeddings_3d[positive_mask, 0], embeddings_3d[positive_mask, 1],
                      embeddings_3d[positive_mask, 2],
                      c=[label_colors[idx]], alpha=0.6, s=10, edgecolors='none',
                      label=f'{label_name} ({positive_mask.sum()})')

    ax.set_title('3D t-SNE: All Labels (Positive Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    ax.set_zlabel('t-SNE 3', fontsize=10)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_interactive_all_labels_3d_html(embeddings_3d, labels, masks, output_path):
    """
    Create interactive 3D visualization with all labels in different colors.
    """
    try:
        import plotly.graph_objects as go

        n_labels = len(mimic_cxr_jpg.chexpert_labels)

        # Plotly color sequence
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896'
        ]

        fig = go.Figure()

        # Add trace for each label
        for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
            valid_mask = masks[:, idx] == 1
            positive_mask = (labels[:, idx] == 1) & valid_mask

            if positive_mask.sum() > 0:
                fig.add_trace(go.Scatter3d(
                    x=embeddings_3d[positive_mask, 0],
                    y=embeddings_3d[positive_mask, 1],
                    z=embeddings_3d[positive_mask, 2],
                    mode='markers',
                    marker=dict(size=3, color=colors[idx % len(colors)], opacity=0.6),
                    name=f'{label_name} ({positive_mask.sum()})'
                ))

        fig.update_layout(
            title='Interactive 3D t-SNE: All Labels',
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            width=1200,
            height=900,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )

        fig.write_html(output_path)
        print(f"Saved interactive 3D visualization: {output_path}")
        return True

    except ImportError:
        print("  Plotly not installed. Skipping interactive 3D visualization.")
        return False


def create_interactive_3d_html(embeddings_3d, labels, masks, output_path):
    """
    Create interactive 3D visualization using plotly (if available).
    Falls back to static matplotlib if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        no_finding_idx = mimic_cxr_jpg.chexpert_labels.index('No Finding')
        no_finding = labels[:, no_finding_idx] == 1
        abnormal_indices = [i for i in range(len(mimic_cxr_jpg.chexpert_labels))
                           if i != no_finding_idx]
        has_abnormality = (labels[:, abnormal_indices] * masks[:, abnormal_indices]).sum(axis=1) > 0

        # Categorize samples
        categories = []
        for i in range(len(labels)):
            if no_finding[i] and not has_abnormality[i]:
                categories.append('No Finding')
            elif has_abnormality[i] and not no_finding[i]:
                categories.append('Abnormal')
            else:
                categories.append('Mixed/Unknown')
        categories = np.array(categories)

        colors_map = {'No Finding': '#2ecc71', 'Abnormal': '#e74c3c', 'Mixed/Unknown': '#95a5a6'}

        fig = go.Figure()

        for cat in ['No Finding', 'Abnormal', 'Mixed/Unknown']:
            mask = categories == cat
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                marker=dict(size=3, color=colors_map[cat], opacity=0.6),
                name=f'{cat} (n={mask.sum()})'
            ))

        fig.update_layout(
            title='Interactive 3D t-SNE: No Finding vs Abnormal',
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            width=1000,
            height=800,
        )

        fig.write_html(output_path)
        print(f"Saved interactive 3D visualization: {output_path}")
        return True

    except ImportError:
        print("  Plotly not installed. Skipping interactive 3D visualization.")
        print("  Install with: pip install plotly")
        return False


def visualize_single_label(embeddings_2d, labels, masks, label_idx, label_name,
                           output_path, compute_boundary=True):
    """
    Visualize t-SNE for a single label with positive/negative samples colored.
    Optionally draw decision boundary.
    """
    # Get valid samples for this label
    valid_mask = masks[:, label_idx] == 1
    valid_embeddings = embeddings_2d[valid_mask]
    valid_labels = labels[valid_mask, label_idx]

    if len(valid_embeddings) == 0:
        print(f"  No valid samples for {label_name}")
        return None

    # Compute clustering metrics
    metrics = {}
    if len(np.unique(valid_labels)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
        except:
            metrics['silhouette_score'] = None
        try:
            metrics['davies_bouldin_index'] = davies_bouldin_score(valid_embeddings, valid_labels)
        except:
            metrics['davies_bouldin_index'] = None
    else:
        metrics['silhouette_score'] = None
        metrics['davies_bouldin_index'] = None

    metrics['n_positive'] = int((valid_labels == 1).sum())
    metrics['n_negative'] = int((valid_labels == 0).sum())
    metrics['n_total'] = len(valid_labels)
    metrics['positive_ratio'] = metrics['n_positive'] / metrics['n_total']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot samples
    colors = ['#3498db', '#e74c3c']  # Blue for negative, Red for positive
    for class_val, color, label_text in [(0, colors[0], 'Negative'), (1, colors[1], 'Positive')]:
        mask = valid_labels == class_val
        ax.scatter(valid_embeddings[mask, 0], valid_embeddings[mask, 1],
                  c=color, label=f'{label_text} (n={mask.sum()})',
                  alpha=0.5, s=15, edgecolors='none')

    # Draw decision boundary if requested
    if compute_boundary and len(np.unique(valid_labels)) > 1:
        try:
            # Create mesh grid
            x_min, x_max = valid_embeddings[:, 0].min() - 1, valid_embeddings[:, 0].max() + 1
            y_min, y_max = valid_embeddings[:, 1].min() - 1, valid_embeddings[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))

            # Fit classifier for boundary visualization
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(valid_embeddings, valid_labels)

            # Predict on mesh
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            # Draw decision boundary contour
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
            ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=colors, alpha=0.1)

        except Exception as e:
            print(f"  Could not draw decision boundary for {label_name}: {e}")

    # Add title and labels
    title = f'{label_name}\n'
    if metrics['silhouette_score'] is not None:
        title += f'Silhouette: {metrics["silhouette_score"]:.3f}, '
    if metrics['davies_bouldin_index'] is not None:
        title += f'DB Index: {metrics["davies_bouldin_index"]:.3f}'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def visualize_all_labels_grid(embeddings_2d, labels, masks, output_path):
    """Create a grid visualization of all labels."""
    n_labels = len(mimic_cxr_jpg.chexpert_labels)
    n_cols = 4
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    colors = ['#3498db', '#e74c3c']

    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        ax = axes[idx]

        valid_mask = masks[:, idx] == 1
        valid_embeddings = embeddings_2d[valid_mask]
        valid_labels = labels[valid_mask, idx]

        if len(valid_embeddings) == 0:
            ax.set_title(f'{label_name}\n(No samples)', fontsize=10)
            ax.axis('off')
            continue

        # Plot
        for class_val, color in [(0, colors[0]), (1, colors[1])]:
            mask = valid_labels == class_val
            ax.scatter(valid_embeddings[mask, 0], valid_embeddings[mask, 1],
                      c=color, alpha=0.4, s=5, edgecolors='none')

        # Compute silhouette score
        silhouette = None
        if len(np.unique(valid_labels)) > 1:
            try:
                silhouette = silhouette_score(valid_embeddings, valid_labels)
            except:
                pass

        n_pos = (valid_labels == 1).sum()
        n_neg = (valid_labels == 0).sum()
        title = f'{label_name}\n+:{n_pos}, -:{n_neg}'
        if silhouette is not None:
            title += f', Sil:{silhouette:.2f}'
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_labels, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('t-SNE Visualization by Label (Blue: Negative, Red: Positive)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_multilabel_overlay(embeddings_2d, labels, masks, output_path):
    """
    Visualize with color encoding for multi-label patterns.
    Color intensity based on number of positive labels.
    """
    # Count positive labels per sample
    n_positive_labels = (labels * masks).sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create colormap
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=n_positive_labels, cmap='YlOrRd',
                        alpha=0.6, s=15, edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Positive Labels', fontsize=12)

    ax.set_title('t-SNE: Multi-label Distribution\n(Color = Number of Positive Labels)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_no_finding_vs_abnormal(embeddings_2d, labels, masks, output_path):
    """
    Visualize No Finding vs any abnormality.
    """
    no_finding_idx = mimic_cxr_jpg.chexpert_labels.index('No Finding')

    # Get No Finding status
    no_finding = labels[:, no_finding_idx] == 1

    # Get any abnormality (any positive label except No Finding)
    abnormal_indices = [i for i in range(len(mimic_cxr_jpg.chexpert_labels))
                       if i != no_finding_idx]
    has_abnormality = (labels[:, abnormal_indices] * masks[:, abnormal_indices]).sum(axis=1) > 0

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = {'No Finding': '#2ecc71', 'Abnormal': '#e74c3c', 'Mixed/Unknown': '#95a5a6'}

    # Categorize samples
    categories = []
    for i in range(len(labels)):
        if no_finding[i] and not has_abnormality[i]:
            categories.append('No Finding')
        elif has_abnormality[i] and not no_finding[i]:
            categories.append('Abnormal')
        else:
            categories.append('Mixed/Unknown')
    categories = np.array(categories)

    for cat, color in colors.items():
        mask = categories == cat
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=f'{cat} (n={mask.sum()})',
                  alpha=0.5, s=15, edgecolors='none')

    ax.set_title('t-SNE: No Finding vs Abnormal', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize test sample embeddings')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/sample_embeddings',
                        help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples to use (for speed)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--no_boundary', action='store_true',
                        help='Skip decision boundary computation')
    args = parser.parse_args()

    # Create output directories
    output_dir = args.output_dir
    per_label_dir = os.path.join(output_dir, 'per_label')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(per_label_dir, exist_ok=True)

    print("=" * 80)
    print("Test Sample Embedding Visualization")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    model_name = config['model']['name']
    print(f"\nModel: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max samples: {args.max_samples}")

    # Set seed
    set_seed(config['system'].get('seed', 42))

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print("\nLoading model...")
    model = get_model(config['model'])

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    print("Checkpoint loaded successfully")

    # Create feature extractor
    feature_extractor = FeatureExtractor(model, model_name)

    # Setup data loader
    print("\nLoading test data...")
    data_config = config['data']
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

    test_loader = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        collate_fn=safe_collate_fn,
    )
    print(f"Test dataset size: {len(test_ds)}")

    # Extract embeddings
    print("\nExtracting feature embeddings...")
    features, labels, masks = extract_embeddings(
        feature_extractor, test_loader, device, max_samples=args.max_samples
    )
    print(f"Extracted embeddings shape: {features.shape}")

    # Compute t-SNE 2D
    print("\nComputing t-SNE 2D projection...")
    embeddings_2d = compute_tsne(features, perplexity=args.perplexity, n_components=2)

    # Compute t-SNE 3D
    print("\nComputing t-SNE 3D projection...")
    embeddings_3d = compute_tsne(features, perplexity=args.perplexity, n_components=3)

    # Save embeddings
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'tsne_embeddings_2d.npy'), embeddings_2d)
    np.save(os.path.join(output_dir, 'tsne_embeddings_3d.npy'), embeddings_3d)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    np.save(os.path.join(output_dir, 'masks.npy'), masks)
    print(f"Saved embeddings to {output_dir}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. All labels in single plot (main visualization)
    print("  Creating all-labels single plot (2D)...")
    visualize_all_labels_single_plot(embeddings_2d, labels, masks,
                                      os.path.join(output_dir, 'all_labels_single_plot.png'))

    # 2. Grid view of all labels
    print("  Creating all-labels grid view...")
    visualize_all_labels_grid(embeddings_2d, labels, masks,
                              os.path.join(output_dir, 'all_labels_grid.png'))

    # 2. Multi-label overlay
    print("  Creating multi-label overlay...")
    visualize_multilabel_overlay(embeddings_2d, labels, masks,
                                 os.path.join(output_dir, 'multilabel_overlay.png'))

    # 3. No Finding vs Abnormal
    print("  Creating No Finding vs Abnormal view...")
    visualize_no_finding_vs_abnormal(embeddings_2d, labels, masks,
                                     os.path.join(output_dir, 'no_finding_vs_abnormal.png'))

    # 4. Per-label visualizations with decision boundaries (2D)
    print("  Creating per-label 2D visualizations...")
    all_metrics = {}
    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        print(f"    Processing {label_name}...")
        safe_name = label_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(per_label_dir, f'{safe_name}.png')

        metrics = visualize_single_label(
            embeddings_2d, labels, masks, idx, label_name, output_path,
            compute_boundary=not args.no_boundary
        )
        if metrics:
            all_metrics[label_name] = metrics

    # 5. 3D Visualizations
    print("\nGenerating 3D visualizations...")
    tsne_3d_dir = os.path.join(output_dir, '3d')
    os.makedirs(tsne_3d_dir, exist_ok=True)

    # 3D all labels in single plot (main 3D visualization)
    print("  Creating 3D all-labels single plot...")
    visualize_all_labels_single_plot_3d(embeddings_3d, labels, masks,
                                         os.path.join(tsne_3d_dir, 'all_labels_single_plot_3d.png'))

    # 3D grid view
    print("  Creating 3D all-labels grid view...")
    visualize_3d_all_labels_grid(embeddings_3d, labels, masks,
                                  os.path.join(tsne_3d_dir, 'all_labels_grid_3d.png'))

    # 3D multi-label overlay
    print("  Creating 3D multi-label overlay...")
    visualize_3d_multilabel_overlay(embeddings_3d, labels, masks,
                                     os.path.join(tsne_3d_dir, 'multilabel_overlay_3d.png'))

    # 3D No Finding vs Abnormal
    print("  Creating 3D No Finding vs Abnormal view...")
    visualize_3d_no_finding_vs_abnormal(embeddings_3d, labels, masks,
                                         os.path.join(tsne_3d_dir, 'no_finding_vs_abnormal_3d.png'))

    # 3D per-label visualizations
    print("  Creating 3D per-label visualizations...")
    per_label_3d_dir = os.path.join(tsne_3d_dir, 'per_label')
    os.makedirs(per_label_3d_dir, exist_ok=True)
    for idx, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        safe_name = label_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(per_label_3d_dir, f'{safe_name}_3d.png')
        visualize_3d_single_label(embeddings_3d, labels, masks, idx, label_name, output_path)

    # Interactive 3D HTML (if plotly available)
    print("  Creating interactive 3D visualizations (HTML)...")
    create_interactive_3d_html(embeddings_3d, labels, masks,
                               os.path.join(tsne_3d_dir, 'interactive_3d_normal_abnormal.html'))
    create_interactive_all_labels_3d_html(embeddings_3d, labels, masks,
                                           os.path.join(tsne_3d_dir, 'interactive_3d_all_labels.html'))

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'label'
    metrics_df.to_csv(os.path.join(output_dir, 'clustering_metrics.csv'))

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Save summary JSON
    summary = {
        'model': model_name,
        'checkpoint': args.checkpoint,
        'n_samples': len(features),
        'feature_dim': int(features.shape[1]),
        'perplexity': args.perplexity,
        'metrics': convert_to_native(all_metrics),
    }
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Clustering Metrics Summary")
    print("=" * 80)
    print(f"{'Label':<30} {'Silhouette':>12} {'DB Index':>12} {'Pos Ratio':>12}")
    print("-" * 70)
    for label_name, metrics in all_metrics.items():
        sil = f"{metrics['silhouette_score']:.3f}" if metrics['silhouette_score'] else "N/A"
        db = f"{metrics['davies_bouldin_index']:.3f}" if metrics['davies_bouldin_index'] else "N/A"
        pos_ratio = f"{metrics['positive_ratio']:.3f}"
        print(f"{label_name:<30} {sil:>12} {db:>12} {pos_ratio:>12}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    print("\nOutput files (2D):")
    print("  - all_labels_single_plot.png: ALL labels in one plot with different colors")
    print("  - all_labels_grid.png: Grid view of all labels")
    print("  - multilabel_overlay.png: Color by number of positive labels")
    print("  - no_finding_vs_abnormal.png: Normal vs abnormal samples")
    print("  - per_label/*.png: Individual label visualizations with decision boundaries")
    print("\nOutput files (3D):")
    print("  - 3d/all_labels_single_plot_3d.png: ALL labels in one 3D plot")
    print("  - 3d/all_labels_grid_3d.png: 3D grid view of all labels")
    print("  - 3d/multilabel_overlay_3d.png: 3D multi-label visualization")
    print("  - 3d/no_finding_vs_abnormal_3d.png: 3D normal vs abnormal")
    print("  - 3d/per_label/*.png: 3D individual label visualizations")
    print("  - 3d/interactive_3d_all_labels.html: Interactive 3D with all labels")
    print("  - 3d/interactive_3d_normal_abnormal.html: Interactive 3D normal vs abnormal")
    print("\nData files:")
    print("  - tsne_embeddings_2d.npy: 2D t-SNE coordinates")
    print("  - tsne_embeddings_3d.npy: 3D t-SNE coordinates")
    print("  - clustering_metrics.csv: Clustering metrics per label")
    print("  - analysis_summary.json: Full analysis summary")


if __name__ == '__main__':
    main()
