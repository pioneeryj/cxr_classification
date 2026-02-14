"""
Label Embedding Visualization and Clustering Analysis

Extracts classifier weights from a trained model and analyzes the label embeddings:
1. Visualizes label embeddings using t-SNE and PCA
2. Computes clustering metrics (silhouette score, Davies-Bouldin index)
3. Analyzes pairwise distances and correlations between label embeddings

Usage:
    python visualize_label_embeddings.py --config configs/resnet50_config.yaml \
        --checkpoint /path/to/checkpoint.pt --output_dir ./outputs/label_embeddings
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import json

from model_factory import get_model
from utils import load_config
import mimic_cxr_jpg


def extract_classifier_weights(model, model_name: str) -> np.ndarray:
    """
    Extract classifier layer weights from the model.

    The classifier weights can be interpreted as label embeddings in the feature space.
    Each row corresponds to a label's "template" in the feature space.

    Args:
        model: PyTorch model
        model_name: Name of the model architecture

    Returns:
        weights: numpy array of shape [num_classes, feature_dim]
    """
    model_name = model_name.lower()

    if 'resnet' in model_name:
        # ResNet: model.fc is the classifier
        weights = model.fc.weight.data.cpu().numpy()
    elif 'densenet' in model_name:
        # DenseNet: model.classifier is the classifier
        weights = model.classifier.weight.data.cpu().numpy()
    elif model_name == 'biovil':
        # BioViL: model.classifier is the classifier
        weights = model.classifier.weight.data.cpu().numpy()
    elif model_name == 'medklip':
        # MedKLIP: model.classifier is the classifier
        weights = model.classifier.weight.data.cpu().numpy()
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return weights


def compute_pairwise_distances(embeddings: np.ndarray) -> pd.DataFrame:
    """Compute pairwise distances between label embeddings."""
    distances = squareform(pdist(embeddings, metric='euclidean'))
    return pd.DataFrame(
        distances,
        index=mimic_cxr_jpg.chexpert_labels,
        columns=mimic_cxr_jpg.chexpert_labels
    )


def compute_pairwise_cosine_similarity(embeddings: np.ndarray) -> pd.DataFrame:
    """Compute pairwise cosine similarity between label embeddings."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute cosine similarity
    similarity = normalized @ normalized.T

    return pd.DataFrame(
        similarity,
        index=mimic_cxr_jpg.chexpert_labels,
        columns=mimic_cxr_jpg.chexpert_labels
    )


def visualize_tsne(embeddings: np.ndarray, labels: list, output_path: str,
                   perplexity: int = 5, random_state: int = 42):
    """Visualize embeddings using t-SNE."""
    # t-SNE requires perplexity < n_samples
    perplexity = min(perplexity, len(labels) - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                n_iter=2000, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    # Create scatter plot
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         s=200, c=range(len(labels)), cmap='tab20', alpha=0.8)

    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=10, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')

    plt.title('Label Embeddings (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved t-SNE visualization: {output_path}")
    return embeddings_2d


def visualize_pca(embeddings: np.ndarray, labels: list, output_path: str):
    """Visualize embeddings using PCA."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    # Create scatter plot
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         s=200, c=range(len(labels)), cmap='tab20', alpha=0.8)

    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=10, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')

    # Add explained variance info
    explained_var = pca.explained_variance_ratio_
    plt.title(f'Label Embeddings (PCA)\nPC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved PCA visualization: {output_path}")
    return embeddings_2d, pca


def plot_distance_heatmap(distance_matrix: pd.DataFrame, output_path: str, title: str):
    """Plot heatmap of pairwise distances."""
    plt.figure(figsize=(14, 12))

    sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap: {output_path}")


def plot_dendrogram(embeddings: np.ndarray, labels: list, output_path: str):
    """Plot hierarchical clustering dendrogram."""
    plt.figure(figsize=(14, 8))

    # Compute linkage
    linkage_matrix = linkage(embeddings, method='ward')

    # Plot dendrogram
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=45, leaf_font_size=10)

    plt.title('Hierarchical Clustering of Label Embeddings', fontsize=14, fontweight='bold')
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved dendrogram: {output_path}")
    return linkage_matrix


def compute_clustering_metrics(embeddings: np.ndarray, labels: list) -> dict:
    """
    Compute clustering metrics for label embeddings.

    Returns metrics for different numbers of clusters.
    """
    n_samples = len(labels)
    metrics = {}

    # Try different numbers of clusters
    for n_clusters in range(2, min(n_samples, 8)):
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Silhouette score (higher is better, range [-1, 1])
        silhouette = silhouette_score(embeddings, cluster_labels)

        # Davies-Bouldin index (lower is better)
        db_index = davies_bouldin_score(embeddings, cluster_labels)

        metrics[n_clusters] = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(db_index),
            'cluster_assignments': {labels[i]: int(cluster_labels[i]) for i in range(n_samples)}
        }

    return metrics


def compute_label_wise_metrics(embeddings: np.ndarray, labels: list,
                                distance_matrix: pd.DataFrame,
                                cosine_sim_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-label clustering/separation metrics.

    For each label, compute:
    - Mean distance to other labels
    - Min distance to other labels (nearest neighbor)
    - Max distance to other labels (farthest neighbor)
    - Mean cosine similarity to other labels
    - Embedding norm
    """
    metrics = []

    for i, label in enumerate(labels):
        # Get distances to other labels (excluding self)
        distances = distance_matrix.loc[label].drop(label).values
        cosine_sims = cosine_sim_matrix.loc[label].drop(label).values

        # Nearest and farthest neighbors
        nearest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)
        other_labels = [l for l in labels if l != label]

        metrics.append({
            'label': label,
            'embedding_norm': float(np.linalg.norm(embeddings[i])),
            'mean_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'std_distance': float(np.std(distances)),
            'nearest_neighbor': other_labels[nearest_idx],
            'nearest_distance': float(distances[nearest_idx]),
            'farthest_neighbor': other_labels[farthest_idx],
            'farthest_distance': float(distances[farthest_idx]),
            'mean_cosine_similarity': float(np.mean(cosine_sims)),
            'max_cosine_similarity': float(np.max(cosine_sims)),
            'min_cosine_similarity': float(np.min(cosine_sims)),
        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description='Visualize label embeddings from classifier weights')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/label_embeddings',
                        help='Output directory for visualizations and metrics')
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Label Embedding Visualization and Clustering Analysis")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    model_name = config['model']['name']
    print(f"\nModel: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")

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
    model.eval()
    print("Checkpoint loaded successfully")

    # Extract classifier weights
    print("\nExtracting classifier weights...")
    embeddings = extract_classifier_weights(model, model_name)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"  - Number of labels: {embeddings.shape[0]}")
    print(f"  - Feature dimension: {embeddings.shape[1]}")

    labels = mimic_cxr_jpg.chexpert_labels

    # Compute pairwise metrics
    print("\nComputing pairwise distances and similarities...")
    distance_matrix = compute_pairwise_distances(embeddings)
    cosine_sim_matrix = compute_pairwise_cosine_similarity(embeddings)

    # Save distance and similarity matrices
    distance_matrix.to_csv(os.path.join(output_dir, 'pairwise_euclidean_distance.csv'))
    cosine_sim_matrix.to_csv(os.path.join(output_dir, 'pairwise_cosine_similarity.csv'))
    print(f"Saved pairwise matrices to {output_dir}")

    # Visualizations
    print("\nGenerating visualizations...")

    # t-SNE
    tsne_2d = visualize_tsne(embeddings, labels,
                             os.path.join(output_dir, 'tsne_visualization.png'))

    # PCA
    pca_2d, pca_model = visualize_pca(embeddings, labels,
                                       os.path.join(output_dir, 'pca_visualization.png'))

    # Distance heatmap
    plot_distance_heatmap(distance_matrix,
                         os.path.join(output_dir, 'euclidean_distance_heatmap.png'),
                         'Pairwise Euclidean Distance (Label Embeddings)')

    # Cosine similarity heatmap
    plot_distance_heatmap(cosine_sim_matrix,
                         os.path.join(output_dir, 'cosine_similarity_heatmap.png'),
                         'Pairwise Cosine Similarity (Label Embeddings)')

    # Dendrogram
    linkage_matrix = plot_dendrogram(embeddings, labels,
                                     os.path.join(output_dir, 'hierarchical_clustering.png'))

    # Compute clustering metrics
    print("\nComputing clustering metrics...")
    clustering_metrics = compute_clustering_metrics(embeddings, labels)

    # Print clustering metrics
    print("\n" + "=" * 80)
    print("Clustering Metrics (K-Means)")
    print("=" * 80)
    print(f"{'n_clusters':<12} {'Silhouette':>12} {'Davies-Bouldin':>15}")
    print("-" * 40)
    for n_clusters, metrics in clustering_metrics.items():
        print(f"{n_clusters:<12} {metrics['silhouette_score']:>12.4f} {metrics['davies_bouldin_index']:>15.4f}")

    # Compute label-wise metrics
    print("\nComputing label-wise metrics...")
    label_metrics = compute_label_wise_metrics(embeddings, labels, distance_matrix, cosine_sim_matrix)
    label_metrics.to_csv(os.path.join(output_dir, 'label_wise_metrics.csv'), index=False)

    # Print label-wise metrics
    print("\n" + "=" * 80)
    print("Label-wise Embedding Metrics")
    print("=" * 80)
    print(f"{'Label':<30} {'Norm':>8} {'MeanDist':>10} {'NearestNeighbor':<25} {'Dist':>8}")
    print("-" * 90)
    for _, row in label_metrics.iterrows():
        print(f"{row['label']:<30} {row['embedding_norm']:>8.2f} {row['mean_distance']:>10.2f} "
              f"{row['nearest_neighbor']:<25} {row['nearest_distance']:>8.2f}")

    # Save all results to JSON
    results = {
        'model': model_name,
        'checkpoint': args.checkpoint,
        'embedding_shape': list(embeddings.shape),
        'labels': labels,
        'clustering_metrics': clustering_metrics,
        'label_wise_metrics': label_metrics.to_dict(orient='records'),
        'pca_explained_variance': pca_model.explained_variance_ratio_.tolist(),
    }

    with open(os.path.join(output_dir, 'embedding_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save embeddings as numpy array for later use
    np.save(os.path.join(output_dir, 'label_embeddings.npy'), embeddings)

    # Save 2D projections
    tsne_df = pd.DataFrame(tsne_2d, columns=['tsne_1', 'tsne_2'], index=labels)
    tsne_df.to_csv(os.path.join(output_dir, 'tsne_coordinates.csv'))

    pca_df = pd.DataFrame(pca_2d, columns=['pc1', 'pc2'], index=labels)
    pca_df.to_csv(os.path.join(output_dir, 'pca_coordinates.csv'))

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)
    print("\nOutput files:")
    print("  - tsne_visualization.png: t-SNE plot of label embeddings")
    print("  - pca_visualization.png: PCA plot of label embeddings")
    print("  - euclidean_distance_heatmap.png: Pairwise distance heatmap")
    print("  - cosine_similarity_heatmap.png: Pairwise similarity heatmap")
    print("  - hierarchical_clustering.png: Dendrogram")
    print("  - pairwise_euclidean_distance.csv: Distance matrix")
    print("  - pairwise_cosine_similarity.csv: Similarity matrix")
    print("  - label_wise_metrics.csv: Per-label metrics")
    print("  - embedding_analysis.json: All metrics in JSON format")
    print("  - label_embeddings.npy: Raw embeddings for further analysis")


if __name__ == '__main__':
    main()
