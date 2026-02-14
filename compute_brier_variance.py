"""Compute per-sample Brier score variance for each label on validation set."""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mimic_cxr_jpg
from model_factory import get_model, apply_lora
from data_transforms import get_val_transform
from utils import load_config, set_seed


def safe_collate_fn(batch):
    images, labels, masks = zip(*batch)
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])
    return images_batch, labels_batch, masks_batch


def compute_brier_variance(config, checkpoint_path, output_dir):
    """Compute per-sample Brier score and its variance for each label."""

    # Setup device
    device_str = config['system']['device']
    if 'cuda' in device_str and not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    set_seed(config['system']['seed'])

    # Setup model
    model = get_model(config['model'])
    model = model.to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Detect LoRA weights in checkpoint
    has_lora = any('lora_' in k for k in new_state_dict.keys())
    if has_lora:
        print("Detected LoRA weights in checkpoint. Applying LoRA adapter...")
        model = apply_lora(model, config['model'])
        model.load_state_dict(new_state_dict, strict=False)
        print("Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        model.eval()
        print("LoRA merge complete.")
    else:
        model.load_state_dict(new_state_dict)

    # Setup validation data
    data_config = config['data']
    model_name = config['model']['name']
    val_transform = get_val_transform(model_name)

    if data_config['split_type'] == 'official':
        _, val_ds, _ = mimic_cxr_jpg.official_split(
            datadir=data_config['datadir'],
            dicom_id_file=data_config['dicom_id_file'],
            image_subdir=data_config['image_subdir'],
            train_transform=val_transform,
            test_transform=val_transform,
            label_method=data_config['label_method'],
            subject_prefix_filter=data_config.get('subject_prefix_filter', None),
        )
    elif data_config['split_type'] == 'cv':
        _, val_ds, _ = mimic_cxr_jpg.cv(
            num_folds=data_config['num_folds'],
            fold=data_config['fold'],
            datadir=data_config['datadir'],
            dicom_id_file=data_config['dicom_id_file'],
            image_subdir=data_config['image_subdir'],
            val_size=data_config['val_size'],
            random_state=data_config['random_state'],
            stratify=data_config['stratify'],
            train_transform=val_transform,
            test_transform=val_transform,
            label_method=data_config['label_method'],
            subject_prefix_filter=data_config.get('subject_prefix_filter', None),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=safe_collate_fn,
    )
    print(f"Validation set size: {len(val_ds)}")

    # Collect per-sample Brier scores
    num_classes = len(mimic_cxr_jpg.chexpert_labels)
    all_brier_scores = [[] for _ in range(num_classes)]  # List per label

    print("\nComputing per-sample Brier scores...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing"):
            X, Y, Ymask = batch
            X = X.to(device).float()
            Y = Y.to(device).float()
            Ymask = Ymask.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            # Brier score per sample per label: (p - y)^2
            brier = (probs - Y) ** 2

            # Collect per label
            for i in range(num_classes):
                mask = Ymask[:, i] == 1
                if mask.sum() > 0:
                    all_brier_scores[i].extend(brier[mask, i].cpu().numpy().tolist())

    # Compute statistics for each label
    results = []
    print("\n" + "=" * 80)
    print("Brier Score Statistics per Label (Validation Set)")
    print("=" * 80)
    print(f"{'Label':<30} {'Mean':>10} {'Std':>10} {'Var':>10} {'N':>10}")
    print("-" * 80)

    for i, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
        scores = np.array(all_brier_scores[i])
        mean_brier = np.mean(scores)
        std_brier = np.std(scores)
        var_brier = np.var(scores)
        n_samples = len(scores)

        print(f"{label_name:<30} {mean_brier:>10.6f} {std_brier:>10.6f} {var_brier:>10.6f} {n_samples:>10}")

        results.append({
            'label_index': i,
            'label_name': label_name,
            'brier_mean': mean_brier,
            'brier_std': std_brier,
            'brier_var': var_brier,
            'n_samples': n_samples,
        })

    # Overall statistics
    all_scores_flat = np.concatenate([np.array(s) for s in all_brier_scores])
    print("-" * 80)
    print(f"{'Overall':<30} {np.mean(all_scores_flat):>10.6f} {np.std(all_scores_flat):>10.6f} {np.var(all_scores_flat):>10.6f} {len(all_scores_flat):>10}")
    print("=" * 80)

    results.append({
        'label_index': -1,
        'label_name': 'Overall',
        'brier_mean': np.mean(all_scores_flat),
        'brier_std': np.std(all_scores_flat),
        'brier_var': np.var(all_scores_flat),
        'n_samples': len(all_scores_flat),
    })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'brier_variance_per_label.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-sample Brier score variance for each label'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config['output']['dir']

    compute_brier_variance(config, args.checkpoint, output_dir)


if __name__ == '__main__':
    main()
