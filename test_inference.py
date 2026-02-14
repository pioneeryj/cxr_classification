"""Inference-only script for CXR classification models.

Loads a trained checkpoint, runs inference on samples specified by a CSV
(e.g. oversampled_dicom_ids.csv), and saves per-sample classification
results (14 CheXpert labels + probabilities) as JSON files.

No evaluation metrics are computed — this is purely for generating
classification outputs to feed into the report generation pipeline.

Usage:
    python test_inference.py \
        --config configs/resnet50_config.yaml \
        --checkpoint /path/to/checkpoint.pt \
        --sample-csv /path/to/oversampled_dicom_ids.csv \
        --output-dir /path/to/output/
"""

import json
import os
import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mimic_cxr_jpg
from peft import set_peft_model_state_dict
from model_factory import get_model, apply_lora
from data_transforms import get_val_transform
from utils import load_config, set_seed
from fit_calibration import TemperatureScaling


def safe_collate_fn(batch):
    """Custom collate function that creates independent tensor copies."""
    images, labels, masks = zip(*batch)
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])
    return images_batch, labels_batch, masks_batch


class Inferencer:
    """Inference-only runner for CXR classification."""

    def __init__(self, config, checkpoint_path, sample_csv,
                 output_dir, temp_scaling=False):
        """
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            sample_csv: Path to CSV with dicom_id, subject_id, study_id, path
            output_dir: Directory to save per-sample JSON results
            temp_scaling: If True, fit calibrators on validation set
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.sample_csv = sample_csv
        self.output_dir = output_dir
        self.temp_scaling = temp_scaling

        # Setup device
        device_str = config['system']['device']
        if 'cuda' in device_str and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU.")
            device_str = 'cpu'
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Set random seed
        set_seed(config['system']['seed'])

        # Initialize model and load checkpoint
        self.model = self._setup_model()
        self._load_checkpoint()

        # Load sample CSV and build dataset directly (bypass split filtering)
        self.sample_df = pd.read_csv(sample_csv)
        print(f"Loaded {len(self.sample_df)} samples from {sample_csv}")

        self.dataset = self._build_dataset_from_csv(self.sample_df)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            collate_fn=safe_collate_fn,
        )
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Batches: {len(self.data_loader)}")

        # Temperature scaling
        if self.temp_scaling:
            print("\n" + "=" * 80)
            print("Temperature Scaling Enabled - Fitting Calibrators")
            print("=" * 80)
            self.val_loader = self._setup_split_data('val')
            self.calibrator_global, self.calibrator_label_wise = self._fit_calibrators()
        else:
            self.calibrator_global = None
            self.calibrator_label_wise = None

    def _setup_model(self):
        """Setup model based on configuration."""
        model = get_model(self.config['model'])
        model = apply_lora(model, self.config['model'])
        model = model.to(self.device)
        model.eval()
        print(f"Model: {self.config['model']['name']}")
        return model

    def _load_checkpoint(self):
        """Load model checkpoint."""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            self.epoch = checkpoint.get('epoch', None)
        else:
            state_dict = checkpoint
            self.epoch = None

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        lora_config = self.config['model'].get('lora', {})
        if lora_config.get('enabled', False):
            is_full_peft = any(k.startswith('base_model.model.') for k in new_state_dict.keys())
            if is_full_peft:
                self.model.load_state_dict(new_state_dict, strict=False)
            else:
                set_peft_model_state_dict(self.model, new_state_dict)
            print(f"LoRA checkpoint loaded (Epoch: {self.epoch})")
            print("Merging LoRA adapters into base model...")
            self.model = self.model.merge_and_unload()
            self.model.eval()
            print("LoRA merge complete.")
        else:
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(new_state_dict.keys())

            needs_remapping = False
            if 'encoder.encoder.conv1.weight' in ckpt_keys and 'encoder.encoder.encoder.conv1.weight' in model_keys:
                needs_remapping = True
                print("Detected checkpoint key mismatch - applying key remapping for BioViL model")

            if needs_remapping:
                remapped_state_dict = {}
                for k, v in new_state_dict.items():
                    new_key = k
                    if k.startswith('encoder.encoder.') and not k.startswith('encoder.encoder.encoder.'):
                        new_key = 'encoder.encoder.encoder.' + k[len('encoder.encoder.'):]
                    elif k == 'encoder.missing_previous_emb':
                        new_key = 'encoder.encoder.missing_previous_emb'
                    elif k.startswith('encoder.backbone_to_vit.'):
                        new_key = 'encoder.encoder.backbone_to_vit.' + k[len('encoder.backbone_to_vit.'):]
                    elif k.startswith('encoder.vit_pooler.'):
                        new_key = 'encoder.encoder.vit_pooler.' + k[len('encoder.vit_pooler.'):]
                    elif k.startswith('projector.'):
                        new_key = 'encoder.projector.' + k[len('projector.'):]
                    remapped_state_dict[new_key] = v
                new_state_dict = remapped_state_dict

            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Checkpoint loaded (Epoch: {self.epoch})")

    def _build_dataset_from_csv(self, sample_df):
        """Build a MIMICCXRJPGDataset directly from a sample CSV.

        This bypasses the train/val/test split logic entirely.
        The CSV must have columns: dicom_id, subject_id, study_id, path.
        We load the full metadata and filter to the requested dicom_ids.
        """
        data_config = self.config['data']
        model_name = self.config['model']['name']
        transform = get_val_transform(model_name)

        # Load full metadata (all splits, no subject_prefix_filter)
        full_meta = mimic_cxr_jpg.load_all_metadata(
            data_dir=data_config['datadir']
        )
        print(f"Full metadata loaded: {len(full_meta)} rows")

        # Filter to requested dicom_ids
        requested_ids = set(sample_df['dicom_id'].astype(str))
        full_meta['dicom_id'] = full_meta['dicom_id'].astype(str)
        filtered = full_meta[full_meta['dicom_id'].isin(requested_ids)].copy()

        if len(filtered) == 0:
            raise ValueError(
                f"No matching dicom_ids found! "
                f"CSV has {len(requested_ids)} IDs, metadata has {len(full_meta)} rows. "
                f"Check that the CSV dicom_ids exist in the metadata."
            )

        if len(filtered) < len(requested_ids):
            found_ids = set(filtered['dicom_id'])
            missing = requested_ids - found_ids
            print(f"WARNING: {len(missing)} dicom_ids not found in metadata "
                  f"(requested {len(requested_ids)}, found {len(filtered)})")
            if len(missing) <= 10:
                for mid in sorted(missing):
                    print(f"  Missing: {mid}")

        filtered = filtered.reset_index(drop=True)
        print(f"Filtered dataset: {len(filtered)} samples")

        dataset = mimic_cxr_jpg.MIMICCXRJPGDataset(
            dataframe=filtered,
            datadir=data_config['datadir'],
            image_subdir=data_config['image_subdir'],
            transform=transform,
            label_method=data_config['label_method'],
        )
        return dataset

    def _setup_split_data(self, split):
        """Setup data loader for a specific split (used for val set calibration)."""
        data_config = self.config['data']
        model_name = self.config['model']['name']
        transform = get_val_transform(model_name)

        if data_config['split_type'] == 'official':
            train_ds, val_ds, test_ds = mimic_cxr_jpg.official_split(
                datadir=data_config['datadir'],
                dicom_id_file=data_config['dicom_id_file'],
                image_subdir=data_config['image_subdir'],
                train_transform=transform,
                test_transform=transform,
                label_method=data_config['label_method'],
                subject_prefix_filter=data_config.get('subject_prefix_filter', None),
            )
        elif data_config['split_type'] == 'cv':
            train_ds, val_ds, test_ds = mimic_cxr_jpg.cv(
                num_folds=data_config['num_folds'],
                fold=data_config['fold'],
                datadir=data_config['datadir'],
                dicom_id_file=data_config['dicom_id_file'],
                image_subdir=data_config['image_subdir'],
                val_size=data_config['val_size'],
                random_state=data_config['random_state'],
                stratify=data_config['stratify'],
                train_transform=transform,
                test_transform=transform,
                label_method=data_config['label_method'],
                subject_prefix_filter=data_config.get('subject_prefix_filter', None),
            )
        else:
            raise ValueError(f"Unknown split_type: {data_config['split_type']}")

        if split == 'val':
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            dataset = train_ds

        data_loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=safe_collate_fn,
        )
        print(f"{split.capitalize()} dataset size: {len(dataset)}")
        return data_loader

    def _fit_calibrators(self):
        """Fit global and label-wise temperature scaling calibrators on validation set."""
        print("\nCollecting logits from validation set...")
        all_logits, all_labels, all_masks = [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val logits"):
                X, Y, Ymask = batch
                X = X.to(self.device).float()
                Y = Y.to(self.device).float()
                Ymask = Ymask.to(self.device)
                logits = self.model(X)
                all_logits.append(logits.cpu())
                all_labels.append(Y.cpu())
                all_masks.append(Ymask.cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        masks = torch.cat(all_masks, dim=0)
        print(f"Collected {logits.shape[0]} validation samples")

        num_classes = logits.shape[1]

        # Global
        print("\nFitting global temperature scaling...")
        calibrator_global = TemperatureScaling(mode='global', num_classes=num_classes)
        temp_global = calibrator_global.fit(logits, labels, masks)
        print(f"Optimal global temperature: {temp_global:.4f}")
        calibrator_global = calibrator_global.to(self.device)

        # Label-wise
        print("\nFitting label-wise temperature scaling...")
        calibrator_label_wise = TemperatureScaling(mode='label_wise', num_classes=num_classes)
        temp_label_wise = calibrator_label_wise.fit(logits, labels, masks)
        print("Optimal temperatures per label:")
        for i, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
            print(f"  {label_name}: {temp_label_wise[i]:.4f}")
        calibrator_label_wise = calibrator_label_wise.to(self.device)

        # Save calibrators
        os.makedirs(self.output_dir, exist_ok=True)
        for name, cal in [('calibrator_global.pkl', calibrator_global),
                          ('calibrator_label_wise.pkl', calibrator_label_wise)]:
            path = os.path.join(self.output_dir, name)
            with open(path, 'wb') as f:
                pickle.dump(cal, f)
            print(f"Saved: {path}")

        return calibrator_global, calibrator_label_wise

    def run(self):
        """Run inference and save all results in a single JSON file.

        Output JSON structure:
        {
            "metadata": { model, checkpoint, calibration_type, num_samples },
            "results": {
                "<dicom_id>": {
                    "subject_id": ...,
                    "study_id": ...,
                    "image_path": ...,
                    "classification": { label: 0/1, ... },
                    "probability": { label: float, ... }
                },
                ...
            }
        }

        The output filename matches the checkpoint filename (e.g. resnet50_xxx.json).
        """
        print("\n" + "=" * 80)
        print("Running Inference")
        print("=" * 80)

        os.makedirs(self.output_dir, exist_ok=True)

        self.model.eval()
        df = self.dataset.dataframe
        label_names = mimic_cxr_jpg.chexpert_labels

        # Determine calibration type
        if self.temp_scaling:
            calibration_type = "label_wise_t"
            print("Using label-wise temperature scaling for probabilities")
        else:
            calibration_type = "uncalibrated"
            print("Using uncalibrated probabilities")

        all_results = {}
        sample_idx = 0

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Inference"):
                X, Y, Ymask = batch
                batch_size = X.shape[0]
                X = X.to(self.device).float()

                # Forward pass
                logits = self.model(X)

                # Apply calibration if available
                if self.temp_scaling:
                    logits_cal = self.calibrator_label_wise(logits)
                    probs = torch.sigmoid(logits_cal).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()

                # Binary predictions (threshold 0.5)
                preds = (probs > 0.5).astype(int)

                for i in range(batch_size):
                    if sample_idx >= len(df):
                        break

                    row = df.iloc[sample_idx]
                    dicom_id = str(row['dicom_id'])

                    all_results[dicom_id] = {
                        "subject_id": int(row['subject_id']),
                        "study_id": int(row['study_id']),
                        "image_path": row['path'],
                        "classification": {
                            label_names[j]: int(preds[i, j])
                            for j in range(len(label_names))
                        },
                        "probability": {
                            label_names[j]: round(float(probs[i, j]), 4)
                            for j in range(len(label_names))
                        },
                    }
                    sample_idx += 1

        # Build output JSON
        output_data = {
            "metadata": {
                "model": self.config['model']['name'],
                "checkpoint": os.path.basename(self.checkpoint_path),
                "calibration_type": calibration_type,
                "num_samples": len(all_results),
            },
            "results": all_results,
        }

        # Filename from checkpoint (e.g. resnet50_calib_distill_latest_kl1.0_9ep.json)
        ckpt_stem = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
        output_path = os.path.join(self.output_dir, f"{ckpt_stem}.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, indent=2, ensure_ascii=False, fp=f)

        print(f"\nInference complete!")
        print(f"  Samples: {len(all_results)}")
        print(f"  Calibration: {calibration_type}")
        print(f"  Saved to: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='CXR classification inference: save per-sample results as JSON'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--sample-csv', type=str, required=True,
        help='Path to CSV with columns: dicom_id, subject_id, study_id, path'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for inference results'
    )
    parser.add_argument(
        '--temp_scaling', type=str, default='false',
        choices=['true', 'false'],
        help='Enable temperature scaling (default: false)'
    )
    parser.add_argument(
        '--override', type=str, nargs='+',
        help='Override config values (e.g., output.dir=./outputs/my_test)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        for override in args.override:
            key_path, value = override.split('=', 1)
            keys = key_path.split('.')
            d = config
            for key in keys[:-1]:
                d = d[key]
            try:
                d[keys[-1]] = eval(value)
            except:
                d[keys[-1]] = value

    temp_scaling = args.temp_scaling.lower() == 'true'

    inferencer = Inferencer(
        config=config,
        checkpoint_path=args.checkpoint,
        sample_csv=args.sample_csv,
        output_dir=args.output_dir,
        temp_scaling=temp_scaling,
    )
    inferencer.run()

    print("\nDone! JSON files ready for report generation.")


if __name__ == '__main__':
    main()
