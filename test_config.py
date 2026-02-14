"""Configuration-based testing script for CXR classification models."""

import os
import argparse
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import pandas as pd
import wandb

import mimic_cxr_jpg
from peft import set_peft_model_state_dict
from model_factory import get_model, apply_lora
from data_transforms import get_val_transform
from utils import load_config, set_seed
from calibration_metrics import CalibrationEvaluator
from fit_calibration import TemperatureScaling


def safe_collate_fn(batch):
    """Custom collate function that creates independent tensor copies.

    This avoids storage sharing issues in multi-processing DataLoader.
    """
    # Separate the batch into components
    images, labels, masks = zip(*batch)

    # Stack and clone to create independent storage
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])

    return images_batch, labels_batch, masks_batch


class Tester:
    """Tester class for CXR classification."""

    def __init__(self, config, checkpoint_path, temp_scaling=False):
        """Initialize tester with configuration.

        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            temp_scaling: If True, fit both global and label_wise calibrators on validation set
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.temp_scaling = temp_scaling
        self.epoch = None  # Will be extracted from checkpoint

        # Setup device
        device_str = config['system']['device']
        if 'cuda' in device_str and not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but not available. Using CPU.")
            device_str = 'cpu'

        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Set random seed
        set_seed(config['system']['seed'])

        # Initialize wandb
        self._setup_wandb()

        # Initialize model
        self.model = self._setup_model()

        # Load checkpoint
        self._load_checkpoint()

        # Setup data
        self.test_loader = self._setup_data('test')

        # If temperature scaling is enabled, fit calibrators on validation set
        if self.temp_scaling:
            print("\n" + "="*80)
            print("Temperature Scaling Enabled - Fitting Calibrators")
            print("="*80)
            self.val_loader = self._setup_data('val')
            self.calibrator_global, self.calibrator_label_wise = self._fit_calibrators()
        else:
            self.calibrator_global = None
            self.calibrator_label_wise = None

    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        # Extract experiment name from output directory
        output_dir = self.config['output']['dir']
        exp_name = os.path.basename(output_dir)

        # Get checkpoint name
        ckpt_name = os.path.basename(self.checkpoint_path)

        tags = ['test', self.config['model']['name']]
        test_name = f"{exp_name}_test"

        if self.temp_scaling:
            tags.append('temperature_scaling')
            test_name = f"{exp_name}_temp_scaling_test"

        wandb.init(
            project="cxr-classification",
            name=test_name,
            config={
                'model': self.config['model']['name'],
                'checkpoint': ckpt_name,
                'temp_scaling': self.temp_scaling,
                'batch_size': self.config['training']['batch_size'],
                'split_type': self.config['data']['split_type'],
                'fold': self.config['data'].get('fold', None),
                'label_method': self.config['data']['label_method'],
            },
            tags=tags,
            reinit=True
        )
        print(f"Wandb initialized for testing: {test_name}")

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

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            # Extract epoch information if available
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
            # Detect checkpoint format:
            # - Full PEFT state_dict: keys start with 'base_model.model.'
            # - Adapter-only (get_peft_model_state_dict): no such prefix
            is_full_peft = any(k.startswith('base_model.model.') for k in new_state_dict.keys())
            if is_full_peft:
                self.model.load_state_dict(new_state_dict, strict=False)
            else:
                set_peft_model_state_dict(self.model, new_state_dict)
            print(f"LoRA checkpoint loaded (Epoch: {self.epoch if self.epoch is not None else 'Unknown'})")
            print("Merging LoRA adapters into base model...")
            self.model = self.model.merge_and_unload()
            self.model.eval()
            print("LoRA merge complete. Model is now a standard nn.Module.")
        else:
            # Full model checkpoint
            # Check if key remapping is needed for BioViL model
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
            print(f"Checkpoint loaded successfully (Epoch: {self.epoch if self.epoch is not None else 'Unknown'})")

        # Log epoch to wandb
        if self.epoch is not None:
            wandb.config.update({'checkpoint_epoch': self.epoch})

    def _fit_calibrators(self):
        """Fit both global and label-wise temperature scaling calibrators on validation set."""
        # Collect logits from validation set
        print("\nCollecting logits from validation set...")
        all_logits = []
        all_labels = []
        all_masks = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Collecting"):
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

        # Fit global calibrator
        print("\nFitting global temperature scaling...")
        num_classes = logits.shape[1]
        calibrator_global = TemperatureScaling(mode='global', num_classes=num_classes)
        temp_global = calibrator_global.fit(logits, labels, masks)
        print(f"Optimal global temperature: {temp_global:.4f}")

        # Move calibrator to device
        calibrator_global = calibrator_global.to(self.device)

        # Fit label-wise calibrator
        print("\nFitting label-wise temperature scaling...")
        calibrator_label_wise = TemperatureScaling(mode='label_wise', num_classes=num_classes)
        temp_label_wise = calibrator_label_wise.fit(logits, labels, masks)
        print("Optimal temperatures per label:")
        for i, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
            print(f"  {label_name}: {temp_label_wise[i]:.4f}")

        # Move calibrator to device
        calibrator_label_wise = calibrator_label_wise.to(self.device)

        # Save calibrators
        output_dir = self.config['output']['dir']
        os.makedirs(output_dir, exist_ok=True)

        global_path = os.path.join(output_dir, 'calibrator_global.pkl')
        with open(global_path, 'wb') as f:
            pickle.dump(calibrator_global, f)
        print(f"\nGlobal calibrator saved to: {global_path}")

        label_wise_path = os.path.join(output_dir, 'calibrator_label_wise.pkl')
        with open(label_wise_path, 'wb') as f:
            pickle.dump(calibrator_label_wise, f)
        print(f"Label-wise calibrator saved to: {label_wise_path}")

        # Log to wandb
        wandb.config.update({
            'temperature_global': temp_global,
            'temperature_label_wise': temp_label_wise.tolist(),
        })

        return calibrator_global, calibrator_label_wise

    def _setup_data(self, split='test'):
        """Setup data loader for specified split.

        Args:
            split: 'train', 'val', or 'test'
        """
        data_config = self.config['data']
        model_name = self.config['model']['name']

        # Get model-specific transforms
        transform = get_val_transform(model_name)

        # Load dataset based on split type
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

        # Select dataset based on split
        if split == 'train':
            dataset = train_ds
        elif split == 'val':
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=safe_collate_fn,
        )

        print(f"{split.capitalize()} dataset size: {len(dataset)}")
        print(f"{split.capitalize()} batches: {len(data_loader)}")

        return data_loader

    def test(self):
        """Run testing on test set with optional temperature scaling."""
        print("\n" + "="*80)
        if self.temp_scaling:
            print("Testing Model - Uncalibrated + Global T + Label-wise T")
        else:
            print("Testing Model - Uncalibrated Only")
        print("="*80)

        self.model.eval()

        # Collect predictions and labels
        all_preds_uncal = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
        all_probs_uncal = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
        all_labels = {task: [] for task in mimic_cxr_jpg.chexpert_labels}

        if self.temp_scaling:
            all_preds_global = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
            all_probs_global = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
            all_preds_label_wise = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
            all_probs_label_wise = {task: [] for task in mimic_cxr_jpg.chexpert_labels}

        # Initialize calibration evaluators
        calibration_eval_uncal = CalibrationEvaluator()
        if self.temp_scaling:
            calibration_eval_global = CalibrationEvaluator()
            calibration_eval_label_wise = CalibrationEvaluator()

        # Collect full predictions for saving (all samples, all classes)
        all_labels_full = []
        all_masks_full = []
        all_probs_uncal_full = []
        if self.temp_scaling:
            all_probs_global_full = []
            all_probs_label_wise_full = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                X, Y, Ymask = batch

                X = X.to(self.device).float()
                Y = Y.to(self.device).float()
                Ymask = Ymask.to(self.device)

                # Forward pass - get uncalibrated logits
                logits_uncal = self.model(X)
                probs_uncal = torch.sigmoid(logits_uncal)

                # Add to uncalibrated calibration evaluator
                calibration_eval_uncal.add_batch(logits_uncal, Y, Ymask)

                # Collect full predictions for saving
                all_labels_full.append(Y.cpu().numpy())
                all_masks_full.append(Ymask.cpu().numpy())
                all_probs_uncal_full.append(probs_uncal.cpu().numpy())

                # Apply temperature scaling if enabled
                if self.temp_scaling:
                    # Global temperature scaling
                    logits_global = self.calibrator_global(logits_uncal)
                    probs_global = torch.sigmoid(logits_global)
                    calibration_eval_global.add_batch(logits_global, Y, Ymask)
                    all_probs_global_full.append(probs_global.cpu().numpy())

                    # Label-wise temperature scaling
                    logits_label_wise = self.calibrator_label_wise(logits_uncal)
                    probs_label_wise = torch.sigmoid(logits_label_wise)
                    calibration_eval_label_wise.add_batch(logits_label_wise, Y, Ymask)
                    all_probs_label_wise_full.append(probs_label_wise.cpu().numpy())

                # Collect predictions and labels for each task
                for i, task in enumerate(mimic_cxr_jpg.chexpert_labels):
                    mask = Ymask[:, i] == 1
                    if mask.sum() > 0:
                        # Uncalibrated
                        all_probs_uncal[task].append(probs_uncal[mask, i].cpu().numpy())
                        all_preds_uncal[task].append((probs_uncal[mask, i] > 0.5).cpu().numpy())
                        all_labels[task].append(Y[mask, i].cpu().numpy())

                        # Temperature scaled
                        if self.temp_scaling:
                            # Global
                            all_probs_global[task].append(probs_global[mask, i].cpu().numpy())
                            all_preds_global[task].append((probs_global[mask, i] > 0.5).cpu().numpy())

                            # Label-wise
                            all_probs_label_wise[task].append(probs_label_wise[mask, i].cpu().numpy())
                            all_preds_label_wise[task].append((probs_label_wise[mask, i] > 0.5).cpu().numpy())

        # Compute classification metrics
        print("\n" + "="*80)
        print("Classification Metrics")
        print("="*80)

        results_uncal = self._compute_classification_metrics(
            all_probs_uncal, all_preds_uncal, all_labels, prefix="Uncalibrated"
        )

        if self.temp_scaling:
            results_global = self._compute_classification_metrics(
                all_probs_global, all_preds_global, all_labels, prefix="Global T"
            )
            results_label_wise = self._compute_classification_metrics(
                all_probs_label_wise, all_preds_label_wise, all_labels, prefix="Label-wise T"
            )
        else:
            results_global = None
            results_label_wise = None

        # Compute calibration metrics
        print("\n" + "="*80)
        print("Calibration Metrics")
        print("="*80)

        print("\nUncalibrated:")
        calibration_results_uncal = calibration_eval_uncal.compute_metrics(n_bins=15)
        calibration_eval_uncal.print_results(calibration_results_uncal, label_names=mimic_cxr_jpg.chexpert_labels)

        if self.temp_scaling:
            print("\nGlobal Temperature Scaling:")
            calibration_results_global = calibration_eval_global.compute_metrics(n_bins=15)
            calibration_eval_global.print_results(calibration_results_global, label_names=mimic_cxr_jpg.chexpert_labels)

            print("\nLabel-wise Temperature Scaling:")
            calibration_results_label_wise = calibration_eval_label_wise.compute_metrics(n_bins=15)
            calibration_eval_label_wise.print_results(calibration_results_label_wise, label_names=mimic_cxr_jpg.chexpert_labels)
        else:
            calibration_results_global = None
            calibration_results_label_wise = None

        # Prepare predictions data for saving
        predictions_data = {
            'labels': np.concatenate(all_labels_full, axis=0),
            'masks': np.concatenate(all_masks_full, axis=0),
            'probs_uncalibrated': np.concatenate(all_probs_uncal_full, axis=0),
        }
        if self.temp_scaling:
            predictions_data['probs_global_t'] = np.concatenate(all_probs_global_full, axis=0)
            predictions_data['probs_label_wise_t'] = np.concatenate(all_probs_label_wise_full, axis=0)

        # Save results
        self._save_results(results_uncal, results_global, results_label_wise,
                          calibration_results_uncal, calibration_results_global, calibration_results_label_wise,
                          predictions_data=predictions_data)

        # Log results to wandb
        self._log_to_wandb(results_uncal, results_global, results_label_wise,
                          calibration_results_uncal, calibration_results_global, calibration_results_label_wise)

        # Finish wandb run
        wandb.finish()

        return results_uncal, results_global, results_label_wise, \
               calibration_results_uncal, calibration_results_global, calibration_results_label_wise

    def _compute_classification_metrics(self, all_probs, all_preds, all_labels, prefix=""):
        """Compute classification metrics from collected predictions."""
        results = {}
        for task in mimic_cxr_jpg.chexpert_labels:
            if len(all_preds[task]) == 0:
                print(f"\nWarning: No valid samples for {task}")
                continue

            # Concatenate all batches
            task_probs = np.concatenate(all_probs[task])
            task_preds = np.concatenate(all_preds[task])
            task_labels = np.concatenate(all_labels[task])

            # Compute metrics
            metrics = {}

            # AUC
            try:
                metrics['AUC'] = roc_auc_score(task_labels, task_probs)
            except ValueError:
                metrics['AUC'] = 0.0

            # Average Precision
            try:
                metrics['AP'] = average_precision_score(task_labels, task_probs)
            except ValueError:
                metrics['AP'] = 0.0

            # Accuracy
            metrics['Accuracy'] = accuracy_score(task_labels, task_preds)

            # F1 Score
            try:
                metrics['F1'] = f1_score(task_labels, task_preds, zero_division=0)
            except ValueError:
                metrics['F1'] = 0.0

            # Support (number of samples)
            metrics['Support'] = len(task_labels)

            results[task] = metrics

        # Print results
        self._print_results(results, prefix)

        return results

    def _print_results(self, results, prefix=""):
        """Print test results in a formatted table."""
        title = f"{prefix} Test Results" if prefix else "Test Results"
        print(f"\n{title}:")
        print("-"*80)
        print(f"{'Disease':<30} {'AUC':>8} {'AP':>8} {'Acc':>8} {'F1':>8} {'Support':>10}")
        print("-"*80)

        auc_list = []
        ap_list = []
        acc_list = []
        f1_list = []

        for task, metrics in results.items():
            print(f"{task:<30} {metrics['AUC']:>8.4f} {metrics['AP']:>8.4f} "
                  f"{metrics['Accuracy']:>8.4f} {metrics['F1']:>8.4f} {metrics['Support']:>10}")

            if metrics['AUC'] > 0:
                auc_list.append(metrics['AUC'])
            if metrics['AP'] > 0:
                ap_list.append(metrics['AP'])
            acc_list.append(metrics['Accuracy'])
            f1_list.append(metrics['F1'])

        print("-"*80)
        print(f"{'Average':<30} {np.mean(auc_list):>8.4f} {np.mean(ap_list):>8.4f} "
              f"{np.mean(acc_list):>8.4f} {np.mean(f1_list):>8.4f} {sum([m['Support'] for m in results.values()]):>10}")
        print("="*80)

    def _save_results(self, results_uncal, results_global, results_label_wise,
                     calibration_results_uncal, calibration_results_global, calibration_results_label_wise,
                     predictions_data=None):
        """Save test results to CSV files."""
        output_dir = self.config['output']['dir']
        os.makedirs(output_dir, exist_ok=True)

        # Save uncalibrated classification results
        df_uncal = pd.DataFrame(results_uncal).T
        df_uncal.to_csv(os.path.join(output_dir, 'test_results_uncalibrated.csv'))
        print(f"\nUncalibrated classification results saved to: {output_dir}/test_results_uncalibrated.csv")

        # Save temperature scaling results if available
        if results_global is not None:
            df_global = pd.DataFrame(results_global).T
            df_global.to_csv(os.path.join(output_dir, 'test_results_global_t.csv'))
            print(f"Global T classification results saved to: {output_dir}/test_results_global_t.csv")

        if results_label_wise is not None:
            df_label_wise = pd.DataFrame(results_label_wise).T
            df_label_wise.to_csv(os.path.join(output_dir, 'test_results_label_wise_t.csv'))
            print(f"Label-wise T classification results saved to: {output_dir}/test_results_label_wise_t.csv")

        # Save uncalibrated calibration metrics
        self._save_calibration_metrics(calibration_results_uncal, 'uncalibrated')

        # Save temperature scaling calibration metrics if available
        if calibration_results_global is not None:
            self._save_calibration_metrics(calibration_results_global, 'global_t')

        if calibration_results_label_wise is not None:
            self._save_calibration_metrics(calibration_results_label_wise, 'label_wise_t')

        # Save predictions (probabilities/confidence) for each calibration type
        if predictions_data is not None:
            self._save_predictions(predictions_data)

    def _save_predictions(self, predictions_data):
        """Save predictions (probabilities/confidence) as numpy arrays.

        Args:
            predictions_data: Dictionary containing:
                - 'labels': Ground truth labels [N, num_classes]
                - 'masks': Valid label masks [N, num_classes]
                - 'probs_uncalibrated': Uncalibrated probabilities [N, num_classes]
                - 'probs_global_t': Global T calibrated probabilities [N, num_classes] (optional)
                - 'probs_label_wise_t': Label-wise T calibrated probabilities [N, num_classes] (optional)
        """
        output_dir = self.config['output']['dir']
        predictions_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)

        # Save ground truth labels and masks
        np.save(os.path.join(predictions_dir, 'labels.npy'), predictions_data['labels'])
        np.save(os.path.join(predictions_dir, 'masks.npy'), predictions_data['masks'])

        # Save label names for reference
        label_names = np.array(mimic_cxr_jpg.chexpert_labels)
        np.save(os.path.join(predictions_dir, 'label_names.npy'), label_names)

        # Save uncalibrated probabilities
        np.save(os.path.join(predictions_dir, 'probs_uncalibrated.npy'),
                predictions_data['probs_uncalibrated'])

        # Save calibrated probabilities if available
        if 'probs_global_t' in predictions_data:
            np.save(os.path.join(predictions_dir, 'probs_global_t.npy'),
                    predictions_data['probs_global_t'])

        if 'probs_label_wise_t' in predictions_data:
            np.save(os.path.join(predictions_dir, 'probs_label_wise_t.npy'),
                    predictions_data['probs_label_wise_t'])

        print(f"\nPredictions saved to: {predictions_dir}/")
        print(f"  - labels.npy: Ground truth labels [{predictions_data['labels'].shape}]")
        print(f"  - masks.npy: Valid label masks [{predictions_data['masks'].shape}]")
        print(f"  - label_names.npy: Label names [{len(label_names)}]")
        print(f"  - probs_uncalibrated.npy: Uncalibrated probabilities [{predictions_data['probs_uncalibrated'].shape}]")
        if 'probs_global_t' in predictions_data:
            print(f"  - probs_global_t.npy: Global T calibrated probabilities [{predictions_data['probs_global_t'].shape}]")
        if 'probs_label_wise_t' in predictions_data:
            print(f"  - probs_label_wise_t.npy: Label-wise T calibrated probabilities [{predictions_data['probs_label_wise_t'].shape}]")

    def _save_calibration_metrics(self, calibration_results, prefix):
        """Save calibration metrics to a single CSV file per case."""
        output_dir = self.config['output']['dir']
        os.makedirs(output_dir, exist_ok=True)

        # Collect all label keys across metrics
        all_keys = set()
        for metric_dict in calibration_results.values():
            all_keys.update(metric_dict.keys())

        # Build rows with all metrics in one table
        data = []
        for key in sorted(all_keys, key=lambda k: (-1 if k == 'overall' else int(k.split('_')[1]))):
            if key == 'overall':
                label_name = 'Overall'
                label_idx = -1
            else:
                label_idx = int(key.split('_')[1])
                label_name = mimic_cxr_jpg.chexpert_labels[label_idx]

            row = {'label_index': label_idx, 'label_name': label_name}
            for metric_name, metric_dict in calibration_results.items():
                row[metric_name] = metric_dict.get(key, None)
            data.append(row)

        # Sort: labels first (by index), then overall at the end
        data.sort(key=lambda r: (r['label_index'] == -1, r['label_index']))

        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f'calibration_{prefix}.csv')
        df.to_csv(csv_path, index=False)

        print(f"{prefix.capitalize()} calibration metrics saved to: {csv_path}")

    def _log_to_wandb(self, results_uncal, results_global, results_label_wise,
                     calibration_results_uncal, calibration_results_global, calibration_results_label_wise):
        """Log all results to wandb."""
        # Log uncalibrated classification metrics
        self._log_classification_metrics(results_uncal, 'uncalibrated')

        # Log temperature scaling classification metrics if available
        if results_global is not None:
            self._log_classification_metrics(results_global, 'global_t')

        if results_label_wise is not None:
            self._log_classification_metrics(results_label_wise, 'label_wise_t')

        # Log uncalibrated calibration metrics
        self._log_calibration_metrics(calibration_results_uncal, 'uncalibrated')

        # Log temperature scaling calibration metrics if available
        if calibration_results_global is not None:
            self._log_calibration_metrics(calibration_results_global, 'global_t')

        if calibration_results_label_wise is not None:
            self._log_calibration_metrics(calibration_results_label_wise, 'label_wise_t')

        print("\nAll results logged to wandb")

    def _log_classification_metrics(self, results, prefix):
        """Log classification metrics to wandb."""
        # Calculate average metrics
        auc_list = []
        ap_list = []
        acc_list = []
        f1_list = []

        # Log per-task metrics
        for task, metrics in results.items():
            wandb.log({
                f'{prefix}/{task}/AUC': metrics['AUC'],
                f'{prefix}/{task}/AP': metrics['AP'],
                f'{prefix}/{task}/Accuracy': metrics['Accuracy'],
                f'{prefix}/{task}/F1': metrics['F1'],
            })

            if metrics['AUC'] > 0:
                auc_list.append(metrics['AUC'])
            if metrics['AP'] > 0:
                ap_list.append(metrics['AP'])
            acc_list.append(metrics['Accuracy'])
            f1_list.append(metrics['F1'])

        # Log average metrics
        wandb.log({
            f'{prefix}/average/AUC': np.mean(auc_list) if auc_list else 0.0,
            f'{prefix}/average/AP': np.mean(ap_list) if ap_list else 0.0,
            f'{prefix}/average/Accuracy': np.mean(acc_list) if acc_list else 0.0,
            f'{prefix}/average/F1': np.mean(f1_list) if f1_list else 0.0,
        })

    def _log_calibration_metrics(self, calibration_results, prefix):
        """Log calibration metrics to wandb."""
        # Log each calibration metric
        for metric_name, metric_dict in calibration_results.items():
            # Log per-task calibration metrics
            for key, value in metric_dict.items():
                if key == 'overall':
                    wandb.log({f'{prefix}/calibration/{metric_name}/overall': value})
                else:
                    label_idx = int(key.split('_')[1])
                    label_name = mimic_cxr_jpg.chexpert_labels[label_idx]
                    wandb.log({f'{prefix}/calibration/{metric_name}/{label_name}': value})


def main():
    parser = argparse.ArgumentParser(
        description='Test CXR classification model with optional temperature scaling'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--temp_scaling',
        type=str,
        default='false',
        choices=['true', 'false'],
        help='Enable temperature scaling (true/false). If true, fits both global and label-wise T on validation set.'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='+',
        help='Override config values (e.g., output.dir=./outputs/my_test)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply overrides if provided
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

    # Convert string to boolean
    temp_scaling = args.temp_scaling.lower() == 'true'

    # Create tester and run test
    tester = Tester(config, args.checkpoint, temp_scaling=temp_scaling)
    tester.test()

    if temp_scaling:
        print("\nTemperature scaling testing completed!")
        print("Results: Uncalibrated + Global T + Label-wise T")
    else:
        print("\nUncalibrated testing completed!")


if __name__ == '__main__':
    main()
