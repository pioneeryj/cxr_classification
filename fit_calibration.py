"""Fit temperature scaling calibrator on validation set."""

import os
import argparse
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mimic_cxr_jpg
from model_factory import get_model
from data_transforms import get_val_transform
from utils import load_config, set_seed


def safe_collate_fn(batch):
    """Custom collate function that creates independent tensor copies."""
    images, labels, masks = zip(*batch)
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])
    return images_batch, labels_batch, masks_batch


class TemperatureScaling(nn.Module):
    """Temperature scaling calibration module.

    Supports two modes:
    - global: Single temperature for all labels
    - label_wise: Separate temperature for each label
    """

    def __init__(self, mode='global', num_classes=14):
        """
        Args:
            mode: 'global' or 'label_wise'
            num_classes: Number of classes (for label_wise mode)
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        if mode == 'global':
            self.temperature = nn.Parameter(torch.ones(1))
        elif mode == 'label_wise':
            self.temperature = nn.Parameter(torch.ones(num_classes))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'global' or 'label_wise'")

    def forward(self, logits):
        """Apply temperature scaling to logits.

        Args:
            logits: [batch_size, num_classes]

        Returns:
            scaled_logits: [batch_size, num_classes]
        """
        if self.mode == 'global':
            return logits / self.temperature
        else:  # label_wise
            # temperature: [num_classes]
            # Broadcasting: [B, num_classes] / [num_classes]
            return logits / self.temperature.unsqueeze(0)

    def fit(self, logits, labels, masks, lr=0.01, max_iter=50):
        """Fit temperature parameter using NLL loss on validation set.

        Args:
            logits: Validation logits [N, num_classes]
            labels: Ground truth labels [N, num_classes]
            masks: Valid label masks [N, num_classes]
            lr: Learning rate for optimization
            max_iter: Maximum number of optimization iterations

        Returns:
            temperature: Fitted temperature value(s)
        """
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits = logits.detach()
        labels = labels.detach()
        masks = masks.detach()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss_per_sample = criterion(scaled_logits, labels)
            loss = (loss_per_sample * masks).sum() / masks.sum()
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        if self.mode == 'global':
            return self.temperature.item()
        else:  # label_wise
            return self.temperature.detach().cpu().numpy()


class CalibrationFitter:
    """Fit temperature scaling calibrator on validation set."""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.checkpoint_path = checkpoint_path

        # Setup device
        device_str = config['system']['device']
        if 'cuda' in device_str and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU.")
            device_str = 'cpu'

        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Set random seed
        set_seed(config['system']['seed'])

        # Setup model
        self.model = self._setup_model()
        self._load_checkpoint()

        # Setup validation data
        self.val_loader = self._setup_data()

    def _setup_model(self):
        """Setup model based on configuration."""
        model = get_model(self.config['model'])
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
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        print("Checkpoint loaded successfully")

    def _setup_data(self):
        """Setup validation data loader."""
        data_config = self.config['data']
        model_name = self.config['model']['name']

        val_transform = get_val_transform(model_name)

        # Load validation dataset
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
        else:
            raise ValueError(f"Unknown split_type: {data_config['split_type']}")

        val_loader = DataLoader(
            val_ds,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            collate_fn=safe_collate_fn,
        )

        print(f"Validation dataset size: {len(val_ds)}")
        print(f"Validation batches: {len(val_loader)}")

        return val_loader

    def collect_logits(self):
        """Collect logits and labels from validation set."""
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

        print(f"Collected {logits.shape[0]} samples")
        return logits, labels, masks

    def fit(self, mode='global', save_path=None):
        """Fit temperature scaling calibrator.

        Args:
            mode: 'global' or 'label_wise' temperature scaling
            save_path: Path to save calibrator (default: output_dir/calibrator_{mode}.pkl)
        """
        # Collect logits from validation set
        logits, labels, masks = self.collect_logits()

        # Create and fit calibrator
        print(f"\nFitting temperature scaling ({mode} mode)...")
        num_classes = logits.shape[1]
        calibrator = TemperatureScaling(mode=mode, num_classes=num_classes)
        temperature = calibrator.fit(logits, labels, masks)

        if mode == 'global':
            print(f"Optimal temperature: {temperature:.4f}")
        else:  # label_wise
            print(f"Optimal temperatures per label:")
            import mimic_cxr_jpg
            for i, label_name in enumerate(mimic_cxr_jpg.chexpert_labels):
                print(f"  {label_name}: {temperature[i]:.4f}")

        # Save calibrator
        if save_path is None:
            output_dir = self.config['output']['dir']
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'calibrator_{mode}.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(calibrator, f)

        print(f"Calibrator saved to: {save_path}")

        return calibrator, save_path


def main():
    parser = argparse.ArgumentParser(
        description='Fit temperature scaling calibrator on validation set'
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
        '--mode',
        type=str,
        default='global',
        choices=['global', 'label_wise'],
        help='Temperature scaling mode: global (single T) or label_wise (per-label T)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save calibrator (default: output_dir/calibrator_{mode}.pkl)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Fit calibrator
    fitter = CalibrationFitter(config, args.checkpoint)
    calibrator, save_path = fitter.fit(mode=args.mode, save_path=args.output)

    print("\nCalibration fitting completed!")


if __name__ == '__main__':
    main()
