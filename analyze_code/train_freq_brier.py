"""
Training script with frequency-aware calibration loss.

Applies group-wise Brier score loss based on label frequency:
- Head group: low calibration weight
- Medium group: medium calibration weight
- Tail group: high calibration weight (stronger calibration)

The calibration loss is applied only after training stabilizes (warmup epochs).

Usage:
    python train_freq_brier.py --config configs/resnet50_config.yaml
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb

import re
import glob

import mimic_cxr_jpg
from meters import CSVMeter
from model_factory import get_model
from data_transforms import get_train_transform, get_val_transform
from utils import load_config, set_seed, setup_output_dir, save_config, count_parameters, get_lr


# CheXpert labels
CHEXPERT_LABELS = mimic_cxr_jpg.chexpert_labels


def compute_label_frequency_from_dataset(dataset, num_samples=None):
    """
    Compute label frequency from training dataset.

    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to use (None for all)

    Returns:
        freq_dict: {label_name: frequency}
        count_dict: {label_name: count}
    """
    print("Computing label frequency from training dataset...")

    n_samples = min(len(dataset), num_samples) if num_samples else len(dataset)
    label_counts = np.zeros(len(CHEXPERT_LABELS))
    valid_counts = np.zeros(len(CHEXPERT_LABELS))

    for i in tqdm(range(n_samples), desc="Counting labels"):
        try:
            _, labels, mask = dataset[i]
            labels = labels.numpy()
            mask = mask.numpy()

            for j in range(len(CHEXPERT_LABELS)):
                if mask[j] == 1:
                    valid_counts[j] += 1
                    if labels[j] == 1:
                        label_counts[j] += 1
        except:
            continue

    freq_dict = {}
    count_dict = {}
    for i, label in enumerate(CHEXPERT_LABELS):
        count_dict[label] = int(label_counts[i])
        freq_dict[label] = label_counts[i] / valid_counts[i] if valid_counts[i] > 0 else 0

    return freq_dict, count_dict


def assign_frequency_groups(freq_dict):
    """
    Assign labels to Head/Medium/Tail groups based on log frequency quantiles.

    Returns:
        group_assignment: {label_name: 'Head'/'Medium'/'Tail'}
        label_to_group_idx: {label_name: 0/1/2} (Head=0, Medium=1, Tail=2)
    """
    # Compute log frequency
    log_freqs = {}
    for label, freq in freq_dict.items():
        count = freq * 1000  # Approximate count for log
        log_freqs[label] = np.log10(max(count, 1))

    # Sort by log frequency
    sorted_labels = sorted(log_freqs.keys(), key=lambda x: log_freqs[x], reverse=True)

    # Compute quantile thresholds
    log_freq_values = sorted([log_freqs[l] for l in sorted_labels], reverse=True)
    n = len(log_freq_values)
    q_high = log_freq_values[n // 3]  # Top 33%
    q_low = log_freq_values[2 * n // 3]  # Bottom 33%

    group_assignment = {}
    label_to_group_idx = {}

    for label in CHEXPERT_LABELS:
        lf = log_freqs[label]
        if lf >= q_high:
            group_assignment[label] = 'Head'
            label_to_group_idx[label] = 0
        elif lf >= q_low:
            group_assignment[label] = 'Medium'
            label_to_group_idx[label] = 1
        else:
            group_assignment[label] = 'Tail'
            label_to_group_idx[label] = 2

    return group_assignment, label_to_group_idx


def find_latest_checkpoint(weight_dir: str, model_name: str) -> tuple:
    """Find the latest checkpoint for a given model."""
    pattern = os.path.join(weight_dir, f'{model_name}_*ep.pt')
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, -1

    max_epoch = -1
    latest_checkpoint = None

    for ckpt in checkpoints:
        match = re.search(rf'{model_name}_(\d+)ep\.pt$', ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = ckpt

    return latest_checkpoint, max_epoch


def safe_collate_fn(batch):
    """Custom collate function that creates independent tensor copies."""
    images, labels, masks = zip(*batch)
    images_batch = torch.stack([img.clone() for img in images])
    labels_batch = torch.stack([lbl.clone() for lbl in labels])
    masks_batch = torch.stack([msk.clone() for msk in masks])
    return images_batch, labels_batch, masks_batch


class FrequencyAwareBrierLoss(nn.Module):
    """
    Frequency-aware Brier score loss for calibration.

    Applies different calibration weights based on label frequency group:
    - Head: alpha_head (small)
    - Medium: alpha_medium (medium)
    - Tail: alpha_tail (large, stronger calibration)
    """

    def __init__(self, label_to_group_idx, alpha_head=0.05, alpha_medium=0.1, alpha_tail=0.2):
        """
        Args:
            label_to_group_idx: {label_name: 0/1/2}
            alpha_head: Brier loss weight for head group
            alpha_medium: Brier loss weight for medium group
            alpha_tail: Brier loss weight for tail group
        """
        super().__init__()
        self.label_to_group_idx = label_to_group_idx

        # Create tensor mapping label index to group alpha
        self.alphas = torch.tensor([alpha_head, alpha_medium, alpha_tail])

        # Create label-to-alpha mapping
        label_alphas = []
        for label in CHEXPERT_LABELS:
            group_idx = label_to_group_idx[label]
            label_alphas.append(self.alphas[group_idx])
        self.label_alphas = torch.stack(label_alphas)  # [num_classes]

        print(f"\nFrequencyAwareBrierLoss initialized:")
        print(f"  Alpha Head: {alpha_head}")
        print(f"  Alpha Medium: {alpha_medium}")
        print(f"  Alpha Tail: {alpha_tail}")

    def forward(self, logits, targets, mask):
        """
        Compute frequency-aware Brier loss.

        Args:
            logits: [B, C] raw logits
            targets: [B, C] binary labels
            mask: [B, C] valid label mask

        Returns:
            brier_loss: scalar tensor
        """
        # Move alphas to same device as logits
        label_alphas = self.label_alphas.to(logits.device)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Brier score: (p - y)^2
        brier = (probs - targets) ** 2

        # Apply label-wise alpha weights
        # label_alphas: [C] -> expand to [1, C] for broadcasting
        weighted_brier = brier * label_alphas.unsqueeze(0)

        # Apply mask and compute mean
        masked_brier = weighted_brier * mask

        if mask.sum() > 0:
            loss = masked_brier.sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss


class TrainerFreqBrier:
    """Trainer class with frequency-aware Brier calibration loss."""

    def __init__(self, config, resume=True, checkpoint_dir=None):
        """Initialize trainer with configuration."""
        self.config = config
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir or '/mnt/HDD/yoonji/mrg/cxr_classification/weight'

        # Calibration loss settings
        self.calibration_config = config.get('calibration', {})
        self.warmup_epochs = self.calibration_config.get('warmup_epochs', 5)
        self.alpha_head = self.calibration_config.get('alpha_head', 0.05)
        self.alpha_medium = self.calibration_config.get('alpha_medium', 0.1)
        self.alpha_tail = self.calibration_config.get('alpha_tail', 0.2)

        # Check CUDA availability
        device_str = config['system']['device']
        if 'cuda' in device_str and not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
            device_str = 'cpu'

        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        # Set random seed
        set_seed(config['system']['seed'])

        # Setup output directory
        self.output_dir = setup_output_dir(config['output']['dir'])
        save_config(config, self.output_dir)

        # Initialize wandb
        wandb.init(
            project="cxr-classification",
            name=f"{config['model']['name']}_freq_brier_fold{config['data'].get('fold', 0)}",
            config=config,
            dir=self.output_dir,
            tags=['freq_brier', 'calibration'],
        )
        print(f"Weights & Biases run: {wandb.run.name}")

        # Initialize model
        self.model = self._setup_model()

        # Setup data (need train dataset for frequency computation)
        self.train_loader, self.val_loader, self.test_loader, self.train_ds = self._setup_data()

        # Compute frequency groups from training data
        self._setup_frequency_groups()

        # Setup training components
        self.criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion_brier = FrequencyAwareBrierLoss(
            self.label_to_group_idx,
            alpha_head=self.alpha_head,
            alpha_medium=self.alpha_medium,
            alpha_tail=self.alpha_tail,
        )

        self.optimizer = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['training']['amp'])

        # Setup metrics tracking
        self.epoch_meter = CSVMeter(os.path.join(self.output_dir, 'epoch_metrics.csv'), buffering=1)
        self.val_meter = CSVMeter(os.path.join(self.output_dir, 'val_metrics.csv'), buffering=1)
        self.iter_meter = CSVMeter(os.path.join(self.output_dir, 'iter_metrics.csv'))

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.total_iters = 0
        self.best_val_loss = float('inf')
        self.val_loss_history = []
        self.epochs_without_improvement = 0

        # Try to resume from checkpoint if enabled
        if self.resume:
            self._try_resume_from_checkpoint()

    def _setup_frequency_groups(self):
        """Compute label frequency from training data and assign groups."""
        print("\n" + "=" * 80)
        print("Computing Frequency Groups from Training Data")
        print("=" * 80)

        # Compute frequency (use subset for speed if dataset is large)
        max_samples = min(len(self.train_ds), 10000)
        self.freq_dict, self.count_dict = compute_label_frequency_from_dataset(
            self.train_ds, num_samples=max_samples
        )

        # Assign groups
        self.group_assignment, self.label_to_group_idx = assign_frequency_groups(self.freq_dict)

        # Print results
        print("\nLabel Frequency Groups (from training data):")
        print("-" * 60)

        for group_name in ['Head', 'Medium', 'Tail']:
            labels_in_group = [l for l, g in self.group_assignment.items() if g == group_name]
            print(f"\n{group_name} Group:")
            for label in labels_in_group:
                print(f"  {label}: {self.freq_dict[label]:.4f} ({self.count_dict[label]} samples)")

        # Save frequency info
        import json
        freq_info = {
            'frequency': self.freq_dict,
            'counts': self.count_dict,
            'groups': self.group_assignment,
            'label_to_group_idx': self.label_to_group_idx,
        }
        with open(os.path.join(self.output_dir, 'frequency_groups.json'), 'w') as f:
            json.dump(freq_info, f, indent=2)

        # Log to wandb
        wandb.config.update({
            'frequency_groups': self.group_assignment,
            'warmup_epochs': self.warmup_epochs,
            'alpha_head': self.alpha_head,
            'alpha_medium': self.alpha_medium,
            'alpha_tail': self.alpha_tail,
        })

    def _try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint."""
        weight_dir = self.checkpoint_dir
        model_name = self.config['model']['name']

        checkpoint_path, last_epoch = find_latest_checkpoint(weight_dir, model_name)

        if checkpoint_path is None:
            print(f"\nNo checkpoint found for {model_name} in {weight_dir}")
            print("Starting training from epoch 0 with pretrained weights")
            return

        print(f"\n" + "="*80)
        print(f"RESUMING TRAINING")
        print(f"="*80)
        print(f"Found checkpoint: {checkpoint_path}")
        print(f"Last completed epoch: {last_epoch}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(new_state_dict)

        self.start_epoch = last_epoch + 1

        num_epochs = self.config['training']['num_epochs']
        remaining_epochs = num_epochs - self.start_epoch

        print(f"Target epochs: {num_epochs}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"="*80 + "\n")

    def _setup_model(self):
        """Setup model based on configuration."""
        model = get_model(self.config['model'])

        total_params, trainable_params = count_parameters(model)
        print(f"Model: {self.config['model']['name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        model = model.to(self.device)

        if self.config['training']['distributed']:
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            dist.init_process_group('nccl')
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        elif self.config['training']['single_node_parallel']:
            model = nn.DataParallel(model)

        return model

    def _setup_data(self):
        """Setup data loaders based on configuration."""
        data_config = self.config['data']
        train_config = self.config['training']
        model_name = self.config['model']['name']

        train_transform = get_train_transform(model_name, self.config['augmentation'])
        val_transform = get_val_transform(model_name)

        if data_config['split_type'] == 'official':
            train_ds, val_ds, test_ds = mimic_cxr_jpg.official_split(
                datadir=data_config['datadir'],
                dicom_id_file=data_config['dicom_id_file'],
                image_subdir=data_config['image_subdir'],
                train_transform=train_transform,
                test_transform=val_transform,
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
                train_transform=train_transform,
                test_transform=val_transform,
                label_method=data_config['label_method'],
                subject_prefix_filter=data_config.get('subject_prefix_filter', None),
            )
        else:
            raise ValueError(f"Unknown split_type: {data_config['split_type']}")

        use_distributed = train_config['distributed']

        train_loader = DataLoader(
            train_ds,
            batch_size=train_config['batch_size'],
            shuffle=not use_distributed,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            sampler=DistributedSampler(train_ds, shuffle=True) if use_distributed else None,
            collate_fn=safe_collate_fn,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            sampler=DistributedSampler(val_ds) if use_distributed else None,
            collate_fn=safe_collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            sampler=DistributedSampler(test_ds) if use_distributed else None,
            collate_fn=safe_collate_fn,
        )

        print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        print(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

        return train_loader, val_loader, test_loader, train_ds

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        train_config = self.config['training']
        optimizer_name = train_config['optimizer'].lower()

        lr = float(train_config['learning_rate'])
        weight_decay = float(train_config.get('weight_decay', 0.0))

        params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")

        print(f"\nOptimizer setup:")
        print(f"  Trainable parameters: {sum(p.numel() for p in params):,}")
        print(f"  Learning rate: {lr}")

        if optimizer_name == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        show_progress = self.config['system']['progress_bar']

        print(f"\n" + "=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(f"  Warmup epochs (BCE only): {self.warmup_epochs}")
        print(f"  Calibration starts at epoch: {self.warmup_epochs}")
        print(f"  Alpha Head: {self.alpha_head}")
        print(f"  Alpha Medium: {self.alpha_medium}")
        print(f"  Alpha Tail: {self.alpha_tail}")
        print("=" * 80 + "\n")

        if self.start_epoch >= num_epochs:
            print(f"Training already completed ({self.start_epoch} >= {num_epochs} epochs)")
        else:
            epoch_bar = range(self.start_epoch, num_epochs)
            if show_progress:
                epoch_bar = tqdm(epoch_bar, desc='Epochs', initial=self.start_epoch, total=num_epochs)

            for epoch in epoch_bar:
                self.current_epoch = epoch

                # Train for one epoch
                train_loss, bce_loss, brier_loss = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Update learning rate
                current_val_loss = val_metrics['val']['loss']
                self._update_learning_rate(current_val_loss)

                # Save checkpoint
                if (epoch + 1) % self.config['output']['save_frequency'] == 0:
                    self.save_checkpoint(epoch)

                # Log metrics
                self.log_epoch_metrics(epoch, train_loss, bce_loss, brier_loss, val_metrics)

                # Check for early stopping
                if self._should_early_stop():
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final validation on test set
        print("\nFinal evaluation on test set:")
        test_metrics = self.validate(use_test=True)
        self.log_test_metrics(test_metrics)

        wandb.finish()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_brier_loss = 0.0
        show_progress = self.config['system']['progress_bar']

        # Check if we should apply calibration loss
        use_calibration = self.current_epoch >= self.warmup_epochs

        iter_bar = self.train_loader
        if show_progress:
            calib_status = "BCE+Brier" if use_calibration else "BCE only"
            iter_bar = tqdm(iter_bar, desc=f'Epoch {self.current_epoch} ({calib_status})', leave=False)

        for batch_idx, batch in enumerate(iter_bar):
            loss, bce_loss, brier_loss = self.train_step(batch, use_calibration)

            if loss is not None:
                epoch_loss += loss
                epoch_bce_loss += bce_loss
                epoch_brier_loss += brier_loss if brier_loss else 0

                self.iter_meter.update(
                    loss=loss,
                    bce_loss=bce_loss,
                    brier_loss=brier_loss if brier_loss else 0,
                    epoch=self.current_epoch
                )

                wandb.log({
                    'train/loss_iter': loss,
                    'train/bce_loss_iter': bce_loss,
                    'train/brier_loss_iter': brier_loss if brier_loss else 0,
                }, step=self.total_iters)

                if show_progress:
                    iter_bar.set_postfix({'loss': f'{loss:.4f}', 'bce': f'{bce_loss:.4f}'})

            self.total_iters += 1

        n_batches = len(self.train_loader)
        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce_loss / n_batches
        avg_brier = epoch_brier_loss / n_batches

        return avg_loss, avg_bce, avg_brier

    def train_step(self, batch, use_calibration=False):
        """Single training step."""
        use_amp = self.config['training']['amp']

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = self.forward_batch(batch, use_calibration)

        if outputs is None:
            return None, None, None

        loss, bce_loss, brier_loss = outputs

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), bce_loss.item(), brier_loss.item() if brier_loss is not None else None

    def forward_batch(self, batch, use_calibration=False):
        """Forward pass for a batch."""
        X, Y, Ymask = batch

        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        Ymask = Ymask.to(self.device)

        weightsum = Ymask.sum().item()
        if weightsum == 0:
            return None

        # Forward pass
        preds = self.model(X)

        # BCE Loss
        bce = self.criterion_bce(preds, Y)
        bce_loss = (bce * Ymask).sum() / weightsum

        # Brier Loss (only after warmup)
        if use_calibration:
            brier_loss = self.criterion_brier(preds, Y, Ymask)
            loss = bce_loss + brier_loss
        else:
            brier_loss = None
            loss = bce_loss

        return loss, bce_loss, brier_loss

    def validate(self, use_test=False):
        """Validate model."""
        self.model.eval()
        metrics = {}

        loaders = [('val', self.val_loader)]
        if use_test:
            loaders.append(('test', self.test_loader))

        for split_name, loader in loaders:
            val_loss = 0.0
            all_preds = {task: [] for task in CHEXPERT_LABELS}
            all_labels = {task: [] for task in CHEXPERT_LABELS}

            with torch.no_grad():
                for batch in loader:
                    X, Y, Ymask = batch
                    X = X.to(self.device).float()
                    Y = Y.to(self.device).float()
                    Ymask = Ymask.to(self.device)

                    weightsum = Ymask.sum().item()
                    if weightsum == 0:
                        continue

                    preds = self.model(X)

                    bce = self.criterion_bce(preds, Y)
                    loss = (bce * Ymask).sum() / weightsum
                    val_loss += loss.item()

                    for i, task in enumerate(CHEXPERT_LABELS):
                        mask = Ymask[:, i] == 1
                        if mask.sum() > 0:
                            all_preds[task].append(preds[mask, i].cpu().numpy())
                            all_labels[task].append(Y[mask, i].cpu().numpy())

            task_metrics = {'AUC': {}, 'AveragePrecision': {}}

            for task in CHEXPERT_LABELS:
                if len(all_preds[task]) == 0:
                    continue

                pred_concat = np.concatenate(all_preds[task])
                label_concat = np.concatenate(all_labels[task])

                ap = average_precision_score(label_concat, pred_concat)
                task_metrics['AveragePrecision'][task] = ap

                try:
                    auc = roc_auc_score(label_concat, pred_concat)
                except ValueError:
                    auc = 0.0
                task_metrics['AUC'][task] = auc

            metrics[split_name] = {
                'loss': val_loss / len(loader),
                'metrics': task_metrics,
            }

        self.model.train()
        return metrics

    def _update_learning_rate(self, val_loss):
        """Update learning rate based on validation loss."""
        lr_config = self.config['training'].get('lr_scheduler', {})
        if not lr_config.get('enabled', False):
            return

        self.val_loss_history.append(val_loss)

        patience = lr_config.get('patience', 3)
        if len(self.val_loss_history) > patience:
            recent_losses = self.val_loss_history[-patience:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                factor = lr_config.get('factor', 0.5)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= factor
                print(f"Reduced learning rate to {get_lr(self.optimizer):.6f}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _should_early_stop(self):
        """Check if training should be stopped early."""
        lr_config = self.config['training'].get('lr_scheduler', {})
        if not lr_config.get('enabled', False):
            return False

        patience = lr_config.get('early_stop_patience', 10)
        return self.epochs_without_improvement >= patience

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        weight_dir = self.checkpoint_dir
        os.makedirs(weight_dir, exist_ok=True)

        model_name = self.config['model']['name']
        checkpoint_path = os.path.join(weight_dir, f'{model_name}_freq_brier_{epoch}ep.pt')

        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'frequency_groups': self.group_assignment,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    def log_epoch_metrics(self, epoch, train_loss, bce_loss, brier_loss, val_metrics):
        """Log metrics for current epoch."""
        use_calibration = epoch >= self.warmup_epochs

        for split, metrics_dict in val_metrics.items():
            for metric_name, task_metrics in metrics_dict['metrics'].items():
                for task, value in task_metrics.items():
                    self.val_meter.update(
                        epoch=epoch,
                        split=split,
                        metric=metric_name,
                        task=task,
                        value=value,
                        loss=metrics_dict['loss'],
                    )

        self.epoch_meter.update(
            epoch=epoch,
            train_loss=train_loss,
            bce_loss=bce_loss,
            brier_loss=brier_loss,
            val_loss=val_metrics['val']['loss'],
            lr=get_lr(self.optimizer),
            use_calibration=use_calibration,
        )

        wandb_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/bce_loss': bce_loss,
            'train/brier_loss': brier_loss,
            'train/use_calibration': int(use_calibration),
            'val/loss': val_metrics['val']['loss'],
            'learning_rate': get_lr(self.optimizer),
        }

        for split, metrics_dict in val_metrics.items():
            for metric_name, task_metrics in metrics_dict['metrics'].items():
                for task, value in task_metrics.items():
                    wandb_log[f'{split}/{metric_name}/{task}'] = value

        wandb.log(wandb_log, step=epoch)

        self.epoch_meter.flush()
        self.val_meter.flush()
        self.iter_meter.flush()

        calib_status = "YES" if use_calibration else "NO (warmup)"
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (BCE: {bce_loss:.4f}, Brier: {brier_loss:.4f})")
        print(f"  Val Loss: {val_metrics['val']['loss']:.4f}")
        print(f"  Calibration: {calib_status}")
        print(f"  Learning Rate: {get_lr(self.optimizer):.6f}")

    def log_test_metrics(self, test_metrics):
        """Log final test metrics."""
        print("\nTest Set Results:")

        test_log = {}
        for metric_name, task_metrics in test_metrics['test']['metrics'].items():
            print(f"\n{metric_name}:")
            for task, value in task_metrics.items():
                print(f"  {task}: {value:.4f}")
                test_log[f'test/{metric_name}/{task}'] = value

        wandb.log(test_log)


def main():
    parser = argparse.ArgumentParser(
        description='Train CXR classification model with frequency-aware Brier calibration loss'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (BCE only) before calibration loss')
    parser.add_argument('--alpha_head', type=float, default=0.05,
                        help='Brier loss weight for head (high-freq) group')
    parser.add_argument('--alpha_medium', type=float, default=0.1,
                        help='Brier loss weight for medium-freq group')
    parser.add_argument('--alpha_tail', type=float, default=0.2,
                        help='Brier loss weight for tail (low-freq) group')
    parser.add_argument('--resume', type=str, default='true', choices=['true', 'false'],
                        help='Resume training from latest checkpoint')
    parser.add_argument('--no-resume', action='store_true',
                        help='Disable resume and start fresh')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save/load checkpoints')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Add calibration config
    config['calibration'] = {
        'warmup_epochs': args.warmup_epochs,
        'alpha_head': args.alpha_head,
        'alpha_medium': args.alpha_medium,
        'alpha_tail': args.alpha_tail,
    }

    # Determine resume mode
    resume = (args.resume.lower() == 'true') and (not args.no_resume)

    # Create trainer and start training
    trainer = TrainerFreqBrier(config, resume=resume, checkpoint_dir=args.checkpoint_dir)
    trainer.train()


if __name__ == '__main__':
    main()
