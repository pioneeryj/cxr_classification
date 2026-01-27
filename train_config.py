"""Configuration-based training script for CXR classification with multiple pretrained models."""

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


def find_latest_checkpoint(weight_dir: str, model_name: str) -> tuple:
    """Find the latest checkpoint for a given model.

    Args:
        weight_dir: Directory containing checkpoints
        model_name: Name of the model

    Returns:
        Tuple of (checkpoint_path, epoch) or (None, -1) if no checkpoint found
    """
    pattern = os.path.join(weight_dir, f'{model_name}_*ep.pt')
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, -1

    # Extract epoch numbers and find the maximum
    max_epoch = -1
    latest_checkpoint = None

    for ckpt in checkpoints:
        # Extract epoch number from filename like "resnet50_29ep.pt"
        match = re.search(rf'{model_name}_(\d+)ep\.pt$', ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = ckpt

    return latest_checkpoint, max_epoch


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


class Trainer:
    """Trainer class for CXR classification."""

    def __init__(self, config, resume=True, checkpoint_dir=None):
        """Initialize trainer with configuration.

        Args:
            config: Configuration dictionary
            resume: If True, try to resume from the latest checkpoint
            checkpoint_dir: Directory to save/load checkpoints (default: /mnt/HDD/yoonji/mrg/cxr_classification/weight)
        """
        self.config = config
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir or '/mnt/HDD/yoonji/mrg/cxr_classification/weight'

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
            name=f"{config['model']['name']}_fold{config['data'].get('fold', 0)}",
            config=config,
            dir=self.output_dir,
        )
        print(f"Weights & Biases run: {wandb.run.name}")

        # Initialize model
        self.model = self._setup_model()

        # Setup data
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        # Setup training components
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
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

    def _try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint.

        Looks for checkpoints in the weight directory matching the model name.
        If found, loads the checkpoint and sets start_epoch accordingly.
        """
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

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats (state_dict directly or wrapped)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Load state dict into model
        # Handle case where model is wrapped in DataParallel
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(new_state_dict)

        # Set start epoch to continue from where we left off
        # last_epoch is the completed epoch, so we start from last_epoch + 1
        self.start_epoch = last_epoch + 1

        num_epochs = self.config['training']['num_epochs']
        remaining_epochs = num_epochs - self.start_epoch

        print(f"Target epochs: {num_epochs}")
        print(f"Starting from epoch: {self.start_epoch}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"="*80 + "\n")

        if remaining_epochs <= 0:
            print(f"WARNING: Already trained for {last_epoch + 1} epochs, target is {num_epochs}")
            print("No additional training needed. Increase num_epochs if you want to continue.")

    def _setup_model(self):
        """Setup model based on configuration."""
        model = get_model(self.config['model'])

        # Print model info
        total_params, trainable_params = count_parameters(model)
        print(f"Model: {self.config['model']['name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Move to device
        model = model.to(self.device)

        # Setup distributed training if needed
        if self.config['training']['distributed']:
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

            dist.init_process_group('nccl')
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        elif self.config['training']['single_node_parallel']:
            model = nn.DataParallel(model)

        # Debug: Print trainable parameters info
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"\nTrainable params check:")
        print(f"  Number of trainable tensors: {len(trainable)}")
        print(f"  Total trainable parameters: {sum(p.numel() for p in trainable):,}")

        # Print which layers are trainable
        print(f"\nTrainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} params")

        return model

    def _setup_data(self):
        """Setup data loaders based on configuration."""
        data_config = self.config['data']
        train_config = self.config['training']
        model_name = self.config['model']['name']

        # Get model-specific transforms
        train_transform = get_train_transform(model_name, self.config['augmentation'])
        val_transform = get_val_transform(model_name)

        # Load datasets based on split type
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

        # Create data loaders
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

        return train_loader, val_loader, test_loader

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        train_config = self.config['training']
        optimizer_name = train_config['optimizer'].lower()

        # Convert learning_rate and weight_decay to float to handle scientific notation
        lr = float(train_config['learning_rate'])
        weight_decay = float(train_config.get('weight_decay', 0.0))

        # Get only trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Check that we have trainable parameters
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found. Check freeze settings.")

        print(f"\nOptimizer setup:")
        print(f"  Trainable parameters: {sum(p.numel() for p in params):,}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")

        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        show_progress = self.config['system']['progress_bar']

        # Check if we need to train at all
        if self.start_epoch >= num_epochs:
            print(f"Training already completed ({self.start_epoch} >= {num_epochs} epochs)")
            print("Skipping to final evaluation...")
        else:
            # Create epoch range starting from start_epoch
            epoch_bar = range(self.start_epoch, num_epochs)
            if show_progress:
                epoch_bar = tqdm(epoch_bar, desc='Epochs', initial=self.start_epoch, total=num_epochs)

            for epoch in epoch_bar:
                self.current_epoch = epoch

                # Train for one epoch
                train_loss = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Update learning rate based on validation loss
                current_val_loss = val_metrics['val']['loss']
                self._update_learning_rate(current_val_loss)

                # Save checkpoint
                if (epoch + 1) % self.config['output']['save_frequency'] == 0:
                    self.save_checkpoint(epoch)

                # Log metrics
                self.log_epoch_metrics(epoch, train_loss, val_metrics)

                # Check for early stopping
                if self._should_early_stop():
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final validation on test set
        print("\nFinal evaluation on test set:")
        test_metrics = self.validate(use_test=True)
        self.log_test_metrics(test_metrics)

        # Finish wandb run
        wandb.finish()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        show_progress = self.config['system']['progress_bar']

        iter_bar = self.train_loader
        if show_progress:
            iter_bar = tqdm(iter_bar, desc=f'Epoch {self.current_epoch}', leave=False)

        for batch_idx, batch in enumerate(iter_bar):
            loss = self.train_step(batch)

            if loss is not None:
                epoch_loss += loss
                self.iter_meter.update(loss=loss, epoch=self.current_epoch)

                # Log to wandb
                wandb.log({'train/loss_iter': loss}, step=self.total_iters)

                if show_progress:
                    iter_bar.set_postfix({'loss': f'{loss:.4f}'})

            self.total_iters += 1

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def train_step(self, batch):
        """Single training step."""
        use_amp = self.config['training']['amp']

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with optional AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = self.forward_batch(batch)

        if outputs is None:
            return None

        preds, loss = outputs

        # Backward pass - scaler handles AMP automatically
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def forward_batch(self, batch):
        """Forward pass for a batch."""
        X, Y, Ymask = batch

        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        Ymask = Ymask.to(self.device)

        # Check if there are any valid labels in this batch
        weightsum = Ymask.sum().item()
        if weightsum == 0:
            return None

        # Forward pass
        preds = self.model(X)

        # Compute loss
        bce = self.criterion(preds, Y)
        loss = (bce * Ymask).sum() / weightsum

        return preds, loss

    def validate(self, use_test=False):
        """Validate model."""
        self.model.eval()
        metrics = {}

        loaders = [('val', self.val_loader)]
        if use_test:
            loaders.append(('test', self.test_loader))

        for split_name, loader in loaders:
            val_loss = 0.0
            all_preds = {task: [] for task in mimic_cxr_jpg.chexpert_labels}
            all_labels = {task: [] for task in mimic_cxr_jpg.chexpert_labels}

            with torch.no_grad():
                for batch in loader:
                    outputs = self.forward_batch(batch)
                    if outputs is None:
                        continue

                    preds, loss = outputs
                    val_loss += loss.item()

                    # Collect predictions and labels for each task
                    X, Y, Ymask = batch
                    Y = Y.to(self.device).float()
                    Ymask = Ymask.to(self.device)

                    for i, task in enumerate(mimic_cxr_jpg.chexpert_labels):
                        mask = Ymask[:, i] == 1
                        if mask.sum() > 0:
                            all_preds[task].append(preds[mask, i].cpu().numpy())
                            all_labels[task].append(Y[mask, i].cpu().numpy())

            # Compute metrics for each task
            task_metrics = {'AUC': {}, 'AveragePrecision': {}}

            for task in mimic_cxr_jpg.chexpert_labels:
                if len(all_preds[task]) == 0:
                    continue

                pred_concat = np.concatenate(all_preds[task])
                label_concat = np.concatenate(all_labels[task])

                # Average Precision
                ap = average_precision_score(label_concat, pred_concat)
                task_metrics['AveragePrecision'][task] = ap

                # AUC
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

        # Check if we should reduce learning rate
        patience = lr_config.get('patience', 3)
        if len(self.val_loss_history) > patience:
            recent_losses = self.val_loss_history[-patience:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                # Learning rate reduction
                factor = lr_config.get('factor', 0.5)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= factor
                print(f"Reduced learning rate to {get_lr(self.optimizer):.6f}")

        # Track best validation loss
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
        # Create weight directory if it doesn't exist
        weight_dir = self.checkpoint_dir
        os.makedirs(weight_dir, exist_ok=True)

        # Save with model name and epoch
        model_name = self.config['model']['name']
        checkpoint_path = os.path.join(
            weight_dir,
            f'{model_name}_{epoch}ep.pt'
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    def log_epoch_metrics(self, epoch, train_loss, val_metrics):
        """Log metrics for current epoch."""
        # Log to CSV
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
            val_loss=val_metrics['val']['loss'],
            lr=get_lr(self.optimizer),
        )

        # Log to wandb
        wandb_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_metrics['val']['loss'],
            'learning_rate': get_lr(self.optimizer),
        }

        # Log task-specific metrics to wandb
        for split, metrics_dict in val_metrics.items():
            for metric_name, task_metrics in metrics_dict['metrics'].items():
                for task, value in task_metrics.items():
                    wandb_log[f'{split}/{metric_name}/{task}'] = value

        wandb.log(wandb_log, step=epoch)

        # Flush meters
        self.epoch_meter.flush()
        self.val_meter.flush()
        self.iter_meter.flush()

        # Print summary
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val']['loss']:.4f}")
        print(f"  Learning Rate: {get_lr(self.optimizer):.6f}")

    def log_test_metrics(self, test_metrics):
        """Log final test metrics."""
        print("\nTest Set Results:")

        # Log test metrics to wandb
        test_log = {}
        for metric_name, task_metrics in test_metrics['test']['metrics'].items():
            print(f"\n{metric_name}:")
            for task, value in task_metrics.items():
                print(f"  {task}: {value:.4f}")
                test_log[f'test/{metric_name}/{task}'] = value

        wandb.log(test_log)


def main():
    parser = argparse.ArgumentParser(
        description='Train CXR classification model with config file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='+',
        help='Override config values (e.g., model.name=BioViL training.batch_size=32)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='Resume training from latest checkpoint if available (default: true)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume and start fresh from pretrained weights'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save/load checkpoints (default: /mnt/HDD/yoonji/mrg/cxr_classification/weight)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply overrides if provided
    if args.override:
        for override in args.override:
            key_path, value = override.split('=')
            keys = key_path.split('.')

            # Navigate to the nested dictionary
            d = config
            for key in keys[:-1]:
                d = d[key]

            # Set the value (try to infer type)
            try:
                d[keys[-1]] = eval(value)
            except:
                d[keys[-1]] = value

    # Determine resume mode
    resume = (args.resume.lower() == 'true') and (not args.no_resume)

    # Create trainer and start training
    trainer = Trainer(config, resume=resume, checkpoint_dir=args.checkpoint_dir)
    trainer.train()


if __name__ == '__main__':
     main()
