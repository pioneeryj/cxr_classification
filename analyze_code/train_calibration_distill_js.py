"""
Training script with calibration distillation using Jensen-Shannon divergence.

Post-hoc calibration을 training time에 적용:
1. Warmup 후, 매 N epoch마다 validation set에서 label-wise T와 global T 계산
2. 각 label에 대해 NLL 개선량 비교하여 더 나은 T 선택
3. 선택된 T로 calibrated probability 계산
4. Jensen-Shannon divergence loss로 모델이 calibrated distribution을 학습하도록 distillation

Loss = BCE + alpha * JSD(calibrated_prob || model_prob)
JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

Usage:
    python train_calibration_distill_js.py --config configs/densenet121_config.yaml
"""

import os
import argparse
import json
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
from model_factory import get_model, apply_lora
from data_transforms import get_train_transform, get_val_transform
from utils import load_config, set_seed, setup_output_dir, save_config, count_parameters, get_lr


CHEXPERT_LABELS = mimic_cxr_jpg.chexpert_labels


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


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self, mode='global', num_classes=14):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        if mode == 'global':
            self.temperature = nn.Parameter(torch.ones(1))
        elif mode == 'label_wise':
            self.temperature = nn.Parameter(torch.ones(num_classes))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, logits):
        if self.mode == 'global':
            return logits / self.temperature
        else:
            return logits / self.temperature.unsqueeze(0)

    def fit(self, logits, labels, masks, lr=0.01, max_iter=50):
        """Fit temperature using NLL loss."""
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
        else:
            return self.temperature.detach().cpu().numpy()


class CalibrationDistillJSLoss(nn.Module):
    """
    Calibration distillation loss using Jensen-Shannon divergence.

    JSD is a symmetric divergence: JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = 0.5*(P+Q)
    Distills post-hoc calibrated probabilities to model output.
    """

    def __init__(self, num_classes=14, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        # Temperature per label (will be updated periodically)
        self.register_buffer('temperatures', torch.ones(num_classes))

        # Track which T type is used per label: 0=global, 1=label_wise
        self.register_buffer('t_type', torch.zeros(num_classes, dtype=torch.long))

        # Global temperature (scalar, replicated for convenience)
        self.register_buffer('t_global', torch.ones(1))

    def update_temperatures(self, t_global, t_label_wise, use_label_wise_mask):
        """
        Update temperatures based on decision.

        Args:
            t_global: scalar, global temperature
            t_label_wise: [num_classes], per-label temperatures
            use_label_wise_mask: [num_classes] bool, True if label-wise T is better
        """
        self.t_global = torch.tensor([t_global], device=self.temperatures.device)

        t_label_wise = torch.tensor(t_label_wise, device=self.temperatures.device)
        use_mask = torch.tensor(use_label_wise_mask, device=self.temperatures.device)

        # Select T based on mask
        self.temperatures = torch.where(use_mask, t_label_wise, self.t_global.expand(self.num_classes))
        self.t_type = use_mask.long()

    def forward(self, logits, targets, mask):
        """
        Compute Jensen-Shannon divergence loss between calibrated probs and model probs.

        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

        For binary distributions per label:
        KL(p||m) = p*log(p/m) + (1-p)*log((1-p)/(1-m))

        Args:
            logits: [B, C] raw model logits
            targets: [B, C] binary labels (not used directly, but kept for interface)
            mask: [B, C] valid label mask

        Returns:
            jsd_loss: scalar
        """
        # Model probabilities
        p_model = torch.sigmoid(logits)

        # Calibrated probabilities (teacher)
        temperatures = self.temperatures.to(logits.device).unsqueeze(0)  # [1, C]
        scaled_logits = logits / temperatures
        p_calibrated = torch.sigmoid(scaled_logits)

        # Jensen-Shannon divergence for binary distribution
        eps = 1e-7
        p_cal = p_calibrated.clamp(eps, 1 - eps)
        p_mod = p_model.clamp(eps, 1 - eps)

        # Mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (p_cal + p_mod)
        m = m.clamp(eps, 1 - eps)

        # KL(P || M) for binary: p_cal * log(p_cal/m) + (1-p_cal) * log((1-p_cal)/(1-m))
        kl_p_m = p_cal * (torch.log(p_cal) - torch.log(m)) + \
                 (1 - p_cal) * (torch.log(1 - p_cal) - torch.log(1 - m))

        # KL(Q || M) for binary: p_mod * log(p_mod/m) + (1-p_mod) * log((1-p_mod)/(1-m))
        kl_q_m = p_mod * (torch.log(p_mod) - torch.log(m)) + \
                 (1 - p_mod) * (torch.log(1 - p_mod) - torch.log(1 - m))

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        jsd = 0.5 * kl_p_m + 0.5 * kl_q_m

        # Apply mask and compute mean
        masked_jsd = jsd * mask

        if mask.sum() > 0:
            loss = self.alpha * masked_jsd.sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss


class TrainerCalibrationDistillJS:
    """Trainer with calibration distillation using Jensen-Shannon divergence."""

    def __init__(self, config, resume=True, checkpoint_dir=None, load_checkpoint=None, start_epoch=None):
        self.config = config
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir or '/mnt/HDD/yoonji/mrg/cxr_classification/weight'
        self.load_checkpoint = load_checkpoint
        self.forced_start_epoch = start_epoch

        # Calibration settings
        self.calibration_config = config.get('calibration', {})
        self.warmup_epochs = self.calibration_config.get('warmup_epochs', 5)
        self.t_update_freq = self.calibration_config.get('t_update_freq', 5)
        self.alpha_jsd = self.calibration_config.get('alpha_jsd', 1.0)
        self.distill_lr_factor = self.calibration_config.get('distill_lr_factor', 0.1)  # LR reduction when distillation starts
        self.jsd_rampup_epochs = self.calibration_config.get('jsd_rampup_epochs', 5)  # Epochs to ramp up JSD loss
        self.current_alpha_jsd = 0.0  # Will be updated during training

        # Device setup
        device_str = config['system']['device']
        if 'cuda' in device_str and not torch.cuda.is_available():
            print("WARNING: CUDA not available. Falling back to CPU.")
            device_str = 'cpu'
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

        set_seed(config['system']['seed'])

        self.output_dir = setup_output_dir(config['output']['dir'])
        save_config(config, self.output_dir)

        wandb.init(
            project="cxr-classification",
            name=f"{config['model']['name']}_calib_distill_js_fold{config['data'].get('fold', 0)}",
            config=config,
            dir=self.output_dir,
            tags=['calibration_distill', 'jensen_shannon'],
        )
        print(f"Weights & Biases run: {wandb.run.name}")

        self.model = self._setup_model()
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        # Loss functions
        self.criterion_bce = self._setup_criterion()
        self.criterion_jsd = CalibrationDistillJSLoss(
            num_classes=len(CHEXPERT_LABELS),
            alpha=self.alpha_jsd,
        )

        self.optimizer = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['training']['amp'])

        # Metrics tracking
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

        # Temperature history
        self.temperature_history = []

        # Load checkpoint: priority is load_checkpoint > resume
        if self.load_checkpoint:
            self._load_specific_checkpoint()
        elif self.resume:
            self._try_resume_from_checkpoint()

    def _setup_model(self):
        """Setup model."""
        model = get_model(self.config['model'])
        model = apply_lora(model, self.config['model'])

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
        """Setup data loaders."""
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

        # Determine train sampler
        use_sqrt_resampling = train_config.get('sqrt_resampling', False)
        if use_distributed:
            train_sampler = DistributedSampler(train_ds, shuffle=True)
        elif use_sqrt_resampling:
            train_sampler = mimic_cxr_jpg.get_sqrt_resampling_sampler(train_ds)
            print(f"Using square root resampling sampler (num_samples={len(train_ds)})")
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_ds,
            batch_size=train_config['batch_size'],
            shuffle=(train_sampler is None),
            num_workers=self.config['system']['num_workers'],
            pin_memory=self.config['system']['pin_memory'],
            sampler=train_sampler,
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

        # Save train dataset for pos_weight computation
        self.train_ds = train_ds

        return train_loader, val_loader, test_loader

    def _get_split_info(self):
        """Generate split info string for pos_weight cache filename."""
        data_config = self.config['data']
        split_type = data_config['split_type']

        prefix_filter = data_config.get('subject_prefix_filter', None)
        if prefix_filter is not None:
            prefix_str = '-'.join(str(p) for p in sorted(prefix_filter))
        else:
            prefix_str = 'all'

        if split_type == 'cv':
            fold = data_config.get('fold', 0)
            num_folds = data_config.get('num_folds', 10)
            return f"cv_fold{fold}_nfolds{num_folds}_p{prefix_str}"
        else:
            return f"official_p{prefix_str}"

    def _compute_pos_weight(self):
        """Compute pos_weight from training dataset DataFrame (with caching)."""
        cache_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        split_info = self._get_split_info()
        return mimic_cxr_jpg.compute_pos_weight(
            self.train_ds,
            cache_dir=cache_dir,
            split_info=split_info,
        )

    def _setup_criterion(self):
        """Setup BCE loss criterion based on configuration."""
        use_pos_weight = self.config['training'].get('pos_weight', False)

        if use_pos_weight:
            pos_weight = self._compute_pos_weight().to(self.device)
            criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
            print(f"\nUsing BCEWithLogitsLoss with pos_weight")
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            print(f"\nUsing BCEWithLogitsLoss without pos_weight")

        return criterion

    def _setup_optimizer(self):
        """Setup optimizer."""
        train_config = self.config['training']
        optimizer_name = train_config['optimizer'].lower()

        lr = float(train_config['learning_rate'])
        weight_decay = float(train_config.get('weight_decay', 0.0))

        params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise RuntimeError("No trainable parameters found.")

        print(f"\nOptimizer: {optimizer_name}, LR: {lr}")

        if optimizer_name == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _load_specific_checkpoint(self):
        """Load a specific checkpoint file."""
        checkpoint_path = self.load_checkpoint

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n" + "=" * 80)
        print("LOADING SPECIFIC CHECKPOINT")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")

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

        # Use forced start epoch if provided, otherwise extract from filename
        if self.forced_start_epoch is not None:
            self.start_epoch = self.forced_start_epoch
            print(f"Starting from epoch: {self.start_epoch} (forced)")
        else:
            # Try to extract epoch from filename (e.g., model_14ep.pt)
            match = re.search(r'_(\d+)ep\.pt$', checkpoint_path)
            if match:
                self.start_epoch = int(match.group(1)) + 1
                print(f"Starting from epoch: {self.start_epoch} (extracted from filename)")
            else:
                self.start_epoch = 0
                print(f"Starting from epoch: {self.start_epoch} (default)")

        print("=" * 80 + "\n")

    def _try_resume_from_checkpoint(self):
        """Try to resume training from the latest checkpoint."""
        weight_dir = self.checkpoint_dir
        model_name = self.config['model']['name']

        checkpoint_path, last_epoch = find_latest_checkpoint(weight_dir, model_name)

        if checkpoint_path is None:
            print(f"\nNo checkpoint found for {model_name} in {weight_dir}")
            print("Starting training from epoch 0")
            return

        print(f"\n" + "=" * 80)
        print("RESUMING TRAINING")
        print("=" * 80)
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
        print(f"Starting from epoch: {self.start_epoch}")
        print("=" * 80 + "\n")

    def _collect_validation_logits(self):
        """Collect logits and labels from validation set."""
        print("\nCollecting validation logits...")

        self.model.eval()
        all_logits = []
        all_labels = []
        all_masks = []

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

        self.model.train()
        return logits, labels, masks

    def _compute_nll(self, logits, labels, masks, temperatures, mode='label_wise'):
        """
        Compute NLL for given temperatures.

        Args:
            logits: [N, C]
            labels: [N, C]
            masks: [N, C]
            temperatures: scalar or [C]
            mode: 'global' or 'label_wise'

        Returns:
            nll_per_label: [C] NLL for each label
        """
        num_classes = logits.shape[1]
        nll_per_label = torch.zeros(num_classes)

        if mode == 'global':
            scaled_logits = logits / temperatures
        else:
            # temperatures: [C]
            scaled_logits = logits / temperatures.unsqueeze(0)

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_per_sample = criterion(scaled_logits, labels)  # [N, C]

        for i in range(num_classes):
            mask_i = masks[:, i] == 1
            if mask_i.sum() > 0:
                nll_per_label[i] = loss_per_sample[mask_i, i].mean()
            else:
                nll_per_label[i] = float('inf')

        return nll_per_label

    def _fit_and_select_temperatures(self):
        """
        Fit global and label-wise temperatures, select better one per label.

        Returns:
            final_temperatures: [num_classes]
            use_label_wise: [num_classes] bool mask
            t_global: scalar
            t_label_wise: [num_classes]
        """
        print("\n" + "=" * 80)
        print("Fitting Temperature Scaling")
        print("=" * 80)

        logits, labels, masks = self._collect_validation_logits()
        num_classes = logits.shape[1]

        # Compute baseline NLL (T=1)
        nll_baseline = self._compute_nll(logits, labels, masks, torch.tensor(1.0), mode='global')

        # Fit global temperature
        print("\nFitting global temperature...")
        calibrator_global = TemperatureScaling(mode='global', num_classes=num_classes)
        t_global = calibrator_global.fit(logits, labels, masks)
        print(f"  Global T: {t_global:.4f}")

        nll_global = self._compute_nll(logits, labels, masks, torch.tensor(t_global), mode='global')

        # Fit label-wise temperatures
        print("\nFitting label-wise temperatures...")
        calibrator_label_wise = TemperatureScaling(mode='label_wise', num_classes=num_classes)
        t_label_wise = calibrator_label_wise.fit(logits, labels, masks)

        nll_label_wise = self._compute_nll(
            logits, labels, masks,
            torch.tensor(t_label_wise),
            mode='label_wise'
        )

        # Compare and select
        print("\n" + "-" * 80)
        print(f"{'Label':<30} {'T_global':>10} {'T_label':>10} {'NLL_base':>10} {'NLL_glob':>10} {'NLL_label':>10} {'Select':>10}")
        print("-" * 80)

        use_label_wise = np.zeros(num_classes, dtype=bool)
        final_temperatures = np.zeros(num_classes)

        for i, label in enumerate(CHEXPERT_LABELS):
            # Calculate improvements
            improve_global = nll_baseline[i] - nll_global[i]
            improve_label = nll_baseline[i] - nll_label_wise[i]

            # Select the one with better improvement (lower NLL)
            if nll_label_wise[i] < nll_global[i]:
                use_label_wise[i] = True
                final_temperatures[i] = t_label_wise[i]
                select = "Label"
            else:
                use_label_wise[i] = False
                final_temperatures[i] = t_global
                select = "Global"

            print(f"{label:<30} {t_global:>10.4f} {t_label_wise[i]:>10.4f} "
                  f"{nll_baseline[i]:>10.4f} {nll_global[i]:>10.4f} {nll_label_wise[i]:>10.4f} "
                  f"{select:>10}")

        print("-" * 80)
        print(f"Labels using Global T: {(~use_label_wise).sum()}")
        print(f"Labels using Label-wise T: {use_label_wise.sum()}")
        print("=" * 80)

        # Save temperature info
        temp_info = {
            'epoch': self.current_epoch,
            't_global': float(t_global),
            't_label_wise': {label: float(t_label_wise[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            'final_temperatures': {label: float(final_temperatures[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            'use_label_wise': {label: bool(use_label_wise[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            'nll_baseline': {label: float(nll_baseline[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            'nll_global': {label: float(nll_global[i]) for i, label in enumerate(CHEXPERT_LABELS)},
            'nll_label_wise': {label: float(nll_label_wise[i]) for i, label in enumerate(CHEXPERT_LABELS)},
        }
        self.temperature_history.append(temp_info)

        # Save to file
        with open(os.path.join(self.output_dir, 'temperature_history.json'), 'w') as f:
            json.dump(self.temperature_history, f, indent=2)

        # Log to wandb
        wandb.log({
            'calibration/t_global': t_global,
            'calibration/num_label_wise': int(use_label_wise.sum()),
            'calibration/num_global': int((~use_label_wise).sum()),
        }, step=self.current_epoch)

        return final_temperatures, use_label_wise, t_global, t_label_wise

    def _update_calibration(self):
        """Update calibration temperatures."""
        final_t, use_label_wise, t_global, t_label_wise = self._fit_and_select_temperatures()

        # Update JSD loss with new temperatures
        self.criterion_jsd.update_temperatures(t_global, t_label_wise, use_label_wise)

        print("\nCalibration temperatures updated for JSD distillation.")

    def _reduce_lr_for_distillation(self):
        """Reduce learning rate when distillation starts."""
        old_lr = get_lr(self.optimizer)
        new_lr = old_lr * self.distill_lr_factor

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"\n[Distillation Start] Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f} (factor={self.distill_lr_factor})")

    def _get_jsd_rampup_weight(self, epoch):
        """
        Get JSD loss weight with linear ramp-up.

        Args:
            epoch: Current epoch

        Returns:
            Current alpha_jsd value (ramped up)
        """
        if epoch < self.warmup_epochs:
            return 0.0

        epochs_since_distill = epoch - self.warmup_epochs

        if self.jsd_rampup_epochs <= 0:
            # No ramp-up, use full weight immediately
            return self.alpha_jsd

        if epochs_since_distill >= self.jsd_rampup_epochs:
            # Ramp-up complete
            return self.alpha_jsd

        # Linear ramp-up: 0 -> alpha_jsd over jsd_rampup_epochs
        rampup_ratio = (epochs_since_distill + 1) / self.jsd_rampup_epochs
        return self.alpha_jsd * rampup_ratio

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        show_progress = self.config['system']['progress_bar']

        print(f"\n" + "=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Starting from epoch: {self.start_epoch}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Output dir: {self.output_dir}")
        print(f"")
        print(f"  [Calibration Distillation - Jensen-Shannon]")
        print(f"  Warmup epochs (BCE only): {self.warmup_epochs}")
        print(f"  Temperature update frequency: every {self.t_update_freq} epochs")
        print(f"  Loss = BCE + alpha_jsd * JSD(calibrated || model)")
        print(f"  Alpha JSD (target): {self.alpha_jsd}")
        print(f"  JSD ramp-up epochs: {self.jsd_rampup_epochs}")
        print(f"  LR reduction at distillation start: x{self.distill_lr_factor}")
        print("=" * 80 + "\n")

        if self.start_epoch >= num_epochs:
            print(f"Training already completed ({self.start_epoch} >= {num_epochs} epochs)")
            return

        epoch_bar = range(self.start_epoch, num_epochs)
        if show_progress:
            epoch_bar = tqdm(epoch_bar, desc='Epochs', initial=self.start_epoch, total=num_epochs)

        for epoch in epoch_bar:
            self.current_epoch = epoch

            # Check if we should update calibration temperatures
            use_distillation = epoch >= self.warmup_epochs

            if use_distillation:
                # Update at warmup end and every t_update_freq epochs after
                epochs_since_warmup = epoch - self.warmup_epochs
                if epochs_since_warmup == 0:
                    # First distillation epoch: reduce learning rate
                    self._reduce_lr_for_distillation()
                    self._update_calibration()
                elif epochs_since_warmup % self.t_update_freq == 0:
                    self._update_calibration()

                # Update JSD weight with ramp-up
                self.current_alpha_jsd = self._get_jsd_rampup_weight(epoch)
                self.criterion_jsd.alpha = self.current_alpha_jsd

            # Train for one epoch
            train_loss, bce_loss, jsd_loss = self.train_epoch(use_distillation)

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            current_val_loss = val_metrics['val']['loss']
            self._update_learning_rate(current_val_loss)

            # Save checkpoint
            if (epoch + 1) % self.config['output']['save_frequency'] == 0:
                self.save_checkpoint(epoch)

            # Log metrics
            self.log_epoch_metrics(epoch, train_loss, bce_loss, jsd_loss, val_metrics)

            if self._should_early_stop():
                print(f"Early stopping at epoch {epoch}")
                break

        # Final test evaluation
        print("\nFinal evaluation on test set:")
        test_metrics = self.validate(use_test=True)
        self.log_test_metrics(test_metrics)

        wandb.finish()

    def train_epoch(self, use_distillation=False):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_jsd_loss = 0.0
        show_progress = self.config['system']['progress_bar']

        iter_bar = self.train_loader
        if show_progress:
            status = "BCE+JSD" if use_distillation else "BCE only"
            iter_bar = tqdm(iter_bar, desc=f'Epoch {self.current_epoch} ({status})', leave=False)

        for batch_idx, batch in enumerate(iter_bar):
            loss, bce_loss, jsd_loss = self.train_step(batch, use_distillation)

            if loss is not None:
                epoch_loss += loss
                epoch_bce_loss += bce_loss
                epoch_jsd_loss += jsd_loss if jsd_loss else 0

                self.iter_meter.update(
                    loss=loss,
                    bce_loss=bce_loss,
                    jsd_loss=jsd_loss if jsd_loss else 0,
                    epoch=self.current_epoch
                )

                wandb.log({
                    'train/loss_iter': loss,
                    'train/bce_loss_iter': bce_loss,
                    'train/jsd_loss_iter': jsd_loss if jsd_loss else 0,
                }, step=self.total_iters)

                if show_progress:
                    iter_bar.set_postfix({'loss': f'{loss:.4f}', 'bce': f'{bce_loss:.4f}'})

            self.total_iters += 1

        n_batches = len(self.train_loader)
        return epoch_loss / n_batches, epoch_bce_loss / n_batches, epoch_jsd_loss / n_batches

    def train_step(self, batch, use_distillation=False):
        """Single training step."""
        use_amp = self.config['training']['amp']

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = self.forward_batch(batch, use_distillation)

        if outputs is None:
            return None, None, None

        loss, bce_loss, jsd_loss = outputs

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), bce_loss.item(), jsd_loss.item() if jsd_loss is not None else None

    def forward_batch(self, batch, use_distillation=False):
        """Forward pass for a batch."""
        X, Y, Ymask = batch

        X = X.to(self.device).float()
        Y = Y.to(self.device).float()
        Ymask = Ymask.to(self.device)

        weightsum = Ymask.sum().item()
        if weightsum == 0:
            return None

        # Forward pass
        logits = self.model(X)

        # BCE Loss
        bce = self.criterion_bce(logits, Y)
        bce_loss = (bce * Ymask).sum() / weightsum

        # JSD Distillation Loss (after warmup)
        if use_distillation:
            jsd_loss = self.criterion_jsd(logits, Y, Ymask)
            loss = bce_loss + jsd_loss
        else:
            jsd_loss = None
            loss = bce_loss

        return loss, bce_loss, jsd_loss

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
            if all(recent_losses[i] >= recent_losses[i - 1] for i in range(1, len(recent_losses))):
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
        checkpoint_path = os.path.join(weight_dir, f'{model_name}_calib_distill_js_{epoch}ep.pt')

        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'temperatures': self.criterion_jsd.temperatures.cpu().tolist(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    def log_epoch_metrics(self, epoch, train_loss, bce_loss, jsd_loss, val_metrics):
        """Log metrics for current epoch."""
        use_distillation = epoch >= self.warmup_epochs

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
            jsd_loss=jsd_loss,
            val_loss=val_metrics['val']['loss'],
            lr=get_lr(self.optimizer),
            use_distillation=use_distillation,
        )

        wandb_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/bce_loss': bce_loss,
            'train/jsd_loss': jsd_loss,
            'train/alpha_jsd': self.current_alpha_jsd if use_distillation else 0.0,
            'train/use_distillation': int(use_distillation),
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

        if use_distillation:
            status = f"YES (alpha_jsd={self.current_alpha_jsd:.3f})"
        else:
            status = "NO (warmup)"
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (BCE: {bce_loss:.4f}, JSD: {jsd_loss:.4f})")
        print(f"  Val Loss: {val_metrics['val']['loss']:.4f}")
        print(f"  Distillation: {status}")
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
        description='Train CXR model with calibration distillation (Jensen-Shannon divergence)'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs (BCE only)')
    parser.add_argument('--t_update_freq', type=int, default=5,
                        help='Temperature update frequency (epochs)')
    parser.add_argument('--alpha_jsd', type=float, default=1.0, help='JSD distillation loss weight')
    parser.add_argument('--distill_lr_factor', type=float, default=0.1,
                        help='LR reduction factor when distillation starts (default: 0.1)')
    parser.add_argument('--jsd_rampup_epochs', type=int, default=5,
                        help='Epochs to ramp up JSD loss from 0 to alpha_jsd (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Total training epochs (overrides config)')
    parser.add_argument('--resume', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to specific checkpoint to load (overrides auto-resume)')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='Starting epoch (use with --load-checkpoint to skip warmup)')
    parser.add_argument('--pos_weight', type=str, default=None, choices=['true', 'false'],
                        help='Enable pos_weight for BCEWithLogitsLoss (overrides config)')
    parser.add_argument('--override', type=str, nargs='+',
                        help='Override config values (e.g., training.num_epochs=30 output.dir=outputs/exp1)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides if provided
    if args.override:
        for override in args.override:
            key_path, value = override.split('=')
            keys = key_path.split('.')
            d = config
            for key in keys[:-1]:
                d = d[key]
            if value.lower() == 'true':
                d[keys[-1]] = True
            elif value.lower() == 'false':
                d[keys[-1]] = False
            else:
                try:
                    d[keys[-1]] = eval(value)
                except:
                    d[keys[-1]] = value

    # Override pos_weight if specified via CLI
    if args.pos_weight is not None:
        config['training']['pos_weight'] = (args.pos_weight.lower() == 'true')

    config['calibration'] = {
        'warmup_epochs': args.warmup_epochs,
        't_update_freq': args.t_update_freq,
        'alpha_jsd': args.alpha_jsd,
        'distill_lr_factor': args.distill_lr_factor,
        'jsd_rampup_epochs': args.jsd_rampup_epochs,
    }

    # num_epochs를 command line에서 직접 지정한 경우 config 덮어쓰기
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs

    resume = (args.resume.lower() == 'true') and (not args.no_resume)

    trainer = TrainerCalibrationDistillJS(
        config,
        resume=resume,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
        start_epoch=args.start_epoch,
    )
    trainer.train()


if __name__ == '__main__':
    main()
