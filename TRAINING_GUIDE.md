# CXR Classification Training Framework - User Guide

A flexible, configuration-based training framework for chest X-ray classification that supports multiple pretrained models including BioViL, MedKLIP, DenseNet, and ResNet.

## Features

- **Multiple Model Support**: Easily switch between BioViL, MedKLIP, DenseNet, and ResNet architectures
- **Config-Based Training**: All hyperparameters and settings in YAML config files
- **Model-Specific Transforms**: Automatic preprocessing based on model requirements
- **Flexible Data Handling**: Support for official splits and cross-validation
- **Advanced Training**: Mixed precision, distributed training, learning rate scheduling
- **Comprehensive Logging**: Automatic metric tracking and checkpoint saving

## Installation

```bash
pip install torch torchvision transformers scikit-learn pyyaml tqdm pandas pillow
```

## Quick Start

### 1. Prepare Configuration

Create or modify a config file (e.g., `config.yaml`):

```yaml
model:
  name: "BioViL"  # Options: BioViL, MedKLIP, densenet121, resnet50
  pretrained: true
  num_classes: 14

data:
  datadir: "/path/to/MIMIC-CXR-JPG"
  split_type: "cv"
  fold: 0

training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 1e-3

output:
  dir: "./outputs/biovil_experiment"
```

### 2. Run Training

```bash
python train_config.py --config config.yaml
```

### 3. Override Config Values (Optional)

```bash
python train_config.py --config config.yaml \
    --override model.name=densenet121 training.batch_size=32
```

## Configuration Guide

### Model Configuration

```yaml
model:
  name: "BioViL"  # Model architecture
  pretrained: true  # Use pretrained weights
  num_classes: 14  # Number of output classes
  freeze_backbone: false  # Freeze pretrained layers
```

**Supported Models:**
- `BioViL`: Microsoft's BiomedVLP-BioViL-T
- `MedKLIP`: Medical Knowledge-enhanced Language-Image Pre-training
- `densenet121`, `densenet161`, `densenet169`, `densenet201`
- `resnet50`, `resnet101`

### Data Configuration

```yaml
data:
  datadir: "/path/to/MIMIC-CXR-JPG"
  image_subdir: "files"
  label_method: "zeros_uncertain_nomask"  # Label handling strategy

  # Split configuration
  split_type: "cv"  # or "official"
  num_folds: 10
  fold: 0
  val_size: 0.1
```

**Label Methods:**
- `ignore_uncertain`: Ignore uncertain labels (U-Ignore)
- `zeros_uncertain`: Treat uncertain as negative (U-Zeros)
- `ones_uncertain`: Treat uncertain as positive (U-Ones)
- `zeros_uncertain_nomask`: U-Zeros without masking
- `ones_uncertain_nomask`: U-Ones without masking
- `three_class`: Multi-class with uncertain category
- `four_class`: Four classes including missing
- `missing_neg`: Treat missing as negative

### Training Configuration

```yaml
training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  optimizer: "adam"  # or "sgd"
  weight_decay: 0.0

  # Learning rate scheduling
  lr_scheduler:
    enabled: true
    patience: 3  # Epochs before reducing LR
    factor: 0.5  # LR reduction factor
    early_stop_patience: 10  # Early stopping patience

  # Advanced options
  amp: false  # Automatic mixed precision
  distributed: false  # Multi-node training
```

### Augmentation Configuration

```yaml
augmentation:
  horizontal_flip: 0.5  # Probability of horizontal flip
  rotation_degrees: 20  # Random rotation range
  random_crop: false
  color_jitter: false
```

## Project Structure

```
cxr_classification/
├── config.yaml              # Main configuration file
├── train_config.py          # Training script
├── model_factory.py         # Model creation factory
├── data_transforms.py       # Model-specific transforms
├── utils.py                 # Utility functions
├── mimic_cxr_jpg.py         # Dataset handling
├── meters.py                # Metric logging
└── configs/                 # Example configs
    ├── biovil_config.yaml
    ├── medklip_config.yaml
    └── densenet_config.yaml
```

## Example Workflows

### Training BioViL

```bash
python train_config.py --config configs/biovil_config.yaml
```

### Training DenseNet with Custom Settings

```bash
python train_config.py --config configs/densenet_config.yaml \
    --override training.learning_rate=5e-4 training.batch_size=128
```

### Cross-Validation Training

Run multiple folds:
```bash
for fold in {0..9}; do
    python train_config.py --config config.yaml \
        --override data.fold=$fold output.dir=./outputs/fold_$fold
done
```

### Fine-tuning with Frozen Backbone

```yaml
model:
  name: "BioViL"
  pretrained: true
  freeze_backbone: true  # Only train classification head
```

## Output Structure

Training outputs are saved to the specified output directory:

```
outputs/experiment_1/
├── config.yaml              # Saved configuration
├── epoch_metrics.csv        # Epoch-level metrics
├── val_metrics.csv          # Validation metrics per task
├── iter_metrics.csv         # Iteration-level metrics
├── model_epoch0.pt          # Saved checkpoints
├── model_epoch1.pt
└── ...
```

## Model-Specific Notes

### BioViL
- Requires RGB input (grayscale images are automatically converted)
- Uses ImageNet normalization
- 768-dimensional image embeddings

### MedKLIP
- Designed for medical imaging
- May work with grayscale images directly
- Requires MedKLIP module in `model/` directory

### DenseNet/ResNet
- Adapted for single-channel grayscale input
- First conv layer initialized from pretrained RGB weights (averaged)
- Uses MIMIC-CXR specific normalization

## Tips

1. **Start with a small learning rate** (1e-4 to 1e-3) when fine-tuning pretrained models
2. **Use learning rate scheduling** to improve convergence
3. **Enable AMP** (`training.amp: true`) to reduce memory and speed up training
4. **Freeze backbone initially** when data is limited, then unfreeze for additional epochs
5. **Use cross-validation** for robust performance estimates

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size`
- Enable `amp: true`
- Use gradient accumulation (not yet implemented)

**Poor Performance:**
- Check label method matches your use case
- Verify data augmentation isn't too aggressive
- Ensure correct normalization for your model
- Try different learning rates

**Model Not Loading:**
- Verify model name spelling
- Check that pretrained weights are accessible
- For BioViL, ensure transformers library is installed
- For MedKLIP, ensure module is in the correct directory
