"""Factory for creating and loading pretrained medical imaging models."""

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class ModelFactory:
    """Factory class for creating medical imaging models."""

    @staticmethod
    def create_model(config):
        """Create a model based on configuration.

        Args:
            config: Dictionary containing model configuration

        Returns:
            PyTorch model
        """
        model_name = config['name'].lower()
        num_classes = config.get('num_classes', 14)
        pretrained = config.get('pretrained', True)
        freeze_backbone = config.get('freeze_backbone', False)

        if model_name == 'biovil':
            model = ModelFactory._create_biovil(num_classes, pretrained, freeze_backbone)
        elif model_name == 'medklip':
            model = ModelFactory._create_medklip(num_classes, pretrained, freeze_backbone)
        elif model_name == 'medclip':
            model = ModelFactory._create_medclip(num_classes, pretrained, freeze_backbone)
        elif model_name == 'biomedclip':
            model = ModelFactory._create_biomedclip(num_classes, pretrained, freeze_backbone)
        elif 'densenet' in model_name:
            model = ModelFactory._create_densenet(model_name, num_classes, pretrained, freeze_backbone)
        elif 'resnet' in model_name:
            model = ModelFactory._create_resnet(model_name, num_classes, pretrained, freeze_backbone)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    @staticmethod
    def _create_biovil(num_classes, pretrained, freeze_backbone):
        """Create BioViL model using the proper MultiImageModel architecture."""
        if not pretrained:
            raise NotImplementedError("BioViL only supports pretrained initialization")

        from health_multimodal.image.model.model import MultiImageModel
        from health_multimodal.image.model.types import ImageEncoderType

        # Load BioViL-T image encoder checkpoint
        ckpt_path = "/home/yoonji/mrg/cxr_classification/pretrained_weight/biovil_image_model.pt"

        # Create multi-image encoder + projector model
        biovil_encoder = MultiImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE.value,
            joint_feature_size=128,
            pretrained_model_path=ckpt_path,
        )

        print(f"Loaded BioViL-T image encoder from: {ckpt_path}")

        class BioViLClassifier(nn.Module):
            def __init__(self, encoder, num_classes, use_projected_128=False, freeze_encoder=False):
                super().__init__()
                self.encoder = encoder  # MultiImageModel
                self.use_projected_128 = use_projected_128

                if freeze_encoder:
                    for p in self.encoder.parameters():
                        p.requires_grad = False

                # Determine feature dimension
                # - projected_global_embedding: [B, 128]
                # - img_embedding: ResNet pooled features (actual dimension varies)
                if use_projected_128:
                    in_dim = 128
                else:
                    # Infer dimension BEFORE optimizer creation (not lazy)
                    with torch.no_grad():
                        device = next(self.encoder.parameters()).device
                        dummy = torch.zeros(2, 3, 224, 224, device=device)
                        out = self.encoder(current_image=dummy, previous_image=None)
                        in_dim = out.img_embedding.shape[-1]

                self.classifier = nn.Linear(in_dim, num_classes)

                # Ensure classifier is trainable
                for p in self.classifier.parameters():
                    p.requires_grad = True

                print(f"BioViL classifier created:")
                print(f"  - Use projected features (128-dim): {use_projected_128}")
                print(f"  - Freeze encoder: {freeze_encoder}")
                print(f"  - Classifier input dim: {in_dim}")
                print(f"  - Classifier requires_grad: {all(p.requires_grad for p in self.classifier.parameters())}")

            def forward(self, x, x_prev=None):
                """
                Forward pass through BioViL encoder and classifier.

                Args:
                    x: Current image [B, 3, H, W]
                    x_prev: Previous image [B, 3, H, W] or None

                Returns:
                    logits: [B, num_classes]
                """
                # Get features from encoder
                out = self.encoder(current_image=x, previous_image=x_prev)

                # Choose which features to use
                if self.use_projected_128:
                    feats = out.projected_global_embedding
                else:
                    feats = out.img_embedding

                # Classify
                logits = self.classifier(feats)
                return logits

        # Create classifier model
        # use_projected_128=False means we use the full ResNet features before projection
        model = BioViLClassifier(
            biovil_encoder,
            num_classes,
            use_projected_128=False,
            freeze_encoder=freeze_backbone
        )

        return model

    @staticmethod
    def _create_medklip(num_classes, pretrained, freeze_backbone):
        """Create MedKLIP model for classification."""
        if not pretrained:
            raise NotImplementedError("MedKLIP only supports pretrained initialization")

        # Load pretrained checkpoint
        ckpt_path = "/home/yoonji/mrg/cxr_classification/pretrained_weight/MedKLIP_checkpoint_final.pt"
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        print(f"Loading MedKLIP checkpoint from: {ckpt_path}")

        class MedKLIPImageEncoder(nn.Module):
            """Simplified MedKLIP model that uses only the image encoder."""
            def __init__(self, num_classes, freeze_encoder=False):
                super().__init__()

                # Create ResNet50 backbone (same as MedKLIP)
                resnet = models.resnet50(pretrained=False)
                # MedKLIP uses up to layer3 (before layer4)
                # Keep 3-channel input to match pretrained weights
                self.res_features = nn.Sequential(*list(resnet.children())[:-3])

                # MedKLIP's projection layers
                num_ftrs = 1024  # ResNet50 layer3 output: 1024 channels
                self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
                self.res_l2 = nn.Linear(num_ftrs, 256)  # d_model=256 in MedKLIP

                # Global average pooling
                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

                # Classification head
                self.classifier = nn.Linear(256, num_classes)

                if freeze_encoder:
                    for param in self.res_features.parameters():
                        param.requires_grad = False
                    for param in self.res_l1.parameters():
                        param.requires_grad = False
                    for param in self.res_l2.parameters():
                        param.requires_grad = False

                print("MedKLIP classifier created:")
                print(f"  - Freeze encoder: {freeze_encoder}")
                print(f"  - Feature dimension: 256")
                print(f"  - Number of classes: {num_classes}")

            def forward(self, x):
                """
                Forward pass through MedKLIP encoder and classifier.

                Args:
                    x: Input image [B, 3, H, W] (grayscale repeated to RGB)

                Returns:
                    logits: [B, num_classes]
                """
                # Extract features with ResNet backbone
                x = self.res_features(x)  # [B, 1024, H/8, W/8]

                # Global pooling
                x = self.global_pool(x)  # [B, 1024, 1, 1]
                x = x.squeeze(-1).squeeze(-1)  # [B, 1024]

                # MedKLIP projection layers
                x = self.res_l1(x)
                x = F.relu(x)
                x = self.res_l2(x)  # [B, 256]

                # Classification
                logits = self.classifier(x)  # [B, num_classes]

                return logits

        # Create model
        model = MedKLIPImageEncoder(num_classes, freeze_encoder=freeze_backbone)

        # Load pretrained weights
        state_dict = checkpoint['model']

        # Filter and load only the image encoder weights
        model_dict = model.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            # Remove 'module.' prefix if present (from DataParallel)
            k_clean = k.replace('module.', '')

            # Map MedKLIP weights to our simplified model
            if k_clean.startswith('res_features'):
                pretrained_dict[k_clean] = v
            elif k_clean.startswith('res_l1'):
                pretrained_dict[k_clean] = v
            elif k_clean.startswith('res_l2'):
                pretrained_dict[k_clean] = v

        # Update model with pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"Loaded {len(pretrained_dict)} layers from pretrained MedKLIP checkpoint")

        return model

    @staticmethod
    def _create_medclip(num_classes, pretrained, freeze_backbone):
        """Create MedCLIP model for classification.

        Uses flax-community/medclip-roco pretrained weights converted to PyTorch.
        """
        if not pretrained:
            raise NotImplementedError("MedCLIP only supports pretrained initialization")

        from transformers import VisionTextDualEncoderModel

        # Load pretrained MedCLIP model (Flax weights auto-converted to PyTorch)
        model_name = "flax-community/medclip-roco"
        print(f"Loading MedCLIP from HuggingFace: {model_name}")

        # VisionTextDualEncoderModel can load FlaxHybridCLIP weights
        clip_model = VisionTextDualEncoderModel.from_pretrained(model_name, from_flax=True)

        class MedCLIPClassifier(nn.Module):
            """MedCLIP model wrapper for image classification."""
            def __init__(self, clip_model, num_classes, freeze_encoder=False):
                super().__init__()
                self.vision_model = clip_model.vision_model
                self.visual_projection = clip_model.visual_projection

                # MedCLIP projection dimension
                projection_dim = clip_model.config.projection_dim  # typically 512

                # Classification head
                self.classifier = nn.Linear(projection_dim, num_classes)

                if freeze_encoder:
                    for param in self.vision_model.parameters():
                        param.requires_grad = False
                    for param in self.visual_projection.parameters():
                        param.requires_grad = False

                print("MedCLIP classifier created:")
                print(f"  - Freeze encoder: {freeze_encoder}")
                print(f"  - Projection dimension: {projection_dim}")
                print(f"  - Number of classes: {num_classes}")

            def forward(self, x):
                """
                Forward pass through MedCLIP vision encoder and classifier.

                Args:
                    x: Input image [B, 3, H, W] (RGB format expected by CLIP)

                Returns:
                    logits: [B, num_classes]
                """
                # Get vision model outputs
                vision_outputs = self.vision_model(pixel_values=x)

                # Get pooled output (CLS token)
                pooled_output = vision_outputs.pooler_output  # [B, hidden_size]

                # Project to CLIP embedding space
                image_embeds = self.visual_projection(pooled_output)  # [B, projection_dim]

                # Classification
                logits = self.classifier(image_embeds)  # [B, num_classes]

                return logits

        # Create classifier model
        model = MedCLIPClassifier(clip_model, num_classes, freeze_encoder=freeze_backbone)

        return model

    @staticmethod
    def _create_biomedclip(num_classes, pretrained, freeze_backbone):
        """Create BiomedCLIP model for classification.

        Uses microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 via open_clip.
        ViT-B/16, embed_dim=512, image_size=224.
        """
        if not pretrained:
            raise NotImplementedError("BiomedCLIP only supports pretrained initialization")

        from open_clip import create_model_from_pretrained

        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        print(f"Loading BiomedCLIP from: {model_name}")

        clip_model, _ = create_model_from_pretrained(model_name)

        class BiomedCLIPClassifier(nn.Module):
            """BiomedCLIP vision encoder wrapper for image classification."""
            def __init__(self, clip_model, num_classes, freeze_encoder=False):
                super().__init__()
                self.visual = clip_model.visual

                # Get embed dimension from the visual encoder
                # open_clip ViT: output is projected to embed_dim (512)
                embed_dim = 512

                # Classification head
                self.classifier = nn.Linear(embed_dim, num_classes)

                if freeze_encoder:
                    for param in self.visual.parameters():
                        param.requires_grad = False

                print("BiomedCLIP classifier created:")
                print(f"  - Backbone: ViT-B/16")
                print(f"  - Freeze encoder: {freeze_encoder}")
                print(f"  - Embed dimension: {embed_dim}")
                print(f"  - Number of classes: {num_classes}")

            def forward(self, x):
                """
                Forward pass through BiomedCLIP vision encoder and classifier.

                Args:
                    x: Input image [B, 3, 224, 224]

                Returns:
                    logits: [B, num_classes]
                """
                image_features = self.visual(x)  # [B, 512]
                logits = self.classifier(image_features)  # [B, num_classes]
                return logits

        model = BiomedCLIPClassifier(clip_model, num_classes, freeze_encoder=freeze_backbone)

        return model

    @staticmethod
    def _create_densenet(arch, num_classes, pretrained, freeze_backbone):
        """Create DenseNet model adapted for grayscale medical images."""
        # Map architecture string to model constructor
        arch_map = {
            'densenet121': models.densenet121,
            'densenet161': models.densenet161,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
        }

        if arch not in arch_map:
            raise ValueError(f"Unknown DenseNet architecture: {arch}")

        # Load base model
        if pretrained:
            model = arch_map[arch](weights='DEFAULT')
        else:
            model = arch_map[arch](weights=None)

        # Modify first conv layer for grayscale input (1 channel)
        num_init_features_map = {
            'densenet121': 64,
            'densenet161': 96,
            'densenet169': 64,
            'densenet201': 64,
        }
        num_init_features = num_init_features_map[arch]

        old_conv = model.features.conv0
        new_conv = nn.Conv2d(
            1, num_init_features,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        # Average the pretrained RGB weights to initialize grayscale conv
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.features.conv0 = new_conv

        # Replace classifier
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        return model

    @staticmethod
    def _create_resnet(arch, num_classes, pretrained, freeze_backbone):
        """Create ResNet model adapted for grayscale medical images."""
        arch_map = {
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }

        if arch not in arch_map:
            raise ValueError(f"Unknown ResNet architecture: {arch}")

        # Load base model
        if pretrained:
            model = arch_map[arch](weights='DEFAULT')
        else:
            model = arch_map[arch](weights=None)

        # Modify first conv layer for grayscale input
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

        # Replace final FC layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

        return model


def get_model(config):
    """Convenience function to get a model from config."""
    return ModelFactory.create_model(config)


# --- LoRA target module configuration per model ---

# Keywords to match in named_modules() for each model type
_LORA_TARGET_KEYWORDS = {
    'medklip': ['conv2'],
    'biovil': ['conv2'],
    'medclip': ['qkv', 'q_proj', 'v_proj', 'query', 'value'],
    'biomedclip': ['query', 'value', 'qkv', 'q_proj', 'v_proj', 'in_proj'],
}

_LORA_MODULES_TO_SAVE = {
    'medklip': ['classifier', 'res_l1', 'res_l2'],
    'biovil': ['classifier'],
    'medclip': ['classifier'],
    'biomedclip': ['classifier'],
}


def _get_lora_target_modules(model, model_name):
    """Determine LoRA target modules by inspecting the model's named_modules().

    For Conv2d targets (ResNet-based models), filters to only the latter half
    of sequential blocks to focus LoRA on deeper layers.

    Args:
        model: PyTorch model
        model_name: Model name string (lowercase)

    Returns:
        target_modules: list of module name strings for LoRA
        modules_to_save: list of module name strings to keep trainable
    """
    keywords = _LORA_TARGET_KEYWORDS.get(model_name, [])
    modules_to_save = _LORA_MODULES_TO_SAVE.get(model_name, ['classifier'])

    if not keywords:
        raise ValueError(f"No LoRA target keywords defined for model: {model_name}")

    # Collect all matching module names
    matched_modules = []
    for name, module in model.named_modules():
        for kw in keywords:
            if kw in name.split('.')[-1]:
                matched_modules.append(name)
                break

    if not matched_modules:
        raise ValueError(
            f"No modules matched LoRA target keywords {keywords} in model {model_name}. "
            f"Check model architecture with model.named_modules()."
        )

    print(f"[LoRA] Model: {model_name}")
    print(f"[LoRA] Target modules ({len(matched_modules)}): {matched_modules[:10]}{'...' if len(matched_modules) > 10 else ''}")
    print(f"[LoRA] Modules to save: {modules_to_save}")

    return matched_modules, modules_to_save


def _filter_latter_half_blocks(module_names):
    """Filter Conv2d module names to keep only the latter half of sequential blocks.

    For ResNet-based models, modules follow patterns like:
    - res_features.4.0.conv2, res_features.6.2.conv3
    - encoder.encoder.encoder.layer3.0.conv2

    This function identifies the block indices and keeps only the latter half.
    """
    import re

    # Group by parent block prefix (everything before the block index)
    # e.g., "res_features.6" -> group all modules under res_features.6.*
    # We want to find the top-level sequential groups and filter latter half

    # Extract the top-level block identifier (first two levels for res_features.N)
    block_indices = set()
    for name in module_names:
        # Match patterns like "res_features.6" or "layer4"
        match = re.match(r'(.*?\.\d+)', name)
        if match:
            block_indices.add(match.group(1))

    if not block_indices:
        return module_names

    # Sort block identifiers and take latter half
    sorted_blocks = sorted(block_indices)
    half = len(sorted_blocks) // 2
    latter_blocks = set(sorted_blocks[half:])

    # Filter module names to those in latter half blocks
    filtered = []
    for name in module_names:
        match = re.match(r'(.*?\.\d+)', name)
        if match and match.group(1) in latter_blocks:
            filtered.append(name)

    if not filtered:
        # Fallback: return all if filtering removed everything
        return module_names

    return filtered


def apply_lora(model, model_config):
    """Apply PEFT LoRA adapter to a model based on config.

    Args:
        model: PyTorch model (already created by ModelFactory)
        model_config: Model configuration dict with optional 'lora' section

    Returns:
        Model with LoRA adapters applied, or original model if LoRA disabled
    """
    lora_config = model_config.get('lora', {})
    if not lora_config.get('enabled', False):
        return model

    model_name = model_config['name'].lower()
    r = lora_config.get('r', 16)
    lora_alpha = lora_config.get('lora_alpha', 32)
    lora_dropout = lora_config.get('lora_dropout', 0.1)

    target_modules, modules_to_save = _get_lora_target_modules(model, model_name)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model
