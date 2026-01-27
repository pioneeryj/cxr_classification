"""Factory for creating and loading pretrained medical imaging models."""

import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F


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
