"""Model-specific data transformations for chest X-ray classification."""

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


class TransformFactory:
    """Factory for creating model-specific transforms."""

    # Normalization statistics for different models
    NORMALIZE_STATS = {
        'mimic_cxr': {
            'mean': [0.449],
            'std': [0.226],
        },
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'biovil': {
            # BioViL uses ImageNet stats but may require specific preprocessing
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'medklip': {
            # MedKLIP-specific normalization (adjust as needed)
            'mean': [0.449],
            'std': [0.226],
        },
        'medclip': {
            # MedCLIP uses CLIP normalization (OpenAI CLIP stats)
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        },
        'biomedclip': {
            # BiomedCLIP uses same CLIP normalization stats
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
        },
    }

    @staticmethod
    def get_transforms(model_name, augmentation_config, is_training=True):
        """Get training and validation transforms for a specific model.

        Args:
            model_name: Name of the model (biovil, medklip, densenet, etc.)
            augmentation_config: Dictionary with augmentation settings
            is_training: Whether transforms are for training or validation

        Returns:
            torchvision.transforms.Compose object
        """
        model_name = model_name.lower()

        if 'biovil' in model_name:
            return TransformFactory._get_biovil_transforms(augmentation_config, is_training)
        elif 'medklip' in model_name:
            return TransformFactory._get_medklip_transforms(augmentation_config, is_training)
        elif 'biomedclip' in model_name:
            return TransformFactory._get_biomedclip_transforms(augmentation_config, is_training)
        elif 'medclip' in model_name:
            return TransformFactory._get_medclip_transforms(augmentation_config, is_training)
        elif 'densenet' in model_name or 'resnet' in model_name:
            return TransformFactory._get_standard_transforms(augmentation_config, is_training)
        else:
            # Default to standard transforms
            return TransformFactory._get_standard_transforms(augmentation_config, is_training)

    @staticmethod
    def _get_biovil_transforms(aug_config, is_training):
        """Get transforms for BioViL model.

        BioViL expects RGB images, so we need to convert grayscale to RGB.
        """
        transform_list = []

        if is_training:
            # Training augmentations
            if aug_config.get('horizontal_flip', 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip'])
                )

            if aug_config.get('rotation_degrees', 0) > 0:
                degrees = aug_config['rotation_degrees']
                transform_list.append(
                    transforms.RandomRotation(degrees=[-degrees, degrees])
                )

            if aug_config.get('random_crop', False):
                transform_list.append(transforms.RandomResizedCrop(224, scale=(0.8, 1.0)))

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Resize to fixed size (512x512)
        transform_list.append(transforms.Resize((512, 512), antialias=True))

        # Convert grayscale to RGB by repeating channels
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))

        # Normalize with ImageNet stats
        stats = TransformFactory.NORMALIZE_STATS['biovil']
        transform_list.append(transforms.Normalize(mean=stats['mean'], std=stats['std']))

        return transforms.Compose(transform_list)

    @staticmethod
    def _get_medklip_transforms(aug_config, is_training):
        """Get transforms for MedKLIP model.

        Follows the original MedKLIP preprocessing pipeline:
        - Train: RandomResizedCrop(224, BICUBIC) -> RandomHorizontalFlip -> ToTensor -> ImageNet normalize
        - Test:  Resize(224) -> ToTensor -> ImageNet normalize
        """
        stats = TransformFactory.NORMALIZE_STATS['imagenet']

        if is_training:
            transform_list = [
                transforms.RandomResizedCrop(512, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize(mean=stats['mean'], std=stats['std']),
            ]
        else:
            transform_list = [
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                transforms.Normalize(mean=stats['mean'], std=stats['std']),
            ]

        return transforms.Compose(transform_list)

    @staticmethod
    def _get_medclip_transforms(aug_config, is_training):
        """Get transforms for MedCLIP model.

        Follows the original MedCLIP preprocessing pipeline:
        Resize(BICUBIC) -> CenterCrop(224) -> float conversion -> CLIP normalize.
        No aggressive augmentation for medical images.
        """
        transform_list = []

        # Convert to tensor (uint8)
        transform_list.append(transforms.ToTensor())

        # Convert grayscale to RGB by repeating channels
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))

        # Resize with BICUBIC then CenterCrop (224 matches pretrained position embeddings)
        transform_list.append(transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True))
        transform_list.append(transforms.CenterCrop(224))

        # Normalize with CLIP stats
        stats = TransformFactory.NORMALIZE_STATS['medclip']
        transform_list.append(transforms.Normalize(mean=stats['mean'], std=stats['std']))

        return transforms.Compose(transform_list)

    @staticmethod
    def _get_biomedclip_transforms(aug_config, is_training):
        """Get transforms for BiomedCLIP model.

        BiomedCLIP uses ViT-B/16 with image_size=224 and CLIP normalization.
        Follows the open_clip preprocessing: Resize(224, BICUBIC) -> CenterCrop(224) -> normalize.
        """
        transform_list = []

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Convert grayscale to RGB by repeating channels
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))

        # Resize with BICUBIC then CenterCrop (224 matches pretrained position embeddings)
        transform_list.append(transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True))
        transform_list.append(transforms.CenterCrop(224))

        # Normalize with CLIP stats
        stats = TransformFactory.NORMALIZE_STATS['biomedclip']
        transform_list.append(transforms.Normalize(mean=stats['mean'], std=stats['std']))

        return transforms.Compose(transform_list)

    @staticmethod
    def _get_standard_transforms(aug_config, is_training):
        """Get standard transforms for DenseNet/ResNet models.

        These models work with grayscale images.
        """
        transform_list = []

        if is_training:
            # Training augmentations
            if aug_config.get('horizontal_flip', 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip'])
                )

            if aug_config.get('rotation_degrees', 0) > 0:
                degrees = aug_config['rotation_degrees']
                transform_list.append(
                    transforms.RandomRotation(degrees=[-degrees, degrees])
                )

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Resize to fixed size (1024x1024)
        transform_list.append(transforms.Resize((1024, 1024), antialias=True))

        # Normalize with MIMIC-CXR stats
        stats = TransformFactory.NORMALIZE_STATS['mimic_cxr']
        transform_list.append(transforms.Normalize(mean=stats['mean'], std=stats['std']))

        return transforms.Compose(transform_list)


def get_train_transform(model_name, augmentation_config):
    """Get training transform for a specific model."""
    return TransformFactory.get_transforms(model_name, augmentation_config, is_training=True)


def get_val_transform(model_name):
    """Get validation transform for a specific model."""
    # No augmentation for validation
    empty_aug_config = {}
    return TransformFactory.get_transforms(model_name, empty_aug_config, is_training=False)
