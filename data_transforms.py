"""Model-specific data transformations for chest X-ray classification."""

import torch
from torchvision import transforms
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

        MedKLIP may work with grayscale images directly.
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

        # Resize to fixed size (1024x1024) for consistent batch processing
        transform_list.append(transforms.Resize((1024, 1024), antialias=True))

        # Convert grayscale to 3-channel by repeating
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x))

        # Normalize with MedKLIP stats (now 3 channels)
        stats = TransformFactory.NORMALIZE_STATS['imagenet']  # Use ImageNet stats for 3 channels
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
