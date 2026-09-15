"""Shared hyperparameters. Architecture-specific differences stay in model classes."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fold generation seed (shared by ALL architectures). Training uses a different
# seed per fold: SPLIT_SEED + 1000 + fold_index.
SPLIT_SEED = 20260830
N_FOLDS = 5

CONFIG = {
    'dataset': {
        'kaggle_id': 'abdallahwagih/retina-blood-vessel',
        'chase_kaggle_id': 'hoverstrange/chase-db1',
        'hrf_kaggle_id': 'lavanya456/hrf-images-for-dr',
        'image_size': 768,
    },
    'training': {
        'epochs': 100,
        'batch_size': 2,
        'lr_backbone': 1e-5,
        'lr_decoder': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'scheduler_patience': 7,
        'scheduler_factor': 0.5,
    },
    'model': {
        'backbone': 'resnet34',
        'pretrained': True,
    },
    'augmentation': {
        # Shared by U-Net / U-Net++ / Attention U-Net (not nnU-Net).
        'rotation': 15,
        'brightness': 0.1,
        'contrast': 0.1,
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
        'shift_limit': 0.05,
        'scale_limit': 0.1,
        'shift_scale_rotate_p': 0.5,
        'grid_num_steps': 5,
        'grid_distort_limit': 0.1,
        'grid_distort_p': 0.3,
    },
    'cv': {
        'n_folds': N_FOLDS,
        'split_seed': SPLIT_SEED,
    },
}

# Encoder parameter-name prefixes for differential LR (pretrained encoder vs decoder).
UNET_ENCODER_PREFIXES = (
    'conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4',
)
ATTENTION_UNET_ENCODER_PREFIXES = (
    'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'pool',
)
UNETPP_ENCODER_PREFIXES = (
    'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'pool',
)
