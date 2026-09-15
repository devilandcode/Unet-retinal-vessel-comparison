"""Retina image/mask dataset and albumentations pipelines (identical for all U-Nets).

CLAHE vessel enhancement lives in RetinaDataset.enhance_vessel_contrast — do not
add another CLAHE step in albumentations (that would double-CLAHE).
nnU-Net keeps its own self-configuring augs; this module is U-Net-family only.
"""

from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CONFIG
from .splits import Sample


class RetinaDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None, enhance_vessels: bool = True):
        self.samples = samples
        self.transform = transform
        self.enhance_vessels = enhance_vessels

    def __len__(self):
        return len(self.samples)

    def enhance_vessel_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample.image_path).convert('RGB'))
        mask = np.array(Image.open(sample.mask_path).convert('L'))

        if self.enhance_vessels:
            image = self.enhance_vessel_contrast(image)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        image = (image - image.mean()) / (image.std() + 1e-8)
        return image, mask


def _shift_scale_rotate(A, aug: dict):
    """Small shift/scale; rotation is a separate Rotate so rotate_limit=0 here."""
    shift = aug['shift_limit']
    scale = aug['scale_limit']
    p = aug['shift_scale_rotate_p']
    if hasattr(A, 'ShiftScaleRotate'):
        return A.ShiftScaleRotate(
            shift_limit=shift,
            scale_limit=scale,
            rotate_limit=0,
            p=p,
        )
    return A.Affine(
        translate_percent={'x': (-shift, shift), 'y': (-shift, shift)},
        scale=(1.0 - scale, 1.0 + scale),
        rotate=0,
        p=p,
    )


def _rotate(A, limit: float, p: float):
    if hasattr(A, 'Rotate'):
        return A.Rotate(limit=limit, p=p)
    return A.Affine(rotate=(-limit, limit), p=p)


def _mild_distortion(A, aug: dict):
    """Conservative geometric warp for thin vessels (prefer grid over aggressive elastic)."""
    p = aug['grid_distort_p']
    if hasattr(A, 'GridDistortion'):
        return A.GridDistortion(
            num_steps=aug['grid_num_steps'],
            distort_limit=aug['grid_distort_limit'],
            p=p,
        )
    if hasattr(A, 'ElasticTransform'):
        return A.ElasticTransform(alpha=20, sigma=6, p=p)
    return A.NoOp()


def get_transforms(phase: str, image_size: Optional[int] = None):
    try:
        import albumentations as A
    except ImportError:
        import os
        os.system('pip install albumentations')
        import albumentations as A

    size = image_size or CONFIG['dataset']['image_size']
    # Val and held-out test: resize only (no random augs).
    if phase in ('val', 'test'):
        return A.Compose([A.Resize(size, size)])

    aug = CONFIG['augmentation']
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=aug['horizontal_flip_p']),
        A.VerticalFlip(p=aug['vertical_flip_p']),
        _rotate(A, aug['rotation'], 0.7),
        _shift_scale_rotate(A, aug),
        _mild_distortion(A, aug),
        A.RandomBrightnessContrast(
            brightness_limit=aug['brightness'],
            contrast_limit=aug['contrast'],
            p=0.5,
        ),
    ])
