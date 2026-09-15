"""
U-Net (ResNet34) on DRIVE — patient-level 5-fold CV on Data/train + held-out Data/test.

Shared protocol (common/): same folds, image size, augmentations, loss,
optimizer, metrics. Historical single-run JSON is not overwritten.

    python u-net.py
    python u-net.py --max-epochs 1 --fold 0
    python u-net.py --splits-only
"""

from common.architectures import ImprovedUNet
from common.config import CONFIG, UNET_ENCODER_PREFIXES
from common.cv_runner import run_cv


def build_model():
    return ImprovedUNet(
        backbone=CONFIG['model']['backbone'],
        pretrained=CONFIG['model']['pretrained'],
    )


if __name__ == '__main__':
    run_cv(
        architecture_name='u-net',
        dataset_name='drive',
        model_factory=build_model,
        encoder_prefixes=UNET_ENCODER_PREFIXES,
    )
