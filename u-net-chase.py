"""
U-Net (ResNet34) on CHASE_DB1 — patient-level 5-fold CV on the train-pool + held-out test.

Fellow eyes of the same child stay in the same split (see common/splits.py).

    python u-net-chase.py
    python u-net-chase.py --splits-only
    python u-net-chase.py --max-epochs 1 --fold 0
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
        dataset_name='chase',
        model_factory=build_model,
        encoder_prefixes=UNET_ENCODER_PREFIXES,
    )
