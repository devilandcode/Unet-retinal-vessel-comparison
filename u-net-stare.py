"""
U-Net (ResNet34) on STARE — patient-level 5-fold CV on the train-pool + held-out test.

Fellow-eye pairs stay in the same fold (see common/splits.py).

    python u-net-stare.py
    python u-net-stare.py --splits-only
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
        dataset_name='stare',
        model_factory=build_model,
        encoder_prefixes=UNET_ENCODER_PREFIXES,
    )
