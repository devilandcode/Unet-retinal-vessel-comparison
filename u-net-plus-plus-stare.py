"""
U-Net++ (ResNet34) on STARE — patient-level 5-fold CV.

    python u-net-plus-plus-stare.py
"""

from common.architectures import UNetPlusPlus
from common.config import CONFIG, UNETPP_ENCODER_PREFIXES
from common.cv_runner import run_cv


def build_model():
    return UNetPlusPlus(
        backbone=CONFIG['model']['backbone'],
        pretrained=CONFIG['model']['pretrained'],
    )


if __name__ == '__main__':
    run_cv(
        architecture_name='u-net-plus-plus',
        dataset_name='stare',
        model_factory=build_model,
        encoder_prefixes=UNETPP_ENCODER_PREFIXES,
    )
