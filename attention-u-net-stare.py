"""
Attention U-Net (ResNet34) on STARE — patient-level 5-fold CV.

    python attention-u-net-stare.py
"""

from common.architectures import AttentionUNet
from common.config import CONFIG, ATTENTION_UNET_ENCODER_PREFIXES
from common.cv_runner import run_cv


def build_model():
    return AttentionUNet(
        backbone=CONFIG['model']['backbone'],
        pretrained=CONFIG['model']['pretrained'],
    )


if __name__ == '__main__':
    run_cv(
        architecture_name='attention-u-net',
        dataset_name='stare',
        model_factory=build_model,
        encoder_prefixes=ATTENTION_UNET_ENCODER_PREFIXES,
    )
