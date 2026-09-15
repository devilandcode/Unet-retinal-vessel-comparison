"""
Attention U-Net (ResNet34 + Attention Gates) on DRIVE — patient-level 5-fold CV.

    python attention-u-net.py
    python attention-u-net.py --max-epochs 1 --fold 0
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
        dataset_name='drive',
        model_factory=build_model,
        encoder_prefixes=ATTENTION_UNET_ENCODER_PREFIXES,
    )
