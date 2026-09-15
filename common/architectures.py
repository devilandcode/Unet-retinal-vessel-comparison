"""
U-Net, U-Net++ and Attention U-Net (ResNet34 encoder).

Copied from the original training scripts without changing skip/attention/nested
decoder logic — those differences are the point of the comparison.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def _sanitize_prob(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf then clamp to [0, 1] so shared BCELoss stays valid."""
    return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _resnet(backbone: str, pretrained: bool):
    if backbone == 'resnet34':
        return models.resnet34(pretrained=pretrained)
    if backbone == 'resnet50':
        return models.resnet50(pretrained=pretrained)
    raise ValueError(f'Unsupported backbone: {backbone}')


class ImprovedUNet(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True, num_classes=1):
        super().__init__()
        base = _resnet(backbone, pretrained)
        # Do not keep `base` as a child: that would leave unused ImageNet `fc`
        # in the optimizer (unfair vs Attention U-Net / U-Net++).
        self.conv1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.up4 = self._make_up_block(512, 256)
        self.up3 = self._make_up_block(256 + 256, 128)
        self.up2 = self._make_up_block(128 + 128, 64)
        self.up1 = self._make_up_block(64 + 64, 32)
        self.final_up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, 1)

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1_pool = self.maxpool(x1)
        x2 = self.layer1(x1_pool)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        up4 = self.up4(x5)
        up3 = self.up3(torch.cat([up4, x4], dim=1))
        up2 = self.up2(torch.cat([up3, x3], dim=1))
        up1 = self.up1(torch.cat([up2, x2], dim=1))
        output = self.final_conv(self.final_up(up1))
        return _sanitize_prob(torch.sigmoid(output))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    ENC_CHANNELS = [64, 64, 128, 256, 512]
    DEC_CHANNELS = [32, 64, 128, 256]

    def __init__(self, backbone: str = 'resnet34', pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        base = _resnet(backbone, pretrained)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4
        e = self.ENC_CHANNELS
        d = self.DEC_CHANNELS
        self.node_0_1 = ConvBlock(e[0] + e[1], d[0])
        self.node_1_1 = ConvBlock(e[1] + e[2], d[1])
        self.node_2_1 = ConvBlock(e[2] + e[3], d[2])
        self.node_3_1 = ConvBlock(e[3] + e[4], d[3])
        self.node_0_2 = ConvBlock(e[0] + d[0] + d[1], d[0])
        self.node_1_2 = ConvBlock(e[1] + d[1] + d[2], d[1])
        self.node_2_2 = ConvBlock(e[2] + d[2] + d[3], d[2])
        self.node_0_3 = ConvBlock(e[0] + d[0] * 2 + d[1], d[0])
        self.node_1_3 = ConvBlock(e[1] + d[1] * 2 + d[2], d[1])
        self.node_0_4 = ConvBlock(e[0] + d[0] * 3 + d[1], d[0])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.head_1 = nn.Conv2d(d[0], num_classes, 1)
        self.head_2 = nn.Conv2d(d[0], num_classes, 1)
        self.head_3 = nn.Conv2d(d[0], num_classes, 1)
        self.head_4 = nn.Conv2d(d[0], num_classes, 1)
        self.final_up = nn.ConvTranspose2d(d[0], d[0], 2, stride=2)
        self.final_conv = nn.Conv2d(d[0], num_classes, 1)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        n3_1 = self.node_3_1(torch.cat([x3, self.up(x4)], dim=1))
        n2_1 = self.node_2_1(torch.cat([x2, self.up(x3)], dim=1))
        n1_1 = self.node_1_1(torch.cat([x1, self.up(x2)], dim=1))
        n0_1 = self.node_0_1(torch.cat([x0, self.up(x1)], dim=1))
        n2_2 = self.node_2_2(torch.cat([x2, n2_1, self.up(n3_1)], dim=1))
        n1_2 = self.node_1_2(torch.cat([x1, n1_1, self.up(n2_1)], dim=1))
        n0_2 = self.node_0_2(torch.cat([x0, n0_1, self.up(n1_1)], dim=1))
        n1_3 = self.node_1_3(torch.cat([x1, n1_1, n1_2, self.up(n2_2)], dim=1))
        n0_3 = self.node_0_3(torch.cat([x0, n0_1, n0_2, self.up(n1_2)], dim=1))
        n0_4 = self.node_0_4(torch.cat([x0, n0_1, n0_2, n0_3, self.up(n1_3)], dim=1))
        out1 = torch.sigmoid(self.final_conv(self.final_up(n0_1)))
        out2 = torch.sigmoid(self.final_conv(self.final_up(n0_2)))
        out3 = torch.sigmoid(self.final_conv(self.final_up(n0_3)))
        out4 = torch.sigmoid(self.final_conv(self.final_up(n0_4)))
        return _sanitize_prob((out1 + out2 + out3 + out4) / 4.0)


class AttentionGate(nn.Module):
    def __init__(self, F_x: int, F_g: int, F_int: int):
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(self.W_x(x) + self.W_g(g)))
        return x * alpha


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNet(nn.Module):
    ENC_CH = [64, 64, 128, 256, 512]

    def __init__(self, backbone: str = 'resnet34', pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        base = _resnet(backbone, pretrained)
        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc2 = base.layer1
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4
        e = self.ENC_CH
        self.ag4 = AttentionGate(F_x=e[3], F_g=e[4], F_int=e[3] // 2)
        self.ag3 = AttentionGate(F_x=e[2], F_g=e[3], F_int=e[2] // 2)
        self.ag2 = AttentionGate(F_x=e[1], F_g=e[2], F_int=e[1] // 2)
        self.ag1 = AttentionGate(F_x=e[0], F_g=e[1], F_int=e[0] // 2)
        self.up4 = UpBlock(e[4], e[3])
        self.up3 = UpBlock(e[3], e[2])
        self.up2 = UpBlock(e[2], e[1])
        self.up1 = UpBlock(e[1], e[0])
        self.final_up = nn.ConvTranspose2d(e[0], 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        d4 = self.up4(x5, self.ag4(x4, x5))
        d3 = self.up3(d4, self.ag3(x3, d4))
        d2 = self.up2(d3, self.ag2(x2, d3))
        d1 = self.up1(d2, self.ag1(x1, d2))
        return _sanitize_prob(torch.sigmoid(self.final_conv(self.final_up(d1))))
