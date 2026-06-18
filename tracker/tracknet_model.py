import torch
import torch.nn as nn

INPUT_W = 512
INPUT_H = 288


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TrackNetV2(nn.Module):
    """
    Input:  (B, 9, H, W)  — 3 RGB frames stacked channel-wise
    Output: (B, 1, H, W)  — shuttle confidence heatmap in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(_Block(9, 64), _Block(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(_Block(64, 128), _Block(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(_Block(128, 256), _Block(256, 256), _Block(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(_Block(256, 512), _Block(512, 512), _Block(512, 512))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(_Block(768, 256), _Block(256, 256), _Block(256, 256))
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(_Block(384, 128), _Block(128, 128))
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(_Block(192, 64), _Block(64, 64))
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        d1 = self.dec1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d3))
