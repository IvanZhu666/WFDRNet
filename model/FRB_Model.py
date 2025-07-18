import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FRB(nn.Module):
    def __init__(self, n_channels=3, o_channels=64):
        super(FRB, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, o_channels)
        self.down0 = Down(o_channels,o_channels)
        self.down1 = Down(o_channels, o_channels*2)
        self.down2 = Down(o_channels*2, o_channels*4)
        self.down3 = Down(o_channels*4, o_channels*8)
        self.down4 = Down(o_channels*8, o_channels*8)
        self.conv512 = nn.Sequential(
            nn.Conv2d(o_channels*8, o_channels*8, kernel_size=1),
            nn.BatchNorm2d(o_channels*8),
            nn.ReLU(inplace=True)
        )
        self.up1 = Up(o_channels*16, o_channels*4)
        self.up2 = Up(o_channels*8, o_channels*2)
        self.up3 = Up(o_channels*4, o_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_ = self.conv512(x5)
        x6 = self.up1(x5_, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        out = F.interpolate(x3, size=(64, 64), mode='bilinear', align_corners=False)
        out = torch.cat([x2, out], dim=1)
        return (x8, x7, x6, x5_), out

