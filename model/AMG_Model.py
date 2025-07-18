import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvModel(nn.Module):
    def __init__(self, inchannel, outchannel, size=[16,16], atrous_rates=[2, 4, 6]):
        super(DilatedConvModel, self).__init__()

        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, padding=0,
                      dilation=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel , out_channels=outchannel, kernel_size=3, padding=1,
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, padding=1,
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.dilated_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, padding=1,
                      dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Upsample(size=size, mode='bilinear')
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=outchannel*4, out_channels=outchannel*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel*2, out_channels=outchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.dilated_conv1(x)
        x1 = self.dilated_conv2(x)
        x2 = self.dilated_conv3(x)
        x3 = self.dilated_conv4(x)
        x_ = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.conv_final(x_)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_downsample = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.downsample(x)
        identity = self.bn_downsample(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


class AMG(nn.Module):
    def __init__(self, o_channels=64):
        super(AMG, self).__init__()
        self.up_st0 = nn.Sequential(
            nn.ConvTranspose2d(o_channels*8, o_channels*8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.up_su0 = nn.Sequential(
            nn.ConvTranspose2d(o_channels*8, o_channels*8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.res1 = ResidualBlockWithSE(o_channels*24, o_channels*16)
        self.dillated_conv1 = DilatedConvModel(o_channels*16, o_channels*4)
        self.conv_final1 = nn.Sequential(
            nn.Conv2d(o_channels*4, o_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(o_channels*4, 1, kernel_size=1)
        )

        self.up_st1 = nn.Sequential(
            nn.ConvTranspose2d(o_channels*2, o_channels*2, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.up_su1 = nn.Sequential(
            nn.ConvTranspose2d(o_channels*2, o_channels*2, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.res2 = ResidualBlockWithSE(o_channels*6, o_channels*4)
        self.dillated_conv2 = DilatedConvModel(o_channels*4, o_channels*4, size=[64,64], atrous_rates=[6, 12, 18])
        self.conv_final2 = nn.Sequential(
            nn.Conv2d(o_channels*4, o_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(o_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(o_channels*4, 1, kernel_size=1)
        )

        self.attention_conv = nn.Conv2d(2, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output_st, output_su):
        assert len(output_st) == len(output_su)
        output_st[3] = self.up_st0(output_st[3])
        output_su[3] = self.up_su0(output_su[3])
        output_32 = torch.cat([output_st[3], output_su[3], output_st[2], output_su[2]], dim=1)
        output_32 = self.res1(output_32)
        output_32 = self.dillated_conv1(output_32)
        output_32 = F.interpolate(output_32, size=[64,64], mode='bilinear')
        output_32 = self.conv_final1(output_32)

        output_st[1] = self.up_st1(output_st[1])
        output_su[1] = self.up_su1(output_su[1])
        output_01 = torch.cat([output_st[1], output_su[1], output_st[0], output_su[0]], dim=1)
        output_01 = self.res2(output_01)
        output_01 = self.dillated_conv2(output_01)
        output_01 = self.conv_final2(output_01)

        combined_prob  = torch.cat([output_32, output_01], dim=1)

        attention_weights = self.attention_conv(combined_prob)

        attention_weights = self.softmax(attention_weights)

        weighted_prob1 = output_32 * attention_weights[:, 0:1, :, :]
        weighted_prob2 = output_01 * attention_weights[:, 1:2, :, :]

        fused_prob = weighted_prob1 + weighted_prob2
        fused_prob = torch.sigmoid(fused_prob)

        return fused_prob
