import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_channels)

    def forward(self, x):
        return x + self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=64, num_blocks=23, growth_channels=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        self.residual_blocks = nn.Sequential(*[
            RRDB(num_feat, growth_channels) for _ in range(num_blocks)
        ])
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.residual_blocks(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        out = self.conv_last(out)
        return out
