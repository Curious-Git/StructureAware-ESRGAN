import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class RRDB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualDenseBlock(64) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True)
        )
        self.final = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.upsample(x)
        x = self.final(x)
        return x
