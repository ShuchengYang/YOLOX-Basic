import torch
from network_blocks import *


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.seq1 = CBL5(384, 128)
        self.seq2 = CBL5(768, 256)
        self.seq3 = CBL5(1024, 512)
        self.up1 = nn.Sequential(
            CBL(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, backbone_output):
        n3 = self.seq3(backbone_output[2])
        n2 = self.seq2(torch.cat([backbone_output[1], self.up2(n3)], dim=1))
        n1 = self.seq1(torch.cat([backbone_output[0], self.up1(n2)], dim=1))
        return [n1, n2, n3]