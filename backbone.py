from network_blocks import *



class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.seq1 = nn.Sequential(
            CBL(3, 32, 3, 1, 1),
            ResX(1, 32, 64, 3, 2, 1),
            ResX(2, 64, 128, 3, 2, 1),
            ResX(8, 128, 256, 3, 2, 1)
        )
        self.seq2 = ResX(8, 256, 512, 3, 2, 1)
        self.seq3 = ResX(4, 512, 1024, 3, 2, 1)

    def forward(self, x):
        b1 = self.seq1(x)
        b2 = self.seq2(b1)
        b3 = self.seq3(b2)
        return [b1, b2, b3]