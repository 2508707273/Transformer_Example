import torch

from FullyConnectedOutput import FullyConnectedOutput
from MutiHead import MultiHead

"""
    Encoder:
    x = [b, 50, 48]
    mask = [b, 1, 50, 50]
    [b, 50, 48] -> [b, 50, 48]
"""


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 自注意力,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        x = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        x = self.fc(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


# Encoder = Encoder()
# x = torch.randn([8, 50, 48])
# mask = torch.ones([8, 1, 50, 50])
# y = Encoder(x, mask)
# print(y.shape)