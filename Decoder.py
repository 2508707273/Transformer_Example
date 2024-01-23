import torch

from FullyConnectedOutput import FullyConnectedOutput
from MutiHead import MultiHead

"""
    Decoder:
    x = [b, 50, 48]
    y = [b, 50, 48]
    mask_pad_x = [b, 1, 50, 50]
    mask_tril_y = [b, 1, 50, 50]
    
    [b, 50, 48] -> [b, 50, 48]
"""


class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 48],[b, 50, 48] -> [b, 50, 48]
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        y = self.fc(y)

        return y


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y
