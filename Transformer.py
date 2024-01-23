import torch

from Decoder import Decoder
from Encoder import Encoder
from MaskPad import mask_pad
from MaskTril import mask_tril
from PositionEmbedding import PositionEmbedding


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.fc = torch.nn.Linear(in_features=48, out_features=48)

    def forward(self, x, y):
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 48]
        # y = [b, 50] -> [b, 50, 48]
        x, y = self.embed_x(x), self.embed_y(y)

        # 编码层计算
        # [b, 50, 48] -> [b, 50, 48]
        x = self.encoder(x, mask_pad_x)

        # 解码层计算
        # [b, 50, 48],[b, 50, 48] -> [b, 50, 48]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        y = self.fc(y)

        return y
