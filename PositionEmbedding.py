import torch
import math

# [b, 50] -> [b, 50, 48]
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu
            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(50, 48)
        for i in range(50):
            for j in range(48):
                pe[i, j] = get_pe(i, j, 48)
        # [50, 48] -> [1, 50, 48]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        # print("pe",pe.shape)

        # 词编码层
        self.embed = torch.nn.Embedding(39, 48)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # print("x",x.device)
        embed = self.embed(x)
        # print("embed",embed.device)
        # print("pe",self.pe.device)
        embed = embed + self.pe
        return embed


model = PositionEmbedding()
x = torch.randint(0, 39, (8, 50))
y = model(x)
print(y.shape)
