import torch

from data2 import y_zidian

# [b, 50] -> [b, 1, 50, 50]
def mask_tril(data):

    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long)).to(data.device)

    mask = data == y_zidian['<PAD>']

    # [b, 1, 50]
    mask = mask.unsqueeze(1).long().to(data.device)

    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask + tril

    # 转布尔型
    mask = mask > 0

    # 增加一个维度,便于后续的计算
    mask = mask.unsqueeze(dim=1)

    return mask



# data = torch.randn([8,50])
# print(data.shape)
# mask = mask_tril(data)
# print(mask.shape)