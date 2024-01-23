import torch

from data2 import x_zidian

# [b, 50] -> [b, 1, 50, 50]
def mask_pad(data):
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 50]
    # 判断每个词是不是<PAD>
    # 在数据（data）中找到所有等于字典中的特定值 zidian_x['<PAD>'] 的元素，并创建一个相同形状的布尔类型的掩码（mask）。
    mask = data == x_zidian['<PAD>']

    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)

    # 在计算注意力时,是计算50个词和50个词相互之间的注意力,所以是个50*50的矩阵
    # 是pad的列是true,意味着任何词对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0
    # 所以是pad的行不是true
    # print("mask", mask)
    # 复制n次
    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50)

    return mask

# data = torch.tensor([[1,3,2]])
# print(data.shape)
# mask = mask_pad(data)
# print(mask.shape)
# print(mask)
# torch.Size([1, 3])
# mask tensor([[[[False, False,  True]]]])
# torch.Size([1, 1, 3, 3])
# tensor([[[[False, False,  True],
#           [False, False,  True],
#           [False, False,  True]]]])

