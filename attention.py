import math
import torch

"""
    Q, K, V = [b, 3, 50, 16]
    mask = [b, 1, 50, 50]
    
    output:
        [b, 50, 48]
"""
def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成48维向量,3个头,每个头分到16维向量
    # Q,K,V = [b, 3, 50, 16]
    # mask = [b, 1, 50, 50] transpose交换后两个维度
    # [b, 3, 50, 16] * [b, 3, 16, 50] -> [b, 3, 50, 50]
    # [b, 3, 50, 50] / math.sqrt(16) -> [b, 3, 50, 50]
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(16)

    # score 盖 mask [b, 1, 50, 50] -> [b, 3, 50, 50] mask空为True
    score = score.masked_fill(mask == True, -1e9)

    # softmax [b, 3, 50, 50] -> [b, 3, 50, 50]
    score = torch.softmax(score, dim=-1)

    # [b, 3, 50, 50] * V[b, 3, 50, 16] -> [b, 3, 50, 16]
    score = torch.matmul(score, V)

    # [b, 3, 50, 16] -> [b, 50, 48]
    score = score.transpose(1, 2).reshape(-1, 50, 48)

    return score

# Q = torch.randn(8, 3, 50, 16)
# K = Q.clone()
# V = Q.clone()
# mask = torch.zeros(8, 1, 50, 50)
# mask[:, :, 0:25, 0:25] = True
# print(attention(Q, K, V, mask).shape)
# torch.Size([8, 50, 48])