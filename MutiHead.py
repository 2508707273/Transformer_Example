import math
import torch

from attention import attention

"""
    Q, K, V = x = [b, 50, 48]
    mask = [b, 1, 50, 50]
    [b, 50, 48] -> [b, 50, 48]
"""
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(48, 48)
        self.fc_K = torch.nn.Linear(48, 48)
        self.fc_V = torch.nn.Linear(48, 48)

        self.out_fc = torch.nn.Linear(48, 48)

        # LN是取不同通道做归一化
        # elementwise_affine=True：指定是否使用可学习的缩放和平移参数。
        # 当设置为 True 时，LayerNorm 层会学习并应用每个特征维度的缩放和平移，从而允许模型更灵活地适应不同的数据分布。
        self.norm = torch.nn.LayerNorm(48)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成48维向量
        # Q,K,V = [b, 50, 48]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        # 多头切分
        # [b, 50, 48] -> [b, 3, 50, 16]
        Q = Q.reshape(b, 50, 3, 16).transpose(1, 2)
        K = K.reshape(b, 50, 3, 16).transpose(1, 2)
        V = V.reshape(b, 50, 3, 16).transpose(1, 2)

        # 计算注意力
        # [b, 3, 50, 16] -> [b, 50, 48]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 48] -> [b, 50, 48]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score

        return score


# x = torch.randn([8, 50, 48])
# mask = torch.zeros(8, 1, 50, 50)
# mask[:, :, 0:25, 0:25] = True
# print(MultiHead()(x, x, x, mask).shape)
