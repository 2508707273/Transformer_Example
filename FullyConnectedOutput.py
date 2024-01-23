import torch

# [b, 50, 48] -> [b, 50, 48]
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=48, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=48),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=48,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 48] -> [b, 50, 48]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out

# model = FullyConnectedOutput()
# x = torch.randn(8, 50, 48)
# y = model(x)
# print(y.shape)