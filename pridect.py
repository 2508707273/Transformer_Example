import torch

from MaskPad import mask_pad
from MaskTril import mask_tril
from Transformer import Transformer
from data2 import loader, y_zidian, zidian_xr, zidian_yr


def predict(x):
    # x = [1, 50]
    # print(x)
    model.eval()

    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)
    # print("mask_pad_x= ", mask_pad_x.device)

    # 初始化输出,这个是固定值
    # [1, 50]
    # [[0,2,2,2...]]
    target = [y_zidian['<SOS>']] + [y_zidian['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [1, 50] -> [1, 50, 32]
    x = model.embed_x(x)
    print("x= ", x.shape)

    # 编码层计算,维度不变
    # [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第49个词
    for i in range(49):
        # [1, 50]
        y = target.to("cuda")

        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y)

        # y编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        # print("y= ", y.device)
        y = model.embed_y(y)

        # 解码层计算,维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target


# 创建一个新的模型实例
model = Transformer().to("cuda")

# 加载保存的模型参数
model.load_state_dict(torch.load('model.pth'))
# 打印模型结构
# print(model)

# 打印模型的训练参数
# for name, param in model.named_parameters():
#     print(f'Parameter: {name}, Size: {param.size()}')

# 测试
for i, (x, y) in enumerate(loader):
    break

for i in range(8):
    print(i)
    print(''.join([zidian_xr[i] for i in x[i].tolist()]))
    print(''.join([zidian_yr[i] for i in y[i].tolist()]))
    print(''.join([zidian_yr[i] for i in predict(x[i].unsqueeze(0).to("cuda"))[0].tolist()]))