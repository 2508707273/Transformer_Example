x_zidian = "<SOS>,<EOS>,<PAD>,<->,0,1,2,3,4,5,6,7,8,9".split(',')

x_zidian = {word: i for i, word in enumerate(x_zidian)}
# {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<->': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13}

zidian_xr = [k for k, v in x_zidian.items()]
# ['<SOS>', '<EOS>', '<PAD>', '<->', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

y_zidian = "<SOS>,<EOS>,<PAD>,(-),(+),0,1,2,3,4,5,6,7,8,9".split(',')
y_zidian = {word: i for i, word in enumerate(y_zidian)}

zidian_yr = [k for k, v in y_zidian.items()]

import random
import numpy as np
import torch

def get_data():
    # 定义词集合
    words = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    # 生成x1
    n1 = random.randint(8, 24)
    x1 = np.random.choice(words, size=n1, replace=True)
    # ['3' '3' '0' '2' '1' '3' '9' '2' '1' '7' '3' '9' '9' '3' '6' '9' '8' '6'
    #  '6' '4' '6' '6' '6' '5' '4' '4' '9' '4' '6' '1']
    x1 = x1.tolist()
    # x转换成数字
    int_x1 = int(''.join(x1))
    x1 = list(str(int_x1))

    # 生成x2
    n2 = random.randint(8, 24)
    x2 = np.random.choice(words, size=n2, replace=True)
    x2 = x2.tolist()
    int_x2 = int(''.join(x2))
    x2 = list(str(int_x2))

    # 生成x
    x = x1 + ['<->'] + x2

    # 生成y
    int_y = int_x1 - int_x2
    y = list(str(abs(int_y)))
    if int_y < 0:
        y = ['(-)'] + y
    else:
        y = ['(+)'] + y

    # ['(+)', '9', '2', '2', '1', '0', '6', '3', '7', '7', '4', '3']

    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]
    # ['<SOS>', '5', '1', '9', '4', '5', '3', '4', '8', '1', '9', '8', '0', '6', '3', '9', '0', '2', '6', '9', '2', '1', '2', '<->', '6', '7', '6', '7', '4', '2', '1', '0', '0', '0', '5', '6', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
    # ['<SOS>', '(+)', '5', '1', '9', '4', '5', '3', '4', '8', '1', '9', '1', '2', '9', '6', '4', '8', '1', '6', '9', '1', '5', '6', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
    # 编码成数据
    x = [x_zidian[i] for i in x]
    y = [y_zidian[i] for i in y]
    # [0, 9, 5, 13, 8, 9, 7, 8, 12, 5, 13, 12, 4, 10, 7, 13, 4, 6, 10, 13, 6, 5, 6, 3, 10, 11, 10, 11, 8, 6, 5, 4, 4, 4, 9, 10, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # [0, 4, 10, 6, 14, 9, 10, 8, 9, 13, 6, 14, 6, 7, 14, 11, 9, 13, 6, 11, 14, 6, 10, 11, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    # print(x.shape) torch.Size([50])
    # print(y.shape) torch.Size([51])
    return x, y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num):
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return get_data()


dataset = Dataset(100000)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
# print(len(loader))
# print(next(iter(loader))[0].shape)
# print(next(iter(loader))[1].shape)
# torch.Size([128, 50])
# torch.Size([128, 51])