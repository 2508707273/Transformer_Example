import torch

from MaskPad import mask_pad
from MaskTril import mask_tril
from Transformer import Transformer
from data2 import y_zidian, loader, zidian_xr, zidian_yr

model = Transformer()
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model = model.cuda()
    loss_func = loss_func.cuda()

for epoch in range(10):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # x = [8, 50]
        # y = [8, 51]

        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
        # [8, 50, 39]

        pred = model(x, y[:, :-1])
        # print("pred", pred.shape)

        # [8, 50, 39] -> [400, 39]
        pred = pred.reshape(-1, 48)

        # [8, 51] -> [400]
        y = y[:, 1:].reshape(-1)

        # 忽略pad
        select = y != y_zidian['<PAD>']
        pred = pred[select]
        y = y[select]

        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 200 == 0:
            # [select, 39] -> [select]
            pred = pred.argmax(1)
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            lr = optim.param_groups[0]['lr']
            print(epoch, i, lr, loss.item(), accuracy)

    sched.step()

torch.save(model.state_dict(), 'model.pth')
