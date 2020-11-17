import sys
sys.path.append("./data")
sys.path.append("./net")

import os
import random
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.dataset import handsDataset
import torch.optim as optim
from net.mobilenetV3 import MobileNetV3_Small
from torch.autograd import Variable
import tqdm


MAX_EPOCCH = 60
BATCH_SIZE = 2
LR = 0.0001
LOG_INTERVAL = 3
VAL_INTERVAL = 3

#构建数据
split_dir = os.path.join(".", "data", "splitData")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)
#对数据集做预处理

train_data = handsDataset(dataset=train_dir)
valid_data = handsDataset(dataset=valid_dir)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)

#加载模型
net = MobileNetV3_Small(num_classes=10)

if torch.cuda.is_available():
    net.cuda()

#加载损失函数
criterion = nn.CrossEntropyLoss()
#优化器
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

#记录训练数据
train_curve = list()
valid_curve = list()
accurancy_global = 0.0
val_accurancy_global = 0.0
#训练
for epoch in range(MAX_EPOCCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    running_loss = 0.
    #train
    net.train()
    for i, data in enumerate(tqdm(train_loader)):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        #前向计算
        out = net(img)
        optimizer.zero_grad()
        loss=criterion(out, label)
        #反向传播
        loss.backward()
        optimizer.step()
        #打印log
        if (i+1)%LOG_INTERVAL==0:
            print('train epoch:{}, lr:{:.4f},loss:{:.4f}'.format(epoch+1, optimizer.param_groups[0]['lr'] ,loss.data.item()))

        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        running_loss += loss.data.item()
        correct += (predicted == label).sum()

    print("="*20)
    accurancy = correct / total
    #保存训练参数
    loss_mean = running_loss / total
    print("train epoch:{} finished, lr:{}, loss_mean:{:.4f}, train_acc{:.4f}".format(epoch+1, optimizer.param_groups[0]['lr'], loss_mean, accurancy))
    state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}

    #valid
    valid_loss_mean = 0.
    valid_correct = 0.
    valid_total = 0.
    valid_running_loss = 0.
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader)):
            img, label = data
            img = Variable(img).cuda()
            label = Variable(img).cuda()

            val_out = net(img)
            val_loss = criterion(val_out, label)
            if i%VAL_INTERVAL == 0:
                print('val epoch:{}, lr:{:.4f}, loss:{:.4f}'.format(epoch+1, optimizer.param_groups[0]['lr'] ,val_loss.data.item()))

            _, predicted = torch.max(out.data, 1)
            valid_total += label.size(0)
            valid_running_loss += val_loss.data.item()
            valid_correct += (predicted == label).sum()
    val_accurancy = valid_correct / valid_total
    valid_loss_mean = valid_running_loss / valid_total
    print("-"*20)
    print("val epoch:{} finished, lr:{}, loss_mean:{:.4f}, train_acc{:.4f}".format(epoch + 1,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     valid_loss_mean, val_accurancy))
    if accurancy > accurancy_global and val_accurancy > val_accurancy_global:
        torch.save(state, "./weights/best.pkl")
        print("准确率由：", accurancy_global, "上升至：", accurancy, "已更新并保存权值为weights/best.pkl")
        accurancy_global = accurancy
        val_accurancy_global = val_accurancy


torch.save(net, "./weights/last.pkl")
print("训练完毕，权重已保存为：weights/last.pkl")