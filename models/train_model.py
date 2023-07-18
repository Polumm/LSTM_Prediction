# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:44
# @Author  : 宋楚嘉
# @FileName: train_model.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import torch
import numpy as np

# 定义计算均方根误差的函数
def compute_rmse(y_pred, y):
    return np.sqrt(((y_pred - y) ** 2).mean())

# 定义模型训练函数
def fit(epochs, model, trainloader, validloader, loss_fn, optimizer):
    train_loss = []
    train_rmse = []
    test_loss = []
    test_rmse = []
    # 对每一个epoch进行循环
    for epoch in range(epochs):
        total = 0
        running_loss = 0.0
        running_rmse = 0.0

        model.train()  # 开始训练
        # 对训练集进行循环
        for x, y in trainloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')  # 如果GPU可用，将数据移到GPU上
            y_pred = model(x)  # 得到模型的预测值
            loss = loss_fn(y_pred, y)  # 计算预测值与真实值之间的损失
            rmse = compute_rmse(y_pred.cpu().detach().numpy(), y.cpu().numpy())  # 计算预测值与真实值之间的均方根误差
            optimizer.zero_grad()  # 清空优化器中的梯度信息
            loss.backward()  # 进行反向传播
            optimizer.step()  # 更新模型参数
            with torch.no_grad():
                total += y.size(0)
                running_loss += loss.item() * y.size(0)
                running_rmse += rmse * y.size(0)

        epoch_loss = running_loss / total
        epoch_rmse = running_rmse / total
        train_loss.append(epoch_loss)
        train_rmse.append(epoch_rmse)

        # 对测试集进行相似的操作
        test_total = 0
        test_running_loss = 0.0
        test_running_rmse = 0.0

        model.eval()  # 开始测试
        with torch.no_grad():
            for x, y in validloader:
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                rmse = compute_rmse(y_pred.cpu().detach().numpy(), y.cpu().numpy())
                test_total += y.size(0)
                test_running_loss += loss.item() * y.size(0)
                test_running_rmse += rmse * y.size(0)

        epoch_test_loss = test_running_loss / test_total
        epoch_test_rmse = test_running_rmse / test_total
        test_loss.append(epoch_test_loss)
        test_rmse.append(epoch_test_rmse)

        # 打印每个epoch的训练和测试的损失和均方根误差
        print('epoch: ', epoch, 'loss： ', round(epoch_loss, 3), 'rmse: ', round(epoch_rmse, 3), 'test_loss： ',
              round(epoch_test_loss, 3), 'test_rmse: ', round(epoch_test_rmse, 3))

    # 返回训练和测试的损失和均方根误差
    return train_loss, train_rmse, test_loss, test_rmse

