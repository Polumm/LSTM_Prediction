# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:44
# @Author  : 宋楚嘉
# @FileName: train_model.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import torch
import numpy as np


def compute_rmse(y_pred, y):
    return np.sqrt(((y_pred - y) ** 2).mean())



def fit(epochs, model, trainloader, testloader, loss_fn, optimizer):
    train_loss = []
    train_rmse = []
    test_loss = []
    test_rmse = []
    for epoch in range(epochs):
        total = 0
        running_loss = 0.0
        running_rmse = 0.0

        model.train()
        for x, y in trainloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            rmse = compute_rmse(y_pred.cpu().detach().numpy(), y.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total += y.size(0)
                running_loss += loss.item() * y.size(0)  # Note the change here
                running_rmse += rmse * y.size(0)  # And here

        epoch_loss = running_loss / total
        epoch_rmse = running_rmse / total
        train_loss.append(epoch_loss)
        train_rmse.append(epoch_rmse)

        # Similar changes for the test set...
        test_total = 0
        test_running_loss = 0.0
        test_running_rmse = 0.0

        model.eval()
        with torch.no_grad():
            for x, y in testloader:
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                rmse = compute_rmse(y_pred.cpu().detach().numpy(), y.cpu().numpy())
                test_total += y.size(0)
                test_running_loss += loss.item() * y.size(0)  # And here
                test_running_rmse += rmse * y.size(0)  # And here

        epoch_test_loss = test_running_loss / test_total
        epoch_test_rmse = test_running_rmse / test_total
        test_loss.append(epoch_test_loss)
        test_rmse.append(epoch_test_rmse)

        print('epoch: ', epoch, 'loss： ', round(epoch_loss, 3), 'rmse: ', round(epoch_rmse, 3), 'test_loss： ',
              round(epoch_test_loss, 3), 'test_rmse: ', round(epoch_test_rmse, 3))

    return train_loss, train_rmse, test_loss, test_rmse
