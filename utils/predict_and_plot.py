# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:39
# @Author  : 宋楚嘉
# @FileName: predict_and_plot.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.plot_utils import plot_and_save

def predict_and_plot(model, trainloader, testloader, loss_fn, train_y, test_y, train_loss, train_metric, metric_name, prediction_step, prefix=''):
    model.eval()
    train_predictions = []
    with torch.no_grad():
        for x, y in trainloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            train_predictions.append(y_pred.cpu().numpy())
    train_predictions = np.concatenate(train_predictions, axis=0)

    test_predictions = []
    test_losses = []
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            test_predictions.append(y_pred.cpu().numpy())
            loss = loss_fn(y_pred, y)
            test_losses.append(loss.item())
    test_predictions = np.concatenate(test_predictions, axis=0)

    plt.ion()  # 交互模式，避免后续阻塞进程

    for step in range(min(1, prediction_step)):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # 创建两个子图

        # 对于每一步预测，我们绘制训练集和测试集的预测与目标值的对比图
        axs[0].plot(train_y[:, step], label="train target")
        axs[0].plot(train_predictions[:, 0] if train_predictions.shape[1] == 1 else train_predictions[:, step],
                    label="train prediction")
        axs[0].set_xlabel("Time steps")
        axs[0].set_ylabel("Values")
        axs[0].legend()
        axs[0].set_title(f'Training Predictions vs Targets at step {step + 1}')

        axs[1].plot(test_y[:, step], label="test target")
        axs[1].plot(test_predictions[:, 0] if test_predictions.shape[1] == 1 else test_predictions[:, step],
                    label="test prediction")
        axs[1].set_xlabel("Time steps")
        axs[1].set_ylabel("Values")
        axs[1].legend()
        axs[1].set_title(f'Testing Predictions vs Targets at step {step + 1}')

        plt.tight_layout()
        plt.savefig(f'./results/{prefix}_predictions_vs_targets_step_{step + 1}.png', dpi=600)  # 将图保存为PNG图
        plt.show()
        plt.pause(2)
        plt.close()

    plot_and_save(train_loss, train_metric, metric_name, prefix + 'train_loss_' + metric_name + '.png')

