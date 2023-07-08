# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 20:06
# @Author  : 宋楚嘉
# @FileName: plot_utils.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import matplotlib.pyplot as plt

def plot_and_save(train_loss, train_metric, metric_name, fig_name):
    plt.ion()  # 交互模式，避免后续阻塞进程
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, "r-", label="Train Loss")  # 可以将训练损失进行绘制
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="r")

    # Set xticks as integers
    ax1.set_xticks(range(len(train_loss)))
    ax1.set_xticklabels(range(1, len(train_loss) + 1))  # if you want epochs count to start from 1

    ax2 = ax1.twinx()
    ax2.plot(train_metric, "b-", label="Train "+metric_name)  # 可以将训练评估指标进行绘制
    ax2.set_ylabel("Train "+metric_name, color="b")

    # Set xticks as integers for the second plot
    ax2.set_xticks(range(len(train_metric)))
    ax2.set_xticklabels(range(1, len(train_metric) + 1))  # if you want epochs count to start from 1

    fig.tight_layout()
    plt.savefig('./results/'+fig_name, format='png', dpi=600)  # 将图保存为PNG图
    plt.show()
    plt.pause(2)
    plt.close()
