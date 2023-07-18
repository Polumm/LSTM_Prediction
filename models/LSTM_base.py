# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:40
# @Author  : 宋楚嘉
# @FileName: LSTM_base.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_base(nn.Module):
    # 定义构造函数
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM_base, self).__init__()
        self.hidden_dim = hidden_dim  # 设置隐藏层维度
        self.num_layers = num_layers  # 设置 LSTM 的层数

        # 定义 LSTM 网络，输入维度为 input_dim，隐藏层维度为 hidden_dim，层数为 num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 定义全连接层，将隐藏层的输出转化为最终的输出，输出维度为 output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 定义前向传播过程
    def forward(self, x):
        # 初始化隐藏状态和细胞状态，维度均为 (层数, batch大小, 隐藏层维度)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # LSTM层前向传播，得到输出以及最新的隐藏状态和细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将 LSTM 层的输出通过全连接层得到最终的输出
        out = self.fc(out[:, -1, :])
        return out
