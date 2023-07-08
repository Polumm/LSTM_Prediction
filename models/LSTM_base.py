# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:40
# @Author  : 宋楚嘉
# @FileName: LSTM_base.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm
import torch
import torch.nn as nn
import torch.nn.functional as F

# 在模型的最后一层去掉unsqueeze操作
class LSTM_base(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTM_base, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out  # 去掉unsqueeze操作
