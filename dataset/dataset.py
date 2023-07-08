# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:41
# @Author  : 宋楚嘉
# @FileName: dataset.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import torch

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return len(self.features)
