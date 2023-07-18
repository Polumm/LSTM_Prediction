# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:39
# @Author  : 宋楚嘉
# @FileName: preprocess.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import pandas as pd
import numpy as np
import datetime

# 创建一个 DataFrame 来存储数据处理的结果
experiment_results = pd.DataFrame(columns=['hidden_size', 'prediction_step', 'features', 'data_split', 'normalization', 'final_train_loss', 'final_test_loss', 'final_train_rmse', 'final_test_rmse'])

def load_and_preprocess_data(filepath, prediction_step, features):
    # 从指定路径加载数据，并进行异常记录剔除与空值填补
    data = pd.read_csv(filepath)
    data = data.iloc[24:].copy()
    data.fillna(method='ffill', inplace=True)
    data.drop('No', axis=1, inplace=True)
    # 创建新的 'time' 列，代表每行数据的时间戳，并设置为索引
    data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']),axis=1)
    data.set_index('time', inplace=True)
    data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)
    # 添加列名并将将 'cbwd' 列的值转换为独热编码，并加入到 DataFrame 中
    data.columns = ['pm2.5', 'dew', 'temp', 'press', 'cbwd', 'iws', 'snow', 'rain']
    data = data.join(pd.get_dummies(data.cbwd))
    del data['cbwd']

    # 选择特定的特征列
    data = data[features]

    # 设置序列长度，由预测步数决定
    sequence_length = prediction_step * 24  # 将原来固定的sequence_length修改为prediction_step*24
    delay = prediction_step
    data_ = []
    # 生成具有特定序列长度的连续时间序列样本
    for i in range(len(data) - sequence_length - delay):
        df = data.iloc[i: i + sequence_length + delay].copy()
        # 对每个序列样本进行标准化
        mean = df.mean()
        std = df.std()
        df = (df - mean) / std
        data_.append(df)
    data_ = np.array([df.values for df in data_])

    # 切分输入和输出
    np.random.shuffle(data_)
    x = data_[:, :-delay, :]
    y = data_[:, -delay:, 0]
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # 划分训练集和测试集
    split_boundary = int(data_.shape[0] * 0.8)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, test_x, train_y, test_y


