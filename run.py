# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 19:33
# @Author  : 宋楚嘉
# @FileName: run.py.py
# @Software: PyCharm
# @Blog    ：https://github.com/Polumm

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from utils.preprocess import load_and_preprocess_data
from utils.predict_and_plot import predict_and_plot
from dataset.dataset import Mydataset
from models.LSTM_base import LSTM_base
from models.train_model import fit
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 开展实验（包括：数据预处理、模型训练、模型评估等过程），具有详细的参数列表
def run_experiment(hidden_size, prediction_step, features, data_split, normalization_method):
    # 打印实验参数
    print(
        f"\nRunning experiment with: hidden_size={hidden_size}, prediction_step={prediction_step}, features={features}, data_split={data_split}, normalization_method={normalization_method}\n")

    # 加载并预处理数据
    train_x, test_x, train_y, test_y = load_and_preprocess_data('./dataset/PRSA_data_2010.1.1-2014.12.31.csv',
                                                                prediction_step, features)

    # 实例化 datasets和dataloaders
    batch_size = 512
    train_ds = Mydataset(train_x, train_y)
    test_ds = Mydataset(test_x, test_y)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    # 实例化 model, loss, optimizer
    input_dim = len(features)
    model = LSTM_base(input_dim, hidden_size)
    if torch.cuda.is_available():
        model.to('cuda')

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 模型训练
    epochs = 20
    train_loss, train_rmse, test_loss, test_rmse = fit(epochs, model, train_dl, test_dl, loss_fn, optimizer)

    # 预测并绘图
    prefix = f"hidden_size_{hidden_size}_prediction_step_{prediction_step}_features_{features}_data_split_{data_split}_normalization_{normalization_method}"
    predict_and_plot(model, train_dl, test_dl, loss_fn, train_y, test_y, train_loss, train_rmse, 'RMSE',
                     prediction_step, prefix)

    # 返回最后一个 epoch 的 train_loss, train_rmse, test_loss, test_rmse
    return train_loss[-1], train_rmse[-1], test_loss[-1], test_rmse[-1]


if __name__ == "__main__":
    # 主函数在列表中定义所有可能的参数组合
    hidden_sizes = [16, 32, 64] # 隐藏层数
    prediction_steps = [3, 5, 7] # 时间尺度（预测步长）
    features_list = [['dew'], ['dew', 'temp'], ['dew', 'temp', 'press']] # 特征选择
    data_splits = [0.7, 0.8, 0.9] #数据集划分比例
    normalization_methods = ['minmax', 'standard'] #数据标准方式

    # 非调参时的一般参数设置
    fixed_hidden_size = 64
    fixed_prediction_step = 5
    fixed_features = ['dew', 'temp', 'press']
    fixed_data_split = 0.8
    fixed_normalization_method = 'minmax'

    experiment_results = []

    # 1、隐藏层数调参
    for hidden_size in hidden_sizes:
        print(
            f"Parameters: hidden_size={hidden_size}, prediction_step={fixed_prediction_step}, features={fixed_features}, data_split={fixed_data_split}, normalization_method={fixed_normalization_method}")

        # 获取 run_experiment 返回的四个度量值
        train_loss, train_rmse, test_loss, test_rmse = run_experiment(hidden_size, fixed_prediction_step, fixed_features, fixed_data_split,
                                        fixed_normalization_method)

        experiment_results.append({
            'hidden_size': hidden_size,
            'prediction_step': fixed_prediction_step,
            'features': str(fixed_features),  # 列表需要转换为字符串才能保存到 CSV
            'data_split': fixed_data_split,
            'normalization_method': fixed_normalization_method,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_loss': test_loss,
            'test_rmse': test_rmse
        })

    # 2、特征选择调参
    for features in features_list:
        print(
            f"Parameters: hidden_size={fixed_hidden_size}, prediction_step={fixed_prediction_step}, features={features}, data_split={fixed_data_split}, normalization_method={fixed_normalization_method}")

        train_loss, train_rmse, test_loss, test_rmse = run_experiment(fixed_hidden_size, fixed_prediction_step,
                                                                      features, fixed_data_split,
                                                                      fixed_normalization_method)

        experiment_results.append({
            'hidden_size': fixed_hidden_size,
            'prediction_step': fixed_prediction_step,
            'features': str(features),  # Lists need to be converted to string to be saved into CSV
            'data_split': fixed_data_split,
            'normalization_method': fixed_normalization_method,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_loss': test_loss,
            'test_rmse': test_rmse
        })

    # 3、数据集划分比例调参
    for data_split in data_splits:
        print(
            f"Parameters: hidden_size={fixed_hidden_size}, prediction_step={fixed_prediction_step}, features={fixed_features}, data_split={data_split}, normalization_method={fixed_normalization_method}")

        train_loss, train_rmse, test_loss, test_rmse = run_experiment(fixed_hidden_size, fixed_prediction_step,
                                                                      fixed_features, data_split,
                                                                      fixed_normalization_method)

        experiment_results.append({
            'hidden_size': fixed_hidden_size,
            'prediction_step': fixed_prediction_step,
            'features': str(fixed_features),
            'data_split': data_split,
            'normalization_method': fixed_normalization_method,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_loss': test_loss,
            'test_rmse': test_rmse
        })

    # 4、数据集标准化方法调参
    for normalization_method in normalization_methods:
        print(
            f"Parameters: hidden_size={fixed_hidden_size}, prediction_step={fixed_prediction_step}, features={fixed_features}, data_split={fixed_data_split}, normalization_method={normalization_method}")

        train_loss, train_rmse, test_loss, test_rmse = run_experiment(fixed_hidden_size, fixed_prediction_step,
                                                                      fixed_features, fixed_data_split,
                                                                      normalization_method)

        experiment_results.append({
            'hidden_size': fixed_hidden_size,
            'prediction_step': fixed_prediction_step,
            'features': str(fixed_features),
            'data_split': fixed_data_split,
            'normalization_method': normalization_method,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_loss': test_loss,
            'test_rmse': test_rmse
        })

    # 5、时间尺度调参
    for prediction_step in prediction_steps:
        print(
            f"Parameters: hidden_size={fixed_hidden_size}, prediction_step={prediction_step}, features={fixed_features}, data_split={fixed_data_split}, normalization_method={fixed_normalization_method}")

        # 获取 run_experiment 返回的四个度量值
        train_loss, train_rmse, test_loss, test_rmse = run_experiment(fixed_hidden_size, prediction_step,
                                                                      fixed_features, fixed_data_split,
                                                                      fixed_normalization_method)

        experiment_results.append({
            'hidden_size': fixed_hidden_size,
            'prediction_step': prediction_step,
            'features': str(fixed_features),  # 列表需要转换为字符串才能保存到 CSV
            'data_split': fixed_data_split,
            'normalization_method': fixed_normalization_method,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'test_loss': test_loss,
            'test_rmse': test_rmse
        })

    df = pd.DataFrame(experiment_results)
    df.to_csv('./results/experiment_results.csv', index=False)