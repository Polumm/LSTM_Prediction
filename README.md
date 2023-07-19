# LSTM_Prediction

关键词：深度学习，时间序列预测，LSTM模型，隐藏层，时间尺度，特征选择，数据集划分比例，标准化方式，模型优化，自动化训练，批量调参，预测精度。	

该项目为时空大数据分析与处理课程实习 第二节 时间序列预测项目的代码实现。

​    本次实习中，我在老师和学姐的指导下探究了深度学习与时间序列预测的科学问题和基本方法、LSTM模型的基本原理及其性能的影响因素。我对LSTM模型的不同隐藏层、不同时间尺度、不同特征选择、不同数据集划分比例、以及不同标准化方式对预测结果的影响进行了较为深入的实验研究。

​    此外我对参考项目进行了重构和优化，将原本的.ipynb文件分为五个关键模块，使其可以开展更加丰富的实验且便于对某个细节进行修改。重构后的项目有着较低的代码耦合性，支持批量实验，并且可以自动地保存实验参数和预测结果可视化。在实习项目中，我设计了一种自动化的训练模式，即通过遍历参数列表的方式进行批量调参。这在一定程度上减轻了科研工作者的实验负担。本次实习的实验结果表明，更丰富且有对预测益的特征、更精细的时间尺度以及Standardization标准化方式都有助于提高LSTM模型的预测精度。

------

Keywords: Deep Learning, Time Series Prediction, LSTM Model, Hidden Layers, Time Scale, Feature Selection, Dataset Split Ratio, Normalization Methods, Model Optimization, Automated Training, Batch Parameter Tuning, Prediction Accuracy.

This project entails the code implementation for the Time Series Prediction Project, which is the second part of the Spatio-Temporal Big Data Analysis and Processing course internship.

In this internship, under the guidance of my professor and senior colleagues, I delved into the scientific questions and basic methods of deep learning and time series prediction, as well as the fundamental principles of the LSTM model and factors influencing its performance. I conducted in-depth experimental research on the impact of different hidden layers, time scales, feature selections, dataset split ratios, and normalization methods on the prediction results of the LSTM model.

Additionally, I refactored and optimized the reference project, dividing the original .ipynb file into five key modules, thus enabling a richer array of experiments and facilitating modification of specific details. The refactored project exhibits low code coupling, supports batch experiments, and can automatically save experiment parameters and visualize prediction results. In the internship project, I designed an automated training mode, namely batch parameter tuning by traversing the parameter list, which to some extent lightens the experimental burden on researchers. The experimental results of this internship indicate that richer features beneficial to prediction, finer time scales, and Standardization normalization methods all contribute to enhancing the prediction accuracy of the LSTM model.
