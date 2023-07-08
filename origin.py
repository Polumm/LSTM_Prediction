import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')
data = data.iloc[24:].copy()
data.fillna(method='ffill', inplace=True)
data.drop('No', axis=1, inplace=True)
data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']),
                          axis=1)
data.set_index('time', inplace=True)
data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)
data.columns = ['pm2.5', 'dew', 'temp', 'press', 'cbwd', 'iws', 'snow', 'rain']
data = data.join(pd.get_dummies(data.cbwd))
del data['cbwd']

sequence_length = 5 * 24
delay = 24
data_ = []
for i in range(len(data) - sequence_length - delay):
    data_.append(data.iloc[i: i + sequence_length + delay])
data_ = np.array([df.values for df in data_])

np.random.shuffle(data_)
x = data_[:, :-delay, :]
y = data_[:, -1, 0]
x = x.astype(np.float32)
y = y.astype(np.float32)

split_boundary = int(data_.shape[0] * 0.8)
train_x = x[: split_boundary]
test_x = x[split_boundary:]
train_y = y[: split_boundary]
test_y = y[split_boundary:]

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std


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


train_ds = Mydataset(train_x, train_y)
test_ds = Mydataset(test_x, test_y)

BTACH_SIZE = 512
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BTACH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BTACH_SIZE)

hidden_size = 64


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(train_x.shape[-1], hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, inputs):
        _, s_o = self.rnn(inputs)
        s_o = s_o[-1]
        x = F.dropout(F.relu(self.fc1(s_o)))
        x = self.fc2(x)
        return torch.squeeze(x)


model = Net(hidden_size)

if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def fit(epoch, model, trainloader, testloader):
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)

    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)

    print('epoch: ', epoch, 'loss： ', round(epoch_loss, 3), 'test_loss： ', round(epoch_test_loss, 3))

    return epoch_loss, epoch_test_loss


epochs = 20
train_loss = []
test_loss = []

for epoch in range(epochs):
    epoch_loss, epoch_test_loss = fit(epoch, model, train_dl, test_dl)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)

plt.plot(train_loss, label="train loss")
plt.plot(test_loss, label="valid loss")
plt.legend()

model.eval()
predictions = []
test_loss = []
with torch.no_grad():
    for x, y in test_dl:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        predictions.append(y_pred.numpy())
        loss = loss_fn(y_pred, y)
        test_loss.append(loss.item())

predictions = np.hstack(predictions)
plt.plot(predictions[-100:], label="predict")
plt.plot(test_y[-100:], label="target")
plt.legend()
