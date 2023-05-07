import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import spiral
import numpy as np
import time


class SpiralDataset:
    def __init__(self, device):
        self.x, self.y = spiral.load_data()
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx].astype(np.float32)).clone().to(device)
        y = self.y[idx].argmax()
        return x, y


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.liner_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        logits = self.liner_relu_stack(x)
        return logits


# GPUを使うかの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # 強制的にCPUを使わせる
print("Using {} device".format(device))

# ハイパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 1000
learning_rate = 1.0

# データの読み込み、モデルとオプティマイザの生成
train_dataloader = DataLoader(
    SpiralDataset(device), batch_size=batch_size, shuffle=True)

model = TwoLayerNet(input_size=2, hidden_size=hidden_size,
                    output_size=3).to(device)

# nn.LogSoftmaxとnn.NLLLoss(Negative Log Likelihood)を結合した損失関数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 学習で使用する変数
data_size = len(train_dataloader.dataset)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

start = time.time()

for epoch in range(max_epoch):
    for iters, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # 勾配を求め、パラメータを更新
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        loss_count += 1

        # if (iters+1) % 10 == 0:
        #     avg_loss = total_loss / loss_count
        #     print('| epoch %d |  iter %d / %d | loss %.2f' %
        #           (epoch + 1, iters + 1, max_iters, avg_loss))
        #     loss_list.append((avg_loss))
        #     total_loss, loss_count = 0, 0

print(time.time() - start)
