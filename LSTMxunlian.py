import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random
import pandas as pd

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据读取和预处理
def read_yolo_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    yolo_data = [list(map(float, line.strip().split())) for line in lines]
    return np.array(yolo_data)

def preprocess_yolo_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.startswith('sample_') and f.endswith('.txt')],
                   key=lambda x: int(x.split('_')[1].split('.')[0]))
    yolo_data_list = [read_yolo_data(os.path.join(folder_path, file)) for file in files]
    return np.concatenate(yolo_data_list, axis=0)

def create_sliding_windows(data, window_size, stride):
    windows, targets = [], []
    for i in range(0, len(data) - window_size - 10, stride):
        windows.append(data[i:i + window_size])
        targets.append(data[i + window_size + 10])
    return np.array(windows), np.array(targets)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 计算指标
def compute_metrics(outputs, labels):
    mse = nn.MSELoss()(outputs, labels)
    rmse = torch.sqrt(mse)
    mae = nn.L1Loss()(outputs, labels)
    return mse.item(), rmse.item(), mae.item()

# 可视化函数
def draw_metrics(metrics_dict, title="Training Metrics"):
    plt.figure(figsize=(10, 5))
    for label, values in metrics_dict.items():
        plt.plot(values, label=label)
    plt.title(title, fontsize=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./metrics/{title.replace(' ', '_')}.png")
    plt.show()

# 创建metrics文件夹
if not os.path.exists('./metrics'):
    os.makedirs('./metrics')

# 指定YOLO数据所在的文件夹路径
current_dir = os.path.dirname(__file__)  # 注意，运行时需要确保这一行正确
yolo_folder_path = os.path.join(current_dir, 'che', 'demo', 'labels')

# 进行预处理，得到整理后的YOLO数据
preprocessed_data = preprocess_yolo_folder(yolo_folder_path)

window_size = 50
stride = 1

sliding_windows, targets = create_sliding_windows(preprocessed_data, window_size, stride)

# 转换为PyTorch张量
sliding_windows = torch.tensor(sliding_windows, dtype=torch.float32).to(device)
targets = torch.tensor(targets, dtype=torch.float32).to(device)

# 数据集划分
dataset = TensorDataset(sliding_windows, targets)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 初始化模型、损失函数和优化器
input_size = sliding_windows.shape[2]
hidden_size = 400  # 可以根据需要调整
output_size = targets.shape[1]
model = LSTMModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率可以根据需要调整

# 将数据包装成 DataLoader
batch_size = 8  # 可以根据需要调整
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=100):
    metrics = {'Train Loss': [], 'Train MSE': [], 'Train RMSE': [], 'Train MAE': [],
               'Test Loss': [], 'Test MSE': [], 'Test RMSE': [], 'Test MAE': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mse, train_rmse, train_mae = [], [], [], []

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            mse, rmse, mae = compute_metrics(outputs, labels)
            train_loss.append(loss.item())
            train_mse.append(mse)
            train_rmse.append(rmse)
            train_mae.append(mae)

        metrics['Train Loss'].append(np.mean(train_loss))
        metrics['Train MSE'].append(np.mean(train_mse))
        metrics['Train RMSE'].append(np.mean(train_rmse))
        metrics['Train MAE'].append(np.mean(train_mae))

        model.eval()
        with torch.no_grad():
            test_loss, test_mse, test_rmse, test_mae = [], [], [], []
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                mse, rmse, mae = compute_metrics(outputs, labels)
                test_loss.append(loss.item())
                test_mse.append(mse)
                test_rmse.append(rmse)
                test_mae.append(mae)

            metrics['Test Loss'].append(np.mean(test_loss))
            metrics['Test MSE'].append(np.mean(test_mse))
            metrics['Test RMSE'].append(np.mean(test_rmse))
            metrics['Test MAE'].append(np.mean(test_mae))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {metrics["Train Loss"][-1]:.4f}, '
                  f'Test Loss: {metrics["Test Loss"][-1]:.4f}, Train MSE: {metrics["Train MSE"][-1]:.4f}, '
                  f'Test MSE: {metrics["Test MSE"][-1]:.4f}, Train RMSE: {metrics["Train RMSE"][-1]:.4f}, '
                  f'Test RMSE: {metrics["Test RMSE"][-1]:.4f}, Train MAE: {metrics["Train MAE"][-1]:.4f}, '
                  f'Test MAE: {metrics["Test MAE"][-1]:.4f}')

    # 绘制指标图像并保存
    for key in metrics:
        draw_metrics({key: metrics[key]}, key)

    # 导出指标到Excel
    df = pd.DataFrame(metrics)
    df.to_excel('./metrics/training_metrics.xlsx', index=False)

train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=100)

# 模型预测示例
model.eval()
with torch.no_grad():
    test_input = sliding_windows[:1]  # 假设我们用第一个样本进行测试
    print(test_input.shape)
    predicted = model(test_input).cpu().numpy()
    print("Predicted:", predicted)
