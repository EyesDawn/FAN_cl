import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_timeseries.normalizations.FAN import FAN
from torch_timeseries.datasets.traffic import Traffic

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 参数设置
seq_len = 96  # 输入序列长度
pred_len = 96  # 预测序列长度
batch_size = 32
epochs = 20
learning_rate = 0.001
freq_topk = 10  # 提取的主导频率成分数量
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 加载Traffic数据集
traffic_data = Traffic(root='./data')
data = traffic_data.data  # (17544, 862)

# 使用数据集的1%
num_features = data.shape[1]    # 获取列数（特征数量 862）
selected_features = np.random.choice(num_features, size=int(num_features * 0.01), replace=False)
data = data[:, selected_features]  # 只使用1%的特征

# 数据预处理
def create_sequences(data, seq_len, pred_len):
    x = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(x), np.array(y)

x_data, y_data = create_sequences(data, seq_len, pred_len)

# 划分训练集和测试集
train_ratio = 0.7
train_size = int(len(x_data) * train_ratio)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

# 转换为PyTorch张量
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义预测模型
class Predictor(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, freq_topk=10):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, enc_in]
        
        # 对每个特征进行预测
        batch_size, _, enc_in = x.shape
        predictions = []
        
        for i in range(enc_in):
            feature = x[:, :, i]
            pred = self.predictor(feature)
            # unsqueeze(-1) 在tensor的最后一个维度上增加一个维度
            predictions.append(pred.unsqueeze(-1))
        
        # 合并所有特征的预测结果
        output = torch.cat(predictions, dim=-1)
        
        return output

# 初始化模型
model = Predictor(seq_len=seq_len, pred_len=pred_len, enc_in=len(selected_features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
def train():
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试模型
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 训练和评估
train_losses = []
test_losses = []

print(f"开始训练，设备: {device}, 特征数量: {len(selected_features)}")
for epoch in range(epochs):
    train_loss = train()
    test_loss = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 测试损失: {test_loss:.4f}")

# 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.title('Training and test loss')

# 预测示例
model.eval()
with torch.no_grad():
    x_sample = x_test[:1]
    y_true = y_test[:1]
    y_pred = model(x_sample)

# 选择一个特征进行可视化
feature_idx = 0
plt.subplot(1, 2, 2)
plt.plot(range(seq_len), x_sample[0, :, feature_idx].cpu().numpy(), label='Historical data')
plt.plot(range(seq_len, seq_len + pred_len), y_true[0, :, feature_idx].cpu().numpy(), label='True value')
plt.plot(range(seq_len, seq_len + pred_len), y_pred[0, :, feature_idx].cpu().numpy(), label='Prediction')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.title(f'Prediction results for feature {feature_idx}')

plt.tight_layout()
plt.savefig('prediction_results.png')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print("模型已保存到 model.pth")