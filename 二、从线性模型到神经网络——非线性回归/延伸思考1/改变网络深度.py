"""
延伸思考1：改变网络深度
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# 1. 数据生成（与任务 2.1 完全相同，保证两次实验可比）
# ---------------------------------------------------------------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)

# ---------------------------------------------------------------
# 2. 划分训练集 / 测试集（同样的 random_state）
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 3. 标准化（同任务 2.1）
# ---------------------------------------------------------------
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train.reshape(-1, 1))
y_test_s = y_scaler.transform(y_test.reshape(-1, 1))

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test_s, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------------------------------------------
# 4. 模型定义：全连接神经网络（MLP）
#    结构：10 -> 64 (ReLU) -> 32 (ReLU) -> 1
#    比线性模型多了两个隐藏层和非线性激活函数
# ---------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, in_features, hidden1=64, hidden2 :int = 128,hidden3=32,hidden4=16 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2,hidden3),
            nn.ReLU(),
            nn.Linear(hidden3,hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLPModel(in_features=X_train.shape[1])
n_params = sum(p.numel() for p in model.parameters())
print(f"MLP 可训练参数量: {n_params}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ---------------------------------------------------------------
# 5. 训练
# ---------------------------------------------------------------
epochs = 300
loss_history = []

for epoch in range(1, epochs + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    loss_history.append(np.mean(epoch_losses))

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train MSE (standardized): {loss_history[-1]:.4f}")

# ---------------------------------------------------------------
# 6. 测试集评估（还原到原始 y 尺度）
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    pred_test_s = model(X_test_t).numpy()

pred_test = y_scaler.inverse_transform(pred_test_s).ravel()

mse = mean_squared_error(y_test, pred_test)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - pred_test))
r2 = r2_score(y_test, pred_test)

print("\n===== 测试集结果 (MLP) =====")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

# ---------------------------------------------------------------
# 7. 可视化
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(loss_history, color="#55A868")
axes[0].set_yscale("log")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training MSE (standardized, log scale)")
axes[0].set_title("MLP Training Loss")
axes[0].grid(alpha=0.3, which="both")

axes[1].scatter(y_test, pred_test, alpha=0.5, s=18, color="#55A868", edgecolor="none")
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
axes[1].plot(lims, lims, "k--", linewidth=1, label="y = ŷ")
axes[1].set_xlabel("True y")
axes[1].set_ylabel("Predicted y")
axes[1].set_title(f"MLP: Predicted vs True (R²={r2:.3f})")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("改变网络深度.png", dpi=150)
plt.show()
