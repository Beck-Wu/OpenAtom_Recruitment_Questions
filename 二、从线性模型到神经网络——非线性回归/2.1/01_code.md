# 完整代码

> 源文件：[`linear_regression_friedman1.py`](./linear_regression_friedman1.py)

```python
"""
任务 2.一、机器学习深度学习基础知识问答：使用 PyTorch 搭建线性回归模型，拟合 make_friedman1 非线性回归数据
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
# 一、机器学习深度学习基础知识问答. 数据生成
# ---------------------------------------------------------------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)

# ---------------------------------------------------------------
# 2. 划分训练集 / 测试集
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 3. 标准化：对特征和目标都做标准化，有利于梯度下降收敛
#    （测试集必须用训练集的均值/方差进行变换，避免信息泄漏）
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
# 4. 模型定义：不含隐藏层的线性回归 —— 等价于 y = Wx + b
# ---------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(in_features=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
# 6. 测试集评估（还原到原始 y 尺度再算指标，更直观）
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    pred_test_s = model(X_test_t).numpy()

pred_test = y_scaler.inverse_transform(pred_test_s).ravel()

mse = mean_squared_error(y_test, pred_test)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - pred_test))
r2 = r2_score(y_test, pred_test)

print("\n===== 测试集结果 =====")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

# ---------------------------------------------------------------
# 7. 可视化：训练损失曲线 + 预测值 vs 真实值散点图
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(loss_history, color="#4C72B0")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training MSE (standardized)")
axes[0].set_title("Linear Model Training Loss")
axes[0].grid(alpha=0.3)

axes[1].scatter(y_test, pred_test, alpha=0.5, s=18, color="#DD8452", edgecolor="none")
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
axes[1].plot(lims, lims, "k--", linewidth=1, label="y = ŷ")
axes[1].set_xlabel("True y")
axes[1].set_ylabel("Predicted y")
axes[1].set_title(f"Linear Model: Predicted vs True (R²={r2:.3f})")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("linear_model_results.png", dpi=150)
plt.show()
```
