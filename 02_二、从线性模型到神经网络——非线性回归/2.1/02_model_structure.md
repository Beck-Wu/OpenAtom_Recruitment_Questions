# 模型结构

```
LinearRegressionModel(
  (linear): Linear(in_features=10, out_features=1)
)
```

- 无隐藏层、无激活函数，本质上是 `y = Wx + b` 的仿射变换
- 可训练参数共 11 个（10 个权重 + 1 个偏置）
- 损失函数：`MSELoss`
- 优化器：`Adam`（学习率 0.01）
- `batch_size = 64`，训练 300 个 epoch
- 训练前对特征 `X` 和目标 `y` 均做了标准化（`StandardScaler`）
