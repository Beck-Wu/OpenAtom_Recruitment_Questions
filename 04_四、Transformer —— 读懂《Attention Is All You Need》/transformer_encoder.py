import torch
import torch.nn as nn

# 01_一、机器学习深度学习基础知识问答. 基本参数
batch_size = 2
seq_len = 4
embed_dim = 8
num_heads = 2
num_layers = 2

# 2. 构造输入
x = torch.randn(
    batch_size,
    seq_len,
    embed_dim
)

# 3. 构建 Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers
)

# 4. 前向传播
out = encoder(x)

# 5. 查看结果
print("--- 初始参数运行结果 ---")
print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("Shape matches expectation:", x.shape == out.shape)

# 6. 尝试修改参数 (将 num_layers 修改为 3，或将 num_heads 修改为 4)
print("\n--- 修改参数后的实验 ---")
num_layers_modified = 3
encoder_layer_modified = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=4,  # 同时将 num_heads 改为 4
    batch_first=True
)
encoder_modified = nn.TransformerEncoder(
    encoder_layer_modified,
    num_layers=num_layers_modified
)

out_modified = encoder_modified(x)
print(f"Modified num_heads=4, num_layers={num_layers_modified}")
print("New Output shape:", out_modified.shape)