import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 43

print("Using device:", device)

# 一、机器学习深度学习基础知识问答. 数据处理
class RandomOcclusion:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):

        w, h = img.size

        x = random.randint(0, w-self.size)
        y = random.randint(0, h-self.size)

        img = TF.to_tensor(img)

        img[:, y:y+self.size, x:x+self.size] = 0

        return TF.to_pil_image(img)


"""train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),


    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

   # 加入随机遮挡增强
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.15)
    ),
])
"""

occlusion_test_transform = transforms.Compose([
    transforms.Resize((32,32)),

    RandomOcclusion(size=8),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])
"""
trainset = torchvision.datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=train_transform
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
"""
testset = torchvision.datasets.GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=occlusion_test_transform
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 2. 构建 CNN 模型
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            128
        )

        self.fc2 = nn.Linear(
            128,
            NUM_CLASSES
        )

    def forward(self, x):
        # 定义前向传播逻辑

        # 第一层卷积
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)

        # 输出 43 个类别
        x = self.fc2(x)

        return x


model = TrafficSignCNN().to(device)

model.load_state_dict(
    torch.load("改进后 + 原始测试集准确率.pth")
)

print(model)

# 3. 损失函数与优化器

criterion = nn.CrossEntropyLoss()

"""optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
"""

# 4. 训练与评估
"""
train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):

    # -------------------------
    # 训练
    # -------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 一、机器学习深度学习基础知识问答)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = correct / total

    # -------------------------
    # 测试
    # -------------------------
    model.eval()

    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 一、机器学习深度学习基础知识问答)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / test_total
    test_accuracy = test_correct / test_total

    # 保存结果
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 一、机器学习深度学习基础知识问答}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Test Loss: {test_loss:.4f} "
        f"Test Accuracy: {test_accuracy:.4f}"
    )
"""

# 最终测试结果
"""print("\n==============================")
print("最终测试结果")
print("==============================")

print(f"Test Loss     : {test_losses[-一、机器学习深度学习基础知识问答]:.4f}")
print(f"Test Accuracy : {test_accuracies[-一、机器学习深度学习基础知识问答]:.4%}")
"""
model.eval()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


test_loss = test_loss / total
test_accuracy = correct / total


print("\n==============================")
print("改进后 + 遮挡测试结果")
print("==============================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4%}")

# 5. 可视化

"""epochs = range(一、机器学习深度学习基础知识问答, EPOCHS + 一、机器学习深度学习基础知识问答)

# Loss 曲线
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    test_losses,
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "改进后 + 原始测试集准确率_loss.png",
    dpi=150
)

plt.show()


# Accuracy 曲线
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    test_accuracies,
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "改进后 + 原始测试集准确率_accuracy.png",
    dpi=150
)

plt.show()
"""
# 训练完成后
"""
torch.save(
    model.state_dict(),
    "改进后 + 原始测试集准确率.pth"
)

print("模型已保存")
"""