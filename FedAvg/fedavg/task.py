import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 Hugging Face datasets 库用于加载标准数据集
from datasets import load_dataset
# 导入 Flower 的联邦数据集相关模块，用于处理分区数据
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
# 导入图像转换工具：转换为 Tensor 和标准化
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """模型（简单的 CNN,改编自 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层：输入通道3(RGB)，输出通道6，卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义最大池化层：核大小2x2，步长2
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层：输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个全连接层：输入特征维度 16*5*5，输出维度 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层：输入维度 120，输出维度 84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层（输出层）：输入维度 84，输出维度 10 (对应10个类别)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> ReLU激活 -> 最大池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积 -> ReLU激活 -> 最大池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平多维特征图为一维向量 (batch_size, 16*5*5)
        x = x.view(-1, 16 * 5 * 5)
        # 全连接 -> ReLU激活
        x = F.relu(self.fc1(x))
        # 全连接 -> ReLU激活
        x = F.relu(self.fc2(x))
        # 输出层（无激活函数，通常配合 CrossEntropyLoss 使用）
        return self.fc3(x)


# 全局变量，用于缓存 FederatedDataset 实例，避免重复加载
fds = None

# 定义 PyTorch 图像预处理流程：
# 1. 将 PIL Image 或 numpy.ndarray 转换为 FloatTensor
# 2. 对图像进行标准化，均值和标准差均为 0.5 (将像素值从 [0, 1] 映射到 [-1, 1])
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """将转换操作应用到 FederatedDataset 的分区数据上。"""
    # 对批次中的每一张图像应用预处理
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """加载分区后的 CIFAR10 数据。"""
    # 仅初始化 `FederatedDataset` 一次（使用全局变量缓存）
    global fds
    if fds is None:
        # 初始化分区器，将数据划分为指定数量的分区（此处为 IID 分区）
        partitioner = IidPartitioner(num_partitions=num_partitions)
        # 创建联邦数据集对象，指定数据集名称和训练集的分区方式
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    # 加载指定 ID 的数据分区
    partition = fds.load_partition(partition_id)
    # 在每个节点上划分数据：80% 用于训练，20% 用于测试
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # 应用数据转换
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    # 构建训练集 DataLoader，开启 shuffle 以打乱数据顺序
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    # 构建测试集 DataLoader
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """加载完整的测试集并返回 DataLoader。"""
    # 加载完整的测试集（来自 Hugging Face Hub）
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    # 设置格式为 PyTorch tensor 并应用转换
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    # 返回 DataLoader，批次大小设为 128
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """在训练集上训练模型。"""
    net.to(device)  # 将模型移动到 GPU（如果可用）
    # 定义损失函数：交叉熵损失（适用于多分类任务）
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 定义优化器：随机梯度下降 (SGD)，包含动量参数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()  # 将模型设置为训练模式（启用 Dropout 等）
    running_loss = 0.0
    # 训练循环
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播：计算预测值
            loss = criterion(net(images), labels)
            # 反向传播：计算梯度
            loss.backward()
            # 参数更新
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
    # 计算平均训练损失
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """在测试集上验证模型。"""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    # 禁用梯度计算，节省内存并加快计算速度
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            # 前向传播：获取模型输出
            outputs = net(images)
            # 累加损失
            loss += criterion(outputs, labels).item()
            # 计算预测正确的样本数
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    # 计算准确率和平均损失
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
