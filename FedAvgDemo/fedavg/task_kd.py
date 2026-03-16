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
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """将转换操作应用到 FederatedDataset 的分区数据上。"""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """加载分区后的 CIFAR10 数据。"""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # 在每个节点上划分数据：80% 用于训练，20% 用于测试
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """加载完整的测试集并返回 DataLoader。"""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, teacher_net, trainloader, epochs, lr, device, kd_alpha=0.0, kd_temperature=1.0):
    """
    在训练集上训练模型，支持知识蒸馏 (Knowledge Distillation)。
    
    参数:
        net: 学生模型 (本地模型)
        teacher_net: 教师模型 (通常是上一轮的全局模型)
        kd_alpha: 蒸馏损失权重 (0.0 表示不使用蒸馏, 只用 CrossEntropy)
        kd_temperature: 蒸馏温度
    """
    net.to(device)
    teacher_net.to(device)
    teacher_net.eval() # 教师模型必须处于评估模式
    
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    # KL 散度损失，注意：PyTorch 的 KLDivLoss 期望 log_softmax 输入
    criterion_kl = torch.nn.KLDivLoss(reduction="batchmean").to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # 1. 学生模型前向传播
            student_outputs = net(images)
            
            # 2. 计算 Hard Loss (CrossEntropy)
            loss_ce = criterion_ce(student_outputs, labels)
            
            loss = loss_ce # 默认只用 CE Loss
            
            # 3. 知识蒸馏逻辑
            if kd_alpha > 0.0:
                with torch.no_grad():
                    # 教师模型前向传播 (不计算梯度)
                    teacher_outputs = teacher_net(images)
                
                # 计算 Soft Loss (KL Divergence)
                # 公式: KL( log_softmax(Student/T), softmax(Teacher/T) ) * T^2
                loss_kl = criterion_kl(
                    F.log_softmax(student_outputs / kd_temperature, dim=1),
                    F.softmax(teacher_outputs / kd_temperature, dim=1)
                ) * (kd_temperature * kd_temperature)
                
                # 4. 混合 Loss
                loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * loss_kl

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """在测试集上验证模型。"""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy