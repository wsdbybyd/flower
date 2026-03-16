import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 Hugging Face datasets 库用于加载标准数据集 (用于地面蒸馏)
from datasets import load_dataset
# 导入 Flower 的联邦数据集相关模块，用于处理分区数据 (用于联邦学习)
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
# 导入图像转换工具
from torchvision.transforms import Compose, Normalize, ToTensor


# ==========================================
# 1. 模型定义
# ==========================================

class Net(nn.Module):
    """
    [学生模型] 轻量级 CNN (改自 'PyTorch: A 60 Minute Blitz')
    这是最终部署到客户端的模型，参数量较小。
    """
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
        # 输出层
        return self.fc3(x)


class BigTeacherNet(nn.Module):
    """
    [教师模型] 大型 CNN
    参数量更大，结构更深/更宽，仅在地面端（服务器）使用，用于指导学生模型。
    """
    def __init__(self):
        super(BigTeacherNet, self).__init__()
        # 增加通道数：6->64, 16->128
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # 经过两次池化 (32->16->8)，特征图大小为 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 2. 数据处理与辅助函数
# ==========================================

# 全局变量，用于缓存 FederatedDataset 实例，避免重复加载
fds = None

# 定义 PyTorch 图像预处理流程
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def count_parameters(model):
    """计算模型可训练参数的数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_transforms(batch):
    """将转换操作应用到 FederatedDataset 的分区数据上。"""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    [FL 客户端] 加载分区后的 CIFAR10 数据。
    用于联邦学习过程中客户端的数据加载。
    """
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

def load_centralized_dataset_train_test():
    """
    [服务端] 加载完整的 CIFAR10 训练集和测试集。
    用于地面端的教师模型训练和蒸馏过程。
    """
    # 加载 train 和 test split
    dataset = load_dataset("uoft-cs/cifar10")
    
    # 转换为 PyTorch 格式并应用预处理
    train_data = dataset["train"].with_format("torch").with_transform(apply_transforms)
    test_data = dataset["test"].with_format("torch").with_transform(apply_transforms)
    
    # 注意：在实际大规模训练时，可以调整 num_workers 等参数
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    return trainloader, testloader

def load_centralized_dataset():
    """
    [服务端] 仅加载测试集。
    兼容旧代码，用于 FL 每一轮的全局评估。
    """
    _, testloader = load_centralized_dataset_train_test()
    return testloader


# ==========================================
# 3. 训练与蒸馏逻辑
# ==========================================

def train_centralized(net, trainloader, epochs, lr, device):
    """
    [服务端] 普通监督训练。
    用于在地面端从头训练教师模型。
    """
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    print(f"Starting standard training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"  [Standard Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def distill_centralized(student, teacher, trainloader, epochs, lr, device, temp=2.0, alpha=0.5):
    """
    [服务端] 地面蒸馏训练。
    利用预训练好的 Teacher 指导 Student。
    
    参数:
        student: 待训练的学生模型
        teacher: 已训练好的教师模型
        temp (temperature): 蒸馏温度，越高概率分布越平滑
        alpha: 蒸馏损失的权重 (1-alpha 为真实标签损失权重)
    """
    student.to(device)
    teacher.to(device)
    teacher.eval() # 冻结教师模型
    student.train()
    
    criterion_ce = nn.CrossEntropyLoss()
    # KLDivLoss 默认 reduction='mean'，但为了数学上的正确性通常使用 'batchmean'
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    
    print(f"Starting distillation (Alpha={alpha}, Temp={temp})...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # 1. 学生模型前向传播
            student_logits = student(images)
            
            # 2. 教师模型前向传播 (不计算梯度)
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            # 3. 计算 Hard Loss (与真实标签对比)
            loss_ce = criterion_ce(student_logits, labels)
            
            # 4. 计算 Soft Loss (与教师输出对比)
            # 公式: KL( log_softmax(Student/T), softmax(Teacher/T) ) * T^2
            loss_kl = criterion_kl(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1)
            ) * (temp * temp)
            
            # 5. 混合 Loss
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kl
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(trainloader)
        print(f"  [Distillation] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def train(net, teacher_net, trainloader, epochs, lr, device, kd_alpha=0.0, kd_temperature=1.0):
    """
    [FL 客户端] 本地训练函数。
    支持在联邦学习过程中进行训练，也可以配置进行本地蒸馏 (Client-side Distillation)。
    
    参数:
        net: 学生模型 (本地模型)
        teacher_net: 教师模型 (通常是上一轮的全局模型，用于本地正则化/蒸馏)
        kd_alpha: 蒸馏损失权重 (0.0 表示不使用蒸馏, 只用 CrossEntropy)
        kd_temperature: 蒸馏温度
    """
    net.to(device)
    teacher_net.to(device)
    teacher_net.eval() # 教师模型必须处于评估模式
    
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
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
            
            # 3. 知识蒸馏逻辑 (如果启用)
            if kd_alpha > 0.0:
                with torch.no_grad():
                    # 教师模型前向传播 (不计算梯度)
                    teacher_outputs = teacher_net(images)
                
                # 计算 Soft Loss (KL Divergence)
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
    """
    [通用] 测试/评估函数。
    计算 Loss 和 Accuracy。
    """
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