import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# -----------------------------------------------------------------------------
# 1. 模型定义 (Model Definitions)
# -----------------------------------------------------------------------------

class Net(nn.Module):
    """
    轻量级学生模型 (Lightweight Student Model)
    用于部署在 LEO 卫星上，计算和存储资源受限。
    
    架构: 简单的 2层 CNN + 3层全连接
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TeacherNet(nn.Module):
    """
    教师模型 (Teacher Model)
    部署在地面站，拥有更深的网络结构和更多的参数，用于指导学生模型。
    
    架构: 3层 CNN + 3层全连接 (通道数更多)
    """
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 32 -> 16
        x = self.pool(F.relu(self.conv2(x))) # 16 -> 8
        x = self.pool(F.relu(self.conv3(x))) # 8 -> 4
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------------------------------------------------------
# 2. 数据处理 (Data Handling)
# -----------------------------------------------------------------------------

fds = None
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    为指定客户端加载分区后的本地数据 (CIFAR-10)。
    使用 IID 分区策略。
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # 80% 训练, 20% 本地验证
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset():
    """
    加载地面站持有的校准数据集 (用于蒸馏)。
    取测试集的一个子集 (1000张) 以模拟少量有标签数据。
    """
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    subset = torch.utils.data.Subset(dataset, range(1000)) 
    return DataLoader(subset, batch_size=64, shuffle=True)

def test_centralized_dataset():
    """加载完整的测试集用于全局模型评估。"""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

# -----------------------------------------------------------------------------
# 3. 核心算法实现 (Algorithms)
# -----------------------------------------------------------------------------

def train_apskd(net, trainloader, epochs, lr, device, a_T):
    """
    [Client Side] 卫星自适应渐进式自蒸馏 (APSKD - Adaptive Progressive Self-Knowledge Distillation)
    
    算法描述 (论文 Section 2.3):
    卫星在本地训练时，不从地面下载额外的 Teacher 模型，而是利用"上一轮 Epoch 的自己"作为 Teacher。
    随着训练进行 (Epoch 增加)，模型越来越信任历史模型 (Teacher)，减少对 Hard Label 的依赖。
    
    Args:
        a_T (float): 线性增长权重的最终值 (Terminal value of a_t).
    """
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    # KLDivLoss: reduction='batchmean' 对应数学上的期望计算
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    net.train()
    
    # 初始状态下没有"上一轮模型"
    prev_model = None
    
    for epoch in range(1, epochs + 1):
        # [核心算法实现 1] 计算当前 Epoch 的混合权重 a_t (论文 Eq. 4)
        # a_t = a_T * (t / T)
        # 随着 t 增大，a_t 增大，表示蒸馏损失 (Soft Loss) 的比重增加
        a_t = a_T * (epoch / epochs)
        
        # [核心算法实现 2] 冻结并保存当前模型作为下一轮的 Teacher
        if epoch > 1:
            prev_model = copy.deepcopy(net)
            prev_model.eval()
            for p in prev_model.parameters():
                p.requires_grad = False

        running_loss = 0.0
        
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # 1. 当前模型 (Student) 前向传播
            outputs_curr = net(images)
            
            # 2. 计算 Hard Label 损失 (Cross Entropy)
            loss_ce = criterion_ce(outputs_curr, labels) 
            
            if prev_model is not None:
                # 3. 获取上一轮模型 (Teacher) 的软标签
                with torch.no_grad():
                    outputs_prev = prev_model(images)
                
                # 4. 计算 Soft Label 损失 (KL Divergence)
                # KL(Student || Teacher)
                loss_kl = criterion_kl(
                    F.log_softmax(outputs_curr, dim=1),
                    F.softmax(outputs_prev, dim=1)
                )
                
                # [核心算法实现 3] 线性混合 Loss (论文 Eq. 3 对应的优化目标)
                # Total Loss = (1 - a_t) * Hard_Loss + a_t * Soft_Loss
                loss = (1.0 - a_t) * loss_ce + a_t * loss_kl
                
            else:
                # 第一个 Epoch 仅使用 Hard Loss
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    return running_loss / len(trainloader)


def distill_teacher_to_student(teacher_net, student_net, dataloader, device, epochs, lr, alpha, temp):
    """
    [Server Side] Stage 1: 地面站下行蒸馏 (Teacher -> Student)
    
    描述 (论文 Section 2.2):
    在将全局模型 (Student) 下发给卫星之前，利用地面站的大模型 (Teacher) 对其进行指导/微调，
    使 Student 获得 Teacher 的部分知识作为良好的初始化参数。
    """
    teacher_net.to(device)
    student_net.to(device)
    teacher_net.eval()  # Teacher 始终固定
    student_net.train() # 更新 Student
    
    optimizer = torch.optim.SGD(student_net.parameters(), lr=lr, momentum=0.9)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    for _ in range(epochs):
        for batch in dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                logits_t = teacher_net(images)
            
            logits_s = student_net(images)
            
            # 标准 KD Loss: Hard Loss + Soft Loss
            loss_ce = criterion_ce(logits_s, labels)
            
            loss_kl = criterion_kl(
                F.log_softmax(logits_s / temp, dim=1),
                F.softmax(logits_t / temp, dim=1)
            ) * (temp * temp)
            
            loss = alpha * loss_ce + (1 - alpha) * loss_kl
            
            loss.backward()
            optimizer.step()


def distill_student_to_teacher(student_net, teacher_net, dataloader, device, epochs, lr):
    """
    [Server Side] Stage 3: 反向蒸馏 (Student -> Teacher)
    
    描述 (论文 Section 2.4):
    联邦聚合结束后，利用聚合后的全局 Student 模型反过来更新 Teacher 模型。
    这使得 Teacher 模型能够吸收卫星端学到的新知识 (Reverse Knowledge Transfer)。
    """
    teacher_net.to(device)
    student_net.to(device)
    student_net.eval()  # 此时聚合后的 Student 充当 Teacher 角色
    teacher_net.train() # Teacher 反而被更新
    
    optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0.9)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    for _ in range(epochs):
        for batch in dataloader:
            images = batch["img"].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                logits_s = student_net(images)
            
            logits_t = teacher_net(images)
            
            # Reverse Distillation Loss: 仅使用 KL 散度让 Teacher 逼近 Student 的分布
            loss = criterion_kl(
                F.log_softmax(logits_t, dim=1),
                F.softmax(logits_s, dim=1)
            )
            
            loss.backward()
            optimizer.step()


def test(net, testloader, device):
    """
    标准模型评估函数。
    计算测试集上的 Loss 和 Accuracy。
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