import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
# [修改点 1] 引入 DirichletPartitioner 以支持 Non-IID 数据划分
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# -----------------------------------------------------------------------------
# 1. 模型定义 (Model Definitions)
# -----------------------------------------------------------------------------

class Net(nn.Module):
    """
    [学生模型] 轻量级 CNN
    用于部署在客户端 (Client)，以及作为全局模型 (Global Model) 下发。
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


class BigTeacherNet(nn.Module):
    """
    [教师模型] 大型 CNN
    部署在地面站/服务器，拥有更深的网络结构。
    """
    def __init__(self):
        super(BigTeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
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

# -----------------------------------------------------------------------------
# 2. 数据处理 (Data Handling)
# -----------------------------------------------------------------------------

fds = None
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Client 端加载本地分区数据"""
    global fds
    if fds is None:
        # [修改点 2] 使用 DirichletPartitioner 实现 Non-IID 划分
        # alpha=0.5 能产生明显的数据异质性，验证双向蒸馏的鲁棒性
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5, 
            min_partition_size=10,
            seed=42
        )
        
        # 如果想要退回普通的 IID，请注释上面的 partitioner 并使用下面这行：
        # partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id)
    # 本地划分 80% 训练, 20% 测试
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset_train_test():
    """Server 端加载完整数据集 (用于地面预训练和 Teacher 评估)"""
    # 注意：这里会下载整个 CIFAR-10
    dataset = load_dataset("uoft-cs/cifar10")
    train_data = dataset["train"].with_format("torch").with_transform(apply_transforms)
    test_data = dataset["test"].with_format("torch").with_transform(apply_transforms)
    
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # 测试集 batch_size 可大一点
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)
    return trainloader, testloader

def load_centralized_dataset():
    """仅返回训练集 (旧接口兼容)"""
    trainloader, _ = load_centralized_dataset_train_test()
    return trainloader

def test_centralized_dataset():
    """
    [修改点 3] 明确返回测试集
    修复了 server.py 中可能错误加载训练集进行评估的隐患
    """
    _, testloader = load_centralized_dataset_train_test()
    return testloader

# -----------------------------------------------------------------------------
# 3. 训练与蒸馏函数 (Training & Distillation Functions)
# -----------------------------------------------------------------------------

# --- A. 地面预训练相关 (Ground Pre-training) ---

def train_centralized(net, trainloader, epochs, lr, device):
    """普通监督训练 (用于从头训练 Teacher)"""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def distill_centralized(student, teacher, trainloader, epochs, lr, device, temp=2.0, alpha=0.5):
    """地面单向蒸馏 (Teacher -> Student, 用于初始化)"""
    student.to(device)
    teacher.to(device)
    teacher.eval()
    student.train()
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            loss_ce = criterion_ce(student_logits, labels)
            loss_kl = criterion_kl(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1)
            ) * (temp * temp)
            
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kl
            loss.backward()
            optimizer.step()

# --- B. 联邦过程相关 (FL Distillation) ---

def distill_teacher_to_student(teacher_net, student_net, dataloader, device, epochs, lr, alpha, temp):
    """Stage 1: 联邦每一轮开始前的下行蒸馏"""
    distill_centralized(student_net, teacher_net, dataloader, epochs, lr, device, temp, alpha)

def distill_student_to_teacher(student_net, teacher_net, dataloader, device, epochs, lr):
    """Stage 3: 联邦每一轮聚合后的反向蒸馏 (Student -> Teacher)"""
    teacher_net.to(device)
    student_net.to(device)
    student_net.eval()  # 全局 Student 作为老师
    teacher_net.train() # 更新 Teacher
    
    optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0.9)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    for _ in range(epochs):
        for batch in dataloader:
            images = batch["img"].to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                logits_s = student_net(images) # Student 输出
            
            logits_t = teacher_net(images)     # Teacher 输出
            
            # 反向只用 KL Loss
            loss = criterion_kl(
                F.log_softmax(logits_t, dim=1),
                F.softmax(logits_s, dim=1)
            )
            loss.backward()
            optimizer.step()

# --- C. 客户端训练 (Client Side) ---

def train_apskd(net, trainloader, epochs, lr, device, a_T):
    """客户端自适应渐进式自蒸馏 (APSKD)"""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction='batchmean').to(device)
    
    net.train()
    prev_model = None
    
    for epoch in range(1, epochs + 1):
        # 线性增长权重 (0 -> a_T)
        a_t = a_T * (epoch / epochs)
        
        if epoch > 1:
            prev_model = copy.deepcopy(net)
            prev_model.eval()
            for p in prev_model.parameters(): p.requires_grad = False

        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            
            outputs_curr = net(images)
            loss_ce = criterion_ce(outputs_curr, labels)
            
            if prev_model is not None:
                with torch.no_grad():
                    outputs_prev = prev_model(images)
                loss_kl = criterion_kl(
                    F.log_softmax(outputs_curr, dim=1),
                    F.softmax(outputs_prev, dim=1)
                )
                loss = (1.0 - a_t) * loss_ce + a_t * loss_kl
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            
    return 0.0

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        net.eval() # 确保 eval 模式
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)