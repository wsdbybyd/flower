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
# 1. 模型定义
# -----------------------------------------------------------------------------


class Net(nn.Module):
    """轻量级学生模型/全局模型。"""

    def __init__(self):
        super().__init__()
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
    """较大的教师模型，用于地面蒸馏。"""

    def __init__(self):
        super().__init__()
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
# 2. 数据处理
# -----------------------------------------------------------------------------

fds = None
pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """客户端加载本地分区数据。"""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset_train_test():
    dataset = load_dataset("uoft-cs/cifar10")
    train_data = dataset["train"].with_transform(apply_transforms)
    test_data = dataset["test"].with_transform(apply_transforms)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)
    return trainloader, testloader


def load_centralized_dataset():
    trainloader, _ = load_centralized_dataset_train_test()
    return trainloader


def test_centralized_dataset():
    _, testloader = load_centralized_dataset_train_test()
    return testloader


# -----------------------------------------------------------------------------
# 3. 训练与蒸馏
# -----------------------------------------------------------------------------


def train_centralized(net, trainloader, epochs, lr, device):
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def distill_centralized(student, teacher, trainloader, epochs, lr, device, temp=2.0, alpha=0.5):
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
                F.softmax(teacher_logits / temp, dim=1),
            ) * (temp * temp)
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kl
            loss.backward()
            optimizer.step()


def distill_teacher_to_student(teacher_net, student_net, dataloader, device, epochs, lr, alpha, temp):
    distill_centralized(student_net, teacher_net, dataloader, epochs, lr, device, temp, alpha)


def distill_student_to_teacher(student_net, teacher_net, dataloader, device, epochs, lr):
    teacher_net.to(device)
    student_net.to(device)
    student_net.eval()
    teacher_net.train()

    optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0.9)
    criterion_kl = nn.KLDivLoss(reduction="batchmean").to(device)

    for _ in range(epochs):
        for batch in dataloader:
            images = batch["img"].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                logits_s = student_net(images)
            logits_t = teacher_net(images)
            loss = criterion_kl(F.log_softmax(logits_t, dim=1), F.softmax(logits_s, dim=1))
            loss.backward()
            optimizer.step()


# ---------------- 客户端本地训练 ----------------

def train_plain_sgd(net, trainloader, epochs, lr, device):
    """干净的普通本地 SGD / FedAvg 客户端训练。"""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0
    num_steps = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_steps += 1

    return running_loss / max(num_steps, 1)


# 为兼容旧代码，保留别名 train
train = train_plain_sgd


def train_apskd(net, trainloader, epochs, lr, device, a_T):
    """客户端 APSKD。a_T<=0 时严格退化为普通 SGD。"""
    if float(a_T) <= 0.0:
        return train_plain_sgd(net, trainloader, epochs, lr, device)

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction="batchmean").to(device)

    net.train()
    prev_model = None
    running_loss = 0.0
    num_steps = 0

    for epoch in range(1, epochs + 1):
        a_t = float(a_T) * (epoch / max(epochs, 1))

        if epoch > 1:
            prev_model = copy.deepcopy(net)
            prev_model.to(device)
            prev_model.eval()
            for p in prev_model.parameters():
                p.requires_grad = False

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
                    F.softmax(outputs_prev, dim=1),
                )
                loss = (1.0 - a_t) * loss_ce + a_t * loss_kl
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_steps += 1

    return running_loss / max(num_steps, 1)


def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)
