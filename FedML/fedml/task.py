import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# -----------------------------------------------------------------------------
# 2. 数据处理 (Data Handling)
# -----------------------------------------------------------------------------

fds = None
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """对批次数据应用图像变换。"""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    为指定客户端加载分区后的本地数据 (CIFAR-10)。
    使用 IID 分区策略。
    
    Args:
        partition_id: 客户端 ID
        num_partitions: 总分区数
        batch_size: 批次大小
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    
    # 划分 80% 训练 (用于 Meta-Training), 20% 测试 (用于 Meta-Testing)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def test_centralized_dataset():
    """
    加载完整的测试集用于全局模型评估 (Server Side)。
    用于监控 Meta-Model 的 Zero-shot 性能（虽然这不是元学习的主要目标）。
    """
    from datasets import load_dataset
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

# -----------------------------------------------------------------------------
# 3. 核心算法实现 (FO-MAML & Meta-Testing)
# -----------------------------------------------------------------------------

def train_fomaml(net, trainloader, device, alpha, beta, num_inner_steps=1):
    """
    [Client Side] FO-MAML (First-Order Model-Agnostic Meta-Learning) Training
    
    算法流程:
    1. 采样 Support Set (Batch A) & Query Set (Batch B)。
    2. Inner Loop: 使用 Batch A 更新模型的临时副本 (Fast Weights)。
    3. Outer Loop: 使用 Batch B 在临时副本上计算梯度。
    4. Meta Update: 将这些梯度应用到原始模型上 (First-Order 近似)。
    
    Args:
        net: 全局元模型 (Meta-Model)
        trainloader: 本地数据迭代器
        device: 计算设备
        alpha (float): Inner Loop Learning Rate (用于 Support Set 微调)
        beta (float): Outer Loop Learning Rate (用于最终模型更新)
        num_inner_steps (int): Inner Loop 的更新步数
    """
    net.to(device)
    net.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    iterator = iter(trainloader)
    
    total_outer_loss = 0.0
    steps = 0
    
    while True:
        try:
            # 1. 获取 Support Set (用于 Inner Loop)
            batch_sup = next(iterator)
            # 2. 获取 Query Set (用于 Outer Loop / Meta-Update)
            batch_qry = next(iterator)
        except StopIteration:
            break

        # ==========================================
        # Inner Loop (Adaptation on Support Set)
        # ==========================================
        # 克隆模型，以免修改原始模型参数 (Fast Weights)
        temp_model = copy.deepcopy(net)
        temp_model.train()
        inner_optimizer = torch.optim.SGD(temp_model.parameters(), lr=alpha)
        
        imgs_sup, lbls_sup = batch_sup["img"].to(device), batch_sup["label"].to(device)
        
        for _ in range(num_inner_steps):
            inner_optimizer.zero_grad()
            outputs_sup = temp_model(imgs_sup)
            loss_sup = criterion(outputs_sup, lbls_sup)
            loss_sup.backward()
            inner_optimizer.step()
            
        # ==========================================
        # Outer Loop (Gradient on Query Set)
        # ==========================================
        # 使用更新后的 temp_model 在 Query Set 上计算梯度
        imgs_qry, lbls_qry = batch_qry["img"].to(device), batch_qry["label"].to(device)
        
        outputs_qry = temp_model(imgs_qry)
        loss_qry = criterion(outputs_qry, lbls_qry)
        
        # FO-MAML 核心：使用 Adapted Model 在 Query Set 上的梯度
        # 作为 Original Model 的更新方向。忽略二阶导数项。
        grads_qry = torch.autograd.grad(loss_qry, temp_model.parameters())
        
        # ==========================================
        # Meta-Update (Update Original Model)
        # theta = theta - beta * grad_query
        # ==========================================
        with torch.no_grad():
            for param, grad in zip(net.parameters(), grads_qry):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                # 手动执行 SGD 更新逻辑
                param.data.sub_(beta * grad)

        total_outer_loss += loss_qry.item()
        steps += 1

    avg_loss = total_outer_loss / steps if steps > 0 else 0.0
    return avg_loss


def test_meta(net, testloader, device, adaptation_steps=5, adaptation_lr=0.01):
    """
    [Client Side] Meta-Testing / Meta-Evaluation
    
    流程:
    1. 从 testloader 中划分出一小部分作为 Support Set。
    2. 使用 Support Set 对模型进行微调 (Adaptation)。
    3. 在剩余的 Query Set 上评估微调后的模型。
    
    这验证了全局模型的“可适应性” (Meta-Generalization)。
    """
    net.to(device)
    
    # 1. 创建模型的深拷贝，以免修改全局评估逻辑中的原始模型
    meta_model = copy.deepcopy(net)
    meta_model.train() # 微调时需要处于训练模式
    
    optimizer = torch.optim.SGD(meta_model.parameters(), lr=adaptation_lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    iterator = iter(testloader)
    
    # --- Phase 1: Adaptation (Fine-tuning on Support Set) ---
    # 尝试从测试集中取出前 N 个 batch 进行微调
    params_updated = False
    for _ in range(adaptation_steps):
        try:
            batch = next(iterator)
            params_updated = True
        except StopIteration:
            # 如果数据不够，重新开始循环 (Edge case)
            iterator = iter(testloader)
            batch = next(iterator)
            
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = meta_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if not params_updated:
        return 0.0, 0.0

    # --- Phase 2: Evaluation (Testing on Query Set) ---
    meta_model.eval()
    correct, total_loss = 0, 0.0
    batches_count = 0
    
    with torch.no_grad():
        # 继续遍历迭代器中剩余的数据 (Query Set)
        for batch in iterator:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = meta_model(images)
            
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            batches_count += 1
            
    # 防止除零错误 (如果 adaptation 消耗了所有数据)
    if batches_count == 0:
        return 0.0, 0.0
            
    accuracy = correct / (batches_count * testloader.batch_size)
    loss = total_loss / batches_count
    
    return loss, accuracy


def test(net, testloader, device):
    """
    标准模型评估函数 (Zero-shot Evaluation)。
    用于服务器端全局评估，或不进行 Adaptation 的基准测试。
    """
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
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy