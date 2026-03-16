import copy
from typing import Optional, Tuple

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


class TeacherNet(nn.Module):
    """
    教师模型 (Teacher Model) —— 地面站维护的更强模型 ω_T
    用于 Ground→Satellite 蒸馏，以及 Student→Teacher 反向蒸馏更新教师。
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# 2. 数据处理 (Data Handling)
# -----------------------------------------------------------------------------

fds = None
pytorch_transforms = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def apply_transforms(batch):
    """对批次数据应用图像变换。"""
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
    """
    from datasets import load_dataset 

    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def load_representative_dataset(
    *,
    split: str,
    num_samples: int,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """
    地面站代表数据集：用于 D_cal（校准/蒸馏）或 D_global（反向蒸馏更新教师）
    """
    from datasets import load_dataset 

    ds = load_dataset("uoft-cs/cifar10", split=split)
    if num_samples is not None and num_samples > 0 and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    ds = ds.with_transform(apply_transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# -----------------------------------------------------------------------------
# 3. 核心算法实现 (FO-MAML & Meta-Testing)
# -----------------------------------------------------------------------------

def train_fomaml(net, trainloader, device, alpha, beta, num_inner_steps=1):
    """
    [Client Side] FO-MAML (First-Order Model-Agnostic Meta-Learning) Training
    """
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss().to(device)
    iterator = iter(trainloader)

    total_outer_loss = 0.0
    steps = 0

    while True:
        try:
            # 1. Support Set (Inner Loop)
            batch_sup = next(iterator)
            # 2. Query Set (Outer Loop)
            batch_qry = next(iterator)
        except StopIteration:
            break

        # ----------------------------
        # Inner Loop (Support Adaptation)
        # ----------------------------
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

        # ----------------------------
        # Outer Loop (Query Gradient)
        # ----------------------------
        imgs_qry, lbls_qry = batch_qry["img"].to(device), batch_qry["label"].to(device)
        outputs_qry = temp_model(imgs_qry)
        loss_qry = criterion(outputs_qry, lbls_qry)

        grads_qry = torch.autograd.grad(loss_qry, temp_model.parameters())

        # ----------------------------
        # Meta-Update on Original Model
        # theta = theta - beta * grad_query
        # ----------------------------
        with torch.no_grad():
            for param, grad in zip(net.parameters(), grads_qry):
                param.data.sub_(beta * grad)

        total_outer_loss += float(loss_qry.item())
        steps += 1

    avg_loss = total_outer_loss / steps if steps > 0 else 0.0
    return avg_loss


def test_meta(net, testloader, device, adaptation_steps=5, adaptation_lr=0.01):
    """
    [Client Side] Meta-Testing / Meta-Evaluation
    Adapt -> Test
    """
    net.to(device)

    meta_model = copy.deepcopy(net)
    meta_model.train()

    optimizer = torch.optim.SGD(meta_model.parameters(), lr=adaptation_lr)
    criterion = nn.CrossEntropyLoss().to(device)

    iterator = iter(testloader)

    # --- Phase 1: Adaptation ---
    params_updated = False
    for _ in range(adaptation_steps):
        try:
            batch = next(iterator)
            params_updated = True
        except StopIteration:
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

    # --- Phase 2: Evaluation ---
    meta_model.eval()
    correct, total_loss = 0, 0.0
    total_samples = 0
    batches_count = 0

    with torch.no_grad():
        for batch in iterator:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = meta_model(images)

            total_loss += float(criterion(outputs, labels).item())
            preds = torch.max(outputs.data, 1)[1]
            correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
            batches_count += 1

    if batches_count == 0 or total_samples == 0:
        return 0.0, 0.0

    accuracy = correct / total_samples
    loss = total_loss / batches_count
    return loss, accuracy


def test(net, testloader, device):
    """
    标准模型评估函数 (Zero-shot Evaluation)。
    """
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)

            loss += float(criterion(outputs, labels).item())
            preds = torch.max(outputs.data, 1)[1]
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    accuracy = correct / total if total > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    return loss, accuracy


# -----------------------------------------------------------------------------
# 4. 双向蒸馏 (Bidirectional KD) + APSKD 
# -----------------------------------------------------------------------------

def kd_loss_lkd(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    y_true: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """
    论文 Eq.(2): L_KD = α * CE(y, σ(z_S)) + (1-α) * KL(σ(z_T/τ), σ(z_S/τ))
    """
    ce = F.cross_entropy(z_student, y_true)

    t = float(temperature)
    log_p_s = F.log_softmax(z_student / t, dim=1)
    p_t = F.softmax(z_teacher / t, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")

    a = float(alpha)
    return a * ce + (1.0 - a) * kl


@torch.no_grad()
def _copy_params_(dst: nn.Module, src: nn.Module) -> None:
    """把 src 参数复制到 dst（同结构模型）"""
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


def distill_teacher_to_student(
    *,
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    temperature: float,
    lr: float,
    epochs: int = 1,
) -> float:
    """
    Ground-Station-to-Satellite KD（正向蒸馏）：
    在地面站用 D_cal 最小化 L_KD(ω_S; ω_T, D_cal)，得到 ω_S^(0)，再下发给卫星。
    """
    teacher.to(device).eval()
    student.to(device).train()

    opt = torch.optim.SGD(student.parameters(), lr=float(lr))

    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)

            with torch.no_grad():
                z_t = teacher(x)

            z_s = student(x)
            loss = kd_loss_lkd(z_s, z_t, y_true=y, alpha=alpha, temperature=temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1

    return total / steps if steps > 0 else 0.0


def reverse_distill_student_to_teacher(
    *,
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    temperature: float,
    lr: float,
    epochs: int = 1,
) -> float:
    """
    Reverse Distillation（反向蒸馏，教师更新）：
    论文 Eq.(10): ω_T,t+1 = ω_T,t - γ ∇ L_KD(ω_T,t; ω_S,t+1, D_global)
    """
    student.to(device).eval()
    teacher.to(device).train()

    opt = torch.optim.SGD(teacher.parameters(), lr=float(lr))

    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)

            with torch.no_grad():
                z_s_as_teacher = student(x)

            z_t_as_student = teacher(x)
            loss = kd_loss_lkd(
                z_student=z_t_as_student,
                z_teacher=z_s_as_teacher,
                y_true=y,
                alpha=alpha,
                temperature=temperature,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1

    return total / steps if steps > 0 else 0.0


def train_apskd(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    temperature: float,
) -> float:
    """
    Satellite-side APSKD（论文 2.3 / Eq.(5)-(8)）
    - 使用上一 epoch 的模型作为 teacher（P^{t-1}）
    - 任务损失 Lt = CE(P^t(x), y)
    - 蒸馏损失 Ld = KL(P^t(x), P^{t-1}(x)) * Lt / (Lt + Lt_prev)
    - 总损失 LAPSKD = Lt + Ld
    """
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    eps = 1e-12
    t = float(temperature)

    prev_epoch_task_loss: Optional[float] = None
    total_loss_all, steps_all = 0.0, 0

    for ep in range(int(epochs)):
        prev_model = copy.deepcopy(model).to(device).eval()

        lt_prev_scalar = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0

        epoch_task_sum, epoch_task_steps = 0.0, 0

        for batch in trainloader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)

            z_cur = model(x)
            lt = F.cross_entropy(z_cur, y)

            with torch.no_grad():
                z_prev = prev_model(x)

            log_p_cur = F.log_softmax(z_cur / t, dim=1)
            p_prev = F.softmax(z_prev / t, dim=1)
            kl = F.kl_div(log_p_cur, p_prev, reduction="batchmean")

            w = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            ld = w * kl

            loss = lt + ld

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss_all += float(loss.item())
            steps_all += 1

            epoch_task_sum += float(lt.item())
            epoch_task_steps += 1

        prev_epoch_task_loss = (epoch_task_sum / epoch_task_steps) if epoch_task_steps > 0 else prev_epoch_task_loss

    return total_loss_all / steps_all if steps_all > 0 else 0.0


def train_fomaml_apskd(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    alpha: float,
    beta: float,
    num_inner_steps: int,
    temperature: float,
    epochs: int = 1,
) -> float:
    """
    [Client Side] 混合模式：FO-MAML + APSKD
    在外循环更新时，同时利用上一 Epoch 的模型进行软标签知识蒸馏，
    使得模型在快速适应新数据（Meta-Learning）的同时，不遗忘历史通用知识（APSKD）。
    """
    net.to(device).train()
    eps = 1e-12
    t = float(temperature)

    prev_epoch_task_loss = None
    total_outer_loss = 0.0
    steps = 0

    for ep in range(epochs):
        # 复制本 Epoch 开始时的模型作为 Teacher (P^{t-1})
        prev_model = copy.deepcopy(net).to(device).eval()
        
        # 如果上一轮没有 Lt_prev，用当前 epoch 的一个默认值
        lt_prev_scalar = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0

        criterion = nn.CrossEntropyLoss().to(device)
        iterator = iter(trainloader)

        epoch_task_sum, epoch_task_steps = 0.0, 0

        while True:
            try:
                # 1. Support Set (Inner Loop)
                batch_sup = next(iterator)
                # 2. Query Set (Outer Loop)
                batch_qry = next(iterator)
            except StopIteration:
                break

            # --- Inner Loop (Support Adaptation) ---
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

            # --- Outer Loop (Query Gradient + APSKD Distillation) ---
            imgs_qry, lbls_qry = batch_qry["img"].to(device), batch_qry["label"].to(device)
            
            # 当前适配后模型在 Query 集上的输出
            outputs_qry = temp_model(imgs_qry)
            lt = criterion(outputs_qry, lbls_qry) # 任务损失 (Task Loss)

            # 旧模型在 Query 集上的输出 (Teacher)
            with torch.no_grad():
                z_prev = prev_model(imgs_qry)

            # KL(P^t || P^{t-1})
            log_p_cur = F.log_softmax(outputs_qry / t, dim=1)
            p_prev = F.softmax(z_prev / t, dim=1)
            kl = F.kl_div(log_p_cur, p_prev, reduction="batchmean")

            # 自适应权重：Lt / (Lt + Lt_prev)
            w = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            ld = w * kl

            # 总损失 = 任务损失 + 蒸馏损失
            loss_total = lt + ld

            # --- Meta-Update on Original Model ---
            grads_qry = torch.autograd.grad(loss_total, temp_model.parameters())

            with torch.no_grad():
                for param, grad in zip(net.parameters(), grads_qry):
                    param.data.sub_(beta * grad)

            total_outer_loss += float(loss_total.item())
            steps += 1
            epoch_task_sum += float(lt.item())
            epoch_task_steps += 1

        prev_epoch_task_loss = (epoch_task_sum / epoch_task_steps) if epoch_task_steps > 0 else prev_epoch_task_loss

    avg_loss = total_outer_loss / steps if steps > 0 else 0.0
    return avg_loss


# -----------------------------------------------------------------------------
# 5. 便捷封装：构造 D_cal / D_global 的 DataLoader（供 server/fedml 调用）
# -----------------------------------------------------------------------------

def build_dcal_loader(
    *,
    num_samples: int = 2048,
    batch_size: int = 64,
    seed: int = 42,
    split: str = "train",
) -> DataLoader:
    """构造校准集 D_cal（默认从 train 子集采样）。"""
    return load_representative_dataset(
        split=split,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )


def build_dglobal_loader(
    *,
    num_samples: int = 4096,
    batch_size: int = 64,
    seed: int = 123,
    split: str = "train",
) -> DataLoader:
    """构造代表集 D_global（用于反向蒸馏更新 teacher）。"""
    return load_representative_dataset(
        split=split,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )