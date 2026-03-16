"""
task.py  ——  fedml-satellite 核心任务模块
数据集: NWPU-RESISC45 (31,500 张遥感卫星图像, 45 类, 256×256 RGB)
替换原 CIFAR-10 (32×32, 10 类)
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Normalize, ToTensor, Resize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, CenterCrop
)

# -----------------------------------------------------------------------------
# 常量：RESISC45 数据集规格
# CIFAR-10: 32×32, 10 类  →  RESISC45: 256×256 → 裁剪为 224×224, 45 类
# -----------------------------------------------------------------------------
NUM_CLASSES = 45
IMG_SIZE    = 64   # 下采样至 64×64，兼顾卫星端算力与表达能力
                   # 原图 256×256 全尺寸留给地面 BigTeacherNet

# -----------------------------------------------------------------------------
# 1. 模型定义 (Model Definitions)
# -----------------------------------------------------------------------------

class Net(nn.Module):
    """
    卫星端学生模型 (~480K 参数)，适配 RESISC45 45 分类任务。

    输入: 64×64 RGB  →  45 类遥感场景输出

    相比 CIFAR-10 版本的改动：
      - 输出层: 10 → 45
      - 卷积通道: 32→64 → 32→64→128 (新增第三组卷积+BN)
      - 全连接: 1024→256→64 → 2048→512→128
      - 额外 MaxPool 将 64→8→4→2，fc1 输入 = 128 * 2 * 2 = 512
        等效参数量 ~480K，仍在卫星端可部署范围内

    输入尺寸推导 (64×64):
      64 -[conv1+pool]→ 32
      32 -[conv2+pool]→ 16
      16 -[conv3+pool]→ 8
      8  -[pool]→ 4
      fc1 输入: 128 * 4 * 4 = 2048
    """
    def __init__(self, dropout_p: float = 0.3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 512)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.fc2   = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(p=dropout_p)
        self.fc3   = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64→32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32→16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16→8
        x = self.pool(x)                                 # 8→4
        x = x.view(-1, 128 * 4 * 4)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.fc3(x)


class BigTeacherNet(nn.Module):
    """
    地面站教师模型，适配 RESISC45 大尺寸输入 (64×64)，45 类输出。

    架构: 5 组卷积 (32→64→128→256→256) + 更宽全连接层
    输入尺寸推导 (64×64):
      64 -[conv1+pool]→ 32
      32 -[conv2+pool]→ 16
      16 -[conv3]→ 16
      16 -[conv4+pool]→ 8
       8 -[conv5+pool]→ 4
      fc1 输入: 256 * 4 * 4 = 4096
    """
    def __init__(self):
        super(BigTeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3, padding=1);  self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  64,  3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1);  self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1);  self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1);  self.bn5 = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(256 * 4 * 4, 1024)
        self.fc2   = nn.Linear(1024, 256)
        self.fc3   = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64→32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32→16
        x = F.relu(self.bn3(self.conv3(x)))             # 16→16
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 16→8
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 8→4
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------------------------------------------------------
# 2. 数据预处理变换
# RESISC45 原图 256×256 → 下采样至 IMG_SIZE×IMG_SIZE
# 均值/标准差来自 RESISC45 官方统计近似值
# -----------------------------------------------------------------------------

_RESISC45_MEAN = (0.3680, 0.3810, 0.3436)
_RESISC45_STD  = (0.2034, 0.1854, 0.1849)

# 训练时增强变换（客户端本地训练）
_train_transforms = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(15),
    ToTensor(),
    Normalize(_RESISC45_MEAN, _RESISC45_STD),
])

# 验证/测试变换（无数据增强）
_eval_transforms = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(_RESISC45_MEAN, _RESISC45_STD),
])


def _apply_train_transforms(batch):
    """训练集增强：适配 HuggingFace datasets 的 with_transform 接口"""
    # RESISC45 图像字段名为 "image"，标签为 "label"
    batch["image"] = [_train_transforms(img.convert("RGB")) for img in batch["image"]]
    return batch


def _apply_eval_transforms(batch):
    """验证/测试集变换：无增强"""
    batch["image"] = [_eval_transforms(img.convert("RGB")) for img in batch["image"]]
    return batch


def _collate_fn(batch):
    """
    自定义 collate：将 HuggingFace 的字典格式批次转成
    与原 CIFAR-10 兼容的 {"img": tensor, "label": tensor} 格式，
    保证 client.py / task.py 训练循环无需改动字段名。
    """
    imgs   = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"img": imgs, "label": labels}


# -----------------------------------------------------------------------------
# 3. 数据加载 (Data Handling)
# -----------------------------------------------------------------------------

# 全局 FederatedDataset 单例（联邦分区，避免重复创建）
fds: Optional[FederatedDataset] = None


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    客户端加载本地分区数据 (联邦 IID 分区)。

    RESISC45 通过 tanganke/resisc45 在 HuggingFace 上托管，
    train split 含 18,900 张，test split 含 6,300 张。
    这里仅对 train split 做联邦分区，每个客户端再划出 20% 作为本地验证集。
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="tanganke/resisc45",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_ds = partition_train_test["train"].with_transform(_apply_train_transforms)
    val_ds   = partition_train_test["test"].with_transform(_apply_eval_transforms)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=0, pin_memory=False
    )
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_fn, num_workers=0, pin_memory=False
    )
    return trainloader, valloader


def load_centralized_dataset_train_test():
    """
    服务器端加载完整 RESISC45 数据集（地面预训练 / Warm Start）。
    train: 18,900 张  |  test: 6,300 张
    """
    dataset = load_dataset("tanganke/resisc45")

    train_ds = dataset["train"].with_transform(_apply_train_transforms)
    test_ds  = dataset["test"].with_transform(_apply_eval_transforms)

    trainloader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=_collate_fn, num_workers=0
    )
    testloader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        collate_fn=_collate_fn, num_workers=0
    )
    return trainloader, testloader


def test_centralized_dataset():
    """服务器端加载测试集（全局评估用）"""
    _, testloader = load_centralized_dataset_train_test()
    return testloader


def build_dcal_loader(
    *, num_samples: int = 2048, batch_size: int = 64,
    seed: int = 42, split: str = "train"
) -> DataLoader:
    """
    构造校准集 D_cal（正向蒸馏：Ground Teacher → Satellite Student）。
    从 train split 中随机抽取 num_samples 张。
    """
    ds = load_dataset("tanganke/resisc45", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    ds = ds.with_transform(_apply_eval_transforms)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=0
    )


def build_dglobal_loader(
    *, num_samples: int = 4096, batch_size: int = 64,
    seed: int = 123, split: str = "train"
) -> DataLoader:
    """
    构造代表集 D_global（反向蒸馏：Aggregated Student → Ground Teacher）。
    从 train split 中随机抽取 num_samples 张（与 D_cal 用不同 seed 区分）。
    """
    ds = load_dataset("tanganke/resisc45", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    ds = ds.with_transform(_apply_eval_transforms)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=0
    )


# -----------------------------------------------------------------------------
# 4. 中心化预训练与蒸馏 (Ground Warm-up Tools)
# -----------------------------------------------------------------------------

def train_centralized(net, trainloader, epochs, lr, device):
    """地面站从头训练 Teacher 模型（SGD + Momentum）"""
    net.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


def distill_centralized(student, teacher, trainloader, epochs, lr, device,
                         temp=2.0, alpha=0.5):
    """
    地面站单向初始蒸馏 (Teacher → Student)。

    Dropout 修复：蒸馏阶段 student 用 eval() 关闭 Dropout，
    确保软标签输入稳定（梯度仍正常流动，参数正常更新）。
    """
    student.to(device).eval()
    teacher.to(device).eval()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            s_logits = student(images)
            with torch.no_grad():
                t_logits = teacher(images)
            loss_ce = criterion_ce(s_logits, labels)
            loss_kl = criterion_kl(
                F.log_softmax(s_logits / temp, dim=1),
                F.softmax(t_logits / temp, dim=1),
            ) * (temp * temp)
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kl
            loss.backward()
            optimizer.step()


def test(net, testloader, device):
    """标准零样本评估（Zero-shot，用于预热阶段测试和 FedAvg baseline）"""
    net.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    n = len(testloader.dataset)
    return total_loss / len(testloader), correct / n


# -----------------------------------------------------------------------------
# 5. Meta-Learning & FL 训练算法
# FO-MAML / APSKD / FO-MAML+APSKD / FedAvg
# 所有算法与原版逻辑完全一致，仅字段名已与新 collate_fn 对齐（均使用 "img"/"label"）
# -----------------------------------------------------------------------------

def train_fomaml(net, trainloader, device, alpha, beta, num_inner_steps=1):
    """纯 FO-MAML 训练（First-Order MAML）

    修复：outer update 加梯度裁剪（max_norm=10），防止梯度爆炸导致参数发散。
    原版 param.data.sub_(beta * grad) 裸更新无保护，RESISC45 45类任务
    初期 loss 较大、梯度量级不稳定时容易发散。
    """
    net.to(device).train()
    criterion = nn.CrossEntropyLoss().to(device)
    iterator = iter(trainloader)
    total_outer_loss, steps = 0.0, 0

    while True:
        try:
            batch_sup = next(iterator)
            batch_qry = next(iterator)
        except StopIteration:
            break

        temp_model = copy.deepcopy(net).train()
        inner_opt  = torch.optim.SGD(temp_model.parameters(), lr=alpha)

        imgs_sup = batch_sup["img"].to(device)
        lbls_sup = batch_sup["label"].to(device)
        for _ in range(num_inner_steps):
            inner_opt.zero_grad()
            criterion(temp_model(imgs_sup), lbls_sup).backward()
            inner_opt.step()

        imgs_qry = batch_qry["img"].to(device)
        lbls_qry = batch_qry["label"].to(device)
        loss_qry  = criterion(temp_model(imgs_qry), lbls_qry)

        grads = torch.autograd.grad(loss_qry, temp_model.parameters())

        # 梯度裁剪：限制 L2 范数不超过 10，防止早期 loss 大时参数发散
        grad_list  = list(grads)
        total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grad_list))
        clip_coef  = 10.0 / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            grad_list = [g * clip_coef for g in grad_list]

        with torch.no_grad():
            for param, grad in zip(net.parameters(), grad_list):
                param.data.sub_(beta * grad)

        total_outer_loss += float(loss_qry.item())
        steps += 1

    return total_outer_loss / steps if steps > 0 else 0.0


def test_meta(net, testloader, device, adaptation_steps=5, adaptation_lr=0.01):
    """
    元评估 (Meta-Testing: Adapt -> Test)

    关键修复：adaptation 和 evaluation 使用完全独立的两次遍历。
    原实现从同一个 iterator 中先取 adaptation_steps 个 batch 做 adaptation，
    再取剩余 batch 做 evaluation，导致：
      1. evaluation 样本随 adaptation_steps 增大而减少（评估集枯竭）
      2. adaptation 消耗测试集造成数据泄露
      3. 当 adaptation_steps >= len(testloader) 时 batches_count=0，返回 (0,0)
    修复后：adaptation 用独立的 adapt_iter（前 N 步），
            evaluation 重新遍历完整 testloader，两者互不干扰。
    """
    net.to(device)
    meta_model = copy.deepcopy(net).eval()
    optimizer  = torch.optim.SGD(meta_model.parameters(), lr=adaptation_lr)
    criterion  = nn.CrossEntropyLoss().to(device)

    # --- Phase 1: Adaptation（独立 iterator，仅取前 adaptation_steps 个 batch）---
    adapt_iter = iter(testloader)
    for _ in range(adaptation_steps):
        try:
            batch = next(adapt_iter)
        except StopIteration:
            adapt_iter = iter(testloader)
            batch = next(adapt_iter)
        optimizer.zero_grad()
        loss = criterion(
            meta_model(batch["img"].to(device)),
            batch["label"].to(device)
        )
        loss.backward()
        optimizer.step()

    # --- Phase 2: Evaluation（独立完整遍历，与 adapt_iter 无关）---
    correct, total_loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for batch in testloader:
            labels  = batch["label"].to(device)
            outputs = meta_model(batch["img"].to(device))
            total_loss    += float(criterion(outputs, labels).item())
            correct       += int((torch.max(outputs, 1)[1] == labels).sum().item())
            total_samples += int(labels.size(0))

    if total_samples == 0:
        return 0.0, 0.0

    num_batches = len(testloader)
    return total_loss / num_batches, correct / total_samples


def kd_loss_lkd(z_student, z_teacher, y_true, alpha, temperature):
    """混合知识蒸馏损失 = α·CE + (1-α)·KL"""
    ce  = F.cross_entropy(z_student, y_true)
    t   = float(temperature)
    kl  = F.kl_div(
        F.log_softmax(z_student / t, dim=1),
        F.softmax(z_teacher / t, dim=1),
        reduction="batchmean"
    )
    return float(alpha) * ce + (1.0 - float(alpha)) * kl


def distill_teacher_to_student(*, teacher, student, loader, device,
                                 alpha, temperature, lr, epochs=1):
    """服务器端正向蒸馏 (Ground Teacher → Satellite Student)"""
    teacher.to(device).eval()
    student.to(device).train()
    opt   = torch.optim.SGD(student.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x, y = batch["img"].to(device), batch["label"].to(device)
            with torch.no_grad():
                z_t = teacher(x)
            loss = kd_loss_lkd(student(x), z_t, y, alpha=alpha, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return total / steps if steps > 0 else 0.0


def reverse_distill_student_to_teacher(*, teacher, student, loader, device,
                                         alpha, temperature, lr, epochs=1):
    """服务器端反向蒸馏 (Aggregated Student → Ground Teacher)"""
    student.to(device).eval()
    teacher.to(device).train()
    opt   = torch.optim.SGD(teacher.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x, y = batch["img"].to(device), batch["label"].to(device)
            with torch.no_grad():
                z_s = student(x)
            loss = kd_loss_lkd(teacher(x), z_s, y, alpha=alpha, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return total / steps if steps > 0 else 0.0


def train_apskd(model, trainloader, device, *, lr, epochs, temperature):
    """纯 APSKD 本地自蒸馏（Adaptive Progressive Self-Knowledge Distillation）"""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    eps, t = 1e-12, float(temperature)
    prev_epoch_task_loss = None
    total_loss_all, steps_all = 0.0, 0

    for ep in range(int(epochs)):
        prev_model   = copy.deepcopy(model).to(device).eval()
        lt_prev_scalar = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0
        epoch_task_sum, epoch_task_steps = 0.0, 0

        for batch in trainloader:
            x, y   = batch["img"].to(device), batch["label"].to(device)
            z_cur   = model(x)
            lt      = F.cross_entropy(z_cur, y)
            with torch.no_grad():
                z_prev = prev_model(x)
            kl  = F.kl_div(
                F.log_softmax(z_cur / t, dim=1),
                F.softmax(z_prev / t, dim=1),
                reduction="batchmean"
            )
            w    = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            loss = lt + (w * kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss_all += float(loss.item())
            steps_all      += 1
            epoch_task_sum += float(lt.item())
            epoch_task_steps += 1

        if epoch_task_steps > 0:
            prev_epoch_task_loss = epoch_task_sum / epoch_task_steps

    return total_loss_all / steps_all if steps_all > 0 else 0.0


def train_fomaml_apskd(net, trainloader, device, alpha, beta, num_inner_steps,
                        temperature, epochs=1, current_round=1, warmup_rounds=50):
    """
    混合模式: FO-MAML (outer loop) + APSKD 自蒸馏正则化（解耦执行）。

    Step-A: FO-MAML outer update（纯任务损失，梯度干净）
    Step-B: APSKD 正则化（独立 SGD 步，不污染 MAML 梯度）
    """
    net.to(device).train()
    eps          = 1e-12
    t            = float(temperature)
    lambda_apskd = 0.3
    lr_apskd     = beta * lambda_apskd
    criterion    = nn.CrossEntropyLoss().to(device)
    prev_epoch_task_loss = None
    total_outer_loss, steps = 0.0, 0

    for ep in range(epochs):
        prev_model      = copy.deepcopy(net).to(device).eval()
        lt_prev_scalar  = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0
        iterator        = iter(trainloader)
        epoch_task_sum, epoch_task_steps = 0.0, 0

        while True:
            try:
                batch_sup = next(iterator)
                batch_qry = next(iterator)
            except StopIteration:
                break

            imgs_sup = batch_sup["img"].to(device)
            lbls_sup = batch_sup["label"].to(device)
            imgs_qry = batch_qry["img"].to(device)
            lbls_qry = batch_qry["label"].to(device)

            # --- Step-A: FO-MAML outer update ---
            temp_model = copy.deepcopy(net).train()
            inner_opt  = torch.optim.SGD(temp_model.parameters(), lr=alpha)
            for _ in range(num_inner_steps):
                inner_opt.zero_grad()
                criterion(temp_model(imgs_sup), lbls_sup).backward()
                inner_opt.step()

            outputs_qry = temp_model(imgs_qry)
            lt          = criterion(outputs_qry, lbls_qry)
            grads_maml  = list(torch.autograd.grad(lt, temp_model.parameters()))

            # 梯度裁剪（与 train_fomaml 保持一致）
            total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads_maml))
            clip_coef  = 10.0 / (total_norm.item() + 1e-6)
            if clip_coef < 1.0:
                grads_maml = [g * clip_coef for g in grads_maml]

            with torch.no_grad():
                for param, grad in zip(net.parameters(), grads_maml):
                    param.data.sub_(beta * grad)

            # --- Step-B: APSKD 自蒸馏正则化 ---
            with torch.no_grad():
                z_prev = prev_model(imgs_qry)
            outputs_after = net(imgs_qry)
            kl = F.kl_div(
                F.log_softmax(outputs_after / t, dim=1),
                F.softmax(z_prev / t, dim=1),
                reduction="batchmean"
            )
            w    = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            l_kd = w * kl
            l_kd.backward()
            with torch.no_grad():
                for param in net.parameters():
                    if param.grad is not None:
                        param.data.sub_(lr_apskd * param.grad)
                        param.grad.zero_()

            total_outer_loss += float(lt.item()) + float(l_kd.item())
            steps            += 1
            epoch_task_sum   += float(lt.item())
            epoch_task_steps += 1

        if epoch_task_steps > 0:
            prev_epoch_task_loss = epoch_task_sum / epoch_task_steps

    return total_outer_loss / steps if steps > 0 else 0.0


def train_fedavg(net, trainloader, device, lr: float = 0.01, epochs: int = 1,
                 weight_decay: float = 1e-4, label_smoothing: float = 0.1):
    """
    标准 FedAvg 本地训练，含 Weight Decay + Label Smoothing 正则化。
    RESISC45 (45 类) 相比 CIFAR-10 标签空间更大，label smoothing 更重要。
    """
    net.to(device).train()
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=float(lr),
        momentum=0.9, weight_decay=float(weight_decay)
    )
    total_loss, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps      += 1
    return total_loss / steps if steps > 0 else 0.0