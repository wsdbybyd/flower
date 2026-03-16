import copy
from typing import Optional, Tuple

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


class BigTeacherNet(nn.Module):
    """
    [教师模型] 与 FedAvg 实验完全对齐的深度 CNN。
    部署在地面站/服务器，用于生成高质量的软标签。
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

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Client 端加载本地分区数据"""
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
    """Server 端加载完整数据集 (用于地面预训练 / Warm Start)"""
    dataset = load_dataset("uoft-cs/cifar10")
    train_data = dataset["train"].with_format("torch").with_transform(apply_transforms)
    test_data = dataset["test"].with_format("torch").with_transform(apply_transforms)
    
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=False)
    return trainloader, testloader

def test_centralized_dataset():
    """Server 端加载测试集 (用于全局评估)"""
    _, testloader = load_centralized_dataset_train_test()
    return testloader

def build_dcal_loader(*, num_samples: int = 2048, batch_size: int = 64, seed: int = 42, split: str = "train") -> DataLoader:
    """构造校准集 D_cal (用于下发前正向蒸馏)"""
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def build_dglobal_loader(*, num_samples: int = 4096, batch_size: int = 64, seed: int = 123, split: str = "train") -> DataLoader:
    """构造代表集 D_global (用于聚合后反向蒸馏)"""
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# 3. 中心化预训练与蒸馏 (Ground Warm-up Tools)
# -----------------------------------------------------------------------------

def train_centralized(net, trainloader, epochs, lr, device):
    """地面站从头训练 Teacher 模型"""
    net.to(device).train()
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
    """地面站单向初始蒸馏 (Teacher -> Student)"""
    student.to(device).train()
    teacher.to(device).eval()
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

def test(net, testloader, device):
    """标准零样本评估 (Zero-shot Evaluation, 用于预热阶段测试)"""
    net.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)

# -----------------------------------------------------------------------------
# 4. Meta-Learning & FL 算法 (FO-MAML / APSKD / Bidirectional KD)
# -----------------------------------------------------------------------------

def train_fomaml(net, trainloader, device, alpha, beta, num_inner_steps=1):
    """纯 FO-MAML 训练"""
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
        inner_optimizer = torch.optim.SGD(temp_model.parameters(), lr=alpha)
        imgs_sup, lbls_sup = batch_sup["img"].to(device), batch_sup["label"].to(device)
        for _ in range(num_inner_steps):
            inner_optimizer.zero_grad()
            loss_sup = criterion(temp_model(imgs_sup), lbls_sup)
            loss_sup.backward()
            inner_optimizer.step()
        imgs_qry, lbls_qry = batch_qry["img"].to(device), batch_qry["label"].to(device)
        loss_qry = criterion(temp_model(imgs_qry), lbls_qry)
        grads_qry = torch.autograd.grad(loss_qry, temp_model.parameters())
        with torch.no_grad():
            for param, grad in zip(net.parameters(), grads_qry):
                param.data.sub_(beta * grad)
        total_outer_loss += float(loss_qry.item())
        steps += 1
    return total_outer_loss / steps if steps > 0 else 0.0


def _get_xy_from_batch(batch, device):
    """Support batches in dict form (HF datasets) or tuple/list (x, y, ...)."""
    if isinstance(batch, dict):
        # Common keys in this project: {"img": ..., "label": ...}
        if "img" in batch:
            x = batch["img"]
        elif "image" in batch:
            x = batch["image"]
        else:
            # Fallback: first tensor-like value
            x = next(iter(batch.values()))
        if "label" in batch:
            y = batch["label"]
        elif "labels" in batch:
            y = batch["labels"]
        else:
            raise KeyError("Batch dict must contain 'label' or 'labels'.")
        return x.to(device), y.to(device)
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        return x.to(device), y.to(device)
    raise TypeError("Expected batch to be dict or (x, y, ...) tuple/list.")


@torch.no_grad()
def _eval_loop_meta(model, loader_or_iter, device):
    """Return (avg_loss, avg_acc, extra_metrics). Loss is sample-weighted."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    # Confidence / calibration diagnostics
    conf_sum = 0.0
    conf_wrong_sum = 0.0
    n_wrong = 0

    batch_losses = []

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    for batch in loader_or_iter:
        x, y = _get_xy_from_batch(batch, device)
        logits = model(x)  # logits only (do NOT softmax before CE)
        loss = criterion(logits, y)

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())

        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        conf_sum += float(conf.sum().item())

        wrong_mask = (pred != y)
        if wrong_mask.any():
            conf_wrong_sum += float(conf[wrong_mask].sum().item())
            n_wrong += int(wrong_mask.sum().item())

        batch_losses.append(float(loss.item()))

    avg_loss = total_loss / max(total_n, 1)
    avg_acc = total_correct / max(total_n, 1)

    avg_conf = conf_sum / max(total_n, 1)
    avg_conf_wrong = conf_wrong_sum / max(n_wrong, 1)

    if batch_losses:
        s = sorted(batch_losses)
        idx = int(0.95 * (len(s) - 1))
        loss_p95 = s[idx]
        loss_max = s[-1]
    else:
        loss_p95 = 0.0
        loss_max = 0.0

    extra = {
        "avg_conf": float(avg_conf),
        "avg_conf_wrong": float(avg_conf_wrong),
        "loss_p95": float(loss_p95),
        "loss_max": float(loss_max),
        "n_eval_samples": float(total_n),
        "n_wrong_samples": float(n_wrong),
    }
    return float(avg_loss), float(avg_acc), extra


def test_meta(
    net,
    testloader,
    device,
    adaptation_steps: int = 5,
    adaptation_lr: float = 0.01,
    *,
    do_adapt: bool = True,
    reset_iterator_after_adapt: bool = True,
    return_extra: bool = False,
):
    """E0-enabled Meta-Testing.

    - do_adapt=False: zero-shot evaluation (no adaptation).
    - do_adapt=True : post-adaptation evaluation (adapt on a few batches, then eval).
    - reset_iterator_after_adapt=True (recommended): evaluate on FULL testloader after adaptation.
      If False: evaluate only on remaining batches (diagnostic for iterator bias).
    - return_extra=True: also return diagnostics dict.
    """
    device = torch.device(device)
    net.to(device)

    # Clone model to avoid polluting the passed-in net during evaluation
    meta_model = copy.deepcopy(net).to(device)

    # Zero-shot
    if (not do_adapt) or adaptation_steps <= 0:
        loss, acc, extra = _eval_loop_meta(meta_model, testloader, device)
        return (loss, acc, extra) if return_extra else (loss, acc)

    # --- Adaptation ---
    meta_model.train()
    optimizer = torch.optim.SGD(meta_model.parameters(), lr=float(adaptation_lr))
    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    iterator = iter(testloader)
    steps_done = 0
    for _ in range(int(adaptation_steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        x, y = _get_xy_from_batch(batch, device)
        optimizer.zero_grad()
        logits = meta_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        steps_done += 1

    if steps_done == 0:
        # Nothing adapted; return zero-shot instead
        loss, acc, extra = _eval_loop_meta(meta_model, testloader, device)
        return (loss, acc, extra) if return_extra else (loss, acc)

    # --- Evaluation ---
    if reset_iterator_after_adapt:
        eval_source = testloader  # full test set
    else:
        eval_source = iterator    # remaining batches only

    loss, acc, extra = _eval_loop_meta(meta_model, eval_source, device)
    return (loss, acc, extra) if return_extra else (loss, acc)


def test_meta_dual(
    net,
    testloader,
    device,
    adaptation_steps: int = 5,
    adaptation_lr: float = 0.01,
    *,
    reset_iterator_after_adapt: bool = True,
):
    """Run both zero-shot and post-adaptation meta-test; return a dict for logging."""
    zs = test_meta(
        net, testloader, device,
        adaptation_steps=adaptation_steps, adaptation_lr=adaptation_lr,
        do_adapt=False, reset_iterator_after_adapt=reset_iterator_after_adapt,
        return_extra=True,
    )
    ad = test_meta(
        net, testloader, device,
        adaptation_steps=adaptation_steps, adaptation_lr=adaptation_lr,
        do_adapt=True, reset_iterator_after_adapt=reset_iterator_after_adapt,
        return_extra=True,
    )
    zs_loss, zs_acc, zs_extra = zs
    ad_loss, ad_acc, ad_extra = ad
    return {
        "zero_shot": {"loss": zs_loss, "acc": zs_acc, **zs_extra},
        "post_adapt": {"loss": ad_loss, "acc": ad_acc, **ad_extra},
    }

def kd_loss_lkd(z_student, z_teacher, y_true, alpha, temperature):
    """计算混合蒸馏损失"""
    ce = F.cross_entropy(z_student, y_true)
    t = float(temperature)
    log_p_s = F.log_softmax(z_student / t, dim=1)
    p_t = F.softmax(z_teacher / t, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return float(alpha) * ce + (1.0 - float(alpha)) * kl

def distill_teacher_to_student(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1):
    """服务器端正向蒸馏 (Ground -> Satellite)"""
    teacher.to(device).eval()
    student.to(device).train()
    opt = torch.optim.SGD(student.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x, y = batch["img"].to(device), batch["label"].to(device)
            with torch.no_grad(): z_t = teacher(x)
            loss = kd_loss_lkd(student(x), z_t, y, alpha=alpha, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return total / steps if steps > 0 else 0.0

def reverse_distill_student_to_teacher(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1, freeze_bn: bool = False):
    """服务器端反向蒸馏 (Satellite -> Ground)

    freeze_bn:
        True  -> 反向蒸馏时冻结 teacher 的 BatchNorm running stats（更稳定，避免后期漂移）
        False -> 默认行为（teacher.train()，BN 统计会更新）
    """
    student.to(device).eval()
    teacher.to(device).train()

    if freeze_bn:
        # Freeze BN running stats while still allowing weight updates
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    opt = torch.optim.SGD(teacher.parameters(), lr=float(lr))
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
    """纯 APSKD 本地自蒸馏"""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    eps, t = 1e-12, float(temperature)
    prev_epoch_task_loss = None
    total_loss_all, steps_all = 0.0, 0
    for ep in range(int(epochs)):
        prev_model = copy.deepcopy(model).to(device).eval()
        lt_prev_scalar = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0
        epoch_task_sum, epoch_task_steps = 0.0, 0
        for batch in trainloader:
            x, y = batch["img"].to(device), batch["label"].to(device)
            z_cur = model(x)
            lt = F.cross_entropy(z_cur, y)
            with torch.no_grad(): z_prev = prev_model(x)
            kl = F.kl_div(F.log_softmax(z_cur / t, dim=1), F.softmax(z_prev / t, dim=1), reduction="batchmean")
            w = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            loss = lt + (w * kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss_all += float(loss.item())
            steps_all += 1
            epoch_task_sum += float(lt.item())
            epoch_task_steps += 1
        prev_epoch_task_loss = (epoch_task_sum / epoch_task_steps) if epoch_task_steps > 0 else prev_epoch_task_loss
    return total_loss_all / steps_all if steps_all > 0 else 0.0

def train_fomaml_apskd(net, trainloader, device, alpha, beta, num_inner_steps, temperature, epochs=1, current_round=1, warmup_rounds=50):
    """混合模式: FO-MAML + APSKD (带动态预热)"""
    net.to(device).train()
    eps, t = 1e-12, float(temperature)
    lambda_kd = min(0.2, float(current_round) / float(max(1, warmup_rounds)))
    prev_epoch_task_loss = None
    total_outer_loss, steps = 0.0, 0
    for ep in range(epochs):
        prev_model = copy.deepcopy(net).to(device).eval()
        lt_prev_scalar = float(prev_epoch_task_loss) if prev_epoch_task_loss is not None else 1.0
        criterion = nn.CrossEntropyLoss().to(device)
        iterator = iter(trainloader)
        epoch_task_sum, epoch_task_steps = 0.0, 0
        while True:
            try:
                batch_sup = next(iterator)
                batch_qry = next(iterator)
            except StopIteration:
                break
            temp_model = copy.deepcopy(net).train()
            inner_optimizer = torch.optim.SGD(temp_model.parameters(), lr=alpha)
            imgs_sup, lbls_sup = batch_sup["img"].to(device), batch_sup["label"].to(device)
            
            # Inner Loop (FO-MAML 快速适应)
            for _ in range(num_inner_steps):
                inner_optimizer.zero_grad()
                loss_sup = criterion(temp_model(imgs_sup), lbls_sup)
                loss_sup.backward()
                inner_optimizer.step()
                
            # Outer Loop (任务损失 + APSKD 蒸馏正则化)
            imgs_qry, lbls_qry = batch_qry["img"].to(device), batch_qry["label"].to(device)
            outputs_qry = temp_model(imgs_qry)
            lt = criterion(outputs_qry, lbls_qry)
            
            with torch.no_grad(): z_prev = prev_model(imgs_qry)
            kl = F.kl_div(F.log_softmax(outputs_qry / t, dim=1), F.softmax(z_prev / t, dim=1), reduction="batchmean")
            w = float(lt.detach().item()) / (float(lt.detach().item()) + lt_prev_scalar + eps)
            
            # 动态权重 L_total = L_task + lambda * L_KD
            loss_total = lt + lambda_kd * (w * kl)
            
            grads_qry = torch.autograd.grad(loss_total, temp_model.parameters())
            with torch.no_grad():
                for param, grad in zip(net.parameters(), grads_qry):
                    param.data.sub_(beta * grad)
                    
            total_outer_loss += float(loss_total.item())
            steps += 1
            epoch_task_sum += float(lt.item())
            epoch_task_steps += 1
            
        prev_epoch_task_loss = (epoch_task_sum / epoch_task_steps) if epoch_task_steps > 0 else prev_epoch_task_loss
    return total_outer_loss / steps if steps > 0 else 0.0