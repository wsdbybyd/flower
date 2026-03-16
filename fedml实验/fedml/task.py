"""
task.py — 模型定义、数据处理、所有 FL 本地训练函数

训练模式：
  train_fedavg        FedAvg (McMahan et al., 2017)
  train_fedprox       FedProx (Li et al., 2020)
  train_scaffold      SCAFFOLD (Karimireddy et al., 2020)
  train_fedkd         FedKD (Wu et al., 2022)
  train_fomaml        FedMeta / FO-MAML
  train_fomaml_apskd  Ours (Full): FO-MAML + APSKD + 双向 KD (v7)
"""
import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# =============================================================================
# 1. 模型定义
# =============================================================================

class Net(nn.Module):
    """轻量级学生模型（卫星端部署）~299K 参数"""
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 4 * 4, 256)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.fc2   = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(p=dropout_p)
        self.fc3   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.fc3(x)


class BigTeacherNet(nn.Module):
    """教师模型（地面站部署）~4.5M 参数"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   64,  3, padding=1); self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  64,  3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1); self.bn4 = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# 2. 数据处理
# =============================================================================

fds = None
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.1):
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                           partition_by="label", alpha=float(alpha), seed=42)
        fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.3, seed=42).with_transform(apply_transforms)
    return (DataLoader(split["train"], batch_size=batch_size, shuffle=True),
            DataLoader(split["test"],  batch_size=batch_size))

def load_centralized_dataset_train_test():
    ds = load_dataset("uoft-cs/cifar10")
    train_data = ds["train"].with_format("torch").with_transform(apply_transforms)
    test_data  = ds["test"].with_format("torch").with_transform(apply_transforms)
    return (DataLoader(train_data, batch_size=64, shuffle=True),
            DataLoader(test_data,  batch_size=128, shuffle=False))

def test_centralized_dataset():
    _, tl = load_centralized_dataset_train_test(); return tl

def build_dcal_loader(*, num_samples=2048, batch_size=64, seed=42, split="train"):
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def build_dglobal_loader(*, num_samples=4096, batch_size=64, seed=123, split="train"):
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# =============================================================================
# 3. 地面预训练工具
# =============================================================================

def train_centralized(net, trainloader, epochs, lr, device):
    net.to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    ce  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for b in trainloader:
            opt.zero_grad()
            ce(net(b["img"].to(device)), b["label"].to(device)).backward()
            opt.step()

def distill_centralized(student, teacher, trainloader, epochs, lr, device, temp=2.0, alpha=0.5):
    student.to(device).eval(); teacher.to(device).eval()
    ce = nn.CrossEntropyLoss(); kl = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for b in trainloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            opt.zero_grad()
            z_s = student(imgs)
            with torch.no_grad(): z_t = teacher(imgs)
            loss = ((1-alpha)*ce(z_s, lbls)
                    + alpha*kl(F.log_softmax(z_s/temp,dim=1),
                               F.softmax(z_t/temp,dim=1)) * temp**2)
            loss.backward(); opt.step()

def test(net, testloader, device):
    net.to(device).eval(); ce = nn.CrossEntropyLoss()
    correct = loss = 0.0
    with torch.no_grad():
        for b in testloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            out = net(imgs); loss += ce(out, lbls).item()
            correct += (out.argmax(1) == lbls).sum().item()
    return loss/len(testloader), correct/len(testloader.dataset)

def kd_loss_lkd(z_s, z_t, y, alpha, temperature):
    t = float(temperature)
    ce = F.cross_entropy(z_s, y)
    kl = F.kl_div(F.log_softmax(z_s/t,dim=1), F.softmax(z_t/t,dim=1), reduction="batchmean")
    return float(alpha)*ce + (1.0-float(alpha))*kl

def distill_teacher_to_student(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1):
    teacher.to(device).eval(); student.to(device).train()
    opt = torch.optim.SGD(student.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for b in loader:
            x, y = b["img"].to(device), b["label"].to(device)
            with torch.no_grad(): z_t = teacher(x)
            loss = kd_loss_lkd(student(x), z_t, y, alpha=alpha, temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
    return total/steps if steps > 0 else 0.0

def reverse_distill_student_to_teacher(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1):
    student.to(device).eval(); teacher.to(device).train()
    opt = torch.optim.SGD(teacher.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for b in loader:
            x, y = b["img"].to(device), b["label"].to(device)
            with torch.no_grad(): z_s = student(x)
            loss = kd_loss_lkd(teacher(x), z_s, y, alpha=alpha, temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
    return total/steps if steps > 0 else 0.0


# =============================================================================
# 4. 统一评估函数（所有方法公用）
# =============================================================================

def test_meta(net, testloader, device, adaptation_steps=5, adaptation_lr=0.01):
    """
    Meta-Testing: Adapt → Test
    eval() 关闭 Dropout 确保 adaptation 梯度方向稳定、结果可复现。
    所有 6 种方法公用此函数，对比公平。
    """
    net.to(device)
    meta = copy.deepcopy(net).eval()
    opt  = torch.optim.SGD(meta.parameters(), lr=adaptation_lr)
    ce   = nn.CrossEntropyLoss().to(device)
    it   = iter(testloader)
    updated = False
    for _ in range(adaptation_steps):
        try:    b = next(it); updated = True
        except StopIteration: it = iter(testloader); b = next(it)
        opt.zero_grad()
        ce(meta(b["img"].to(device)), b["label"].to(device)).backward()
        opt.step()
    if not updated: return 0.0, 0.0
    correct = total_loss = total = n = 0
    with torch.no_grad():
        for b in it:
            lbls = b["label"].to(device); out = meta(b["img"].to(device))
            total_loss += float(ce(out, lbls).item())
            correct    += int((out.argmax(1) == lbls).sum().item())
            total      += int(lbls.size(0)); n += 1
    return (total_loss/n if n > 0 else 0.0,
            correct/total if total > 0 else 0.0)


# =============================================================================
# 5. 本地训练函数 — 6 种 FL 方法
# =============================================================================

# ---------------------------------------------------------------------------
# 5-1  FedAvg
# ---------------------------------------------------------------------------
def train_fedavg(net, trainloader, device,
                 lr=0.01, epochs=1, weight_decay=1e-4, label_smoothing=0.1):
    """FedAvg (McMahan et al., 2017) + Weight Decay + Label Smoothing"""
    net.to(device).train()
    ce  = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr),
                          momentum=0.9, weight_decay=float(weight_decay))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for b in trainloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            opt.zero_grad(); loss = ce(net(imgs), lbls); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
    return total/steps if steps > 0 else 0.0


# ---------------------------------------------------------------------------
# 5-2  FedProx
# ---------------------------------------------------------------------------
def train_fedprox(net, trainloader, device,
                  lr=0.01, epochs=1, mu=0.01, weight_decay=1e-4, label_smoothing=0.1):
    """
    FedProx (Li et al., 2020)
    L = CE(w) + (mu/2)*||w - w_global||^2
    近端项约束本地更新幅度，缓解 Non-IID 下的 client drift。
    """
    net.to(device).train()
    w_global = [p.detach().clone() for p in net.parameters()]
    ce  = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr),
                          momentum=0.9, weight_decay=float(weight_decay))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for b in trainloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            opt.zero_grad()
            loss_ce = ce(net(imgs), lbls)
            prox = sum(((p - g)**2).sum() for p, g in zip(net.parameters(), w_global))
            (loss_ce + (float(mu)/2.0)*prox).backward(); opt.step()
            total += float(loss_ce.item()); steps += 1
    return total/steps if steps > 0 else 0.0


# ---------------------------------------------------------------------------
# 5-3  SCAFFOLD
# ---------------------------------------------------------------------------
def train_scaffold(net, trainloader, device,
                   lr=0.01, epochs=1,
                   c_global: Optional[List[torch.Tensor]] = None,
                   c_local:  Optional[List[torch.Tensor]] = None,
                   weight_decay=1e-4):
    """
    SCAFFOLD (Karimireddy et al., 2020)
    梯度修正：grad ← grad - c_i + c  （消除 client drift）
    控制变量更新（Option II）：
      c_i_new = c_i - c + (w_0 - w_T)/(K*lr)
    返回 (train_loss, delta_c)；delta_c 由 server 汇总更新全局 c。
    """
    net.to(device).train()
    params = list(net.parameters())
    if c_global is None: c_global = [torch.zeros_like(p) for p in params]
    if c_local  is None: c_local  = [torch.zeros_like(p) for p in params]
    w0  = [p.detach().clone() for p in params]
    ce  = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    total, K = 0.0, 0
    for _ in range(int(epochs)):
        for b in trainloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            opt.zero_grad(); loss = ce(net(imgs), lbls); loss.backward()
            with torch.no_grad():
                for p, ci, cg in zip(net.parameters(), c_local, c_global):
                    if p.grad is not None:
                        p.grad.add_(-ci.to(device) + cg.to(device))
            opt.step(); total += float(loss.item()); K += 1
    K = max(K, 1)
    delta_c = []
    with torch.no_grad():
        for p, ci, cg, w0i in zip(net.parameters(), c_local, c_global, w0):
            ci_new = ci.to(device) - cg.to(device) + (w0i.to(device) - p.detach())/(K*float(lr))
            delta_c.append((ci_new - ci.to(device)).cpu())
            ci.copy_(ci_new.cpu())
    return total/K, delta_c


# ---------------------------------------------------------------------------
# 5-4  FedKD
# ---------------------------------------------------------------------------
def train_fedkd(net, trainloader, device,
                lr=0.01, epochs=1, teacher_model=None,
                kd_temperature=4.0, kd_alpha=0.5, weight_decay=1e-4):
    """
    FedKD (Wu et al., 2022)
    L = alpha*CE + (1-alpha)*KL(student/T || teacher/T)
    teacher_model = server 下发的 BigTeacherNet（eval 模式）。
    区别于 Ours：无元学习、无 APSKD 自蒸馏。
    """
    net.to(device).train()
    if teacher_model is not None: teacher_model.to(device).eval()
    ce  = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=float(lr),
                          momentum=0.9, weight_decay=float(weight_decay))
    t = float(kd_temperature); alpha = float(kd_alpha)
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for b in trainloader:
            imgs, lbls = b["img"].to(device), b["label"].to(device)
            opt.zero_grad(); z_s = net(imgs); loss_ce = ce(z_s, lbls)
            if teacher_model is not None:
                with torch.no_grad(): z_t = teacher_model(imgs)
                loss_kl = F.kl_div(F.log_softmax(z_s/t,dim=1),
                                   F.softmax(z_t/t,dim=1),
                                   reduction="batchmean") * (t*t)
                loss = alpha*loss_ce + (1.0-alpha)*loss_kl
            else:
                loss = loss_ce
            loss.backward(); opt.step()
            total += float(loss_ce.item()); steps += 1
    return total/steps if steps > 0 else 0.0


# ---------------------------------------------------------------------------
# 5-5  FedMeta / FO-MAML
# ---------------------------------------------------------------------------
def train_fomaml(net, trainloader, device,
                 alpha=0.01, beta=0.001, num_inner_steps=5):
    """
    FedMeta / FO-MAML (Finn et al., 2017; Chen et al., 2018 FO 近似)
    inner: support set CE → 更新 temp_model（步长 alpha）
    outer: query  set CE → 对 temp 求梯度，更新 net（步长 beta）
    """
    net.to(device).train()
    ce = nn.CrossEntropyLoss().to(device)
    it = iter(trainloader)
    total, steps = 0.0, 0
    while True:
        try:    sup = next(it); qry = next(it)
        except StopIteration: break
        temp = copy.deepcopy(net).train()
        iopt = torch.optim.SGD(temp.parameters(), lr=alpha)
        imgs_s, lbls_s = sup["img"].to(device), sup["label"].to(device)
        for _ in range(num_inner_steps):
            iopt.zero_grad(); ce(temp(imgs_s), lbls_s).backward(); iopt.step()
        imgs_q, lbls_q = qry["img"].to(device), qry["label"].to(device)
        l_q = ce(temp(imgs_q), lbls_q)
        grads = torch.autograd.grad(l_q, temp.parameters())
        with torch.no_grad():
            for p, g in zip(net.parameters(), grads): p.data.sub_(beta*g)
        total += float(l_q.item()); steps += 1
    return total/steps if steps > 0 else 0.0


# ---------------------------------------------------------------------------
# 5-6  Ours (Full): FO-MAML + APSKD + 双向 KD  (v7)
# ---------------------------------------------------------------------------
def train_apskd(model, trainloader, device, *, lr, epochs, temperature):
    """纯 APSKD 自蒸馏（供独立使用）"""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    eps, t = 1e-12, float(temperature)
    prev_loss = None; total, steps = 0.0, 0
    for _ in range(int(epochs)):
        prev = copy.deepcopy(model).to(device).eval()
        lp   = float(prev_loss) if prev_loss is not None else 1.0
        ep_s, ep_n = 0.0, 0
        for b in trainloader:
            x, y = b["img"].to(device), b["label"].to(device)
            z = model(x); lt = F.cross_entropy(z, y)
            with torch.no_grad(): zp = prev(x)
            kl = F.kl_div(F.log_softmax(z/t,dim=1), F.softmax(zp/t,dim=1), reduction="batchmean")
            w  = float(lt.detach().item())/(float(lt.detach().item())+lp+eps)
            loss = lt + w*kl; opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
            ep_s += float(lt.item()); ep_n += 1
        prev_loss = ep_s/ep_n if ep_n > 0 else prev_loss
    return total/steps if steps > 0 else 0.0


def train_fomaml_apskd(net, trainloader, device,
                       alpha=0.01, beta=0.01, num_inner_steps=7,
                       temperature=3.0, epochs=1,
                       teacher_model=None,
                       current_round=1, warmup_rounds=20):
    """
    Ours (Full) v7: MAML-KD 联合 inner-loop + Adapted-Teacher APSKD 精调

    Stage 1 — MAML-KD inner-loop:
      inner: CE + lambda_kd * KL(temp || snapshot)
      outer: 纯 query CE（净元梯度）

    Stage 2 — APSKD 精调（教师 = adapted temp_model）:
      L_refine = CE(net,y) + w_apskd * KL(net || adapted_teacher)

    lambda_kd 线性 warmup 0→0.4 （前 warmup_rounds 轮后固定）
    """
    net.to(device).train()
    t = float(temperature); eps = 1e-12
    ce = nn.CrossEntropyLoss().to(device)
    total, steps = 0.0, 0
    snapshot = teacher_model if teacher_model is not None else copy.deepcopy(net).to(device).eval()
    lambda_kd = min(float(current_round)/max(int(warmup_rounds),1), 1.0) * 0.4
    apskd_lr  = float(beta)
    last_adapted: Optional[torch.nn.Module] = None

    for _ in range(epochs):
        it = iter(trainloader)
        # ── Stage 1: MAML-KD ─────────────────────────────────────────────
        while True:
            try:    sup = next(it); qry = next(it)
            except StopIteration: break
            imgs_s = sup["img"].to(device); lbls_s = sup["label"].to(device)
            imgs_q = qry["img"].to(device); lbls_q = qry["label"].to(device)
            temp = copy.deepcopy(net).train()
            iopt = torch.optim.SGD(temp.parameters(), lr=alpha)
            for _ in range(num_inner_steps):
                iopt.zero_grad()
                z_tmp = temp(imgs_s); l_ce = ce(z_tmp, lbls_s)
                if lambda_kd > 0.0:
                    with torch.no_grad(): z_sn = snapshot(imgs_s)
                    l_kd = F.kl_div(F.log_softmax(z_tmp/t,dim=1),
                                    F.softmax(z_sn/t,dim=1), reduction="batchmean")
                    (l_ce + lambda_kd*l_kd).backward()
                else:
                    l_ce.backward()
                iopt.step()
            l_q   = ce(temp(imgs_q), lbls_q)
            grads = torch.autograd.grad(l_q, temp.parameters())
            with torch.no_grad():
                for p, g in zip(net.parameters(), grads): p.data.sub_(beta*g)
            total += float(l_q.item()); steps += 1
            last_adapted = temp.eval()
        # ── Stage 2: APSKD 精调 ──────────────────────────────────────────
        ts2 = last_adapted if last_adapted is not None else snapshot
        with torch.no_grad():
            try:
                fb = next(iter(trainloader))
                ce_t = float(ce(ts2(fb["img"].to(device)), fb["label"].to(device)).item())
            except Exception: ce_t = 1.0
        aopt = torch.optim.SGD(net.parameters(), lr=apskd_lr)
        net.train()
        for b in trainloader:
            x, y = b["img"].to(device), b["label"].to(device)
            z_n = net(x); l_ce = F.cross_entropy(z_n, y)
            with torch.no_grad(): z_t = ts2(x)
            ce_c = float(l_ce.detach().item())
            w    = ce_c/(ce_c+ce_t+eps)
            l_kd = F.kl_div(F.log_softmax(z_n/t,dim=1), F.softmax(z_t/t,dim=1), reduction="batchmean")
            aopt.zero_grad(); (l_ce+w*l_kd).backward(); aopt.step()
    return total/steps if steps > 0 else 0.0