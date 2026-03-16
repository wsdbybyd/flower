import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# -----------------------------------------------------------------------------
# 1. 模型定义 (Model Definitions)
# -----------------------------------------------------------------------------

class Net(nn.Module):
    """
    升级版学生模型 (Upgraded Student Model) ~300K 参数。
    在保持卫星端可部署的前提下，给 FO-MAML inner-loop 适应提供
    足够的参数冗余，同时缓解 APSKD 蒸馏正则化对小模型的过度压制。

    架构变化（原 → 新）：
      Conv 通道: 3→6→16       升级为  3→32→64  (含 BN，稳定训练)
      Pool:      2次            升级为  3次 (额外压缩一次特征图)
      FC 维度:   400→120→84   升级为  1024→256→64
      Dropout:   无             新增 fc1/fc2 后各加一层 (p=0.3)
      总参数量:  ~62K          升级为  ~299K

    Dropout 设计说明：
      - 只加在全连接层之间，不加在卷积层（卷积特征图较小，Dropout 效果差）
      - p=0.3：随机屏蔽 30% 神经元，强迫模型不依赖单一特征路径
      - FedAvg 场景下每个客户端用不同随机 mask 训练，
        聚合后的全局模型天然具有更强泛化性，有效抑制 client drift 过拟合
      - 推断时（model.eval()）Dropout 自动关闭，不影响评估结果

    输入尺寸推导 (CIFAR-10, 32×32):
      32 -[conv3x3 pad1]→ 32 -[pool2x2]→ 16
      16 -[conv3x3 pad1]→ 16 -[pool2x2]→ 8
                                8 -[pool2x2]→ 4
      fc1 输入: 64 * 4 * 4 = 1024
    """
    def __init__(self, dropout_p: float = 0.3):
        super(Net, self).__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 4 * 4, 256)
        self.drop1   = nn.Dropout(p=dropout_p)   # fc1 后
        self.fc2     = nn.Linear(256, 64)
        self.drop2   = nn.Dropout(p=dropout_p)   # fc2 后
        self.fc3     = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(x)                                 # 8→4
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
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

def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.1):
    """
    Client 端加载本地分区数据。

    分区方式：Dirichlet Non-IID
      alpha 由配置文件传入（默认 0.1，高度 Non-IID），
      不再硬编码，确保代码与 pyproject.toml 的 dirichlet-alpha 一致。

      test_size=0.3：给本地验证集留更多数据，
      在 Non-IID α=0.1 下每个客户端类别极少，
      0.2 时验证集可能只有几十张，test_meta 20步 adaptation
      几乎把验证集用完，剩余评估数据极少导致结果噪声大。
      0.3 能保证 adaptation 后还有足够数据评估。
    """
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=float(alpha),
            seed=42,
        )
        fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.3, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(partition_train_test["test"],  batch_size=batch_size)
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
    """
    地面站单向初始蒸馏 (Teacher -> Student)。

    Dropout 修复：student 使用 eval() 模式接收软标签。
    蒸馏阶段 student 的输出需要稳定（作为 KL 散度的 log-softmax 输入），
    train() 模式下 Dropout 随机屏蔽神经元会让 student 输出带有噪声，
    导致软标签匹配不稳定，蒸馏质量下降。
    注意：eval() 不影响梯度计算，optimizer.step() 仍正常更新参数。
    """
    student.to(device).eval()   # eval() 关闭 Dropout，但梯度仍正常流动
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

def test_meta(net, testloader, device, adaptation_steps=5, adaptation_lr=0.01):
    """
    元评估 (Meta-Testing: Adapt -> Test)

    Dropout 修复说明：
      adaptation 阶段使用 eval() 而非 train()。
      原因：adaptation 只是用少量样本微调参数方向，不需要 Dropout 的随机噪声；
      train() 模式下每次 forward 的随机 mask 不同，梯度方向带有噪声，
      导致 meta-test 结果在不同运行间不一致，评估曲线抖动增大。
      eval() 模式下 Dropout 关闭，adaptation 梯度干净，评估结果可复现。
    """
    net.to(device)
    # [修复] 用 eval() 而非 train()，关闭 Dropout 确保 adaptation 梯度稳定
    meta_model = copy.deepcopy(net).eval()
    optimizer = torch.optim.SGD(meta_model.parameters(), lr=adaptation_lr)
    criterion = nn.CrossEntropyLoss().to(device)
    iterator = iter(testloader)
    params_updated = False

    # Adaptation（eval 模式，Dropout 关闭，梯度方向稳定可复现）
    for _ in range(adaptation_steps):
        try:
            batch = next(iterator)
            params_updated = True
        except StopIteration:
            iterator = iter(testloader)
            batch = next(iterator)
        optimizer.zero_grad()
        loss = criterion(meta_model(batch["img"].to(device)), batch["label"].to(device))
        loss.backward()
        optimizer.step()

    if not params_updated: return 0.0, 0.0

    # Evaluation（同样 eval 模式）
    correct, total_loss, total_samples, batches_count = 0, 0.0, 0, 0
    with torch.no_grad():
        for batch in iterator:
            labels = batch["label"].to(device)
            outputs = meta_model(batch["img"].to(device))
            total_loss += float(criterion(outputs, labels).item())
            correct += int((torch.max(outputs.data, 1)[1] == labels).sum().item())
            total_samples += int(labels.size(0))
            batches_count += 1
    return (total_loss / batches_count if batches_count > 0 else 0.0,
            correct / total_samples if total_samples > 0 else 0.0)

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

def reverse_distill_student_to_teacher(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1):
    """服务器端反向蒸馏 (Satellite -> Ground)"""
    student.to(device).eval()
    teacher.to(device).train()
    opt = torch.optim.SGD(teacher.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x, y = batch["img"].to(device), batch["label"].to(device)
            with torch.no_grad(): z_s = student(x)
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

def train_fomaml_apskd(net, trainloader, device, alpha, beta, num_inner_steps,
                       temperature, epochs=1, teacher_model=None,
                       current_round=1, warmup_rounds=20):
    """
    混合模式 v7: MAML-KD Joint Inner-Loop + Adapted-Teacher APSKD Refinement

    ── 设计思路 ─────────────────────────────────────────────────────────────
    Stage 1（MAML-KD 联合 inner-loop）：
      inner-loop: CE + λ_kd * KL(temp || snapshot)
        → temp_model 在适应本地数据的同时保留全局知识（snapshot 作为锚点）
        → adapted temp_model 携带"本地个性化 + 全局知识"的双重信息
      outer-loop: 纯 query CE（元梯度质量优先）
        → 跨任务泛化信号，驱动 net 向快速可适应的初始化收敛

    Stage 2（APSKD 精调，教师 = adapted temp_model）：
      教师软标签来自 adapted temp_model（非原始 snapshot）
        → KL 约束来自"本地适应后的分布"，非零且有信息量
        → net 被引导向"高质量本地适应点"收敛
      步长 = beta（与 outer-loop 相同量级，适度精调）
      Loss: CE(net,y) + w_apskd * KL(net || adapted_teacher)

    λ_kd 线性 warmup（0 → 0.4，前 warmup_rounds 轮，之后固定 0.4）：
      早期全局模型质量低，KD 约束应弱；
      随全局模型收敛，KD 信号质量提高，约束增强至上限 0.4。

    协同效果：
      MAML outer-loop → 优化跨任务泛化初始化
      inner-loop KD   → 使 adapted 结果携带全局知识（Mixed 的核心增益）
      Stage2 APSKD    → 用 adapted teacher 轻量稳定输出分布
      → Mixed 同时具备：快速适应（MAML）+ 知识保留（inner KD）+ 分布稳定（APSKD）
    """
    net.to(device).train()
    t = float(temperature)
    eps = 1e-12
    criterion = nn.CrossEntropyLoss().to(device)
    total_outer_loss, steps = 0.0, 0

    # ── KD 教师 snapshot：跨轮次全局模型 ───────────────────────────────
    if teacher_model is not None:
        snapshot = teacher_model  # 已在 client.py 中设为 eval()
    else:
        snapshot = copy.deepcopy(net).to(device).eval()

    # ── λ_kd：线性 warmup 0→0.4，warmup_rounds 轮后固定 ───────────────
    warmup_rounds = max(int(warmup_rounds), 1)
    lambda_kd = min(float(current_round) / warmup_rounds, 1.0) * 0.4

    # ── Stage2 步长：与 outer-loop beta 相同量级 ───────────────────────
    apskd_lr = float(beta)

    # 保存最后一个 adapted temp_model 用于 Stage2（每 epoch 更新）
    last_adapted_teacher: Optional[torch.nn.Module] = None

    for ep in range(epochs):
        iterator = iter(trainloader)

        # ================================================================
        # Stage 1：MAML-KD 联合 inner-loop
        #   inner：CE + λ_kd * KL(temp || snapshot)  → 携带全局知识
        #   outer：纯 query CE → 净元梯度，跨任务泛化
        # ================================================================
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

            temp_model = copy.deepcopy(net).train()
            inner_opt = torch.optim.SGD(temp_model.parameters(), lr=alpha)

            # inner-loop：CE + KD（snapshot 作为知识锚点）
            for _ in range(num_inner_steps):
                inner_opt.zero_grad()
                z_temp = temp_model(imgs_sup)
                l_ce_inner = criterion(z_temp, lbls_sup)
                if lambda_kd > 0.0:
                    with torch.no_grad():
                        z_snap_inner = snapshot(imgs_sup)
                    l_kd_inner = F.kl_div(
                        F.log_softmax(z_temp / t, dim=1),
                        F.softmax(z_snap_inner / t, dim=1),
                        reduction="batchmean",
                    )
                    l_inner = l_ce_inner + lambda_kd * l_kd_inner
                else:
                    l_inner = l_ce_inner
                l_inner.backward()
                inner_opt.step()

            # outer-loop：纯 query CE
            outputs_qry = temp_model(imgs_qry)
            lt = criterion(outputs_qry, lbls_qry)
            grads_maml = torch.autograd.grad(lt, temp_model.parameters())
            with torch.no_grad():
                for param, grad in zip(net.parameters(), grads_maml):
                    param.data.sub_(beta * grad)

            total_outer_loss += float(lt.item())
            steps += 1

            # 保留最后一个 adapted temp_model 作为 Stage2 教师
            last_adapted_teacher = temp_model.eval()

        # ================================================================
        # Stage 2：APSKD 精调
        #   教师：last_adapted_teacher（本地适应 + 全局知识的混合分布）
        #   步长：apskd_lr = beta（适度精调，不大幅覆盖元梯度）
        #   Loss：CE(net,y) + w_apskd * KL(net || adapted_teacher)
        # ================================================================
        if last_adapted_teacher is None:
            last_adapted_teacher = snapshot  # fallback

        # 预计算 adapted_teacher 的任务损失（APSKD 自适应权重分母）
        with torch.no_grad():
            try:
                fb = next(iter(trainloader))
                x0, y0 = fb["img"].to(device), fb["label"].to(device)
                ce_teacher = float(criterion(last_adapted_teacher(x0), y0).item())
            except Exception:
                ce_teacher = 1.0

        apskd_opt = torch.optim.SGD(net.parameters(), lr=apskd_lr)
        net.train()
        for batch in trainloader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)

            z_net = net(x)
            l_ce = F.cross_entropy(z_net, y)

            with torch.no_grad():
                z_teacher = last_adapted_teacher(x)

            ce_cur = float(l_ce.detach().item())
            # 自适应权重：当前损失相对教师损失越高，KD 约束越弱（避免早期 KD 过强）
            w_apskd = ce_cur / (ce_cur + ce_teacher + eps)

            l_kd_stage2 = F.kl_div(
                F.log_softmax(z_net / t, dim=1),
                F.softmax(z_teacher / t, dim=1),
                reduction="batchmean",
            )
            l_refine = l_ce + w_apskd * l_kd_stage2
            apskd_opt.zero_grad()
            l_refine.backward()
            apskd_opt.step()

    return total_outer_loss / steps if steps > 0 else 0.0

def train_fedavg(net, trainloader, device, lr: float = 0.01, epochs: int = 1,
                 weight_decay: float = 1e-4, label_smoothing: float = 0.1):
    """
    标准 FedAvg 本地训练 (McMahan et al., 2017)，加入两处轻量正则化：

    1. Weight Decay (1e-4)：L2 惩罚项，直接限制参数幅度膨胀。
       多轮 FedAvg 聚合后模型容易对本地数据过拟合，
       参数绝对值持续增大，导致 logits 幅度失控、CrossEntropy Loss 爬升。

    2. Label Smoothing (0.1)：将硬标签软化为 [ε/K, ..., 1-ε+ε/K, ...]。
       防止模型将某个类的 logit 推得无限大（即使预测正确），
       从而在 Accuracy 不变的情况下稳定 Loss 数值。

    两者均不改变 FedAvg 的聚合语义，只作用于本地训练步骤。
    """
    net.to(device).train()
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=float(lr),
        momentum=0.9,
        weight_decay=float(weight_decay),
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
            steps += 1
    return total_loss / steps if steps > 0 else 0.0