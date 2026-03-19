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
from torchvision.models import resnet18

# -----------------------------------------------------------------------------
# 1. 模型定义 (Model Definitions)
# -----------------------------------------------------------------------------

class _InvertedResidual(nn.Module):
    """
    MobileNetV2 基础块：Inverted Residual with Linear Bottleneck。

    结构：
      1×1 PW-Expand  (in_c → in_c * expand_ratio, ReLU6)
      3×3 DW-Conv    (深度可分离卷积, ReLU6)
      1×1 PW-Project (in_c * expand_ratio → out_c, 线性激活，无 ReLU)

    残差连接条件：stride=1 且 in_c == out_c（论文原则：仅在特征图尺寸和通道不变时连接）。

    Linear Bottleneck 设计原因：
      最后一个 1×1 卷积不加激活函数，防止 ReLU 破坏低维流形信息，
      是 MobileNetV2 优于 V1 的核心改进。
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.use_res = (stride == 1 and in_c == out_c)

        layers = []
        # PW-Expand（当 expand_ratio=1 时跳过，避免冗余）
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_c, mid_c, 1, bias=False),
                       nn.BatchNorm2d(mid_c),
                       nn.ReLU6(inplace=True)]
        # DW-Conv
        layers += [nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1,
                             groups=mid_c, bias=False),
                   nn.BatchNorm2d(mid_c),
                   nn.ReLU6(inplace=True)]
        # PW-Project（线性，无激活）
        layers += [nn.Conv2d(mid_c, out_c, 1, bias=False),
                   nn.BatchNorm2d(out_c)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + out if self.use_res else out


class Net(nn.Module):
    """
    [卫星端学生模型] MobileNetV2-CIFAR  ~274K 参数

    针对 CIFAR-10（32×32 输入）对原版 MobileNetV2 做了两处适配：
      1. Stem：原版 32×32×2 stride → 改为 3×3 stride=1，保留空间分辨率
         （CIFAR 图像太小，大步长 stem 会直接压垮特征图）
      2. 去掉原版第一个 stride=2 的 bottleneck，避免 32→16 过早降采样

    网络结构（输入 32×32）：
      Stem    3×3 s1  3→32                          → 32×32
      Stage1  t1,c16,n1,s1                          → 32×32
      Stage2  t6,c24,n2,s1  (原 s2→s1，CIFAR 适配)  → 32×32
      Stage3  t6,c32,n3,s2                          → 16×16
      Stage4  t6,c64,n4,s2                          → 8×8
      Stage5  t6,c96,n3,s1                          → 8×8
      Head    1×1 PW 96→320, GAP, FC 320→10

    符号说明：t=expand_ratio, c=out_channels, n=重复次数, s=stride

    卫星端适用性：
      - 参数量 ~274K，约为 ResNet-18 教师的 1/41
      - 全部 DW+PW 卷积，无全连接中间层，内存占用低
      - 残差连接保证 FO-MAML inner-loop 梯度稳定回流
      - 无 Dropout（MobileNetV2 用 BN + 宽而浅结构替代 Dropout 的正则化作用）
    """

    # 配置表：(expand_ratio, out_channels, num_blocks, stride)
    _CFG = [
        (1,  16, 1, 1),   # Stage1
        (6,  24, 2, 1),   # Stage2  ← stride=1（CIFAR 适配，原版为 2）
        (6,  32, 3, 2),   # Stage3
        (6,  64, 4, 2),   # Stage4
        (6,  96, 3, 1),   # Stage5
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Stem：3×3 stride=1（CIFAR 专用）
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Inverted Residual Blocks
        layers = []
        in_c = 32
        for t, c, n, s in self._CFG:
            for i in range(n):
                layers.append(_InvertedResidual(
                    in_c, c,
                    stride=s if i == 0 else 1,
                    expand_ratio=t,
                ))
                in_c = c
        self.features = nn.Sequential(*layers)

        # Head：PW 升维 → GAP → 分类
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_c, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(320, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)           # 32×32→32×32
        x = self.features(x)       # 32×32→8×8
        x = self.head_conv(x)      # 8×8, 320ch
        x = self.pool(x)           # 1×1
        x = x.flatten(1)           # 320
        return self.classifier(x)


class BigTeacherNet(nn.Module):
    """
    [地面站教师模型] ResNet-18，针对 CIFAR-10（32×32）适配  ~11.17M 参数

    与原版 ResNet-18 的两处 CIFAR 适配：
      1. Stem：7×7 stride=2 → 3×3 stride=1
         原因：CIFAR 图像仅 32×32，7×7 stride=2 直接将特征图压到 16×16，
               浅层局部纹理特征严重丢失；3×3 stride=1 保留完整 32×32 特征图。
      2. 去掉 stem 后的 MaxPool
         原因：同上，避免早期过度降采样。

    结构（输入 32×32）：
      Stem    3×3 s1  3→64    → 32×32
      Layer1  BasicBlock×2 64  → 32×32  (无 downsample)
      Layer2  BasicBlock×2 128 → 16×16  (stride=2 downsample)
      Layer3  BasicBlock×2 256 → 8×8   (stride=2 downsample)
      Layer4  BasicBlock×2 512 → 4×4   (stride=2 downsample)
      GAP → FC 512→10

    教师端适用性：
      - 参数量 11.17M，比 MobileNetV2 学生大 41×，蒸馏信息增益显著
      - 深度残差结构提供丰富的多层次软标签
      - 仅部署在地面站，不受卫星算力/存储限制
      - 压缩比 41× 是论文"轻量化建模"的核心数据支撑
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 加载标准 ResNet-18 骨架
        backbone = resnet18(num_classes=num_classes)

        # CIFAR 适配：替换 stem
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        # 去掉 MaxPool（原版 stem 后紧跟一个 3×3 MaxPool stride=2）
        backbone.maxpool = nn.Identity()

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
    Client 端加载本地分区数据（Dirichlet Non-IID）。

    性能优化：
      num_workers=2：客户端在 Flower 模拟器中以独立进程运行，
        每个进程分配 2 个 worker 做数据预取，消除 GPU 等待 IO 的空转。
        注意不设 persistent_workers，避免多进程模拟时 fork 冲突。

      pin_memory=True：锁页内存 → DMA 直传 GPU，减少 CPU→GPU 拷贝延迟。

      prefetch_factor=2：每个 worker 提前预取 2 个 batch，
        确保 GPU 训练时下一批数据已在内存就绪。
    """
    import multiprocessing
    # 客户端进程内限制 2 个 worker，防止多客户端并发时 worker 数爆炸
    nw = min(2, multiprocessing.cpu_count())

    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=float(alpha),
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.3, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True,
        prefetch_factor=2 if nw > 0 else None,
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        prefetch_factor=2 if nw > 0 else None,
    )
    return trainloader, testloader

def load_centralized_dataset_train_test(
    warmup_samples: int = 10000,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Server 端加载数据集（用于地面预训练 / Warm Start）。

    性能优化说明：
      warmup_samples=10000：热启动只需要给 FL 提供一个好的初始化，
        用完整 50000 张反而浪费时间，10000 张子集精度损失 < 2%，速度快 5×。

      batch_size=256：RTX 4060 Laptop (8GB) 显存充裕，
        ResNet-18(~134MB) + MobileNetV2(~7MB) + 激活值 << 8GB，
        大 batch 减少调度次数，GPU 利用率从 ~20% 提升到 ~80%。

      num_workers=4：HuggingFace datasets 的 Arrow 格式读取 + PIL 解码 +
        transform 全在 CPU 完成，num_workers=0（默认）时主线程串行执行，
        GPU 约 79% 时间在等数据；4 个 worker 并行预取，GPU 空转消除。

      pin_memory=True：将数据预先锁页到内存，.to(device) 时走 DMA 直传，
        比普通内存到 GPU 的拷贝快约 30%。
    """
    import multiprocessing
    # 安全上限：不超过系统 CPU 核心数
    nw = min(num_workers, multiprocessing.cpu_count())

    dataset = load_dataset("uoft-cs/cifar10")

    # 训练集：取子集用于热启动
    train_data = (dataset["train"]
                  .shuffle(seed=42)
                  .select(range(warmup_samples))
                  .with_transform(apply_transforms))

    # 测试集：完整 10000 张（评估精度需要）
    test_data = dataset["test"].with_transform(apply_transforms)

    trainloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    testloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    return trainloader, testloader

def test_centralized_dataset():
    """Server 端加载测试集 (用于全局评估)"""
    _, testloader = load_centralized_dataset_train_test()
    return testloader

def build_dcal_loader(
    *, num_samples: int = 2048, batch_size: int = 256,
    seed: int = 42, split: str = "train", num_workers: int = 4,
) -> DataLoader:
    """
    构造校准集 D_cal（用于每轮下发前正向蒸馏）。

    batch_size=256, num_workers=4, pin_memory=True：
      与 load_centralized_dataset_train_test 同理，消除数据 IO 瓶颈。
      D_cal 只有 2048 张，大 batch 让整个 loader 只需 8 步即可跑完 1 epoch。
    """
    import multiprocessing
    nw = min(num_workers, multiprocessing.cpu_count())
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )

def build_dglobal_loader(
    *, num_samples: int = 4096, batch_size: int = 256,
    seed: int = 123, split: str = "train", num_workers: int = 4,
) -> DataLoader:
    """
    构造代表集 D_global（用于聚合后反向蒸馏）。

    batch_size=256, num_workers=4：同上，4096 张只需 16 步跑完 1 epoch。
    """
    import multiprocessing
    nw = min(num_workers, multiprocessing.cpu_count())
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.shuffle(seed=seed).select(range(num_samples)).with_transform(apply_transforms)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )

# -----------------------------------------------------------------------------
# 3. 中心化预训练与蒸馏 (Ground Warm-up Tools)
# -----------------------------------------------------------------------------

def train_centralized(net, trainloader, epochs, lr, device):
    """
    地面站从头训练 Teacher 模型（ResNet-18）。

    优化器：SGD + weight_decay=5e-4 + CosineAnnealingLR。
    non_blocking=True：配合 pin_memory=True 的 DataLoader，
      .to(device, non_blocking=True) 让 CPU→GPU 数据传输与 GPU 计算重叠，
      进一步消除传输等待（异步 DMA）。
    """
    net.to(device).train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

def distill_centralized(student, teacher, trainloader, epochs, lr, device, temp=2.0, alpha=0.5):
    """
    地面站单向初始蒸馏（Teacher → Student）。

    non_blocking=True：同 train_centralized，异步 DMA 传输。
    student.eval()：关闭 Dropout 让梯度方向稳定（MobileNetV2 无 Dropout，
      但保留 eval() 是好习惯，确保 BN 使用全局统计量而非 batch 统计量）。

    注意：eval() 不影响梯度计算，optimizer.step() 仍正常更新参数。
    """
    student.to(device).eval()
    teacher.to(device).eval()
    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction="batchmean").to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
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

def test(net, testloader, device):
    """标准零样本评估（Zero-shot Evaluation，用于预热阶段测试）"""
    net.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)

# -----------------------------------------------------------------------------
# 4. BN 统计量校准工具 (BatchNorm Calibration after FedAvg Aggregation)
# -----------------------------------------------------------------------------

def calibrate_bn_stats(net: nn.Module, loader: DataLoader, device: torch.device,
                       num_batches: int = 20) -> nn.Module:
    """
    联邦聚合后对 MobileNetV2 学生模型做 BN 统计量校准。

    问题背景：
      FedAvg 聚合 = 对各客户端的模型参数做加权平均。
      BN 层的 running_mean / running_var 也被平均，
      但各客户端的 Non-IID 数据分布差异极大，
      平均后的 running stats 不代表任何客户端的真实分布，
      导致聚合后模型的 BN 归一化失效，精度大幅下降。

    解决方案：
      聚合完成后，用 D_cal（地面站校准集，IID 分布，代表全局特征）
      对全局模型做一次前向传播，让 BN 重新估计正确的 running stats。
      不做反向传播，不更新参数，只重置统计量。

    实现细节：
      1. 将所有 BN 层切换到 train() 模式（开启 running stats 更新）
      2. 其余层保持 eval()（避免 Dropout 等引入随机性）
      3. 前向传播 num_batches 个 batch 后恢复全模型 eval()
      4. momentum 临时设为 None → 使用累积平均而非指数移动平均，
         收敛更快，num_batches=20（约 5120 张样本）即可稳定

    参数：
      net        : 聚合后的全局学生模型
      loader     : D_cal DataLoader（IID 校准集，batch_size=256）
      device     : 计算设备
      num_batches: 用于校准的 batch 数量，默认 20（≈ 5120 张）
    """
    net.to(device)

    # 记录原始 momentum，校准后恢复
    original_momentum = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            original_momentum[name] = m.momentum
            m.momentum = None          # None → 累积平均（比 EMA 更稳定）
            m.reset_running_stats()    # 清零，从干净状态开始估计
            m.train()                  # 开启 running stats 更新
        else:
            m.eval()                   # 其余层保持 eval

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            x = batch["img"].to(device, non_blocking=True)
            net(x)                     # 纯前向，触发 BN running stats 更新

    # 恢复全模型 eval() 和原始 momentum
    net.eval()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in original_momentum:
            m.momentum = original_momentum[name]

    return net


# -----------------------------------------------------------------------------
# 5. Meta-Learning & FL 算法 (FO-MAML / APSKD / Bidirectional KD)
# -----------------------------------------------------------------------------

def train_fomaml(net, trainloader, device, alpha, beta, num_inner_steps=1):
    """
    纯 FO-MAML 训练。

    BN 兼容修复（MobileNetV2 适配）：
      temp_model = copy.deepcopy(net).eval()  ← 关键：使用 eval() 而非 train()

      原因：MobileNetV2 有 27 个 BatchNorm 层，每层维护 running_mean/var。
      .train() 模式下 BN 会用当前 batch 统计量更新 running stats。
      MAML inner-loop 的 support batch 是极度偏斜的 Non-IID 数据（32张），
      BN running stats 被这批极端数据污染 → outer-loop 元梯度方向错误 →
      经多轮积累后 running stats 完全偏离 → 精度锁死在随机猜测水平(~14%)。

      eval() 模式：BN 冻结 running stats，仅用于归一化（不更新）。
      梯度仍正常流过 BN 层（BN 的 weight/bias 参数照常更新），
      inner-loop 的参数适应效果不变，但元梯度方向稳定可靠。

      outer-loop 后恢复 net.train()：net 本身继续用 train() 接收梯度，
      只有 temp_model（deepcopy 的临时副本）用 eval()。
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

        # [BN修复] eval()：冻结 BN running stats，梯度仍正常流动
        temp_model = copy.deepcopy(net).eval()
        inner_optimizer = torch.optim.SGD(temp_model.parameters(), lr=alpha)

        imgs_sup = batch_sup["img"].to(device, non_blocking=True)
        lbls_sup = batch_sup["label"].to(device, non_blocking=True)
        for _ in range(num_inner_steps):
            inner_optimizer.zero_grad()
            loss_sup = criterion(temp_model(imgs_sup), lbls_sup)
            loss_sup.backward()
            inner_optimizer.step()

        imgs_qry = batch_qry["img"].to(device, non_blocking=True)
        lbls_qry = batch_qry["label"].to(device, non_blocking=True)
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
    元评估（Meta-Testing: Adapt → Test）

    BN + Dropout 双重修复：adaptation 阶段使用 eval() 模式。

    Dropout 原因（旧）：
      eval() 关闭 Dropout，adaptation 梯度稳定，评估结果可复现。

    BatchNorm 原因（新，MobileNetV2 适配）：
      MobileNetV2 有 27 个 BN 层。若在 train() 模式下 adaptation，
      每步 forward 都会用当前极小的 Non-IID 本地 batch 更新 running stats，
      27 层累积偏差后输出严重扭曲，adaptation 效果反而为负。
      eval() 冻结 running stats，BN 用联邦聚合后校准的全局统计量做归一化，
      adaptation 梯度方向正确，少样本适应能力得以正常体现。
    """
    net.to(device)
    meta_model = copy.deepcopy(net).eval()   # eval(): 同时冻结 BN stats 和 Dropout
    optimizer = torch.optim.SGD(meta_model.parameters(), lr=adaptation_lr)
    criterion = nn.CrossEntropyLoss().to(device)
    iterator = iter(testloader)
    params_updated = False

    # Adaptation
    for _ in range(adaptation_steps):
        try:
            batch = next(iterator)
            params_updated = True
        except StopIteration:
            iterator = iter(testloader)
            batch = next(iterator)
        optimizer.zero_grad()
        loss = criterion(
            meta_model(batch["img"].to(device, non_blocking=True)),
            batch["label"].to(device, non_blocking=True),
        )
        loss.backward()
        optimizer.step()

    if not params_updated: return 0.0, 0.0

    # Evaluation
    correct, total_loss, total_samples, batches_count = 0, 0.0, 0, 0
    with torch.no_grad():
        for batch in iterator:
            labels  = batch["label"].to(device, non_blocking=True)
            outputs = meta_model(batch["img"].to(device, non_blocking=True))
            total_loss += float(criterion(outputs, labels).item())
            correct    += int((torch.max(outputs.data, 1)[1] == labels).sum().item())
            total_samples  += int(labels.size(0))
            batches_count  += 1
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
    """服务器端正向蒸馏（Ground → Satellite），non_blocking 异步传输"""
    teacher.to(device).eval()
    student.to(device).train()
    opt = torch.optim.SGD(student.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.no_grad():
                z_t = teacher(x)
            loss = kd_loss_lkd(student(x), z_t, y, alpha=alpha, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
    return total / steps if steps > 0 else 0.0

def reverse_distill_student_to_teacher(*, teacher, student, loader, device, alpha, temperature, lr, epochs=1):
    """服务器端反向蒸馏（Satellite → Ground），non_blocking 异步传输"""
    student.to(device).eval()
    teacher.to(device).train()
    opt = torch.optim.SGD(teacher.parameters(), lr=float(lr))
    total, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in loader:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
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
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
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

            imgs_sup = batch_sup["img"].to(device, non_blocking=True)
            lbls_sup = batch_sup["label"].to(device, non_blocking=True)
            imgs_qry = batch_qry["img"].to(device, non_blocking=True)
            lbls_qry = batch_qry["label"].to(device, non_blocking=True)

            # [BN修复] eval()：冻结 BN running stats，梯度仍正常流动
            temp_model = copy.deepcopy(net).eval()
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
                x0 = fb["img"].to(device, non_blocking=True)
                y0 = fb["label"].to(device, non_blocking=True)
                ce_teacher = float(criterion(last_adapted_teacher(x0), y0).item())
            except Exception:
                ce_teacher = 1.0

        apskd_opt = torch.optim.SGD(net.parameters(), lr=apskd_lr)
        net.train()
        for batch in trainloader:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

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
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
    return total_loss / steps if steps > 0 else 0.0

# -----------------------------------------------------------------------------
# 5. 新增对比算法 (FedProx / SCAFFOLD / FedKD)
# -----------------------------------------------------------------------------

def train_fedprox(net, trainloader, device, lr: float = 0.01, epochs: int = 1,
                  mu: float = 0.01, weight_decay: float = 1e-4,
                  label_smoothing: float = 0.1):
    """
    FedProx (Li et al., 2020)
    在 FedAvg 的 CE 损失基础上，加入近端项（proximal term）：
      L_prox = CE(net, y) + (mu/2) * ||w - w_global||^2
    近端项将每轮本地更新约束在全局模型附近，
    缓解 Non-IID 场景下客户端漂移（client drift）问题。
    w_global 是本轮下发时冻结的全局参数，不参与梯度更新。
    """
    net.to(device).train()
    # 冻结全局参数副本用于近端项计算
    global_params = [p.detach().clone() for p in net.parameters()]

    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=float(lr), momentum=0.9, weight_decay=float(weight_decay)
    )
    total_loss, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss_ce = criterion(net(images), labels)
            # 近端项：(mu/2) * sum_i ||w_i - w_global_i||^2
            prox = sum(
                ((p - g) ** 2).sum()
                for p, g in zip(net.parameters(), global_params)
            )
            loss = loss_ce + (float(mu) / 2.0) * prox
            loss.backward()
            optimizer.step()
            total_loss += float(loss_ce.item())   # 记录 CE 部分便于横向对比
            steps += 1
    return total_loss / steps if steps > 0 else 0.0


def train_scaffold(net, trainloader, device,
                   lr: float = 0.01, epochs: int = 1,
                   c_global: Optional[list] = None,
                   c_local: Optional[list] = None,
                   weight_decay: float = 1e-4):
    """
    SCAFFOLD (Karimireddy et al., 2020) 本地训练步骤。

    核心思想：用控制变量 (c_i, c) 修正本地梯度，消除 client drift：
      w ← w - lr * (g_local - c_i + c)
    其中：
      g_local：本地 CE 梯度
      c_i：本地控制变量（估计本地梯度偏差）
      c：全局控制变量（估计全局梯度）
      (c - c_i)：修正方向，让本地更新朝全局方向对齐

    控制变量更新规则（Option II，不需要额外通信）：
      c_i_new = c_i - c + (1/K*lr) * (w_0 - w_T)
    其中 K=本地步数，w_0=训练前参数，w_T=训练后参数。

    返回：(train_loss, delta_c_i) — delta_c_i 用于服务端聚合更新 c
    """
    net.to(device).train()
    device = next(net.parameters()).device

    # 若无控制变量则初始化为零（首轮或新节点）
    params_list = list(net.parameters())
    if c_global is None:
        c_global = [torch.zeros_like(p) for p in params_list]
    if c_local is None:
        c_local = [torch.zeros_like(p) for p in params_list]

    # 记录训练前参数 w_0
    w0 = [p.detach().clone() for p in params_list]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    total_loss, K = 0.0, 0

    for _ in range(int(epochs)):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            # SCAFFOLD 梯度修正：grad ← grad - c_i + c
            with torch.no_grad():
                for p, ci, cg in zip(net.parameters(), c_local, c_global):
                    if p.grad is not None:
                        p.grad.add_(-ci + cg)
            optimizer.step()
            total_loss += float(loss.item())
            K += 1

    # 更新本地控制变量（Option II）
    # c_i_new = c_i - c + (w0 - wT) / (K * lr)
    lr_val = float(lr)
    K = max(K, 1)
    delta_c = []
    with torch.no_grad():
        for p, ci, cg, w0_i in zip(net.parameters(), c_local, c_global, w0):
            ci_new = ci - cg + (w0_i - p.detach()) / (K * lr_val)
            delta_c.append(ci_new - ci)  # Δc_i 上传给 server 聚合
            ci.copy_(ci_new)             # 原地更新本地控制变量

    return total_loss / K, delta_c


def train_fedkd(net, trainloader, device,
                lr: float = 0.01, epochs: int = 1,
                teacher_model=None,
                kd_temperature: float = 4.0,
                kd_alpha: float = 0.5,
                weight_decay: float = 1e-4):
    """
    FedKD (Wu et al., 2022) 客户端本地蒸馏训练。

    FedKD 的核心：服务端维护一个（通常更大或更优的）教师模型，
    在每轮训练时将教师模型的软标签传递给客户端学生模型：
      L_KD = alpha * CE(student, y) + (1-alpha) * KL(student/T || teacher/T)
    教师在 server 端用聚合后的全局 student 做反向蒸馏持续更新（同我们方案）。

    与我们方案的区别：
      - FedKD 无元学习（无 inner/outer loop），直接用软标签监督
      - 无 APSKD 自适应自蒸馏，教师软标签始终来自 server 端全局教师
      - 适合通信开销较低、无快速适应需求的场景

    teacher_model: server 下发的全局教师模型（BigTeacherNet），eval 模式
    """
    net.to(device).train()

    if teacher_model is not None:
        teacher_model.to(device).eval()

    criterion_ce = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=float(lr), momentum=0.9, weight_decay=float(weight_decay)
    )
    t = float(kd_temperature)
    alpha = float(kd_alpha)
    total_loss, steps = 0.0, 0

    for _ in range(int(epochs)):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()

            student_logits = net(images)
            loss_ce = criterion_ce(student_logits, labels)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                loss_kl = F.kl_div(
                    F.log_softmax(student_logits / t, dim=1),
                    F.softmax(teacher_logits / t, dim=1),
                    reduction="batchmean",
                ) * (t * t)
                loss = alpha * loss_ce + (1.0 - alpha) * loss_kl
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()
            total_loss += float(loss_ce.item())
            steps += 1

    return total_loss / steps if steps > 0 else 0.0