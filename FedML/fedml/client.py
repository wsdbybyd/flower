import math
import random
from typing import Any, Dict
from logging import INFO

import torch

# Flower 框架相关导入
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

# 项目内部模块导入
from fedml.task import Net, load_data
# ---- Modified: Import Meta-Learning functions ----
from fedml.task import test_meta as test_fn      # 元学习评估 (Adapt -> Test)
from fedml.task import train_fomaml as train_fn  # FO-MAML 训练


app = ClientApp()

# -----------------------------------------------------------------------------
# 全局状态缓存
# -----------------------------------------------------------------------------
# 缓存节点的距离，确保同一个节点在不同轮次（Round）中保持距离不变
_DISTANCE_CACHE_KM: Dict[int, float] = {}


# -----------------------------------------------------------------------------
# 辅助工具函数
# -----------------------------------------------------------------------------
def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """安全读取配置字典中的键值。"""
    try:
        return cfg[key]
    except Exception:
        return default


def _state_dict_size_bits(state_dict: Dict[str, torch.Tensor]) -> int:
    """计算 PyTorch 模型参数字典的总大小（单位：比特 bit）。"""
    total_bytes = 0
    for _, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            total_bytes += t.numel() * t.element_size()
    return int(total_bytes * 8)


# -----------------------------------------------------------------------------
# 物理层通信模拟模块 (保持不变)
# -----------------------------------------------------------------------------
def compute_link_metrics(
    *,
    model_bits: int,        # 模型大小 (bits)
    d_km: float,            # 通信距离 (km)
    tau_d_s: float,         # 最大允许时延/时间窗口 (seconds)
    A_T: float,             # 发射天线增益 (dB)
    A_R: float,             # 接收天线增益 (dB)
    f_c_hz: float,          # 载波频率 (Hz)
    G_H: float,             # 硬件损耗因子 (0~1)
    delta_rician: float,    # 瑞利/莱斯衰落因子
    psi_db_per_km: float,   # 大气衰减系数 (dB/km)
    zeta_km: float,         # 衰减有效距离/标度长度 (km)
    w_u_hz: float,          # 上行链路带宽 (Hz)
    p_u_w: float,           # 上行发射功率 (Watts)
    w_d_hz: float,          # 下行链路带宽 (Hz)
    p_d_w: float,           # 下行发射功率 (Watts)
    sigma2: float,          # 噪声功率 (Watts)
) -> Dict[str, float]:
    """计算物理层链路预算 (Link Budget) 及通信性能指标。"""
    c = 299_792_458.0
    d_m = max(float(d_km), 1e-9) * 1000.0
    zeta_km = max(float(zeta_km), 1e-9)

    # 1. 大气衰减
    A_atm = 10.0 ** ((3.0 * float(psi_db_per_km) * float(d_km)) / (10.0 * zeta_km))

    # 2. 自由空间路径损耗 (FSPL) 与综合信道系数
    fspl = (c / (4.0 * math.pi * d_m * max(float(f_c_hz), 1e-9))) ** 2
    A_C = fspl * float(G_H) * A_atm * float(delta_rician)

    # 3. 天线增益线性化
    A_T_lin = (10.0 ** (float(A_T) / 10.0)) if float(A_T) > 10.0 else float(A_T)
    A_R_lin = (10.0 ** (float(A_R) / 10.0)) if float(A_R) > 10.0 else float(A_R)

    rho = A_T_lin * A_C * A_R_lin

    sigma2 = max(float(sigma2), 1e-18)
    w_u_hz = max(float(w_u_hz), 1e-9)
    w_d_hz = max(float(w_d_hz), 1e-9)

    # 4. 信噪比 (SNR)
    snr_u = (float(p_u_w) * (rho ** 2)) / sigma2
    snr_d = (float(p_d_w) * (rho ** 2)) / sigma2

    # 5. 信道容量 (Shannon)
    F_u = w_u_hz * math.log2(1.0 + max(snr_u, 0.0))
    F_d = w_d_hz * math.log2(1.0 + max(snr_d, 0.0))

    # 6. 传输时延
    T_u = (float(model_bits) / F_u) if F_u > 1e-12 else float("inf")
    T_d = (float(model_bits) / F_d) if F_d > 1e-12 else float("inf")
    T_total = T_u + T_d
    
    # 7. 窗口约束判断
    comm_ok = (T_total <= float(tau_d_s))

    return {
        "d_km": float(d_km),
        "tau_d_s": float(tau_d_s),
        "rho": float(rho),
        "A_atm": float(A_atm),
        "snr_u": float(snr_u),
        "snr_d": float(snr_d),
        "F_u_bps": float(F_u),
        "F_d_bps": float(F_d),
        "T_u_s": float(T_u),
        "T_d_s": float(T_d),
        "T_total_s": float(T_total),
        "window_margin_s": float(float(tau_d_s) - T_total) if math.isfinite(T_total) else float("-inf"),
        "comm_ok": 1.0 if comm_ok else 0.0,
        "model_bits": float(model_bits),
    }


def _merge_cfg(msg: Message, context: Context) -> Dict[str, Any]:
    """配置合并工具。"""
    cfg_server = dict(msg.content.get("config", {}))
    cfg_run = dict(context.run_config)
    cfg_node = dict(context.node_config)

    cfg: Dict[str, Any] = {}
    cfg.update(cfg_run)
    cfg.update(cfg_server)
    cfg.update(cfg_node)
    return cfg


def _rand_distance_km(cfg: Dict[str, Any], partition_id: int) -> float:  
    """为指定节点生成或获取缓存的随机通信距离。"""
    if partition_id in _DISTANCE_CACHE_KM:
        return _DISTANCE_CACHE_KM[partition_id]

    d_min = float(_get(cfg, "distance-km-min", 600.0))
    d_max = float(_get(cfg, "distance-km-max", 2000.0))
    base_seed = int(_get(cfg, "distance-seed", 2026))

    rng = random.Random(base_seed + int(partition_id))
    d_km = rng.uniform(d_min, d_max)

    _DISTANCE_CACHE_KM[partition_id] = d_km
    return d_km


# -----------------------------------------------------------------------------
# 客户端核心逻辑 (FO-MAML Train & Meta-Evaluate)
# -----------------------------------------------------------------------------
@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    客户端训练主入口。
    
    流程:
    1. 接收全局元模型参数 (Initial Meta-Weights)。
    2. 加载本地数据。
    3. 执行 FO-MAML (First-Order MAML) 训练：
       - Inner Loop: Support Set 微调
       - Outer Loop: Query Set 求导
       - Client Update: 返回更新后的参数
    4. 计算物理层链路指标。
    """
    
    # ---- Step 1: 初始化与加载模型 ----
    model = Net()
    state_dict_arrays = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict_arrays)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Step 2: 准备本地数据集 ----
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # ---- Step 3: 读取 FO-MAML 超参数 ----
    cfg = _merge_cfg(msg, context)
    
    # Inner Loop LR (Support Set)
    fomaml_alpha = float(_get(cfg, "fomaml-alpha", 0.01))
    # Outer Loop LR (Query Set / Client Update)
    fomaml_beta = float(_get(cfg, "fomaml-beta", 0.001))
    # Inner Steps
    fomaml_inner_steps = int(_get(cfg, "fomaml-inner-steps", 1))

    # ---- Step 4: 执行本地训练 (FO-MAML) ----
    train_loss = train_fn(
        model, 
        trainloader,
        device,
        alpha=fomaml_alpha,
        beta=fomaml_beta,
        num_inner_steps=fomaml_inner_steps
    )

    # ---- Step 5: 打包训练后的模型 ----
    state_dict = model.state_dict()
    model_record = ArrayRecord(state_dict)

    # ---- Step 6: 物理层通信仿真 (Physics Simulation) ----
    if "distance-km" in cfg:
        d_km = float(cfg["distance-km"])
    else:
        d_km = _rand_distance_km(cfg, partition_id)

    # 读取链路预算参数
    tau_d_s = float(_get(cfg, "tau-d-s", _get(cfg, "default-tau-d-s", 5.0)))
    A_T = float(_get(cfg, "A-T", 1.0))
    A_R = float(_get(cfg, "A-R", 1.0))
    f_c_hz = float(_get(cfg, "f-c-hz", 20e9))
    G_H = float(_get(cfg, "G-H", 1.0))
    delta_rician = float(_get(cfg, "delta", 1.0))
    psi_db_per_km = float(_get(cfg, "psi-db-per-km", 0.0))
    zeta_km = float(_get(cfg, "zeta-km", 550.0))
    w_u_hz = float(_get(cfg, "w-u-hz", 1e6))
    w_d_hz = float(_get(cfg, "w-d-hz", 1e6))
    p_u_w = float(_get(cfg, "p-u-w", 1.0))
    p_d_w = float(_get(cfg, "p-d-w", 1.0))
    sigma2 = float(_get(cfg, "sigma2", 1e-9))

    model_bits = _state_dict_size_bits(state_dict)

    comm_metrics = compute_link_metrics(
        model_bits=model_bits,
        d_km=d_km,
        tau_d_s=tau_d_s,
        A_T=A_T,
        A_R=A_R,
        f_c_hz=f_c_hz,
        G_H=G_H,
        delta_rician=delta_rician,
        psi_db_per_km=psi_db_per_km,
        zeta_km=zeta_km,
        w_u_hz=w_u_hz,
        p_u_w=p_u_w,
        w_d_hz=w_d_hz,
        p_d_w=p_d_w,
        sigma2=sigma2,
    )

    # ---- Step 7: 记录日志与回传 ----
    log(
        INFO, 
        "[node %s] FO-MAML Loss=%.4f, d_km=%.1f, comm_ok=%s, margin=%.2f", 
        partition_id, 
        train_loss,
        comm_metrics["d_km"], 
        int(comm_metrics["comm_ok"]), 
        comm_metrics["window_margin_s"]
    )

    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        # 回传链路指标
        "tau_d_s": comm_metrics["tau_d_s"],
        "T_total_s": comm_metrics["T_total_s"],
        "window_margin_s": comm_metrics["window_margin_s"],
        "comm_ok": comm_metrics["comm_ok"],
        "d_km": comm_metrics["d_km"],
        "F_u_bps": comm_metrics["F_u_bps"],
        "F_d_bps": comm_metrics["F_d_bps"],
        "A_atm": comm_metrics["A_atm"],
        "snr_u": comm_metrics["snr_u"],
        "snr_d": comm_metrics["snr_d"],
        "rho": comm_metrics["rho"],
        "model_bits": comm_metrics["model_bits"],
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    客户端评估回调函数 (Meta-Evaluation)。
    执行 "Adaptation -> Evaluation" 流程，验证模型的少样本适应能力。
    """
    # ---- Step 1: 加载模型 ----
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Step 2: 加载验证集 ----
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # ---- Step 3: 读取 Meta-Test 参数 ----
    cfg = _merge_cfg(msg, context)
    
    meta_adapt_steps = int(_get(cfg, "meta-adapt-steps", 5))
    meta_adapt_lr = float(_get(cfg, "meta-adapt-lr", 0.01))

    # ---- Step 4: 执行 Meta-Test (Adapt -> Test) ----
    eval_loss, eval_acc = test_fn(
        model, 
        valloader, 
        device, 
        adaptation_steps=meta_adapt_steps, 
        adaptation_lr=meta_adapt_lr
    )

    # ---- Step 5: 物理层通信仿真 ----
    if "distance-km" in cfg:
        d_km = float(cfg["distance-km"])
    else:
        d_km = _rand_distance_km(cfg, partition_id)

    # 读取链路参数
    tau_d_s = float(_get(cfg, "tau-d-s", _get(cfg, "default-tau-d-s", 5.0)))
    A_T = float(_get(cfg, "A-T", 1.0))
    A_R = float(_get(cfg, "A-R", 1.0))
    f_c_hz = float(_get(cfg, "f-c-hz", 20e9))
    G_H = float(_get(cfg, "G-H", 1.0))
    delta_rician = float(_get(cfg, "delta", 1.0))
    psi_db_per_km = float(_get(cfg, "psi-db-per-km", 0.0))
    zeta_km = float(_get(cfg, "zeta-km", 550.0))
    w_u_hz = float(_get(cfg, "w-u-hz", 1e6))
    w_d_hz = float(_get(cfg, "w-d-hz", 1e6))
    p_u_w = float(_get(cfg, "p-u-w", 1.0))
    p_d_w = float(_get(cfg, "p-d-w", 1.0))
    sigma2 = float(_get(cfg, "sigma2", 1e-9))

    model_bits = _state_dict_size_bits(model.state_dict())

    comm_metrics = compute_link_metrics(
        model_bits=model_bits,
        d_km=d_km,
        tau_d_s=tau_d_s,
        A_T=A_T,
        A_R=A_R,
        f_c_hz=f_c_hz,
        G_H=G_H,
        delta_rician=delta_rician,
        psi_db_per_km=psi_db_per_km,
        zeta_km=zeta_km,
        w_u_hz=w_u_hz,
        p_u_w=p_u_w,
        w_d_hz=w_d_hz,
        p_d_w=p_d_w,
        sigma2=sigma2,
    )

    # ---- Step 6: 回传评估结果 ----
    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
        # 链路指标
        "tau_d_s": comm_metrics["tau_d_s"],
        "T_total_s": comm_metrics["T_total_s"],
        "window_margin_s": comm_metrics["window_margin_s"],
        "comm_ok": comm_metrics["comm_ok"],
        "d_km": comm_metrics["d_km"],
        "A_atm": comm_metrics["A_atm"],
        "snr_u": comm_metrics["snr_u"],
        "snr_d": comm_metrics["snr_d"],
        "rho": comm_metrics["rho"],
        "model_bits": comm_metrics["model_bits"],
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)