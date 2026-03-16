"""
client.py  ——  fedml-satellite ClientApp
数据集: NWPU-RESISC45 (45 类遥感场景分类, 64×64 RGB)
物理层通信仿真：LEO 卫星链路预算模型
"""

import math
import random
from typing import Any, Dict
from logging import INFO

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

from fedml.task import Net, load_data, NUM_CLASSES

# ---- 训练算法导入 ----
from fedml.task import train_fomaml    as train_fomaml_fn
from fedml.task import test_meta       as test_meta_fn
from fedml.task import train_apskd     as train_apskd_fn
from fedml.task import train_fomaml_apskd as train_fomaml_apskd_fn
from fedml.task import train_fedavg    as train_fedavg_fn

app = ClientApp()

# ---- 节点距离缓存（同一节点跨轮次保持距离不变）----
_DISTANCE_CACHE_KM: Dict[int, float] = {}


# -----------------------------------------------------------------------------
# 辅助工具
# -----------------------------------------------------------------------------
def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except Exception:
        return default


def _state_dict_size_bits(state_dict: Dict[str, torch.Tensor]) -> int:
    """计算模型参数字典总大小（bits）"""
    total_bytes = 0
    for _, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            total_bytes += t.numel() * t.element_size()
    return int(total_bytes * 8)


def _merge_cfg(msg: Message, context: Context) -> Dict[str, Any]:
    """配置合并：run_config < server config < node_config"""
    cfg_server = dict(msg.content.get("config", {}))
    cfg_run    = dict(getattr(context, "run_config", {}) or {})
    cfg_node   = dict(getattr(context, "node_config", {}) or {})
    cfg: Dict[str, Any] = {}
    cfg.update(cfg_run)
    cfg.update(cfg_server)
    cfg.update(cfg_node)
    return cfg


def _safe_int_from_context(context: Context, key: str, default: int) -> int:
    try:
        return int(context.node_config[key])
    except Exception:
        return int(default)


def _rand_distance_km(cfg: Dict[str, Any], partition_id: int) -> float:
    """为指定节点生成或获取缓存的随机通信距离（LEO 轨道高度范围）"""
    if partition_id in _DISTANCE_CACHE_KM:
        return _DISTANCE_CACHE_KM[partition_id]
    d_min    = float(_get(cfg, "distance-km-min", 300.0))
    d_max    = float(_get(cfg, "distance-km-max", 1000.0))
    base_seed = int(_get(cfg, "distance-seed", 2026))
    rng      = random.Random(base_seed + int(partition_id))
    d_km     = rng.uniform(d_min, d_max)
    _DISTANCE_CACHE_KM[partition_id] = d_km
    return d_km


# -----------------------------------------------------------------------------
# 物理层通信仿真（LEO 卫星链路预算）
# -----------------------------------------------------------------------------
def compute_link_metrics(
    *, model_bits: int, d_km: float, tau_d_s: float,
    A_T: float, A_R: float, f_c_hz: float, G_H: float,
    delta_rician: float, psi_db_per_km: float, zeta_km: float,
    w_u_hz: float, p_u_w: float, w_d_hz: float, p_d_w: float, sigma2: float,
) -> Dict[str, float]:
    """计算物理层链路预算及通信性能指标"""
    c       = 299_792_458.0
    d_m     = max(float(d_km), 1e-9) * 1000.0
    zeta_km = max(float(zeta_km), 1e-9)

    A_atm = 10.0 ** ((3.0 * float(psi_db_per_km) * float(d_km)) / (10.0 * zeta_km))
    fspl  = (c / (4.0 * math.pi * d_m * max(float(f_c_hz), 1e-9))) ** 2
    A_C   = fspl * float(G_H) * A_atm * float(delta_rician)

    A_T_lin = (10.0 ** (float(A_T) / 10.0)) if float(A_T) > 10.0 else float(A_T)
    A_R_lin = (10.0 ** (float(A_R) / 10.0)) if float(A_R) > 10.0 else float(A_R)
    rho     = A_T_lin * A_C * A_R_lin

    sigma2  = max(float(sigma2), 1e-18)
    w_u_hz  = max(float(w_u_hz), 1e-9)
    w_d_hz  = max(float(w_d_hz), 1e-9)

    snr_u = (float(p_u_w) * (rho ** 2)) / sigma2
    snr_d = (float(p_d_w) * (rho ** 2)) / sigma2

    F_u = w_u_hz * math.log2(1.0 + max(snr_u, 0.0))
    F_d = w_d_hz * math.log2(1.0 + max(snr_d, 0.0))

    T_u     = (float(model_bits) / F_u) if F_u > 1e-12 else float("inf")
    T_d     = (float(model_bits) / F_d) if F_d > 1e-12 else float("inf")
    T_total = T_u + T_d
    comm_ok = (T_total <= float(tau_d_s))

    return {
        "d_km":           float(d_km),
        "tau_d_s":        float(tau_d_s),
        "rho":            float(rho),
        "A_atm":          float(A_atm),
        "snr_u":          float(snr_u),
        "snr_d":          float(snr_d),
        "F_u_bps":        float(F_u),
        "F_d_bps":        float(F_d),
        "T_u_s":          float(T_u),
        "T_d_s":          float(T_d),
        "T_total_s":      float(T_total),
        "window_margin_s": float(float(tau_d_s) - T_total) if math.isfinite(T_total) else float("-inf"),
        "comm_ok":        1.0 if comm_ok else 0.0,
        "model_bits":     float(model_bits),
    }


def _extract_comm_params(cfg: Dict[str, Any]) -> Dict[str, float]:
    """从配置字典提取物理层参数"""
    return dict(
        tau_d_s       = float(_get(cfg, "tau-d-s",        _get(cfg, "default-tau-d-s", 5.0))),
        A_T           = float(_get(cfg, "A-T",            1.0)),
        A_R           = float(_get(cfg, "A-R",            1.0)),
        f_c_hz        = float(_get(cfg, "f-c-hz",         20e9)),
        G_H           = float(_get(cfg, "G-H",            1.0)),
        delta_rician  = float(_get(cfg, "delta",          1.0)),
        psi_db_per_km = float(_get(cfg, "psi-db-per-km",  0.0)),
        zeta_km       = float(_get(cfg, "zeta-km",        550.0)),
        w_u_hz        = float(_get(cfg, "w-u-hz",         1e6)),
        w_d_hz        = float(_get(cfg, "w-d-hz",         1e6)),
        p_u_w         = float(_get(cfg, "p-u-w",          1.0)),
        p_d_w         = float(_get(cfg, "p-d-w",          1.0)),
        sigma2        = float(_get(cfg, "sigma2",         1e-9)),
    )


# -----------------------------------------------------------------------------
# 客户端核心逻辑
# -----------------------------------------------------------------------------
@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    客户端训练回调。
    支持: FedAvg / FO-MAML / APSKD / FO-MAML+APSKD
    数据集: NWPU-RESISC45 (45 类遥感场景)
    """
    cfg = _merge_cfg(msg, context)

    # ---- 加载模型（45 类）----
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- 加载本地分区数据 ----
    partition_id   = _safe_int_from_context(context, "partition-id",   0)
    num_partitions = _safe_int_from_context(context, "num-partitions", max(partition_id + 1, 2))
    batch_size     = int(_get(cfg, "batch-size", 32))
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # ---- 选择训练模式 ----
    train_mode = str(_get(cfg, "client-train-mode", "fomaml")).strip().lower()

    if train_mode == "fedavg":
        train_loss = train_fedavg_fn(
            model, trainloader, device,
            lr              = float(_get(cfg, "learning-rate",          0.01)),
            epochs          = int(  _get(cfg, "fedavg-local-epochs",    1)),
            weight_decay    = float(_get(cfg, "fedavg-weight-decay",    1e-4)),
            label_smoothing = float(_get(cfg, "fedavg-label-smoothing", 0.1)),
        )
    elif train_mode == "apskd":
        train_loss = train_apskd_fn(
            model, trainloader, device,
            lr          = float(_get(cfg, "apskd-lr",        0.01)),
            epochs      = int(  _get(cfg, "apskd-epochs",    1)),
            temperature = float(_get(cfg, "kd-temperature",  4.0)),
        )
    elif train_mode == "fomaml+apskd":
        train_loss = train_fomaml_apskd_fn(
            model, trainloader, device,
            alpha           = float(_get(cfg, "fomaml-alpha",       0.01)),
            beta            = float(_get(cfg, "fomaml-beta",        0.001)),
            num_inner_steps = int(  _get(cfg, "fomaml-inner-steps", 1)),
            temperature     = float(_get(cfg, "kd-temperature",     4.0)),
            epochs          = int(  _get(cfg, "apskd-epochs",       1)),
            current_round   = int(  _get(cfg, "server-round",       1)),
            warmup_rounds   = int(  _get(cfg, "kd-warmup-rounds",   50)),
        )
    else:  # 默认：纯 FO-MAML
        train_loss = train_fomaml_fn(
            model, trainloader, device,
            alpha           = float(_get(cfg, "fomaml-alpha",       0.01)),
            beta            = float(_get(cfg, "fomaml-beta",        0.001)),
            num_inner_steps = int(  _get(cfg, "fomaml-inner-steps", 1)),
        )

    # ---- 打包训练后的模型 ----
    state_dict   = model.state_dict()
    model_record = ArrayRecord(state_dict)

    # ---- 物理层通信仿真 ----
    d_km = float(cfg["distance-km"]) if "distance-km" in cfg \
           else _rand_distance_km(cfg, partition_id)
    comm_params  = _extract_comm_params(cfg)
    model_bits   = _state_dict_size_bits(state_dict)
    comm_metrics = compute_link_metrics(model_bits=model_bits, d_km=d_km, **comm_params)

    log(
        INFO,
        "[node %s | RESISC45/%d cls] mode=%s loss=%.4f d_km=%.1f comm_ok=%s margin=%.2fs",
        partition_id, NUM_CLASSES, train_mode, float(train_loss),
        comm_metrics["d_km"], int(comm_metrics["comm_ok"]), comm_metrics["window_margin_s"],
    )

    # ---- 回传 ----
    train_mode_id = {"apskd": 1.0, "fomaml+apskd": 2.0, "fedavg": 3.0}.get(train_mode, 0.0)

    metrics = {
        "train_mode":      train_mode_id,
        "train_loss":      float(train_loss),
        "num-examples":    len(getattr(trainloader, "dataset", [])),
        "tau_d_s":         comm_metrics["tau_d_s"],
        "T_total_s":       comm_metrics["T_total_s"],
        "window_margin_s": comm_metrics["window_margin_s"],
        "comm_ok":         comm_metrics["comm_ok"],
        "d_km":            comm_metrics["d_km"],
        "F_u_bps":         comm_metrics["F_u_bps"],
        "F_d_bps":         comm_metrics["F_d_bps"],
        "A_atm":           comm_metrics["A_atm"],
        "snr_u":           comm_metrics["snr_u"],
        "snr_d":           comm_metrics["snr_d"],
        "rho":             comm_metrics["rho"],
        "model_bits":      comm_metrics["model_bits"],
    }

    metric_record = MetricRecord(metrics)
    content       = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    客户端评估回调（Meta-Evaluation）。
    FedMeta 系列：Adapt → Test；FedAvg：zero-shot。
    """
    cfg = _merge_cfg(msg, context)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id   = _safe_int_from_context(context, "partition-id",   0)
    num_partitions = _safe_int_from_context(context, "num-partitions", max(partition_id + 1, 2))
    batch_size     = int(_get(cfg, "batch-size", 32))
    _, valloader   = load_data(partition_id, num_partitions, batch_size)

    train_mode       = str(_get(cfg, "client-train-mode", "fomaml")).strip().lower()
    meta_adapt_steps = int(  _get(cfg, "meta-adapt-steps", 5))
    meta_adapt_lr    = float(_get(cfg, "meta-adapt-lr",    0.01))

    if train_mode == "fedavg":
        eval_loss, eval_acc = test_meta_fn(
            model, valloader, device,
            adaptation_steps=meta_adapt_steps,
            adaptation_lr=meta_adapt_lr,
        )
    else:
        eval_loss, eval_acc = test_meta_fn(
            model, valloader, device,
            adaptation_steps=meta_adapt_steps,
            adaptation_lr=meta_adapt_lr,
        )

    # ---- 物理层通信仿真（评估阶段同样上报，用于服务器调度）----
    d_km = float(cfg["distance-km"]) if "distance-km" in cfg \
           else _rand_distance_km(cfg, partition_id)
    comm_params  = _extract_comm_params(cfg)
    model_bits   = _state_dict_size_bits(model.state_dict())
    comm_metrics = compute_link_metrics(model_bits=model_bits, d_km=d_km, **comm_params)

    metrics = {
        "eval_loss":       float(eval_loss),
        "eval_acc":        float(eval_acc),
        "num-examples":    len(getattr(valloader, "dataset", [])),
        "tau_d_s":         comm_metrics["tau_d_s"],
        "T_total_s":       comm_metrics["T_total_s"],
        "window_margin_s": comm_metrics["window_margin_s"],
        "comm_ok":         comm_metrics["comm_ok"],
        "d_km":            comm_metrics["d_km"],
        "A_atm":           comm_metrics["A_atm"],
        "snr_u":           comm_metrics["snr_u"],
        "snr_d":           comm_metrics["snr_d"],
        "rho":             comm_metrics["rho"],
        "model_bits":      comm_metrics["model_bits"],
    }

    metric_record = MetricRecord(metrics)
    content       = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)