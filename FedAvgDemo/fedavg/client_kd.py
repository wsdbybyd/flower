import math
import random
from typing import Any, Dict

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log
from logging import INFO

from fedavg.task import Net, load_data
from fedavg.task import test as test_fn
from fedavg.task import train as train_fn


app = ClientApp()

_DISTANCE_CACHE_KM: Dict[int, float] = {}


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except Exception:
        return default


def _state_dict_size_bits(state_dict: Dict[str, torch.Tensor]) -> int:
    total_bytes = 0
    for _, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            total_bytes += t.numel() * t.element_size()
    return int(total_bytes * 8)


def compute_link_metrics(
    *,
    model_bits: int,
    d_km: float,
    tau_d_s: float,
    A_T: float,
    A_R: float,
    f_c_hz: float,
    G_H: float,
    delta_rician: float,
    psi_db_per_km: float,
    zeta_km: float,
    w_u_hz: float,
    p_u_w: float,
    w_d_hz: float,
    p_d_w: float,
    sigma2: float,
) -> Dict[str, float]:
    c = 299_792_458.0
    d_m = max(float(d_km), 1e-9) * 1000.0
    zeta_km = max(float(zeta_km), 1e-9)

    A_atm = 10.0 ** ((3.0 * float(psi_db_per_km) * float(d_km)) / (10.0 * zeta_km))
    fspl = (c / (4.0 * math.pi * d_m * max(float(f_c_hz), 1e-9))) ** 2
    A_C = fspl * float(G_H) * A_atm * float(delta_rician)

    A_T_lin = (10.0 ** (float(A_T) / 10.0)) if float(A_T) > 10.0 else float(A_T)
    A_R_lin = (10.0 ** (float(A_R) / 10.0)) if float(A_R) > 10.0 else float(A_R)

    rho = A_T_lin * A_C * A_R_lin

    sigma2 = max(float(sigma2), 1e-18)
    w_u_hz = max(float(w_u_hz), 1e-9)
    w_d_hz = max(float(w_d_hz), 1e-9)

    snr_u = (float(p_u_w) * (rho ** 2)) / sigma2
    snr_d = (float(p_d_w) * (rho ** 2)) / sigma2

    F_u = w_u_hz * math.log2(1.0 + max(snr_u, 0.0))
    F_d = w_d_hz * math.log2(1.0 + max(snr_d, 0.0))

    T_u = (float(model_bits) / F_u) if F_u > 1e-12 else float("inf")
    T_d = (float(model_bits) / F_d) if F_d > 1e-12 else float("inf")

    T_total = T_u + T_d
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
    cfg_server = dict(msg.content.get("config", {}))
    cfg_run = dict(context.run_config)
    cfg_node = dict(context.node_config)
    cfg: Dict[str, Any] = {}
    cfg.update(cfg_run)
    cfg.update(cfg_server)
    cfg.update(cfg_node)
    return cfg


def _rand_distance_km(cfg: Dict[str, Any], partition_id: int) -> float:  
    if partition_id in _DISTANCE_CACHE_KM:
        return _DISTANCE_CACHE_KM[partition_id]
    d_min = float(_get(cfg, "distance-km-min", 600.0))
    d_max = float(_get(cfg, "distance-km-max", 2000.0))
    base_seed = int(_get(cfg, "distance-seed", 2026))
    rng = random.Random(base_seed + int(partition_id))
    d_km = rng.uniform(d_min, d_max)
    _DISTANCE_CACHE_KM[partition_id] = d_km
    return d_km


@app.train()
def train(msg: Message, context: Context) -> Message:
    # ---- 1. 准备模型 (Student) ----
    model = Net()
    state_dict_arrays = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict_arrays)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- 2. 准备教师模型 (Teacher) - 知识蒸馏 ----
    # 教师模型使用接收到的全局参数（即本轮训练的起点），并在本地保持冻结
    teacher_model = Net()
    teacher_model.load_state_dict(state_dict_arrays) # 加载相同的全局参数
    teacher_model.to(device)
    # 冻结教师模型参数，不消耗梯度计算资源
    for param in teacher_model.parameters():
        param.requires_grad = False

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # 读取 KD 参数
    cfg_temp = _merge_cfg(msg, context) # 临时合并用于读取训练参数
    # 如果 config 里没有配，默认为 0.0 (不启用)
    kd_alpha = float(_get(cfg_temp, "kd-alpha", 0.0))
    kd_temperature = float(_get(cfg_temp, "kd-temperature", 1.0))

    # ---- 3. 执行本地训练 (带蒸馏) ----
    train_loss = train_fn(
        model, 
        teacher_model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(msg.content["config"]["lr"]),
        device,
        kd_alpha=kd_alpha,
        kd_temperature=kd_temperature
    )

    # ---- 4. 训练后参数打包 ----
    state_dict = model.state_dict()
    model_record = ArrayRecord(state_dict)

    # ---- 5. 链路与通信指标计算 (逻辑不变) ----
    cfg = _merge_cfg(msg, context)
    if "distance-km" in cfg:
        d_km = float(cfg["distance-km"])
    else:
        d_km = _rand_distance_km(cfg, partition_id)

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
        model_bits=model_bits, d_km=d_km, tau_d_s=tau_d_s, A_T=A_T, A_R=A_R,
        f_c_hz=f_c_hz, G_H=G_H, delta_rician=delta_rician, psi_db_per_km=psi_db_per_km,
        zeta_km=zeta_km, w_u_hz=w_u_hz, p_u_w=p_u_w, w_d_hz=w_d_hz, p_d_w=p_d_w, sigma2=sigma2,
    )

    log(INFO, "[node %s] d_km=%.1f, comm_ok=%s, margin=%.2f, T_total=%.2f, KD_alpha=%.2f", 
        partition_id, comm_metrics["d_km"], int(comm_metrics["comm_ok"]), 
        comm_metrics["window_margin_s"], comm_metrics["T_total_s"], kd_alpha)

    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
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
    # Evaluate 逻辑保持不变
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    _, valloader = load_data(partition_id, num_partitions, batch_size)
    eval_loss, eval_acc = test_fn(model, valloader, device)

    cfg = _merge_cfg(msg, context)
    if "distance-km" in cfg:
        d_km = float(cfg["distance-km"])
    else:
        d_km = _rand_distance_km(cfg, partition_id)

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
        model_bits=model_bits, d_km=d_km, tau_d_s=tau_d_s, A_T=A_T, A_R=A_R,
        f_c_hz=f_c_hz, G_H=G_H, delta_rician=delta_rician, psi_db_per_km=psi_db_per_km,
        zeta_km=zeta_km, w_u_hz=w_u_hz, p_u_w=p_u_w, w_d_hz=w_d_hz, p_d_w=p_d_w, sigma2=sigma2,
    )

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
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