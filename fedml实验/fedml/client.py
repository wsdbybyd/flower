"""
client.py — 统一客户端，支持 6 种 FL 方法：
  fedavg / fedprox / scaffold / fedkd / fedmeta / ours
"""
import math, random
from typing import Any, Dict, List, Optional
from logging import INFO

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log

from fedml.task import (
    Net, BigTeacherNet, load_data,
    train_fedavg, train_fedprox, train_scaffold,
    train_fedkd, train_fomaml, train_fomaml_apskd,
    test_meta,
)

app = ClientApp()

# 跨 Round 缓存
_DISTANCE_CACHE: Dict[int, float] = {}
_SCAFFOLD_C_LOCAL: Dict[int, List[torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _get(cfg, key, default):
    try:    return cfg[key]
    except: return default

def _merge_cfg(msg, context):
    cfg = {}
    cfg.update(dict(getattr(context, "run_config",  {}) or {}))
    cfg.update(dict(msg.content.get("config", {})))
    cfg.update(dict(getattr(context, "node_config", {}) or {}))
    return cfg

def _safe_int(ctx, key, default):
    try:    return int(ctx.node_config[key])
    except: return default

def _state_dict_bits(sd):
    return sum(t.numel()*t.element_size() for t in sd.values()
               if isinstance(t, torch.Tensor)) * 8

def _rand_dist(cfg, pid):
    if pid in _DISTANCE_CACHE: return _DISTANCE_CACHE[pid]
    rng = random.Random(int(_get(cfg,"distance-seed",2026)) + pid)
    d = rng.uniform(float(_get(cfg,"distance-km-min",600)),
                    float(_get(cfg,"distance-km-max",2000)))
    _DISTANCE_CACHE[pid] = d; return d

def _comm(cfg, pid, sd):
    d_km   = float(cfg["distance-km"]) if "distance-km" in cfg else _rand_dist(cfg, pid)
    tau    = float(_get(cfg,"tau-d-s", _get(cfg,"default-tau-d-s",5.0)))
    sigma2 = max(float(_get(cfg,"sigma2",1e-9)), 1e-18)
    w_u    = max(float(_get(cfg,"w-u-hz",1e6)), 1e-9)
    w_d    = max(float(_get(cfg,"w-d-hz",1e6)), 1e-9)
    p_u    = float(_get(cfg,"p-u-w",1.0))
    p_d    = float(_get(cfg,"p-d-w",1.0))
    fc     = max(float(_get(cfg,"f-c-hz",20e9)), 1e-9)
    A_T    = float(_get(cfg,"A-T",1.0)); A_R = float(_get(cfg,"A-R",1.0))
    GH     = float(_get(cfg,"G-H",1.0)); delta=float(_get(cfg,"delta",1.0))
    psi    = float(_get(cfg,"psi-db-per-km",0.0))
    zeta   = max(float(_get(cfg,"zeta-km",550.0)), 1e-9)
    c = 299_792_458.0; dm = d_km*1000.0
    A_atm  = 10**((3*psi*d_km)/(10*zeta))
    fspl   = (c/(4*math.pi*dm*fc))**2
    A_C    = fspl*GH*A_atm*delta
    AT_l   = 10**(A_T/10) if A_T>10 else A_T
    AR_l   = 10**(A_R/10) if A_R>10 else A_R
    rho    = AT_l*A_C*AR_l
    bits   = _state_dict_bits(sd)
    Fu = w_u*math.log2(1+max(p_u*rho**2/sigma2,0))
    Fd = w_d*math.log2(1+max(p_d*rho**2/sigma2,0))
    Tu = bits/Fu if Fu>1e-12 else float("inf")
    Td = bits/Fd if Fd>1e-12 else float("inf")
    Tt = Tu+Td
    return dict(d_km=d_km, tau_d_s=tau, rho=rho, A_atm=A_atm,
                snr_u=p_u*rho**2/sigma2, snr_d=p_d*rho**2/sigma2,
                F_u_bps=Fu, F_d_bps=Fd, T_u_s=Tu, T_d_s=Td, T_total_s=Tt,
                window_margin_s=tau-Tt if math.isfinite(Tt) else float("-inf"),
                comm_ok=1.0 if Tt<=tau else 0.0, model_bits=float(bits))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = _merge_cfg(msg, context)
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pid  = _safe_int(context, "partition-id", 0)
    npar = _safe_int(context, "num-partitions", max(pid+1, 2))
    bs   = int(_get(cfg,"batch-size",32))
    dalpha = float(_get(cfg,"dirichlet-alpha",0.1))
    trainloader, _ = load_data(pid, npar, bs, alpha=dalpha)

    mode = str(_get(cfg,"client-train-mode","fedavg")).strip().lower()
    lr   = float(_get(cfg,"learning-rate",0.01))
    ep   = int(_get(cfg,"local-epochs",1))
    wd   = float(_get(cfg,"weight-decay",1e-4))
    ls   = float(_get(cfg,"label-smoothing",0.1))
    extra: Dict[str,float] = {}

    # ── FedAvg ───────────────────────────────────────────────────────────
    if mode == "fedavg":
        loss = train_fedavg(model, trainloader, device,
                            lr=lr, epochs=ep, weight_decay=wd, label_smoothing=ls)

    # ── FedProx ──────────────────────────────────────────────────────────
    elif mode == "fedprox":
        mu = float(_get(cfg,"fedprox-mu",0.01))
        loss = train_fedprox(model, trainloader, device,
                             lr=lr, epochs=ep, mu=mu, weight_decay=wd, label_smoothing=ls)
        extra["fedprox_mu"] = mu

    # ── SCAFFOLD ─────────────────────────────────────────────────────────
    elif mode == "scaffold":
        # 解析 server 传来的全局控制变量（展平 float 字符串）
        cg_str = str(_get(cfg,"scaffold-c-global","")).strip()
        c_global = None
        if cg_str:
            try:
                flat   = [float(v) for v in cg_str.split(",")]
                shapes = [p.shape for p in model.parameters()]
                idx, c_global = 0, []
                for sh in shapes:
                    numel = 1
                    for s in sh: numel *= s
                    c_global.append(
                        torch.tensor(flat[idx:idx+numel]).reshape(sh).to(device))
                    idx += numel
            except Exception:
                c_global = None

        c_local = _SCAFFOLD_C_LOCAL.get(pid, None)
        loss, delta_c = train_scaffold(model, trainloader, device,
                                       lr=lr, epochs=ep,
                                       c_global=c_global, c_local=c_local,
                                       weight_decay=wd)
        # 更新本地控制变量缓存
        _SCAFFOLD_C_LOCAL[pid] = [
            ((c_local[i]+delta_c[i]) if c_local is not None else delta_c[i])
            for i in range(len(delta_c))]
        extra["scaffold_delta_c_norm"] = float(
            torch.cat([d.flatten() for d in delta_c]).norm().item())

    # ── FedKD ─────────────────────────────────────────────────────────────
    elif mode == "fedkd":
        teacher = None
        if "teacher_arrays" in msg.content:
            try:
                teacher = BigTeacherNet()
                teacher.load_state_dict(msg.content["teacher_arrays"].to_torch_state_dict())
                teacher.to(device).eval()
            except Exception:
                teacher = None
        kd_t = float(_get(cfg,"kd-temperature",4.0))
        kd_a = float(_get(cfg,"kd-alpha",0.5))
        loss  = train_fedkd(model, trainloader, device,
                            lr=lr, epochs=ep, teacher_model=teacher,
                            kd_temperature=kd_t, kd_alpha=kd_a, weight_decay=wd)
        extra["kd_temperature"] = kd_t; extra["kd_alpha"] = kd_a

    # ── FedMeta (FO-MAML) ────────────────────────────────────────────────
    elif mode == "fedmeta":
        loss = train_fomaml(model, trainloader, device,
                            alpha=float(_get(cfg,"fomaml-alpha",0.01)),
                            beta =float(_get(cfg,"fomaml-beta", 0.001)),
                            num_inner_steps=int(_get(cfg,"fomaml-inner-steps",5)))

    # ── Ours (FO-MAML + APSKD + KD) ─────────────────────────────────────
    else:
        prev = Net()
        prev.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        prev.to(device).eval()
        loss = train_fomaml_apskd(
            model, trainloader, device,
            alpha=float(_get(cfg,"fomaml-alpha",0.01)),
            beta =float(_get(cfg,"fomaml-beta", 0.01)),
            num_inner_steps=int(_get(cfg,"fomaml-inner-steps",7)),
            temperature=float(_get(cfg,"kd-temperature",3.0)),
            epochs=int(_get(cfg,"apskd-epochs",1)),
            current_round=int(_get(cfg,"server-round",1)),
            warmup_rounds=int(_get(cfg,"kd-warmup-rounds",20)),
            teacher_model=prev)

    # ── 通信仿真 & 回传 ──────────────────────────────────────────────────
    sd   = model.state_dict()
    cm   = _comm(cfg, pid, sd)
    mode_id = {"fedavg":0,"fedprox":1,"scaffold":2,"fedkd":3,"fedmeta":4,"ours":5}.get(mode,0)
    metrics = {
        "train_mode":mode_id, "train_loss":float(loss),
        "num-examples":len(getattr(trainloader,"dataset",[])),
        **{k:cm[k] for k in ("tau_d_s","T_total_s","window_margin_s","comm_ok",
                              "d_km","F_u_bps","F_d_bps","A_atm","snr_u","snr_d",
                              "rho","model_bits")},
    }
    metrics.update(extra)
    log(INFO,"[node %s] mode=%s loss=%.4f d_km=%.1f comm_ok=%s",
        pid, mode, float(loss), cm["d_km"], int(cm["comm_ok"]))
    return Message(content=RecordDict({"arrays":ArrayRecord(sd),"metrics":MetricRecord(metrics)}),
                   reply_to=msg)


# ---------------------------------------------------------------------------
# Evaluate — 所有方法统一 Meta-Evaluation
# ---------------------------------------------------------------------------
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg = _merge_cfg(msg, context)
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pid  = _safe_int(context,"partition-id",0)
    npar = _safe_int(context,"num-partitions",max(pid+1,2))
    _, valloader = load_data(pid, npar,
                             int(_get(cfg,"batch-size",32)),
                             alpha=float(_get(cfg,"dirichlet-alpha",0.1)))

    eval_loss, eval_acc = test_meta(
        model, valloader, device,
        adaptation_steps=int(_get(cfg,"meta-adapt-steps",5)),
        adaptation_lr   =float(_get(cfg,"meta-adapt-lr",0.01)))

    cm = _comm(cfg, pid, model.state_dict())
    metrics = {
        "eval_loss":float(eval_loss), "eval_acc":float(eval_acc),
        "num-examples":len(getattr(valloader,"dataset",[])),
        **{k:cm[k] for k in ("tau_d_s","T_total_s","window_margin_s","comm_ok",
                              "d_km","A_atm","snr_u","snr_d","rho","model_bits")},
    }
    return Message(content=RecordDict({"metrics":MetricRecord(metrics)}), reply_to=msg)