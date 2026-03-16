import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedavg.fedavg import FedAvg
from fedavg.task import Net, load_centralized_dataset, test

app = ServerApp()


@app.main() 
def main(grid: Grid, context: Context) -> None:
    # ---------- 基础训练参数 ----------  
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.5))
    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))
    lr: float = float(context.run_config.get("learning-rate", 0.1))

    # ---------- 知识蒸馏参数 (New) ----------
    # 从 pyproject.toml 读取 KD 参数
    kd_alpha: float = float(context.run_config.get("kd-alpha", 0.0))
    kd_temperature: float = float(context.run_config.get("kd-temperature", 1.0))

    # ---------- 调度退出加入参数 ---------- 
    selection_mode: str = str(context.run_config.get("selection-mode", "random"))
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", False))

    turnover_T: int = int(context.run_config.get("turnover-T", 10))
    deltaQ: int = int(context.run_config.get("deltaQ", 999999))
    deltaP: int = int(context.run_config.get("deltaP", 999999))
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))

    # ---------- 通信链路参数 ----------  
    sigma2 = float(context.run_config.get("sigma2", 1e-9))

    comm_defaults = {  
        "sigma2": sigma2,
        "default-distance-km": float(context.run_config.get("default-distance-km", 700.0)),
        "default-tau-d-s": float(context.run_config.get("default-tau-d-s", 5.0)),
        "w-u-hz": float(context.run_config.get("w-u-hz", 1e6)),
        "w-d-hz": float(context.run_config.get("w-d-hz", 1e6)),
        "p-u-w": float(context.run_config.get("p-u-w", 1.0)),
        "p-d-w": float(context.run_config.get("p-d-w", 1.0)),
        "f-c-hz": float(context.run_config.get("f-c-hz", 20e9)),
        "A-T": float(context.run_config.get("A-T", 1.0)),
        "A-R": float(context.run_config.get("A-R", 1.0)),
        "G-H": float(context.run_config.get("G-H", 1.0)),
        "delta": float(context.run_config.get("delta", 1.0)),
        "psi-db-per-km": float(context.run_config.get("psi-db-per-km", 0.0)),
        "zeta-km": float(context.run_config.get("zeta-km", 550.0)),
    }

    # ---------- 初始化全局模型 ----------  
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ---------- 构建策略 ----------  
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        selection_mode=selection_mode,
        ks_train=int(ks_train) if ks_train is not None else None,
        ks_evaluate=int(ks_eval) if ks_eval is not None else None,
        enforce_comm=enforce_comm,
        turnover_T=turnover_T,
        deltaQ=deltaQ,
        deltaP=deltaP,
        probe_blocked_k=probe_blocked_k,
    )

    # ---------- 下发给客户端的配置 ----------  
    # 将 kd-alpha 和 kd-temperature 加入到训练配置中
    train_cfg = ConfigRecord({
        "lr": lr, 
        "kd-alpha": kd_alpha, 
        "kd-temperature": kd_temperature, 
        **comm_defaults
    })
    
    eval_cfg = ConfigRecord({**comm_defaults})

    # ---------- 启动联邦训练 ----------  
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})