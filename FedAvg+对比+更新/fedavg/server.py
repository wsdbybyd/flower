import copy
import torch
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedavg.fedavg import FedAvg
from fedavg.task import (
    BigTeacherNet,
    Net,
    distill_centralized,
    load_centralized_dataset,
    test_centralized_dataset,
    train_centralized,
    test,
)

app = ServerApp()


def _bool_cfg(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    testloader = test_centralized_dataset()
    test_loss, test_acc = test(model, testloader, device)
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})


def run_strategy_session(*, title: str, grid: Grid, context: Context, initial_model: Net,
                         local_train_mode: str, apskd_alpha: float,
                         enable_dual_distillation: bool) -> object:
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 1.0))
    num_rounds = int(context.run_config.get("num-server-rounds", 50))
    lr = float(context.run_config.get("learning-rate", 0.01))

    selection_mode = str(context.run_config.get("selection-mode", "random"))
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    enforce_comm = _bool_cfg(context.run_config.get("enforce-comm", False), False)
    turnover_T = int(context.run_config.get("turnover-T", 10))
    deltaQ = int(context.run_config.get("deltaQ", 999999))
    deltaP = int(context.run_config.get("deltaP", 999999))
    probe_blocked_k = int(context.run_config.get("probe-blocked-k", 1))

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
        "distance-km-min": float(context.run_config.get("distance-km-min", 300.0)),
        "distance-km-max": float(context.run_config.get("distance-km-max", 1000.0)),
        "distance-seed": int(context.run_config.get("distance-seed", 2026)),
    }

    arrays = ArrayRecord(copy.deepcopy(initial_model.state_dict()))
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
        enable_dual_distillation=enable_dual_distillation,
    )

    train_cfg = ConfigRecord({
        "lr": lr,
        "local-train-mode": local_train_mode,
        "apskd-a-T": float(apskd_alpha),
        **comm_defaults,
    })
    eval_cfg = ConfigRecord({**comm_defaults})

    print(f"\n>>>>>>>>>>>>>>>>>>>> 开始运行策略: {title} <<<<<<<<<<<<<<<<<<<<")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    return result


def plot_comparison(res_baseline, res_only_ground, res_dual):
    def get_series(result):
        metrics = result.evaluate_metrics_serverapp
        rounds = sorted(metrics.keys())
        acc = [metrics[r].get("accuracy", 0.0) for r in rounds]
        loss = [metrics[r].get("loss", 0.0) for r in rounds]
        return rounds, acc, loss

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for result, label in [
        (res_baseline, "Baseline"),
        (res_only_ground, "Only Ground Distill"),
        (res_dual, "Dual Distill"),
    ]:
        rounds, acc, _ = get_series(result)
        plt.plot(rounds, acc, marker="o", label=label)
    plt.title("Server Accuracy Comparison")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    for result, label in [
        (res_baseline, "Baseline"),
        (res_only_ground, "Only Ground Distill"),
        (res_dual, "Dual Distill"),
    ]:
        rounds, _, loss = get_series(result)
        plt.plot(rounds, loss, marker="o", label=label)
    plt.title("Server Loss Comparison")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig("comparison_three_way.png", dpi=300)
    print("\n📈 三模式对比图已保存为 comparison_three_way.png")


@app.main()
def main(grid: Grid, context: Context) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- 地面教师预训练与初始化蒸馏 ----------
    teacher_pretrain_epochs = int(context.run_config.get("teacher-pretrain-epochs", 3))
    teacher_lr = float(context.run_config.get("teacher-pretrain-lr", 0.01))
    gs_distill_epochs = int(context.run_config.get("gs-distill-epochs", 3))
    gs_distill_lr = float(context.run_config.get("gs-distill-lr", 0.01))
    gs_distill_alpha = float(context.run_config.get("gs-distill-alpha", 0.5))
    gs_distill_temp = float(context.run_config.get("gs-distill-temp", 2.0))
    target_apskd = float(context.run_config.get("apskd-a-T", 0.5))

    trainloader = load_centralized_dataset()

    teacher = BigTeacherNet().to(device)
    train_centralized(teacher, trainloader, teacher_pretrain_epochs, teacher_lr, device)

    random_model = Net()
    distilled_model = Net()
    distill_centralized(
        distilled_model,
        teacher,
        trainloader,
        gs_distill_epochs,
        gs_distill_lr,
        device,
        temp=gs_distill_temp,
        alpha=gs_distill_alpha,
    )

    # 1) Baseline: 随机初始化 + 干净 FedAvg
    res_baseline = run_strategy_session(
        title="1. Baseline",
        grid=grid,
        context=context,
        initial_model=random_model,
        local_train_mode="fedavg",
        apskd_alpha=0.0,
        enable_dual_distillation=False,
    )

    # 2) Only Ground: 地面蒸馏 warm start + 干净 FedAvg
    res_only_ground = run_strategy_session(
        title="2. Only Ground Distill",
        grid=grid,
        context=context,
        initial_model=distilled_model,
        local_train_mode="fedavg",
        apskd_alpha=0.0,
        enable_dual_distillation=False,
    )

    # 3) Dual Distill: 地面蒸馏 warm start + APSKD + 服务端双向蒸馏开关
    res_dual = run_strategy_session(
        title="3. Dual Distill",
        grid=grid,
        context=context,
        initial_model=distilled_model,
        local_train_mode="apskd",
        apskd_alpha=target_apskd,
        enable_dual_distillation=True,
    )

    plot_comparison(res_baseline, res_only_ground, res_dual)
