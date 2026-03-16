import torch  # 导入 PyTorch
import matplotlib.pyplot as plt  # 导入绘图库
from logging import INFO, WARNING

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # Flower 核心数据结构
from flwr.common import log
from flwr.serverapp import Grid, ServerApp  # Flower ServerApp

# ---- FedMeta 策略（含通信感知调度 + 成员管理 + 双向蒸馏）----
from fedml.fedml import FedMeta

# ---- Task: 学生模型 + 中心化测试集 + 元评估（Adapt->Test）----
from fedml.task import Net, test_centralized_dataset, test_meta

app = ServerApp()


# ============================================================
#  绘图工具函数
# ============================================================
def plot_metrics(result, filename: str = "training_metrics.png") -> None:
    """
    解析 Result 对象并绘制 Server-side Accuracy 和 Loss 的折线图。
    注意：这里画的是 server 侧 centralized meta-test（Adapt->Test）的曲线。
    """
    history = getattr(result, "evaluate_metrics_serverapp", None)
    if not history:
        log(INFO, "No server-side evaluation metrics found to plot.")
        return

    rounds = sorted(history.keys())
    accuracies = []
    losses = []
    for r in rounds:
        metrics = history[r] or {}
        accuracies.append(metrics.get("accuracy", None))
        losses.append(metrics.get("loss", None))

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Accuracy ---
    valid_acc = [(r, v) for r, v in zip(rounds, accuracies) if v is not None]
    if valid_acc:
        rs, vs = zip(*valid_acc)
        ax1.plot(rs, vs, marker="o", linewidth=2, label="Accuracy (Post-Adaptation)")
        ax1.set_title("Server-side Meta-Test Accuracy", fontsize=14)
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.grid(True)
        ax1.legend()
        from matplotlib.ticker import MaxNLocator

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Loss ---
    valid_loss = [(r, v) for r, v in zip(rounds, losses) if v is not None]
    if valid_loss:
        rs, vs = zip(*valid_loss)
        ax2.plot(rs, vs, marker="o", linewidth=2, label="Loss")
        ax2.set_title("Server-side Meta-Test Loss", fontsize=14)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.grid(True)
        ax2.legend()
        from matplotlib.ticker import MaxNLocator

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    log(INFO, f"Training plots saved to: {filename}")


# ============================================================
#  构造 server-side centralized meta-evaluate 闭包
# ============================================================
def make_global_evaluate(meta_adapt_steps: int, meta_adapt_lr: float) -> callable:
    """
    返回一个 evaluate_fn(server_round, arrays) -> MetricRecord
    用闭包把 meta-adapt-steps / meta-adapt-lr 带进去，避免硬编码。
    """

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 完整测试集（HF datasets）
        test_dataloader = test_centralized_dataset()

        # 元评估：先适应再测试（更符合 FO-MAML / Meta-Learning 的目标）
        test_loss, test_acc = test_meta(
            model,
            test_dataloader,
            device,
            adaptation_steps=int(meta_adapt_steps),
            adaptation_lr=float(meta_adapt_lr),
        )

        return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})

    return global_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Flower ServerApp 主入口。
    负责：
      1) 读取 pyproject.toml 的 run_config
      2) 构建 FedMeta（含调度/成员管理/双向蒸馏）
      3) 启动训练与 server-side meta-test
      4) 保存最终 student/teacher 与绘图
    """

    # ============================================================
    # 1. 基础训练参数
    # ============================================================
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 1.0))
    num_rounds: int = int(context.run_config.get("num-server-rounds", 5))

    # ============================================================
    # 2. 调度与成员管理参数
    # ============================================================
    selection_mode: str = str(context.run_config.get("selection-mode", "window"))
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", True))

    turnover_T: int = int(context.run_config.get("turnover-T", 5))
    deltaQ: int = int(context.run_config.get("deltaQ", 2))
    deltaP: int = int(context.run_config.get("deltaP", 3))
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))

    # ============================================================
    # 3. FO-MAML 训练参数（下发给 Client）
    # ============================================================
    batch_size: int = int(context.run_config.get("batch-size", 32))
    fomaml_config = {
        "fomaml-alpha": float(context.run_config.get("fomaml-alpha", 0.01)),
        "fomaml-beta": float(context.run_config.get("fomaml-beta", 0.001)),
        "fomaml-inner-steps": int(context.run_config.get("fomaml-inner-steps", 1)),
        "batch-size": batch_size,
    }

    # ============================================================
    # 4. Meta-Testing（评估）参数（Client evaluate + Server evaluate）
    # ============================================================
    meta_adapt_steps: int = int(context.run_config.get("meta-adapt-steps", 10))
    meta_adapt_lr: float = float(context.run_config.get("meta-adapt-lr", 0.01))
    meta_eval_config = {
        "meta-adapt-steps": meta_adapt_steps,
        "meta-adapt-lr": meta_adapt_lr,
        "batch-size": batch_size,
    }

    # ============================================================
    # 5. 双向蒸馏（Bidirectional KD）参数（Server 侧）
    # ============================================================
    kd_enable: bool = bool(context.run_config.get("kd-enable", False))
    kd_alpha: float = float(context.run_config.get("kd-alpha", 0.5))
    kd_temperature: float = float(context.run_config.get("kd-temperature", 4.0))
    kd_batch_size: int = int(context.run_config.get("kd-batch-size", 64))
    kd_cal_samples: int = int(context.run_config.get("kd-cal-samples", 2048))
    kd_global_samples: int = int(context.run_config.get("kd-global-samples", 4096))
    kd_forward_epochs: int = int(context.run_config.get("kd-forward-epochs", 1))
    kd_forward_lr: float = float(context.run_config.get("kd-forward-lr", 0.05))
    kd_reverse_epochs: int = int(context.run_config.get("kd-reverse-epochs", 1))
    kd_reverse_lr: float = float(context.run_config.get("kd-reverse-lr", 0.01))

    # 可选：客户端本地训练模式（默认 FO-MAML；可切 APSKD）
    client_train_mode: str = str(context.run_config.get("client-train-mode", "fomaml"))
    apskd_epochs: int = int(context.run_config.get("apskd-epochs", 1))
    apskd_lr: float = float(context.run_config.get("apskd-lr", 0.01))

    if kd_enable:
        log(
            INFO,
            "Bidirectional KD enabled: alpha=%.3f, T=%.2f, cal=%d, global=%d, fwd(ep=%d,lr=%.4f), rev(ep=%d,lr=%.4f)",
            kd_alpha,
            kd_temperature,
            kd_cal_samples,
            kd_global_samples,
            kd_forward_epochs,
            kd_forward_lr,
            kd_reverse_epochs,
            kd_reverse_lr,
        )
    else:
        log(INFO, "Bidirectional KD disabled.")

    # ============================================================
    # 6. 通信链路参数（下发给 Client，用于链路时延/窗口判定）
    # ============================================================
    sigma2 = float(context.run_config.get("sigma2", 1e-9))
    comm_defaults = {
        "sigma2": sigma2,
        "default-distance-km": float(context.run_config.get("default-distance-km", 780.0)),
        "default-tau-d-s": float(context.run_config.get("default-tau-d-s", 50000.0)),
        "w-u-hz": float(context.run_config.get("w-u-hz", 4e6)),
        "w-d-hz": float(context.run_config.get("w-d-hz", 4e6)),
        "p-u-w": float(context.run_config.get("p-u-w", 100.0)),
        "p-d-w": float(context.run_config.get("p-d-w", 100.0)),
        "f-c-hz": float(context.run_config.get("f-c-hz", 20e9)),
        "A-T": float(context.run_config.get("A-T", 60.0)),
        "A-R": float(context.run_config.get("A-R", 30.0)),
        "G-H": float(context.run_config.get("G-H", 0.8)),
        "delta": float(context.run_config.get("delta", 2.0)),
        "psi-db-per-km": float(context.run_config.get("psi-db-per-km", 0.5)),
        "zeta-km": float(context.run_config.get("zeta-km", 500.0)),
    }

    # ============================================================
    # 7. 初始化全局学生模型
    # ============================================================
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ============================================================
    # 8. 构建策略（FedMeta）
    # ============================================================
    strategy = FedMeta(
        fraction_evaluate=fraction_evaluate,
        selection_mode=selection_mode,
        ks_train=int(ks_train) if ks_train is not None else None,
        ks_evaluate=int(ks_eval) if ks_eval is not None else None,
        enforce_comm=enforce_comm,
        turnover_T=turnover_T,
        deltaQ=deltaQ,
        deltaP=deltaP,
        probe_blocked_k=probe_blocked_k,
        # --- Bidirectional KD ---
        kd_enable=kd_enable,
        kd_alpha=kd_alpha,
        kd_temperature=kd_temperature,
        kd_cal_samples=kd_cal_samples,
        kd_global_samples=kd_global_samples,
        kd_batch_size=kd_batch_size,
        kd_forward_epochs=kd_forward_epochs,
        kd_forward_lr=kd_forward_lr,
        kd_reverse_epochs=kd_reverse_epochs,
        kd_reverse_lr=kd_reverse_lr,
    )

    # ============================================================
    # 9. 构造下发配置（train/evaluate）
    # ============================================================
    train_cfg_dict = {
        **comm_defaults,
        **fomaml_config,
        # 可选：客户端本地训练模式开关
        "client-train-mode": client_train_mode,
        "apskd-epochs": apskd_epochs,
        "apskd-lr": apskd_lr,
        # APSKD 也会用到温度
        "kd-temperature": kd_temperature,
        # 下面两项不一定会被 client 用到，但保留便于未来扩展（例如 client 侧也做 KD）
        "kd-alpha": kd_alpha,
        "kd-enable": kd_enable,
    }
    train_cfg = ConfigRecord(train_cfg_dict)

    eval_cfg_dict = {**comm_defaults, **meta_eval_config}
    eval_cfg = ConfigRecord(eval_cfg_dict)

    # server-side evaluate_fn：用闭包把 meta 参数带入
    evaluate_fn = make_global_evaluate(meta_adapt_steps=meta_adapt_steps, meta_adapt_lr=meta_adapt_lr)

    # ============================================================
    # 10. 启动联邦训练
    # ============================================================
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # ============================================================
    # 11. 保存最终模型（student + 可选 teacher）
    # ============================================================
    log(INFO, "Saving final meta-model (student) to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")

    if kd_enable:
        teacher = getattr(strategy, "teacher_model", None)
        if teacher is not None:
            try:
                log(INFO, "Saving teacher model to disk...")
                torch.save(teacher.state_dict(), "teacher_model.pt")
            except Exception as e:
                log(WARNING, "Failed to save teacher model: %s", str(e))

    # ============================================================
    # 12. 绘图
    # ============================================================
    log(INFO, "Generating training plots...")
    plot_metrics(result, filename="training_metrics.png")
