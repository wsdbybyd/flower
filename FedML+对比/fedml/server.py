import os
import json
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


def plot_multi_mode_comparison():
    """
    扫描当前目录下的 JSON 记录文件，如果存在多个模式的数据，
    则自动绘制在同一张图表上进行对比验证。
    """
    MODES = ["fomaml", "apskd", "fomaml+apskd"]
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    LABELS = ["Pure FO-MAML", "Pure APSKD", "FO-MAML + APSKD (Mixed)"]
    
    plt.figure(figsize=(10, 6))
    plt.style.use('default')

    success_count = 0
    for mode, color, label in zip(MODES, COLORS, LABELS):
        json_file = f"metrics_{mode}.json"
        if not os.path.exists(json_file):
            continue
            
        with open(json_file, "r") as f:
            data = json.load(f)
            
        rounds = sorted([int(k) for k in data.keys()])
        accuracies = [data[str(r)]["accuracy"] for r in rounds if data[str(r)].get("accuracy") is not None]
        valid_rounds = [r for r in rounds if data[str(r)].get("accuracy") is not None]
        
        if accuracies:
            plt.plot(valid_rounds, accuracies, label=label, color=color, linewidth=2, marker='o', markersize=4)
            success_count += 1

    if success_count > 0:
        plt.title("Meta-Test Accuracy Comparison across Client Modes", fontsize=14)
        plt.xlabel("Server Round", fontsize=12)
        plt.ylabel("Server-side Meta-Test Accuracy", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)
        
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        save_path = "comparison_result_accuracy.png"
        plt.savefig(save_path, dpi=300)
        log(INFO, f"Multi-mode comparison plot automatically updated and saved to: {save_path}")


# ============================================================
#  构造 server-side centralized meta-evaluate 闭包
# ============================================================
def make_global_evaluate(meta_adapt_steps: int, meta_adapt_lr: float) -> callable:
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_dataloader = test_centralized_dataset()

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
    # ============================================================
    # 1. 统一读取和解析基础运行配置 (不随模式改变)
    # ============================================================
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 1.0))
    num_rounds: int = int(context.run_config.get("num-server-rounds", 5))

    selection_mode: str = str(context.run_config.get("selection-mode", "window"))
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", True))

    turnover_T: int = int(context.run_config.get("turnover-T", 5))
    deltaQ: int = int(context.run_config.get("deltaQ", 2))
    deltaP: int = int(context.run_config.get("deltaP", 3))
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))

    batch_size: int = int(context.run_config.get("batch-size", 32))
    fomaml_config = {
        "fomaml-alpha": float(context.run_config.get("fomaml-alpha", 0.01)),
        "fomaml-beta": float(context.run_config.get("fomaml-beta", 0.001)),
        "fomaml-inner-steps": int(context.run_config.get("fomaml-inner-steps", 1)),
        "batch-size": batch_size,
    }

    meta_adapt_steps: int = int(context.run_config.get("meta-adapt-steps", 10))
    meta_adapt_lr: float = float(context.run_config.get("meta-adapt-lr", 0.01))
    meta_eval_config = {
        "meta-adapt-steps": meta_adapt_steps,
        "meta-adapt-lr": meta_adapt_lr,
        "batch-size": batch_size,
    }

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

    apskd_epochs: int = int(context.run_config.get("apskd-epochs", 1))
    apskd_lr: float = float(context.run_config.get("apskd-lr", 0.01))

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
    # 2. 依次按三种模式执行完整的联邦训练流程
    # ============================================================
    modes_to_run = ["fomaml", "apskd", "fomaml+apskd"]
    
    log(INFO, "="*60)
    log(INFO, "🚀 Starting automated sequential experiments for modes: %s", modes_to_run)
    log(INFO, "="*60)

    for client_train_mode in modes_to_run:
        log(INFO, "\n" + "*"*60)
        log(INFO, "🔵 NOW RUNNING MODE: %s", client_train_mode.upper())
        log(INFO, "*"*60)

        # 每次都重新初始化全局学生模型，保证实验从头开始对比
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # 每次都重构策略实例（包含清空节点连接历史和教师模型）
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

        # 动态将当前遍历到的模式注入给客户端的训练配置
        train_cfg_dict = {
            **comm_defaults,
            **fomaml_config,
            "client-train-mode": client_train_mode,
            "apskd-epochs": apskd_epochs,
            "apskd-lr": apskd_lr,
            "kd-temperature": kd_temperature,
            "kd-alpha": kd_alpha,
            "kd-enable": kd_enable,
        }
        train_cfg = ConfigRecord(train_cfg_dict)

        eval_cfg_dict = {**comm_defaults, **meta_eval_config}
        eval_cfg = ConfigRecord(eval_cfg_dict)

        evaluate_fn = make_global_evaluate(meta_adapt_steps=meta_adapt_steps, meta_adapt_lr=meta_adapt_lr)

        # 启动当前模式的联邦训练
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_cfg,
            evaluate_config=eval_cfg,
            num_rounds=num_rounds,
            evaluate_fn=evaluate_fn,
        )

        # 保存本模式的最终模型权重
        torch.save(result.arrays.to_torch_state_dict(), f"final_model_{client_train_mode}.pt")

        if kd_enable:
            teacher = getattr(strategy, "teacher_model", None)
            if teacher is not None:
                try:
                    torch.save(teacher.state_dict(), f"teacher_model_{client_train_mode}.pt")
                except Exception as e:
                    log(WARNING, "Failed to save teacher model: %s", str(e))

        # 绘制本模式的单线图并保存 JSON 数据
        plot_filename = f"training_metrics_{client_train_mode}.png"
        plot_metrics(result, filename=plot_filename)
        
        history = getattr(result, "evaluate_metrics_serverapp", None)
        if history:
            metrics_to_save = {}
            for r in history:
                metrics_to_save[r] = {
                    "accuracy": history[r].get("accuracy"), 
                    "loss": history[r].get("loss")
                }
            json_filename = f"metrics_{client_train_mode}.json"
            with open(json_filename, "w") as f:
                json.dump(metrics_to_save, f)
            log(INFO, f"Metrics successfully saved to {json_filename}.")

    # ============================================================
    # 3. 全部循环结束后，自动绘制并生成对比图表
    # ============================================================
    log(INFO, "\n" + "="*60)
    log(INFO, "🎉 ALL EXPERIMENTS COMPLETED! Generating final comparison plot...")
    log(INFO, "="*60)
    plot_multi_mode_comparison()