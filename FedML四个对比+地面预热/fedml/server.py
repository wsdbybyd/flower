import os
import json
import copy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from logging import INFO, WARNING

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from fedml.fedml import FedMeta
from fedml.task import (
    Net,
    BigTeacherNet,
    test_centralized_dataset,
    test_meta,
    load_centralized_dataset_train_test,
    train_centralized,
    distill_centralized,
    test
)

app = ServerApp()

# ============================================================
# 1. 地面预蒸馏模块 (Ground Pre-training / Warm Start)
# ============================================================
def perform_ground_distillation_experiment(device):
    log(INFO, "\n" + "="*60)
    log(INFO, "🚀 [预处理] 启动地面蒸馏 (Ground Warm-up) for FedMeta")
    log(INFO, "="*60)

    trainloader, testloader = load_centralized_dataset_train_test()
    teacher = BigTeacherNet()
    student = Net()

    epochs = 2
    lr = 0.01

    log(INFO, "1. 训练 Teacher 模型...")
    train_centralized(teacher, trainloader, epochs=epochs, lr=lr, device=device)

    log(INFO, "2. 蒸馏 Student 模型 (Teacher -> Student)...")
    distill_centralized(student, teacher, trainloader, epochs=epochs, lr=lr, device=device, temp=2.0, alpha=0.5)

    loss, acc = test(student, testloader, device)
    log(INFO, f"✅ 地面蒸馏完成，Student 初始 Baseline Accuracy: {acc:.2%}")

    return student, teacher

# ============================================================
# 2. 绘图工具函数
# ============================================================
def plot_metrics(result, filename: str = "training_metrics.png") -> None:
    """绘制单次运行的 Accuracy 和 Loss 折线图"""
    history = getattr(result, "evaluate_metrics_serverapp", None)
    if not history:
        log(INFO, "No server-side evaluation metrics found to plot.")
        return

    rounds = sorted(history.keys())
    accuracies = [history[r].get("accuracy") for r in rounds]
    losses = [history[r].get("loss") for r in rounds]

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    valid_acc = [(r, v) for r, v in zip(rounds, accuracies) if v is not None]
    if valid_acc:
        rs, vs = zip(*valid_acc)
        ax1.plot(rs, vs, marker="o", linewidth=2, label="Accuracy (Post-Adaptation)")
        ax1.set_title("Server-side Meta-Test Accuracy", fontsize=14)
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.grid(True)
        ax1.legend()

    valid_loss = [(r, v) for r, v in zip(rounds, losses) if v is not None]
    if valid_loss:
        rs, vs = zip(*valid_loss)
        ax2.plot(rs, vs, marker="o", linewidth=2, label="Loss")
        ax2.set_title("Server-side Meta-Test Loss", fontsize=14)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    log(INFO, f"Training plots saved to: {filename}")


def plot_multi_mode_comparison():
    """扫描目录下的 JSON，绘制 4 种模式的 Test Accuracy 对比图（含 FedAvg baseline）"""
    MODES  = ["fedavg",             "fomaml",       "apskd",        "fomaml+apskd"]
    COLORS = ["#d62728",            "#1f77b4",       "#ff7f0e",      "#2ca02c"]
    STYLES = ["--",                 "-",             "-",            "-"]
    LABELS = ["FedAvg (Baseline)",  "Pure FO-MAML", "Pure APSKD",   "FO-MAML + APSKD (Mixed)"]

    plt.figure(figsize=(10, 6))
    plt.style.use('default')

    success_count = 0
    for mode, color, style, label in zip(MODES, COLORS, STYLES, LABELS):
        json_file = f"metrics_{mode}.json"
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        rounds = sorted([int(k) for k in data.keys()])
        accuracies = [data[str(r)]["accuracy"] for r in rounds if data[str(r)].get("accuracy") is not None]
        valid_rounds = [r for r in rounds if data[str(r)].get("accuracy") is not None]
        if accuracies:
            plt.plot(valid_rounds, accuracies, label=label, color=color,
                     linestyle=style, linewidth=2, marker='o', markersize=4)
            success_count += 1

    if success_count > 0:
        plt.title("Accuracy Comparison: FedAvg Baseline vs FedMeta Variants (Non-IID, With Warm Start)", fontsize=13)
        plt.xlabel("Server Round", fontsize=12)
        plt.ylabel("Server-side Meta-Test Accuracy (Post-Adaptation)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        save_path = "comparison_result_accuracy.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        log(INFO, f"Comparison plot saved to: {save_path}")


def plot_train_loss_comparison():
    """扫描目录下的 JSON，绘制 4 种模式的 Train Loss 对比图（含 FedAvg baseline）"""
    MODES  = ["fedavg",             "fomaml",       "apskd",        "fomaml+apskd"]
    COLORS = ["#d62728",            "#1f77b4",       "#ff7f0e",      "#2ca02c"]
    STYLES = ["--",                 "-",             "-",            "-"]
    LABELS = ["FedAvg (Baseline)",  "Pure FO-MAML", "Pure APSKD",   "FO-MAML + APSKD (Mixed)"]

    plt.figure(figsize=(10, 6))
    plt.style.use('default')

    success_count = 0
    for mode, color, style, label in zip(MODES, COLORS, STYLES, LABELS):
        json_file = f"metrics_{mode}.json"
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        rounds = sorted([int(k) for k in data.keys()])
        train_losses = [data[str(r)].get("train_loss") for r in rounds if data[str(r)].get("train_loss") is not None]
        valid_rounds = [r for r in rounds if data[str(r)].get("train_loss") is not None]
        if train_losses:
            plt.plot(valid_rounds, train_losses, label=label, color=color,
                     linestyle=style, linewidth=2, marker='o', markersize=4)
            success_count += 1

    if success_count > 0:
        plt.title("Client-side Aggregated Train Loss Comparison (With Warm Start)", fontsize=13)
        plt.xlabel("Server Round", fontsize=12)
        plt.ylabel("Train Loss (Aggregated)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        save_path = "comparison_result_train_loss.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        log(INFO, f"Train Loss comparison plot saved to: {save_path}")


# ============================================================
# 3. 构造评估闭包
#    所有模式统一使用 test_meta（post-adaptation）评估，确保公平对比：
#      - FedMeta 系列本就以快速适应为训练目标，post-adaptation 评估合理
#      - FedAvg baseline 同样用 post-adaptation 评估，
#        暴露其"不可快速适应"的结构性弱点，这正是与 FedMeta 对比的核心价值
#    评估标准统一后，图中 FedAvg 的相对劣势才能真实反映 Meta-Learning 的增益
# ============================================================
def make_global_evaluate(meta_adapt_steps: int, meta_adapt_lr: float) -> callable:
    """所有模式统一：执行 meta-adaptation 后再评估（公平对比）"""
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_dataloader = test_centralized_dataset()
        test_loss, test_acc = test_meta(
            model, test_dataloader, device,
            adaptation_steps=int(meta_adapt_steps),
            adaptation_lr=float(meta_adapt_lr),
        )
        return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})
    return global_evaluate


# ============================================================
# 4. 主程序入口
# ============================================================
@app.main()
def main(grid: Grid, context: Context) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # 执行地面预热 (获取热启动的初始参数)
    # ============================================================
    distilled_student, distilled_teacher = perform_ground_distillation_experiment(device)

    # 读取全局配置
    cfg = context.run_config
    fraction_evaluate = float(cfg.get("fraction-evaluate", 1.0))
    num_rounds = int(cfg.get("num-server-rounds", 10))
    selection_mode = str(cfg.get("selection-mode", "window"))
    ks_train = cfg.get("ks-train", None)
    ks_eval = cfg.get("ks-eval", None)
    enforce_comm = bool(cfg.get("enforce-comm", True))

    turnover_T = int(cfg.get("turnover-T", 5))
    deltaQ = int(cfg.get("deltaQ", 2))
    deltaP = int(cfg.get("deltaP", 3))
    probe_blocked_k = int(cfg.get("probe-blocked-k", 1))

    batch_size = int(cfg.get("batch-size", 32))
    fomaml_config = {
        "fomaml-alpha": float(cfg.get("fomaml-alpha", 0.01)),
        "fomaml-beta": float(cfg.get("fomaml-beta", 0.01)),
        "fomaml-inner-steps": int(cfg.get("fomaml-inner-steps", 5)),
        "batch-size": batch_size,
    }

    meta_adapt_steps = int(cfg.get("meta-adapt-steps", 10))
    meta_adapt_lr = float(cfg.get("meta-adapt-lr", 0.01))
    meta_eval_config = {
        "meta-adapt-steps": meta_adapt_steps,
        "meta-adapt-lr": meta_adapt_lr,
        "batch-size": batch_size,
    }

    kd_enable = bool(cfg.get("kd-enable", True))
    kd_alpha = float(cfg.get("kd-alpha", 0.5))
    kd_temperature = float(cfg.get("kd-temperature", 2.0))
    kd_cal_samples = int(cfg.get("kd-cal-samples", 2048))
    kd_global_samples = int(cfg.get("kd-global-samples", 4096))
    kd_batch_size = int(cfg.get("kd-batch-size", 64))
    kd_forward_epochs = int(cfg.get("kd-forward-epochs", 1))
    kd_forward_lr = float(cfg.get("kd-forward-lr", 0.05))
    kd_reverse_epochs = int(cfg.get("kd-reverse-epochs", 1))
    kd_reverse_lr = float(cfg.get("kd-reverse-lr", 0.01))

    apskd_epochs = int(cfg.get("apskd-epochs", 5))
    apskd_lr = float(cfg.get("apskd-lr", 0.01))
    kd_warmup_rounds = int(cfg.get("kd-warmup-rounds", 2))

    comm_defaults = {
        "sigma2": float(cfg.get("sigma2", 1e-9)),
        "default-distance-km": float(cfg.get("default-distance-km", 780.0)),
        "default-tau-d-s": float(cfg.get("default-tau-d-s", 50000.0)),
        "w-u-hz": float(cfg.get("w-u-hz", 4e6)),
        "w-d-hz": float(cfg.get("w-d-hz", 4e6)),
        "p-u-w": float(cfg.get("p-u-w", 100.0)),
        "p-d-w": float(cfg.get("p-d-w", 100.0)),
        "f-c-hz": float(cfg.get("f-c-hz", 20e9)),
        "A-T": float(cfg.get("A-T", 60.0)),
        "A-R": float(cfg.get("A-R", 30.0)),
        "G-H": float(cfg.get("G-H", 0.8)),
        "delta": float(cfg.get("delta", 2.0)),
        "psi-db-per-km": float(cfg.get("psi-db-per-km", 0.5)),
        "zeta-km": float(cfg.get("zeta-km", 500.0)),
    }

    # ============================================================
    # 依次执行 4 个模式（fedavg 作为 baseline 第一个跑）
    # ============================================================
    modes_to_run = ["fedavg", "fomaml", "apskd", "fomaml+apskd"]

    log(INFO, "="*60)
    log(INFO, "🚀 Starting automated sequential experiments for modes: %s", modes_to_run)
    log(INFO, "="*60)

    for client_train_mode in modes_to_run:
        log(INFO, "\n" + "*"*60)
        if client_train_mode == "fedavg":
            log(INFO, "⚪ NOW RUNNING MODE: FEDAVG (BASELINE, WITH WARM START)")
        else:
            log(INFO, "🔵 NOW RUNNING MODE: %s (WITH WARM START)", client_train_mode.upper())
        log(INFO, "*"*60)

        # 所有模式均使用地面蒸馏后的 student 作为热启动初始参数
        global_model = copy.deepcopy(distilled_student)
        arrays = ArrayRecord(global_model.state_dict())

        # ---- 按模式分别构建 strategy / train_cfg / evaluate_fn ----
        if client_train_mode == "fedavg":
            # FedAvg baseline：复用 FedMeta 策略，关闭双向 KD
            strategy = FedMeta(
                fraction_evaluate=fraction_evaluate,
                selection_mode=selection_mode,
                ks_train=int(ks_train) if ks_train is not None else None,
                ks_evaluate=int(ks_eval) if ks_eval is not None else None,
                enforce_comm=enforce_comm,
                turnover_T=turnover_T, deltaQ=deltaQ, deltaP=deltaP,
                probe_blocked_k=probe_blocked_k,
                kd_enable=False,        # FedAvg 不使用双向 KD
                initial_teacher_model=None,
            )
            train_cfg = ConfigRecord({
                **comm_defaults,
                "client-train-mode": "fedavg",
                "learning-rate":           float(cfg.get("learning-rate",           0.01)),
                "fedavg-local-epochs":     int(  cfg.get("fedavg-local-epochs",     1)),
                "fedavg-weight-decay":     float(cfg.get("fedavg-weight-decay",     1e-4)),
                "fedavg-label-smoothing":  float(cfg.get("fedavg-label-smoothing",  0.1)),
                "batch-size": batch_size,
                "kd-enable": False,
            })
            # FedAvg 统一使用 meta_eval_config，与 FedMeta 系列评估标准完全一致
            eval_cfg = ConfigRecord({**comm_defaults, **meta_eval_config})
            # 统一用 test_meta 评估：暴露 FedAvg 不可快速适应的结构性弱点
            evaluate_fn = make_global_evaluate(
                meta_adapt_steps=meta_adapt_steps, meta_adapt_lr=meta_adapt_lr
            )

        else:
            # FedMeta 系列三种模式：保持原有完整配置不变
            strategy = FedMeta(
                fraction_evaluate=fraction_evaluate,
                selection_mode=selection_mode,
                ks_train=int(ks_train) if ks_train is not None else None,
                ks_evaluate=int(ks_eval) if ks_eval is not None else None,
                enforce_comm=enforce_comm,
                turnover_T=turnover_T, deltaQ=deltaQ, deltaP=deltaP,
                probe_blocked_k=probe_blocked_k,
                kd_enable=kd_enable, kd_alpha=kd_alpha, kd_temperature=kd_temperature,
                kd_cal_samples=kd_cal_samples, kd_global_samples=kd_global_samples,
                kd_batch_size=kd_batch_size,
                kd_forward_epochs=kd_forward_epochs, kd_forward_lr=kd_forward_lr,
                kd_reverse_epochs=kd_reverse_epochs, kd_reverse_lr=kd_reverse_lr,
                initial_teacher_model=copy.deepcopy(distilled_teacher),
            )
            train_cfg = ConfigRecord({
                **comm_defaults,
                **fomaml_config,
                "client-train-mode": client_train_mode,
                "apskd-epochs": apskd_epochs,
                "apskd-lr": apskd_lr,
                "kd-temperature": kd_temperature,
                "kd-alpha": kd_alpha,
                "kd-enable": kd_enable,
                "kd-warmup-rounds": kd_warmup_rounds,
            })
            eval_cfg = ConfigRecord({**comm_defaults, **meta_eval_config})
            # FedMeta 系列统一用 meta-adaptation 评估
            evaluate_fn = make_global_evaluate(
                meta_adapt_steps=meta_adapt_steps, meta_adapt_lr=meta_adapt_lr
            )

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_cfg,
            evaluate_config=eval_cfg,
            num_rounds=num_rounds,
            evaluate_fn=evaluate_fn,
        )

        torch.save(result.arrays.to_torch_state_dict(), f"final_model_{client_train_mode}.pt")

        if client_train_mode != "fedavg" and kd_enable:
            teacher = getattr(strategy, "teacher_model", None)
            if teacher is not None:
                try:
                    torch.save(teacher.state_dict(), f"teacher_model_{client_train_mode}.pt")
                except Exception as e:
                    log(WARNING, "Failed to save teacher model: %s", str(e))

        plot_filename = f"training_metrics_{client_train_mode}.png"
        plot_metrics(result, filename=plot_filename)

        history_eval = getattr(result, "evaluate_metrics_serverapp", None)
        history_train = getattr(result, "train_metrics_clientapp", None)

        if history_eval:
            metrics_to_save = {}
            for r in history_eval:
                m_eval = history_eval[r]
                m_train = history_train.get(r, {}) if history_train else {}
                metrics_to_save[int(r)] = {
                    "accuracy":   float(m_eval["accuracy"])     if "accuracy"   in m_eval  else None,
                    "loss":       float(m_eval["loss"])         if "loss"       in m_eval  else None,
                    "train_loss": float(m_train["train_loss"])  if "train_loss" in m_train else None,
                }
            json_filename = f"metrics_{client_train_mode}.json"
            with open(json_filename, "w") as f:
                json.dump(metrics_to_save, f)
            log(INFO, f"Metrics successfully saved to {json_filename}.")

    # ============================================================
    # 全部循环结束后，绘制对比图
    # ============================================================
    log(INFO, "\n" + "="*60)
    log(INFO, "🎉 ALL EXPERIMENTS COMPLETED! Generating final comparison plots...")
    log(INFO, "="*60)

    plot_multi_mode_comparison()
    plot_train_loss_comparison()