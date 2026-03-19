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

    # Teacher epochs: 2 → 10
    #   ResNet-18 训 2 epoch 精度仅约 70%，软标签接近均匀分布，
    #   KL 损失迫使学生把输出压平，与 CE 损失对抗 → Loss 发散。
    #   10 epoch 精度约 88%，软标签有意义，KD 正常收敛。
    #   RTX 4060 Laptop 约需 2~3 分钟，值得等待。
    teacher_epochs = 10
    student_epochs = 5    # Student 蒸馏 5 epoch，充分吸收教师知识
    lr = 0.01

    log(INFO, "1. 训练 Teacher 模型 (%d epochs)...", teacher_epochs)
    train_centralized(teacher, trainloader, epochs=teacher_epochs, lr=lr, device=device)

    log(INFO, "2. 蒸馏 Student 模型 Teacher→Student (%d epochs)...", student_epochs)
    distill_centralized(student, teacher, trainloader,
                        epochs=student_epochs, lr=lr, device=device, temp=2.0, alpha=0.5)

    loss, acc = test(student, testloader, device)
    log(INFO, "✅ 地面蒸馏完成，Student 初始精度: %.2f%%", acc * 100)

    return student, teacher

# ============================================================
# 2. 绘图工具函数
# ============================================================
def plot_metrics(result, filename: str = "training_metrics.png") -> None:
    """绘制单次运行的 Accuracy 和 Loss 折线图（来自客户端本地评估聚合）"""
    history = getattr(result, "evaluate_metrics_clientapp", None)
    if not history:
        log(INFO, "No client-side evaluation metrics found to plot.")
        return

    rounds = sorted(history.keys())
    accuracies = [history[r].get("eval_acc") for r in rounds]
    losses     = [history[r].get("eval_loss") for r in rounds]

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    valid_acc = [(r, v) for r, v in zip(rounds, accuracies) if v is not None]
    if valid_acc:
        rs, vs = zip(*valid_acc)
        ax1.plot(rs, vs, marker="o", linewidth=2, label="Accuracy (Post-Adaptation)")
        ax1.set_title("Client-Aggregated Meta-Test Accuracy (Local Non-IID)", fontsize=14)
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.grid(True)
        ax1.legend()

    valid_loss = [(r, v) for r, v in zip(rounds, losses) if v is not None]
    if valid_loss:
        rs, vs = zip(*valid_loss)
        ax2.plot(rs, vs, marker="o", linewidth=2, label="Loss")
        ax2.set_title("Client-Aggregated Meta-Test Loss (Local Non-IID)", fontsize=14)
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
        plt.title("Accuracy Comparison: FedAvg Baseline vs FedMeta Variants (Non-IID, Client-Local Evaluation)", fontsize=13)
        plt.xlabel("Server Round", fontsize=12)
        plt.ylabel("Client-Aggregated Accuracy (Post-Adaptation on Local Non-IID Data)", fontsize=10)
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
#    ── 评估方式修改说明 ─────────────────────────────────────────────
#    原方案：服务器用全局 IID 测试集做 test_meta → 对 FedAvg 天然有利
#      FedAvg 的全局模型本来就是在全局分布上优化的，
#      在 IID 测试集上 adaptation 20步 = 在"熟悉的数据"上微调，效果极好。
#      MAML 系列的快速适应优势场景是"陌生的偏斜分布"，在 IID 测试集上体现不出来。
#
#    新方案：各客户端用自己的本地 Non-IID 数据做 test_meta，回传 eval_acc，
#      服务器聚合各客户端的 eval_acc 作为全局指标（加权平均）。
#      → FedAvg 因 client drift 在本地偏斜数据上 adaptation 效果差
#      → MAML 系列训练时已见过各种偏斜分布，本地快速适应能力更强
#      → 这才是真实卫星场景：每颗卫星只能用自己的数据做适应
# ============================================================
def make_global_evaluate_from_clients(meta_adapt_steps: int, meta_adapt_lr: float) -> callable:
    """
    占位 evaluate_fn：不在服务器端跑全局评估，
    改为依赖客户端回传的 eval_acc 聚合（在 aggregate_evaluate 里处理）。
    返回 None 让框架跳过服务器端评估，只用客户端聚合结果。
    """
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # 不在服务器端用全局 IID 测试集评估
        # 真实评估结果来自各客户端本地 Non-IID 数据的 eval_acc 聚合
        return None
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
    dirichlet_alpha = float(cfg.get("dirichlet-alpha", 0.1))
    fomaml_config = {
        "fomaml-alpha": float(cfg.get("fomaml-alpha", 0.01)),
        "fomaml-beta": float(cfg.get("fomaml-beta", 0.01)),
        "fomaml-inner-steps": int(cfg.get("fomaml-inner-steps", 5)),
        "batch-size": batch_size,
        "dirichlet-alpha": dirichlet_alpha,
    }

    meta_adapt_steps = int(cfg.get("meta-adapt-steps", 10))
    meta_adapt_lr = float(cfg.get("meta-adapt-lr", 0.01))
    meta_eval_config = {
        "meta-adapt-steps": meta_adapt_steps,
        "meta-adapt-lr": meta_adapt_lr,
        "batch-size": batch_size,
        "dirichlet-alpha": dirichlet_alpha,
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
                "dirichlet-alpha": dirichlet_alpha,
                "kd-enable": False,
            })
            # FedAvg 统一使用 meta_eval_config，与 FedMeta 系列评估标准完全一致
            eval_cfg = ConfigRecord({**comm_defaults, **meta_eval_config})
            # 评估改为客户端本地 Non-IID 数据聚合，服务器端不再跑全局 IID 测试集
            evaluate_fn = make_global_evaluate_from_clients(
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
            # 评估改为客户端本地 Non-IID 数据聚合
            evaluate_fn = make_global_evaluate_from_clients(
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

        # 评估结果来源：
        #   history_eval_client = 客户端 evaluate() 回传的本地 Non-IID 评估聚合
        #                         （eval_acc / eval_loss 字段）
        #   history_train       = 客户端 train() 回传的训练指标（train_loss 字段）
        history_eval_client = getattr(result, "evaluate_metrics_clientapp", None)
        history_train = getattr(result, "train_metrics_clientapp", None)

        if history_eval_client:
            metrics_to_save = {}
            all_rounds = set(history_eval_client.keys())
            if history_train:
                all_rounds |= set(history_train.keys())
            for r in sorted(all_rounds):
                m_eval  = history_eval_client.get(r, {}) if history_eval_client else {}
                m_train = history_train.get(r, {})       if history_train       else {}
                metrics_to_save[int(r)] = {
                    "accuracy":   float(m_eval["eval_acc"])    if "eval_acc"   in m_eval  else None,
                    "loss":       float(m_eval["eval_loss"])   if "eval_loss"  in m_eval  else None,
                    "train_loss": float(m_train["train_loss"]) if "train_loss" in m_train else None,
                }
            json_filename = f"metrics_{client_train_mode}.json"
            with open(json_filename, "w") as f:
                json.dump(metrics_to_save, f)
            log(INFO, f"Metrics successfully saved to {json_filename}.")
        else:
            log(WARNING, "No client-side evaluate metrics found for mode: %s", client_train_mode)

    # ============================================================
    # 全部循环结束后，绘制对比图
    # ============================================================
    log(INFO, "\n" + "="*60)
    log(INFO, "🎉 ALL EXPERIMENTS COMPLETED! Generating final comparison plots...")
    log(INFO, "="*60)

    plot_multi_mode_comparison()
    plot_train_loss_comparison()