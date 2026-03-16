import torch  # 导入 PyTorch
import matplotlib.pyplot as plt  # 导入绘图库
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # 导入 Flower 的核心数据结构
from flwr.serverapp import Grid, ServerApp  # 导入 Flower 服务器端组件
from flwr.common import log
from logging import INFO

# ---- 使用 FedMeta 策略 (注意包名已改为 fedml) ----
from fedml.fedml import FedMeta 
# ---- 引入 test_meta 用于正确的元学习评估 ----
from fedml.task import Net, test_centralized_dataset, test, test_meta

app = ServerApp()

# ============================================================
#  绘图工具函数 (直接定义在此文件中)
# ============================================================
def plot_metrics(result, filename="training_metrics.png"):
    """
    解析 Result 对象并绘制 Server-side Accuracy 和 Loss 的折线图。
    """
    # 1. 提取数据
    history = result.evaluate_metrics_serverapp
    
    if not history:
        log(INFO, "No server-side evaluation metrics found to plot.")
        return

    # 按轮次排序 (Round 0, 1, 2...)
    rounds = sorted(history.keys())
    accuracies = []
    losses = []

    for r in rounds:
        metrics = history[r]
        # 提取 accuracy 和 loss，如果不存在则设为 None
        accuracies.append(metrics.get("accuracy", None))
        losses.append(metrics.get("loss", None))

    # 2. 设置绘图 (双子图)
    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- 左图: Server-side Accuracy ---
    # 过滤 None 值
    valid_acc = [(r, v) for r, v in zip(rounds, accuracies) if v is not None]
    if valid_acc:
        rs, vs = zip(*valid_acc)
        ax1.plot(rs, vs, color='blue', marker='o', label='Accuracy (Post-Adaptation)', linewidth=2)
        ax1.set_title('Server-side Meta-Test Accuracy', fontsize=14)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True)
        ax1.legend()
        # 强制 X 轴显示整数
        from matplotlib.ticker import MaxNLocator
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 右图: Server-side Loss ---
    valid_loss = [(r, v) for r, v in zip(rounds, losses) if v is not None]
    if valid_loss:
        rs, vs = zip(*valid_loss)
        ax2.plot(rs, vs, color='red', marker='o', label='Loss', linewidth=2)
        ax2.set_title('Server-side Meta-Test Loss', fontsize=14)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True)
        ax2.legend()
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. 保存图片
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    log(INFO, f"Training plots saved to: {filename}")


@app.main() 
def main(grid: Grid, context: Context) -> None: 
    """
    Flower ServerApp 主入口函数。
    """

    # ============================================================
    # 1. 读取基础训练参数
    # ============================================================
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 1.0))
    num_rounds: int = int(context.run_config.get("num-server-rounds", 5))
    lr: float = float(context.run_config.get("learning-rate", 0.01))

    # ============================================================
    # 2. 读取调度与成员管理参数
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
    # 3. 读取 Meta-Learning 相关参数 (FO-MAML)
    # ============================================================
    
    # [Train Config] FO-MAML 训练超参数
    fomaml_config = {
        "fomaml-alpha": float(context.run_config.get("fomaml-alpha", 0.01)), # Inner loop LR
        "fomaml-beta": float(context.run_config.get("fomaml-beta", 0.001)),  # Outer loop LR
        "fomaml-inner-steps": int(context.run_config.get("fomaml-inner-steps", 1)),
        "batch-size": int(context.run_config.get("batch-size", 32)),
    }

    # [Evaluate Config] Meta-Testing 微调参数
    meta_eval_config = {
        "meta-adapt-steps": int(context.run_config.get("meta-adapt-steps", 5)),
        "meta-adapt-lr": float(context.run_config.get("meta-adapt-lr", 0.01)),
        "batch-size": int(context.run_config.get("batch-size", 32)),
    }

    # ============================================================
    # 4. 读取通信链路参数
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
    # 5. 初始化全局模型
    # ============================================================
    global_model = Net() 
    arrays = ArrayRecord(global_model.state_dict())

    # ============================================================
    # 6. 构建策略 (FedMeta)
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
    )

    # ============================================================
    # 7. 构造下发配置
    # ============================================================
    train_cfg_dict = {**comm_defaults, **fomaml_config}
    train_cfg = ConfigRecord(train_cfg_dict)
    
    eval_cfg_dict = {**comm_defaults, **meta_eval_config}
    eval_cfg = ConfigRecord(eval_cfg_dict)

    # ============================================================
    # 8. 启动联邦训练
    # ============================================================
    result = strategy.start(  
        grid=grid, 
        initial_arrays=arrays,  
        train_config=train_cfg,  
        evaluate_config=eval_cfg, 
        num_rounds=num_rounds, 
        evaluate_fn=global_evaluate,  # 使用修正后的元学习评估函数
    )

    # 保存最终训练好的 Global Meta-Model
    log(INFO, "Saving final meta-model to disk...")  
    state_dict = result.arrays.to_torch_state_dict()  
    torch.save(state_dict, "final_model.pt")  

    # ============================================================
    # 9. 绘制并保存结果图
    # ============================================================
    log(INFO, "Generating training plots...")
    plot_metrics(result, filename="training_metrics.png")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:  
    """
    服务器端全局评估函数 (Centralized Evaluation)。
    
    [重要修正]: 
    使用 test_meta (Adapt -> Test) 而不是 test (Zero-shot)。
    这能真实反映元学习模型在经过少量微调后的性能。
    """
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model.to(device) 

    # 加载完整的测试集 (在 task.py 中定义)
    test_dataloader = test_centralized_dataset()
    
    # 执行元学习评估 (例如：给予 10 步微调机会)
    # adaptation_steps=10, adaptation_lr=0.01 建议与 toml 保持一致或稍大
    test_loss, test_acc = test_meta(
        model, 
        test_dataloader, 
        device, 
        adaptation_steps=10, 
        adaptation_lr=0.01
    )  

    # 返回的字典键名必须与 plot_metrics 中读取的一致
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})