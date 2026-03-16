import torch
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fedavg.fedavg import FedAvg
# 导入任务相关的模型定义和辅助函数
from fedavg.task import (
    Net, 
    BigTeacherNet, 
    load_centralized_dataset, 
    load_centralized_dataset_train_test,
    train_centralized, 
    distill_centralized, 
    test, 
    count_parameters
)

app = ServerApp()

# ============================================================
# 1. 地面蒸馏实验逻辑 (Ground Distillation)
# ============================================================
def perform_ground_distillation_experiment(device):
    """
    [地面蒸馏实验核心逻辑]
    1. 加载地面数据。
    2. 定义大小两个模型 (Teacher & Student)。
    3. 训练 Teacher 作为知识源。
    4. 将 Teacher 的知识蒸馏给 Student。
    5. 对比两者的性能与参数量。
    """
    print("\n" + "="*60)
    print("启动地面轻量化蒸馏实验 (Ground Distillation Experiment)")
    print("="*60)

    # 1. 准备数据 (加载完整的训练集和测试集)
    # 注意：在模拟环境为了速度，epochs 设置较小，实际生产可增加
    trainloader, testloader = load_centralized_dataset_train_test()
    
    # 2. 初始化模型
    teacher = BigTeacherNet() # 大模型 (Teacher)
    student = Net()           # 小模型 (Student)
    
    # 计算并对比参数量
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    compression_ratio = student_params / teacher_params
    
    print(f"模型轻量化指标对比:")
    print(f"   - Teacher (大模型) 参数量: {teacher_params:,}")
    print(f"   - Student (小模型) 参数量: {student_params:,}")
    print(f"   - 压缩率: Student 为 Teacher 的 {compression_ratio:.2%}")

    # 3. 训练教师模型 (Teacher)
    # 这里设置 epochs=2 仅做演示，实际建议 10+
    ground_epochs = 2 
    ground_lr = 0.01
    
    print(f"\n[Step 1] 正在训练教师模型 (Teacher Training)...")
    train_centralized(teacher, trainloader, epochs=ground_epochs, lr=ground_lr, device=device)
    
    # 评估教师模型性能
    t_loss, t_acc = test(teacher, testloader, device)
    print(f"Teacher 训练完成 -> Loss: {t_loss:.4f}, Accuracy: {t_acc:.4%}")

    # 4. 蒸馏学生模型 (Distillation)
    # alpha=0.5 表示 Loss 一半来自真实标签(Hard)，一半来自老师(Soft)
    # temp=2.0 软化概率分布
    distill_alpha = 0.5
    distill_temp = 2.0
    
    print(f"\n[Step 2] 正在进行知识蒸馏 (Teacher -> Student)...")
    distill_centralized(
        student, teacher, trainloader, 
        epochs=ground_epochs, lr=ground_lr, device=device,
        temp=distill_temp, alpha=distill_alpha
    )
    
    # 评估学生模型性能
    s_loss, s_acc = test(student, testloader, device)
    print(f"Student 蒸馏完成 -> Loss: {s_loss:.4f}, Accuracy: {s_acc:.4%}")

    # 5. 实验总结输出
    print("\n🏆 地面蒸馏实验总结:")
    print(f"| Model   | Params    | Accuracy | Loss   |")
    print(f"|---------|-----------|----------|--------|")
    print(f"| Teacher | {teacher_params:9,}| {t_acc:7.2%} | {t_loss:.4f} |")
    print(f"| Student | {student_params:9,}| {s_acc:7.2%} | {s_loss:.4f} |")
    print("="*60 + "\n")
    
    # 返回蒸馏后的学生模型，用于后续联邦学习
    return student


# ============================================================
# 2. 绘图辅助函数 (Plotting)
# ============================================================
def plot_metrics(history_metrics: dict, filename: str = "server_metrics.png"):
    """
    绘制服务器端评估的 Accuracy 和 Loss 曲线并保存为图片。
    
    Args:
        history_metrics: result.evaluate_metrics_serverapp 字典
                         Key是轮次(int), Value是MetricRecord
        filename: 保存的文件名
    """
    if not history_metrics:
        print("没有服务器端评估数据，跳过绘图。")
        return

    # 1. 提取数据
    # history_metrics 的格式是 {round_num: MetricRecord{'accuracy': ..., 'loss': ...}}
    rounds = sorted(history_metrics.keys())
    accuracies = []
    losses = []

    for r in rounds:
        metrics = history_metrics[r]
        # 注意：MetricRecord 类似于字典，直接取值
        accuracies.append(metrics["accuracy"])
        losses.append(metrics["loss"])

    # 2. 创建画布 (包含两个子图)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制 Accuracy
    ax1.plot(rounds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    ax1.set_title('Server-side Accuracy per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()

    # 绘制 Loss
    ax2.plot(rounds, losses, marker='o', linestyle='-', color='r', label='Loss')
    ax2.set_title('Server-side Loss per Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    # 3. 保存图片
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n绘图完成！结果已保存至: {filename}")
    plt.close()


@app.main() 
def main(grid: Grid, context: Context) -> None:
    # 检查计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # 3. 执行地面蒸馏 (Ground Phase)
    # ============================================================
    # 先在服务器端利用中心化数据，通过大模型带小模型的方式，
    # 获得一个性能较好的初始 Student 模型。
    distilled_student_model = perform_ground_distillation_experiment(device)

    # ============================================================
    # 4. 联邦学习配置读取 (FL Phase)
    # ============================================================
    # ---------- 基础训练参数 ----------  
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.5))
    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))
    lr: float = float(context.run_config.get("learning-rate", 0.1))

    # ---------- 知识蒸馏参数 (用于 FL 阶段的本地训练) ----------
    # 这些参数决定了客户端在本地训练时，是否还要继续用旧全局模型做蒸馏
    kd_alpha: float = float(context.run_config.get("kd-alpha", 0.0))
    kd_temperature: float = float(context.run_config.get("kd-temperature", 1.0))

    # ---------- 调度与节点选择参数 ---------- 
    selection_mode: str = str(context.run_config.get("selection-mode", "random"))
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", False))

    turnover_T: int = int(context.run_config.get("turnover-T", 10))
    deltaQ: int = int(context.run_config.get("deltaQ", 999999))
    deltaP: int = int(context.run_config.get("deltaP", 999999))
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))

    # ---------- 通信链路模拟参数 ----------  
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

    # ============================================================
    # 5. 初始化全局模型 (Warm Start)
    # ============================================================
    # 使用地面蒸馏后的 Student 模型作为联邦学习的起点
    global_model = distilled_student_model
    arrays = ArrayRecord(global_model.state_dict())
    
    print(f"联邦学习准备就绪，使用地面蒸馏后的模型参数作为初始状态。")

    # ============================================================
    # 6. 构建策略与启动
    # ============================================================
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

    # 下发训练配置 (包含 KD 参数)
    train_cfg = ConfigRecord({
        "lr": lr, 
        "kd-alpha": kd_alpha, 
        "kd-temperature": kd_temperature, 
        **comm_defaults
    })
    
    # 下发评估配置
    eval_cfg = ConfigRecord({**comm_defaults})

    # 启动 ServerApp
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

    # ============================================================
    # 7. 绘制结果图
    # ============================================================
    # result.evaluate_metrics_serverapp 存储了由 global_evaluate 返回的每轮数据
    plot_metrics(result.evaluate_metrics_serverapp, filename="server_metrics.png")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """
    [服务端全局评估]
    在每一轮联邦学习结束后，使用中心化测试集评估当前的全局模型。
    """
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 这里的 load_centralized_dataset 只返回测试集
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})