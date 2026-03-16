import torch
import copy
import matplotlib
# 设置后端为 Agg，防止在无 GUI 环境下报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

# 引入我们自定义的策略和任务
from fedavg.fedavg import FedAvg 
from fedavg.task import Net, BigTeacherNet, load_centralized_dataset, load_centralized_dataset_train_test, train_centralized, distill_centralized, test, count_parameters

app = ServerApp()

# ============================================================
# 1. 地面预蒸馏模块 (Ground Pre-training / Warm Start)
# ============================================================
def perform_ground_distillation_experiment(device):
    """
    在 FL 开始前，利用服务器端数据预训练一个 Student 模型。
    """
    print("\n" + "="*60)
    print("🚀 [预处理] 启动地面蒸馏 (Ground Distillation)")
    print("="*60)

    # 加载地面数据
    trainloader, testloader = load_centralized_dataset_train_test()
    
    teacher = BigTeacherNet() # 大模型
    student = Net()           # 小模型
    
    # 训练参数 (演示用 epoch=2)
    epochs = 2
    lr = 0.01

    print(f"1. 训练 Teacher 模型...")
    train_centralized(teacher, trainloader, epochs=epochs, lr=lr, device=device)
    
    print(f"2. 蒸馏 Student 模型 (Teacher -> Student)...")
    distill_centralized(student, teacher, trainloader, epochs=epochs, lr=lr, device=device, temp=2.0, alpha=0.5)
    
    loss, acc = test(student, testloader, device)
    print(f"✅ 地面蒸馏完成，Student 初始 Accuracy: {acc:.2%}")
    
    return student

# ============================================================
# 2. 绘图函数集
# ============================================================
def plot_comparison(results_dict: dict, filename: str = "comparison_three_way.png"):
    """
    绘制三条曲线的对比图 (Baseline vs Ground vs Dual)
    """
    if not results_dict:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义样式
    styles = [
        {'color': 'gray',   'linestyle': '--', 'linewidth': 1.5, 'marker': 'x'}, # Baseline
        {'color': 'blue',   'linestyle': '-.', 'linewidth': 1.5, 'marker': '^'}, # Only Ground
        {'color': 'red',    'linestyle': '-',  'linewidth': 2.5, 'marker': 'o'}  # Dual Distill
    ]

    keys = list(results_dict.keys())
    
    for idx, label_name in enumerate(keys):
        history_metrics = results_dict[label_name]
        if not history_metrics:
            continue
            
        rounds = sorted(history_metrics.keys())
        accuracies = [history_metrics[r]["accuracy"] for r in rounds]
        losses = [history_metrics[r]["loss"] for r in rounds]
        
        style = styles[idx % len(styles)]

        ax1.plot(rounds, accuracies, label=label_name, **style)
        ax2.plot(rounds, losses, label=label_name, **style)

    ax1.set_title('Accuracy Comparison (3 Strategies)')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.set_title('Loss Comparison (3 Strategies)')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n📊 三方对比图已保存至: {filename}")
    plt.close()

def plot_student_mentor_style(
    student_metrics: dict, 
    teacher_metrics: dict, 
    filename: str = "student_mentor_style.png"
):
    """
    [新增] 复现 IEEE 风格的 Student vs Mentor 对比图
    """
    if not student_metrics or not teacher_metrics:
        print("⚠️ 数据不足，无法绘制 Student-Mentor 对比图")
        return

    # 提取数据
    rounds = sorted(student_metrics.keys())
    # 确保 Teacher 数据也对齐 (有些轮次可能还没跑完)
    valid_rounds = [r for r in rounds if r in teacher_metrics]
    
    if not valid_rounds:
        print("⚠️ Student 和 Mentor 数据轮次未对齐，跳过绘图")
        return

    st_acc = [student_metrics[r]["accuracy"] * 100 for r in valid_rounds] # 转为百分比
    te_acc = [teacher_metrics[r]["accuracy"] * 100 for r in valid_rounds]
    
    # 设置风格
    plt.figure(figsize=(6, 5)) 
    
    # 绘制 Student (红色三角形)
    plt.plot(valid_rounds, st_acc, 
             label='student', 
             color='#d62728',    # 砖红色
             marker='^',         # 三角形标记
             markersize=5,       
             linewidth=1.0,      
             linestyle='-',      
             alpha=0.8)

    # 绘制 Mentor (蓝色圆形)
    plt.plot(valid_rounds, te_acc, 
             label='mentor', 
             color='#1f77b4',    # 经典的蓝
             marker='o',         # 圆形标记
             markersize=5, 
             linewidth=1.0,
             linestyle='-',
             alpha=0.8)

    plt.xlabel('Index of iteration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # 设置网格
    plt.grid(True, axis='y', linestyle='-', alpha=0.5) 
    plt.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    
    # 自动调整 Y 轴范围，稍微留点余量
    min_val = min(min(st_acc), min(te_acc))
    plt.ylim(max(0, min_val - 5), 100)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300) 
    print(f"\n📊 Student-Mentor 对比图已保存至: {filename}")
    plt.close()

# ============================================================
# 3. 单次实验运行器
# ============================================================
def run_strategy_session(
    grid: Grid, 
    initial_model: Net, 
    context: Context, 
    session_name: str,
    enable_dual_distill: bool, 
    apskd_alpha: float         
):
    print(f"\n" + ">"*20 + f" 开始运行策略: {session_name} " + "<"*20)
    
    # 深拷贝初始模型，确保实验间不互相污染
    current_model = copy.deepcopy(initial_model)
    arrays = ArrayRecord(current_model.state_dict())

    # 读取参数
    run_cfg = context.run_config
    
    # 初始化策略
    strategy = FedAvg(
        fraction_evaluate=float(run_cfg.get("fraction-evaluate", 1.0)),
        selection_mode=str(run_cfg.get("selection-mode", "window")),
        ks_train=int(run_cfg.get("ks-train")) if run_cfg.get("ks-train") else None,
        ks_evaluate=int(run_cfg.get("ks-eval")) if run_cfg.get("ks-eval") else None,
        enforce_comm=bool(run_cfg.get("enforce-comm", True)),
        turnover_T=int(run_cfg.get("turnover-T", 5)),
        deltaQ=int(run_cfg.get("deltaQ", 2)),
        deltaP=int(run_cfg.get("deltaP", 3)),
        probe_blocked_k=int(run_cfg.get("probe-blocked-k", 1)),
        
        # --- 核心开关: 双向蒸馏 (Teacher<->Student) ---
        enable_dual_distillation=enable_dual_distill 
    )
    
    strategy.server_teacher_lr = float(run_cfg.get("server-teacher-lr", 0.001))

    # 构造配置
    comm_defaults = {  
        "sigma2": float(run_cfg.get("sigma2", 1e-9)),  
        "default-distance-km": float(run_cfg.get("default-distance-km", 780.0)),
        "default-tau-d-s": float(run_cfg.get("default-tau-d-s", 50000.0)),
        "w-u-hz": float(run_cfg.get("w-u-hz", 4e6)),       
        "w-d-hz": float(run_cfg.get("w-d-hz", 4e6)),       
        "p-u-w": float(run_cfg.get("p-u-w", 100.0)),       
        "p-d-w": float(run_cfg.get("p-d-w", 100.0)),       
        "f-c-hz": float(run_cfg.get("f-c-hz", 20e9)),      
    }
    
    distill_config = {
        "apskd-a-T": apskd_alpha, 
        "gs-distill-epochs": int(run_cfg.get("gs-distill-epochs", 1)),
        "gs-distill-alpha": float(run_cfg.get("gs-distill-alpha", 0.5)),
        "gs-distill-temp": float(run_cfg.get("gs-distill-temp", 3.0)),
        "lr": float(run_cfg.get("learning-rate", 0.01)),
    }

    train_cfg = ConfigRecord({**comm_defaults, **distill_config})
    eval_cfg = ConfigRecord({**comm_defaults})

    # 启动
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=int(run_cfg.get("num-server-rounds", 5)),
        evaluate_fn=global_evaluate,
    )
    
    # [修改] 返回 (Result, Strategy) 元组
    # 这样我们才能在外面访问 strategy 内部记录的 Teacher 数据
    return result, strategy

# ============================================================
# 主入口 (Experiment Orchestrator)
# ============================================================
@app.main() 
def main(grid: Grid, context: Context) -> None: 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 步骤 1: 准备两种初始模型 (Random vs Distilled)
    # ---------------------------------------------------------
    print("\n🛠️ [Init] 准备随机模型 (for Baseline)...")
    random_model = Net()

    print("\n🛠️ [Init] 准备地面蒸馏模型 (for Warm Start)...")
    distilled_model = perform_ground_distillation_experiment(device)

    # ---------------------------------------------------------
    # 步骤 2: 依次运行三个实验
    # ---------------------------------------------------------
    
    # 实验 A: Baseline
    res_baseline, _ = run_strategy_session(
        grid=grid, context=context,
        session_name="1. Baseline",
        initial_model=random_model,
        enable_dual_distill=False,
        apskd_alpha=0.0
    )

    # 实验 B: Only Ground Distillation
    res_only_ground, _ = run_strategy_session(
        grid=grid, context=context,
        session_name="2. Ground Distillation",
        initial_model=distilled_model,  
        enable_dual_distill=False,     
        apskd_alpha=0.0                 
    )

    # 实验 C: Dual Distillation (完整方案)
    target_apskd = float(context.run_config.get("apskd-a-T", 0.5))
    
    # [关键] 这里我们要捕获返回的 strategy 对象
    res_dual, strategy_dual = run_strategy_session(
        grid=grid, context=context,
        session_name="3. Dual Distillation",
        initial_model=distilled_model,  
        enable_dual_distill=True,       
        apskd_alpha=target_apskd        
    )

    # ---------------------------------------------------------
    # 步骤 3: 绘图与保存
    # ---------------------------------------------------------
    
    # 3.1 绘制原有的三方对比图
    comparison_data = {
        "Baseline": res_baseline.evaluate_metrics_serverapp,
        "Only Ground": res_only_ground.evaluate_metrics_serverapp,
        "Dual Distill": res_dual.evaluate_metrics_serverapp
    }
    plot_comparison(comparison_data, filename="comparison_three_way.png")
    
    # 3.2 [新增] 绘制 Student vs Mentor 风格图
    # 数据来源：Global Student 的 Metrics (Result) 和 Teacher 的 Metrics (Strategy)
    plot_student_mentor_style(
        student_metrics=res_dual.evaluate_metrics_serverapp,
        teacher_metrics=strategy_dual.teacher_metrics_history,
        filename="student_mentor_style.png"
    )
    
    print("\nSaving Best Model (Dual Distill) to disk...")
    torch.save(res_dual.arrays.to_torch_state_dict(), "final_model_dual.pt")
    print("\n✅ 所有实验完成！请查看 comparison.png 和 student_mentor_style.png")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:  
    """全局评估回调"""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model.to(device) 
    
    try:
        from fedavg.task import test_centralized_dataset
        test_dataloader = test_centralized_dataset()
    except ImportError:
        test_dataloader = load_centralized_dataset()

    test_loss, test_acc = test(model, test_dataloader, device)  
    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})