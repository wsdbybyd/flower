import torch  # 导入 PyTorch 用于模型参数保存与设备选择
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # 导入 Flower 的记录类型与上下文类型
from flwr.serverapp import Grid, ServerApp  # 导入 Flower 服务器端网格与 ServerApp
import matplotlib.pyplot as plt  # 新增：导入绘图库

from fedavg.fedavg import FedAvg  # 导入自定义的 FedAvg 策略实现
from fedavg.task import Net, load_centralized_dataset, test  # 导入模型结构与集中测试数据加载与测试函数

app = ServerApp()  # 创建 Flower ServerApp 实例


def plot_metrics(result):
    """提取联邦学习训练结果，并绘制 Loss 和 Accuracy 双轨对比图"""
    
    # --- 1. 提取服务器端（集中式）评估指标 ---
    server_metrics = result.evaluate_metrics_serverapp
    server_rounds = sorted(list(server_metrics.keys()))
    server_acc = [server_metrics[r].get("accuracy", 0) for r in server_rounds]
    server_loss = [server_metrics[r].get("loss", 0) for r in server_rounds]

    # --- 2. 提取客户端（联邦式）聚合评估指标 ---
    client_metrics = result.evaluate_metrics_clientapp
    client_rounds = sorted(list(client_metrics.keys()))
    # 注意：这里的键名对应 client.py 中返回的指标名称
    client_acc = [client_metrics[r].get("eval_acc", 0) for r in client_rounds]
    client_loss = [client_metrics[r].get("eval_loss", 0) for r in client_rounds]

    if not server_metrics and not client_metrics:
        print("\n未找到任何评估指标，跳过绘图。")
        return

    # --- 3. 开始绘制图表 ---
    plt.figure(figsize=(14, 6))

    # 绘制 Loss 子图
    plt.subplot(1, 2, 1)
    if server_rounds:
        plt.plot(server_rounds, server_loss, marker='o', color='b', label='Server (Global) Loss', linewidth=2)
    if client_rounds:
        plt.plot(client_rounds, client_loss, marker='s', color='c', label='Client (Aggregated) Loss', linestyle='--')
    plt.title('Model Loss over Communication Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # 绘制 Accuracy 子图
    plt.subplot(1, 2, 2)
    if server_rounds:
        plt.plot(server_rounds, server_acc, marker='o', color='darkorange', label='Server (Global) Accuracy', linewidth=2)
    if client_rounds:
        plt.plot(client_rounds, client_acc, marker='s', color='gold', label='Client (Aggregated) Accuracy', linestyle='--')
    plt.title('Model Accuracy over Communication Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # 调整布局并保存
    plt.tight_layout()
    filename = "federated_learning_metrics.png"
    plt.savefig(filename, dpi=300)
    print(f"\n📈 训练指标对比图表已成功保存至: '{filename}'")


@app.main() 
def main(grid: Grid, context: Context) -> None:  # 定义服务器主函数接收网格与上下文
    # ---------- 基础训练参数 ----------  
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.5))  
    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))  
    lr: float = float(context.run_config.get("learning-rate", 0.1))  

    # ---------- 调度退出加入参数 从 pyproject toml 读取 ---------- 
    selection_mode: str = str(context.run_config.get("selection-mode", "random"))  
    ks_train = context.run_config.get("ks-train", None)  
    ks_eval = context.run_config.get("ks-eval", None)  
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", False))  

    turnover_T: int = int(context.run_config.get("turnover-T", 10))  
    deltaQ: int = int(context.run_config.get("deltaQ", 999999))  
    deltaP: int = int(context.run_config.get("deltaP", 999999))  
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))  

    # ---------- 通信链路参数 下发给客户端计算 comm_ok 与 T_total 等 ----------  
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
    train_cfg = ConfigRecord({"lr": lr, **comm_defaults})  
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

    # ---------- 绘制并保存训练指标图表 ----------
    plot_metrics(result)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:  
    model = Net()  
    model.load_state_dict(arrays.to_torch_state_dict())  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model.to(device)  

    test_dataloader = load_centralized_dataset()  
    test_loss, test_acc = test(model, test_dataloader, device)  

    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})