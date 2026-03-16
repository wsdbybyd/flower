import torch  # 导入 PyTorch，用于模型加载、保存及设备管理
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # 导入 Flower 的核心数据结构
from flwr.serverapp import Grid, ServerApp  # 导入 Flower 服务器端组件

from fedavg.fedavg import FedAvg 
from fedavg.task import Net, test_centralized_dataset, test  # 导入模型定义及测试工具

app = ServerApp()


@app.main() 
def main(grid: Grid, context: Context) -> None: 
    """
    Flower ServerApp 主入口函数。
    负责读取配置、初始化全局模型、配置策略并启动联邦学习流程。
    """

    # ============================================================
    # 1. 读取基础训练参数 (Basic Training Hyperparameters)
    # ============================================================
    # 评估采样比例 (1.0 表示评估所有可用节点)
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 1.0))
    # 联邦学习总轮数
    num_rounds: int = int(context.run_config.get("num-server-rounds", 5))
    # 客户端学习率 (将通过 config 下发给卫星)
    lr: float = float(context.run_config.get("learning-rate", 0.01))

    # ============================================================
    # 2. 读取调度与成员管理参数 (Scheduling & Turnover)
    # 对应论文中卫星可见性窗口与动态拓扑管理逻辑
    # ============================================================
    # 节点选择模式: "window" (基于通信窗口余量) 或 "random" (随机)
    selection_mode: str = str(context.run_config.get("selection-mode", "window"))
    
    # 每一轮参与训练/评估的最少节点数
    ks_train = context.run_config.get("ks-train", None)
    ks_eval = context.run_config.get("ks-eval", None)
    
    # 是否强制过滤掉通信超时的节点 (True: 严格窗口限制)
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", True))

    # 动态成员管理参数 (Turnover Management)
    turnover_T: int = int(context.run_config.get("turnover-T", 5))      # 历史记录窗口长度
    deltaQ: int = int(context.run_config.get("deltaQ", 2))               # 入网阈值 (连续成功次数)
    deltaP: int = int(context.run_config.get("deltaP", 3))               # 退网阈值 (连续失败次数)
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1)) # 每轮探测的被屏蔽节点数

    # ============================================================
    # 3. 读取蒸馏相关参数 (Distillation Configuration)
    # 对应论文 Section 2.2, 2.3, 2.4
    # ============================================================
    # [Server Side] 地面站 Teacher 模型的学习率 (用于 Stage 3 反向蒸馏更新)
    server_teacher_lr = float(context.run_config.get("server-teacher-lr", 0.001))
    
    # 构造蒸馏配置字典 (将被打包进 train_config)
    distill_config = {
        # [Client Side] APSKD 参数: 控制卫星本地自蒸馏的强度 (线性增长终值)
        "apskd-a-T": float(context.run_config.get("apskd-a-T", 0.5)),
        
        # [Server Side] Stage 1: 地面站 Teacher -> Student 蒸馏参数
        "gs-distill-epochs": int(context.run_config.get("gs-distill-epochs", 1)),
        "gs-distill-alpha": float(context.run_config.get("gs-distill-alpha", 0.5)),
        "gs-distill-temp": float(context.run_config.get("gs-distill-temp", 3.0)),
        
        # 将基础学习率传递给 Client
        "lr": lr,
    }

    

    # ============================================================
    # 4. 读取通信链路参数 (Physics Simulation Parameters)
    # 这些参数将下发给 Client，用于计算 SNR、速率和时延
    # ============================================================
    sigma2 = float(context.run_config.get("sigma2", 1e-9))
    
    comm_defaults = {  
        "sigma2": sigma2,  
        "default-distance-km": float(context.run_config.get("default-distance-km", 780.0)),
        "default-tau-d-s": float(context.run_config.get("default-tau-d-s", 50000.0)), # 可见时间窗口
        "w-u-hz": float(context.run_config.get("w-u-hz", 4e6)),       # 上行带宽
        "w-d-hz": float(context.run_config.get("w-d-hz", 4e6)),       # 下行带宽
        "p-u-w": float(context.run_config.get("p-u-w", 100.0)),       # 上行功率
        "p-d-w": float(context.run_config.get("p-d-w", 100.0)),       # 下行功率
        "f-c-hz": float(context.run_config.get("f-c-hz", 20e9)),      # 载波频率
        "A-T": float(context.run_config.get("A-T", 60.0)),            # 发射增益
        "A-R": float(context.run_config.get("A-R", 30.0)),            # 接收增益
        "G-H": float(context.run_config.get("G-H", 0.8)),             # 硬件损耗
        "delta": float(context.run_config.get("delta", 2.0)),         # 衰落因子
        "psi-db-per-km": float(context.run_config.get("psi-db-per-km", 0.5)), # 大气衰减
        "zeta-km": float(context.run_config.get("zeta-km", 500.0)),   # 衰减尺度
    }

    # ============================================================
    # 5. 初始化全局模型 (Global Student Model)
    # ============================================================
    # 这是将被分发到卫星上的轻量化模型
    global_model = Net() 
    arrays = ArrayRecord(global_model.state_dict())

    # ============================================================
    # 6. 构建策略 (Strategy Initialization)
    # 使用自定义的 FedAvg 类，注入双向蒸馏和卫星调度逻辑
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
    
    # 注入 Teacher 的学习率 (策略内部的 aggregate_train 会用到)
    strategy.server_teacher_lr = server_teacher_lr

    # ============================================================
    # 7. 构造下发配置 (Config Construction)
    # ============================================================
    # train_cfg 包含了：
    # 1. Client 需要的物理参数 (comm_defaults)
    # 2. Client 需要的 APSKD 参数 (distill_config)
    # 3. Server 策略在 configure_train 中需要的 GS 蒸馏参数 (distill_config)
    train_cfg_dict = {**comm_defaults, **distill_config}
    train_cfg = ConfigRecord(train_cfg_dict)
    
    # 评估阶段只需要通信参数
    eval_cfg = ConfigRecord({**comm_defaults})

    # ============================================================
    # 8. 启动联邦训练 (Start Simulation)
    # ============================================================
    result = strategy.start(  
        grid=grid, 
        initial_arrays=arrays,  
        train_config=train_cfg,  
        evaluate_config=eval_cfg, 
        num_rounds=num_rounds, 
        evaluate_fn=global_evaluate,  # 指定服务器端全局评估函数
    )

    # 保存最终训练好的 Global Student 模型
    print("\nSaving final model to disk...")  
    state_dict = result.arrays.to_torch_state_dict()  
    torch.save(state_dict, "final_model.pt")  


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:  
    """
    服务器端全局评估函数 (Centralized Evaluation)。
    
    在每一轮结束后，使用地面站持有的测试集 (Test Set) 对当前的 Global Student 模型
    进行性能评估，以监控全局模型的收敛情况。
    """
    model = Net()  # 实例化 Student 模型结构
    model.load_state_dict(arrays.to_torch_state_dict())  # 加载当前轮次的全局参数

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model.to(device) 

    # 加载完整的测试集 (在 task.py 中定义)
    test_dataloader = test_centralized_dataset()
    
    # 执行推理
    test_loss, test_acc = test(model, test_dataloader, device)  

    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})