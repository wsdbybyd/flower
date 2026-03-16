import torch  # 导入 PyTorch 用于模型参数保存与设备选择
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # 导入 Flower 的记录类型与上下文类型
from flwr.serverapp import Grid, ServerApp  # 导入 Flower 服务器端网格与 ServerApp

from fedavg.fedavg import FedAvg  # 导入自定义的 FedAvg 策略实现
from fedavg.task import Net, load_centralized_dataset, test  # 导入模型结构与集中测试数据加载与测试函数

app = ServerApp()  # 创建 Flower ServerApp 实例


@app.main() 
def main(grid: Grid, context: Context) -> None:  # 定义服务器主函数接收网格与上下文
    # ---------- 基础训练参数 ----------  
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.5))  # 读取评估采样比例默认 0.5
    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))  # 读取联邦训练轮数默认 3
    lr: float = float(context.run_config.get("learning-rate", 0.1))  # 读取学习率默认 0.1

    # ---------- 调度退出加入参数 从 pyproject toml 读取 ---------- 
    selection_mode: str = str(context.run_config.get("selection-mode", "random"))  # 读取选择模式支持 random 与 window
    ks_train = context.run_config.get("ks-train", None)  # 读取训练每轮选择的节点数可能为空
    ks_eval = context.run_config.get("ks-eval", None)  # 读取评估每轮选择的节点数可能为空
    enforce_comm: bool = bool(context.run_config.get("enforce-comm", False))  # 读取是否强制满足通信约束默认否

    turnover_T: int = int(context.run_config.get("turnover-T", 10))  # 读取统计窗口长度默认 10
    deltaQ: int = int(context.run_config.get("deltaQ", 999999))  # 读取入网阈值默认极大表示不启用
    deltaP: int = int(context.run_config.get("deltaP", 999999))  # 读取退网阈值默认极大表示不启用
    probe_blocked_k: int = int(context.run_config.get("probe-blocked-k", 1))  # 读取每轮探测被屏蔽节点数量默认 1

    # ---------- 通信链路参数 下发给客户端计算 comm_ok 与 T_total 等 ----------  
    sigma2 = float(context.run_config.get("sigma2", 1e-9))  # 读取噪声功率默认 1e-9

    comm_defaults = {  
        "sigma2": sigma2,  # 写入噪声功率
        "default-distance-km": float(context.run_config.get("default-distance-km", 700.0)),  # 写入默认距离用于客户端未指定时
        "default-tau-d-s": float(context.run_config.get("default-tau-d-s", 5.0)),  # 写入默认窗口阈值用于客户端未指定时
        "w-u-hz": float(context.run_config.get("w-u-hz", 1e6)),  # 写入上行带宽默认 1e6
        "w-d-hz": float(context.run_config.get("w-d-hz", 1e6)),  # 写入下行带宽默认 1e6
        "p-u-w": float(context.run_config.get("p-u-w", 1.0)),  # 写入上行功率默认 1
        "p-d-w": float(context.run_config.get("p-d-w", 1.0)),  # 写入下行功率默认 1
        "f-c-hz": float(context.run_config.get("f-c-hz", 20e9)),  # 写入载波频率默认 20e9
        "A-T": float(context.run_config.get("A-T", 1.0)),  # 写入发射端参数默认 1
        "A-R": float(context.run_config.get("A-R", 1.0)),  # 写入接收端参数默认 1
        "G-H": float(context.run_config.get("G-H", 1.0)),  # 写入方向增益参数默认 1
        "delta": float(context.run_config.get("delta", 1.0)),  # 写入衰落系数参数默认 1
        "psi-db-per-km": float(context.run_config.get("psi-db-per-km", 0.0)),  # 写入每公里衰减默认 0
        "zeta-km": float(context.run_config.get("zeta-km", 550.0)),  # 写入衰减尺度参数默认 550
    }  # 结束链路默认参数构造

    # ---------- 初始化全局模型 ----------  
    global_model = Net()  # 实例化全局模型结构
    arrays = ArrayRecord(global_model.state_dict())  # 将全局模型参数封装为 ArrayRecord

    # ---------- 构建策略 ----------  
    strategy = FedAvg(  # 创建策略实例并传入调度与成员管理参数
        fraction_evaluate=fraction_evaluate,  # 设置评估采样比例
        selection_mode=selection_mode,  # 设置节点选择模式
        ks_train=int(ks_train) if ks_train is not None else None,  # 如果配置了训练 K 则转为整数否则为空
        ks_evaluate=int(ks_eval) if ks_eval is not None else None,  # 如果配置了评估 K 则转为整数否则为空
        enforce_comm=enforce_comm,  # 设置是否强制通信约束
        turnover_T=turnover_T,  # 设置统计窗口长度
        deltaQ=deltaQ,  # 设置入网阈值
        deltaP=deltaP,  # 设置退网阈值
        probe_blocked_k=probe_blocked_k,  # 设置探测被屏蔽节点数量
    )  # 结束策略初始化

    # ---------- 下发给客户端的配置 ----------  
    train_cfg = ConfigRecord({"lr": lr, **comm_defaults})  # 构造训练配置包含学习率与链路默认参数
    eval_cfg = ConfigRecord({**comm_defaults})  # 构造评估配置仅包含链路默认参数

    # ---------- 启动联邦训练 ----------  
    result = strategy.start(  # 调用策略 start 开始联邦训练并返回结果
        grid=grid,  # 传入节点管理网格
        initial_arrays=arrays,  # 传入初始全局模型参数
        train_config=train_cfg,  # 传入训练配置记录
        evaluate_config=eval_cfg,  # 传入评估配置记录
        num_rounds=num_rounds,  # 传入训练总轮数
        evaluate_fn=global_evaluate,  # 传入服务器端全局评估函数
    )  # 结束训练启动调用

    print("\nSaving final model to disk...")  # 打印提示准备保存最终模型
    state_dict = result.arrays.to_torch_state_dict()  # 将最终聚合后的参数转换为 state_dict
    torch.save(state_dict, "final_model.pt")  # 将最终模型参数保存到本地文件


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:  # 定义服务器端全局评估函数
    model = Net()  # 实例化模型结构用于评估
    model.load_state_dict(arrays.to_torch_state_dict())  # 将全局参数加载到评估模型中

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择可用设备优先 GPU
    model.to(device)  # 将模型迁移到目标设备

    test_dataloader = load_centralized_dataset()  # 加载服务器端集中测试数据
    test_loss, test_acc = test(model, test_dataloader, device)  # 在集中测试集上计算损失与准确率

    return MetricRecord({"accuracy": float(test_acc), "loss": float(test_loss)})  # 将评估结果封装为 MetricRecord 返回
