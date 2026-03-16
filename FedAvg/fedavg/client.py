import math  # 导入数学库用于常量与对数等计算
import random  # 导入随机库用于生成节点距离
from typing import Any, Dict  # 导入类型注解用于声明字典与任意类型

import torch  # 导入 PyTorch 用于模型与张量计算

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict  # 导入 Flower 的记录类型与上下文消息类型
from flwr.clientapp import ClientApp  # 导入 Flower 的客户端应用封装
from flwr.common import log  # 导入 Flower 的日志函数
from logging import INFO  # 导入日志级别常量

from fedavg.task import Net, load_data  # 导入模型结构与数据加载函数
from fedavg.task import test as test_fn  # 导入测试函数并重命名
from fedavg.task import train as train_fn  # 导入训练函数并重命名


app = ClientApp()  # 创建 Flower ClientApp 实例

_DISTANCE_CACHE_KM: Dict[int, float] = {}  # 定义距离缓存字典键为节点编号值为距离


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:  # 定义安全读取配置键的函数
    """安全读取字典键，兼容 node_config run_config config，避免 KeyError。"""  # 说明该函数用于安全读取配置
    try:  # 尝试读取键值
        return cfg[key]  # 返回配置中对应键的值
    except Exception:  # 捕获任何异常避免程序中断
        return default  # 读取失败时返回默认值


def _state_dict_size_bits(state_dict: Dict[str, torch.Tensor]) -> int:  # 定义计算模型大小的函数
    """计算模型大小 I，单位 bit，用于通信时延计算。"""  # 说明该函数输出 bit 大小
    total_bytes = 0  # 初始化总字节数计数器
    for _, t in state_dict.items():  # 遍历模型参数字典中的每个张量
        if isinstance(t, torch.Tensor):  # 判断当前对象是否为张量
            total_bytes += t.numel() * t.element_size()  # 按元素个数乘以单元素字节数累加
    return int(total_bytes * 8)  # 将字节转换为 bit 并返回整数


def compute_link_metrics(  # 定义链路指标计算函数
    *,  # 强制后续参数必须以关键字形式传入
    model_bits: int,  # 传入模型大小 bit
    d_km: float,  # 传入链路距离 km
    tau_d_s: float,  # 传入时延窗口阈值秒
    A_T: float,  # 传入发射端增益或系数
    A_R: float,  # 传入接收端增益或系数
    f_c_hz: float,  # 传入载波频率 Hz
    G_H: float,  # 传入高增益项或方向性系数
    delta_rician: float,  # 传入瑞利或莱斯衰落相关系数
    psi_db_per_km: float,  # 传入每公里衰减系数 dB
    zeta_km: float,  # 传入衰减尺度参数 km
    w_u_hz: float,  # 传入上行带宽 Hz
    p_u_w: float,  # 传入上行功率 W
    w_d_hz: float,  # 传入下行带宽 Hz
    p_d_w: float,  # 传入下行功率 W
    sigma2: float,  # 传入噪声功率
) -> Dict[str, float]:  # 声明返回值为浮点字典
    """
    ρ^t = A_T · A_C · A_R  # 给出合成增益定义
    A_C = (c/(4π d f_c))^2 · G_H · A d · δ  # 给出信道系数定义
    A d = 10^  3 ψ d  /  10 ζ  # 给出大气衰减项定义
    F_u = w_u log2 1 + p_u ρ^2 / σ^2  # 给出上行速率定义
    T_u = I / F_u  # 给出上行传输时延定义
    F_d = w_d log2 1 + p_d ρ^2 / σ^2  # 给出下行速率定义
    T_d = I / F_d  # 给出下行传输时延定义
    约束：T_u + T_d ≤ τ_d  # 给出通信可行性约束
    """  # 结束公式说明
    c = 299_792_458.0  # 定义光速常量 m 每秒
    d_m = max(float(d_km), 1e-9) * 1000.0  # 将距离从 km 转为 m 并避免零值
    zeta_km = max(float(zeta_km), 1e-9)  # 将衰减尺度限制为非零避免除零

    # 大气雨衰项  # 标注下面计算大气衰减
    A_atm = 10.0 ** ((3.0 * float(psi_db_per_km) * float(d_km)) / (10.0 * zeta_km))  # 根据距离与参数计算衰减项

    fspl = (c / (4.0 * math.pi * d_m * max(float(f_c_hz), 1e-9))) ** 2  # 计算自由空间路径损耗项并避免频率为零
    A_C = fspl * float(G_H) * A_atm * float(delta_rician)  # 合成信道系数包含增益衰减与莱斯因子

    # A_T A_R：若看起来像 dB 就转线性否则直接当线性  # 标注发射接收增益可能需要转线性
    A_T_lin = (10.0 ** (float(A_T) / 10.0)) if float(A_T) > 10.0 else float(A_T)  # 将发射端增益转换为线性值
    A_R_lin = (10.0 ** (float(A_R) / 10.0)) if float(A_R) > 10.0 else float(A_R)  # 将接收端增益转换为线性值

    rho = A_T_lin * A_C * A_R_lin  # 计算整体链路增益系数

    sigma2 = max(float(sigma2), 1e-18)  # 将噪声功率限制为非零避免除零
    w_u_hz = max(float(w_u_hz), 1e-9)  # 将上行带宽限制为非零
    w_d_hz = max(float(w_d_hz), 1e-9)  # 将下行带宽限制为非零

    snr_u = (float(p_u_w) * (rho ** 2)) / sigma2  # 计算上行信噪比
    snr_d = (float(p_d_w) * (rho ** 2)) / sigma2  # 计算下行信噪比

    F_u = w_u_hz * math.log2(1.0 + max(snr_u, 0.0))  # 计算上行速率并保证对数输入非负
    F_d = w_d_hz * math.log2(1.0 + max(snr_d, 0.0))  # 计算下行速率并保证对数输入非负

    T_u = (float(model_bits) / F_u) if F_u > 1e-12 else float("inf")  # 计算上行传输时延速率过小则视为无穷
    T_d = (float(model_bits) / F_d) if F_d > 1e-12 else float("inf")  # 计算下行传输时延速率过小则视为无穷

    T_total = T_u + T_d  # 计算上下行总传输时延
    comm_ok = (T_total <= float(tau_d_s))  # 判断总时延是否满足窗口约束

    return {  # 返回所有链路指标字典
        "d_km": float(d_km),  # 回传距离数值
        "tau_d_s": float(tau_d_s),  # 回传窗口阈值数值
        "rho": float(rho),  # 回传整体链路增益系数
        "A_atm": float(A_atm),  # 回传大气衰减项
        "snr_u": float(snr_u),  # 回传上行信噪比
        "snr_d": float(snr_d),  # 回传下行信噪比
        "F_u_bps": float(F_u),  # 回传上行速率比特每秒
        "F_d_bps": float(F_d),  # 回传下行速率比特每秒
        "T_u_s": float(T_u),  # 回传上行时延秒
        "T_d_s": float(T_d),  # 回传下行时延秒
        "T_total_s": float(T_total),  # 回传总时延秒
        "window_margin_s": float(float(tau_d_s) - T_total) if math.isfinite(T_total) else float("-inf"),  # 回传窗口余量并处理无穷情况
        "comm_ok": 1.0 if comm_ok else 0.0,  # 回传通信可行性用数值表示
        "model_bits": float(model_bits),  # 回传模型大小 bit
    }  # 结束返回字典


def _merge_cfg(msg: Message, context: Context) -> Dict[str, Any]:  # 定义统一合并配置的函数
    """统一合并配置：run_config < server_config < node_config。"""  # 说明合并顺序与覆盖关系
    cfg_server = dict(msg.content.get("config", {}))  # 从消息内容读取服务器下发配置
    cfg_run = dict(context.run_config)  # 从运行配置读取全局运行参数
    cfg_node = dict(context.node_config)  # 从节点配置读取节点级参数

    cfg: Dict[str, Any] = {}  # 初始化合并后的配置字典
    cfg.update(cfg_run)  # 先写入运行配置作为低优先级
    cfg.update(cfg_server)  # 再写入服务器配置覆盖运行配置
    cfg.update(cfg_node)  # 最后写入节点配置覆盖前两者
    return cfg  # 返回合并后的配置字典


def _rand_distance_km(cfg: Dict[str, Any], partition_id: int) -> float:  
    """
    方案:每个节点启动时随机一次距离并固定  
    可在 pyproject.toml 的 tool flwr app config 配 distance-km-min distance-km-max distance-seed  
    """  
    if partition_id in _DISTANCE_CACHE_KM:  # 判断该节点是否已有缓存距离
        return _DISTANCE_CACHE_KM[partition_id]  # 直接返回缓存距离

    d_min = float(_get(cfg, "distance-km-min", 600.0))  # 读取最小距离默认 600
    d_max = float(_get(cfg, "distance-km-max", 2000.0))  # 读取最大距离默认 2000
    base_seed = int(_get(cfg, "distance-seed", 2026))  # 读取基础随机种子默认 2026

    rng = random.Random(base_seed + int(partition_id))  # 使用基础种子加节点编号构造独立随机源
    d_km = rng.uniform(d_min, d_max)  # 在范围内均匀采样距离

    _DISTANCE_CACHE_KM[partition_id] = d_km  # 将采样距离写入缓存
    return d_km  # 返回本节点固定距离


@app.train()  # 注册训练处理函数到 ClientApp
def train(msg: Message, context: Context) -> Message:  # 定义训练回调函数
    # ---- 模型与数据 ----  # 标注下面加载模型与数据
    model = Net()  # 实例化模型结构
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  # 从服务器下发参数加载到模型

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择可用设备优先 GPU
    model.to(device)  # 将模型迁移到目标设备

    partition_id = int(context.node_config["partition-id"])  # 读取本节点分区编号
    num_partitions = int(context.node_config["num-partitions"])  # 读取总分区数量
    batch_size = int(context.run_config["batch-size"])  # 读取批大小配置

    trainloader, _ = load_data(partition_id, num_partitions, batch_size)  # 加载本节点训练数据与验证数据占位

    train_loss = train_fn(  # 调用训练函数执行本地训练
        model,  # 传入模型
        trainloader,  # 传入训练数据加载器
        int(context.run_config["local-epochs"]),  # 传入本地训练轮数
        float(msg.content["config"]["lr"]),  # 传入学习率来自服务器配置
        device,  # 传入设备
    )  

    # ---- 训练后参数打包 ----  
    state_dict = model.state_dict()  # 获取训练后的参数字典
    model_record = ArrayRecord(state_dict)  # 将参数字典封装为 ArrayRecord

    # ---- 合并配置 + 读取或生成 distance ----  
    cfg = _merge_cfg(msg, context)  # 合并三类配置得到统一配置字典

    # 如果手动给了 distance-km 就用它否则随机并缓存  
    if "distance-km" in cfg:  # 判断配置中是否显式提供距离
        d_km = float(cfg["distance-km"])  # 使用显式提供的距离
    else:  # 进入未提供距离的分支
        d_km = _rand_distance_km(cfg, partition_id)  # 按方案生成并缓存距离

    tau_d_s = float(_get(cfg, "tau-d-s", _get(cfg, "default-tau-d-s", 5.0)))  # 读取窗口阈值优先 tau-d-s 否则默认
    A_T = float(_get(cfg, "A-T", 1.0))  # 读取发射端增益参数
    A_R = float(_get(cfg, "A-R", 1.0))  # 读取接收端增益参数
    f_c_hz = float(_get(cfg, "f-c-hz", 20e9))  # 读取载波频率参数
    G_H = float(_get(cfg, "G-H", 1.0))  # 读取方向增益参数
    delta_rician = float(_get(cfg, "delta", 1.0))  # 读取衰落系数参数
    psi_db_per_km = float(_get(cfg, "psi-db-per-km", 0.0))  # 读取每公里衰减参数
    zeta_km = float(_get(cfg, "zeta-km", 550.0))  # 读取衰减尺度参数
    w_u_hz = float(_get(cfg, "w-u-hz", 1e6))  # 读取上行带宽参数
    w_d_hz = float(_get(cfg, "w-d-hz", 1e6))  # 读取下行带宽参数
    p_u_w = float(_get(cfg, "p-u-w", 1.0))  # 读取上行功率参数
    p_d_w = float(_get(cfg, "p-d-w", 1.0))  # 读取下行功率参数
    sigma2 = float(_get(cfg, "sigma2", 1e-9))  # 读取噪声功率参数

    model_bits = _state_dict_size_bits(state_dict)  # 计算模型参数总大小 bit

    comm_metrics = compute_link_metrics(  # 调用链路计算函数得到链路指标
        model_bits=model_bits,  # 传入模型大小
        d_km=d_km,  # 传入距离
        tau_d_s=tau_d_s,  # 传入窗口阈值
        A_T=A_T,  # 传入发射端参数
        A_R=A_R,  # 传入接收端参数
        f_c_hz=f_c_hz,  # 传入载波频率
        G_H=G_H,  # 传入方向增益
        delta_rician=delta_rician,  # 传入瑞利莱斯系数
        psi_db_per_km=psi_db_per_km,  # 传入每公里衰减
        zeta_km=zeta_km,  # 传入衰减尺度
        w_u_hz=w_u_hz,  # 传入上行带宽
        p_u_w=p_u_w,  # 传入上行功率
        w_d_hz=w_d_hz,  # 传入下行带宽
        p_d_w=p_d_w,  # 传入下行功率
        sigma2=sigma2,  # 传入噪声功率
    )  # 结束链路指标计算

    # 可选：打印每个节点的距离与可行性便于观察  # 标注下面是调试日志
    log(  # 调用日志输出
        INFO,  # 使用信息级别
        "[node %s] d_km=%.1f, comm_ok=%s, margin=%.2f, T_total=%.2f",  # 定义日志输出模板
        partition_id,  # 输出节点编号
        comm_metrics["d_km"],  # 输出距离数值
        int(comm_metrics["comm_ok"]),  # 输出可行性标志
        comm_metrics["window_margin_s"],  # 输出窗口余量
        comm_metrics["T_total_s"],  # 输出总时延
    )  # 结束日志调用

    metrics = {  # 构造回传给服务器的指标字典
        "train_loss": float(train_loss),  # 回传训练损失
        "num-examples": len(trainloader.dataset),  # 回传本节点样本量用于加权

        # 回传链路指标用于服务器端窗口调度与成员管理  # 标注下面是链路相关指标
        "tau_d_s": comm_metrics["tau_d_s"],  # 回传窗口阈值
        "T_total_s": comm_metrics["T_total_s"],  # 回传总时延
        "window_margin_s": comm_metrics["window_margin_s"],  # 回传窗口余量
        "comm_ok": comm_metrics["comm_ok"],  # 回传通信可行性数值
        "d_km": comm_metrics["d_km"],  # 回传距离

        "F_u_bps": comm_metrics["F_u_bps"],  # 回传上行速率
        "F_d_bps": comm_metrics["F_d_bps"],  # 回传下行速率
        "A_atm": comm_metrics["A_atm"],  # 回传大气衰减项
        "snr_u": comm_metrics["snr_u"],  # 回传上行信噪比
        "snr_d": comm_metrics["snr_d"],  # 回传下行信噪比
        "rho": comm_metrics["rho"],  # 回传链路增益系数
        "model_bits": comm_metrics["model_bits"],  # 回传模型大小
    }  # 结束指标字典构造

    metric_record = MetricRecord(metrics)  # 将指标字典封装为 MetricRecord
    content = RecordDict({"arrays": model_record, "metrics": metric_record})  # 构造回复消息内容包含参数与指标
    return Message(content=content, reply_to=msg)  # 返回回复消息给服务器


@app.evaluate()  # 注册评估处理函数到 ClientApp
def evaluate(msg: Message, context: Context) -> Message:  # 定义评估回调函数
    # ---- 模型与数据 ----  
    model = Net()  # 实例化模型结构
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())  # 从服务器下发参数加载到模型

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择可用设备优先 GPU
    model.to(device)  # 将模型迁移到目标设备

    partition_id = int(context.node_config["partition-id"])  # 读取本节点分区编号
    num_partitions = int(context.node_config["num-partitions"])  # 读取总分区数量
    batch_size = int(context.run_config["batch-size"])  # 读取批大小配置

    _, valloader = load_data(partition_id, num_partitions, batch_size)  # 加载本节点验证数据

    eval_loss, eval_acc = test_fn(model, valloader, device)  # 调用测试函数得到损失与准确率

    # ---- 合并配置 + 读取或生成 distance ----  
    cfg = _merge_cfg(msg, context)  # 合并三类配置得到统一配置字典

    if "distance-km" in cfg:  # 判断配置中是否显式提供距离
        d_km = float(cfg["distance-km"])  # 使用显式提供的距离
    else:  # 进入未提供距离的分支
        d_km = _rand_distance_km(cfg, partition_id)  # 按方案生成并缓存距离

    tau_d_s = float(_get(cfg, "tau-d-s", _get(cfg, "default-tau-d-s", 5.0)))  # 读取窗口阈值优先 tau-d-s 否则默认
    A_T = float(_get(cfg, "A-T", 1.0))  # 读取发射端增益参数
    A_R = float(_get(cfg, "A-R", 1.0))  # 读取接收端增益参数
    f_c_hz = float(_get(cfg, "f-c-hz", 20e9))  # 读取载波频率参数
    G_H = float(_get(cfg, "G-H", 1.0))  # 读取方向增益参数
    delta_rician = float(_get(cfg, "delta", 1.0))  # 读取衰落系数参数
    psi_db_per_km = float(_get(cfg, "psi-db-per-km", 0.0))  # 读取每公里衰减参数
    zeta_km = float(_get(cfg, "zeta-km", 550.0))  # 读取衰减尺度参数
    w_u_hz = float(_get(cfg, "w-u-hz", 1e6))  # 读取上行带宽参数
    w_d_hz = float(_get(cfg, "w-d-hz", 1e6))  # 读取下行带宽参数
    p_u_w = float(_get(cfg, "p-u-w", 1.0))  # 读取上行功率参数
    p_d_w = float(_get(cfg, "p-d-w", 1.0))  # 读取下行功率参数
    sigma2 = float(_get(cfg, "sigma2", 1e-9))  # 读取噪声功率参数

    model_bits = _state_dict_size_bits(model.state_dict())  # 计算当前模型大小 bit

    comm_metrics = compute_link_metrics(  # 调用链路计算函数得到链路指标
        model_bits=model_bits,  # 传入模型大小
        d_km=d_km,  # 传入距离
        tau_d_s=tau_d_s,  # 传入窗口阈值
        A_T=A_T,  # 传入发射端参数
        A_R=A_R,  # 传入接收端参数
        f_c_hz=f_c_hz,  # 传入载波频率
        G_H=G_H,  # 传入方向增益
        delta_rician=delta_rician,  # 传入瑞利莱斯系数
        psi_db_per_km=psi_db_per_km,  # 传入每公里衰减
        zeta_km=zeta_km,  # 传入衰减尺度
        w_u_hz=w_u_hz,  # 传入上行带宽
        p_u_w=p_u_w,  # 传入上行功率
        w_d_hz=w_d_hz,  # 传入下行带宽
        p_d_w=p_d_w,  # 传入下行功率
        sigma2=sigma2,  # 传入噪声功率
    )  # 结束链路指标计算

    metrics = {  # 构造回传给服务器的评估指标字典
        "eval_loss": float(eval_loss),  # 回传评估损失
        "eval_acc": float(eval_acc),  # 回传评估准确率
        "num-examples": len(valloader.dataset),  # 回传本节点样本量用于加权

        "tau_d_s": comm_metrics["tau_d_s"],  # 回传窗口阈值
        "T_total_s": comm_metrics["T_total_s"],  # 回传总时延
        "window_margin_s": comm_metrics["window_margin_s"],  # 回传窗口余量
        "comm_ok": comm_metrics["comm_ok"],  # 回传通信可行性数值
        "d_km": comm_metrics["d_km"],  # 回传距离

        "A_atm": comm_metrics["A_atm"],  # 回传大气衰减项
        "snr_u": comm_metrics["snr_u"],  # 回传上行信噪比
        "snr_d": comm_metrics["snr_d"],  # 回传下行信噪比
        "rho": comm_metrics["rho"],  # 回传链路增益系数
        "model_bits": comm_metrics["model_bits"],  # 回传模型大小
    }  # 结束评估指标字典构造

    metric_record = MetricRecord(metrics)  # 将评估指标字典封装为 MetricRecord
    content = RecordDict({"metrics": metric_record})  # 构造回复消息内容仅包含指标
    return Message(content=content, reply_to=msg)  # 返回回复消息给服务器
