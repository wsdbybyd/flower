from __future__ import annotations  # 启用延迟注解解析行为

import random  # 导入随机模块用于打乱与采样
from collections.abc import Callable, Iterable  # 导入可调用与可迭代抽象类型
from dataclasses import dataclass  # 导入数据类装饰器
from logging import INFO, WARNING  # 导入日志级别常量
from time import sleep  # 导入休眠函数用于等待
from typing import Any  # 导入任意类型注解

from flwr.common import (  # 从 Flower 公共模块导入通用类型
    ArrayRecord,  # 导入参数数组记录类型
    ConfigRecord,  # 导入配置记录类型
    Message,  # 导入消息类型
    MessageType,  # 导入消息类型枚举
    MetricRecord,  # 导入指标记录类型
    RecordDict,  # 导入记录字典类型
    log,  # 导入 Flower 日志函数
)  # 结束导入列表
from flwr.server import Grid  # 导入服务器端节点网格管理器

from .strategy import Strategy  # 导入策略基类
from .strategy_utils import (  # 从策略工具模块导入工具函数
    aggregate_arrayrecords,  # 导入模型参数聚合函数
    aggregate_metricrecords,  # 导入指标聚合函数
    sample_nodes,  # 导入节点采样函数
    validate_message_reply_consistency,  # 导入回复一致性校验函数
)  # 结束导入列表


@dataclass  # 声明数据类以便自动生成初始化等方法
class _LinkState:  # 定义链路状态缓存结构
    tau_d_s: float | None = None  # 记录窗口长度数值
    window_margin_s: float | None = None  # 记录窗口余量数值
    comm_ok: bool | None = None  # 记录通信是否可行标志
    t_total_s: float | None = None  # 记录总耗时数值


def _metricrecord_to_dict(mr: Any) -> dict[str, Any]:  # 定义将指标记录转为字典的兼容函数
    if mr is None:  # 判断输入是否为空
        return {}  # 返回空字典
    try:  # 尝试直接用字典构造
        return dict(mr)  # 返回直接转换结果
    except Exception:  # 捕获转换异常
        pass  # 忽略异常继续尝试
    for attr in ("to_dict", "as_dict", "dict"):  # 遍历可能存在的转换方法名
        if hasattr(mr, attr):  # 判断对象是否具有该方法
            try:  # 尝试调用该方法
                return getattr(mr, attr)()  # 调用方法并返回结果
            except Exception:  # 捕获调用异常
                pass  # 忽略异常继续尝试
    try:  # 尝试用迭代方式兜底提取
        out = {}  # 初始化输出字典
        for k in mr:  # 遍历对象键
            out[k] = mr[k]  # 将键值对写入字典
        return out  # 返回兜底转换结果
    except Exception:  # 捕获兜底异常
        return {}  # 返回空字典


class FedAvg(Strategy):  # 定义联邦平均策略类
    def __init__(  # 定义初始化方法
        self,  # 传入实例自身
        fraction_train: float = 1.0,  # 设置训练采样比例默认值
        fraction_evaluate: float = 1.0,  # 设置评估采样比例默认值
        min_train_nodes: int = 2,  # 设置最小训练节点数默认值
        min_evaluate_nodes: int = 2,  # 设置最小评估节点数默认值
        min_available_nodes: int = 2,  # 设置最小在线节点数默认值
        weighted_by_key: str = "num-examples",  # 设置加权聚合使用的键名默认值
        arrayrecord_key: str = "arrays",  # 设置参数记录在消息中的键名默认值
        configrecord_key: str = "config",  # 设置配置记录在消息中的键名默认值
        train_metrics_aggr_fn: (  # 声明训练指标聚合函数参数类型
            Callable[[list[RecordDict], str], MetricRecord] | None  # 指定可选聚合函数签名
        ) = None,  # 设置训练指标聚合函数默认值
        evaluate_metrics_aggr_fn: (  # 声明评估指标聚合函数参数类型
            Callable[[list[RecordDict], str], MetricRecord] | None  # 指定可选聚合函数签名
        ) = None,  # 设置评估指标聚合函数默认值
        selection_mode: str = "random",  # 设置节点选择模式默认值
        ks_train: int | None = None,  # 设置训练阶段固定选择数量默认值
        ks_evaluate: int | None = None,  # 设置评估阶段固定选择数量默认值
        enforce_comm: bool = False,  # 设置是否强制通信可行性约束默认值
        turnover_T: int = 10,  # 设置统计窗口长度默认值
        deltaQ: int = 999999,  # 设置加入阈值默认值
        deltaP: int = 999999,  # 设置退出阈值默认值
        probe_blocked_k: int = 1,  # 设置每轮探测被屏蔽节点数量默认值
    ) -> None:  # 声明无返回值
        self.fraction_train = fraction_train  # 保存训练采样比例到实例
        self.fraction_evaluate = fraction_evaluate  # 保存评估采样比例到实例
        self.min_train_nodes = min_train_nodes  # 保存最小训练节点数到实例
        self.min_evaluate_nodes = min_evaluate_nodes  # 保存最小评估节点数到实例
        self.min_available_nodes = min_available_nodes  # 保存最小在线节点数到实例
        self.weighted_by_key = weighted_by_key  # 保存加权键到实例
        self.arrayrecord_key = arrayrecord_key  # 保存参数键到实例
        self.configrecord_key = configrecord_key  # 保存配置键到实例
        self.train_metrics_aggr_fn = train_metrics_aggr_fn or aggregate_metricrecords  # 设置训练指标聚合函数
        self.evaluate_metrics_aggr_fn = evaluate_metrics_aggr_fn or aggregate_metricrecords  # 设置评估指标聚合函数

        if self.fraction_evaluate == 0.0:  # 判断评估比例是否为零
            self.min_evaluate_nodes = 0  # 将最小评估节点数置零
            log(WARNING, "fraction_evaluate is set to 0.0. Federated evaluate will be disabled.")  # 记录评估被禁用日志

        self.selection_mode = selection_mode  # 保存节点选择模式到实例
        self.ks_train = ks_train  # 保存训练固定选择数量到实例
        self.ks_evaluate = ks_evaluate  # 保存评估固定选择数量到实例
        self.enforce_comm = enforce_comm  # 保存强制通信约束标志到实例

        self._link_state: dict[int, _LinkState] = {}  # 初始化链路状态缓存字典

        self.turnover_T = max(int(turnover_T), 1)  # 将统计窗口长度规范到至少为一
        self.deltaQ = int(deltaQ)  # 保存加入阈值到实例
        self.deltaP = int(deltaP)  # 保存退出阈值到实例
        self._connected_history: list[set[int]] = []  # 初始化在线节点集合历史列表

        # --- 退出/加入（超窗）机制：基于comm_ok历史 ---  # 标注该段功能为基于通信可行性的成员管理
        self._comm_hist: dict[int, list[int]] = {}  # 初始化每节点通信可行性历史字典
        self._blocked: set[int] = set()  # 初始化被屏蔽节点集合
        self.probe_blocked_k = max(int(probe_blocked_k), 0)  # 将探测数量规范到非负整数

    def summary(self) -> None:  # 定义输出摘要方法
        log(INFO, "\t└── Summary: FedAvg(selection_mode=%s, enforce_comm=%s)", self.selection_mode, self.enforce_comm)  # 输出策略模式摘要日志
        log(INFO, "\t    fraction_train=%s, fraction_evaluate=%s", self.fraction_train, self.fraction_evaluate)  # 输出采样比例摘要日志
        log(INFO, "\t    min_train_nodes=%s, min_evaluate_nodes=%s, min_available_nodes=%s", self.min_train_nodes, self.min_evaluate_nodes, self.min_available_nodes)  # 输出节点数量约束摘要日志
        log(INFO, "\t    ks_train=%s, ks_evaluate=%s", self.ks_train, self.ks_evaluate)  # 输出固定选择数量摘要日志
        log(INFO, "\t    turnover_T=%s, deltaQ=%s, deltaP=%s, probe_blocked_k=%s", self.turnover_T, self.deltaQ, self.deltaP, self.probe_blocked_k)  # 输出成员管理参数摘要日志

    def _construct_messages(  # 定义构造消息列表的内部方法
        self,  # 传入实例自身
        record: RecordDict,  # 传入要发送的记录内容
        node_ids: list[int],  # 传入目标节点编号列表
        message_type: MessageType,  # 传入消息类型
    ) -> list[Message]:  # 声明返回消息列表
        messages: list[Message] = []  # 初始化消息列表容器
        for node_id in node_ids:  # 遍历每个目标节点编号
            message = Message(  # 构造单条消息对象
                content=record,  # 设置消息内容
                message_type=message_type,  # 设置消息类型
                dst_node_id=node_id,  # 设置目标节点编号
            )  # 完成消息对象构造
            messages.append(message)  # 将消息加入列表
        return messages  # 返回消息列表

    def _wait_for_nodes(self, grid: Grid, min_available_nodes: int) -> list[int]:  # 定义等待节点连接的方法
        while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:  # 循环直到在线节点数满足要求
            log(  # 输出等待日志
                INFO,  # 使用信息级别
                "Waiting for nodes to connect: %d connected (minimum required: %d).",  # 日志模板字符串
                len(all_nodes),  # 当前在线节点数
                min_available_nodes,  # 最小要求节点数
            )  # 完成日志调用
            sleep(1)  # 休眠一秒后重试
        return all_nodes  # 返回最终在线节点列表

    def _log_turnover_warning(self) -> None:  # 定义在线集合变化告警方法
        """统计在线集合的入网/退网变化（仅告警，不影响采样）。"""  # 说明该函数用途
        if len(self._connected_history) < 2:  # 判断历史记录是否不足两轮
            return  # 直接返回不处理
        window = self._connected_history[-self.turnover_T :]  # 取最近统计窗口内的在线集合
        if len(window) < 2:  # 判断窗口内集合是否不足两轮
            return  # 直接返回不处理
        joins = 0  # 初始化加入计数
        leaves = 0  # 初始化退出计数
        for prev, cur in zip(window[:-1], window[1:]):  # 遍历相邻两轮在线集合
            joins += len(cur - prev)  # 累计新增在线节点数量
            leaves += len(prev - cur)  # 累计离线节点数量
        if joins > self.deltaQ:  # 判断加入变化是否超过阈值
            log(WARNING, "Turnover warning: joins=%d > deltaQ=%d in last %d rounds", joins, self.deltaQ, len(window))  # 输出加入告警日志
        if leaves > self.deltaP:  # 判断退出变化是否超过阈值
            log(WARNING, "Turnover warning: leaves=%d > deltaP=%d in last %d rounds", leaves, self.deltaP, len(window))  # 输出退出告警日志

    # -----------------------------  
    # --- 超窗退出/加入：comm_ok --- 
    # -----------------------------  
    def _push_comm_hist(self, nid: int, ok: bool) -> None:  # 定义追加通信可行性历史的方法
        hist = self._comm_hist.get(nid, [])  # 获取该节点历史列表若不存在则用空列表
        hist.append(1 if ok else 0)  # 将本轮结果追加到历史中
        if len(hist) > self.turnover_T:  # 判断历史长度是否超过窗口
            hist = hist[-self.turnover_T :]  # 截断为最近窗口长度
        self._comm_hist[nid] = hist  # 将更新后的历史写回字典

    def _update_membership(self, nid: int) -> None:  # 定义根据历史更新成员状态的方法
        """根据最近T轮comm_ok计数决定退出/加入。
        - fail_cnt >= deltaP -> block
        - ok_cnt   >= deltaQ -> unblock
        """
        hist = self._comm_hist.get(nid, [])  # 获取节点通信可行性历史
        if not hist:  # 判断历史是否为空
            return  # 直接返回不处理

        ok_cnt = sum(hist)  # 计算可行次数
        fail_cnt = len(hist) - ok_cnt  # 计算不可行次数

        if (nid not in self._blocked) and (fail_cnt >= self.deltaP):  # 判断是否需要加入屏蔽集合
            self._blocked.add(nid)  # 将节点加入屏蔽集合
            log(WARNING, "Node %d blocked (fail_cnt=%d in last %d rounds)", nid, fail_cnt, len(hist))  # 输出屏蔽日志

        if (nid in self._blocked) and (ok_cnt >= self.deltaQ):  # 判断是否需要从屏蔽集合移除
            self._blocked.remove(nid)  # 将节点从屏蔽集合移除
            log(INFO, "Node %d unblocked (ok_cnt=%d in last %d rounds)", nid, ok_cnt, len(hist))  # 输出恢复日志

    # -----------------------------  
    # --- window调度：按余量排序 ---  
    # -----------------------------  
    def _select_by_window(self, node_ids: list[int], k: int) -> list[int]:  # 定义按窗口余量排序选择节点的方法
        def score(nid: int) -> tuple[int, float, float]:  # 定义节点评分函数
            st = self._link_state.get(nid, _LinkState())  # 获取节点链路状态若无则用默认状态
            tau = st.tau_d_s if st.tau_d_s is not None else -1e18  # 获取窗口长度若无则给极小值
            margin = st.window_margin_s if st.window_margin_s is not None else -1e18  # 获取窗口余量若无则给极小值
            comm_ok = bool(st.comm_ok) if st.comm_ok is not None else False  # 获取通信可行性标志若无则为假
            feasible = (not self.enforce_comm) or comm_ok  # 计算在强制约束下是否可行
            return (1 if feasible else 0, margin, tau)  # 返回评分三元组用于排序

        ranked = sorted(node_ids, key=score, reverse=True)  # 根据评分从高到低排序节点

        if self.enforce_comm:  # 判断是否启用强制通信可行性
            feasible = [nid for nid in ranked if (self._link_state.get(nid, _LinkState()).comm_ok is True)]  # 过滤出通信可行节点
            picked = feasible[:k]  # 先选择前k个可行节点
            if len(picked) < k:  # 判断可行节点是否不足k个
                for nid in ranked:  # 从排序列表中继续补齐
                    if nid not in picked:  # 判断是否已被选择
                        picked.append(nid)  # 将节点加入选择列表
                    if len(picked) >= k:  # 判断是否已补足数量
                        break  # 满足数量后跳出循环
            return picked  # 返回最终选择节点列表

        return ranked[:k]  # 在不强制约束时直接返回前k个节点

    # -----------------------------  
    # --- 采样辅助：过滤+探测 ---  
    # -----------------------------  
    def _filter_and_probe(self, all_nodes: list[int], sample_size: int) -> tuple[list[int], list[int], list[int]]:  # 定义过滤与探测方法
        """返回 (candidates, probe, blocked_list)。"""  # 说明返回值含义
        blocked = set(self._blocked)  # 获取当前被屏蔽节点集合副本
        candidates = [nid for nid in all_nodes if nid not in blocked]  # 过滤得到候选节点列表
        blocked_list = list(blocked)  # 将屏蔽集合转换为列表

        # 如果候选为空，至少提供一个可采样集合（避免崩）  # 标注候选为空的处理逻辑
        if not candidates and blocked_list:  # 判断候选为空且存在屏蔽节点
            candidates = blocked_list[:]  # 兜底使用屏蔽列表作为候选

        # 探测少量blocked节点（用于恢复加入）  # 标注探测逻辑用途
        probe: list[int] = []  # 初始化探测列表
        if blocked_list and self.probe_blocked_k > 0:  # 判断是否存在可探测屏蔽节点且探测数量大于零
            # 简单取前K个，也可以random.sample  # 标注当前探测策略说明
            k = min(self.probe_blocked_k, len(blocked_list))  # 计算实际探测数量
            probe = blocked_list[:k]  # 选择前k个屏蔽节点作为探测节点

        # 若 sample_size > candidates，后续会自动截断/补齐，这里不强制  # 标注不在此处强制长度原因
        return candidates, probe, blocked_list  # 返回候选列表探测列表屏蔽列表

    # -----------------------------  
    # --- Strategy接口：TRAIN ---  
    # -----------------------------  
    def configure_train(  # 定义训练阶段的配置下发方法
        self,  # 传入实例自身
        server_round: int,  # 传入当前轮次编号
        arrays: ArrayRecord,  # 传入当前全局模型参数
        config: ConfigRecord,  # 传入训练配置记录
        grid: Grid,  # 传入节点网格对象
    ) -> Iterable[Message]:  # 声明返回可迭代消息
        if self.fraction_train == 0.0:  # 判断训练是否被禁用
            return []  # 返回空消息列表

        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)  # 等待足够节点在线并获取在线列表

        # 记录在线历史（这是“在线变化告警”，与block/unblock无关）  # 标注该历史用于告警
        self._connected_history.append(set(all_nodes))  # 将本轮在线集合追加到历史
        if len(self._connected_history) > self.turnover_T:  # 判断历史长度是否超过窗口
            self._connected_history = self._connected_history[-self.turnover_T :]  # 截断为最近窗口长度
        self._log_turnover_warning()  # 触发在线变化告警计算

        # 计算本轮训练需要采样的数量  # 标注采样数量计算逻辑
        if self.selection_mode == "window" and self.ks_train is not None:  # 判断是否使用窗口模式且指定了固定数量
            sample_size = max(int(self.ks_train), self.min_train_nodes)  # 计算训练采样数量并满足最小训练节点数
        else:  # 进入按比例采样分支
            num_nodes = int(len(all_nodes) * self.fraction_train)  # 计算按比例采样的节点数
            sample_size = max(num_nodes, self.min_train_nodes)  # 取按比例与最小训练节点数中的较大值

        # 过滤被block节点 + 生成probe节点  # 标注候选与探测生成
        candidates, probe, _ = self._filter_and_probe(all_nodes, sample_size)  # 获取候选节点探测节点与屏蔽列表

        # 采样  # 标注采样执行部分
        if self.selection_mode == "window":  # 判断是否使用窗口调度
            node_ids = self._select_by_window(candidates, sample_size)  # 按窗口余量选择节点
        else:  # 进入随机采样分支
            # 只在 candidates 内随机采样（避免把blocked选进来）  # 标注随机池来源
            pool = candidates[:]  # 拷贝候选列表作为采样池
            random.shuffle(pool)  # 打乱采样池顺序
            node_ids = pool[:sample_size]  # 截取前sample_size个节点作为本轮选择

        # 合并 probe（去重）  # 标注探测节点合并逻辑
        for nid in probe:  # 遍历探测节点
            if nid not in node_ids:  # 判断探测节点是否未被选择
                node_ids.append(nid)  # 将探测节点加入选择列表

        log(INFO, "configure_train: Selected %s nodes (online=%s, mode=%s, blocked=%s, probe=%s)", len(node_ids), len(all_nodes), self.selection_mode, len(self._blocked), len(probe))  # 输出训练采样结果日志

        # 下发轮次信息  # 标注向客户端下发轮次
        config["server-round"] = server_round  # 将轮次写入配置记录
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})  # 构造要发送的记录字典
        return self._construct_messages(record, node_ids, MessageType.TRAIN)  # 构造并返回训练消息列表

    def _check_and_log_replies(  # 定义回复检查与日志方法
        self,  # 传入实例自身
        replies: Iterable[Message],  # 传入客户端回复消息集合
        is_train: bool,  # 传入是否为训练阶段标志
        validate: bool = True,  # 传入是否执行一致性校验标志
    ) -> tuple[list[Message], list[Message]]:  # 声明返回有效与错误回复列表
        valid_replies: list[Message] = []  # 初始化有效回复列表
        error_replies: list[Message] = []  # 初始化错误回复列表

        for msg in replies:  # 遍历所有回复消息
            if msg.has_error():  # 判断消息是否包含错误
                error_replies.append(msg)  # 将错误消息加入错误列表
            else:  # 进入无错误分支
                valid_replies.append(msg)  # 将正常消息加入有效列表

        log(INFO, "%s: Received %s results and %s failures", "aggregate_train" if is_train else "aggregate_evaluate", len(valid_replies), len(error_replies))  # 输出聚合前统计日志

        for msg in error_replies:  # 遍历错误回复列表
            log(INFO, "\t> Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error.reason)  # 输出每个错误节点的原因

        if validate and valid_replies:  # 判断是否需要校验且存在有效回复
            validate_message_reply_consistency(  # 调用一致性校验函数
                replies=[msg.content for msg in valid_replies],  # 提取有效回复内容列表
                weighted_by_key=self.weighted_by_key,  # 传入加权键名
                check_arrayrecord=is_train,  # 训练阶段检查参数记录一致性
            )  # 结束一致性校验调用

        return valid_replies, error_replies  # 返回有效与错误回复列表

    def _update_link_state_from_replies(self, replies: list[Message]) -> None:  # 定义从回复更新链路状态的方法
        for msg in replies:  # 遍历所有有效回复
            nid = msg.metadata.src_node_id  # 获取回复来源节点编号
            content = msg.content  # 获取回复内容字典
            if "metrics" not in content:  # 判断是否包含指标字段
                continue  # 不包含则跳过该消息

            md = _metricrecord_to_dict(content["metrics"])  # 将指标记录转换为字典

            tau = md.get("tau_d_s", None)  # 读取窗口长度指标
            margin = md.get("window_margin_s", None)  # 读取窗口余量指标
            comm_ok = md.get("comm_ok", None)  # 读取通信可行性指标
            t_total = md.get("T_total_s", None)  # 读取总耗时指标

            st = self._link_state.get(nid, _LinkState())  # 获取该节点当前链路状态若无则创建默认状态

            try:  # 尝试更新窗口长度
                st.tau_d_s = float(tau) if tau is not None else st.tau_d_s  # 将窗口长度写入缓存
            except Exception:  # 捕获转换异常
                pass  # 忽略异常保持原值

            try:  # 尝试更新窗口余量
                st.window_margin_s = float(margin) if margin is not None else st.window_margin_s  # 将窗口余量写入缓存
            except Exception:  # 捕获转换异常
                pass  # 忽略异常保持原值

            if comm_ok is not None:  # 判断通信可行性指标是否存在
                try:  # 尝试按整数转换为布尔
                    st.comm_ok = bool(int(comm_ok))  # 将可行性标志写入缓存
                except Exception:  # 捕获整数转换异常
                    try:  # 尝试直接布尔化
                        st.comm_ok = bool(comm_ok)  # 将可行性标志写入缓存
                    except Exception:  # 捕获布尔化异常
                        pass  # 忽略异常保持原值

            try:  # 尝试更新总耗时
                st.t_total_s = float(t_total) if t_total is not None else st.t_total_s  # 将总耗时写入缓存
            except Exception:  # 捕获转换异常
                pass  # 忽略异常保持原值

            self._link_state[nid] = st  # 将更新后的链路状态写回缓存

            # --- 新增：维护comm_ok历史 + 超窗退出/加入 ---  # 标注该段为成员管理更新
            if st.comm_ok is not None:  # 判断缓存中是否已有可行性结果
                self._push_comm_hist(nid, bool(st.comm_ok))  # 将可行性结果写入历史
                self._update_membership(nid)  # 基于历史更新屏蔽或恢复

    def aggregate_train(  # 定义训练阶段聚合方法
        self,  # 传入实例自身
        server_round: int,  # 传入当前轮次编号
        replies: Iterable[Message],  # 传入客户端回复消息集合
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:  # 声明返回聚合后的参数与指标
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)  # 检查回复并得到有效回复列表

        if valid_replies:  # 判断是否存在有效回复
            self._update_link_state_from_replies(valid_replies)  # 从有效回复中更新链路状态缓存

        arrays, metrics = None, None  # 初始化聚合结果占位变量
        if valid_replies:  # 判断是否存在有效回复用于聚合
            reply_contents = [msg.content for msg in valid_replies]  # 提取有效回复内容列表
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)  # 聚合模型参数得到新的全局参数
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)  # 聚合训练指标得到汇总指标
        return arrays, metrics  # 返回聚合后的参数与指标

    # -----------------------------  
    # --- Strategy接口：EVALUATE ---  
    # -----------------------------  
    def configure_evaluate(  # 定义评估阶段的配置下发方法
        self,  # 传入实例自身
        server_round: int,  # 传入当前轮次编号
        arrays: ArrayRecord,  # 传入当前全局模型参数
        config: ConfigRecord,  # 传入评估配置记录
        grid: Grid,  # 传入节点网格对象
    ) -> Iterable[Message]:  # 声明返回可迭代消息
        if self.fraction_evaluate == 0.0:  # 判断评估是否被禁用
            return []  # 返回空消息列表

        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)  # 等待足够节点在线并获取在线列表

        if self.selection_mode == "window" and self.ks_evaluate is not None:  # 判断是否使用窗口模式且指定了固定数量
            sample_size = max(int(self.ks_evaluate), self.min_evaluate_nodes)  # 计算评估采样数量并满足最小评估节点数
        else:  # 进入按比例采样分支
            num_nodes = int(len(all_nodes) * self.fraction_evaluate)  # 计算按比例采样的节点数
            sample_size = max(num_nodes, self.min_evaluate_nodes)  # 取按比例与最小评估节点数中的较大值

        candidates, probe, _ = self._filter_and_probe(all_nodes, sample_size)  # 获取候选节点探测节点与屏蔽列表

        if self.selection_mode == "window":  # 判断是否使用窗口调度
            node_ids = self._select_by_window(candidates, sample_size)  # 按窗口余量选择节点
        else:  # 进入随机采样分支
            pool = candidates[:]  # 拷贝候选列表作为采样池
            random.shuffle(pool)  # 打乱采样池顺序
            node_ids = pool[:sample_size]  # 截取前sample_size个节点作为本轮选择

        for nid in probe:  # 遍历探测节点列表
            if nid not in node_ids:  # 判断探测节点是否未被选择
                node_ids.append(nid)  # 将探测节点加入选择列表

        log(INFO, "configure_evaluate: Selected %s nodes (online=%s, mode=%s, blocked=%s, probe=%s)", len(node_ids), len(all_nodes), self.selection_mode, len(self._blocked), len(probe))  # 输出评估采样结果日志

        config["server-round"] = server_round  # 将轮次写入配置记录
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})  # 构造要发送的记录字典
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)  # 构造并返回评估消息列表

    def aggregate_evaluate(  # 定义评估阶段聚合方法
        self,  # 传入实例自身
        server_round: int,  # 传入当前轮次编号
        replies: Iterable[Message],  # 传入客户端回复消息集合
    ) -> MetricRecord | None:  # 声明返回聚合后的指标或空值
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)  # 检查回复并得到有效回复列表

        if valid_replies:  # 判断是否存在有效回复
            self._update_link_state_from_replies(valid_replies)  # 从有效回复中更新链路状态缓存

        metrics = None  # 初始化指标聚合结果占位变量
        if valid_replies:  # 判断是否存在有效回复用于聚合
            reply_contents = [msg.content for msg in valid_replies]  # 提取有效回复内容列表
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)  # 聚合评估指标得到汇总指标

        return metrics  # 返回聚合后的评估指标
