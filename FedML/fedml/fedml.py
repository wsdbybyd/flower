from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from logging import INFO, WARNING
from time import sleep
from typing import Any, Optional

import torch
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid
from fedml.strategy import Strategy


from fedml.strategy_utils import (
    aggregate_arrayrecords, 
    aggregate_metricrecords,
    validate_message_reply_consistency
)

from fedml.exception import InconsistentMessageReplies

# 导入 Task 模块 (不再需要 TeacherNet 和 Distill 函数)
from fedml.task import Net


@dataclass
class _LinkState:
    """
    链路状态缓存结构体。
    用于记录每个客户端节点的通信状况，辅助调度决策。
    
    Attributes:
        tau_d_s: 当前链路的最大允许时延窗口 (秒)
        window_margin_s: 时间窗口余量 (窗口 - 实际时延)，正值表示通信可行
        comm_ok: 通信是否成功的布尔标志
        t_total_s: 实际传输总时延
    """
    tau_d_s: float | None = None
    window_margin_s: float | None = None
    comm_ok: bool | None = None
    t_total_s: float | None = None


def _metricrecord_to_dict(mr: Any) -> dict[str, Any]:
    """辅助函数：将 MetricRecord 对象转换为标准 Python 字典以便读取。"""
    if mr is None:
        return {}
    try:
        return dict(mr)
    except Exception:
        pass
    for attr in ("to_dict", "as_dict", "dict"):
        if hasattr(mr, attr):
            try:
                return getattr(mr, attr)()
            except Exception:
                pass
    try:
        out = {}
        for k in mr:
            out[k] = mr[k]
        return out
    except Exception:
        return {}


class FedMeta(Strategy):
    """
    自定义 FedMeta (Federated Meta-Learning) 策略实现。
    
    主要功能：
    1. Meta-Learning Aggregation:
       - 负责分发全局元模型 (Meta-Model)。
       - 聚合客户端回传的更新后参数 (对于 FO-MAML/Reptile，这等价于聚合 Meta-Gradient)。
       
    2. Satellite Communication-Aware Scheduling (卫星通信感知调度):
       - 基于物理链路预算计算窗口余量 (Window Margin)。
       - 优先选择通信状况良好 (Margin > 0) 的卫星节点。
       - 动态成员管理 (Turnover): 自动屏蔽连续失败的节点，并在其恢复后重新接纳。
    """
    
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Callable[[list[RecordDict], str], MetricRecord] | None = None,
        evaluate_metrics_aggr_fn: Callable[[list[RecordDict], str], MetricRecord] | None = None,
        
        # --- 自定义调度参数 ---
        selection_mode: str = "random",  # "window" 或 "random"
        ks_train: int | None = None,     # 训练采样数
        ks_evaluate: int | None = None,  # 评估采样数
        enforce_comm: bool = False,      # 是否强制剔除超时节点
        turnover_T: int = 10,            # 历史记录窗口长度
        deltaQ: int = 999999,            # 入网阈值 (恢复所需的连续成功次数)
        deltaP: int = 999999,            # 退网阈值 (屏蔽所需的连续失败次数)
        probe_blocked_k: int = 1,        # 每轮探测的被屏蔽节点数
    ) -> None:
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.train_metrics_aggr_fn = train_metrics_aggr_fn or aggregate_metricrecords
        self.evaluate_metrics_aggr_fn = evaluate_metrics_aggr_fn or aggregate_metricrecords

        if self.fraction_evaluate == 0.0:
            self.min_evaluate_nodes = 0
            log(WARNING, "fraction_evaluate is set to 0.0. Federated evaluate will be disabled.")

        # --- 初始化调度与链路状态 ---
        self.selection_mode = selection_mode
        self.ks_train = ks_train
        self.ks_evaluate = ks_evaluate
        self.enforce_comm = enforce_comm
        self._link_state: dict[int, _LinkState] = {}
        
        # 成员管理状态初始化
        self.turnover_T = max(int(turnover_T), 1)
        self.deltaQ = int(deltaQ)
        self.deltaP = int(deltaP)
        self._connected_history: list[set[int]] = []
        self._comm_hist: dict[int, list[int]] = {}  # 记录节点历史通信成功/失败状态
        self._blocked: set[int] = set()             # 当前被屏蔽的节点集合
        self.probe_blocked_k = max(int(probe_blocked_k), 0)

        # 注意：此处移除了原有的 self.teacher_model 和 self.gs_dataloader

    def summary(self) -> None:
        log(INFO, "\t└── Summary: FedMeta(selection_mode=%s, enforce_comm=%s)", self.selection_mode, self.enforce_comm)
        log(INFO, "\t    Ground-Assisted LEO FL with FO-MAML/Reptile Enabled.")

    # -----------------------------
    # --- 卫星调度与成员管理逻辑 ---
    # -----------------------------
    
    def _construct_messages(self, record: RecordDict, node_ids: list[int], message_type: MessageType) -> list[Message]:
        """构造发送给选定客户端的消息列表。"""
        messages: list[Message] = []
        for node_id in node_ids:
            message = Message(content=record, message_type=message_type, dst_node_id=node_id)
            messages.append(message)
        return messages

    def _wait_for_nodes(self, grid: Grid, min_available_nodes: int) -> list[int]:
        """阻塞直到有足够数量的节点上线。"""
        while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
            log(INFO, "Waiting for nodes to connect: %d connected (minimum required: %d).", len(all_nodes), min_available_nodes)
            sleep(1)
        return all_nodes

    def _push_comm_hist(self, nid: int, ok: bool) -> None:
        """更新节点的通信历史记录 (滑动窗口)。"""
        hist = self._comm_hist.get(nid, [])
        hist.append(1 if ok else 0)
        if len(hist) > self.turnover_T:
            hist = hist[-self.turnover_T :]
        self._comm_hist[nid] = hist

    def _update_membership(self, nid: int) -> None:
        """
        动态成员管理 (Dynamic Membership Management)。
        根据节点的近期通信成功率，决定将其移入或移出屏蔽列表 (Blocked Set)。
        """
        hist = self._comm_hist.get(nid, [])
        if not hist:
            return
        ok_cnt = sum(hist)
        fail_cnt = len(hist) - ok_cnt
        
        # 连续失败次数过多 -> 屏蔽
        if (nid not in self._blocked) and (fail_cnt >= self.deltaP):
            self._blocked.add(nid)
            log(WARNING, "Node %d blocked (fail_cnt=%d in last %d rounds)", nid, fail_cnt, len(hist))
        
        # 连续成功次数达标 -> 恢复
        if (nid in self._blocked) and (ok_cnt >= self.deltaQ):
            self._blocked.remove(nid)
            log(INFO, "Node %d unblocked (ok_cnt=%d in last %d rounds)", nid, ok_cnt, len(hist))

    def _select_by_window(self, node_ids: list[int], k: int) -> list[int]:
        """
        基于窗口余量 (Window Margin) 的节点选择策略。
        优先选择 Margin 大（通信时间充裕）的节点。
        """
        def score(nid: int) -> tuple[int, float, float]:
            st = self._link_state.get(nid, _LinkState())
            tau = st.tau_d_s if st.tau_d_s is not None else -1e18
            margin = st.window_margin_s if st.window_margin_s is not None else -1e18
            comm_ok = bool(st.comm_ok) if st.comm_ok is not None else False
            feasible = (not self.enforce_comm) or comm_ok
            # 排序优先级: 1. 是否可行 (feasible)  2. 窗口余量 (margin)
            return (1 if feasible else 0, margin, tau)

        ranked = sorted(node_ids, key=score, reverse=True)
        
        if self.enforce_comm:
            # 强制模式：只选 comm_ok=True 的节点
            feasible = [nid for nid in ranked if (self._link_state.get(nid, _LinkState()).comm_ok is True)]
            picked = feasible[:k]
            # 如果可行节点不足 k 个，尝试从剩余节点中补充（虽可能超时）
            if len(picked) < k:
                for nid in ranked:
                    if nid not in picked:
                        picked.append(nid)
                    if len(picked) >= k:
                        break
            return picked
        return ranked[:k]

    def _filter_and_probe(self, all_nodes: list[int], sample_size: int) -> tuple[list[int], list[int], list[int]]:
        """
        将节点分为 候选集合 (Candidates) 和 探测集合 (Probe)。
        """
        blocked = set(self._blocked)
        candidates = [nid for nid in all_nodes if nid not in blocked]
        blocked_list = list(blocked)
        
        # 如果所有节点都被屏蔽，强制使用屏蔽节点
        if not candidates and blocked_list:
            candidates = blocked_list[:]
            
        probe: list[int] = []
        if blocked_list and self.probe_blocked_k > 0:
            k = min(self.probe_blocked_k, len(blocked_list))
            probe = blocked_list[:k]
        return candidates, probe, blocked_list

    def _update_link_state_from_replies(self, replies: list[Message]) -> None:
        """
        从客户端回传的 Metrics 中提取物理层链路状态，更新本地缓存和成员状态。
        """
        for msg in replies:
            nid = msg.metadata.src_node_id
            content = msg.content
            if "metrics" not in content:
                continue
            md = _metricrecord_to_dict(content["metrics"])
            
            st = self._link_state.get(nid, _LinkState())
            
            # 更新 LinkState 缓存
            st.tau_d_s = float(md.get("tau_d_s", st.tau_d_s or 0.0))
            st.window_margin_s = float(md.get("window_margin_s", st.window_margin_s or 0.0))
            st.t_total_s = float(md.get("T_total_s", st.t_total_s or 0.0))
            
            raw_comm_ok = md.get("comm_ok", None)
            if raw_comm_ok is not None:
                st.comm_ok = bool(int(raw_comm_ok))
            
            self._link_state[nid] = st
            
            # 更新历史记录并触发成员管理检查
            if st.comm_ok is not None:
                self._push_comm_hist(nid, bool(st.comm_ok))
                self._update_membership(nid)

    def _check_and_log_replies(self, replies: Iterable[Message], is_train: bool) -> tuple[list[Message], list[Message]]:
        """检查回复消息的有效性并记录日志。"""
        valid_replies = []
        error_replies = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)
        
        log(INFO, "%s: Received %s results and %s failures", "aggregate_train" if is_train else "aggregate_evaluate", len(valid_replies), len(error_replies))
        
        if valid_replies:
             validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )
        return valid_replies, error_replies

    # -------------------------------------------------------------------------
    # --- Strategy 核心方法重写 (Override) ---
    # -------------------------------------------------------------------------

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """
        [Meta-Train Phase] 配置训练阶段。
        
        关键逻辑：
        1. 节点调度：根据物理链路窗口选择卫星节点。
        2. 分发元模型：将 Global Meta-Model 分发给选定的节点。
        """
        if self.fraction_train == 0.0:
            return []

        # ============================================================
        # 节点选择与分发 (Satellite Selection)
        # ============================================================
        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)
        
        # 记录在线历史
        self._connected_history.append(set(all_nodes))
        if len(self._connected_history) > self.turnover_T:
            self._connected_history = self._connected_history[-self.turnover_T :]
        
        # 计算本轮需要的采样数
        if self.selection_mode == "window" and self.ks_train is not None:
            sample_size = max(int(self.ks_train), self.min_train_nodes)
        else:
            num_nodes = int(len(all_nodes) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)

        # 过滤掉被屏蔽的节点，并尝试探测部分屏蔽节点
        candidates, probe, _ = self._filter_and_probe(all_nodes, sample_size)

        # 执行采样
        if self.selection_mode == "window":
            # 基于窗口余量排序选择
            node_ids = self._select_by_window(candidates, sample_size)
        else:
            # 随机选择
            pool = candidates[:]
            random.shuffle(pool)
            node_ids = pool[:sample_size]

        # 将探测节点加入选择列表
        for nid in probe:
            if nid not in node_ids:
                node_ids.append(nid)

        log(INFO, "configure_train: Selected %s nodes (online=%s, mode=%s, blocked=%s, probe=%s)", 
            len(node_ids), len(all_nodes), self.selection_mode, len(self._blocked), len(probe))

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        [Meta-Update Phase] 聚合训练结果。
        
        关键逻辑：
        1. 接收卫星回传的参数。在 FO-MAML 中，卫星回传的是经过 (Support Set微调 + Query Set求导) 更新后的参数。
        2. 服务器端求平均。这等价于对所有 Task 的 Meta-Gradient 求平均并更新 Global Meta-Model。
        """
        # 1. 检查回复并更新链路状态 (用于后续调度决策)
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        # 2. 执行聚合 (Reptile / FO-MAML Update)
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if arrays is None:
            return None, None

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """配置评估阶段 (Meta-Testing)。"""
        if self.fraction_evaluate == 0.0:
            return []

        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)

        if self.selection_mode == "window" and self.ks_evaluate is not None:
            sample_size = max(int(self.ks_evaluate), self.min_evaluate_nodes)
        else:
            num_nodes = int(len(all_nodes) * self.fraction_evaluate)
            sample_size = max(num_nodes, self.min_evaluate_nodes)

        candidates, probe, _ = self._filter_and_probe(all_nodes, sample_size)

        if self.selection_mode == "window":
            node_ids = self._select_by_window(candidates, sample_size)
        else:
            pool = candidates[:]
            random.shuffle(pool)
            node_ids = pool[:sample_size]

        for nid in probe:
            if nid not in node_ids:
                node_ids.append(nid)

        log(INFO, "configure_evaluate: Selected %s nodes", len(node_ids))

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """聚合评估结果。"""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return metrics