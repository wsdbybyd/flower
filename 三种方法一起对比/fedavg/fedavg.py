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
from fedavg.strategy import Strategy

from fedavg.strategy_utils import (
    aggregate_arrayrecords, 
    aggregate_metricrecords,
    validate_message_reply_consistency
)

from fedavg.exception import InconsistentMessageReplies

# 导入 Task 模块中的模型定义与蒸馏函数
# BigTeacherNet: 地面站大模型
# Net: 卫星端小模型 (Student)
from fedavg.task import (
    BigTeacherNet, 
    Net, 
    load_centralized_dataset, 
    load_centralized_dataset_train_test, # [新增] 用于加载测试集
    distill_teacher_to_student, 
    distill_student_to_teacher,
    test # [新增] 用于评估 Teacher
)


@dataclass
class _LinkState:
    """
    链路状态缓存结构体。
    用于记录每个客户端节点的通信状况，辅助调度决策。
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


class FedAvg(Strategy):
    """
    自定义 FedAvg 策略实现。
    支持可选的 Ground-Assisted Dual Distillation (地面辅助双向蒸馏)。
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
        deltaQ: int = 999999,            # 入网阈值
        deltaP: int = 999999,            # 退网阈值
        probe_blocked_k: int = 1,        # 每轮探测的被屏蔽节点数
        
        # --- [New] 是否启用双向蒸馏 (Control Variate) ---
        enable_dual_distillation: bool = False, 
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
        
        self.turnover_T = max(int(turnover_T), 1)
        self.deltaQ = int(deltaQ)
        self.deltaP = int(deltaP)
        self._connected_history: list[set[int]] = []
        self._comm_hist: dict[int, list[int]] = {}
        self._blocked: set[int] = set()
        self.probe_blocked_k = max(int(probe_blocked_k), 0)

        # --- 初始化地面站蒸馏模块 ---
        self.enable_dual_distillation = enable_dual_distillation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # [新增] Teacher 历史记录，用于绘制 Mentor 曲线
        self.teacher_metrics_history = {}

        # 仅在启用双向蒸馏实验时加载 Teacher 和数据
        if self.enable_dual_distillation:
            log(INFO, "🚀 Dual Distillation Enabled: Initializing BigTeacherNet...")
            self.teacher_model = BigTeacherNet()
            self.teacher_model.to(self.device)
            
            # 加载地面校准数据 (复用 task.py 中的函数)
            # [修改] 使用 load_centralized_dataset_train_test 同时获取测试集用于 Teacher 评估
            self.gs_dataloader, self.teacher_test_loader = load_centralized_dataset_train_test()
        else:
            log(INFO, "⏸️ Dual Distillation Disabled: Running Standard Mode.")
            self.teacher_model = None
            self.gs_dataloader = None
            self.teacher_test_loader = None
        
        # 默认 Teacher 学习率 (会被 ServerApp 覆盖)
        self.server_teacher_lr = 0.01

    def summary(self) -> None:
        log(INFO, "\t└── Summary: FedAvg(selection_mode=%s, enforce_comm=%s)", self.selection_mode, self.enforce_comm)
        log(INFO, "\t    Dual Distillation: %s", "ENABLED" if self.enable_dual_distillation else "DISABLED")

    # -----------------------------
    # --- 卫星调度与成员管理逻辑 ---
    # -----------------------------
    
    def _construct_messages(self, record: RecordDict, node_ids: list[int], message_type: MessageType) -> list[Message]:
        messages: list[Message] = []
        for node_id in node_ids:
            message = Message(content=record, message_type=message_type, dst_node_id=node_id)
            messages.append(message)
        return messages

    def _wait_for_nodes(self, grid: Grid, min_available_nodes: int) -> list[int]:
        while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
            log(INFO, "Waiting for nodes to connect: %d connected (minimum required: %d).", len(all_nodes), min_available_nodes)
            sleep(1)
        return all_nodes

    def _push_comm_hist(self, nid: int, ok: bool) -> None:
        hist = self._comm_hist.get(nid, [])
        hist.append(1 if ok else 0)
        if len(hist) > self.turnover_T:
            hist = hist[-self.turnover_T :]
        self._comm_hist[nid] = hist

    def _update_membership(self, nid: int) -> None:
        hist = self._comm_hist.get(nid, [])
        if not hist:
            return
        ok_cnt = sum(hist)
        fail_cnt = len(hist) - ok_cnt
        
        if (nid not in self._blocked) and (fail_cnt >= self.deltaP):
            self._blocked.add(nid)
            log(WARNING, "Node %d blocked (fail_cnt=%d in last %d rounds)", nid, fail_cnt, len(hist))
        
        if (nid in self._blocked) and (ok_cnt >= self.deltaQ):
            self._blocked.remove(nid)
            log(INFO, "Node %d unblocked (ok_cnt=%d in last %d rounds)", nid, ok_cnt, len(hist))

    def _select_by_window(self, node_ids: list[int], k: int) -> list[int]:
        def score(nid: int) -> tuple[int, float, float]:
            st = self._link_state.get(nid, _LinkState())
            tau = st.tau_d_s if st.tau_d_s is not None else -1e18
            margin = st.window_margin_s if st.window_margin_s is not None else -1e18
            comm_ok = bool(st.comm_ok) if st.comm_ok is not None else False
            feasible = (not self.enforce_comm) or comm_ok
            return (1 if feasible else 0, margin, tau)

        ranked = sorted(node_ids, key=score, reverse=True)
        
        if self.enforce_comm:
            feasible = [nid for nid in ranked if (self._link_state.get(nid, _LinkState()).comm_ok is True)]
            picked = feasible[:k]
            if len(picked) < k:
                for nid in ranked:
                    if nid not in picked:
                        picked.append(nid)
                    if len(picked) >= k:
                        break
            return picked
        return ranked[:k]

    def _filter_and_probe(self, all_nodes: list[int], sample_size: int) -> tuple[list[int], list[int], list[int]]:
        blocked = set(self._blocked)
        candidates = [nid for nid in all_nodes if nid not in blocked]
        blocked_list = list(blocked)
        
        if not candidates and blocked_list:
            candidates = blocked_list[:]
            
        probe: list[int] = []
        if blocked_list and self.probe_blocked_k > 0:
            k = min(self.probe_blocked_k, len(blocked_list))
            probe = blocked_list[:k]
        return candidates, probe, blocked_list

    def _update_link_state_from_replies(self, replies: list[Message]) -> None:
        for msg in replies:
            nid = msg.metadata.src_node_id
            content = msg.content
            if "metrics" not in content:
                continue
            md = _metricrecord_to_dict(content["metrics"])
            
            st = self._link_state.get(nid, _LinkState())
            
            st.tau_d_s = float(md.get("tau_d_s", st.tau_d_s or 0.0))
            st.window_margin_s = float(md.get("window_margin_s", st.window_margin_s or 0.0))
            st.t_total_s = float(md.get("T_total_s", st.t_total_s or 0.0))
            
            raw_comm_ok = md.get("comm_ok", None)
            if raw_comm_ok is not None:
                st.comm_ok = bool(int(raw_comm_ok))
            
            self._link_state[nid] = st
            
            if st.comm_ok is not None:
                self._push_comm_hist(nid, bool(st.comm_ok))
                self._update_membership(nid)

    def _check_and_log_replies(self, replies: Iterable[Message], is_train: bool) -> tuple[list[Message], list[Message]]:
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
        if self.fraction_train == 0.0:
            return []

        # ============================================================
        # 1. Ground Station Distillation (Teacher -> Global Student)
        # 仅在启用双向蒸馏时执行
        # ============================================================
        if self.enable_dual_distillation:
            log(INFO, f"[Round {server_round}] Ground Station: Distilling Teacher -> Global Student...")

            # 将 ArrayRecord 转换为可训练的 PyTorch 模型
            student_model = Net()
            student_model.load_state_dict(arrays.to_torch_state_dict())
            student_model.to(self.device)

            # 读取蒸馏配置
            gs_epochs = int(config.get("gs-distill-epochs", 1))
            gs_alpha = float(config.get("gs-distill-alpha", 0.5))
            gs_temp = float(config.get("gs-distill-temp", 3.0))
            gs_lr = float(config.get("lr", 0.01))

            # 执行蒸馏 (Stage 1)
            distill_teacher_to_student(
                self.teacher_model, student_model, self.gs_dataloader,
                self.device, gs_epochs, gs_lr, gs_alpha, gs_temp
            )

            # 更新分发参数
            arrays_to_send = ArrayRecord(student_model.state_dict())
        else:
            # 不进行蒸馏，直接分发
            arrays_to_send = arrays

        # ============================================================
        # 2. 节点选择与分发 (Satellite Selection)
        # ============================================================
        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)
        
        self._connected_history.append(set(all_nodes))
        if len(self._connected_history) > self.turnover_T:
            self._connected_history = self._connected_history[-self.turnover_T :]
        
        if self.selection_mode == "window" and self.ks_train is not None:
            sample_size = max(int(self.ks_train), self.min_train_nodes)
        else:
            num_nodes = int(len(all_nodes) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)

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

        log(INFO, "configure_train: Selected %s nodes (online=%s, mode=%s, blocked=%s, probe=%s)", 
            len(node_ids), len(all_nodes), self.selection_mode, len(self._blocked), len(probe))

        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays_to_send, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if arrays is None:
            return None, None

        # ============================================================
        # 3. Reverse Distillation (Student -> Teacher)
        # 仅在启用双向蒸馏时执行
        # ============================================================
        if self.enable_dual_distillation:
            log(INFO, f"[Round {server_round}] Ground Station: Reverse Distilling Global Student -> Teacher...")

            agg_student_model = Net()
            agg_student_model.load_state_dict(arrays.to_torch_state_dict())
            agg_student_model.to(self.device)

            distill_student_to_teacher(
                agg_student_model, self.teacher_model, self.gs_dataloader,
                self.device, epochs=1, lr=self.server_teacher_lr
            )
            
            # [新增] 蒸馏完后，立即评估 Teacher (Mentor) 的性能
            # 这样我们就能画出 Mentor 的精度曲线了
            log(INFO, f"[Round {server_round}] Evaluating Teacher (Mentor) performance...")
            t_loss, t_acc = test(self.teacher_model, self.teacher_test_loader, self.device)
            self.teacher_metrics_history[server_round] = {"accuracy": t_acc, "loss": t_loss}
            log(INFO, f"[Round {server_round}] Teacher Accuracy: {t_acc:.2%}")

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
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
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return metrics