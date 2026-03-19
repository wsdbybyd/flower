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

from fedml.exception import InconsistentMessageReplies
from fedml.strategy import Strategy
from fedml.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    validate_message_reply_consistency,
)

# 引入更新后的任务模块：使用 BigTeacherNet 对齐 FedAvg 实验的深度网络
from fedml.task import (
    Net,
    BigTeacherNet,
    build_dcal_loader,
    build_dglobal_loader,
    distill_teacher_to_student,
    reverse_distill_student_to_teacher,
    calibrate_bn_stats,
)


@dataclass
class _LinkState:
    """链路状态缓存（用于通信感知调度与成员管理）"""
    tau_d_s: float | None = None
    window_margin_s: float | None = None
    comm_ok: bool | None = None
    t_total_s: float | None = None


def _metricrecord_to_dict(mr: Any) -> dict[str, Any]:
    """将 MetricRecord 转成普通 dict，方便读取 key。"""
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


def _cfg_get(cfg: ConfigRecord, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except Exception:
        return default


def _cfg_bool(cfg: ConfigRecord, key: str, default: bool) -> bool:
    v = _cfg_get(cfg, key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _cfg_int(cfg: ConfigRecord, key: str, default: int) -> int:
    v = _cfg_get(cfg, key, default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _cfg_float(cfg: ConfigRecord, key: str, default: float) -> float:
    v = _cfg_get(cfg, key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


class FedMeta(Strategy):
    """
    FedMeta（FO-MAML/Reptile 风格） + 通信感知调度 + 动态成员管理
    + 双向蒸馏 Bidirectional KD (带有地面热启动 Teacher 支持)

    (1) Ground→Satellite：在每轮下发前，用 D_cal 让 student 向 teacher 蒸馏，得到更好的初始化再下发
    (2) Student→Teacher：每轮聚合后，用 D_global 让 teacher 向聚合 student 反向蒸馏更新
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

        # --- 调度参数 ---
        selection_mode: str = "random",  # "window" or "random"
        ks_train: int | None = None,
        ks_evaluate: int | None = None,
        enforce_comm: bool = False,
        turnover_T: int = 10,
        deltaQ: int = 999999,
        deltaP: int = 999999,
        probe_blocked_k: int = 1,

        # --- 双向蒸馏参数 ---
        kd_enable: bool = False,
        kd_alpha: float = 0.5,
        kd_temperature: float = 4.0,
        kd_cal_samples: int = 2048,
        kd_global_samples: int = 4096,
        kd_batch_size: int = 64,
        kd_forward_epochs: int = 1,
        kd_forward_lr: float = 0.05,
        kd_reverse_epochs: int = 1,
        kd_reverse_lr: float = 0.01,
        kd_cal_split: str = "train",
        kd_global_split: str = "train",
        kd_device: str | None = None,  # "cuda" / "cpu" / None(自动)
        
        # --- [新增] 接收来自 server.py 的热启动 Teacher 模型 ---
        initial_teacher_model: Optional[torch.nn.Module] = None,
    ) -> None:
        # --- Flower/FedMeta 基础 ---
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

        # --- 调度与链路状态 ---
        self.selection_mode = selection_mode
        self.ks_train = ks_train
        self.ks_evaluate = ks_evaluate
        self.enforce_comm = enforce_comm
        self._link_state: dict[int, _LinkState] = {}

        # --- 成员管理 ---
        self.turnover_T = max(int(turnover_T), 1)
        self.deltaQ = int(deltaQ)
        self.deltaP = int(deltaP)
        self._connected_history: list[set[int]] = []
        self._comm_hist: dict[int, list[int]] = {}
        self._blocked: set[int] = set()
        self.probe_blocked_k = max(int(probe_blocked_k), 0)

        # --- 双向蒸馏（Teacher & Datasets）---
        self._kd_enable = bool(kd_enable)
        self._kd_alpha = float(kd_alpha)
        self._kd_temperature = float(kd_temperature)
        self._kd_cal_samples = int(kd_cal_samples)
        self._kd_global_samples = int(kd_global_samples)
        self._kd_batch_size = int(kd_batch_size)
        self._kd_forward_epochs = int(kd_forward_epochs)
        self._kd_forward_lr = float(kd_forward_lr)
        self._kd_reverse_epochs = int(kd_reverse_epochs)
        self._kd_reverse_lr = float(kd_reverse_lr)
        self._kd_cal_split = str(kd_cal_split)
        self._kd_global_split = str(kd_global_split)
        self._kd_device_pref = kd_device  

        # [新增] 初始化 Teacher 为传入的热启动模型
        self.teacher_model = initial_teacher_model
        
        self._dcal_loader = None
        self._dglobal_loader = None

        # 记录本轮 forward KD loss
        self._last_kd_forward_round: Optional[int] = None
        self._last_kd_forward_loss: Optional[float] = None

    # -------------------------------------------------------------------------
    # 基础工具
    # -------------------------------------------------------------------------

    def summary(self) -> None:
        log(INFO, "\t└── Summary: FedMeta(selection_mode=%s, enforce_comm=%s, kd_enable=%s)",
            self.selection_mode, self.enforce_comm, self._kd_enable)
        if self.teacher_model is not None:
            log(INFO, "\t    [Initialized with Pre-trained Ground Teacher]")

    def _construct_messages(self, record: RecordDict, node_ids: list[int], message_type: MessageType) -> list[Message]:
        messages: list[Message] = []
        for node_id in node_ids:
            messages.append(Message(content=record, message_type=message_type, dst_node_id=node_id))
        return messages

    def _wait_for_nodes(self, grid: Grid, min_available_nodes: int) -> list[int]:
        while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
            log(INFO, "Waiting for nodes to connect: %d connected (minimum required: %d).",
                len(all_nodes), min_available_nodes)
            sleep(1)
        return all_nodes

    # -------------------------------------------------------------------------
    # 成员管理 / 调度
    # -------------------------------------------------------------------------

    def _push_comm_hist(self, nid: int, ok: bool) -> None:
        hist = self._comm_hist.get(nid, [])
        hist.append(1 if ok else 0)
        if len(hist) > self.turnover_T:
            hist = hist[-self.turnover_T:]
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
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(INFO, "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies), len(error_replies))

        if valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies

    # -------------------------------------------------------------------------
    # 双向蒸馏：初始化/数据准备
    # -------------------------------------------------------------------------

    def _kd_device(self) -> torch.device:
        if self._kd_device_pref is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pref = str(self._kd_device_pref).strip().lower()
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _kd_read_overrides(self, cfg: ConfigRecord) -> None:
        """允许从 ConfigRecord 覆盖（便于从 pyproject/server 动态配置）"""
        self._kd_enable = _cfg_bool(cfg, "kd-enable", self._kd_enable)
        self._kd_alpha = _cfg_float(cfg, "kd-alpha", self._kd_alpha)
        self._kd_temperature = _cfg_float(cfg, "kd-temperature", self._kd_temperature)
        self._kd_cal_samples = _cfg_int(cfg, "kd-cal-samples", self._kd_cal_samples)
        self._kd_global_samples = _cfg_int(cfg, "kd-global-samples", self._kd_global_samples)
        self._kd_batch_size = _cfg_int(cfg, "kd-batch-size", self._kd_batch_size)
        self._kd_forward_epochs = _cfg_int(cfg, "kd-forward-epochs", self._kd_forward_epochs)
        self._kd_forward_lr = _cfg_float(cfg, "kd-forward-lr", self._kd_forward_lr)
        self._kd_reverse_epochs = _cfg_int(cfg, "kd-reverse-epochs", self._kd_reverse_epochs)
        self._kd_reverse_lr = _cfg_float(cfg, "kd-reverse-lr", self._kd_reverse_lr)

    def _ensure_kd_objects(self) -> None:
        """延迟初始化 teacher 与 D_cal/D_global"""
        if not self._kd_enable:
            return

        # 如果没有传入热启动的 teacher，则回退到重新初始化 BigTeacherNet
        if self.teacher_model is None:
            self.teacher_model = BigTeacherNet()

        if self._dcal_loader is None:
            self._dcal_loader = build_dcal_loader(
                num_samples=self._kd_cal_samples,
                batch_size=self._kd_batch_size,
                split=self._kd_cal_split,
            )

        if self._dglobal_loader is None:
            self._dglobal_loader = build_dglobal_loader(
                num_samples=self._kd_global_samples,
                batch_size=self._kd_batch_size,
                split=self._kd_global_split,
            )

    # -------------------------------------------------------------------------
    # Strategy override
    # -------------------------------------------------------------------------

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []

        self._kd_read_overrides(config)

        # ============================================================
        # Ground→Satellite 正向蒸馏：teacher -> student（下发前）
        # ============================================================
        arrays_to_send = arrays
        if self._kd_enable:
            try:
                self._ensure_kd_objects()
                device = self._kd_device()

                student = Net()
                student.load_state_dict(arrays.to_torch_state_dict())

                fwd_loss = distill_teacher_to_student(
                    teacher=self.teacher_model,  # type: ignore[arg-type]
                    student=student,
                    loader=self._dcal_loader,    # type: ignore[arg-type]
                    device=device,
                    alpha=self._kd_alpha,
                    temperature=self._kd_temperature,
                    lr=self._kd_forward_lr,
                    epochs=self._kd_forward_epochs,
                )

                arrays_to_send = ArrayRecord(student.state_dict())
                self._last_kd_forward_round = server_round
                self._last_kd_forward_loss = float(fwd_loss)

                log(INFO, "KD Forward (round=%s): loss=%.6f (alpha=%.3f, T=%.2f, epochs=%d)",
                    server_round, fwd_loss, self._kd_alpha, self._kd_temperature, self._kd_forward_epochs)

            except Exception as e:
                log(WARNING, "KD Forward failed, fallback to raw arrays. err=%s", str(e))
                arrays_to_send = arrays

        # ============================================================
        # 节点选择与分发
        # ============================================================
        all_nodes = self._wait_for_nodes(grid, self.min_available_nodes)

        self._connected_history.append(set(all_nodes))
        if len(self._connected_history) > self.turnover_T:
            self._connected_history = self._connected_history[-self.turnover_T:]

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

        # v5 Mixed 模式改为用全局 student 模型（arrays）作为 APSKD snapshot
        # 不再需要下发 BigTeacherNet，减少通信开销
        train_mode = str(_cfg_get(config, "client-train-mode", "fomaml")).strip().lower()
        record_dict: dict = {self.arrayrecord_key: arrays_to_send, self.configrecord_key: config}

        record = RecordDict(record_dict)
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # 1) 检查回复并更新链路状态
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        # 2) 聚合客户端更新
        arrays_out: ArrayRecord | None = None
        metrics_out: MetricRecord | None = None

        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays_out = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics_out = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if arrays_out is None:
            return None, None

        # 3) BN 统计量校准：聚合平均后 running_mean/var 失效，用 D_cal 重新估计
        #    MobileNetV2 有 27 个 BN 层，不校准会导致精度大幅下滑
        try:
            self._ensure_kd_objects()   # 确保 _dcal_loader 已初始化
            device = self._kd_device()
            student_for_bn = Net()
            student_for_bn.load_state_dict(arrays_out.to_torch_state_dict())
            student_for_bn = calibrate_bn_stats(
                student_for_bn,
                loader=self._dcal_loader,
                device=device,
                num_batches=20,
            )
            arrays_out = ArrayRecord(student_for_bn.state_dict())
            log(INFO, "BN Calibration done (round=%s)", server_round)
        except Exception as e:
            log(WARNING, "BN Calibration failed, using raw aggregated stats. err=%s", str(e))

        # 4) Student→Teacher 反向蒸馏：用聚合 student 更新 teacher
        if self._kd_enable:
            try:
                self._ensure_kd_objects()
                device = self._kd_device()

                student_aggr = Net()
                student_aggr.load_state_dict(arrays_out.to_torch_state_dict())

                rev_loss = reverse_distill_student_to_teacher(
                    teacher=self.teacher_model,   # type: ignore[arg-type]
                    student=student_aggr,
                    loader=self._dglobal_loader,  # type: ignore[arg-type]
                    device=device,
                    alpha=self._kd_alpha,
                    temperature=self._kd_temperature,
                    lr=self._kd_reverse_lr,
                    epochs=self._kd_reverse_epochs,
                )

                if metrics_out is None:
                    metrics_out = MetricRecord({})
                metrics_out["kd_reverse_loss"] = float(rev_loss)

                if self._last_kd_forward_round == server_round and self._last_kd_forward_loss is not None:
                    metrics_out["kd_forward_loss"] = float(self._last_kd_forward_loss)

                metrics_out["kd_alpha"] = float(self._kd_alpha)
                metrics_out["kd_temperature"] = float(self._kd_temperature)

                log(INFO, "KD Reverse (round=%s): loss=%.6f (epochs=%d)",
                    server_round, rev_loss, self._kd_reverse_epochs)

            except Exception as e:
                log(WARNING, "KD Reverse failed (teacher not updated). err=%s", str(e))

        return arrays_out, metrics_out

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
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

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if valid_replies:
            self._update_link_state_from_replies(valid_replies)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        return metrics