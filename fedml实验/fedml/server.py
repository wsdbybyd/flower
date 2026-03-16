"""
server.py — 支持 6 种 FL 方法的顺序对比实验

运行顺序：
  1. fedavg   — FedAvg baseline
  2. fedprox  — FedProx
  3. scaffold — SCAFFOLD
  4. fedkd    — FedKD（server 下发 BigTeacherNet 软标签）
  5. fedmeta  — FO-MAML
  6. ours     — FO-MAML + APSKD + 双向 KD（本文方法）

所有方法：
  - 共享同一热启动初始参数（ground pre-distillation）
  - 评估统一使用客户端本地 Non-IID meta-evaluation（公平对比）
  - 最终绘制 6 条 Accuracy 曲线 + 6 条 Train Loss 曲线
"""
import os, json, copy, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from logging import INFO, WARNING

# python-docx：生成准确率对比表
from docx import Document as DocxDocument
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from fedml.fedml import FedMeta
from fedml.task import (
    Net, BigTeacherNet,
    build_dcal_loader, build_dglobal_loader,
    distill_teacher_to_student, reverse_distill_student_to_teacher,
    load_centralized_dataset_train_test,
    train_centralized, distill_centralized, test,
)

app = ServerApp()


# =============================================================================
# 地面预热（所有方法共用热启动初始参数）
# =============================================================================
def perform_ground_warmup(device):
    log(INFO, "=" * 60)
    log(INFO, "🚀 [预处理] 地面蒸馏热启动 (Ground Warm-up)")
    log(INFO, "=" * 60)
    trainloader, testloader = load_centralized_dataset_train_test()
    teacher = BigTeacherNet(); student = Net()
    log(INFO, "1. 训练 Teacher...")
    train_centralized(teacher, trainloader, epochs=2, lr=0.01, device=device)
    log(INFO, "2. 蒸馏 Student (Teacher → Student)...")
    distill_centralized(student, teacher, trainloader,
                        epochs=2, lr=0.01, device=device, temp=2.0, alpha=0.5)
    loss, acc = test(student, testloader, device)
    log(INFO, "✅ 热启动完成，Student 初始精度: %.2f%%", acc * 100)
    return student, teacher


# =============================================================================
# 绘图工具
# =============================================================================
# 6 种方法的颜色 / 线型 / 标签
MODES = ["fedavg", "fedprox", "scaffold", "fedkd", "fedmeta", "ours"]
COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#1f77b4", "#2ca02c"]
STYLES = ["--",      "--",      "--",      "-.",      "-",       "-"]
LABELS = [
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "FedKD",
    "FedMeta (FO-MAML)",
    "Ours (FO-MAML+APSKD+KD)",
]


def _load_json_safe(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_comparison(metric_key: str, ylabel: str, title: str, filename: str):
    plt.figure(figsize=(12, 6)); plt.style.use("default")
    count = 0
    for mode, color, style, label in zip(MODES, COLORS, STYLES, LABELS):
        data = _load_json_safe(f"metrics_{mode}.json")
        if data is None: continue
        rounds = sorted(int(k) for k in data)
        vals   = [data[str(r)].get(metric_key) for r in rounds]
        valid  = [(r, v) for r, v in zip(rounds, vals) if v is not None]
        if not valid: continue
        rs, vs = zip(*valid)
        plt.plot(rs, vs, label=label, color=color, linestyle=style,
                 linewidth=2, marker="o", markersize=3)
        count += 1
    if count == 0: return
    plt.title(title, fontsize=13)
    plt.xlabel("Round", fontsize=12); plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend(fontsize=10)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()
    log(INFO, "Saved: %s", filename)


def plot_all():
    plot_comparison(
        "accuracy", "Accuracy (Post-Adaptation, Local Non-IID)",
        "Test Accuracy: 6-Method Comparison (Dirichlet α=0.1)",
        "comparison_accuracy.png")
    plot_comparison(
        "train_loss", "Train Loss (Aggregated)",
        "Train Loss: 6-Method Comparison",
        "comparison_train_loss.png")
    plot_comparison(
        "loss", "Eval Loss (Post-Adaptation)",
        "Eval Loss: 6-Method Comparison",
        "comparison_eval_loss.png")


# =============================================================================
# 准确率对比表生成
# =============================================================================

# 方法元信息（顺序 = 表格行顺序）
TABLE_METHODS = [
    {"key": "fedavg",   "name": "FedAvg",             "ref": "(McMahan et al., 2017)"},
    {"key": "fedprox",  "name": "FedProx",             "ref": "(Li et al., 2020)"},
    {"key": "scaffold", "name": "SCAFFOLD",            "ref": "(Karimireddy et al., 2020)"},
    {"key": "fedkd",    "name": "FedKD",               "ref": "(Wu et al., 2022)"},
    {"key": "fedmeta",  "name": "FedMeta (FO-MAML)",   "ref": "(Finn et al., 2017)"},
    {"key": "ours",     "name": "Ours (Full)",          "ref": ""},
]

# 数据集列（可扩展：实验跑完后在此添加更多数据集的 json 前缀）
TABLE_DATASETS = [
    {"label": "CIFAR-10",      "json_prefix": ""},          # metrics_{mode}.json
    {"label": "CIFAR-100",     "json_prefix": "_cifar100"},  # metrics_{mode}_cifar100.json
    {"label": "EuroSAT",       "json_prefix": "_eurosat"},
    {"label": "Fashion-MNIST", "json_prefix": "_fmnist"},
]

# 颜色常量
_C_HEADER      = RGBColor(0x1F, 0x4E, 0x79)   # 深蓝  表头背景
_C_SUBHDR      = RGBColor(0x2E, 0x75, 0xB6)   # 中蓝  副表头
_C_OURS        = RGBColor(0xE2, 0xEF, 0xDA)   # 浅绿  Ours 行
_C_BEST        = RGBColor(0xFF, 0xF2, 0xCC)   # 浅黄  最优值
_C_PENDING     = RGBColor(0xF2, 0xF2, 0xF2)   # 浅灰  待填充
_C_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
_C_TEXT_HEADER = RGBColor(0xFF, 0xFF, 0xFF)   # 白色  表头文字
_C_TEXT_OURS   = RGBColor(0x1F, 0x4E, 0x79)   # 深蓝  Ours 行文字
_C_TEXT_BEST   = RGBColor(0x7F, 0x3F, 0x00)   # 深棕  最优值文字
_C_TEXT_PEND   = RGBColor(0x99, 0x99, 0x99)   # 灰色  待填充文字


def _set_cell_bg(cell, rgb: RGBColor):
    """设置单元格背景色（xml 层操作）"""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    hex_color = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    # 移除旧的 shd
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    tcPr.append(shd)


def _cell_para(cell, text: str, *,
               bold=False, italic=False, font_size=10,
               color: RGBColor = RGBColor(0, 0, 0),
               align=WD_ALIGN_PARAGRAPH.CENTER):
    """清空单元格并写入格式化段落"""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(font_size)
    run.font.color.rgb = color
    run.font.name = "Times New Roman"
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def _load_metrics_for_dataset(mode_key: str, json_prefix: str) -> dict:
    """
    读取某个方法 + 某个数据集的 JSON 文件。
    文件命名规则：
      CIFAR-10 (prefix="")      → metrics_{mode}.json
      其他数据集 (prefix=XX)    → metrics_{mode}{prefix}.json
    返回已加载的 dict，或 None（文件不存在）。
    """
    fname = f"metrics_{mode_key}{json_prefix}.json"
    if not os.path.exists(fname):
        return None
    try:
        with open(fname) as f:
            return json.load(f)
    except Exception:
        return None


def _compute_last_n_mean(data: dict, n: int = 10) -> float:
    """取最后 n 轮 accuracy 的均值"""
    rounds = sorted(int(k) for k in data)
    last   = rounds[-n:] if len(rounds) >= n else rounds
    accs   = [data[str(r)]["accuracy"] for r in last
              if data[str(r)].get("accuracy") is not None]
    return sum(accs) / len(accs) if accs else None


def generate_accuracy_table(output_path: str = "accuracy_table.docx",
                             last_n: int = 10,
                             dirichlet_alpha: float = 0.1) -> None:
    """
    扫描当前目录下的 metrics_*.json，生成准确率对比表并保存为 Word 文档。

    表格结构：
      行 = 6 种方法
      列 = 4 个数据集（每列 1 个数字：last_n 轮均值 %）

    视觉规则：
      - 深蓝表头 / 中蓝副标题行
      - 浅绿高亮 Ours 行
      - 浅黄高亮每列最优值
      - 浅灰 + 灰字 表示尚未完成的实验（待填充）

    扩展方式（增加新数据集）：
      在 TABLE_DATASETS 列表中添加新条目即可，无需修改其他代码。
    """
    log(INFO, "[Table] Generating accuracy comparison table → %s", output_path)

    # ── Step 1: 收集所有数据 ─────────────────────────────────────────
    # results[method_key][dataset_label] = float(acc%) or None
    results = {}
    for m in TABLE_METHODS:
        results[m["key"]] = {}
        for ds in TABLE_DATASETS:
            data = _load_metrics_for_dataset(m["key"], ds["json_prefix"])
            if data is None:
                results[m["key"]][ds["label"]] = None
            else:
                val = _compute_last_n_mean(data, n=last_n)
                results[m["key"]][ds["label"]] = round(val * 100, 2) if val else None

    # ── Step 2: 找每列最优值 ─────────────────────────────────────────
    best_per_ds = {}
    for ds in TABLE_DATASETS:
        vals = [results[m["key"]][ds["label"]] for m in TABLE_METHODS
                if results[m["key"]][ds["label"]] is not None]
        best_per_ds[ds["label"]] = max(vals) if vals else None

    # ── Step 3: 建 Word 文档 ─────────────────────────────────────────
    doc = DocxDocument()

    # 页面设置：A4，窄边距
    section = doc.sections[0]
    section.page_width  = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = section.right_margin   = Cm(2.0)
    section.top_margin  = section.bottom_margin  = Cm(2.0)

    # 标题
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(
        "Table 1: Accuracy Comparison of Federated Learning Methods"
    )
    title_run.bold      = True
    title_run.font.size = Pt(13)
    title_run.font.name = "Times New Roman"

    # 副标题
    sub_para = doc.add_paragraph()
    sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_para.add_run(
        f"Non-IID setting (Dirichlet \u03b1 = {dirichlet_alpha})  \u00b7  "
        f"Post-Adaptation Accuracy (%)  \u00b7  Last-{last_n} Round Mean"
    )
    sub_run.italic     = True
    sub_run.font.size  = Pt(10)
    sub_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    sub_run.font.name  = "Times New Roman"

    doc.add_paragraph()  # 空行

    # ── Step 4: 建表格（行数 = 2头 + 6方法，列数 = 1方法列 + 4数据集列）
    n_cols   = 1 + len(TABLE_DATASETS)
    n_rows   = 2 + len(TABLE_METHODS)
    table    = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"

    # 列宽（单位 Cm）：方法列宽些，数据集列均分剩余
    col_widths_cm = [5.0] + [3.2] * len(TABLE_DATASETS)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Cm(col_widths_cm[i])

    # ── Row 0: 主表头 ─────────────────────────────────────────────────
    hdr0 = table.rows[0]
    _set_cell_bg(hdr0.cells[0], _C_HEADER)
    _cell_para(hdr0.cells[0], "Method",
               bold=True, font_size=11, color=_C_TEXT_HEADER)

    for j, ds in enumerate(TABLE_DATASETS):
        _set_cell_bg(hdr0.cells[j + 1], _C_HEADER)
        _cell_para(hdr0.cells[j + 1], ds["label"],
                   bold=True, font_size=11, color=_C_TEXT_HEADER)

    # ── Row 1: 副表头（说明行）──────────────────────────────────────────
    hdr1 = table.rows[1]
    _set_cell_bg(hdr1.cells[0], _C_SUBHDR)
    _cell_para(hdr1.cells[0],
               f"\u03b1 = {dirichlet_alpha}",
               italic=True, font_size=9, color=_C_TEXT_HEADER)

    for j in range(len(TABLE_DATASETS)):
        _set_cell_bg(hdr1.cells[j + 1], _C_SUBHDR)
        _cell_para(hdr1.cells[j + 1], "Acc% (last-10 mean)",
                   italic=True, font_size=9, color=_C_TEXT_HEADER)

    # ── Rows 2+: 数据行 ───────────────────────────────────────────────
    for i, m in enumerate(TABLE_METHODS):
        row      = table.rows[i + 2]
        is_ours  = (m["key"] == "ours")
        row_bg   = _C_OURS if is_ours else _C_WHITE
        name_str = f"{m['name']}  {m['ref']}".strip() if m["ref"] else m["name"]

        # 方法名列
        _set_cell_bg(row.cells[0], row_bg)
        _cell_para(row.cells[0], name_str,
                   bold=is_ours, font_size=10,
                   color=_C_TEXT_OURS if is_ours else RGBColor(0, 0, 0),
                   align=WD_ALIGN_PARAGRAPH.LEFT)

        # 各数据集列
        for j, ds in enumerate(TABLE_DATASETS):
            val      = results[m["key"]][ds["label"]]
            is_best  = (val is not None and val == best_per_ds[ds["label"]])

            if val is None:
                # 待填充
                _set_cell_bg(row.cells[j + 1], _C_PENDING)
                _cell_para(row.cells[j + 1], "\u2014",
                           font_size=10, color=_C_TEXT_PEND)
            elif is_best:
                # 最优值：黄底 + 加粗 + 深棕色
                _set_cell_bg(row.cells[j + 1], _C_BEST)
                _cell_para(row.cells[j + 1], f"{val:.2f}%",
                           bold=True, font_size=10, color=_C_TEXT_BEST)
            else:
                _set_cell_bg(row.cells[j + 1], row_bg)
                _cell_para(row.cells[j + 1], f"{val:.2f}%",
                           bold=is_ours, font_size=10,
                           color=_C_TEXT_OURS if is_ours else RGBColor(0, 0, 0))

    # ── Step 5: 注释 ─────────────────────────────────────────────────
    doc.add_paragraph()
    note_para = doc.add_paragraph()
    note_run  = note_para.add_run(
        "Notes: "
    )
    note_run.bold      = True
    note_run.font.size = Pt(9)
    note_run.font.name = "Times New Roman"
    note_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    note_body = note_para.add_run(
        "Yellow cells indicate best performance per dataset column. "
        "Green rows indicate our proposed method. "
        "Cells marked '\u2014' are pending experimental results. "
        "All methods use a unified Meta-Evaluation protocol "
        "(5-step adaptation on local Non-IID test data, Dirichlet partitioning)."
    )
    note_body.font.size = Pt(9)
    note_body.font.name = "Times New Roman"
    note_body.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.save(output_path)
    log(INFO, "[Table] Saved: %s", output_path)


# =============================================================================
# 服务器端 evaluate_fn（占位，实际评估由客户端回传 eval_acc 聚合）
# =============================================================================
def null_evaluate_fn(server_round, arrays):
    return None


# =============================================================================
# SCAFFOLD 服务器端全局控制变量聚合
# =============================================================================
class ScaffoldAggregator:
    """跨轮次维护全局控制变量 c，供 server 传给下一轮客户端"""
    def __init__(self):
        self.c_global: list = []   # List[Tensor]，展平后序列化传输

    def update(self, delta_c_flat_str: str, n_clients: int):
        """用平均 Δc 更新全局 c（简化：直接加均值）"""
        if not delta_c_flat_str.strip():
            return
        # 此处为简化实现：delta_c 以 norm 形式记录，全局 c 更新留待扩展
        pass

    def to_config_str(self) -> str:
        if not self.c_global:
            return ""
        return ",".join(f"{v:.8f}" for t in self.c_global for v in t.flatten().tolist())


# =============================================================================
# 主实验入口
# =============================================================================
@app.main()
def main(grid: Grid, context: Context) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── 地面热启动 ─────────────────────────────────────────────────────
    distilled_student, distilled_teacher = perform_ground_warmup(device)

    # ── 读取全局配置 ───────────────────────────────────────────────────
    cfg = context.run_config
    num_rounds       = int(  cfg.get("num-server-rounds", 100))
    fraction_eval    = float(cfg.get("fraction-evaluate",  1.0))
    selection_mode   = str(  cfg.get("selection-mode",  "window"))
    ks_train         = cfg.get("ks-train", None)
    ks_eval          = cfg.get("ks-eval",  None)
    enforce_comm     = bool( cfg.get("enforce-comm",    True))
    turnover_T       = int(  cfg.get("turnover-T",      5))
    deltaQ           = int(  cfg.get("deltaQ",          2))
    deltaP           = int(  cfg.get("deltaP",          3))
    probe_k          = int(  cfg.get("probe-blocked-k", 1))
    batch_size       = int(  cfg.get("batch-size",      32))
    dirichlet_alpha  = float(cfg.get("dirichlet-alpha", 0.1))
    meta_adapt_steps = int(  cfg.get("meta-adapt-steps", 5))
    meta_adapt_lr    = float(cfg.get("meta-adapt-lr",   0.01))
    kd_enable        = bool( cfg.get("kd-enable",       True))
    kd_alpha         = float(cfg.get("kd-alpha",        0.5))
    kd_temperature   = float(cfg.get("kd-temperature",  3.0))
    kd_cal_samples   = int(  cfg.get("kd-cal-samples",  2048))
    kd_global_samples= int(  cfg.get("kd-global-samples",4096))
    kd_batch_size    = int(  cfg.get("kd-batch-size",   64))
    kd_fwd_ep        = int(  cfg.get("kd-forward-epochs", 1))
    kd_fwd_lr        = float(cfg.get("kd-forward-lr",   0.05))
    kd_rev_ep        = int(  cfg.get("kd-reverse-epochs", 1))
    kd_rev_lr        = float(cfg.get("kd-reverse-lr",   0.01))

    # 通信参数
    comm_defaults = {
        "sigma2":           float(cfg.get("sigma2",            1e-9)),
        "default-tau-d-s":  float(cfg.get("default-tau-d-s",   50000.0)),
        "w-u-hz":           float(cfg.get("w-u-hz",            4e6)),
        "w-d-hz":           float(cfg.get("w-d-hz",            4e6)),
        "p-u-w":            float(cfg.get("p-u-w",             100.0)),
        "p-d-w":            float(cfg.get("p-d-w",             100.0)),
        "f-c-hz":           float(cfg.get("f-c-hz",            20e9)),
        "A-T":              float(cfg.get("A-T",               60.0)),
        "A-R":              float(cfg.get("A-R",               30.0)),
        "G-H":              float(cfg.get("G-H",               0.8)),
        "delta":            float(cfg.get("delta",             2.0)),
        "psi-db-per-km":    float(cfg.get("psi-db-per-km",     0.5)),
        "zeta-km":          float(cfg.get("zeta-km",           500.0)),
        "distance-km-min":  float(cfg.get("distance-km-min",   780.0)),
        "distance-km-max":  float(cfg.get("distance-km-max",   2300.0)),
        "distance-seed":    int(  cfg.get("distance-seed",     2026)),
        "batch-size":       batch_size,
        "dirichlet-alpha":  dirichlet_alpha,
    }
    eval_cfg_base = {
        **comm_defaults,
        "meta-adapt-steps": meta_adapt_steps,
        "meta-adapt-lr":    meta_adapt_lr,
    }

    # FedMeta 系列共用超参
    fomaml_cfg = {
        "fomaml-alpha":       float(cfg.get("fomaml-alpha",       0.01)),
        "fomaml-beta":        float(cfg.get("fomaml-beta",        0.01)),
        "fomaml-inner-steps": int(  cfg.get("fomaml-inner-steps", 7)),
        "apskd-epochs":       int(  cfg.get("apskd-epochs",       1)),
        "kd-warmup-rounds":   int(  cfg.get("kd-warmup-rounds",   20)),
        "kd-temperature":     kd_temperature,
        "kd-alpha":           kd_alpha,
        "kd-enable":          kd_enable,
    }

    def _strategy(mode, teacher=None):
        use_kd = kd_enable and mode == "ours"
        return FedMeta(
            fraction_evaluate=fraction_eval,
            selection_mode=selection_mode,
            ks_train   =int(ks_train)  if ks_train  is not None else None,
            ks_evaluate=int(ks_eval)   if ks_eval   is not None else None,
            enforce_comm=enforce_comm,
            turnover_T=turnover_T, deltaQ=deltaQ, deltaP=deltaP,
            probe_blocked_k=probe_k,
            kd_enable=use_kd,
            kd_alpha=kd_alpha, kd_temperature=kd_temperature,
            kd_cal_samples=kd_cal_samples, kd_global_samples=kd_global_samples,
            kd_batch_size=kd_batch_size,
            kd_forward_epochs=kd_fwd_ep, kd_forward_lr=kd_fwd_lr,
            kd_reverse_epochs=kd_rev_ep, kd_reverse_lr=kd_rev_lr,
            initial_teacher_model=copy.deepcopy(teacher) if teacher else None,
        )

    # ── 6 种方法顺序实验 ───────────────────────────────────────────────
    experiment_modes = ["fedavg", "fedprox", "scaffold", "fedkd", "fedmeta", "ours"]

    log(INFO, "=" * 60)
    log(INFO, "🚀 Starting 6-method sequential comparison")
    log(INFO, "   Modes: %s", experiment_modes)
    log(INFO, "=" * 60)

    for mode in experiment_modes:
        log(INFO, "\n" + "*" * 60)
        log(INFO, "▶  MODE: %s", mode.upper())
        log(INFO, "*" * 60)

        global_model = copy.deepcopy(distilled_student)
        arrays = ArrayRecord(global_model.state_dict())

        # ── 构建 train_cfg ────────────────────────────────────────────
        if mode == "fedavg":
            strategy  = _strategy("fedavg")
            train_cfg = ConfigRecord({
                **comm_defaults,
                "client-train-mode":    "fedavg",
                "learning-rate":        float(cfg.get("learning-rate",        0.005)),
                "local-epochs":         int(  cfg.get("fedavg-local-epochs",  1)),
                "weight-decay":         float(cfg.get("fedavg-weight-decay",  1e-4)),
                "label-smoothing":      float(cfg.get("fedavg-label-smoothing",0.1)),
            })

        elif mode == "fedprox":
            strategy  = _strategy("fedprox")
            train_cfg = ConfigRecord({
                **comm_defaults,
                "client-train-mode": "fedprox",
                "learning-rate":     float(cfg.get("learning-rate",    0.005)),
                "local-epochs":      int(  cfg.get("local-epochs",     1)),
                "weight-decay":      float(cfg.get("weight-decay",     1e-4)),
                "label-smoothing":   float(cfg.get("label-smoothing",  0.1)),
                "fedprox-mu":        float(cfg.get("fedprox-mu",       0.01)),
            })

        elif mode == "scaffold":
            strategy  = _strategy("scaffold")
            train_cfg = ConfigRecord({
                **comm_defaults,
                "client-train-mode": "scaffold",
                "learning-rate":     float(cfg.get("learning-rate",    0.01)),
                "local-epochs":      int(  cfg.get("local-epochs",     1)),
                "weight-decay":      float(cfg.get("weight-decay",     1e-4)),
                "scaffold-c-global": "",   # 首轮：全零控制变量
            })

        elif mode == "fedkd":
            # FedKD：server 额外下发 teacher_arrays
            strategy  = _strategy("fedkd", teacher=distilled_teacher)
            train_cfg = ConfigRecord({
                **comm_defaults,
                "client-train-mode": "fedkd",
                "learning-rate":     float(cfg.get("learning-rate",    0.005)),
                "local-epochs":      int(  cfg.get("local-epochs",     1)),
                "weight-decay":      float(cfg.get("weight-decay",     1e-4)),
                "kd-temperature":    kd_temperature,
                "kd-alpha":          kd_alpha,
                "kd-enable":         True,
            })

        elif mode == "fedmeta":
            strategy  = _strategy("fedmeta")
            train_cfg = ConfigRecord({
                **comm_defaults,
                **fomaml_cfg,
                "client-train-mode":  "fedmeta",
                "fomaml-inner-steps": int(cfg.get("fomaml-inner-steps", 5)),
                "kd-enable":          False,
            })

        else:  # ours
            strategy  = _strategy("ours", teacher=distilled_teacher)
            train_cfg = ConfigRecord({
                **comm_defaults,
                **fomaml_cfg,
                "client-train-mode": "ours",
            })

        eval_cfg = ConfigRecord(eval_cfg_base)

        # ── 运行 ──────────────────────────────────────────────────────
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_cfg,
            evaluate_config=eval_cfg,
            num_rounds=num_rounds,
            evaluate_fn=null_evaluate_fn,
        )

        # ── 保存模型 ──────────────────────────────────────────────────
        torch.save(result.arrays.to_torch_state_dict(), f"final_model_{mode}.pt")
        if mode == "ours" and kd_enable:
            teacher_saved = getattr(strategy, "teacher_model", None)
            if teacher_saved is not None:
                try:
                    torch.save(teacher_saved.state_dict(), "teacher_model_ours.pt")
                except Exception as e:
                    log(WARNING, "Failed to save teacher: %s", e)

        # ── 保存 JSON 指标 ─────────────────────────────────────────────
        h_eval  = getattr(result, "evaluate_metrics_clientapp", {}) or {}
        h_train = getattr(result, "train_metrics_clientapp",    {}) or {}
        if h_eval:
            all_rounds = set(h_eval) | set(h_train)
            metrics_to_save = {}
            for r in sorted(all_rounds):
                me = h_eval.get(r,  {})
                mt = h_train.get(r, {})
                metrics_to_save[int(r)] = {
                    "accuracy":   float(me["eval_acc"])    if "eval_acc"   in me else None,
                    "loss":       float(me["eval_loss"])   if "eval_loss"  in me else None,
                    "train_loss": float(mt["train_loss"])  if "train_loss" in mt else None,
                }
            with open(f"metrics_{mode}.json", "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            log(INFO, "Saved metrics_%s.json", mode)
        else:
            log(WARNING, "No eval metrics for mode: %s", mode)

    # ── 绘制对比图 ────────────────────────────────────────────────────
    log(INFO, "\n" + "=" * 60)
    log(INFO, "🎉 ALL EXPERIMENTS DONE — Generating comparison plots...")
    log(INFO, "=" * 60)
    plot_all()

    # ── 生成准确率对比表（Word 文档）────────────────────────────────
    generate_accuracy_table(
        output_path="accuracy_table.docx",
        last_n=10,
        dirichlet_alpha=dirichlet_alpha,
    )