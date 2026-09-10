"""Read-only viewer of saved experimental evidence. Start from repository root."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from driving_data.common import CLASSES, read_json

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/pilot"
st.set_page_config(page_title="Driving Data Engine", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#0c1320;color:#e7edf6}
.block-container {padding-top:4rem;max-width:1440px}
[data-testid="stHeader"] {background:#0c1320}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:#152236;border:1px solid #293c54;padding:18px;border-radius:10px}
[data-testid="stMetricValue"] {color:#69ddc0}
</style>""",
    unsafe_allow_html=True,
)
st.caption("DRIVING DATA ENGINE  /  实验工作台  /  v0.1")
st.title("让每一批道路数据，都有可追踪的选择依据")
st.write("真实道路图像 → 模型推理 → 样本筛选 → 检测头微调 → 独立评测")
st.info("研究试验：公开数据的小规模重划分；仅微调检测器 ROI 预测头。未验证标注成本、真车安全或生产规模。")
if not (REPORTS / "data_receipt.json").exists():
    st.warning("尚无数据。请先在终端执行 driving-data prepare。")
    st.stop()
receipt = read_json(REPORTS / "data_receipt.json")
manifest = read_json(REPORTS / "sample_manifest.json")["rows"]
by_id = {r["sample_id"]: r for r in manifest}
experiment = read_json(REPORTS / "experiment.json") if (REPORTS / "experiment.json").exists() else None
cols = st.columns(4)
for c, label, value in zip(
    cols,
    ["真实图像", "候选池", "固定测试集", "已完成策略实验"],
    [
        receipt["images"],
        receipt["splits"]["pool"],
        receipt["splits"]["test"],
        len(experiment["records"]) if experiment else 0,
    ],
):
    c.metric(label, value)
overview, review, sampling, evidence = st.tabs(["实验对照", "失败样本", "数据选择", "追踪与复现"])

with overview:
    st.subheader("同样新增 12 张，哪种选择更有效？")
    if experiment:
        df = pd.DataFrame(experiment["aggregates"])
        display = df[["strategy", "test_AP_mean", "test_AP_seed_std", "mean_delta_vs_random"]].copy()
        display.columns = ["策略", "测试 AP 均值", "随机种子标准差", "相对随机 AP 差"]
        st.bar_chart(df.set_index("strategy")[["test_AP_mean"]], color="#69ddc0")
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("AP 为 0–1；三个种子仅描述训练波动，不能当成总体显著性证据。")
        st.warning(
            "当前多样性策略的平均优势很小，不能声称显著收益。PASS 只表示满足预设离线门槛，不代表安全可用。"
        )
        st.subheader("每次运行都有自己的回归判定")
        runs = [
            {
                "运行": r["name"],
                "训练图像": r["train_images"],
                "测试 AP": r["test_AP"],
                "判定": r["gate"]["decision"],
                "夜间召回变化(pp)": round(100 * r["gate"]["night_recall_delta"], 2)
                if r["gate"]["night_recall_delta"] is not None
                else None,
                "原因": " / ".join(r["gate"]["reasons"]),
            }
            for r in experiment["records"]
        ]
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    else:
        st.warning("策略实验尚未完成。当前展示已保存的真实数据，不填充示意指标。")

with review:
    runs = sorted(p.parent.name for p in REPORTS.glob("*/evaluation.json"))
    if not runs:
        st.write("评测报告生成后，此处显示失败样本。")
    else:
        chosen_run = st.selectbox("模型运行", runs, index=runs.index("baseline") if "baseline" in runs else 0)
        metrics = read_json(REPORTS / chosen_run / "sample_metrics.json")
        timeofday = st.selectbox("光照切片", ["全部"] + sorted({r["timeofday"] for r in metrics}))
        candidates = sorted(
            [r for r in metrics if timeofday == "全部" or r["timeofday"] == timeofday],
            key=lambda r: (-r["fn"], r["sample_id"]),
        )
        sid = st.selectbox("样本（按漏检数量排序）", [r["sample_id"] for r in candidates])
        row = by_id[sid]
        m = next(r for r in candidates if r["sample_id"] == sid)
        a, b = st.columns([3, 1])
        local_image = ROOT / row["image_path"]
        labels_file = ROOT / f"data/labels/{sid}.json"
        if local_image.exists() and labels_file.exists():
            im = Image.open(local_image).convert("RGB")
            draw = ImageDraw.Draw(im)
            for g in read_json(labels_file):
                draw.rectangle(g["box"], outline="#69ddc0", width=3)
            predictions = read_json(REPORTS / chosen_run / "predictions.json")["rows"]
            ps = next(r for r in predictions if r["sample_id"] == sid)
            for p in ps["predictions"]:
                if p["score"] >= 0.5:
                    draw.rectangle(p["box"], outline="#ffaf70", width=3)
                    draw.text(p["box"][:2], f"{CLASSES[p['label']]} {p['score']:.2f}", fill="#ffaf70")
            a.image(im, caption="绿色：公开标注  /  橙色：模型预测 ≥ 0.5", width="stretch")
        else:
            a.info("图像未随代码分发。执行 driving-data prepare 可下载固定版本样本。")
        b.metric("漏检 FN", m["fn"])
        b.metric("误检 FP", m["fp"])
        b.write(
            {"光照": row["timeofday"], "天气": row["weather"], "场景": row["scene"], "划分": row["split"]}
        )
        b.caption("漏检是现象，不自动等于根因。需要结合尺度、遮挡和标注检查。")

with sampling:
    if (REPORTS / "selections.json").exists():
        selections = read_json(REPORTS / "selections.json")
        plans = selections["plans"]
        plan_name = st.selectbox("采样计划", [f"{p['strategy']} / seed {p['seed']}" for p in plans])
        plan = plans[[f"{p['strategy']} / seed {p['seed']}" for p in plans].index(plan_name)]
        st.dataframe(pd.DataFrame(plan["selected"]), hide_index=True, width="stretch")
        st.write("采样输入只包含图像身份、预测和图像特征；不包含真实框、天气或时段标签。")
        st.caption("不确定性来自预测置信度熵，多样性来自检测器骨干特征。两者都不是价值保证。")
    else:
        st.write("尚未生成采样计划。")

with evidence:
    st.subheader("固定版本 · 原始证据 · 可复现命令")
    st.json(receipt, expanded=False)
    st.code(
        "uv sync --extra dashboard --extra dev\nuv run driving-data prepare\nuv run driving-data verify\nuv run driving-data experiment\nuv run streamlit run dashboard.py"
    )
    st.download_button("下载数据来源凭证", (REPORTS / "data_receipt.json").read_bytes(), "data_receipt.json")
    if experiment:
        st.download_button("下载完整实验对照", (REPORTS / "experiment.json").read_bytes(), "experiment.json")
    st.write("可运行：数据接入、真实推理、三策略、真实检测头训练、回归门禁、故障恢复。")
    st.write("后续：真实 VLM 辅助质检、人工复核成本实验、多机执行、更大规模与路线隔离评测。")
