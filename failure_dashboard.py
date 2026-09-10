"""Fresh-test training evidence, distinct from historical pilot/VLM inference."""

from pathlib import Path

import pandas as pd
import streamlit as st

from driving_data.common import read_json


def show(root: Path):
    out = root / "reports/failure_v2"
    st.title("失败驱动选样，是否优于随机？")
    st.caption(
        "真实 ROI 预测头训练 · 固定 112 步 · 定向/随机新增 12 图，对照不新增 · 3 个配对种子 · 120 张新评测图"
    )
    if not (out / "summary.json").exists():
        st.info("协议已冻结，实验未全部完成。")
        return
    s = read_json(out / "summary.json")
    c = st.columns(3)
    for col, arm in zip(c, ["targeted", "random", "seed_only"]):
        values = [r["AP"] for r in s["records"] if r["arm"] == arm]
        col.metric(arm + " · 平均 AP ×100", f"{sum(values) / len(values) * 100:.3f}")
    ci = s["group_bootstrap_ci95"]
    st.write(
        f"定向 − 随机：{s['mean_AP_delta'] * 100:.3f} AP 百分点；组级配对区间 [{ci[0] * 100:.3f}, {ci[1] * 100:.3f}]。"
    )
    st.warning("三种子、同源小样本研究；区间描述样本不确定性，不证明生产收益、标注降本或驾驶安全。")
    st.dataframe(
        pd.DataFrame(
            [{k: r[k] for k in ["name", "AP", "precision", "seconds", "steps"]} for r in s["records"]]
        ),
        hide_index=True,
    )
    st.subheader("固定失败切片与非目标退化")
    st.dataframe(
        [
            dict(
                name=r["name"],
                target_recall=r["slices"]["target"]["recall"],
                complement_recall=r["slices"]["complement"]["recall"],
                night_recall=r["slices"].get("time:night", {}).get("recall"),
            )
            for r in s["records"]
        ],
        hide_index=True,
    )
    st.subheader("失败证据 → 选样规则")
    d = read_json(out / "dev_diagnosis.json")
    st.write(
        "固定模型降低分数阈值，检查哪些真值由未命中变为命中；只用这些开发图的特征形成原型。候选池只提供图像身份和预测，不提供真值。"
    )
    with st.expander("开发失败干预原始记录"):
        st.json(d)
    plans = read_json(out / "selections.json")
    choice = st.selectbox("采样运行", [f"{p['strategy']}-s{p['seed']}" for p in plans["plans"]])
    plan = next(p for p in plans["plans"] if f"{p['strategy']}-s{p['seed']}" == choice)
    st.dataframe(plan["selected"], hide_index=True)
    st.caption("公开标注在入选 ID 冻结后用于训练，模拟标注获取；未进行人工标注或人工根因复核。")
    with st.expander("冻结协议与边界"):
        st.json(read_json(out / "protocol.json"))
