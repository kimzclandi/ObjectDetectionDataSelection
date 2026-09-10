"""Read-only development diagnostic evidence."""

from pathlib import Path

import streamlit as st

from driving_data.common import read_json

st.title("定向策略为何没有优于随机？")
st.caption("开发集探索性诊断 · 没有新训练 · 未使用测试标签")
r = read_json(Path("reports/diagnosis_v3/result.json"))
a, b, c = st.columns(3)
a.metric("完整评分 vs 低分质量 · Spearman", f"{r['pool_stats']['mass']['spearman_full']:.3f}")
b.metric("原型留组 AUC", f"{r['dev_auc']['similarity']:.3f}")
c.metric("新增低分框未匹配比例", f"{r['totals']['low_fp'] / r['totals']['low_count']:.1%}")
st.warning("未匹配框不等于已确认的背景误检；关联诊断不能解释全部 AP 因果差值。")
st.subheader("排序成分")
st.dataframe(r["pool_stats"])
st.subheader("开发集逐图证据")
st.dataframe(r["dev_records"], hide_index=True)
st.write("决策：停止追加同类训练；先补足视频场景级挖掘。")
with st.expander("协议与边界"):
    st.json(read_json(Path("reports/diagnosis_v3/protocol.json")))
    st.write(r["limitations"])
