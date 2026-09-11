# 运行说明

## 浏览当前结果

```bash
uv sync --locked --python 3.12 --extra dashboard --extra dev
uv run python scripts/verify_artifacts.py
uv run streamlit run dashboard.py --server.address 127.0.0.1
```

侧栏选择「失败驱动训练 · 新评测集」。`verify_artifacts.py` 校验发布清单，未执行模型训练或重新计算 AP。CI 状态见首页徽标，其测试范围以工作流为准。

## 数据准备与历史训练

以下命令会下载数据与固定权重，并执行真实 CPU 训练。仅在需要重跑时使用；参考结果、失败和协议不可覆盖。

```bash
uv run driving-data prepare
uv run driving-data verify
uv run driving-data experiment
uv run driving-data controls
```

前两条准备并核验旧 240 图；`experiment` 生成或恢复共同 warmup 及旧 pilot 对照，`controls` 为历史等步数消融。原始图像、标注、权重不在 Git 中。依赖、数据与权重首次获取需要网络，模型运行无需付费 API。

最新训练使用上述共同 warmup；准备完成后按[失败驱动协议](FAILURE_V2_PROTOCOL.md)运行 `failure_loop prepare/run/analyze`。必须满足冻结权重和输入摘要；不要刷新协议来接受差异。已有参考结果与新执行记录需分开保存，不能把缓存复用计成新训练。

## 扩展实验

旧 pilot Dashboard 中包含独立的 VLM 昼夜标签与 Ray 计时；它们不属于当前选样主结果。实际结果和边界分别见[VLM 质检](VLM_QUALITY.md)与[架构说明](ARCHITECTURE.md)。以下可选命令分别运行真实 VLM 推理与单机处理计时，不属于只读结果回放：

```bash
uv sync --locked --python 3.12 --extra dashboard --extra dev --extra distributed --extra vlm
uv run driving-data vlm-quality
uv run driving-data vlm-quality --constrained
uv run driving-data benchmark
```

## 文档维护与冻结

`reports/artifact_manifest.json` 由 `scripts/qa.py` 定义为可分发文件的发布清单，收录 README、说明文档、源代码与证据。2026-09-11 整理入口时，仅更新被编辑文档的条目并加入新说明页，以继续检查当前发布内容；未重新生成旧 QA 收据。

它不同于 `reports/failure_v2/protocol.json`、各实验 artifact manifest、数据清单和 `reports/diagnosis_v3/protocol.json` 等研究冻结文件。后者及所有预测、训练记录、失败与统计结果保持原字节。文档编辑不授权修改实验身份或重算历史哈希。
