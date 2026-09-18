# failure-v2 实现与验证索引

| 功能 | 实现位置 | 可审阅运行证据 | 未验证边界 |
|---|---|---|---|
| 失败驱动数据挖掘 | `failure_loop.py::diagnose/rank_pool` | `reports/failure_v2/dev_diagnosis.json`、`selections.json` | 根因竞争存在；未证明策略优于随机，未做组件消融 |
| 标注获取与预算控制 | `rank_pool/select/schedule` | 冻结入选12图、公开标签hash、每组112步 | 不等于人工标注12图的成本或质量 |
| 真实训练与控制变量 | `train_run` | 九份完整loss/权重摘要、共同初始权重、三个seed | 仅466,375参数ROI预测头，不是全量训练 |
| 独立模型验证 | `metrics/analyze` | 120张新评测图，逐图预测、AP、固定失败切片、组bootstrap | 前缀非已验证驾驶会话，同源小数据 |
| 非目标退化检查 | `metrics` | 固定target/complement/time切片及支持数量 | 不代表线上事故风险或全部长尾类别 |
| 数据质量与防泄漏 | `prepare/validate`、`data.py` | 固定来源、排除旧组、hash、近重复审计 | 近重复启发式不保证所有语义重复被发现 |
| 流水线恢复与幂等 | `train_run/predictions/immutable` | `resume_verification.json`：真实重放逐张量/loss比较 | 一次CPU恢复，不是容灾或多机可用性保证 |
| 证据可重算 | `scripts/verify_failure_v2.py` | 原始预测重新计算10组AP、召回与分母 | 新AP重算需要按许可下载相同参考标签；CI不重新训练 |
| VLM自动标签质检（旧轨） | `vlm_quality.py` | `reports/pilot/vlm_quality*` | 场景标签参考未独立人工裁决，无自动标注降本收益 |
| 视觉评测有效性（另一仓库） | [grounding.py](https://github.com/kimzclandi/vlm-data-flywheel-lab/blob/main/src/flywheel/grounding.py) | [真实/空白/错配实验](https://github.com/kimzclandi/vlm-data-flywheel-lab/tree/main/reports/grounding_v3) | 合成小图像、无VLM训练，规则历史单独展示 |

## 验证范围

本轮未证明定向策略优于随机。功能数量、测试数量及恢复成功不能替代算法效果验证。尚未验证全量检测器训练、人工标注成本、多机调度或生产部署。方法见 [数据闭环说明](FAILURE_V2_METHOD.md)。
