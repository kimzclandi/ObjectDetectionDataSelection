# 岗位能力—代码—运行证据—边界

| 岗位能力 | 实现位置 | 可审阅运行证据 | 未验证边界 |
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

## 严格面试官结论

能展示提出可检验假设、设计对照、固定数据与模型身份、执行小规模真实训练、保留负结果和排查工程故障的能力。只有看清本轮结果才能决定是否存在算法增益；功能数和测试数不构成算法有效性证据。

仍不能据此认定成熟的主动学习算法创新、端到端检测训练能力、商业标注ROI、生产数据治理、海量视频去重或车端安全经验。无显著收益时，项目卖点是实验判断和工程可信性，而非“算法提升X%”。

生产才能回答的问题包括：数据触发回传预算、驾驶员/路线隐私与授权、标注团队一致性与工期、在线分布漂移、灰度与回滚、场景复现、安全评审以及多团队接口。可以讲设计，但必须明确未经实际生产验证。

本人必须现场掌握：采样输入字段为何不含oracle；梯度参数与BN状态；112步由何而来；IoU/一对一匹配；AP与阈值召回的不同；组bootstrap；缓存键；optimizer/RNG恢复；如何从原始预测重算一个失败；为什么负结果后不继续用保留集调参。学习验收题见 `DATA_LOOP_INTERVIEW.md`。
