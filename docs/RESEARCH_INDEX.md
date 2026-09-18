# 实验与附件索引

| 阶段 | 内容与结果入口 |
|---|---|
| 当前训练对照 | [失败驱动选样报告](FAILURE_V2_REPORT.md)：新 120 图评测，等 112 步，未支持定向优于随机 |
| 后续机制诊断 | [开发集诊断](DIAGNOSIS_V3_REPORT.md)：排序分解与留组原型检查，无新增训练 |
| 早期 pilot | [实验报告](EXPERIMENT_REPORT.md)：旧 240 图、三策略及后验等步数消融；不得混用新旧评测数字 |
| VLM 扩展 | [昼夜标签质检](VLM_QUALITY.md)：36 张开发图上的真实推理，格式合法不等于语义正确 |
| 数据处理扩展 | [架构与单机实验](ARCHITECTURE.md)：Ray 小负载慢于串行，未验证多机部署 |

[旧 pilot 界面截图](../assets/dashboard.png)属于历史实验，不是当前新评测集结果。

## 方法文档

[实现讲解](METHOD_NOTES.md)、[选样方法](FAILURE_V2_METHOD.md)、[实现地图](IMPLEMENTATION_MAP.md)、[后续实现索引](FAILURE_V2_IMPLEMENTATION.md)提供各轮方法、实现位置及验证边界。
