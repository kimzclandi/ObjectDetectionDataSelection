# 2026-09-19：评测输出保护

本次工程维护发生在既有实验之后，没有新训练或选样收益。

发现：评测与复现脚本会无条件写回 `evaluation.json` 和 `sample_metrics.json`；即使只是核验也改变文件，预测变更后还会覆盖旧结论。
修复：两个结果均先比较；相同结果只读返回，不同结果明确失败。需要保存新结果时使用独立目录：

```bash
uv run driving-data evaluate --name baseline --output work/evaluation-20260919
```

输入仍为保存的预测及标注，不下载原图、不运行推理或训练。返回 COCO AP 与切片指标。测试覆盖相同重放不触碰 mtime、不同结果拒绝覆盖，以及显式新目录保存。

`baseline/` 保存本次维护前的相关文件，按历史 release/presentation 清单原哈希核验。冻结清单、配置、预测、报告不变；当前源码由 Git diff 与新的回归结果审阅，不能把原实验源码身份归给新实现。

## 逐图指标重算

原 `scripts/verify_failure_v2.py` 会调用完整模型运行的校验器，因此需要未入库的图像、标注、权重及原源码环境；不能把它标成纯公开离线核验。新的入口只需参考标注与仓库保存的预测：

```bash
uv run python scripts/replay_metrics.py --labels-dir /path/to/data/labels --output work/replay-20260919
```

输入为原数据准备流程生成的 120 个评测图像的标签 JSON，逐个对照冻结 manifest 的 `labels_sha256`；缺失或不一致立即失败。已有本地数据无需下载。新机器仍需遵循[数据准备说明](../../RUNNING.md)获取 BDD100K 许可数据；标签没有新增分发到 Git。本次不把本机标签重算称为无数据下载的全流程异机复现。

产物是十组共 1,200 条预测重算的 COCO AP、precision、recall 和逐图切片，全部与历史记录比较。不加载原图或模型，不改变旧预测/报告。`scripts/verify_artifacts.py` 仍是无需许可数据的完整性入口，二者职责不同。

修改型 CLI（prepare、infer、select、experiment、benchmark、controls、vlm-quality、catalog）发现根目录含冻结发布清单时，会在执行前拒绝。新运行使用 `--root` 指向不含旧发布记录的独立执行目录，并准备所需配置和许可数据；不要删除原仓库清单来绕过保护。evaluate 的相同重算仍可只读进行。直接调用历史研究函数需遵循其独立 checkout 约束。

本次本地验证：69 项测试通过；命令和修改后源码摘要见 [validation.json](validation.json)。这不是未推送提交的远端 CI 结果。

本次 [指标重算摘要](evidence/metric-replay.json) 保持原结论：定向、随机、seed-only 的平均 AP 分别为 8.595、9.229、7.609（×100）。另保存一次 [dev 图像真实检测冒烟](evidence/inference-smoke.json)，没有新增训练或泛化结论。
