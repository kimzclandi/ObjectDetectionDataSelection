# 数据与评测卡

## 来源与许可

- 原始项目：[BDD100K](https://github.com/bdd100k/bdd100k)。
- 论文：[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html)。
- 本次可访问镜像：[dgural/bdd100k](https://huggingface.co/datasets/dgural/bdd100k)，上游验证集 10,000 个样本。
- 固定 revision：`c2e7f266756bcd07b87f1a45a35937c8eac20241`。
- `samples.json` SHA-256：`1e8853904e2926c434557bb90a36b1638ac03ebb77adfc21f04d821a6a998a09`。
- 许可依据：[原仓库数据许可](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)，副本见 `BDD100K_LICENSE.rst`。

镜像顶部的 `bsd` 标签不能替代原始数据条款。原始条款允许教育、研究、非营利用途，商业权限有额外条件。本项目为教学研究，Git 不分发道路图像、原始标注或训练权重。衍生来源记录仍须保留上游声明。

## 抽样与划分

从完整镜像索引按 `SHA256(["subset-v1", filepath])` 排序取前 240 张，不按标注数量、模型错误或画面质量筛选。

按文件名前缀构造 group key，再以固定哈希桶划为 seed/pool/dev/test。实际数量分别为 **44/101/36/59**。相同前缀不能跨集合；相同图片 SHA-256 直接拒绝。该前缀只是保守近邻隔离措施，不能证明驾驶路线、驾驶员或采集会话独立。近重复图像尚未全面识别。

划分、图像哈希、标签规范化哈希、来源 revision 一起进入 manifest version。图像或标签被修改后，验证和评测会拒绝继续。

`data/pool.json` 只允许 `sample_id/group_id/split/image_path/sha256` 五个字段。采样读取的额外输入仅来自真实像素推理得到的预测框、分数和骨干特征；不接受带真实标注或天气/时段字段的候选池。

隔离是代码契约，不是安全沙箱：文件仍在同一台机器上，具有文件权限的人能读取。目标是阻止正常实验路径误用标签，不声称密码学保密。

## 标签转换

镜像框为归一化 `x,y,width,height`，转换到原图像素 `x1,y1,x2,y2`。非有限值、非正尺寸和完全越界框失败；轻微越界裁剪到图像范围。

| BDD 类别 | COCO ID / 当前语义 |
|---|---|
| pedestrian/person/rider | 1 / person（包含骑行者） |
| bicycle/bike | 2 / bicycle |
| car | 3 / car |
| motorcycle/motor | 4 / motorcycle |
| bus | 6 / bus |
| truck | 8 / truck |
| traffic light | 10 / traffic light |

其他类不在当前任务范围。保留遮挡、截断标记用于分析；只对上述范围内标签训练和评估。

## 模型与指标

模型：[Torchvision MobileNetV3 Faster R-CNN 320](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)。官方 COCO_V1 权重下载后保存 SHA-256。模型内部短边 320、长边最大 640，输出框映射回原始图像坐标。

输入：每张 `float32[3,H,W]` RGB 张量，值域 0–1。输出：`boxes[N,4]`、`labels[N]`、`scores[N]`，以及首个 FPN 特征图的全局平均池化、L2 归一化特征 `[256]`。特征用于多样性采样，未使用真值框。

AP 使用 pycocotools 的 bbox 指标，IoU 0.50:0.05:0.95，在当前七类上汇总；没有真值支持的类别遵循 pycocotools 的排除规则。预测保留门槛为 0.05；该截断影响可实现召回，故不是不受阈值限制的模型上限。

固定运行点指标：score ≥ 0.5、IoU ≥ 0.5、同类一对一匹配。按置信度从高到低匹配，每个 GT 最多命中一次。一个图像若推理失败，整个评测停止，不能从分母静默删除。空预测对有目标图像产生漏检；空标注切片不伪造召回。

- 天气/时段切片：全图预测与全图标注共同评估，包含 FP。
- 小目标/遮挡/person 对象切片：在完整匹配之后统计对应 GT 的召回，不将范围外目标算成 FP。
- 小目标：原图框面积 < 1,024 像素，不是网络缩放后的面积。
- 样本级失误不自动证明根因，需人工检查标注、尺度、遮挡、域偏移等解释。

公开小测试集只能支持局部实验，不支持真实驾驶事故率、安全性、总体分布效果或模型预训练无污染保证。

