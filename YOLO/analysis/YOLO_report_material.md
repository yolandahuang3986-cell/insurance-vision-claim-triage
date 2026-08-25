# YOLO Segmentation — 方法、结果与误差分析（素材整理，供最终报告撰写参考）

## 3. Methods

本方法基于 Ultralytics YOLO11n-seg 预训练权重（`yolo11n-seg.pt`）微调，属于单阶段 anchor-free 实例分割模型。骨干网络提取特征后，检测头直接回归边界框、类别和原型掩码系数，再通过原型掩码（mask prototypes）与系数线性组合得到实例分割结果，不需要像 Mask R-CNN 那样先做 RoI 提取再逐候选框预测掩码。

**训练设置**

| 项目 | 取值 |
|---|---|
| 预训练权重 | yolo11n-seg.pt |
| 训练轮数 (epochs) | 50 |
| 输入分辨率 | 640 × 640 |
| 随机种子 | 42 |
| 硬件 | Apple Silicon (macOS 15.7.4, arm64), device=mps |
| 置信度阈值 | 0.5 |
| IoU 匹配阈值 | 0.5 |
| 训练耗时 | 10644.3 s（约 2.96 小时） |
| 数据配置 | data/yolo/data.yaml（CarDD 官方 train/val/test = 2816/810/374 划分转换为 YOLO 格式） |

如果之后想换成 YOLOv8-seg 或更大的 YOLO11 变体（s/m/l），只需要换预训练权重文件和对应的模型配置文件，`train_yolo.py` 里的训练/推理接口不需要动。评估脚本（`mask_iou.py`、`merge_results.py`）也不用改，因为它们只依赖 Ultralytics 统一的 `model.predict()` 输出（`result.masks.xy`、`result.boxes`）。

---

## 4. Evaluation and comparison

**结果表（YOLO 行）**

| Method | mask IoU | Precision | Recall | mAP@50 | mAP@50:95 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO Segmentation | 0.8207 | 0.8138 | 0.6243 | 0.5912 | 0.4699 | 8.63 |



**指标定义与计算方式说明**

mask IoU 不是 Ultralytics 训练/验证时自带的指标，是我们在 `evaluation/mask_iou.py` 里单独算的：在 CarDD 官方 test 划分（374 张图）上重新推理一遍，把每个预测掩码和同类别、IoU 最高的 GT 实例做一对一贪心匹配（每个 GT 只会被匹配一次），只对匹配成功且 IoU ≥ 0.5 的预测对取平均 IoU。算出来的 0.8207 反映的是预测一旦匹配上 GT 之后，掩码轮廓贴合得有多好，不代表整体检测覆盖率。

Precision / Recall / mAP@50 / mAP@50:95 取自 Ultralytics 训练脚本输出的 mask 指标，而不是 box 指标，因为我们关心的是分割质量而不只是框住损伤区域。`training_summary.json` 里同时留了 box 指标（precision 0.8090、recall 0.6228）作为对照，正文如果没特别说明都以 mask 指标为准。

matched_pairs = 400，unmatched_predictions = 126，后者是模型给出但没能匹配到任何同类别 GT 实例的预测（IoU < 0.5，或附近根本没有对应类别的 GT），本质上就是误检。下面误差分析部分会展开。

latency 是 8.63 ms/image，单张图像端到端推理耗时（预处理+推理+后处理，明细见 `latency_breakdown_ms`），在 mps（Apple Silicon GPU）上测的，跨设备对比的时候要注明这一点。

整体来看，YOLO Segmentation 的 mask IoU（0.82）和 precision（0.81）都不错，说明检测到损伤之后定位和形状拟合比较准；但 recall 只有 0.62，相当一部分真实损伤没被检出，这块结合下面的 missed_damage 数量和具体案例展开。

---

## 5. Error analysis

### 5.1 整体误差分布

在 CarDD test 划分（374 张图）上，以 conf=0.5、IoU 匹配阈值=0.5 重新推理后，`evaluation/mask_iou.py` 的匹配结果如下：

| 类别 | 数量 | 说明 |
|---|---:|---|
| Correct（正确匹配） | 400 | 预测掩码与同类别 GT 实例匹配上，IoU ≥ 0.5 |
| Missed damage（漏检） | 385 | GT 损伤实例附近没有任何同类别、IoU 达标的预测，模型完全没检测到 |
| Unmatched predictions（误检/假阳性） | 126 | 模型给出了预测，但附近没有同类别 GT 与之匹配 |

385 已经快追上 400 了，说明现在模型最主要的问题其实是漏检，而不是"检测到了但形状不准"——这也和结果表里 recall 只有 0.6243、precision 却有 0.8138 对得上：模型给出的预测大多是对的，就是覆盖率不够。126 个 unmatched predictions 算误检，会拉低实际部署时的可信度（人工审核会看到模型报了损伤但其实没有，或者类别/位置报错了）。

评估方法上有个局限需要说明：`error_analysis.classify_error()` 本来设计了五类标签（missed_damage / wrong_damage_class / poor_localization / low_confidence_correct / bad_input_quality），但因为 `mask_iou.py` 是按类别做一对一匹配、而且只对 conf≥0.5 的预测推理，实际跑出来只有 correct 和 missed_damage 两类。这不是说模型没有分类错误或定位偏差的问题，而是当前的匹配逻辑把这些错误都"藏"进了 missed_damage（GT 侧）或 unmatched_predictions（预测侧）里，没有单独拆出来。报告里这块要写清楚，不然读者可能会误以为模型完全没有分类混淆的问题。

### 5.2 分类别表现差异

匹配成功的实例里，mask IoU 按类别差得挺多，从 0.64 到 0.95 都有：

| 类别 | Mean mask IoU |
|---|---:|
| tire flat | 0.9529 |
| glass shatter | 0.9379 |
| lamp broken | 0.9097 |
| dent | 0.7869 |
| scratch | 0.7545 |
| crack | 0.6412 |

大致规律是：形状规则、区域大、边界清晰的损伤（爆胎、玻璃碎裂、大灯破损）掩码质量高；裂纹和划痕这种细长、边界模糊的损伤明显更差。原因大概两点：裂纹本身就只有几个像素宽，标注和预测稍微差一点，IoU 就会掉得很厉害（union 相对交集的变化太敏感）；划痕还经常和车身反光、清洁痕迹混在一起，边界很难精确到像素级。

### 5.3 crack 类别专项案例

crack 是 mask IoU 最低的类别（0.6412），单独用 `run_case_study.py` 在 test 集里挑出全部 48 张含 crack 标注的图片，对前 5 张跑了一遍 `model.predict()`（conf=0.5）看效果，结果如下：

| 图片 | 结果 |
|---|---|
| 000044.jpg | 漏检。保险杠上那道很明显的裂纹完全没被框出来，模型只标出了旁边一处 scratch（conf 0.50）。 |
| 000520.jpg | 检出 crack（conf 0.64），同时检出 dent（conf 0.86）。 |
| 001039.jpg | 检出 crack（conf 0.65），同时检出 lamp broken（conf 0.96）。 |
| 001979.jpg | 检出 crack（conf 0.92），边界框基本贴合裂纹范围，是 5 张里效果最好的一张；同时检出 dent（conf 0.62）。 |
| 002294.jpg | 检出 crack（conf 0.67），但框选范围偏小，只圈住了裂纹的一小部分，没有覆盖车门上完整的裂痕。 |

5 张里 4 张检出了 crack（检出率和整体 recall 偏低的情况大致吻合），置信度在 0.64～0.92 之间波动。两个问题比较突出：

一是漏检（000044）。裂纹本身对比度不低，人眼很容易看出来，模型却完全没反应，说明 5.1 提到的漏检问题在 crack 类别上确实存在，不是个例。

二是即便检出了，边界框/掩码范围也经常偏保守、覆盖不全（002294），这和 5.2 里给出的"裂纹细长、掩码对像素级误差敏感、IoU 容易被拉低"的解释是对得上的——不只是漏检拉低了 recall，检出的部分掩码质量也不够稳定，两者共同导致 crack 的 mask IoU 是六个类别里最低的。

如果报告里要放对比图，000044（漏检）和 001979（检出最好）可以作为一组反差案例：同样是裂纹损伤，一个完全没检出，一个检出置信度 0.92 且边界贴合，比较直观地体现模型在这个类别上的不稳定性。

---

## 附：原始数据来源

- 结果表 / 训练配置：`runs/training_summary.json`
- mask IoU 汇总 / 匹配统计：`evaluation/mask_iou_summary.json`
- 最终结果行：`evaluation/final_results_row.json`
- 图表原始数据对应 `runs/segment/val/` 下的曲线图（confusion matrix、PR/F1 曲线等），如报告需要配图可直接从中选用
