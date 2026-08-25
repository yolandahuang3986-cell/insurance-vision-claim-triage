# CarDD YOLO Segmentation — CA6129 课程项目（YOLO 部分）

本仓库包含 CA6129 课程项目中 YOLO Segmentation 分支的代码、训练结果与分析。项目对比 YOLO Segmentation 与 Mask R-CNN 在 CarDD 车损数据集上的实例分割效果，本仓库仅负责 YOLO 部分。

## 目录结构

```
dataset/        数据准备与格式转换脚本
training/       训练脚本
evaluation/     评估脚本与结果（mask IoU、指标合并、错误分析）
runs/           训练/验证输出（指标 JSON、曲线图）
data/yolo/      数据集配置（data.yaml、转换摘要）
analysis/       错误分析草稿与可视化图表
```

## 数据集

使用 CarDD 官方 train/val/test 划分（2816/810/374 张图片），6 个类别：
`dent, scratch, crack, glass shatter, lamp broken, tire flat`

原始图片与标注不包含在本仓库中（体积过大），需自行下载并按 `data/yolo/data.yaml` 中的路径组织。

## 关键结果（test split，374 张图片）

| 指标 | 数值 |
|---|---:|
| Mask IoU (mean matched-pair) | 0.8207 |
| Mask Precision | 0.8138 |
| Mask Recall | 0.6243 |
| mAP50 | 0.5912 |
| mAP50:95 | 0.4699 |
| Latency | 8.63 ms/image |

固定随机种子 42，confidence threshold 0.5，IoU match threshold 0.5，模型 yolo11n-seg，训练 50 epochs（设备：Apple Silicon / MPS）。

详细结果见 `runs/training_summary.json` 和 `evaluation/final_results_row.json`；分类别 mask IoU 见 `evaluation/mask_iou_summary.json`；错误类型分析见 `analysis/`。

## 复现步骤

1. `dataset/prepare_cardd.py` — 准备 CarDD 原始数据
2. `dataset/convert_coco_to_yolo.py` — 转换为 YOLO 格式
3. `training/train_yolo.py` — 训练模型
4. `evaluation/mask_iou.py` + `evaluation/metrics.py` — 计算指标
5. `evaluation/merge_results.py` — 合并结果
6. `evaluation/error_analysis.py` — 错误类型分析
