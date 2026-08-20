# Training entry points

课程项目比较两个实例分割方法：YOLO Segmentation 和 Mask R-CNN。两个入口只负责建立模型和实验输出目录；数据适配、优化循环、权重缓存和硬件选择必须在小组本地完成，并记录到报告中。

为了公平比较，两个方法应使用相同的数据划分、类别映射、评测阈值和测试集。模型权重和 checkpoints 只保存在本地，`.gitignore` 与提交打包脚本会排除它们。

CarDD 的官方 COCO 标注可以先转换为 YOLO segmentation 格式：

```bash
python dataset/convert_coco_to_yolo.py \
  --source-dir data/raw/CarDD_release \
  --output-dir data/processed/cardd_yolo
```

转换默认创建图片 symlink，不会复制原始图片；如训练环境不支持 symlink，可增加 `--link-mode copy`。输出的 `data/processed/cardd_yolo/data.yaml` 可直接作为 `training/train_yolo.py --data-yaml` 的输入。
