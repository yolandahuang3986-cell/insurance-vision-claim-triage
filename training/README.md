# Training entry points

课程项目比较两个实例分割方法：YOLO Segmentation 和 Mask R-CNN。YOLO 入口保留框架边界；Mask R-CNN 入口已经包含 CarDD Dataset adapter、训练/验证 loss loop 和 checkpoint 保存。

为了公平比较，两个方法应使用相同的数据划分、类别映射、评测阈值和测试集。模型权重和 checkpoints 只保存在本地，`.gitignore` 与提交打包脚本会排除它们。

CarDD 的官方 COCO 标注可以先转换为 YOLO segmentation 格式：

```bash
python dataset/convert_coco_to_yolo.py \
  --source-dir data/raw/CarDD_release \
  --output-dir data/processed/cardd_yolo
```

转换默认创建图片 symlink，不会复制原始图片；如训练环境不支持 symlink，可增加 `--link-mode copy`。输出的 `data/processed/cardd_yolo/data.yaml` 可直接作为 `training/train_yolo.py --data-yaml` 的输入。

## Mask R-CNN 训练

先运行 COCO → YOLO 转换（用于 YOLO 方法），Mask R-CNN 直接读取 CarDD manifest：

```bash
python training/train_mask_rcnn.py \
  --manifest data/processed/cardd_manifest.json \
  --output-dir outputs/mask_rcnn \
  --epochs 20 \
  --image-size 800 \
  --batch-size 2 \
  --pretrained
```

`--pretrained` 会使用 torchvision COCO 权重，首次运行可能下载权重；不指定时从随机初始化开始。可先用 `--max-train-images 2 --max-val-images 1 --epochs 1 --image-size 256` 做 smoke test。训练输出包括 `mask_rcnn_best.pt`、逐 epoch checkpoint 和 `training_summary.json`，均被 Git 排除。
