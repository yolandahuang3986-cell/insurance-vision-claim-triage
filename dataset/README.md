# CarDD 数据准备

CarDD 是本课程项目的主数据集。数据集、标注文件和任何下载脚本产生的原始文件只保存在本地 `data/raw/CarDD_release/`，不会提交到 GitHub，也不会进入最终代码 zip。

## 使用约定

1. 小组成员根据已获得的数据集版本核对许可证和目录结构。
2. 将官方压缩包解压到 `data/raw/CarDD_release/`。本项目使用其中的 `CarDD_COCO/`，不使用 `CarDD_SOD/`。
3. 运行 `prepare_cardd.py` 读取 COCO 图片和实例分割标注：

```bash
python dataset/prepare_cardd.py \
  --source-dir data/raw/CarDD_release \
  --output data/processed/cardd_manifest.json
```

脚本保留 CarDD 官方 train/validation/test 划分，并把 COCO polygon、bbox、面积、类别和图片尺寸写入 manifest。不要重新随机切分官方测试集，也不要在没有完成真实训练和评测的情况下报告实验数字。

## 转换为 YOLO Segmentation

在生成 manifest 后，为 YOLO 训练生成 labels、`data.yaml` 和图片链接：

```bash
python dataset/convert_coco_to_yolo.py \
  --source-dir data/raw/CarDD_release \
  --output-dir data/processed/cardd_yolo
```

默认使用 symlink，不复制 5GB 以上的原始图片；如运行环境不支持 symlink，再使用 `--link-mode copy`。转换结果仍在 `data/processed/`，不会进入 Git。

## Manifest 最小结构

```json
{
  "dataset": "CarDD",
  "format": "COCO-instance-segmentation",
  "seed": 42,
  "classes": ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"],
  "images": [
    {
      "image_id": "train:1",
      "relative_path": "train2017/000001.jpg",
      "split": "train",
      "width": 1000,
      "height": 750,
      "annotations": [{"category_id": 2, "category": "scratch", "segmentation": [], "bbox": [0, 0, 1, 1]}]
    }
  ]
}
```
