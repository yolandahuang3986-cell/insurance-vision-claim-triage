# CarDD 数据准备

CarDD 是本课程项目的主数据集。数据集、标注文件和任何下载脚本产生的原始文件只保存在本地 `data/raw/CarDD/`，不会提交到 GitHub，也不会进入最终代码 zip。

## 使用约定

1. 小组成员根据已获得的数据集版本核对许可证和目录结构。
2. 将图片和标注放入 `data/raw/CarDD/`。
3. 运行 `prepare_cardd.py` 生成可复现的图片 manifest：

```bash
python dataset/prepare_cardd.py \
  --source-dir data/raw/CarDD \
  --output data/processed/cardd_manifest.json
```

当前脚本只建立图片索引和 train/validation/test 划分；CarDD 的具体 mask 标注布局需要按本地版本接入 `annotations` 字段后，才能用于真实训练。不要在没有验证标注映射的情况下报告实验数字。

## Manifest 最小结构

```json
{
  "dataset": "CarDD",
  "seed": 42,
  "classes": ["dent", "scratch", "crack", "glass_shatter"],
  "images": [
    {
      "image_id": "example-001",
      "relative_path": "images/example-001.jpg",
      "split": "train",
      "annotations": []
    }
  ]
}
```
