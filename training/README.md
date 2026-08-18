# Training entry points

课程项目比较两个实例分割方法：YOLO Segmentation 和 Mask R-CNN。两个入口只负责建立模型和实验输出目录；数据适配、优化循环、权重缓存和硬件选择必须在小组本地完成，并记录到报告中。

为了公平比较，两个方法应使用相同的数据划分、类别映射、评测阈值和测试集。模型权重和 checkpoints 只保存在本地，`.gitignore` 与提交打包脚本会排除它们。

CarDD manifest 不是任何框架的最终配置文件。运行真实训练前，应把 manifest 中的图片路径和 mask 标注转换为对应框架所需的 YOLO YAML/labels 或 torchvision Dataset 适配器。
