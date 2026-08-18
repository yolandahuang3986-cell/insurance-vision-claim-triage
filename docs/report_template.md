# group_XX_report

## Abstract

说明保险理赔图片预审问题、比较的两个实例分割方法和主要发现。

## 1. Problem and motivation

说明车损实例分割如何帮助人工理赔预审，并明确不自动决定赔付。

## 2. Dataset and protocol

记录 CarDD 版本、类别、样本数量、划分、许可证、预处理、随机种子和硬件。不要把数据集复制进代码包。

## 3. Methods

分别说明 YOLO Segmentation 与 Mask R-CNN 的架构、训练设置、输入分辨率和可替换实现边界。

## 4. Evaluation and comparison

报告 mask IoU、precision、recall、mAP@50、mAP@50:95、latency，并解释指标定义。

## 5. Error analysis

展示漏检、错分类、定位不准、低质量输入和低置信度正确预测的例子与原因。

## 6. Insurance triage workflow

说明视觉输出如何进入质量门禁、重复图提示和人工审核路由；不要把视觉模型输出写成最终理赔结论。

## 7. Limitations and future work

讨论数据偏差、遮挡、反光、小划痕、域迁移和真实部署风险。

## References

按课程要求列出数据集和方法论文/官方文档。提交代码包本身不应依赖外部云盘链接。
