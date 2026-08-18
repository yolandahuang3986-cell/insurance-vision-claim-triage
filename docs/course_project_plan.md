# CA6129 课程项目计划

## 研究问题

在相同 CarDD 测试划分和类别映射下，YOLO Segmentation 与 Mask R-CNN 哪一种更适合保险理赔图片的车损实例分割预审？比较重点是分割质量、召回率、推理延迟和错误类型，而不是自动核赔金额。

## 公平比较协议

- 固定随机种子：42；
- 固定 train/validation/test 比例：70%/15%/15%；
- 两个方法使用相同的类别映射与测试集；
- 记录输入分辨率、训练轮数、硬件、置信度阈值和 IoU 匹配阈值；
- 报告 mask IoU、precision、recall、mAP@50、mAP@50:95 和单图 latency；
- 额外抽样分析漏检、错分类、边界定位差、低质量图片和低置信度正确预测。

## 结果表（训练完成后填写）

| Method | mask IoU | Precision | Recall | mAP@50 | mAP@50:95 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO Segmentation | TBD | TBD | TBD | TBD | TBD | TBD |
| Mask R-CNN | TBD | TBD | TBD | TBD | TBD | TBD |

任何 `TBD` 都必须在真实实验后替换，不能用 mock 结果代替。
