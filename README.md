# Insurance Vision Claim Triage

车险图片理赔预审 MVP：对案件图片做质量门禁、损伤结构化、重复图片风险提示，并给出可解释的人工审核路由。

这个仓库与现有的宠物险服务项目完全独立。当前版本只验证工程边界和业务流程，使用 mock 图片元数据与 mock 视觉检测结果；它不自动核赔、不预测维修金额、不认定欺诈，也不替代理赔人员。

## MVP 目标

```text
Claim + images
    -> Image quality gate
    -> Damage instance adapter
    -> Duplicate-image signal
    -> Rule-based triage
    -> Auditable JSON result
```

输出包含每张图片的质量检查、损伤实例、风险信号、严重程度（light/medium/severe）和下一步动作：

- `continue_claim_assessment`：证据初步充分，可进入后续理赔评估；
- `request_more_evidence`：图片质量或完整性不足，请补拍/补传；
- `manual_review`：存在重复图片、低置信度或其他需要人工核验的风险信号。

严重程度只是视觉预审标签，不是赔付结论。

## 目录

```text
src/insurance_vision_claim_triage/
  config.py          环境变量配置
  models.py          输入/输出数据契约
  quality.py         图片质量与重复检查
  vision.py          可替换的视觉 provider 接口与 mock 实现
  triage.py          风险规则与路由
  pipeline.py        端到端 MVP pipeline
examples/mock_claim.json 可直接运行的示例案件
examples/run_mock.py 示例入口
tests/               单元测试
docs/architecture.md 设计边界和演进路线
```

## 运行

仓库运行时仅依赖 Python 标准库；测试使用 pytest：

```bash
cd insurance-vision-claim-triage
python examples/run_mock.py
python -m pytest -q
```

也可以从仓库根目录执行：

```bash
PYTHONPATH=src python examples/run_mock.py
```

## 模型替换边界

`VisionProvider` 是唯一的视觉模型接入边界。未来可以实现 `YoloSegmentationProvider`、`CarDDProvider` 或远程推理 provider，而不用改动数据模型、质量门禁和路由规则。本 MVP 默认 provider 为 `mock`，没有下载权重或外部 API 调用。

## 当前明确不做

- 不接入真实图片存储或上传服务；
- 不训练或下载 CarDD/YOLO 权重；
- 不输出维修金额或自动批准/拒绝理赔；
- 不把图片相似度直接称为“欺诈”；
- 不把课程中的 segmentation 概念硬编码为业务规则。

## 下一步（需单独确认）

1. 确认车险业务字段与拍摄视角要求；
2. 选择真实数据集及其许可证；
3. 选择 YOLO/Mask R-CNN/SAM 等模型和部署方式；
4. 以标注集补充 mask IoU、damage recall、duplicate false-alert 等评测；
5. 再决定是否接入前端、对象存储、向量库或报告模型。
