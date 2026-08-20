# Insurance Vision Claim Triage

车险图片理赔预审 MVP：对案件图片做质量门禁、损伤结构化、重复图片风险提示，并给出可解释的人工审核路由。

这个仓库与现有的宠物险服务项目完全独立。当前版本只验证工程边界和业务流程，使用 mock 图片元数据与 mock 视觉检测结果；它不自动核赔、不预测维修金额、不认定欺诈，也不替代理赔人员。

## CA6129 课程项目轨道

本项目的正式技术评审题目是：**Vehicle Damage Instance Segmentation for Insurance Claim Triage**。

- 主数据集：CarDD（数据集只在本地准备，不提交到 GitHub 或课程代码包）；
- 对比方法：YOLO Segmentation 与 Mask R-CNN；
- 统一评测：mask IoU、precision、recall、mAP@50、mAP@50:95、单图延迟，并进行按错误类型的分析；
- 研究边界：模型输出只用于视觉证据预审，不直接作出赔付或拒赔决定。

课程轨道的实验脚手架、数据准备入口、报告模板和离线打包工具已经放在 `dataset/`、`training/`、`evaluation/`、`docs/` 和 `scripts/`。真实模型训练需要成员在本地安装可选依赖并填入数据集的实际标注格式；仓库不会下载或保存权重。

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
dataset/             CarDD 本地数据准备说明与 manifest 生成器
training/            YOLO Segmentation / Mask R-CNN 训练入口与 CarDD Dataset adapter
evaluation/          指标、方法对比和错误分类
configs/             可复现实验配置
scripts/             离线课程提交包构建工具
examples/mock_claim.json 可直接运行的示例案件
examples/run_mock.py 示例入口
tests/               单元测试
docs/architecture.md 设计边界和演进路线
docs/course_project_plan.md 课程项目实验计划
docs/report_template.md 报告结构模板
docs/contribution_template.md 成员贡献说明模板
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

## 离线课程提交包

课程要求代码包不包含数据集、模型权重、云端文件或外部链接。完成本地实验后，可在仓库根目录执行：

```bash
PYTHONPATH=src python scripts/build_submission_zip.py --group-id XX
```

脚本会生成 `submissions/group_XX_code.zip` 和一份内容清单；其中 `XX` 替换为课程分配的组号。

## 当前明确不做

- 不接入真实图片存储或上传服务；
- 不训练或下载 CarDD/YOLO 权重；
- 不输出维修金额或自动批准/拒绝理赔；
- 不把图片相似度直接称为“欺诈”；
- 不把课程中的 segmentation 概念硬编码为业务规则。

## 当前仍需由小组补齐

1. 按课程要求填写成员的学生卡全名、学号和贡献记录；
2. 根据 CarDD 实际下载版本核对类别映射与标注格式；
3. 在同一数据划分和随机种子下完成两个方法的训练与评测；
4. 将真实结果填入 `docs/report_template.md`，不要把示例或未运行的数字当作实验结果。
