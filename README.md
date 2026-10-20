# Insurance Vision Claim Triage — CA6129 Group 15

车险图片理赔预审 MVP：对案件图片做质量门禁、损伤结构化、重复图片风险提示，并给出可解释的人工审核路由。

这个仓库与现有的宠物险服务项目完全独立。当前版本只验证工程边界和业务流程，使用 mock 图片元数据与 mock 视觉检测结果；它不自动核赔、不预测维修金额、不认定欺诈，也不替代理赔人员。

## CA6129 课程项目轨道

本项目的正式技术评审题目是：**Vehicle Damage Instance Segmentation for Insurance Claim Triage**。

- 主数据集：CarDD（数据集只在本地准备，不提交到 GitHub 或课程代码包）；
- 对比方法：YOLO Segmentation 与 Mask R-CNN；
- 统一评测：mask IoU、precision、recall、mAP@50、mAP@50:95、单图延迟，并进行按错误类型的分析；
- 研究边界：模型输出只用于视觉证据预审，不直接作出赔付或拒赔决定。

课程报告中的训练与补充实验已经完成。`experiments/` 包含统一评估、尺寸重训练、组件分析、验证集重建选择、阈值校准、bootstrap 和重复计时代码。应用仍用 mock provider，未接入真实权重。历史 YOLO 逐 epoch 日志未找回；后续重训练日志已保留。

## 交付与复现导航

- [实验与代码对应表](docs/experiment_map.md)
- [完整运行指南](docs/reproduction.md)：安装环境、数据目录、运行顺序和输出位置
- [AI 与使用许可](docs/academic_and_licensing.md)
- [第三方说明](THIRD_PARTY_NOTICES.md)

无 GPU 核验（Python 3.12 为本次交付测试环境）：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install 'pytest>=8,<9' -r experiments/requirements-analysis.txt
python examples/run_mock.py
python -m pytest -q
python -m unittest discover -s experiments -p 'test_*.py' -v
python reporting/verify_metrics.py
python reporting/plot_metrics.py --output outputs/report_figures
```

`reporting/metrics.json` 是带来源哈希的汇总：九种配置的指标、区间、组件分析、验证集选择、五轮计时与训练曲线数值。不含图片、原始标注、逐实例掩码或权重。汇总算术核验不等于重新训练或独立重算 AP。

GPU 实验另建 Linux/CUDA 环境。记录环境为 Python 3.13.5、A40、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Ultralytics 8.4.138；依赖见 `experiments/requirements-gpu.txt`。不要将 CUDA 依赖装到 Mac 的 mock 环境。

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
python scripts/build_submission_zip.py --group-id 15
```

脚本生成 `submissions/group_15_code.zip` 和 `SUBMISSION_MANIFEST.json`，扫描 JSON 数据载荷、云盘替代文件链接与文件大小，排除图片、数据、权重、运行输出和符号链接，验证 ZIP 并记录文件哈希。按课程“不得提供外部链接”的要求，打包文件中出现任何 http(s) URL 都会使构建失败。

## 当前明确不做

- 不接入真实图片存储或上传服务；
- mock 应用启动时不训练或下载模型；研究训练由独立入口执行；
- 不输出维修金额或自动批准/拒绝理赔；
- 不把图片相似度直接称为“欺诈”；
- 不把课程中的 segmentation 概念硬编码为业务规则。

## 提交前仍须确认

1. 核对报告、PPT 与独立贡献说明中的成员标识及实际贡献；
2. 核实 CarDD 的事先研究授权，以及公开的 `CarDD project/manifests/cardd_manifest.json` 标注与 Git 历史中旧图片样例的再分发权限；
3. 从 CA6129 的 NTULearn 或教师指引确认 AI 使用范围、声明格式与提交位置；
4. 提交两份 PDF 与代码 ZIP，GitHub 或演示链接不能代替课程附件；
5. 保留测试集曾暴露、历史 Mask R-CNN 测试集选点、单种子等局限。计时衡量低分数门槛下的 RGB-to-RLE 流水线成本，不是单独模型前向速度。
