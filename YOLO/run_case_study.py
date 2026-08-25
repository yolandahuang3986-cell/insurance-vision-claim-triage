"""
在本机项目根目录（能看到 data/ 和 runs/ 文件夹的那一层）运行本脚本：

    python run_case_study.py

会自动完成：
1. 加载训练好的模型 (best.pt)
2. 对 test 集所有图片跑一遍推理，保存可视化结果
3. 找出标注里含 crack（类别索引 2）的 test 图片
4. 把前 5 张对应的预测结果图复制到 case_study_crack/ 文件夹

跑完后，把 case_study_crack 文件夹整个发回来即可。
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

# 1. 加载训练好的模型
model = YOLO("runs/segment/runs/yolo_segmentation/weights/best.pt")

# 2. 对 test 集所有图片跑一遍推理，保存可视化结果
predict_results = model.predict(
    source="data/yolo/images/test",
    conf=0.5,
    save=True,
    project="runs/segment",
    name="test_predictions",
    exist_ok=True,
)

# 实际保存目录以 Ultralytics 返回的 save_dir 为准，避免本地 Ultralytics
# settings（runs_dir 等）导致的路径与预期不一致，从而复制不到任何文件。
# 不同版本的 Ultralytics，save_dir 有时是 str，有时是 Path，这里统一转成 Path。
pred_dir = Path(predict_results[0].save_dir) if predict_results else Path("runs/segment/test_predictions")
print(f"预测结果实际保存在: {pred_dir}")

# 3. 自动找出标注里含 crack（类别索引 2）的 test 图片
label_dir = Path("data/yolo/labels/test")
crack_images = []
for label_file in label_dir.glob("*.txt"):
    with open(label_file) as f:
        lines = f.readlines()
    if any(line.startswith("2 ") for line in lines):
        crack_images.append(label_file.stem)

print(f"找到 {len(crack_images)} 张含 crack 标注的 test 图片")

# 4. 把前 5 张对应的预测结果图自动复制到一个新文件夹
out_dir = Path("case_study_crack")
out_dir.mkdir(exist_ok=True)

copied = 0
for name in crack_images[:5]:
    src = pred_dir / f"{name}.jpg"
    if src.exists():
        shutil.copy(src, out_dir / f"{name}.jpg")
        copied += 1

print(f"已复制 {copied} 张预测结果图到 case_study_crack/ 文件夹")
print("请把 case_study_crack 这个文件夹上传给我")
