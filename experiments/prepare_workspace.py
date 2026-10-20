"""Create the documented runtime layout without copying data or weights into Git.

This only links trusted local inputs. It never downloads or deserializes weights.
Relative config paths are resolved against the config file's parent directory.
"""
import argparse
import json
from pathlib import Path

WEIGHT_LAYOUT = {
    "historical_yolo640": "yolo_best.pt",
    "historical_mask800": "mask_rcnn_final.pt",
    "retrained_mask640": "results/retrain640/checkpoints/best.pt",
    "retrained_mask800": "results/retrain800/checkpoints/best.pt",
    "retrained_yolo640": "results/yolo_pair_20261009/yolo640/training/weights/best.pt",
    "retrained_yolo800": "results/yolo_pair_20261009/yolo800/training/weights/best.pt",
}


def prepare(config_path, output):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return (config_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    data = resolve(config["dataset_root"])
    inputs = {target: resolve(config["weights"][key]) for key, target in WEIGHT_LAYOUT.items()}
    for split in ["train", "val", "test"]:
        assert (data / f"{split}2017").is_dir(), f"Missing images: {split}2017"
        annotations = data / f"annotations/instances_{split}2017.json"
        assert annotations.is_file(), f"Missing annotations: {annotations}"
        inputs[f"evidence/instances_{split}2017.json"] = annotations
    inputs["train_mask_rcnn.py"] = Path(__file__).with_name("legacy_maskrcnn.py").resolve()
    for target, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(f"{target} requires trusted local file: {source}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh runtime directory; no files will be overwritten")
    output.mkdir(parents=True)
    for target, source in inputs.items():
        link = output / target
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    (output / "data").mkdir()
    (output / "data/dataset_root.txt").write_text(str(data) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.config, args.output))


if __name__ == "__main__":
    main()
