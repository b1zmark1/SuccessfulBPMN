import argparse
from pathlib import Path


EXP_TEMPLATE = """from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = {num_classes}
        self.data_dir = r\"{data_dir}\"
        self.train_ann = \"annotations/instances_train.json\"
        self.val_ann = \"annotations/instances_val.json\"
        self.test_ann = \"annotations/instances_test.json\"
        self.input_size = ({img_size}, {img_size})
        self.test_size = ({img_size}, {img_size})
"""


def _load_num_classes(dataset_root: Path) -> int:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    return len([line for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolox-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--exp-name", default="yolox_tiny_bpmn")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--num-classes", type=int, default=None)
    args = ap.parse_args()

    yolox_dir = Path(args.yolox_dir)
    dataset_root = Path(args.dataset_root)
    num_classes = args.num_classes or _load_num_classes(dataset_root)

    exp_dir = yolox_dir / "exps" / "bpmn"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_path = exp_dir / f"{args.exp_name}.py"

    exp_path.write_text(
        EXP_TEMPLATE.format(
            num_classes=num_classes,
            data_dir=str(dataset_root),
            img_size=args.img_size,
        ),
        encoding="utf-8",
    )
    print(str(exp_path))


if __name__ == "__main__":
    main()
