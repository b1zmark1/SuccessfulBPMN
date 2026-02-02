#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import torch


def _add_yolox_to_path(root_dir):
    yolox_dir = os.path.join(root_dir, "YOLOX")
    if os.path.isdir(yolox_dir) and yolox_dir not in sys.path:
        sys.path.insert(0, yolox_dir)


def _get_image_list(path):
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for base, _, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in image_ext:
                out.append(os.path.join(base, name))
    out.sort()
    return out


def _load_class_names(dataset_root):
    classes_path = os.path.join(dataset_root, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    try:
        from yolox.data.datasets import COCO_CLASSES
        return COCO_CLASSES
    except Exception:
        return None


class Predictor:
    def __init__(self, model, exp, cls_names, device, fp16, legacy):
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess

        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.postprocess = postprocess

    def inference(self, img):
        if img is None:
            return None, None
        h, w = img.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)
        img_info = {"height": h, "width": w, "ratio": ratio, "raw_img": img}

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visualize(self, output, img_info, conf_override=None):
        from yolox.utils import vis

        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= img_info["ratio"]

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        cls_conf = self.confthre if conf_override is None else conf_override
        return vis(img, bboxes, scores, cls, cls_conf, self.cls_names)


def main():
    parser = argparse.ArgumentParser("Render YOLOX inference bboxes on test images")
    parser.add_argument("--exp-file", default="results/yolox_tiny_bpmn.py")
    parser.add_argument("--ckpt", default="results/best_ckpt.pth")
    parser.add_argument("--dataset-root", default="datasets/bpmn_full")
    parser.add_argument("--images", default="datasets/bpmn_full/images/test")
    parser.add_argument("--out", default="results/images")
    parser.add_argument("--device", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--nms", type=float, default=None)
    parser.add_argument("--tsize", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _add_yolox_to_path(repo_root)

    from yolox.exp import get_exp
    from yolox.utils import fuse_model
    from preprocanddetect.preprocess import preprocess, PreprocessConfig

    if args.device is None:
        args.device = "gpu" if torch.cuda.is_available() else "cpu"

    exp = get_exp(args.exp_file, None)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)

    cls_names = _load_class_names(args.dataset_root)
    predictor = Predictor(model, exp, cls_names, args.device, args.fp16, args.legacy)

    image_list = _get_image_list(args.images)
    if args.limit and args.limit > 0:
        image_list = image_list[: args.limit]

    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    saved = 0
    pre_cfg = PreprocessConfig()
    for img_path in image_list:
        pre = preprocess(img_path, pre_cfg)
        img_for_model = pre["model_bgr"]
        outputs, img_info = predictor.inference(img_for_model)
        if img_info is None:
            continue
        vis_img = predictor.visualize(outputs[0] if outputs else None, img_info)

        rel = os.path.relpath(img_path, args.images)
        out_path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis_img)
        saved += 1

    dt = time.time() - t0
    print(f"Saved {saved} images to {args.out} in {dt:.1f}s")


if __name__ == "__main__":
    main()
