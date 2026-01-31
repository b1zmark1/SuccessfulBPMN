# YOLOX Tiny (BPMN/UML)

## 1) Convert YOLO txt to COCO
```powershell
python tools\\yolox\\convert_yolo_to_coco.py --dataset-root datasets\\bpmn_full
```
Outputs:
- `datasets\\bpmn_full\\annotations\\instances_train.json`
- `datasets\\bpmn_full\\annotations\\instances_val.json`
- `datasets\\bpmn_full\\annotations\\instances_test.json`

## 2) Install YOLOX
```powershell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -r requirements.txt
pip install -v -e .
```

## 3) Create experiment file for this dataset
```powershell
python ..\\tools\\yolox\\create_yolox_exp.py --yolox-dir . --dataset-root ..\\datasets\\bpmn_full --img-size 1024
```
This creates: `YOLOX\\exps\\bpmn\\yolox_tiny_bpmn.py`

## 4) Fine-tune YOLOX-Tiny (T4)
```powershell
python tools\\train.py -f exps\\bpmn\\yolox_tiny_bpmn.py -d 1 -b 8 --fp16 -o -c yolox_tiny.pth
```

## 5) Export to ONNX (for CPU inference)
```powershell
python tools\\export_onnx.py -f exps\\bpmn\\yolox_tiny_bpmn.py -c <PATH_TO_BEST_CKPT> --output-name yolox_tiny_bpmn.onnx
```

## 6) Evaluate
```powershell
python tools\\eval.py -f exps\\bpmn\\yolox_tiny_bpmn.py -c <PATH_TO_BEST_CKPT> -b 1 -d 1
```
