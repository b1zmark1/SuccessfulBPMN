param(
    [string]$InputImage = "DATASET/BPMN/BPMN.png",
    [string]$OutRoot = "result/run_bpmn_full",
    [string]$PythonExe = "python",
    [string]$ModelPath = "narrator/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
    [string]$OutputFormat = "narrative",
    [int]$NCtx = 4096,
    [int]$Threads = 12,
    [double]$DetectorConf = 0.4,
    [int]$OcrMaxSide = 4096,
    [double]$OcrUpscaleFactor = 2.0,
    [int]$OcrPadPx = 10,
    [int]$OcrInnerCropPx = 0,
    [int]$OcrPsmBlock = 6,
    [int]$OcrPsmSingle = 7,
    [int]$OcrPsmRawLine = 13,
    [double]$OcrCcMaxAreaFrac = 0.18,
    [int]$OcrCcMinAreaPx = 20,
    [int]$OcrJobs = 4,
    [switch]$SkipNarrator
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CmdArgs
    )
    Write-Host ""
    Write-Host "==> $Name"
    if (-not $CmdArgs -or $CmdArgs.Count -eq 0) {
        throw "No arguments provided for step: $Name"
    }
    Write-Host "$PythonExe $($CmdArgs -join ' ')"
    & $PythonExe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$outYolox = Join-Path $OutRoot "out_yolox"
$outText = Join-Path $OutRoot "out_text"
$outOcr = Join-Path $OutRoot "out_ocr"
$outLabeled = Join-Path $OutRoot "out_labeled"

if (-not (Test-Path $OutRoot)) {
    New-Item -ItemType Directory -Path $OutRoot | Out-Null
}

Invoke-Step "1/5 YOLOX + text detect ensemble" @(
    "preprocanddetect/ensemble_infer.py",
    "--images", $InputImage,
    "--out", $outYolox,
    "--exp-file", "results/yolox_tiny_bpmn.py",
    "--ckpt", "results/best_ckpt.pth",
    "--dataset-root", "datasets/bpmn_full",
    "--conf", "$DetectorConf"
)

Invoke-Step "2/5 Text blocks detect/group" @(
    "preprocanddetect/detect_res.py",
    "--input", $InputImage,
    "--outdir", $outText
)

$modelImagePath = Join-Path $outText "00_model.png"
if (-not (Test-Path $modelImagePath)) {
    throw "Model-space image not found: $modelImagePath"
}

$ocrArgs = @(
    "preprocanddetect/ocr_tesseract_fast.py",
    "--input", $modelImagePath,
    "--blocks", (Join-Path $outText "text_blocks.json"),
    "--outdir", $outOcr,
    "--max-side", "$OcrMaxSide",
    "--upscale-factor", "$OcrUpscaleFactor",
    "--pad-px", "$OcrPadPx",
    "--inner-crop-px", "$OcrInnerCropPx",
    "--psm-block", "$OcrPsmBlock",
    "--psm-single", "$OcrPsmSingle",
    "--psm-raw-line", "$OcrPsmRawLine",
    "--try-rotate-90",
    "--refine-text-bbox",
    "--cc-max-area-frac", "$OcrCcMaxAreaFrac",
    "--cc-min-area-px", "$OcrCcMinAreaPx",
    "--jobs", "$OcrJobs"
)
Invoke-Step "3/5 OCR on text blocks" $ocrArgs

$inputStem = [System.IO.Path]::GetFileNameWithoutExtension($InputImage)
$ensembleJson = Join-Path $outYolox ("{0}_ensemble.json" -f $inputStem)
$mergedEnsemble = Join-Path $outLabeled ("{0}_ensemble_merged_labeled.json" -f $inputStem)

Invoke-Step "4/5 Label assign + auto merge ensemble" @(
    "preprocanddetect/label_res.py",
    "--ensemble", $ensembleJson,
    "--text-blocks", (Join-Path $outText "text_blocks.json"),
    "--ocr", (Join-Path $outOcr "ocr.json"),
    "--image", $modelImagePath,
    "--outdir", $outLabeled
)

if (-not (Test-Path $mergedEnsemble)) {
    throw "Merged ensemble file not found: $mergedEnsemble"
}

if (-not $SkipNarrator) {
    if ($OutputFormat -ne "narrative" -and $OutputFormat -ne "table") {
        throw "Invalid OutputFormat: $OutputFormat. Allowed: narrative, table"
    }
    $e2ePy = @"
import json
import sys
from pathlib import Path

ROOT = Path(r"$root")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_builder.pipeline import build_graph_from_ensemble
from narrator.orchestrator import run_narration

inp = Path(r"$mergedEnsemble")
out_dir = inp.parent
graph_path = out_dir / "graph_output.json"
narr_path = out_dir / "narration_output.json"
narr_txt = out_dir / "narration_text.txt"

obj = json.loads(inp.read_text(encoding="utf-8"))
graph = build_graph_from_ensemble(obj)
res = run_narration(
    graph_payload=graph,
    policy_overrides={
        "output_format": "$OutputFormat",
    },
    runtime_overrides={
        "model_path": r"$ModelPath",
        "n_ctx": $NCtx,
        "n_threads": $Threads
    }
)

graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
narr_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
narr_txt.write_text(str(res.get("text", "")), encoding="utf-8")

print("status:", res.get("status"))
print("graph_output:", graph_path)
print("narration_output:", narr_path)
print("narration_text:", narr_txt)
"@
    $tmpPy = Join-Path $outLabeled "_run_narrator_tmp.py"
    $e2ePy | Set-Content -Path $tmpPy -Encoding UTF8
    try {
        Invoke-Step "5/5 Graph builder + Narrator" @($tmpPy)
    }
    finally {
        if (Test-Path $tmpPy) {
            Remove-Item -Path $tmpPy -Force
        }
    }
}
else {
    Write-Host ""
    Write-Host "Skipped narrator stage (--SkipNarrator)."
}

Write-Host ""
Write-Host "Done."
Write-Host "Merged ensemble: $mergedEnsemble"
