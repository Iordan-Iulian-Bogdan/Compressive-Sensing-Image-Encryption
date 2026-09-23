# Downloads CLIP ONNX models + tokenizer (models/clip/) and the ONNX
# Runtime binaries (third_party/onnxruntime/). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\download_clip.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$dir = Join-Path $PSScriptRoot "..\models\clip"
New-Item -ItemType Directory -Force $dir | Out-Null
$base = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
$files = @(
    @{ url = "$base/vocab.json";                        out = "vocab.json" },
    @{ url = "$base/merges.txt";                        out = "merges.txt" },
    @{ url = "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx"; out = "qdrant_vision.onnx" },
    @{ url = "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/main/model.onnx";   out = "qdrant_text.onnx" }
)
foreach ($f in $files) {
    $dest = Join-Path $dir $f.out
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
        Write-Host "skip  $($f.out) (exists)"
        continue
    }
    Write-Host "fetch $($f.out) ..."
    Invoke-WebRequest -Uri $f.url -OutFile $dest -MaximumRedirection 5
    Write-Host "  ok   $((Get-Item $dest).Length) bytes"
}
# ONNX Runtime (inference backend for the scorer)
$ort = Join-Path $PSScriptRoot "..\third_party\onnxruntime"
if (-not ((Test-Path (Join-Path $ort "lib\onnxruntime.dll")) -and (Test-Path (Join-Path $ort "include\onnxruntime_cxx_api.h")))) {
    New-Item -ItemType Directory -Force $ort | Out-Null
    $zip = Join-Path ([IO.Path]::GetTempPath()) "ort_clip.zip"
    Write-Host "fetch onnxruntime ..."
    Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip" -OutFile $zip -MaximumRedirection 5
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "ort_clip_ex"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Move-Item (Join-Path $inner.FullName "include") (Join-Path $ort "include") -Force
    Move-Item (Join-Path $inner.FullName "lib") (Join-Path $ort "lib") -Force
    Remove-Item -Recurse -Force $tmp
    Remove-Item -Force $zip
    Write-Host "  ok   $ort"
} else {
    Write-Host "skip  onnxruntime (exists)"
}
Write-Host "done -> $((Resolve-Path $dir).Path)"
