# run_gpu.ps1 -- Run the J1 pipeline with GPU acceleration.
#
# Why this script exists instead of `conda activate j1_gpu; python main.py`:
#   - `conda run -n j1_gpu ...` is flaky on this machine (crashes with an
#     internal conda error as soon as heavier packages like tensorflow are
#     imported) -- see README_gpu_setup.md. Direct invocation of the env's
#     python.exe with its Library\bin prepended to PATH (so the CUDA 11.2 /
#     cuDNN 8.1 DLLs conda installed are found) works reliably instead.
#   - PYTHONNOUSERSITE=1 avoids a protobuf version conflict: this machine
#     has a newer TensorFlow installed in the Windows user site-packages
#     directory (%APPDATA%\Python\Python310\site-packages) that silently
#     shadows the conda env's own TensorFlow 2.10 unless user-site is
#     disabled.
#
# Usage: .\run_gpu.ps1 [--mode full|quick|predict_only]

$ErrorActionPreference = "Stop"
$EnvDir = "C:\Users\soufi\miniconda3\envs\j1_gpu"

if (-not (Test-Path "$EnvDir\python.exe")) {
    throw "GPU conda env not found at $EnvDir -- see README_gpu_setup.md to recreate it."
}

$env:PYTHONNOUSERSITE = "1"
$env:PATH = "$EnvDir\Library\bin;$EnvDir;$EnvDir\Scripts;" + $env:PATH

& "$EnvDir\python.exe" (Join-Path $PSScriptRoot "main.py") @args
