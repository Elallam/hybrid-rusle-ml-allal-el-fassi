# GPU setup for Article J1

This machine has an NVIDIA GeForce GTX 1650 Ti (4 GB VRAM). WSL2 is not
enabled (needs a BIOS virtualization change), so native-Windows GPU support
was used instead of the tensorflow-directml or WSL2 routes.

## What runs on GPU

| Library | GPU support | How |
|---|---|---|
| XGBoost | Yes | `device='cuda'` -- works in ANY environment (the pip wheel bundles its own CUDA runtime). Already enabled in `config.py` (`USE_GPU = True`). |
| CatBoost | Yes | `task_type='GPU'` -- same, bundles its own CUDA runtime. Already enabled in `config.py`. |
| TensorFlow (ANN-Residual, CNN-LSTM-Attention, Transformer) | Yes, but only in the `j1_gpu` conda env | TensorFlow >=2.11 dropped native-Windows GPU support entirely. TF 2.10.1 is the last version with it, and it needs an external CUDA 11.2 + cuDNN 8.1 install that only conda provides cleanly (no NVIDIA Developer portal login needed via conda-forge). |
| Random Forest, Gradient Boosting (sklearn) | No | sklearn has no GPU tree implementation. Stay CPU-bound; this is fine, they train in well under a second. |

## The `j1_gpu` conda environment

Cloned from the sibling C2 project's already-working `c2ml_gpu` env
(`conda create --name j1_gpu --clone c2ml_gpu`), then `catboost` was added
via pip. Key pinned versions (older than the CPU-only `j1ml` venv, because
TF 2.10 predates numpy 2.0 / Keras 3):

```
python        3.10.20
tensorflow    2.10.1   (last native-Windows-GPU release)
cudatoolkit   11.2.2   (conda-forge)
cudnn         8.1.0.77 (conda-forge)
numpy         1.23.5
scikit-learn  1.3.2
xgboost       2.1.4
catboost      1.2.10   (added on top of the clone)
```

To recreate it from scratch on another machine (no `c2ml_gpu` to clone):

```
conda create -n j1_gpu -c conda-forge python=3.10 cudatoolkit=11.2.2 cudnn=8.1.0.77
conda run -n j1_gpu pip install tensorflow==2.10.1 numpy==1.23.5 pandas scikit-learn==1.3.2 ^
    xgboost==2.1.4 catboost shap rasterio geopandas statsmodels seaborn matplotlib joblib tqdm
```

## Two gotchas that cost real debugging time

1. **`conda run -n j1_gpu <cmd>` crashes** as soon as the command imports
   tensorflow together with catboost/xgboost/rasterio -- it exits with a
   generic "An unexpected error has occurred" conda report, which looks
   like a conda bug but is actually the *subprocess* dying and conda
   mis-reporting it. Direct invocation of `envs\j1_gpu\python.exe` with the
   env's `Library\bin` prepended to `PATH` (so the CUDA/cuDNN DLLs are
   found) does NOT crash -- this is what `run_gpu.ps1` does. Root cause was
   never fully isolated; it's specific to `conda run`'s wrapper, not a real
   TensorFlow/CatBoost DLL conflict (confirmed by the direct invocation
   working cleanly with both imported in the same process).

2. **A stray user-site TensorFlow 2.16.1** lives in
   `%APPDATA%\Roaming\Python\Python310\site-packages` (leftover from some
   earlier `pip install --user`) and silently shadows the conda env's own
   TensorFlow 2.10.1 unless `PYTHONNOUSERSITE=1` is set -- surfaces as
   `ImportError: cannot import name 'builder' from 'google.protobuf.internal'`
   (a protobuf version mismatch between the two TF installs). `run_gpu.ps1`
   sets this automatically.

## Running

```
.\run_gpu.ps1 --mode quick   # smoke test, ~10 min
.\run_gpu.ps1 --mode full    # real run, n=3000, full hyperparameter search, 300 DL epochs
```

The CPU-only `C:\Users\soufi\venvs\j1ml` venv (see `requirements.txt`) still
works and is a safe fallback if the GPU env ever breaks -- `config.USE_GPU`
only toggles XGBoost/CatBoost's device flags; TensorFlow always uses
whatever GPU (if any) is visible in the environment it's running in.
