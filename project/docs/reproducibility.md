# Reproducibility Guide

## 1. Fixed Random Seeds

| Component | Seed Value | Notes |
|---|---|---|
| Data splitting | 42 | Fixed for all experiments |
| Model training | [42, 123, 456, 789, 1024] | 5 seeds per experiment |
| NumPy global | Set to model seed | `np.random.seed(seed)` |
| PyTorch global | Set to model seed | `torch.manual_seed(seed)` |
| CUDA | Set to model seed | `torch.cuda.manual_seed_all(seed)` |
| Python hash | 0 | `PYTHONHASHSEED=0` |

### Seed Utility

```python
# src/utils/seed.py
def set_seed(seed: int = 42) -> None:
    import random, numpy as np, torch, os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 2. Persisted Artifacts

Every experiment produces the following reproducibility artifacts:

| Artifact | Purpose |
|---|---|
| `data/splits/split_manifest.csv` | Exact train/val/test assignment for each sample |
| `configs/*.yaml` | All hyperparameters and settings |
| `artifacts/models/{id}_best.pt` | Model weights at best validation loss |
| `artifacts/metrics/{id}_config.yaml` | Frozen config snapshot for this run |
| `artifacts/metrics/{id}_training_log.csv` | Per-epoch loss and metrics |
| `artifacts/metrics/{id}_test_results.json` | Final test set results |

## 3. Software Environment

### Pinned Dependencies
All dependencies are specified in `requirements.txt` with version constraints.

### Python Version
Python 3.10+ required. Tested on Python 3.12.

### Operating System
Developed and tested on Windows 11. Should work on Linux/macOS with path adjustments.

## 4. Data Provenance

| Item | Source |
|---|---|
| Raw data | MLOmics Pan-cancer (Zenodo archive 28729127) |
| Label verification | GDC API batch queries (script: `verify_tcga_labels.py`) |
| Data integrity | Verified by `scripts/audit_dataset.py` |

Raw data is NEVER modified. All processing creates new files.

## 5. Reproducing Results

```powershell
# 1. Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure data path
# Edit configs/data.yaml to point to your MLOmics data directory

# 3. Run audit
python scripts/audit_dataset.py

# 4. Prepare data and splits
python scripts/prepare_dataset.py

# 5. Run baselines
python scripts/run_baselines.py

# 6. Run missing-modality experiments
python scripts/run_missing_modality_experiments.py
```

## 6. Known Sources of Non-Determinism

Even with fixed seeds, the following can introduce minor variation:
- CUDA floating-point atomics (non-deterministic reductions on GPU)
- Different GPU hardware (different rounding behavior)
- Python dictionary ordering (deterministic since Python 3.7, but set operations are not)
- Multi-threaded data loading (ordering variation)

**Mitigation:** Set `torch.backends.cudnn.deterministic = True` and use `num_workers=0` for data loading during reproducibility-critical runs.
