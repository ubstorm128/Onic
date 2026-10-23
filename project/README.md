# A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities

## Overview

This research project develops and evaluates a multimodal machine learning framework that classifies four cancer types using four omics modalities (mRNA expression, miRNA expression, copy number variation, and DNA methylation) and remains robust when one or more modalities are unavailable at inference time.

The framework addresses a critical clinical challenge: in real-world settings, not all molecular assays may be available for every patient due to cost, tissue availability, or technical failures. A model that degrades gracefully under missing modalities is more practically useful than one requiring all inputs.

## Research Objective

1. **Multi-cancer classification:** Accurately distinguish BRCA, HNSC, LGG, and THCA using multimodal omics data
2. **Missing-modality robustness:** Maintain classification performance when one or more modalities are absent
3. **Modality fusion:** Evaluate whether cross-modal integration improves over single-modality and simple concatenation baselines
4. **Explainability:** Identify which modalities and features drive classification decisions

## Dataset

- **Source:** MLOmics Pan-cancer Classification Dataset (TCGA-derived)
- **Samples:** 2,263 patients across 4 cancer types
- **Modalities:** mRNA (3,217 features), miRNA (383), CNV (3,105), DNA Methylation (3,139)
- **Class distribution:** BRCA=757, HNSC=506, LGG=506, THCA=494

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for complete data documentation.

## Project Status

| Stage | Status | Description |
|---|---|---|
| A: Dataset Audit | ✅ Complete | All 26 checks passed |
| B: Documentation | ✅ Complete | All required docs and protocols created |
| C: Data Pipeline | ✅ Complete | Filtering, stratified splitting, leak-free preprocessing |
| D: Baselines | ✅ Complete | 6 single-modality and early fusion baselines |
| E: Fusion Model | ✅ Complete | Multi-head cross-attention fusion architecture |
| F: Missing Modality | ✅ Complete | All 15 missingness conditions (C0–C14) evaluated |
| G: Explainability | ✅ Complete | Modality ablation/permutation & biomarker discovery |


## Repository Structure

```
project/
├── README.md                   # This file
├── implementation.md           # Central implementation blueprint
├── requirements.txt            # Python dependencies
├── CHANGELOG.md                # Version history
├── .gitignore                  # Git ignore rules
├── configs/                    # YAML configuration files
│   ├── data.yaml
│   ├── model.yaml
│   └── experiment.yaml
├── docs/                       # Research documentation
│   ├── dataset_audit.md
│   ├── data_dictionary.md
│   ├── research_questions.md
│   ├── experimental_protocol.md
│   ├── model_architecture.md
│   ├── missing_modality_protocol.md
│   ├── evaluation_protocol.md
│   ├── explainability_protocol.md
│   ├── reproducibility.md
│   ├── limitations_and_risks.md
│   ├── decision_log.md
│   └── experiment_log.md
├── src/                        # Source code
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model architectures
│   ├── training/               # Training and evaluation loops
│   ├── explainability/         # Explanation methods
│   └── utils/                  # Utilities (config, seed, logging)
├── scripts/                    # Executable scripts
│   ├── audit_dataset.py
│   ├── prepare_dataset.py
│   ├── run_baselines.py
│   └── run_missing_modality_experiments.py
├── tests/                      # Automated tests
├── reports/                    # Generated reports and metrics
├── artifacts/                  # Saved models, figures, metrics
│   ├── models/
│   ├── metrics/
│   └── figures/
└── data/                       # Processed data (not tracked in git)
    ├── processed/
    └── splits/
```

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- ~8 GB disk space for source data (not included in repo)

### Setup

```powershell
# Clone the repository
git clone <repo-url>
cd project

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

The raw data is NOT included in the repository. Configure the data path:

1. Set the data directory in `configs/data.yaml`
2. Or set the environment variable: `$env:MLOMICS_DATA_DIR = "path/to/Pan-cancer/Original"`

### Running

```powershell
# Stage A: Audit dataset
python scripts/audit_dataset.py

# Stage C: Prepare filtered dataset and splits
python scripts/prepare_dataset.py

# Stage D: Run baseline models
python scripts/run_baselines.py

# Stage F: Run missing-modality experiments
python scripts/run_missing_modality_experiments.py
```

## Documentation

| Document | Description |
|---|---|
| [`implementation.md`](implementation.md) | Central implementation blueprint |
| [`docs/dataset_audit.md`](docs/dataset_audit.md) | Dataset verification report |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | File and column reference |
| [`docs/research_questions.md`](docs/research_questions.md) | Research questions and hypotheses |
| [`docs/experimental_protocol.md`](docs/experimental_protocol.md) | Experiment design and procedures |
| [`docs/model_architecture.md`](docs/model_architecture.md) | Architecture details and rationale |
| [`docs/missing_modality_protocol.md`](docs/missing_modality_protocol.md) | Missing-modality experiment design |
| [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) | Metrics and evaluation procedures |
| [`docs/explainability_protocol.md`](docs/explainability_protocol.md) | Explainability methods |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducibility guarantees |
| [`docs/limitations_and_risks.md`](docs/limitations_and_risks.md) | Known limitations and risks |
| [`docs/decision_log.md`](docs/decision_log.md) | Key design decisions |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Experiment results log |

## Reproducibility

- All random seeds are fixed and documented
- Train/validation/test splits are persisted as CSV manifests
- Configuration files capture all hyperparameters
- Model checkpoints include training metadata

See [`docs/reproducibility.md`](docs/reproducibility.md) for details.

## Limitations

- This is a **research prototype**, not a clinical tool
- Missing modalities are **simulated**, not naturally observed
- Results are specific to the four selected TCGA cancer types
- No external validation cohort is used
- Attention weights should not be interpreted as causal explanations

See [`docs/limitations_and_risks.md`](docs/limitations_and_risks.md) for the complete statement.

## License

This project is for academic research purposes. The underlying TCGA data is publicly available through the Genomic Data Commons.

## Citation

If referencing this work, cite both this project and the MLOmics dataset:

> MLOmics: A Multi-Omics Benchmark for Machine Learning. Zenodo. DOI: 10.5281/zenodo.28729127
