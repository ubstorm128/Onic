# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-16

### Added
- Created `PROJECT_STATUS.md` comprehensive project execution dashboard.
- Stage C: Implemented leak-free data preprocessing, stratified 70/15/15 splitting (`prepare_dataset.py`).
- Stage D: Executed all 6 baseline models across 5 random seeds (mRNA MLP, miRNA MLP, CNV MLP, Methy MLP, Early Fusion MLP, LightGBM GBDT).
- Stage E: Implemented and trained proposed Multimodal Cross-Attention Architecture with learned default tokens and attention masking under Strategy S1 (complete) and Strategy S2 (modality dropout $p=0.3$).
- Stage F: Evaluated missing-modality robustness across all 15 conditions (C0–C14) across 5 seeds, demonstrating an +8.52% Macro-F1 improvement for S2 under noisy single-modality conditions.
- Stage G: Implemented explainability pipeline (`run_explainability.py`) with modality ablation, permutation importance, inter-modality attention weight analysis, and top biomarker discovery per cancer type.
- Stage H: Generated publication-grade figures in `artifacts/figures/` (`fig1_baselines_comparison.png`, `fig2_missing_modality_robustness.png`, `fig3_attention_heatmap.png`).

## [0.2.0] - 2026-10-15


### Added
- Created complete documentation suite in `docs/`:
  - `research_questions.md`
  - `experimental_protocol.md`
  - `model_architecture.md`
  - `missing_modality_protocol.md`
  - `evaluation_protocol.md`
  - `explainability_protocol.md`
  - `reproducibility.md`
  - `limitations_and_risks.md`
  - `decision_log.md` (DEC-001 through DEC-007)
  - `experiment_log.md`
- Created configuration files in `configs/`:
  - `data.yaml`
  - `model.yaml`
  - `experiment.yaml`
- Created `requirements.txt` and `.gitignore`

## [0.1.0] - 2026-10-15

### Added
- Completed Stage A comprehensive dataset audit (26/26 verification checks passed)
- Generated `reports/dataset_audit.json`, `reports/class_distribution.csv`, and `reports/sample_alignment_report.csv`
- Documented verified label mapping for BRCA (2), HNSC (8), LGG (13), THCA (27)
- Created initial `README.md`, `implementation.md`, and `docs/dataset_audit.md`, `docs/data_dictionary.md`
