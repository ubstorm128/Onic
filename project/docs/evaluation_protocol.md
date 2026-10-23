# Evaluation Protocol

## 1. Metrics

### 1.1 Primary Metric
- **Macro-F1:** Unweighted average of per-class F1 scores. Chosen because it equally weights all classes regardless of sample count, which is appropriate given our moderate class imbalance.

### 1.2 Secondary Metrics
- **Balanced Accuracy:** Average of per-class recall (sensitivity). Equivalent to accuracy on a class-balanced test set.
- **Overall Accuracy:** Total correct / total samples. Reported for completeness but not used for model selection (biased toward majority class).

### 1.3 Per-Class Metrics
For each cancer type (BRCA, HNSC, LGG, THCA):
- Precision
- Recall (sensitivity)
- F1-score

### 1.4 Confusion Matrix
4×4 matrix with true labels on rows, predicted labels on columns. Reported for each experimental condition.

## 2. Data Splits

| Split | Purpose | Used for Model Selection? | Used for Reporting? |
|---|---|---|---|
| Train (70%) | Model parameter optimization | No | No |
| Validation (15%) | Hyperparameter tuning, early stopping, model selection | Yes | Yes (for development) |
| Test (15%) | Final evaluation | **Never** | Yes (final results only) |

> [!CAUTION]
> The test set must NEVER be used for: model selection, hyperparameter tuning, threshold optimization, feature selection, or any decision that affects the model. It is evaluated exactly once per final model configuration.

## 3. Multi-Seed Protocol

### 3.1 Seeds
Five random seeds: `[42, 123, 456, 789, 1024]`

Each seed controls:
- Model weight initialization
- Modality dropout sampling (during training)
- DataLoader shuffling

### 3.2 Split Consistency
The data split is fixed (seed=42 for splitting). All five model seeds use the same train/val/test split.

### 3.3 Reporting
- Report: **mean ± standard deviation** across 5 seeds
- Never select and report only the best seed
- If a run fails or diverges, document it (do not silently exclude)

## 4. Statistical Testing

### 4.1 Pairwise Comparisons
For comparing two models (e.g., concatenation vs. attention fusion):
- Use **Wilcoxon signed-rank test** on paired metrics across 5 seeds
- Significance level: α = 0.05
- Report p-values alongside effect sizes

### 4.2 Multiple Comparisons
When comparing >2 models:
- Use Friedman test first (non-parametric repeated measures)
- If significant, follow up with pairwise Wilcoxon tests with Bonferroni correction

### 4.3 Limitations
With only 5 seeds, statistical power is limited. Report this limitation. Consider results as indicative rather than definitive.

## 5. Missing-Modality Evaluation

### 5.1 Per-Condition Reporting
For each of the 15 missingness conditions:
- All metrics from Section 1
- Degradation relative to C0 (complete): `delta = metric_C0 - metric_Cx`
- Relative degradation: `delta_pct = delta / metric_C0 * 100`

### 5.2 Aggregated Reporting
- Average degradation when each modality is absent (across all conditions where it's missing)
- Average degradation by number of missing modalities (0, 1, 2, 3)
- Worst-case single-missing condition
- Worst-case two-missing condition

## 6. Validation vs. Test Reporting

| Stage | Report | Set |
|---|---|---|
| During development | Validation metrics, training curves | Validation |
| Hyperparameter tuning | Validation metrics | Validation |
| Final model evaluation | All metrics, confusion matrices | Test |
| Missing-modality experiments | All metrics per condition | Test |
| Ablation studies | Key metrics | Validation (test for final selected config) |

## 7. Artifacts per Experiment

| Artifact | Format | Location |
|---|---|---|
| Training log | CSV | `artifacts/metrics/{id}_training_log.csv` |
| Validation metrics | JSON | `artifacts/metrics/{id}_val_results.json` |
| Test metrics | JSON | `artifacts/metrics/{id}_test_results.json` |
| Confusion matrix | CSV | `artifacts/metrics/{id}_confusion.csv` |
| Model checkpoint | PyTorch | `artifacts/models/{id}_best.pt` |
| Configuration | YAML | `artifacts/metrics/{id}_config.yaml` |
