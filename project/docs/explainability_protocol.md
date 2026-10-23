# Explainability Protocol

## 1. Goals

1. Identify which **modalities** are most important for classification
2. Identify which **features** within each modality drive predictions
3. Provide **local explanations** for individual sample predictions
4. Assess **stability** of explanations under missing-modality conditions
5. Clearly communicate the **limitations** of each explanation method

## 2. Modality-Level Importance

### 2.1 Permutation Importance
- **Method:** For each modality, permute (shuffle) its features across test samples and measure the drop in macro-F1
- **Repetitions:** 10 permutations per modality, report mean ± std drop
- **Interpretation:** Larger drop = more important modality
- **Cost:** 4 modalities × 10 permutations × 1 forward pass each = 40 forward passes (cheap)

### 2.2 Leave-One-Out Ablation
- **Method:** Evaluate the model with each modality removed (conditions C1–C4)
- **Interpretation:** Larger degradation = more important modality
- **Note:** Already computed as part of missing-modality experiments (no extra cost)

### 2.3 Attention Weight Analysis (If Using Attention Fusion)
- **Method:** Extract mean attention weights from the fusion module
- **Report:** Average attention weight per modality across test samples, broken down by class

> [!WARNING]
> Attention weights indicate where the model "looks" but do NOT necessarily reflect feature importance or causal contribution. Report attention weights as descriptive statistics, not as explanations. Cite Jain & Wallace (2019) "Attention is not Explanation."

## 3. Feature-Level Importance

### 3.1 Integrated Gradients
- **Method:** Compute attributions from a zero baseline to the input for each feature
- **Implementation:** Using the Captum library
- **Aggregation:** Mean absolute attribution across test samples, per class
- **Output:** Top-20 features per modality, per class
- **Cost:** O(steps × test_samples × forward_passes). With 50 integration steps and ~340 test samples: ~17,000 forward passes (moderate, 10–30 minutes on GPU)

### 3.2 SHAP (Conditional on Feasibility)
- **Method:** Kernel SHAP or Deep SHAP
- **Limitation:** Kernel SHAP is expensive for high-dimensional inputs. May only be feasible for the smallest modality (miRNA, 383 features) or on a sample of test predictions.
- **Fallback:** If SHAP is too slow, rely on integrated gradients.

### 3.3 Per-Modality Feature Rankings
For each modality, produce a ranked list of features by importance:
1. Compute mean absolute integrated gradient attribution across correctly classified test samples
2. Report top 20 features per modality
3. Cross-reference with known cancer biomarkers (literature check, not validated)

## 4. Local Explanations

### 4.1 Sample-Level Attributions
For selected individual predictions:
- Correctly classified examples (1 per class, 4 total)
- Incorrectly classified examples (up to 4, if available)
- Compute integrated gradient attributions per feature

### 4.2 Visualization
- Waterfall plots showing top contributing features for individual predictions
- Modality-level importance bar charts per sample

## 5. Stability Under Missing Modalities

### 5.1 Question
Do feature attributions for a given sample change when a modality is removed?

### 5.2 Method
1. Select a set of correctly classified test samples under C0 (complete)
2. Compute attributions under C0
3. Compute attributions under C1–C4 (each single-missing condition)
4. Measure correlation between C0 attributions and Cx attributions for the remaining features
5. Report average Spearman correlation

### 5.3 Interpretation
High correlation = explanations are stable (remaining features maintain their relative importance). Low correlation = the model redistributes importance when a modality is absent.

## 6. Limitations and Caveats

1. **No causal claims.** Feature attributions show what the model uses, not what biologically causes cancer. The distinction between correlation-in-model and biological causation must be clearly stated.

2. **Attention ≠ explanation.** Self-attention weights can be uniform, concentrated, or adversarially manipulated without changing predictions. They are descriptive, not explanatory.

3. **Baseline dependence.** Integrated gradients results depend on the choice of baseline (we use zero). Different baselines can produce different rankings.

4. **Aggregation smooths.** Mean attributions across samples may mask important per-sample variation. Local explanations complement global rankings.

5. **Model-specific.** Explanations are specific to this model architecture and training. A different model may produce different attributions for the same data.

6. **Not validated externally.** We do not validate feature importance against external experimental biology results.
