# Research Questions

## Primary Questions

### RQ1: Multimodal Advantage
**Does integrating four omics modalities (mRNA, miRNA, CNV, DNA methylation) improve cancer classification accuracy compared to using any single modality?**

- **Hypothesis:** Multimodal fusion will outperform all single-modality baselines because different omics capture complementary biological signals.
- **Evaluation:** Compare macro-F1 of fusion model vs. each single-modality model on the same test set.
- **Expected outcome:** Moderate improvement (5–15% macro-F1 gain over worst single modality; 2–5% over best).

### RQ2: Fusion Strategy
**Does attention-based fusion provide better cancer classification than simple feature concatenation?**

- **Hypothesis:** Attention-based fusion will outperform concatenation by learning adaptive modality weighting, but the margin may be small given the structured tabular nature of the data.
- **Evaluation:** Compare attention fusion vs. concatenation baseline on identical splits and metrics.
- **Null hypothesis to test:** There is no statistically significant difference between concatenation and attention fusion (Wilcoxon signed-rank test across seeds).

### RQ3: Missing-Modality Degradation
**How does classification performance degrade when one or more modalities are systematically absent at inference time?**

- **Hypothesis:** Performance will degrade gracefully for single-modality-missing conditions and more severely for two-modality-missing conditions, but some combinations will degrade more than others depending on modality informativeness.
- **Evaluation:** Report macro-F1 for all 15 missingness conditions. Quantify degradation relative to complete input.

### RQ4: Modality Dropout Training
**Does training with random modality dropout improve robustness to missing modalities at inference compared to training with all modalities present?**

- **Hypothesis:** Models trained with modality dropout will show smaller performance drops under missing-modality conditions, at a possible cost to complete-input performance.
- **Evaluation:** Compare training strategies S1 (no dropout) vs. S2 (random dropout) vs. S3 (curriculum dropout) across all missingness conditions.

### RQ5: Feature and Modality Importance
**Which modalities and features are most informative for distinguishing each cancer type, and does this change under missing-modality conditions?**

- **Hypothesis:** mRNA expression will be the most informative single modality overall, but importance rankings may vary by cancer type.
- **Evaluation:** Permutation importance at modality level; integrated gradients or SHAP at feature level.

## Secondary Questions

### RQ6: Representation Quality
Does the shared representation dimension (D) affect the balance between single-modality and multimodal performance?

### RQ7: Missing-Modality Default Strategy
Is a learned default embedding superior to zero-masking for representing absent modalities?

### RQ8: Class-Specific Robustness
Are some cancer types more robust to missing modalities than others? Does this correlate with their molecular distinctiveness?
