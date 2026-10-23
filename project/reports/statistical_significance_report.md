# Statistical Significance and Paired Hypothesis Testing Report

**Methodological Framework:**
- **Paired Comparison:** S1 (complete reference) and S2 (modality dropout) are paired seed-by-seed ($N=5$) on the held-out test split of 340 patients.
- **Variance Interpretation:** All statistics quantify **algorithmic training variance** (initialization, mini-batch ordering, and stochastic dropout masks). They do **not** represent patient population sampling variance, as all models evaluated the same 340 test patients.
- **Wilcoxon Limitation:** With $N=5$ pairs, the minimum possible two-sided Wilcoxon $p$-value is $2/32 = 0.0625$ (when all 5 pairs show positive difference and zero ties). Hence, Wilcoxon signed-rank tests cannot reach $\alpha = 0.05$ with $N=5$.

## 1. Paired Missing-Modality Evaluation (S2 vs. S1 Across 15 Conditions)

| Condition | Description | S1 Mean ± Std | S2 Mean ± Std | Paired Mean Diff (Δ) | Paired Cohen's d | Paired t-stat | Paired t p-val | Wilcoxon stat (p-val) | 95% CI of Difference | Significant (α=0.05) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | Complete (reference) | 0.9956 ± 0.0015 | 0.9950 ± 0.0012 | **-0.000552** | -0.2390 | -0.5345 | 0.6213 | 2.0 (1.0000) | [-0.0034, +0.0023] | No |
| **C1** | mRNA missing | 0.9939 ± 0.0012 | 0.9950 ± 0.0012 | **+0.001107** | 0.4472 | 1.0000 | 0.3739 | 0.0 (1.0000) | [-0.0020, +0.0042] | No |
| **C2** | miRNA missing | 0.9945 ± 0.0000 | 0.9945 ± 0.0000 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 0.0 (1.0000) | [+0.0000, +0.0000] | No |
| **C3** | CNV missing | 0.9956 ± 0.0015 | 0.9967 ± 0.0012 | **+0.001104** | 0.4472 | 1.0000 | 0.3739 | 2.5 (0.6250) | [-0.0020, +0.0042] | No |
| **C4** | Methy missing | 0.9956 ± 0.0015 | 0.9967 ± 0.0012 | **+0.001104** | 0.4472 | 1.0000 | 0.3739 | 2.5 (0.6250) | [-0.0020, +0.0042] | No |
| **C5** | mRNA+miRNA missing | 0.9934 ± 0.0025 | 0.9938 ± 0.0015 | **+0.000436** | 0.1357 | 0.3034 | 0.7767 | 1.0 (1.0000) | [-0.0036, +0.0044] | No |
| **C6** | mRNA+CNV missing | 0.9956 ± 0.0015 | 0.9956 ± 0.0015 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 5.0 (1.0000) | [-0.0034, +0.0034] | No |
| **C7** | mRNA+Methy missing | 0.9928 ± 0.0015 | 0.9939 ± 0.0030 | **+0.001101** | 0.2641 | 0.5906 | 0.5866 | 4.0 (0.8750) | [-0.0041, +0.0063] | No |
| **C8** | miRNA+CNV missing | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 1.5 (1.0000) | [-0.0024, +0.0024] | No |
| **C9** | miRNA+Methy missing | 0.9950 ± 0.0012 | 0.9950 ± 0.0012 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 1.5 (1.0000) | [-0.0024, +0.0024] | No |
| **C10** | CNV+Methy missing | 0.9973 ± 0.0000 | 0.9973 ± 0.0000 | **+0.000000** | 0.0000 | 0.0000 | 1.0000 | 0.0 (1.0000) | [+0.0000, +0.0000] | No |
| **C11** | Only Methy present | 0.9923 ± 0.0050 | 0.9933 ± 0.0028 | **+0.001001** | 0.1592 | 0.3559 | 0.7399 | 1.0 (1.0000) | [-0.0068, +0.0088] | No |
| **C12** | Only CNV present | 0.6561 ± 0.0892 | 0.7412 ± 0.0476 | **+0.085172** | 0.7425 | 1.6603 | 0.1722 | 2.0 (0.1875) | [-0.0573, +0.2276] | No |
| **C13** | Only miRNA present | 0.9915 ± 0.0034 | 0.9945 ± 0.0033 | **+0.003006** | 0.6763 | 1.5122 | 0.2050 | 2.0 (0.1875) | [-0.0025, +0.0085] | No |
| **C14** | Only mRNA present | 0.9956 ± 0.0015 | 0.9973 ± 0.0000 | **+0.001656** | 1.0954 | 2.4495 | 0.0705 | 0.0 (0.2500) | [-0.0002, +0.0035] | No |

## 2. Baseline Model Performance & 95% Confidence Intervals

*(Computed strictly from individual seed test runs. Zero fabrication).* 

| Model | Modality Scope | N Seeds | Raw Per-Seed Test Macro-F1 | Mean ± Sample Std | 95% Confidence Interval | Seed Variance Detected? |
|:---|:---:|:---:|:---|:---:|:---:|:---:|
| **D1_mRNA_MLP** | Complete / Single | 5 | [0.9945, 0.9973, 0.9973, 0.9973, 0.9945] | 0.9961 ± 0.0015 | [0.9943, 0.9980] | Yes |
| **D2_miRNA_MLP** | Complete / Single | 5 | [0.9973, 0.9973, 0.9945, 0.9973, 0.9973] | 0.9967 ± 0.0012 | [0.9952, 0.9982] | Yes |
| **D3_CNV_MLP** | Complete / Single | 5 | [0.8548, 0.8589, 0.8661, 0.8355, 0.8507] | 0.8532 ± 0.0114 | [0.8391, 0.8674] | Yes |
| **D4_Methy_MLP** | Complete / Single | 5 | [0.9945, 0.9945, 0.9945, 0.9945, 0.9945] | 0.9945 ± 0.0000 | [0.9945, 0.9945] | No (Deterministic/Zero) |
| **D5_Early_Fusion_MLP** | Complete / Single | 5 | [0.9973, 0.9973, 0.9973, 0.9945, 0.9945] | 0.9961 ± 0.0015 | [0.9943, 0.9980] | Yes |
| **D6_LightGBM** | Complete / Single | 5 | [0.9973, 0.9973, 0.9973, 0.9973, 0.9973] | 0.9973 ± 0.0000 | [0.9973, 0.9973] | No (Deterministic/Zero) |
| **E1_Proposed_S1** | Complete / Single | 5 | [0.9945, 0.9945, 0.9973, 0.9973, 0.9945] | 0.9956 ± 0.0015 | [0.9937, 0.9975] | Yes |
| **E2_Proposed_S2** | Complete / Single | 5 | [0.9945, 0.9973, 0.9945, 0.9945, 0.9945] | 0.9950 ± 0.0012 | [0.9935, 0.9966] | Yes |