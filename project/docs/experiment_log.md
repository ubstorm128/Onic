# Experiment Log

This document records the results of each experiment as it is completed.

---

## Status Key
- 🔲 Pending
- 🔄 Running
- ✅ Complete
- ❌ Failed

---

## Stage A: Dataset Audit

| Experiment | Status | Date | Notes |
|---|---|---|---|
| A1: Full dataset audit | ✅ | 2026-10-15 | 26/26 checks passed. See `reports/dataset_audit.json` |

---

## Stage D: Baselines

| Experiment | Status | Date | Val Macro-F1 | Test Macro-F1 | Notes |
|---|---|---|---|---|---|
| D1: mRNA single-modality | ✅ | 2026-10-15 | 1.0000 ± 0.0000 | 0.9962 ± 0.0014 | Near-perfect expression separation |
| D2: miRNA single-modality | ✅ | 2026-10-15 | 1.0000 ± 0.0000 | 0.9967 ± 0.0011 | Extremely high discriminative power |
| D3: CNV single-modality | ✅ | 2026-10-15 | 0.8554 ± 0.0094 | 0.8532 ± 0.0102 | Structural variations less cancer-specific |
| D4: Methy single-modality | ✅ | 2026-10-15 | 1.0000 ± 0.0000 | 0.9945 ± 0.0000 | High epigenetic signal |
| D5: Concatenation baseline | ✅ | 2026-10-15 | 1.0000 ± 0.0000 | 0.9962 ± 0.0014 | Early fusion MLP (9,844 features) |
| D6: LightGBM baseline | ✅ | 2026-10-15 | 1.0000 ± 0.0000 | 0.9973 ± 0.0000 | Strongest baseline (Macro-F1 0.9973) |


---

## Stage E: Fusion Model

| Experiment | Status | Date | Val Macro-F1 | Test Macro-F1 | Notes |
|---|---|---|---|---|---|
| E1: Cross-attention fusion (S1) | ✅ | 2026-10-16 | 1.0000 ± 0.0000 | 0.9956 ± 0.0014 | Complete input reference model |
| E2: Cross-attention fusion (S2) | ✅ | 2026-10-16 | 1.0000 ± 0.0000 | 0.9950 ± 0.0011 | Trained with modality dropout p=0.3 |

---

## Stage F: Missing-Modality Experiments

| Experiment | Status | Date | Best Strategy | C0 F1 | Avg Degradation | Notes |
|---|---|---|---|---|---|---|
| F1: Strategy S1 (no dropout) | ✅ | 2026-10-16 | S1 | 0.9956 | -0.0249 | Catastrophic drop on C12 (F1=0.6561) |
| F2: Strategy S2 (random dropout) | ✅ | 2026-10-16 | **S2** | 0.9950 | **-0.0182** | **+8.52% gain on C12 (F1=0.7412)** |

---

## Stage G: Explainability

| Experiment | Status | Date | Notes |
|---|---|---|---|
| G1: Modality importance | ✅ | 2026-10-16 | mRNA & Methy attention ~34%, CNV downweighted to 5.7% |
| G2: Feature importance | ✅ | 2026-10-16 | Input*Gradient identified SFRP1 (BRCA), ANO1 (HNSC), PDGFRA (LGG) |


---

## Ablation Studies
 
| Experiment | Status | Date | C0 Macro-F1 | C12 (CNV Only) F1 | Key Finding |
|---|---|---|---|---|---|
| A1: Representation dim ($D \in \{64, 128, 256\}$) | ✅ | 2026-10-16 | 0.9954 (64) / 0.9945 (256) | **0.7011** (64) vs 0.6633 (256) | $D=64$ regularizes against overfitting and excels under missingness |
| A2: Attention heads ($H \in \{1, 2, 4\}$) | ✅ | 2026-10-16 | 0.9973 (1) / 0.9954 (4) | 0.5907 (1) vs **0.7426** (2) | Single-head attention collapses on C12 (0.5907); multi-head routing is mandatory |
| A3: Dropout rate ($p \in \{0.1, 0.3, 0.5\}$) | ✅ | 2026-10-16 | 0.9945 (0.5) / 0.9945 (0.1) | 0.6605 (0.1) $\to$ **0.7897** (0.5) | Monotonic gain in missing resilience; $p=0.5$ delivers +13.36% gain over S1 |
| A4: Learned default vs. zero masking | ✅ | 2026-10-16 | 0.9954 (Learned) / 0.9945 (Zero) | 0.6768 ± 0.09 vs 0.6807 ± 0.13 | Learned token reduces variance by 35% compared to zero masking |

