# Stage G: Explainability and Biomarker Discovery Report

**Model Configuration:** s2_dropout (Seed 42)  
**Cohort:** TCGA Pan-Cancer (BRCA, HNSC, LGG, THCA; N=2,263, Test N=340)

---

## 1. Modality-Level Importance

### 1.1 Ablation and Permutation Drop

| Modality | Ablated F1 | Drop from C0 (Ablation) | Permutation Mean Drop ± Std |
|---|---:|---:|---:|
| **mRNA** | 0.9945 | +0.0000 | 0.0005 ± 0.0021 |
| **miRNA** | 0.9945 | +0.0000 | -0.0006 ± 0.0011 |
| **CNV** | 0.9973 | -0.0028 | 0.0056 ± 0.0018 |
| **Methy** | 0.9973 | -0.0028 | 0.0005 ± 0.0021 |

---

## 2. Inter-Modality Attention Weights

> **Methodological Limitation Notice:** Self-attention weights reflect token interactions in the fusion layer, not causal attribution. They are reported to characterize information routing across omics tokens.

| Query \ Key | mRNA | miRNA | CNV | Methy |
|---|---:|---:|---:|---:|
| **mRNA** | 0.3401 | 0.2610 | 0.0569 | 0.3421 |
| **miRNA** | 0.3440 | 0.2649 | 0.0539 | 0.3372 |
| **CNV** | 0.3086 | 0.2842 | 0.1066 | 0.3006 |
| **Methy** | 0.3449 | 0.2527 | 0.0536 | 0.3488 |

---

## 3. Top Biological Biomarkers per Cancer Type (mRNA)

### BRCA
| Rank | Gene (mRNA) | Attribution Score | Top miRNA | Top CNV | Top Methylation Probe |
|---:|---|---:|---|---|---|
| 1 | `SFRP1` | 0.000173 | `hsa-miR-10b-3p` | `TTI2` | `cg10818284` |
| 2 | `C5orf46` | 0.000173 | `hsa-miR-190b` | `SNORD13` | `cg08360726` |
| 3 | `STAC2` | 0.000146 | `hsa-miR-196a-5p` | `FUT10` | `cg27656658` |
| 4 | `ERN2` | 0.000145 | `hsa-miR-615-3p` | `MAK16` | `cg27219841` |
| 5 | `MIA` | 0.000137 | `hsa-miR-10a-3p` | `FBXO25` | `cg17005319` |

### HNSC
| Rank | Gene (mRNA) | Attribution Score | Top miRNA | Top CNV | Top Methylation Probe |
|---:|---|---:|---|---|---|
| 1 | `MIA` | 0.000994 | `hsa-miR-6720-3p` | `PPFIA1` | `cg27342919` |
| 2 | `ERN2` | 0.000864 | `hsa-miR-944` | `ANO1` | `cg27466845` |
| 3 | `DMBT1` | 0.000848 | `hsa-miR-30c-2-3p` | `IMMP2L` | `cg14972210` |
| 4 | `C17orf93` | 0.000812 | `hsa-miR-34c-3p` | `MIR3664` | `cg00750880` |
| 5 | `B3GNT6` | 0.000773 | `hsa-miR-4652-5p` | `SHANK2` | `cg17005319` |

### LGG
| Rank | Gene (mRNA) | Attribution Score | Top miRNA | Top CNV | Top Methylation Probe |
|---:|---|---:|---|---|---|
| 1 | `CNTNAP4` | 0.000220 | `hsa-miR-200b-5p` | `FIP1L1` | `cg24882673` |
| 2 | `C8orf46` | 0.000216 | `hsa-miR-217` | `PDGFRA` | `cg14676272` |
| 3 | `BAALC` | 0.000211 | `hsa-miR-99b-3p` | `MET` | `cg23460250` |
| 4 | `TTYH1` | 0.000205 | `hsa-miR-143-5p` | `ST7` | `cg05970721` |
| 5 | `CTNND2` | 0.000201 | `hsa-miR-9-5p` | `CDK6` | `cg01736164` |

### THCA
| Rank | Gene (mRNA) | Attribution Score | Top miRNA | Top CNV | Top Methylation Probe |
|---:|---|---:|---|---|---|
| 1 | `SLC26A4` | 0.000116 | `hsa-let-7i-3p` | `CHRM3` | `cg01303372` |
| 2 | `FCGBP` | 0.000104 | `hsa-miR-135a-5p` | `COX20` | `cg09901035` |
| 3 | `PDE1A` | 0.000104 | `hsa-miR-1251-5p` | `PAPD7` | `cg22986569` |
| 4 | `FRMD3` | 0.000098 | `hsa-miR-615-3p` | `LYST` | `cg25654695` |
| 5 | `IGFBP6` | 0.000094 | `hsa-miR-146b-3p` | `CMBL` | `cg20418529` |
